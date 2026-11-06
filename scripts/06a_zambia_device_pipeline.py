"""
scripts/06a_zambia_device_pipeline.py

Zambia device-generalization training (single-device and multi-device).

Two experiments, both on the Zambia phone/recorder devices at the Chawama and Kanyama
sites, using the same VGGish mel-spectrogram pipeline (10-8000 Hz):

  single_device : train on ONE device, evaluate on the held-out fold of that device
                  (within) and on every other device (cross-device).
  multi_device  : train on THREE devices, evaluate on the held-out fold (within) and
                  on the fourth, unseen device.

Per-device features are subgroup-balanced (location x label) and augmented, built once
and shared by both experiments. Predictions are saved to OUTPUT_DIR for scoring by
06b_zambia_device_analysis.py.

All parameters are defined in the CONFIGURATION section below.

Usage:
    python scripts/06a_zambia_device_pipeline.py
"""

from __future__ import annotations

import os
import pickle
import random as _random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils_audio_aug import aug_add_noise, aug_brownian_tape_speed, aug_pitch_shift
from utils_dl import (
    AUDIO_DURATION, N_MEL, SR,
    build_model, get_input_adapter, model_eval_roc, pad_audio, prep_mel_spectrogram,
    prep_X_y, random_stratified_kfold, train_from_arrays,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_CACHE = Path("data/processed/zambia_device")
OUTPUT_DIR    = Path("results/metrics/zambia_device")

VGGISH_WEIGHTS = Path("data/pytorch_vggish.pth")   # optional; random init if missing
if not VGGISH_WEIGHTS.exists():
    print(f"Warning: pretrained VGGish weights not found at {VGGISH_WEIGHTS}. "
          "The model will be trained from scratch.")

FMIN, FMAX = 10, 8000
RANDOM_SEED       = 42
N_RANDOM_TESTSETS = 10
N_INNER_FOLDS     = 4
EPOCHS            = 40

PARAM = {
    "batch_size": 32, "lr_backbone": 1e-4, "lr_head": 1e-5, "weight_decay": 0,
    "dropout_rate": 0.5, "if_mixup": True, "if_label_smoothing": False,
    "hidden_nodes": 16, "if_mixstyle": False, "mixstyle_p": 0, "mixstyle_alpha": 0.1,
}

AUG_PITCH = (aug_pitch_shift,         {"max_steps": 4.0, "p_apply": 1})
AUG_BROWN = (aug_brownian_tape_speed, {"max_rate_perturb": 0.1, "n_walk_steps": 5, "p_apply": 0.8})
AUG_NOISE = (aug_add_noise,           {"snr_db": 10, "p_apply": 1})
AUGMENTATION_OPTIONS = [
    [AUG_PITCH, AUG_BROWN], [AUG_PITCH, AUG_NOISE], [AUG_PITCH], [AUG_BROWN], [AUG_NOISE],
]

DEVICES   = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]
LOCATIONS = ["Chainda", "Chawama", "Kanyama"]
TRAIN_LOCATIONS = ["Chawama", "Kanyama"]

ZAMBIA_INTERIM  = Path("data/interim/zambia")
ZAMBIA_METADATA = Path(
    "data/raw/zambia/Metadata and Codebook/"
    "Metadata and Codebook/GHAI_Final_Data_2023.csv"
)


# =============================================================================
# PER-DEVICE FEATURES (subgroup-balanced + augmented; cached)
# =============================================================================

@dataclass
class DeviceData:
    features: np.ndarray = None       # [N, 1, mel, time]
    labels: np.ndarray = None
    groups: np.ndarray = None
    devices: np.ndarray = None
    locations: np.ndarray = None
    dll: np.ndarray = None            # device_location_label stratify key


def _augment(audio):
    for func, params in AUGMENTATION_OPTIONS[np.random.choice(len(AUGMENTATION_OPTIONS))]:
        audio = func(audio, SR, **params)
    return audio


def _build_device_index(device: str):
    """List (path, label, subject, location_id) for one device at the train sites."""
    df = pd.read_csv(ZAMBIA_METADATA)
    label_by = dict(zip(df["barcode"].astype(str), df["ground_truth_tb"].astype(str)))
    rows = []
    for location in TRAIN_LOCATIONS:
        folder = ZAMBIA_INTERIM / f"{device}_{location}"
        if not folder.is_dir():
            continue
        for wav in sorted(folder.glob("*.wav")):
            subject = wav.stem.split("_")[0]
            raw = label_by.get(subject, "nan")
            if raw not in ("pos", "neg"):
                continue
            rows.append((str(wav), 1 if raw == "pos" else 0, subject, LOCATIONS.index(location)))
    return rows


def extract_device_features(device: str) -> DeviceData:
    cache = FEATURE_CACHE / f"{device}.npz"
    if cache.exists():
        print(f"  [{device}] loading cached features")
        d = np.load(cache, allow_pickle=True)
        return DeviceData(**{k: d[k] for k in d.files})

    rows = _build_device_index(device)
    dev_id = DEVICES.index(device)
    print(f"  [{device}] {len(rows)} clips")
    subgroups = np.array([f"{loc}_{lab}" for _, lab, _, loc in rows])
    uniq = np.unique(subgroups)
    counts = {sg: int((subgroups == sg).sum()) for sg in uniq}
    max_n = max(counts.values())
    print(f"  [{device}] per-(loc_label) counts {counts} -> {max_n} each")
    print(f"  [{device}] samples before resampling: {len(rows)}, after: {max_n * len(uniq)}")
    seeds = np.random.default_rng(RANDOM_SEED).integers(0, 10_000, len(uniq))

    feats, labels, groups, locations = [], [], [], []
    audio_cache: dict[str, np.ndarray] = {}
    for sg, seed in zip(uniq, seeds):
        pos = np.where(subgroups == sg)[0]
        n = len(pos)
        np.random.seed(int(seed)); _random.seed(int(seed))
        sel = np.concatenate([np.arange(n), np.random.choice(n, max_n - n, replace=True)]) \
            if n < max_n else np.arange(max_n)
        for occ in sel:
            path, lab, subject, loc = rows[pos[occ]]
            if path not in audio_cache:
                audio_cache[path], _ = librosa.load(path, sr=SR)
            audio = pad_audio(_augment(audio_cache[path]), SR, AUDIO_DURATION, pad_by="repeating")
            feats.append(prep_mel_spectrogram(audio, fmin=FMIN, fmax=FMAX, sr=SR, n_mels=N_MEL))
            labels.append(lab); groups.append(subject); locations.append(loc)

    features = np.asarray(feats, dtype=np.float32)   # [N, 1, mel, time]; adapter transposes per batch
    labels = np.array(labels); groups = np.array(groups, dtype=str)
    locations = np.array(locations); devices = np.full(len(labels), dev_id)
    dll = np.array([f"{dev_id}_{locations[i]}_{labels[i]}" for i in range(len(labels))])

    order = np.argsort(groups)   # keep a subject's clips together
    data = DeviceData(features[order], labels[order], groups[order],
                      devices[order], locations[order], dll[order])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=data.features, labels=data.labels, groups=data.groups,
             devices=data.devices, locations=data.locations, dll=data.dll)
    return data


# =============================================================================
# NESTED-CV TRAIN + EVAL
# =============================================================================

def nested_cv_train_eval(X, y, groups, dll, targets: dict, vggish_weights):
    """Train over 10 outer x 4 inner folds; return within preds and per-target preds."""
    input_adapter = get_input_adapter("VGGish16")
    seeds_testset = np.random.RandomState(RANDOM_SEED).randint(0, 100, int(np.ceil(N_RANDOM_TESTSETS / 5)))
    seeds_inner = np.random.RandomState(RANDOM_SEED).randint(0, 100, N_RANDOM_TESTSETS)

    within = {"y_true": [], "y_pred": [], "groups": []}
    target_preds = {name: [] for name in targets}

    for ts in range(N_RANDOM_TESTSETS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seeds_testset[ts // 5])
        for i, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), dll, groups=groups)):
            if i == ts % 5:
                break
        X_tr_all, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        within["y_true"].append([]); within["y_pred"].append([])
        within["groups"].append(groups[test_idx].tolist())
        for name in targets:
            target_preds[name].append([])

        coarse = random_stratified_kfold(X_tr_all, dll[train_idx], groups[train_idx],
                                         n_splits=N_INNER_FOLDS, random_state=seeds_inner[ts])
        for inner in range(N_INNER_FOLDS):
            print(f"  test set {ts + 1}/{N_RANDOM_TESTSETS} | inner {inner + 1}/{N_INNER_FOLDS}", flush=True)
            X_train, y_train, X_val, y_val, _, _ = prep_X_y(X_tr_all, y_tr, coarse, inner)
            model = build_model("VGGish16", hidden_nodes=PARAM["hidden_nodes"],
                                dropout_rate=PARAM["dropout_rate"], if_mixstyle=PARAM["if_mixstyle"],
                                mixstyle_p=PARAM["mixstyle_p"], mixstyle_alpha=PARAM["mixstyle_alpha"],
                                vggish_weights=vggish_weights)
            model = train_from_arrays(
                model, X_train, y_train, X_val, y_val,
                input_adapter=input_adapter,
                epochs=EPOCHS, batch_size=PARAM["batch_size"], lr_backbone=PARAM["lr_backbone"],
                lr_head=PARAM["lr_head"], weight_decay=PARAM["weight_decay"], if_mixup=PARAM["if_mixup"],
                if_label_smoothing=PARAM["if_label_smoothing"], mixup_alpha=PARAM["mixstyle_alpha"],
                verbose=True,
            )
            _, preds = model_eval_roc(model, X_te, y_te, input_adapter)
            within["y_true"][ts].append(y_te.tolist())
            within["y_pred"][ts].append(preds.tolist())
            for name, X_target in targets.items():
                _, tp = model_eval_roc(model, X_target, np.ones(len(X_target)), input_adapter)
                target_preds[name][ts].append(tp.tolist())

    return within, target_preds


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vggish_weights = VGGISH_WEIGHTS if VGGISH_WEIGHTS.exists() else None

    print("=== Building per-device features ===")
    data = {d: extract_device_features(d) for d in DEVICES}
    device_meta = {d: {"groups": data[d].groups, "labels": data[d].labels} for d in DEVICES}

    # --- experiment 1: single-device training ---
    print("\n=== Experiment 1: single-device training ===")
    single = {"devices": DEVICES, "device_meta": device_meta, "per_train": {}}
    for d in DEVICES:
        print(f"\n--- train on {d} ---")
        targets = {o: data[o].features for o in DEVICES if o != d}
        within, target_preds = nested_cv_train_eval(
            data[d].features, data[d].labels, data[d].groups, data[d].dll, targets, vggish_weights)
        single["per_train"][d] = {"within": within, "cross": target_preds}
    with open(OUTPUT_DIR / "single_device.pkl", "wb") as f:
        pickle.dump(single, f)
    print(f"  saved -> {OUTPUT_DIR / 'single_device.pkl'}")

    # --- experiment 2: multi-device training (hold one out) ---
    print("\n=== Experiment 2: multi-device training ===")
    multi = {"devices": DEVICES, "device_meta": device_meta, "per_heldout": {}}
    for d in DEVICES:
        print(f"\n--- hold out {d} (train on the other three) ---")
        train_devices = [o for o in DEVICES if o != d]
        X = np.concatenate([data[o].features for o in train_devices])
        y = np.concatenate([data[o].labels for o in train_devices])
        groups = np.concatenate([data[o].groups for o in train_devices])
        dll = np.concatenate([data[o].dll for o in train_devices])
        within, target_preds = nested_cv_train_eval(X, y, groups, dll, {d: data[d].features}, vggish_weights)
        multi["per_heldout"][d] = {"within": within, "heldout": target_preds[d]}
    with open(OUTPUT_DIR / "multi_device.pkl", "wb") as f:
        pickle.dump(multi, f)
    print(f"  saved -> {OUTPUT_DIR / 'multi_device.pkl'}")

    print("\nDone. Run scripts/06b_zambia_device_analysis.py to score and plot.")


if __name__ == "__main__":
    main()
