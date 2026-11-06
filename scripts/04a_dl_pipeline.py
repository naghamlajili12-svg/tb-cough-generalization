"""
scripts/04_dl_pipeline.py

Deep-learning pipeline (cross-dataset), merged for all three source datasets.

For each source dataset (Zambia / TBScreen passive / CODA), a VGGish model is trained
on mel-spectrogram inputs with subject-level nested CV, then evaluated on every other
dataset. The optimal spectrogram frequency range is fixed per source (paper Fig. 10):

    Zambia            : 100-8000 Hz
    TBScreen Passive  :  10-5000 Hz
    CODA              : 200-6000 Hz

Per source the procedure is repeated over N_RANDOM_TESTSETS random outer splits, each
with N_INNER_FOLDS inner folds (a fresh model per inner fold), producing many estimates.
Predicted probabilities for the source's held-out test fold and for every external
dataset are saved to OUTPUT_DIR/<source>.pkl for scoring by 04_dl_analysis.py.

Source training data is subgroup-balanced (device x location x label) and augmented;
external datasets use all samples in raw form. Datasets are indexed from data/interim
+ metadata, identically to 03a.

All parameters are defined in the CONFIGURATION section below.

Usage:
    python scripts/04_dl_pipeline.py
"""

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
from tqdm import tqdm

from utils_audio_aug import aug_add_noise, aug_brownian_tape_speed, aug_pitch_shift
from utils_dl import (
    AUDIO_DURATION, N_MEL, SR,
    build_model, get_input_adapter, model_eval_roc, pad_audio,
    prep_mel_spectrogram, prep_X_y, random_stratified_kfold, train_from_arrays,
)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_CACHE = Path("data/processed/dl_features")          # cached mel-spectrograms
OUTPUT_DIR    = Path("results/metrics/dl")                   # saved predictions

MODEL_NAME = "VGGish16"          # selected backbone (mel-spectrogram input)
# Pretrained VGGish weights (optional). If the file is missing the backbone is
# initialised randomly and only the classifier head is learned afresh.
VGGISH_WEIGHTS = Path("data/pytorch_vggish.pth")
if not VGGISH_WEIGHTS.exists():
    print(f"Warning: pretrained VGGish weights not found at {VGGISH_WEIGHTS}. "
          "The model will be trained from scratch.")

RANDOM_SEED       = 42
N_RANDOM_TESTSETS = 10
N_INNER_FOLDS     = 4
EPOCHS            = 40

PARAM = {
    "batch_size": 32, "lr_backbone": 1e-4, "lr_head": 1e-5, "weight_decay": 0,
    "dropout_rate": 0.5, "if_mixup": True, "if_label_smoothing": False,
    "hidden_nodes": 16, "if_mixstyle": False, "mixstyle_p": 0, "mixstyle_alpha": 0.1,
}

# Augmentation (applied to the source training set only); one option chosen at random.
AUG_PITCH   = (aug_pitch_shift,         {"max_steps": 4.0, "p_apply": 1})
AUG_BROWN   = (aug_brownian_tape_speed, {"max_rate_perturb": 0.1, "n_walk_steps": 5, "p_apply": 0.8})
AUG_NOISE   = (aug_add_noise,           {"snr_db": 10, "p_apply": 1})
AUGMENTATION_OPTIONS = [
    [AUG_PITCH, AUG_BROWN], [AUG_PITCH, AUG_NOISE], [AUG_PITCH], [AUG_BROWN], [AUG_NOISE],
]

# Datasets (display name, file key); order matches 03b and the analysis heatmap.
DATASETS = ["Zambia", "Zambia - Audio Recorder", "TBScreen Passive", "TBScreen Forced", "CODA"]
KEY = {
    "Zambia": "zambia", "Zambia - Audio Recorder": "zambia_AR",
    "TBScreen Passive": "tbscreen_passive", "TBScreen Forced": "tbscreen_forced", "CODA": "coda",
}
# Datasets sharing a subject pool (cross-eval must restrict to held-out subjects).
COHORT = {"zambia": "zambia", "zambia_AR": "zambia",
          "tbscreen_passive": "tbscreen", "tbscreen_forced": "tbscreen", "coda": "coda"}

# Source datasets with their selected frequency range.
SOURCES = {
    "Zambia":           {"key": "zambia",            "fmin": 100, "fmax": 8000},
    "TBScreen Passive": {"key": "tbscreen_passive",  "fmin": 10,  "fmax": 5000},
    "CODA":             {"key": "coda",              "fmin": 200, "fmax": 6000},
}
SOURCES_TO_RUN = list(SOURCES)   # restrict to e.g. ["Zambia"] to run one at a time

# --- interim audio + metadata paths (see 03a for repo-relative defaults) ----
ZAMBIA_INTERIM  = Path("data/interim/zambia")
ZAMBIA_METADATA = Path(
    "data/raw/zambia/Metadata and Codebook/Metadata and Codebook/GHAI_Final_Data_2023.csv"
)
TBSCREEN_INTERIM = Path("data/interim/tbscreen")
TBSCREEN_RAW     = Path("data/raw/tbscreen")
CODA_INTERIM   = Path("data/interim/coda")
CODA_CLINICAL  = Path("data/raw/coda/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv")
CODA_EXTRA     = Path("data/raw/coda/meta_data/Clinical/CODA_TB_additional_variables_train.csv")
CODA_SOLICITED = Path("data/raw/coda/meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv")

DEVICES   = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]
LOCATIONS = ["Chainda", "Chawama", "Kanyama"]
ZAMBIA_PHONE_DEVICES   = ["Phone A", "Phone B", "Phone C"]
ZAMBIA_TRAIN_LOCATIONS = ["Chawama", "Kanyama"]
TBSCREEN_DEVICE_IDS = {"pixel": 0, "yeti": 1, "codec": 2}
TBSCREEN_PASSIVE_MAX_COUGHS = 100
CODA_COUNTRIES = ["MG", "TZ", "VN"]


# =============================================================================
# SAMPLE INDEX (repo-consistent, identical provenance to 03a)
# =============================================================================

@dataclass
class SampleIndex:
    paths: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    device: list[int] = field(default_factory=list)
    location: list[int] = field(default_factory=list)
    subgroup: list[str] = field(default_factory=list)

    def add(self, path, label, group, device, location, subgroup):
        self.paths.append(path); self.labels.append(label); self.groups.append(group)
        self.device.append(device); self.location.append(location); self.subgroup.append(subgroup)

    def __len__(self):
        return len(self.paths)

    def subset(self, positions):
        out = SampleIndex()
        for i in positions:
            out.add(self.paths[i], self.labels[i], self.groups[i],
                    self.device[i], self.location[i], self.subgroup[i])
        return out


def _cap_per_subject(idx: SampleIndex, max_coughs: int) -> SampleIndex:
    rng = np.random.default_rng(RANDOM_SEED)
    groups = np.array(idx.groups)
    keep = []
    for subject in np.unique(groups):
        pos = np.where(groups == subject)[0]
        if len(pos) > max_coughs:
            pos = rng.choice(pos, max_coughs, replace=False)
        keep.extend(pos.tolist())
    return idx.subset(sorted(keep))


def build_index(key: str) -> SampleIndex:
    if key in ("zambia", "zambia_AR"):
        devices = ["Audio-Recorder"] if key == "zambia_AR" else ZAMBIA_PHONE_DEVICES
        df = pd.read_csv(ZAMBIA_METADATA)
        label_by = dict(zip(df["barcode"].astype(str), df["ground_truth_tb"].astype(str)))
        idx = SampleIndex()
        for device in devices:
            for location in ZAMBIA_TRAIN_LOCATIONS:
                folder = ZAMBIA_INTERIM / f"{device}_{location}"
                if not folder.is_dir():
                    continue
                for wav in sorted(folder.glob("*.wav")):
                    subject = wav.stem.split("_")[0]
                    raw = label_by.get(subject, "nan")
                    if raw not in ("pos", "neg"):
                        continue
                    lab = 1 if raw == "pos" else 0
                    d, lo = DEVICES.index(device), LOCATIONS.index(location)
                    idx.add(str(wav), lab, subject, d, lo, f"{d}_{lo}_{lab}")
        return idx

    if key in ("tbscreen_passive", "tbscreen_forced"):
        cough_type = "Passive_coughs" if key == "tbscreen_passive" else "Forced_coughs"
        meta = pd.read_csv(TBSCREEN_RAW / cough_type / f"{cough_type}.csv").dropna(subset=["Label", "subject", "path"])
        meta = meta[meta["Label"].astype(str).isin(["TB", "NTB"])]
        idx = SampleIndex()
        for _, row in meta.iterrows():
            stem = str(row["path"])
            wav = TBSCREEN_INTERIM / cough_type / f"{stem}.wav"
            if not wav.exists():
                continue
            dev = next((d for k, d in TBSCREEN_DEVICE_IDS.items() if k in stem), None)
            if dev is None:
                continue
            lab = 1 if str(row["Label"]) == "TB" else 0
            idx.add(str(wav), lab, str(row["subject"]), dev, -1, f"{dev}_{lab}")
        if key == "tbscreen_passive":
            idx = _cap_per_subject(idx, TBSCREEN_PASSIVE_MAX_COUGHS)
        return idx

    if key == "coda":
        clinical = pd.read_csv(CODA_CLINICAL)
        extra = pd.read_csv(CODA_EXTRA)
        solicited = pd.read_csv(CODA_SOLICITED)
        label_by = dict(zip(clinical["participant"].astype(str), clinical["tb_status"]))
        country_by = dict(zip(extra["participant"].astype(str), extra["Country"].astype(str)))
        audio_by_name = {p.name: p for p in CODA_INTERIM.rglob("*.wav")}
        idx = SampleIndex()
        for _, row in solicited.iterrows():
            subject = str(row["participant"])
            country = country_by.get(subject)
            if country not in CODA_COUNTRIES:
                continue
            wav = audio_by_name.get(Path(str(row["filename"])).name)
            label = label_by.get(subject)
            if wav is None or label is None or pd.isna(label):
                continue
            lab = int(label)
            c = CODA_COUNTRIES.index(country)
            idx.add(str(wav), lab, subject, c, -1, f"{country}_{lab}")
        return idx

    raise ValueError(f"Unknown dataset key: {key}")


# =============================================================================
# MEL-SPECTROGRAM EXTRACTION (cached per dataset x frequency range)
# =============================================================================

def _augment(audio):
    option = AUGMENTATION_OPTIONS[np.random.choice(len(AUGMENTATION_OPTIONS))]
    for func, params in option:
        audio = func(audio, SR, **params)
    return audio


def _mel(audio, fmin, fmax):
    audio = pad_audio(audio, SR, AUDIO_DURATION, pad_by="repeating")
    return prep_mel_spectrogram(audio, fmin=fmin, fmax=fmax, n_mels=N_MEL)


def extract_features(key: str, fmin: int, fmax: int, augment: bool) -> dict:
    """Build (and cache) mel-spectrograms + metadata for one dataset at a frequency range."""
    suffix = "train" if augment else "eval"
    cache = FEATURE_CACHE / f"{key}_{fmin}_{fmax}_{suffix}.npz"
    if cache.exists():
        print(f"  [{key} {fmin}-{fmax} {suffix}] loading cached features")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    idx = build_index(key)
    print(f"  [{key} {fmin}-{fmax} {suffix}] {len(idx)} clips")
    feats, labels, groups, devices, locations = [], [], [], [], []
    audio_cache: dict[str, np.ndarray] = {}

    if augment:
        # subgroup-balanced oversampling, then augment every (resampled) clip
        subgroups = np.array(idx.subgroup)
        uniq = np.unique(subgroups)
        max_n = max(int((subgroups == sg).sum()) for sg in uniq)
        print(f"  [{key}] samples before resampling: {len(idx)}, after: {max_n * len(uniq)} "
              f"({len(uniq)} subgroups x {max_n})")
        seeds = np.random.default_rng(RANDOM_SEED).integers(0, 10_000, len(uniq))
        for sg, seed in zip(tqdm(uniq, desc=f"{key} feat", leave=False), seeds):
            pos = np.where(subgroups == sg)[0]
            n = len(pos)
            np.random.seed(int(seed)); _random.seed(int(seed))
            sel = np.concatenate([np.arange(n), np.random.choice(n, max_n - n, replace=True)]) \
                if n < max_n else np.arange(max_n)
            for occ in sel:
                p = pos[occ]
                if idx.paths[p] not in audio_cache:
                    audio_cache[idx.paths[p]], _ = librosa.load(idx.paths[p], sr=SR)
                feats.append(_mel(_augment(audio_cache[idx.paths[p]]), fmin, fmax))
                labels.append(idx.labels[p]); groups.append(idx.groups[p])
                devices.append(idx.device[p]); locations.append(idx.location[p])
    else:
        for p in tqdm(range(len(idx)), desc=f"{key} feat", leave=False):
            audio, _ = librosa.load(idx.paths[p], sr=SR)
            feats.append(_mel(audio, fmin, fmax))
            labels.append(idx.labels[p]); groups.append(idx.groups[p])
            devices.append(idx.device[p]); locations.append(idx.location[p])

    out = {
        "features": np.asarray(feats, dtype=np.float32),
        "labels": np.array(labels), "groups": np.array(groups, dtype=str),
        "devices": np.array(devices), "locations": np.array(locations),
    }
    # sort by subject so a participant's clips stay together
    order = np.argsort(out["groups"])
    out = {k: v[order] for k, v in out.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


# =============================================================================
# TRAINING DRIVER (one source dataset)
# =============================================================================

def run_source(source_name: str) -> None:
    cfg = SOURCES[source_name]
    src_key, fmin, fmax = cfg["key"], cfg["fmin"], cfg["fmax"]
    print(f"\n=== Source: {source_name}  ({fmin}-{fmax} Hz) ===")

    input_adapter = get_input_adapter(MODEL_NAME)   # per-batch channel/transpose/resize
    vggish_weights = VGGISH_WEIGHTS if (MODEL_NAME == "VGGish16" and VGGISH_WEIGHTS.exists()) else None

    # --- source training data (resampled + augmented); kept 1-channel [N,1,mel,time] ---
    src = extract_features(src_key, fmin, fmax, augment=True)
    X_src = src["features"]
    y_src, g_src, dev_src, loc_src = src["labels"], src["groups"], src["devices"], src["locations"]
    dll_src = np.array([f"{int(dev_src[i])}_{int(loc_src[i])}_{int(y_src[i])}" for i in range(len(y_src))])

    # --- external targets (all samples, raw) at the same frequency range ---
    target_keys = [KEY[d] for d in DATASETS if KEY[d] != src_key]
    targets = {}
    for tkey in target_keys:
        t = extract_features(tkey, fmin, fmax, augment=False)
        targets[tkey] = {"X": t["features"], "labels": t["labels"],
                         "groups": t["groups"], "devices": t["devices"], "locations": t["locations"]}

    seeds_testset = np.random.RandomState(RANDOM_SEED).randint(0, 100, int(np.ceil(N_RANDOM_TESTSETS / 5)))
    seeds_inner = np.random.RandomState(RANDOM_SEED).randint(0, 100, N_RANDOM_TESTSETS)

    within = {"y_true": [], "y_pred": [], "groups": [], "devices": [], "locations": []}
    target_preds = {tkey: [] for tkey in target_keys}

    for ts in range(N_RANDOM_TESTSETS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seeds_testset[ts // 5])
        for i, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(y_src)), dll_src, groups=g_src)):
            if i == ts % 5:
                break

        X_tr_all, y_tr = X_src[train_idx], y_src[train_idx]
        X_te, y_te = X_src[test_idx], y_src[test_idx]
        within["y_true"].append([]); within["y_pred"].append([])
        within["groups"].append(g_src[test_idx].tolist())
        within["devices"].append(dev_src[test_idx].tolist())
        within["locations"].append(loc_src[test_idx].tolist())
        for tkey in target_keys:
            target_preds[tkey].append([])

        coarse = random_stratified_kfold(X_tr_all, dll_src[train_idx], g_src[train_idx],
                                         n_splits=N_INNER_FOLDS, random_state=seeds_inner[ts])
        for inner in range(N_INNER_FOLDS):
            print(f"\n[{source_name}] test set {ts + 1}/{N_RANDOM_TESTSETS} | "
                  f"inner fold {inner + 1}/{N_INNER_FOLDS}", flush=True)
            X_train, y_train, X_val, y_val, _, _ = prep_X_y(X_tr_all, y_tr, coarse, inner)
            model = build_model(MODEL_NAME, hidden_nodes=PARAM["hidden_nodes"],
                                dropout_rate=PARAM["dropout_rate"], if_mixstyle=PARAM["if_mixstyle"],
                                mixstyle_p=PARAM["mixstyle_p"], mixstyle_alpha=PARAM["mixstyle_alpha"],
                                vggish_weights=vggish_weights)
            model = train_from_arrays(
                model, X_train, y_train, X_val, y_val,
                input_adapter=input_adapter,
                epochs=EPOCHS, batch_size=PARAM["batch_size"],
                lr_backbone=PARAM["lr_backbone"], lr_head=PARAM["lr_head"],
                weight_decay=PARAM["weight_decay"], if_mixup=PARAM["if_mixup"],
                if_label_smoothing=PARAM["if_label_smoothing"], mixup_alpha=PARAM["mixstyle_alpha"],
                verbose=True,
            )

            _, preds = model_eval_roc(model, X_te, y_te, input_adapter)
            within["y_true"][ts].append(y_te.tolist())
            within["y_pred"][ts].append(preds.tolist())
            for tkey in target_keys:
                _, tp = model_eval_roc(model, targets[tkey]["X"],
                                       np.ones(len(targets[tkey]["labels"])), input_adapter)
                target_preds[tkey][ts].append(tp.tolist())

    # --- assemble + save ---
    result = {
        "source": source_name, "fmin": fmin, "fmax": fmax, "cohort": COHORT[src_key],
        "within": within,
        "targets": {
            tkey: {"y_pred": target_preds[tkey], "labels": targets[tkey]["labels"],
                   "groups": targets[tkey]["groups"], "devices": targets[tkey]["devices"],
                   "locations": targets[tkey]["locations"], "cohort": COHORT[tkey]}
            for tkey in target_keys
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"{src_key}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"  saved predictions -> {OUTPUT_DIR / f'{src_key}.pkl'}")


def main() -> None:
    for source_name in SOURCES_TO_RUN:
        run_source(source_name)
    print("\nDone. Run scripts/04_dl_analysis.py to score predictions and make figures.")


if __name__ == "__main__":
    main()
