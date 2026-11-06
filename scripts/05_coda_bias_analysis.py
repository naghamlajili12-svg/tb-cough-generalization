"""
scripts/05_coda_bias_analysis.py

CODA location/device bias analysis.

Trains a TB classifier on CODA (all countries) under two settings and asks whether the
predicted TB probabilities are driven by country-level prevalence rather than the disease
state of individual coughs:

  challenge_replication : reproduces the best CODA TB DREAM Challenge pipeline
                          (ResNet-34, raw 44.1 kHz spectrogram, Hann 1024/64, 50-15000 Hz,
                          no cough segmentation, resampling balanced by TB label only).
  conservative          : a more cautious pipeline (VGGish, 16 kHz mel-spectrogram <= 8000 Hz,
                          cough segmentation, resampling balanced jointly by TB label and country).

For each setting, subject-level predicted probabilities on the held-out folds are grouped
by country and TB label, and the mean predicted probability per country is regressed against
the country-level disease rate (slope and R^2, separately for TB- and TB+ subjects).

Hyperparameters are fixed to the selected best configuration (hidden_nodes=32, lr_head=5e-5);
no grid search is performed.

Outputs (under OUTPUT_DIR):
    predicted_probability_by_country_label_1.pdf   challenge_replication (paper Fig. 3a)
    predicted_probability_by_country_label_2.pdf   conservative          (paper Fig. 3b)

Usage:
    python scripts/05_coda_bias_analysis.py
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # noqa: F401  (kept for parity / quick checks)
from sklearn.model_selection import StratifiedGroupKFold

from utils_audio_aug import aug_add_noise, aug_brownian_tape_speed, aug_pitch_shift
from utils_dl import (
    AUDIO_DURATION,
    build_model, get_input_adapter, model_eval_roc, pad_audio, prep_mel_spectrogram,
    prep_X_y, random_stratified_kfold, train_from_arrays,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_CACHE = Path("data/processed/coda_bias")
PRED_DIR      = Path("results/metrics/coda_bias")
OUTPUT_DIR    = Path("results/figures/coda_bias")

VGGISH_WEIGHTS = Path("data/pytorch_vggish.pth")   # optional; random init if missing
if not VGGISH_WEIGHTS.exists():
    print(f"Warning: pretrained VGGish weights not found at {VGGISH_WEIGHTS}. "
          "The model will be trained from scratch.")

RANDOM_SEED       = 42
N_RANDOM_TESTSETS = 10
N_INNER_FOLDS     = 4
EPOCHS            = 40

# Fixed best hyperparameters (no search).
PARAM = {
    "batch_size": 32, "lr_backbone": 1e-4, "lr_head": 5e-5, "weight_decay": 0,
    "dropout_rate": 0.5, "if_mixup": True, "if_label_smoothing": False,
    "hidden_nodes": 32, "if_mixstyle": False, "mixstyle_p": 0, "mixstyle_alpha": 0.1,
}

AUG_PITCH = (aug_pitch_shift,         {"max_steps": 4.0, "p_apply": 1})
AUG_BROWN = (aug_brownian_tape_speed, {"max_rate_perturb": 0.1, "n_walk_steps": 5, "p_apply": 0.8})
AUG_NOISE = (aug_add_noise,           {"snr_db": 10, "p_apply": 1})
AUGMENTATION_OPTIONS = [
    [AUG_PITCH, AUG_BROWN], [AUG_PITCH, AUG_NOISE], [AUG_PITCH], [AUG_BROWN], [AUG_NOISE],
]

CODA_COUNTRIES = ["IN", "MG", "PH", "SA", "TZ", "UG", "VN"]   # all countries used here
MIN_SAMPLES    = 2048                                          # skip clips shorter than this

# --- two analysis settings ---
SETTINGS = {
    "challenge_replication": {
        "fig_index": 1, "model": "Resnet34", "feature": "spectrogram_raw", "sr": 44_100,
        "resample_by": ["label"],            # balance TB label only
        "spec": {"n_fft": 1024, "hop": 64, "fmin": 50, "fmax": 15000, "clip_s": 0.5},
        "audio": "raw",                      # released clips, no segmentation
    },
    "conservative": {
        "fig_index": 2, "model": "VGGish16", "feature": "mel", "sr": 16_000,
        "resample_by": ["label", "country"], # balance TB label x country
        "mel": {"fmin": 0, "fmax": 8000, "n_mels": 64},
        "audio": "segmented",                # energy-segmented coughs
    },
}
SETTINGS_TO_RUN = list(SETTINGS)

# --- metadata + audio paths ---
CODA_CLINICAL  = Path("data/raw/coda/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv")
CODA_EXTRA     = Path("data/raw/coda/meta_data/Clinical/CODA_TB_additional_variables_train.csv")
CODA_SOLICITED = Path("data/raw/coda/meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv")
# Raw released clips (challenge replication) and energy-segmented clips (conservative).
# NOTE: both must cover all 7 countries; 01_prepare_data.py only segments MG/TZ/VN by
# default, so re-run segmentation for every country before the conservative setting.
CODA_RAW_AUDIO       = Path("data/raw/coda")
CODA_SEGMENTED_AUDIO = Path("data/interim/coda")


# =============================================================================
# SAMPLE INDEX (CODA, all countries)
# =============================================================================

@dataclass
class SampleIndex:
    paths: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    country: list[int] = field(default_factory=list)   # index into CODA_COUNTRIES

    def add(self, path, label, group, country):
        self.paths.append(path); self.labels.append(label)
        self.groups.append(group); self.country.append(country)

    def __len__(self):
        return len(self.paths)


def build_coda_index(audio_dir: Path) -> SampleIndex:
    clinical = pd.read_csv(CODA_CLINICAL)
    extra = pd.read_csv(CODA_EXTRA)
    solicited = pd.read_csv(CODA_SOLICITED)
    label_by = dict(zip(clinical["participant"].astype(str), clinical["tb_status"]))
    country_by = dict(zip(extra["participant"].astype(str), extra["Country"].astype(str)))
    audio_by_name = {p.name: p for p in audio_dir.rglob("*.wav")}

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
        idx.add(str(wav), int(label), subject, CODA_COUNTRIES.index(country))
    return idx


def resample_balanced(idx: SampleIndex, by: list[str]) -> SampleIndex:
    """Oversample (with replacement) so every (key-combination) cell has equal size."""
    labels = np.array(idx.labels)
    country = np.array(idx.country)
    key = labels.astype(str)
    if "country" in by:
        key = np.char.add(np.char.add(labels.astype(str), "_"), country.astype(str))
    cells = np.unique(key)
    counts = {c: int((key == c).sum()) for c in cells}
    max_n = max(counts.values())
    print(f"    resampling by {by}: per-cell counts {counts} -> {max_n} each")
    print(f"    samples before resampling: {len(idx)}, after: {max_n * len(cells)}")
    rng = np.random.default_rng(RANDOM_SEED)

    out = SampleIndex()
    for c in cells:
        pos = np.where(key == c)[0]
        sel = np.concatenate([pos, rng.choice(pos, max_n - len(pos), replace=True)]) \
            if len(pos) < max_n else pos
        for p in sel:
            out.add(idx.paths[p], idx.labels[p], idx.groups[p], idx.country[p])
    return out


# =============================================================================
# FEATURE EXTRACTION (per setting; cached)
# =============================================================================

def _augment(audio, sr):
    for func, params in AUGMENTATION_OPTIONS[np.random.choice(len(AUGMENTATION_OPTIONS))]:
        audio = func(audio, sr, **params)
    return audio


def _spectrogram_raw(y, spec, freq_idx, n_time):
    S = np.abs(librosa.stft(y, n_fft=spec["n_fft"], hop_length=spec["hop"], window="hann")) ** 2
    S = S[freq_idx, :]
    if S.shape[1] > n_time:
        S = S[:, :n_time]
    elif S.shape[1] < n_time:
        S = np.concatenate([S, np.zeros((S.shape[0], n_time - S.shape[1]))], axis=1)
    return np.expand_dims(librosa.power_to_db(S, ref=np.max), axis=0)


def extract_features(setting_name: str) -> dict:
    cfg = SETTINGS[setting_name]
    cache = FEATURE_CACHE / f"{setting_name}.npz"
    if cache.exists():
        print(f"  [{setting_name}] loading cached features")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    audio_dir = CODA_RAW_AUDIO if cfg["audio"] == "raw" else CODA_SEGMENTED_AUDIO
    idx = build_coda_index(audio_dir)
    print(f"  [{setting_name}] {len(idx)} clips before resampling")
    idx = resample_balanced(idx, cfg["resample_by"])
    print(f"  [{setting_name}] {len(idx)} clips after resampling")

    sr = cfg["sr"]
    if cfg["feature"] == "spectrogram_raw":
        spec = cfg["spec"]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=spec["n_fft"])
        freq_idx = np.where((freqs >= spec["fmin"]) & (freqs <= spec["fmax"]))[0]
        n_time = int(round(spec["clip_s"] * sr / spec["hop"]))

    feats, labels, groups, country = [], [], [], []
    np.random.seed(RANDOM_SEED)
    for p in range(len(idx)):
        audio, _ = librosa.load(idx.paths[p], sr=sr)
        if len(audio) < MIN_SAMPLES:
            continue
        audio = _augment(audio, sr)
        if cfg["feature"] == "mel":
            audio = pad_audio(audio, sr, AUDIO_DURATION, pad_by="repeating")
            feat = prep_mel_spectrogram(audio, fmin=cfg["mel"]["fmin"], fmax=cfg["mel"]["fmax"],
                                        sr=sr, n_mels=cfg["mel"]["n_mels"])
        else:
            feat = _spectrogram_raw(audio, spec, freq_idx, n_time)
        feats.append(feat)
        labels.append(idx.labels[p]); groups.append(idx.groups[p]); country.append(idx.country[p])
        if (p + 1) % 500 == 0:
            print(f"    {p + 1}/{len(idx)} features", flush=True)

    out = {"features": np.asarray(feats, dtype=np.float32), "labels": np.array(labels),
           "groups": np.array(groups, dtype=str), "country": np.array(country)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


# =============================================================================
# TRAINING (nested CV, within-CODA predictions)
# =============================================================================

def run_setting(setting_name: str) -> dict:
    cfg = SETTINGS[setting_name]
    print(f"\n=== Setting: {setting_name} ({cfg['model']}, {cfg['feature']}) ===")
    data = extract_features(setting_name)
    X, y, groups, country = data["features"], data["labels"], data["groups"], data["country"]
    label_country = np.array([f"{int(country[i])}_{int(y[i])}" for i in range(len(y))])

    input_adapter = get_input_adapter(cfg["model"])   # per-batch channel/transpose/resize
    vggish_weights = VGGISH_WEIGHTS if (cfg["model"] == "VGGish16" and VGGISH_WEIGHTS.exists()) else None

    seeds_testset = np.random.RandomState(RANDOM_SEED).randint(0, 100, int(np.ceil(N_RANDOM_TESTSETS / 5)))
    seeds_inner = np.random.RandomState(RANDOM_SEED).randint(0, 100, N_RANDOM_TESTSETS)

    record = {"y_true": [], "y_pred": [], "groups": [], "country": [], "countries": CODA_COUNTRIES}
    for ts in range(N_RANDOM_TESTSETS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seeds_testset[ts // 5])
        for i, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), label_country, groups=groups)):
            if i == ts % 5:
                break
        X_tr_all, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        record["y_true"].append([]); record["y_pred"].append([])
        record["groups"].append(groups[test_idx].tolist())
        record["country"].append(country[test_idx].tolist())

        coarse = random_stratified_kfold(X_tr_all, label_country[train_idx], groups[train_idx],
                                         n_splits=N_INNER_FOLDS, random_state=seeds_inner[ts])
        for inner in range(N_INNER_FOLDS):
            print(f"\n[{setting_name}] test set {ts + 1}/{N_RANDOM_TESTSETS} | inner {inner + 1}/{N_INNER_FOLDS}",
                  flush=True)
            X_train, y_train, X_val, y_val, _, _ = prep_X_y(X_tr_all, y_tr, coarse, inner)
            model = build_model(cfg["model"], hidden_nodes=PARAM["hidden_nodes"],
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
            record["y_true"][ts].append(y_te.tolist())
            record["y_pred"][ts].append(preds.tolist())

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRED_DIR / f"{setting_name}.pkl", "wb") as f:
        pickle.dump(record, f)
    print(f"  saved predictions -> {PRED_DIR / f'{setting_name}.pkl'}")
    return record


# =============================================================================
# ANALYSIS (predicted probability vs country prevalence)
# =============================================================================

def cough_level_table(record: dict) -> pd.DataFrame:
    """Cough-level predictions pooled across folds and test sets (one row per cough x fold).

    Matches the provided post-analysis: predicted probabilities are grouped by country
    and TB label at the cough level (not aggregated to subject level).
    """
    rows = []
    for ts in range(len(record["y_pred"])):
        groups = np.array(record["groups"][ts], dtype=str)
        country = np.array(record["country"][ts])
        labels = np.array(record["y_true"][ts][0], dtype=float)
        for fold_preds in record["y_pred"][ts]:
            preds = np.asarray(fold_preds, dtype=float)
            for k in range(len(preds)):
                rows.append({"subject": groups[k], "country": int(country[k]),
                             "label": int(labels[k]), "pred": float(preds[k])})
    return pd.DataFrame(rows)


def true_country_disease_rate() -> dict[int, float]:
    """Subject-level TB+ rate per country, computed from the full CODA metadata.

    The model is trained on resampled (class/country-balanced) data, so per-country
    prevalence must come from the original cohort, not from the prediction records.
    """
    clinical = pd.read_csv(CODA_CLINICAL)
    extra = pd.read_csv(CODA_EXTRA)
    label_by = dict(zip(clinical["participant"].astype(str), clinical["tb_status"]))
    country_by = dict(zip(extra["participant"].astype(str), extra["Country"].astype(str)))
    per_country: dict[int, list[int]] = {}
    for subject, country in country_by.items():
        if country not in CODA_COUNTRIES:
            continue
        label = label_by.get(subject)
        if label is None or pd.isna(label):
            continue
        per_country.setdefault(CODA_COUNTRIES.index(country), []).append(int(label))
    return {c: float(np.mean(v)) for c, v in per_country.items()}


def regress_country_prevalence(df: pd.DataFrame, countries: list[str], rate_by_country: dict[int, float]):
    """Per country: mean cough-level predicted prob (TB- / TB+); regress prob on true prevalence."""
    stats = {}
    for c in df["country"].unique():
        sub = df[df["country"] == c]
        stats[c] = {
            "name": countries[c],
            "disease_rate": rate_by_country.get(c, np.nan),
            "mean_pred_neg": sub[sub["label"] == 0]["pred"].mean(),
            "mean_pred_pos": sub[sub["label"] == 1]["pred"].mean(),
        }
    order = sorted(stats, key=lambda c: stats[c]["disease_rate"])
    rate = np.array([stats[c]["disease_rate"] for c in order])

    fits = {}
    for key in ("mean_pred_neg", "mean_pred_pos"):
        yv = np.array([stats[c][key] for c in order], dtype=float)
        ok = ~np.isnan(yv)
        if ok.sum() >= 2:
            slope = np.polyfit(rate[ok], yv[ok], 1)[0]
            r2 = np.corrcoef(rate[ok], yv[ok])[0, 1] ** 2
        else:
            slope, r2 = np.nan, np.nan
        fits[key] = (slope, r2)
    return stats, order, fits


def plot_predicted_probability(df, record, setting_name, out_path: Path, rate_by_country) -> None:
    countries = list(record["countries"])
    stats, order, fits = regress_country_prevalence(df, countries, rate_by_country)
    names = [stats[c]["name"] for c in order]
    rates = [stats[c]["disease_rate"] for c in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    for offset, label_val, color in [(-width / 2, 0, "#5b9bd5"), (width / 2, 1, "#ed7d7d")]:
        data = [df[(df["country"] == c) & (df["label"] == label_val)]["pred"].values for c in order]
        ax.boxplot(data, positions=np.arange(len(order)) + offset, widths=width,
                   patch_artist=True, boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="black"), showfliers=False)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Predicted Probability of TB")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Country (ordered by disease rate)")

    ax2 = ax.twinx()
    ax2.plot(np.arange(len(order)), rates, "o-", color="blue", label="Disease rate")
    ax2.set_ylabel("Disease Rate", color="blue")
    ax2.set_ylim(0, max(rates) * 1.1 if rates else 1)

    s0, r0 = fits["mean_pred_neg"]
    s1, r1 = fits["mean_pred_pos"]
    ax.set_title(f"{setting_name}\nTB-: slope={s0:.2f}, R²={r0:.2f} | TB+: slope={s1:.2f}, R²={r1:.2f}")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#5b9bd5", markersize=10, label="TB-"),
               plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#ed7d7d", markersize=10, label="TB+")]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [{setting_name}] TB- slope={s0:.3f} R²={r0:.3f} | TB+ slope={s1:.3f} R²={r1:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rate_by_country = true_country_disease_rate()
    for setting_name in SETTINGS_TO_RUN:
        pred_path = PRED_DIR / f"{setting_name}.pkl"
        if pred_path.exists():
            with open(pred_path, "rb") as f:
                record = pickle.load(f)
            print(f"\n=== Setting: {setting_name} (loaded cached predictions) ===")
        else:
            record = run_setting(setting_name)

        df = cough_level_table(record)
        fig_idx = SETTINGS[setting_name]["fig_index"]
        plot_predicted_probability(
            df, record, setting_name,
            OUTPUT_DIR / f"predicted_probability_by_country_label_{fig_idx}.pdf",
            rate_by_country,
        )
    print(f"\nDone. Figures written under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
