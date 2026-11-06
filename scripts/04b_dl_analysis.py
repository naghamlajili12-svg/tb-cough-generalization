"""
scripts/04_dl_analysis.py

Score the DL pipeline predictions (saved by 04_dl_pipeline.py) and produce figures.

For every source dataset and every target dataset, cough-level predicted probabilities
are aggregated to subject level (median over folds and a subject's coughs) and scored
with subject-level ROC-AUC. Results are summarised as mean +/- std over the random
outer test sets.

Outputs (under OUTPUT_DIR):
    best_model_performance_DL.pdf          best DL performance heatmap (paper Fig. 10)
    stratified/<source>_by_device.pdf      within-dataset ROC by recording device (appendix)
    stratified/zambia_by_location.pdf      within-dataset ROC by location (appendix)

Usage:
    python scripts/04_dl_analysis.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PRED_DIR   = Path("results/metrics/dl")          # <source_key>.pkl from 04a_dl_pipeline.py
OUTPUT_DIR = Path("results/figures/dl")

# Heatmap layout (rows = source datasets, columns = target datasets).
SOURCES  = ["Zambia", "TBScreen Passive", "CODA"]
DATASETS = ["Zambia", "Zambia - Audio Recorder", "TBScreen Passive", "TBScreen Forced", "CODA"]
KEY = {
    "Zambia": "zambia", "Zambia - Audio Recorder": "zambia_AR",
    "TBScreen Passive": "tbscreen_passive", "TBScreen Forced": "tbscreen_forced", "CODA": "coda",
}

# Device / location id -> readable name, per source dataset (for stratified plots).
DEVICE_NAMES = {
    "zambia": {1: "Phone A", 2: "Phone B", 3: "Phone C"},
    "tbscreen_passive": {0: "pixel", 1: "yeti", 2: "codec"},
    "coda": {0: "MG", 1: "TZ", 2: "VN"},
}
LOCATION_NAMES = {"zambia": {1: "Chawama", 2: "Kanyama"}}


# =============================================================================
# SUBJECT-LEVEL AGGREGATION
# =============================================================================

def _pool_subject(groups, labels, pred_folds, keep_mask):
    """Pool predictions across folds and a subject's coughs (median) -> subject arrays."""
    preds = np.vstack([np.asarray(f, dtype=float) for f in pred_folds])  # (n_folds, n_samples)
    subj_labels, subj_scores = [], []
    for subj in np.unique(groups[keep_mask]):
        m = (groups == subj) & keep_mask
        if not m.any():
            continue
        subj_labels.append(labels[m][0])
        subj_scores.append(np.nanmedian(preds[:, m]))
    subj_labels = np.array(subj_labels, dtype=float)
    subj_scores = np.array(subj_scores, dtype=float)
    valid = ~np.isnan(subj_labels) & ~np.isnan(subj_scores)
    return subj_labels[valid], subj_scores[valid]


def within_roc_per_testset(result: dict, sample_mask_fn=None):
    """Subject-level (labels, scores) for the source's held-out folds, per random test set."""
    within = result["within"]
    out = []
    for ts in range(len(within["y_pred"])):
        groups = np.array(within["groups"][ts], dtype=str)
        labels = np.array(within["y_true"][ts][0], dtype=float)   # identical across folds
        keep = np.ones(len(groups), bool) if sample_mask_fn is None else sample_mask_fn(within, ts)
        if keep.sum() == 0:
            continue
        out.append(_pool_subject(groups, labels, within["y_pred"][ts], keep))
    return out


def target_roc_per_testset(result: dict, tkey: str):
    """Subject-level (labels, scores) for an external target, per random test set."""
    tgt = result["targets"][tkey]
    groups = np.array(tgt["groups"], dtype=str)
    labels = np.array(tgt["labels"], dtype=float)
    same_cohort = tgt["cohort"] == result["cohort"]
    test_groups = result["within"]["groups"]
    out = []
    for ts in range(len(tgt["y_pred"])):
        keep = np.isin(groups, np.array(test_groups[ts], dtype=str)) if same_cohort else np.ones(len(groups), bool)
        if keep.sum() == 0:
            continue
        out.append(_pool_subject(groups, labels, tgt["y_pred"][ts], keep))
    return out


def mean_std_auc(roc_per_ts):
    aucs = [roc_auc_score(lab, score) for lab, score in roc_per_ts
            if len(np.unique(lab)) == 2]
    return (float(np.nanmean(aucs)), float(np.nanstd(aucs))) if aucs else (np.nan, np.nan)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_performance_heatmap(auc_matrix, results, out_path: Path) -> None:
    annot = np.empty_like(auc_matrix[:, :, 0], dtype=object)
    for i in range(auc_matrix.shape[0]):
        for j in range(auc_matrix.shape[1]):
            m, s = auc_matrix[i, j]
            annot[i, j] = "n/a" if np.isnan(m) else f"{m:.3f}\n±{s:.3f}"

    ylabels = []
    for src in SOURCES:
        r = results.get(KEY[src])
        ylabels.append(f"{src}:\n{r['fmin']}-{r['fmax']} Hz" if r else f"{src}:\nn/a")

    plt.figure(figsize=(6, 4))
    sns.heatmap(auc_matrix[:, :, 0], annot=annot, fmt="", cmap="viridis",
                xticklabels=DATASETS, yticklabels=ylabels, cbar=True,
                cbar_kws={"label": "AUC"}, vmin=0.5, vmax=0.8, annot_kws={"size": 9})
    plt.title("Best Model Performance for Each Training Dataset")
    plt.xlabel("Testing Dataset")
    plt.ylabel("Training Dataset and Model Choices")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stratified_roc(result: dict, field: str, name_map: dict, title: str, out_path: Path) -> None:
    """Within-dataset ROC stratified by a metadata field (device or location)."""
    strata = list(name_map.items())
    fig, axes = plt.subplots(1, len(strata), figsize=(5 * len(strata), 5), squeeze=False)
    for ax, (value, label) in zip(axes[0], strata):
        def mask_fn(within, ts, value=value):
            return np.array(within[field][ts]) == value
        roc_per_ts = within_roc_per_testset(result, mask_fn)
        aucs = []
        for lab, score in roc_per_ts:
            if len(np.unique(lab)) < 2:
                continue
            fpr, tpr, _ = roc_curve(lab, score)
            aucs.append(auc(fpr, tpr))
            ax.plot(fpr, tpr, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--")
        mean, std = (np.nanmean(aucs), np.nanstd(aucs)) if aucs else (np.nan, np.nan)
        ax.set_title(f"{label}: ROC-AUC {mean:.3f} ± {std:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stratified").mkdir(parents=True, exist_ok=True)

    results = {}
    for src in SOURCES:
        path = PRED_DIR / f"{KEY[src]}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                results[KEY[src]] = pickle.load(f)
        else:
            print(f"  missing predictions for {src}: {path}")

    # --- performance heatmap ---
    auc_matrix = np.full((len(SOURCES), len(DATASETS), 2), np.nan)
    for i, src in enumerate(SOURCES):
        result = results.get(KEY[src])
        if result is None:
            continue
        for j, target in enumerate(DATASETS):
            if KEY[target] == KEY[src]:
                auc_matrix[i, j] = mean_std_auc(within_roc_per_testset(result))   # within-dataset
            elif KEY[target] in result["targets"]:
                auc_matrix[i, j] = mean_std_auc(target_roc_per_testset(result, KEY[target]))
        print(f"  {src}: " + ", ".join(
            f"{t}={auc_matrix[i, j, 0]:.3f}" for j, t in enumerate(DATASETS) if not np.isnan(auc_matrix[i, j, 0])
        ))
    plot_performance_heatmap(auc_matrix, results, OUTPUT_DIR / "best_model_performance_DL.pdf")

    # --- stratified within-dataset ROC (appendix) ---
    for src in SOURCES:
        result = results.get(KEY[src])
        if result is None:
            continue
        src_key = KEY[src]
        if src_key in DEVICE_NAMES:
            plot_stratified_roc(
                result, "devices", DEVICE_NAMES[src_key],
                f"{src} Within-dataset Performance by Device",
                OUTPUT_DIR / "stratified" / f"{src_key}_by_device.pdf",
            )
        if src_key in LOCATION_NAMES:
            plot_stratified_roc(
                result, "locations", LOCATION_NAMES[src_key],
                f"{src} Within-dataset Performance by Location",
                OUTPUT_DIR / "stratified" / f"{src_key}_by_location.pdf",
            )

    print(f"\nDone. Figures written under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
