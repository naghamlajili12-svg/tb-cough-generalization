"""
scripts/06b_zambia_device_analysis.py

Score the Zambia device-generalization predictions (from 06a_zambia_device_pipeline.py)
and produce the figures.

Subject-level ROC-AUC is computed as follows (matching the original analysis):
  * within-device (diagonal / "on training devices"): per random test set, pool a
    subject's cough predictions across inner folds (mean), score, average over test sets;
  * cross-device / held-out device: pool a subject's predictions across all folds and
    test sets (median), restricted to the held-out subjects of each test set, single AUC.

Outputs (under OUTPUT_DIR):
    zambia_single_device_generalization.pdf   4x4 generalization heatmap + difference
    zambia_multi_device_generalization.pdf     train-on-3 vs held-out-device heatmap
    Comparison_on_training_devices.pdf         single- vs multi-device bar comparison

Usage:
    python scripts/06b_zambia_device_analysis.py
"""

from __future__ import annotations

import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PRED_DIR   = Path("results/metrics/zambia_device")
OUTPUT_DIR = Path("results/figures/zambia_device")


# =============================================================================
# SUBJECT-LEVEL SCORING
# =============================================================================

def _safe_auc(labels, scores):
    labels, scores = np.asarray(labels, float), np.asarray(scores, float)
    valid = ~np.isnan(labels) & ~np.isnan(scores)
    labels, scores = labels[valid], scores[valid]
    return roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan


def within_auc(within: dict) -> float:
    """Mean over test sets of the within-device subject-level AUC (subject pred = mean)."""
    aucs = []
    for ts in range(len(within["y_pred"])):
        groups = np.array(within["groups"][ts], dtype=str)
        labels = np.array(within["y_true"][ts][0], dtype=float)
        preds = np.vstack([np.asarray(f, float) for f in within["y_pred"][ts]])  # (folds, M)
        s_lab, s_score = [], []
        for subj in np.unique(groups):
            m = groups == subj
            s_lab.append(labels[m][0]); s_score.append(np.mean(preds[:, m]))
        aucs.append(_safe_auc(s_lab, s_score))
    return float(np.nanmean(aucs))


def cross_auc(cross_per_ts, target_groups, target_labels, test_groups_per_ts) -> float:
    """Single pooled AUC on a target device (subject pred = median over folds/test sets).

    Only subjects held out in a given test set contribute that test set's predictions,
    avoiding train/test subject leakage (devices share subjects).
    """
    target_groups = np.asarray(target_groups, dtype=str)
    target_labels = np.asarray(target_labels, dtype=float)
    subj_scores: dict[str, list] = defaultdict(list)
    subj_label: dict[str, float] = {}
    for ts in range(len(cross_per_ts)):
        test_subjects = set(np.asarray(test_groups_per_ts[ts], dtype=str))
        keep = np.isin(target_groups, list(test_subjects))
        if not keep.any():
            continue
        for fold_preds in cross_per_ts[ts]:
            fp = np.asarray(fold_preds, float)
            for subj in np.unique(target_groups[keep]):
                m = (target_groups == subj)
                subj_scores[subj].extend(fp[m].tolist())
                subj_label[subj] = target_labels[m][0]
    labels = [subj_label[s] for s in subj_scores]
    scores = [np.median(v) for v in subj_scores.values()]
    return _safe_auc(labels, scores)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_single_device(single: dict, out_path: Path) -> np.ndarray:
    devices = single["devices"]
    n = len(devices)
    gen = np.full((n, n), np.nan)   # rows = training device, cols = testing device
    for i, train in enumerate(devices):
        per = single["per_train"][train]
        gen[i, i] = within_auc(per["within"])
        for j, test in enumerate(devices):
            if i == j:
                continue
            gen[i, j] = cross_auc(
                per["cross"][test], single["device_meta"][test]["groups"],
                single["device_meta"][test]["labels"], per["within"]["groups"],
            )

    diff = gen.copy()
    for i in range(n):
        diff[i, i] = np.nan
        for j in range(n):
            if i != j:
                diff[i, j] = gen[i, j] - gen[i, i]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(gen, annot=True, fmt=".3f", cmap="Blues", xticklabels=devices, yticklabels=devices, ax=ax1)
    ax1.set_xlabel("Testing Device"); ax1.set_ylabel("Training Device")
    ax1.set_title("Generalization Performance (ROC AUC)")
    ax1.tick_params(axis="x", rotation=45)
    sns.heatmap(diff, annot=True, fmt=".3f", cmap="RdBu", center=0,
                xticklabels=devices, yticklabels=devices, ax=ax2)
    ax2.set_xlabel("Testing Device"); ax2.set_ylabel("Training Device")
    ax2.set_title("Generalization Performance Difference (ROC AUC)")
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return gen


def analyse_multi_device(multi: dict):
    devices = multi["devices"]
    on_train = np.full(len(devices), np.nan)     # within of the 3-device model
    on_heldout = np.full(len(devices), np.nan)   # generalization to the held-out device
    for j, d in enumerate(devices):
        per = multi["per_heldout"][d]
        on_train[j] = within_auc(per["within"])
        on_heldout[j] = cross_auc(
            per["heldout"], multi["device_meta"][d]["groups"],
            multi["device_meta"][d]["labels"], per["within"]["groups"],
        )
    return devices, on_train, on_heldout


def plot_multi_device(devices, on_train, on_heldout, out_path: Path) -> None:
    table = np.vstack([on_train, on_heldout])
    plt.figure(figsize=(5, 3))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="Blues", xticklabels=devices,
                yticklabels=["On training\ndevices", "Generalize to\ntesting device"])
    plt.ylabel("Scenario"); plt.xlabel("Testing Device")
    plt.yticks(rotation=0); plt.xticks(rotation=45)
    plt.title("Generalization Performance (ROC AUC)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_comparison_bar(devices, gen_single, on_heldout, out_path: Path) -> None:
    # single-device: mean over training devices of cross performance onto each test device
    single_mean = np.full(len(devices), np.nan)
    masked = gen_single.copy()
    np.fill_diagonal(masked, np.nan)
    single_mean = np.nanmean(masked, axis=0)   # column mean (per testing device)

    x = np.arange(len(devices))
    width = 0.35
    plt.figure(figsize=(5.5, 3.5))
    plt.bar(x - width / 2, single_mean, width, label="Train on single device (mean)")
    plt.bar(x + width / 2, on_heldout, width, label="Train on other three devices")
    plt.xlabel("Testing Device"); plt.ylabel("Subject-level ROC AUC")
    plt.title("Performance Comparison on Testing on the Same Device")
    plt.xticks(x, devices, rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRED_DIR / "single_device.pkl", "rb") as f:
        single = pickle.load(f)
    with open(PRED_DIR / "multi_device.pkl", "rb") as f:
        multi = pickle.load(f)

    gen_single = plot_single_device(single, OUTPUT_DIR / "zambia_single_device_generalization.pdf")
    devices, on_train, on_heldout = analyse_multi_device(multi)
    plot_multi_device(devices, on_train, on_heldout, OUTPUT_DIR / "zambia_multi_device_generalization.pdf")
    plot_comparison_bar(devices, gen_single, on_heldout, OUTPUT_DIR / "Comparison_on_training_devices.pdf")

    print("Single-device generalization (rows=train, cols=test):")
    print(np.round(gen_single, 3))
    print("Multi-device (held-out device):")
    for d, t, h in zip(devices, on_train, on_heldout):
        print(f"  {d:15s} on-train={t:.3f}  held-out={h:.3f}")
    print(f"\nDone. Figures written under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
