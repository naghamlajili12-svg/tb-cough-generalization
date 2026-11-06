"""
scripts/02_clinical_baseline.py

Clinical-variable baseline for cross-dataset TB classification.

A logistic-regression model is trained on a shared set of demographic/clinical
variables and evaluated under the same dataset-wise external-validation protocol
as the acoustic pipelines:

  * within-dataset performance via nested CV (5 outer / 4 inner folds),
  * external performance by applying each source dataset's 5 outer-fold models
    to the other datasets,
  * feature importance via model coefficients and permutation importance.

Outputs (figures reproduced from the paper) are written to OUTPUT_DIR.
All parameters are defined in the CONFIGURATION section below.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory for reproduced figures (created if missing).
OUTPUT_DIR = Path("results/figures/clinical_baseline")

RANDOM_SEED      = 42
N_OUTER_FOLDS    = 5
N_INNER_FOLDS    = 4
N_PERM_SHUFFLES  = 100
PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "max_iter": [100, 500, 1000],
}

# Shared clinical variables (order defines the feature-vector layout).
CLINICAL_PARAMS = [
    "age",
    "gender",          # 0: female, 1: male
    "smoker",
    "HIV_status",
    "prev_TB",
    "cough_duration",  # expressed in days
    "fever",
    "weight_loss",
    "night_sweats",
]
CLINICAL_PARAMS_CONTINUOUS = ["age", "cough_duration"]


# --- Metadata source paths --------------------------------------------------
# NOTE: data is NOT included in this repo; place each downloaded dataset under
# data/raw/<dataset>/ (see data/README.md).

ZAMBIA_METADATA = Path(
    "data/raw/zambia/Metadata and Codebook/"
    "Metadata and Codebook/GHAI_Final_Data_2023.csv"
)

CODA_METADATA = Path(
    "data/raw/coda/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv"
)
CODA_METADATA_EXTRA = Path(
    "data/raw/coda/meta_data/Clinical/CODA_TB_additional_variables_train.csv"
)

TBSCREEN_METADATA = Path("data/raw/tbscreen/Metadata.csv")


# =============================================================================
# DATASET SPECIFICATIONS
#
# Each spec maps the shared CLINICAL_PARAMS onto a dataset's raw column names
# (`column_map`) and harmonises categorical encodings onto the shared scheme
# (`value_map`). In `value_map`, each entry is {target_value: [raw values...]};
# the sentinel 'rem' marks raw values that should become NaN (and the row dropped).
# =============================================================================

@dataclass
class DatasetSpec:
    name: str
    csv_paths: list[Path]
    column_map: dict[str, str]
    value_map: dict[str, dict]
    subject_col: str
    label_col: str
    merge_key: str | None = None             # merge multiple CSVs on this key
    cough_duration_in_weeks: bool = False     # if True, multiply by 7 -> days


ZAMBIA_SPEC = DatasetSpec(
    name="Zambia",
    csv_paths=[ZAMBIA_METADATA],
    subject_col="barcode",
    label_col="ground_truth_tb",
    column_map={
        "age": "age",
        "gender": "sex",
        "HIV_status": "hiv_status",
        "prev_TB": "prev_tb",
        "smoker": "tobacco_use",
        "cough_duration": "coughdur",
        "fever": "fever",
        "weight_loss": "weightloss",
        "night_sweats": "ngtsweats",
    },
    value_map={
        "gender":         {0: ["f"], 1: ["m"], "rem": ["nan"]},
        "TB":             {0: ["neg"], 1: ["pos"], "rem": ["nan"]},
        "HIV_status":     {0: ["neg"], 1: ["pos"], "rem": ["nan"]},
        "prev_TB":        {0: ["no"], 1: ["yes"], "rem": ["nan"]},
        "smoker":         {0: ["never", "not disclosed"], 1: ["current", "stopped"], "rem": ["nan"]},
        "cough_duration": {0.5 * 7: ["<1wk"], 1.5 * 7: ["1-2wks"], 3.5 * 7: ["3-4wks"],
                           5 * 7: [">4weeks"], 0: ["no cough"], "rem": ["nan"]},
        "fever":          {0: ["no"], 1: ["yes"], "rem": ["nan"]},
        "weight_loss":    {0: ["no"], 1: ["yes"], "rem": ["nan"]},
        "night_sweats":   {0: ["no"], 1: ["yes"], "rem": ["nan"]},
    },
)

CODA_SPEC = DatasetSpec(
    name="CODA",
    csv_paths=[CODA_METADATA, CODA_METADATA_EXTRA],
    merge_key="participant",
    subject_col="participant",
    label_col="tb_status",
    column_map={
        "age": "age",
        "gender": "sex",
        "HIV_status": "HIVstatus",
        "prev_TB": "tb_prior",
        "smoker": "smoke_lweek",
        "cough_duration": "reported_cough_dur",
        "fever": "fever",
        "weight_loss": "weight_loss",
        "night_sweats": "night_sweats",
    },
    value_map={
        "gender":       {0: ["Female"], 1: ["Male"], "rem": ["nan"]},
        "TB":           {0: ["0"], 1: ["1"], "rem": ["nan"]},
        "HIV_status":   {0: ["Negative"], 1: ["Positive"], "rem": ["nan", "Unknown"]},
        "prev_TB":      {0: ["No"], 1: ["Yes"], "rem": ["Not sure", "nan"]},
        "smoker":       {0: ["No"], 1: ["Yes"], "rem": ["nan"]},
        "fever":        {0: ["No"], 1: ["Yes"], "rem": ["nan"]},
        "weight_loss":  {0: ["No"], 1: ["Yes"], "rem": ["nan"]},
        "night_sweats": {0: ["No"], 1: ["Yes"], "rem": ["nan"]},
    },
)

TBSCREEN_SPEC = DatasetSpec(
    name="TBScreen",
    csv_paths=[TBSCREEN_METADATA],
    subject_col="subject",
    label_col="Label",
    cough_duration_in_weeks=True,   # cough_drtn is in weeks -> convert to days
    column_map={
        "age": "age",
        "gender": "gndr_a",
        "HIV_status": "hiv_infected_a",
        "prev_TB": "prev_tb",
        "smoker": "smkr_a",
        "cough_duration": "cough_drtn",
        "fever": "fever",
        "weight_loss": "un_wght_loss",
        "night_sweats": "nght_swts",
    },
    value_map={
        "gender":       {0: ["1.0"], 1: ["0.0"], "rem": ["nan"]},
        "TB":           {0: ["NTB"], 1: ["TB"], "rem": ["nan"]},
        "HIV_status":   {0: ["0.0"], 1: ["1.0"], "rem": ["nan", "Unknown"]},
        "prev_TB":      {0: ["0.0"], 1: ["1.0", "2.0"], "rem": ["nan"]},
        "smoker":       {0: ["0.0"], 1: ["1.0", "2.0"], "rem": ["nan"]},
        "fever":        {0: ["0.0"], 1: ["1.0"], "rem": ["nan"]},
        "weight_loss":  {0: ["0.0"], 1: ["1.0"], "rem": ["nan"]},
        "night_sweats": {0: ["0.0"], 1: ["1.0"], "rem": ["nan"]},
    },
)

# Order is shared by every downstream array, plot axis, and the AUC matrix.
DATASET_SPECS = [ZAMBIA_SPEC, TBSCREEN_SPEC, CODA_SPEC]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def _remap_categorical(column: pd.Series, mapping: dict) -> pd.Series:
    """Replace raw category strings with their harmonised target values."""
    column = column.copy()
    for target, raw_values in mapping.items():
        for raw in raw_values:
            column = column.replace(raw, target)
    return column


def build_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset and return (features, subject_ids, labels) at subject level.

    Features follow the CLINICAL_PARAMS order. Rows with any missing feature or
    a missing label are dropped, and duplicate subjects are reduced to their
    first occurrence.
    """
    df = pd.read_csv(spec.csv_paths[0])
    for extra_path in spec.csv_paths[1:]:
        df = pd.merge(df, pd.read_csv(extra_path), on=spec.merge_key, how="left")

    # --- features ---
    columns = []
    for param in CLINICAL_PARAMS:
        raw = df[spec.column_map[param]].astype(str)
        if param in spec.value_map:
            raw = _remap_categorical(raw, spec.value_map[param])
        raw = raw.replace("rem", np.nan).astype(float)
        columns.append(raw)
    features = np.column_stack(columns)

    # --- labels (harmonised to 0/1) ---
    labels = df[spec.label_col].values.astype(str)
    for target, raw_values in spec.value_map["TB"].items():
        if target == "rem":
            continue
        for raw in raw_values:
            labels[labels == raw] = target
    labels = labels.astype(float)

    subjects = df[spec.subject_col].values.astype(str)

    if spec.cough_duration_in_weeks:
        idx = CLINICAL_PARAMS.index("cough_duration")
        features[:, idx] *= 7

    # --- drop incomplete rows ---
    valid = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    features, subjects, labels = features[valid], subjects[valid], labels[valid]

    # --- deduplicate subjects (keep first occurrence) ---
    if len(np.unique(subjects)) < len(subjects):
        print(f"  [{spec.name}] repeated subject_ids found; keeping first occurrence.")
        _, first_idx = np.unique(subjects, return_index=True)
        first_idx.sort()
        features, subjects, labels = features[first_idx], subjects[first_idx], labels[first_idx]

    n_pos = int(labels.sum())
    print(f"  [{spec.name}] {len(labels)} subjects | TB+ {n_pos} | TB- {len(labels) - n_pos}")
    return features, subjects, labels


# =============================================================================
# MODELLING
# =============================================================================

def permutation_importance_auc(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_shuffles: int = N_PERM_SHUFFLES,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Per-feature ROC-AUC drop after permuting that feature, averaged over shuffles."""
    rng = np.random.default_rng(random_seed)
    baseline = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_shuffles):
            X_shuffled = X.copy()
            X_shuffled[:, j] = rng.permutation(X_shuffled[:, j])
            scores.append(roc_auc_score(y, clf.predict_proba(X_shuffled)[:, 1]))
        importances[j] = baseline - np.mean(scores)
    return importances


@dataclass
class NestedCVResult:
    models: list = field(default_factory=list)            # one per outer fold
    within_auc_mean: float = 0.0
    within_auc_std: float = 0.0
    coeff_importance: np.ndarray = None                   # (n_outer_folds, n_features)
    perm_importance: np.ndarray = None                    # (n_outer_folds, n_features)
    fold_train: list = field(default_factory=list)        # (y_true, y_score) per fold
    fold_test: list = field(default_factory=list)         # (y_true, y_score) per fold


def run_nested_cv(features: np.ndarray, labels: np.ndarray) -> NestedCVResult:
    """Subject-level nested CV: inner grid search, outer fold for evaluation.

    Features are already subject-level (one row per subject), so a dummy group
    per row makes StratifiedGroupKFold behave as subject-wise stratified CV.
    """
    groups = np.arange(len(labels))
    outer_cv = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    result = NestedCVResult(
        coeff_importance=np.zeros((N_OUTER_FOLDS, features.shape[1])),
        perm_importance=np.zeros((N_OUTER_FOLDS, features.shape[1])),
    )
    test_aucs = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(features, labels, groups)):
        X_train, y_train, g_train = features[train_idx], labels[train_idx], groups[train_idx]
        X_test, y_test = features[test_idx], labels[test_idx]

        search = GridSearchCV(
            LogisticRegression(),
            PARAM_GRID,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=g_train)
        clf = search.best_estimator_

        y_train_pred = clf.predict_proba(X_train)[:, 1]
        y_test_pred = clf.predict_proba(X_test)[:, 1]

        result.models.append(clf)
        result.coeff_importance[fold] = clf.coef_.flatten()
        result.perm_importance[fold] = permutation_importance_auc(X_test, y_test, clf)
        result.fold_train.append((y_train, y_train_pred))
        result.fold_test.append((y_test, y_test_pred))
        test_aucs.append(roc_auc_score(y_test, y_test_pred))

    result.within_auc_mean = float(np.nanmean(test_aucs))
    result.within_auc_std = float(np.nanstd(test_aucs))
    return result


def evaluate_external(
    models: list, features: np.ndarray, labels: np.ndarray
) -> tuple[float, float, list[np.ndarray]]:
    """Apply each outer-fold model to an external dataset; return mean/std AUC and ROC curves."""
    aucs, curves = [], []
    for model in models:
        scores = model.predict_proba(features)[:, 1]
        fpr, tpr, _ = roc_curve(labels, scores)
        aucs.append(auc(fpr, tpr))
        curves.append((fpr, tpr, aucs[-1]))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs)), curves


# =============================================================================
# PLOTTING
# =============================================================================

def plot_within_roc(name: str, result: NestedCVResult, out_path: Path) -> None:
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Trained and tested on {name}", fontsize=14)

    for fold, ((y_tr, p_tr), (y_te, p_te)) in enumerate(zip(result.fold_train, result.fold_test), 1):
        for ax, (y, p) in ((ax_train, (y_tr, p_tr)), (ax_test, (y_te, p_te))):
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"Fold {fold} (AUC = {auc(fpr, tpr):.2f})")

    train_aucs = [auc(*roc_curve(y, p)[:2]) for y, p in result.fold_train]
    ax_train.set_title(f"Train ROC (AUC = {np.nanmean(train_aucs):.2f} ± {np.nanstd(train_aucs):.2f})")
    ax_test.set_title(
        f"Test ROC (AUC = {result.within_auc_mean:.2f} ± {result.within_auc_std:.2f})"
    )
    for ax in (ax_train, ax_test):
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_external_roc(
    source_name: str,
    targets: list[tuple[str, list]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), squeeze=False)
    for ax, (target_name, curves) in zip(axes[0], targets):
        aucs = [a for _, _, a in curves]
        for i, (fpr, tpr, a) in enumerate(curves, 1):
            ax.plot(fpr, tpr, label=f"Model {i} (AUC = {a:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title(
            f"Trained on {source_name}, tested on {target_name}\n"
            f"(AUC = {np.nanmean(aucs):.2f} ± {np.nanstd(aucs):.2f})"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_performance_heatmap(auc_matrix: np.ndarray, names: list[str], out_path: Path) -> None:
    annot = np.empty_like(auc_matrix[:, :, 0], dtype=object)
    for i in range(len(names)):
        for j in range(len(names)):
            annot[i, j] = f"{auc_matrix[i, j, 0]:.3f}\n±{auc_matrix[i, j, 1]:.3f}"

    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(
        auc_matrix[:, :, 0], annot=annot, fmt="", cmap="viridis",
        xticklabels=names, yticklabels=names, cbar=True, cbar_kws={"label": "AUC"},
        annot_kws={"size": 10}, vmin=0.5, vmax=0.8,
    )
    plt.title("Clinical Variable Generalization Performance")
    plt.xlabel("Testing Dataset")
    plt.ylabel("Training Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(
    coeff_importance: np.ndarray,   # (n_datasets, n_folds, n_features)
    perm_importance: np.ndarray,
    names: list[str],
    out_path: Path,
) -> None:
    x = np.arange(len(CLINICAL_PARAMS))
    width = 0.25
    fig, (ax_coeff, ax_perm) = plt.subplots(1, 2, figsize=(12, 5))

    for d, name in enumerate(names):
        ax_coeff.bar(x + d * width, np.nanmean(coeff_importance[d], axis=0),
                     yerr=np.nanstd(coeff_importance[d], axis=0), capsize=5,
                     width=width, label=f"Trained on {name}")
        ax_perm.bar(x + d * width, np.nanmean(perm_importance[d], axis=0),
                    yerr=np.nanstd(perm_importance[d], axis=0), capsize=5,
                    width=width, label=f"Trained on {name}")

    for ax, title, ylabel in (
        (ax_coeff, "Feature Importance from Model Coefficients", "Mean Coefficient Value"),
        (ax_perm, "Feature Importance from Permutation Importance", "Mean Decrease in AUC"),
    ):
        ax.set_xticks(x + width)
        ax.set_xticklabels(CLINICAL_PARAMS, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Clinical Features")
        ax.set_title(title)
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [spec.name for spec in DATASET_SPECS]

    # --- load all datasets ---
    print("Loading datasets...")
    datasets = [build_dataset(spec) for spec in DATASET_SPECS]
    features_list = [f for f, _, _ in datasets]
    labels_list = [lab for _, _, lab in datasets]

    # --- standardise continuous features (scaler fit on all datasets pooled) ---
    for param in CLINICAL_PARAMS_CONTINUOUS:
        idx = CLINICAL_PARAMS.index(param)
        scaler = StandardScaler().fit(
            np.concatenate([f[:, idx] for f in features_list]).reshape(-1, 1)
        )
        for f in features_list:
            f[:, idx] = scaler.transform(f[:, idx].reshape(-1, 1)).flatten()

    n_datasets = len(DATASET_SPECS)
    n_features = len(CLINICAL_PARAMS)
    auc_matrix = np.zeros((n_datasets, n_datasets, 2))               # (train, test, [mean, std])
    coeff_importance = np.zeros((n_datasets, N_OUTER_FOLDS, n_features))
    perm_importance = np.zeros((n_datasets, N_OUTER_FOLDS, n_features))

    # --- train per source dataset, then evaluate externally ---
    for src, (src_name, X, y) in enumerate(zip(names, features_list, labels_list)):
        print(f"\nTraining and within-dataset evaluation: {src_name}")
        result = run_nested_cv(X, y)
        auc_matrix[src, src] = (result.within_auc_mean, result.within_auc_std)
        coeff_importance[src] = result.coeff_importance
        perm_importance[src] = result.perm_importance
        plot_within_roc(src_name, result, OUTPUT_DIR / f"{src_name}_within_roc.png")

        print(f"Cross-dataset evaluation with models trained on {src_name}...")
        targets = []
        for tgt in range(n_datasets):
            if tgt == src:
                continue
            mean_auc, std_auc, curves = evaluate_external(
                result.models, features_list[tgt], labels_list[tgt]
            )
            auc_matrix[src, tgt] = (mean_auc, std_auc)
            targets.append((names[tgt], curves))
        plot_external_roc(src_name, targets, OUTPUT_DIR / f"{src_name}_external_roc.png")

    # --- summary figures (reproduced from paper) ---
    plot_performance_heatmap(auc_matrix, names, OUTPUT_DIR / "best_model_performance_clinical.pdf")
    plot_feature_importance(
        coeff_importance, perm_importance, names, OUTPUT_DIR / "feature_importance_clinical.pdf"
    )
    print(f"\nDone. Figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
