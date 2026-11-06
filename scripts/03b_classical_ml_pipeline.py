"""
scripts/03b_classical_ml_pipeline.py

Classical ML pipeline (cross-dataset).

For each source dataset, a classifier is selected with subject-level nested CV and
then evaluated on every target dataset. To keep the reproduction tractable, only the
best-performing configuration per source dataset (as reported in the paper) is run:

    Zambia            : Random Forest    | CLAP embeddings     | without RFE
    TBScreen Passive  : Logistic Reg.    | CLAP embeddings     | with RFE
    CODA              : Random Forest    | time/freq features  | with RFE

The full grid over classifiers / feature families / feature-selection settings is not
reproduced here. Hyperparameter grids can be expanded freely (see PARAM_GRIDS).

Produces the best-model performance heatmap (paper: best_model_performance_classicalML.pdf).

Inputs (precomputed by scripts/03a_extract_features.py), per feature type:
    FEATURES_DIR/<feature_type>/<key>_features.npy            (n_samples, n_features)
    FEATURES_DIR/<feature_type>/<key>_labels.npy              (n_samples,)
    FEATURES_DIR/<feature_type>/<key>_groups.npy              (n_samples,) subject ids
    FEATURES_DIR/<feature_type>/<key>_device_location_label.npy (n_samples,) stratify key
  feature_type in {clap, timefreq}; key in
    {zambia, zambia_AR, tbscreen_passive, tbscreen_forced, coda}

Usage:
    python scripts/03b_classical_ml_pipeline.py
"""

from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = Path("data/processed/features")        # produced by 03a
OUTPUT_DIR   = Path("results/figures/classical_ml")
METRICS_DIR  = Path("results/metrics/classical_ml")

RANDOM_SEED   = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

# RFE (backward elimination) settings
RFE_MIN_FEATURES     = 100
RFE_ELIMINATION_STEP = 50
RFE_PERM_SHUFFLES    = 20

# Datasets: display name, file key, and whether it is a train-eligible source.
DATASETS = [
    ("Zambia",                 "zambia",            True),
    ("Zambia - Audio Recorder", "zambia_AR",        False),
    ("TBScreen Passive",       "tbscreen_passive",  True),
    ("TBScreen Forced",        "tbscreen_forced",   False),
    ("CODA",                   "coda",              True),
]
DISPLAY_NAMES = [d[0] for d in DATASETS]
KEY_BY_NAME   = {d[0]: d[1] for d in DATASETS}

# Best configuration per source dataset (model, feature type, use RFE).
BEST_CONFIGS = {
    "Zambia":           {"model": "rf",      "feature": "clap",     "rfe": False},
    "TBScreen Passive": {"model": "log_reg", "feature": "clap",     "rfe": True},
    "CODA":             {"model": "rf",      "feature": "timefreq", "rfe": True},
}

# Hyperparameter grids (expand as desired; the paper search was broader).
PARAM_GRIDS = {
    "log_reg": {"C": [0.001, 0.01, 0.1, 1, 10, 100], "max_iter": [1000]},
    "rf":      {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "min_samples_split": [5, 10]},
    "gb":      {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
}

MODEL_LABELS = {"log_reg": "LR", "rf": "RF", "gb": "GB"}


# =============================================================================
# MODELS
# =============================================================================

def make_model(model_name: str, params: dict):
    if model_name == "log_reg":
        return LogisticRegression(**params)
    if model_name == "rf":
        return RandomForestClassifier(random_state=RANDOM_SEED, **params)
    if model_name == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_SEED, **params)
    raise ValueError(f"Unknown model: {model_name}")


def param_combinations(model_name: str) -> list[dict]:
    grid = PARAM_GRIDS[model_name]
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# =============================================================================
# FEATURE SELECTION (RFE via permutation-importance backward elimination)
# =============================================================================

def permutation_importance(X, y, clf, n_shuffles=RFE_PERM_SHUFFLES, seed=RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    baseline = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_shuffles):
            Xs = X.copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            scores.append(roc_auc_score(y, clf.predict_proba(Xs)[:, 1]))
        importances[j] = baseline - np.mean(scores)
    return importances


def backward_elimination(X_tr, y_tr, X_val, y_val, clf, min_features, elimination_step):
    clf.fit(X_tr, y_tr)
    best_perf = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    best_features = np.arange(X_tr.shape[1])
    while len(best_features) > min_features:
        n_drop = max(elimination_step, int(elimination_step / 100 * len(best_features)))
        importances = permutation_importance(X_val[:, best_features], y_val, clf)
        drop = np.argsort(importances)[:n_drop]
        candidate = np.delete(best_features, drop)
        clf.fit(X_tr[:, candidate], y_tr)
        perf = roc_auc_score(y_val, clf.predict_proba(X_val[:, candidate])[:, 1])
        if perf >= best_perf:
            best_perf, best_features = perf, candidate
        else:
            break
    return best_features


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(key: str, feature_type: str) -> dict:
    """Load one dataset's feature arrays; standardise + clip time/freq features."""
    base = FEATURES_DIR / feature_type
    features = np.load(base / f"{key}_features.npy")
    if feature_type == "timefreq":
        features = StandardScaler().fit_transform(
            np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        features = np.clip(features, -5, 5)  # guard against extreme values
    return {
        "features": features,
        "labels": np.load(base / f"{key}_labels.npy"),
        "groups": np.load(base / f"{key}_groups.npy", allow_pickle=True).astype(str),
        "dll": np.load(base / f"{key}_device_location_label.npy", allow_pickle=True).astype(str),
    }


def aggregate_to_subject(labels, scores, groups):
    """Aggregate cough-level scores to subject level (median); return subject arrays."""
    subjects = np.unique(groups)
    subj_labels = np.zeros(len(subjects))
    subj_scores = np.zeros(len(subjects))
    for i, subj in enumerate(subjects):
        mask = groups == subj
        subj_labels[i] = labels[mask][0]
        subj_scores[i] = np.nanmedian(scores[mask])
    valid = ~np.isnan(subj_scores) & ~np.isnan(subj_labels)
    return subj_labels[valid], subj_scores[valid]


# =============================================================================
# NESTED CV
# =============================================================================

def run_nested_cv(data: dict, model_name: str, use_rfe: bool):
    """Train per outer fold; return models, feature masks, held-out subjects, within AUC."""
    X, y, groups, dll = data["features"], data["labels"], data["groups"], data["dll"]
    grid = param_combinations(model_name)
    outer = StratifiedGroupKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner = StratifiedGroupKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    models, masks, fold_subjects, within_aucs = [], [], [], []

    # One progress bar over all inner work units so it advances smoothly:
    #   hyperparameter fits + (RFE inner folds if enabled) + final refit, per outer fold.
    units_per_fold = N_INNER_FOLDS * len(grid) + (N_INNER_FOLDS if use_rfe else 0) + 1
    pbar = tqdm(total=N_OUTER_FOLDS * units_per_fold, desc="  nested CV", leave=False)

    for fold, (tv_idx, test_idx) in enumerate(outer.split(X, dll, groups), 1):
        X_tv, y_tv, g_tv = X[tv_idx], y[tv_idx], groups[tv_idx]
        X_te, y_te, g_te = X[test_idx], y[test_idx], groups[test_idx]

        # --- inner hyperparameter search (mean validation AUC) ---
        inner_splits = list(inner.split(X_tv, y_tv, g_tv))
        perf = np.full((len(inner_splits), len(grid)), np.nan)
        for v, (tr_i, val_i) in enumerate(inner_splits):
            for p, params in enumerate(grid):
                clf = make_model(model_name, params)
                clf.fit(X_tv[tr_i], y_tv[tr_i])
                perf[v, p] = roc_auc_score(y_tv[val_i], clf.predict_proba(X_tv[val_i])[:, 1])
                pbar.update(1)
                pbar.set_postfix(fold=f"{fold}/{N_OUTER_FOLDS}", stage="hp-search")
        best_params = grid[int(np.argmax(np.nanmean(perf, axis=0)))]

        # --- feature selection (features kept in >= 50% of inner folds) ---
        if use_rfe:
            vote = np.zeros((len(inner_splits), X.shape[1]), dtype=bool)
            for v, (tr_i, val_i) in enumerate(inner_splits):
                clf = make_model(model_name, best_params)
                feats = backward_elimination(
                    X_tv[tr_i], y_tv[tr_i], X_tv[val_i], y_tv[val_i], clf,
                    RFE_MIN_FEATURES, RFE_ELIMINATION_STEP,
                )
                vote[v, feats] = True
                pbar.update(1)
                pbar.set_postfix(fold=f"{fold}/{N_OUTER_FOLDS}", stage="RFE")
            mask = vote.mean(axis=0) >= 0.5
        else:
            mask = np.ones(X.shape[1], dtype=bool)

        # --- refit on full train, evaluate on held-out fold (subject-level) ---
        clf = make_model(model_name, best_params)
        clf.fit(X_tv[:, mask], y_tv)
        scores = clf.predict_proba(X_te[:, mask])[:, 1]
        s_lab, s_score = aggregate_to_subject(y_te, scores, g_te)

        models.append(clf)
        masks.append(mask)
        fold_subjects.append(np.unique(g_te))
        within_aucs.append(roc_auc_score(s_lab, s_score))
        pbar.update(1)
        pbar.set_postfix(fold=f"{fold}/{N_OUTER_FOLDS}", fold_auc=f"{within_aucs[-1]:.3f}",
                         n_feat=int(mask.sum()))

    pbar.close()
    return models, masks, fold_subjects, float(np.nanmean(within_aucs)), float(np.nanstd(within_aucs))


def evaluate_external(models, masks, fold_subjects, data, restrict_to_fold_subjects: bool):
    """Apply each fold model to an external dataset; subject-level AUC mean/std over models."""
    X, y, groups = data["features"], data["labels"], data["groups"]
    aucs = []
    for clf, mask, subjects in zip(models, masks, fold_subjects):
        scores = clf.predict_proba(X[:, mask])[:, 1]
        keep = np.isin(groups, subjects) if restrict_to_fold_subjects else np.ones(len(groups), bool)
        s_lab, s_score = aggregate_to_subject(y[keep], scores[keep], groups[keep])
        aucs.append(roc_auc_score(s_lab, s_score))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))


# =============================================================================
# PLOTTING
# =============================================================================

def plot_best_model_heatmap(auc_matrix, train_names, configs, out_path: Path) -> None:
    annot = np.empty_like(auc_matrix[:, :, 0], dtype=object)
    for i in range(auc_matrix.shape[0]):
        for j in range(auc_matrix.shape[1]):
            annot[i, j] = f"{auc_matrix[i, j, 0]:.3f}\n±{auc_matrix[i, j, 1]:.3f}"

    ylabels = [
        f"{name}:\n{MODEL_LABELS[cfg['model']]}, {cfg['feature']}, "
        f"{'w/ RFE' if cfg['rfe'] else 'w/o RFE'}"
        for name, cfg in zip(train_names, configs)
    ]
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        auc_matrix[:, :, 0], annot=annot, fmt="", cmap="viridis",
        xticklabels=DISPLAY_NAMES, yticklabels=ylabels,
        cbar=True, cbar_kws={"label": "AUC"}, vmin=0.5, vmax=0.8, annot_kws={"size": 9},
    )
    plt.title("Best Model Performance for Each Training Dataset")
    plt.xlabel("Testing Dataset")
    plt.ylabel("Training Dataset and Model Choices")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_names = [name for name in DISPLAY_NAMES if name in BEST_CONFIGS]
    configs = [BEST_CONFIGS[name] for name in train_names]
    auc_matrix = np.zeros((len(train_names), len(DISPLAY_NAMES), 2))  # (train, test, [mean, std])

    for row, (train_name, cfg) in enumerate(zip(train_names, configs)):
        print(f"\n=== Source: {train_name}  ({MODEL_LABELS[cfg['model']]}, "
              f"{cfg['feature']}, {'w/ RFE' if cfg['rfe'] else 'w/o RFE'}) ===")

        # All datasets must share the source's feature space for cross-evaluation.
        datasets = {name: load_dataset(KEY_BY_NAME[name], cfg["feature"]) for name in DISPLAY_NAMES}

        models, masks, fold_subjects, within_mean, within_std = run_nested_cv(
            datasets[train_name], cfg["model"], cfg["rfe"]
        )
        print(f"  within-dataset subject-level AUC = {within_mean:.3f} ± {within_std:.3f}")

        for col, test_name in enumerate(DISPLAY_NAMES):
            if test_name == train_name:
                auc_matrix[row, col] = (within_mean, within_std)
                continue
            # Restrict to held-out subjects when both sides are Zambia (shared subjects).
            restrict = ("Zambia" in train_name) and ("Zambia" in test_name)
            mean_auc, std_auc = evaluate_external(
                models, masks, fold_subjects, datasets[test_name], restrict
            )
            auc_matrix[row, col] = (mean_auc, std_auc)
            print(f"  -> {test_name:25s} AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    np.save(METRICS_DIR / "best_model_performance.npy", auc_matrix)
    plot_best_model_heatmap(
        auc_matrix, train_names, configs, OUTPUT_DIR / "best_model_performance_classicalML.pdf"
    )
    print(f"\nDone. Figure written to {OUTPUT_DIR}/best_model_performance_classicalML.pdf")


if __name__ == "__main__":
    main()
