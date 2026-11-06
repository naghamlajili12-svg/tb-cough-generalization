"""
scripts/clinical_baseline_eval.py

Clinical-variable baseline for TB classification on the Zambia dataset.

Logistic regression trained and evaluated using the pre-assigned patient-level
5-fold CV splits (seed 42) from patients.csv.

Features used (safe — no leakage from lab/imaging results):
    age, sex, coughdur, fever, weightloss, ngtsweats,
    hiv_status, tobacco_use, bmi, prev_tb
    (+ haemoptysis in a sensitivity analysis)

Excluded per instructions: lam, smear, xpert, culture, abn_*, tb_* columns.

Two evaluation settings:
  1. PRIMARY — evaluate on the 317 patients with has_audio==True
               (the same set used for acoustic models), using folds 0-4.
  2. SECONDARY — evaluate on all 1,621 patients.  The 1,304 non-audio
                 patients lack fold assignments, so we generate new
                 patient-level stratified 5-fold splits (seed 42).

AUC is reported with a bootstrap 95% CI (2000 resamples).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PATIENTS_CSV = ROOT / "data" / "meta" / "patients.csv"
OUTPUT_DIR = ROOT / "results" / "clinical_baseline"

RANDOM_SEED = 42
N_BOOTSTRAP = 2000

# ── hyperparameter grid (same as the paper's script) ─────────────────────────
PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "max_iter": [500, 1000],
}

# ── feature definitions ──────────────────────────────────────────────────────
#
# Encoding choices follow the existing 02_clinical_baseline.py where possible.
# haemoptysis is kept separate because it has high missingness (235/1621
# overall, 36/317 in the audio subset).

FEATURE_SPECS_CORE: dict[str, dict] = {
    # continuous — left as-is (will be z-scored)
    "age": {"type": "continuous"},
    "bmi": {"type": "continuous"},
    # binary yes/no
    "fever":       {"type": "binary", "map": {"yes": 1, "no": 0}},
    "weightloss":  {"type": "binary", "map": {"yes": 1, "no": 0}},
    "ngtsweats":   {"type": "binary", "map": {"yes": 1, "no": 0}},
    "prev_tb":     {"type": "binary", "map": {"yes": 1, "no": 0}},
    # binary with string mapping
    "sex":         {"type": "binary", "map": {"m": 1, "f": 0}},
    "hiv_status":  {"type": "binary", "map": {"pos": 1, "neg": 0}},
    # ordinal (encode cough duration as midpoint in days)
    "coughdur": {
        "type": "ordinal",
        "map": {
            "no cough": 0,
            "<1wk":     0.5 * 7,
            "1-2wks":   1.5 * 7,
            "3-4wks":   3.5 * 7,
            ">4weeks":  5.0 * 7,
        },
    },
    # ordinal tobacco use  (never=0, stopped=1, current=2)
    "tobacco_use": {
        "type": "ordinal",
        "map": {
            "never": 0,
            "not disclosed": 0,  # treated as "never" following 02_clinical_baseline
            "stopped": 1,
            "current": 2,
        },
    },
}

FEATURE_SPECS_EXTRA = {
    "haemoptysis": {"type": "binary", "map": {"yes": 1, "no": 0}},
}


def get_feature_specs(include_haemoptysis: bool) -> dict[str, dict]:
    specs = dict(FEATURE_SPECS_CORE)
    if include_haemoptysis:
        specs.update(FEATURE_SPECS_EXTRA)
    return specs


# =============================================================================
# HELPERS
# =============================================================================

def encode_features(df: pd.DataFrame, feature_specs: dict[str, dict]) -> pd.DataFrame:
    """Encode clinical columns into numeric feature columns."""
    encoded = pd.DataFrame(index=df.index)
    for feat, spec in feature_specs.items():
        if spec["type"] == "continuous":
            encoded[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            encoded[feat] = df[feat].map(spec["map"])
    return encoded


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (AUC, lower_CI, upper_CI) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(y_true[idx], y_score[idx])
    aucs = aucs[~np.isnan(aucs)]
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    point = roc_auc_score(y_true, y_score)
    return point, lower, upper


# =============================================================================
# MAIN EVALUATION ROUTINE
# =============================================================================

def run_fold_evaluation(
    df: pd.DataFrame,
    folds: np.ndarray,
    setting_name: str,
    feature_specs: dict[str, dict],
) -> dict:
    """Train logistic regression in each fold, collect pooled OOF predictions."""
    feature_names = list(feature_specs.keys())
    continuous_features = [k for k, v in feature_specs.items() if v["type"] == "continuous"]

    X_enc = encode_features(df, feature_specs)

    # ── handle missing values ────────────────────────────────────────────
    valid_mask = X_enc.notna().all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  [{setting_name}] Dropping {n_dropped}/{len(df)} rows with missing features.")
        for col in feature_names:
            n_miss = X_enc[col].isna().sum()
            if n_miss > 0:
                print(f"    {col}: {n_miss} missing")

    X_all = X_enc[valid_mask].values.astype(float)
    y_all = df.loc[valid_mask, "label"].values.astype(int)
    folds_valid = folds[valid_mask]

    unique_folds = sorted(np.unique(folds_valid))
    n_folds = len(unique_folds)
    print(f"  [{setting_name}] {len(X_all)} patients, {y_all.sum()} TB+, "
          f"{len(X_all) - y_all.sum()} TB-, {n_folds} folds")

    # ── per-fold training with inner CV for hyperparameter selection ──────
    oof_scores = np.full(len(X_all), np.nan)
    per_fold_auc = []

    for fold_id in unique_folds:
        test_mask = folds_valid == fold_id
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        # Standardise continuous features (fit on train, transform both)
        scaler = StandardScaler()
        cont_idx = [feature_names.index(c) for c in continuous_features]
        X_train_sc = X_train.copy()
        X_test_sc = X_test.copy()
        X_train_sc[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_test_sc[:, cont_idx] = scaler.transform(X_test[:, cont_idx])

        # Inner CV for hyperparameter search (4-fold stratified)
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        grid = GridSearchCV(
            LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED),
            PARAM_GRID,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(X_train_sc, y_train)
        best_clf = grid.best_estimator_

        y_prob = best_clf.predict_proba(X_test_sc)[:, 1]
        oof_scores[test_mask] = y_prob

        if len(np.unique(y_test)) == 2:
            fold_auc = roc_auc_score(y_test, y_prob)
            per_fold_auc.append(fold_auc)
            print(f"    Fold {int(fold_id)}: AUC = {fold_auc:.4f}  "
                  f"(best C={grid.best_params_['C']}, "
                  f"n_train={train_mask.sum()}, n_test={test_mask.sum()})")
        else:
            print(f"    Fold {int(fold_id)}: single class in test set — skipped AUC")

    # ── pooled OOF evaluation ────────────────────────────────────────────
    valid_oof = ~np.isnan(oof_scores)
    y_true = y_all[valid_oof]
    y_score = oof_scores[valid_oof]

    auc_point, ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_score)

    print(f"\n  [{setting_name}] Pooled OOF AUC = {auc_point:.4f}  "
          f"[95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")
    if per_fold_auc:
        print(f"  [{setting_name}] Per-fold AUC: mean = {np.mean(per_fold_auc):.4f}, "
              f"std = {np.std(per_fold_auc):.4f}")

    return {
        "auc": auc_point,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "per_fold_auc": per_fold_auc,
        "y_true": y_true,
        "y_score": y_score,
        "n_patients": len(y_true),
        "n_tb_pos": int(y_true.sum()),
        "features": feature_names,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PATIENTS_CSV)
    print(f"Loaded {len(df)} patients from {PATIENTS_CSV.name}")
    print(f"  Label distribution: TB+ {(df['label']==1).sum()}, TB- {(df['label']==0).sum()}")
    print()

    all_results = {}

    # ──────────────────────────────────────────────────────────────────────
    # PRIMARY: 317 patients with audio, using pre-assigned folds
    # ──────────────────────────────────────────────────────────────────────
    df_audio = df[df["has_audio"] == True].copy().reset_index(drop=True)
    folds_audio = df_audio["fold"].values.astype(int)

    # (a) Without haemoptysis — all 317 patients retained
    print("=" * 70)
    print("PRIMARY: 317 audio patients, 10 features (excl. haemoptysis)")
    print("=" * 70)
    specs_core = get_feature_specs(include_haemoptysis=False)
    r1 = run_fold_evaluation(df_audio, folds_audio, "audio-317-10feat", specs_core)
    all_results["primary_10feat"] = r1

    # (b) With haemoptysis — 281 patients after dropping missing
    print()
    print("=" * 70)
    print("SENSITIVITY: audio patients, 11 features (incl. haemoptysis)")
    print("=" * 70)
    specs_all = get_feature_specs(include_haemoptysis=True)
    r2 = run_fold_evaluation(df_audio, folds_audio, "audio-11feat", specs_all)
    all_results["sensitivity_11feat"] = r2

    # ──────────────────────────────────────────────────────────────────────
    # SECONDARY: all 1,621 patients (new stratified 5-fold split)
    # ──────────────────────────────────────────────────────────────────────
    df_all = df.copy().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds_all = np.full(len(df_all), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(df_all, df_all["label"])):
        folds_all[test_idx] = fold_id

    # (a) Without haemoptysis — minimal missingness
    print()
    print("=" * 70)
    print("SECONDARY: all 1,621 patients, 10 features (excl. haemoptysis)")
    print("=" * 70)
    r3 = run_fold_evaluation(df_all, folds_all, "all-1621-10feat", specs_core)
    all_results["secondary_10feat"] = r3

    # (b) With haemoptysis — significant missingness
    print()
    print("=" * 70)
    print("SECONDARY (sensitivity): all patients, 11 features (incl. haemoptysis)")
    print("=" * 70)
    r4 = run_fold_evaluation(df_all, folds_all, "all-1621-11feat", specs_all)
    all_results["secondary_11feat"] = r4

    # ──────────────────────────────────────────────────────────────────────
    # Save summary
    # ──────────────────────────────────────────────────────────────────────
    lines = [
        "Clinical Baseline — Logistic Regression",
        "=" * 60,
        "",
    ]

    for key, label in [
        ("primary_10feat",     "PRIMARY: 317 audio patients, 10 features (excl. haemoptysis)"),
        ("sensitivity_11feat", "SENSITIVITY: audio patients, 11 features (incl. haemoptysis)"),
        ("secondary_10feat",   "SECONDARY: all 1,621 patients, 10 features (excl. haemoptysis)"),
        ("secondary_11feat",   "SECONDARY: all patients, 11 features (incl. haemoptysis)"),
    ]:
        r = all_results[key]
        lines.append(label)
        lines.append(f"  Features: {', '.join(r['features'])}")
        lines.append(f"  Patients evaluated: {r['n_patients']} (TB+ {r['n_tb_pos']}, "
                     f"TB- {r['n_patients'] - r['n_tb_pos']})")
        lines.append(f"  Pooled OOF AUC = {r['auc']:.4f}  "
                     f"[95% CI: {r['ci_lower']:.4f} - {r['ci_upper']:.4f}]")
        lines.append(f"  Per-fold AUCs: {[round(a, 4) for a in r['per_fold_auc']]}")
        lines.append(f"  Mean +/- std:  {np.mean(r['per_fold_auc']):.4f} +/- "
                     f"{np.std(r['per_fold_auc']):.4f}")
        lines.append("")

    summary_path = OUTPUT_DIR / "clinical_baseline_results.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    # Save OOF predictions
    for tag, key in [
        ("audio_317_10feat", "primary_10feat"),
        ("audio_11feat", "sensitivity_11feat"),
        ("all_1621_10feat", "secondary_10feat"),
        ("all_1621_11feat", "secondary_11feat"),
    ]:
        r = all_results[key]
        out = pd.DataFrame({"y_true": r["y_true"], "y_score": r["y_score"]})
        pred_path = OUTPUT_DIR / f"oof_predictions_{tag}.csv"
        out.to_csv(pred_path, index=False)
        print(f"  OOF predictions -> {pred_path.name}")


if __name__ == "__main__":
    main()
