import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

FEATURES_DIR = Path("data/processed/features")
RESULTS_DIR = Path("results/cache_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["CODA", "TBscreen", "Zambia"]
# Xa excluded: leakage audit found features with |correlation| > 0.95.
FEATURE_SETS = ["Xc"]


def load_dataset(name):
    path = FEATURES_DIR / f"{name}_cache.npz"
    data = np.load(path, allow_pickle=True)

    return {
        "Xc": np.nan_to_num(
            data["Xc"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "Xa": np.nan_to_num(
            data["Xa"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "y": data["y"].astype(int),
        "pids": data["pids"].astype(str),
        "subs": data["subs"].astype(str),
    }

def bootstrap_auc_ci(
    y_true,
    y_score,
    n_bootstrap=2000,
    confidence=0.95,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    auc_values = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, n_samples)
        sampled_y = y_true[indices]

        if len(np.unique(sampled_y)) < 2:
            continue

        sampled_scores = y_score[indices]
        auc_values.append(
            roc_auc_score(sampled_y, sampled_scores)
        )

    if not auc_values:
        return np.nan, np.nan

    alpha = 1.0 - confidence
    lower = np.quantile(auc_values, alpha / 2)
    upper = np.quantile(auc_values, 1 - alpha / 2)

    return float(lower), float(upper)
datasets = {
    name: load_dataset(name)
    for name in DATASETS
}

results = []

for feature_key in FEATURE_SETS:
    print(f"\n{'=' * 60}")
    print(f"Feature set: {feature_key}")
    print(f"{'=' * 60}")

    for source_name in DATASETS:
        for target_name in DATASETS:
            if source_name == target_name:
                continue

            source = datasets[source_name]
            target = datasets[target_name]

            X_train = source[feature_key]
            y_train = source["y"]

            X_test = target[feature_key]
            y_test = target["y"]

            if X_train.shape[1] != X_test.shape[1]:
                print(
                    f"SKIPPED {source_name} -> {target_name}: "
                    f"incompatible dimensions "
                    f"{X_train.shape[1]} vs {X_test.shape[1]}"
                )
                continue

            if len(np.unique(y_train)) < 2:
                print(
                    f"SKIPPED {source_name}: "
                    "training labels contain only one class"
                )
                continue

            if len(np.unique(y_test)) < 2:
                print(
                    f"SKIPPED {target_name}: "
                    "test labels contain only one class"
                )
                continue

            model = RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)
            ci_lower, ci_upper = bootstrap_auc_ci(  y_test, probabilities,)
            result = {
                "feature_set": feature_key,
                "train_dataset": source_name,
                "test_dataset": target_name,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": X_train.shape[1],
                "roc_auc": round(float(auc), 4),
                "roc_auc": round(float(auc), 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
            }
            results.append(result)

            print(
                f"{source_name:8s} -> {target_name:8s} | "
                f"train={len(y_train):3d} | "
                f"test={len(y_test):3d} | "
                f"AUC={auc:.3f} "
                f"[95% CI: {ci_lower:.3f}-{ci_upper:.3f}]"
            )

output_path = RESULTS_DIR / "cross_dataset_results.csv"

with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "feature_set",
            "train_dataset",
            "test_dataset",
            "n_train",
            "n_test",
            "n_features",
            "roc_auc",
            "ci_lower",
            "ci_upper",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {output_path}")