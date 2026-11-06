import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURES = Path("data/processed/features")

for name in ["CODA", "TBscreen", "Zambia"]:
    d = np.load(FEATURES / f"{name}_cache.npz", allow_pickle=True)
    y = d["y"]

    print(f"\n=== {name} ===")
    for key in ["Xc", "Xa"]:
        X = np.nan_to_num(d[key], nan=0.0, posinf=0.0, neginf=0.0)

        counts = np.bincount(y.astype(int))
        n_splits = min(5, int(counts.min()))
        if n_splits < 2:
            print(f"{key}: not enough samples per class")
            continue

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        scores = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        print(f"{key}: shape={X.shape}, ROC-AUC={roc_auc_score(y, scores):.3f}")

