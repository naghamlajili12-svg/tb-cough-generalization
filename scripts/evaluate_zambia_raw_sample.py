from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


FEATURES_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("results/zambia_raw_sample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_TYPES = ["clap", "timefreq"]
N_SPLITS = 5
RANDOM_STATE = 42


def load_features(feature_type: str):
    folder = FEATURES_DIR / feature_type

    features = np.load(
        folder / "zambia_AR_features.npy"
    )
    labels = np.load(
        folder / "zambia_AR_labels.npy"
    ).astype(int)
    groups = np.load(
        folder / "zambia_AR_groups.npy",
        allow_pickle=True,
    ).astype(str)

    return features, labels, groups


def aggregate_to_patient(features, labels, groups):
    """
    Average all cough-level features belonging to the same patient.

    This guarantees that each patient contributes exactly one row,
    preventing cough clips from the same patient from appearing in
    both training and testing sets.
    """
    patient_features = []
    patient_labels = []
    patient_ids = []

    for patient_id in np.unique(groups):
        mask = groups == patient_id
        patient_x = features[mask]
        patient_y = labels[mask]

        unique_labels = np.unique(patient_y)

        if len(unique_labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels: "
                f"{unique_labels}"
            )

        patient_features.append(
            np.mean(patient_x, axis=0)
        )
        patient_labels.append(
            int(unique_labels[0])
        )
        patient_ids.append(patient_id)

    return (
        np.asarray(patient_features),
        np.asarray(patient_labels),
        np.asarray(patient_ids),
    )


results = []
patient_predictions = []

for feature_type in FEATURE_TYPES:
    print(f"\n=== Feature type: {feature_type} ===")

    X, y, groups = load_features(feature_type)

    X_patient, y_patient, patient_ids = aggregate_to_patient(
        X,
        y,
        groups,
    )

    if feature_type == "timefreq":
        X_patient = StandardScaler().fit_transform(
            np.nan_to_num(
                X_patient,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
    else:
        X_patient = np.nan_to_num(
            X_patient,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    class_counts = np.bincount(y_patient)

    print(f"Patients: {len(patient_ids)}")
    print(f"TB negative: {class_counts[0]}")
    print(f"TB positive: {class_counts[1]}")
    print(f"Feature dimensions: {X_patient.shape[1]}")

    if class_counts.min() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} patients per class are required."
        )

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        model,
        X_patient,
        y_patient,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    auc = roc_auc_score(
        y_patient,
        probabilities,
    )

    print(f"Patient-level ROC-AUC: {auc:.3f}")

    results.append(
        {
            "feature_type": feature_type,
            "n_patients": len(patient_ids),
            "n_tb_negative": int(class_counts[0]),
            "n_tb_positive": int(class_counts[1]),
            "n_features": X_patient.shape[1],
            "cv_folds": N_SPLITS,
            "roc_auc": round(float(auc), 4),
        }
    )

    for patient_id, label, probability in zip(
        patient_ids,
        y_patient,
        probabilities,
    ):
        patient_predictions.append(
            {
                "feature_type": feature_type,
                "patient_id": patient_id,
                "label": int(label),
                "predicted_probability": round(
                    float(probability),
                    6,
                ),
            }
        )


results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(patient_predictions)

results_path = OUTPUT_DIR / "patient_level_results.csv"
predictions_path = OUTPUT_DIR / "patient_level_predictions.csv"

results_df.to_csv(
    results_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

print("\n=== Summary ===")
print(results_df.to_string(index=False))

print(f"\nSaved results: {results_path}")
print(f"Saved predictions: {predictions_path}")

print(
    "\nImportant limitation: this is a technical smoke test "
    "using a small patient cohort. Results must not be interpreted "
    "as reliable estimates of clinical performance."
)