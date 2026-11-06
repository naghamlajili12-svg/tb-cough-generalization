from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results/cache_baseline")
INPUT_CSV = RESULTS_DIR / "cross_dataset_results.csv"
OUTPUT_PNG = RESULTS_DIR / "cross_dataset_heatmap.png"
OUTPUT_PDF = RESULTS_DIR / "cross_dataset_heatmap.pdf"

DATASET_ORDER = ["CODA", "TBscreen", "Zambia"]

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError(f"No results found in {INPUT_CSV}")

if set(df["feature_set"]) != {"Xc"}:
    raise ValueError(
        "Expected leakage-safe Xc results only. "
        f"Found feature sets: {sorted(df['feature_set'].unique())}"
    )

matrix = df.pivot(
    index="train_dataset",
    columns="test_dataset",
    values="roc_auc",
)

matrix = matrix.reindex(
    index=DATASET_ORDER,
    columns=DATASET_ORDER,
)

plt.figure(figsize=(7, 5.5))

ax = sns.heatmap(
    matrix,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0.35,
    vmax=0.80,
    center=0.50,
    linewidths=1,
    linecolor="white",
    cbar_kws={"label": "ROC-AUC"},
)

ax.set_title(
    "Cross-dataset TB classification performance\n"
    "Random Forest using leakage-safe Xc features"
)
ax.set_xlabel("Test dataset")
ax.set_ylabel("Training dataset")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close()

mean_auc = df["roc_auc"].mean()
min_row = df.loc[df["roc_auc"].idxmin()]
max_row = df.loc[df["roc_auc"].idxmax()]

print(f"Mean cross-dataset ROC-AUC: {mean_auc:.3f}")
print(
    "Weakest transfer: "
    f"{min_row['train_dataset']} -> {min_row['test_dataset']} "
    f"(AUC={min_row['roc_auc']:.3f})"
)
print(
    "Strongest transfer: "
    f"{max_row['train_dataset']} -> {max_row['test_dataset']} "
    f"(AUC={max_row['roc_auc']:.3f})"
)
print(f"Saved PNG: {OUTPUT_PNG}")
print(f"Saved PDF: {OUTPUT_PDF}")