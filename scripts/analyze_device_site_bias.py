from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("results/device_site_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["CODA", "TBscreen", "Zambia"]

rows = []

for dataset_name in DATASETS:
    file_path = FEATURES_DIR / f"{dataset_name}_cache.npz"
    data = np.load(file_path, allow_pickle=True)

    labels = data["y"].astype(int)
    subgroups = data["subs"].astype(str)

    for subgroup in np.unique(subgroups):
        mask = subgroups == subgroup
        subgroup_labels = labels[mask]

        n_total = len(subgroup_labels)
        n_positive = int(subgroup_labels.sum())
        n_negative = n_total - n_positive
        prevalence = n_positive / n_total

        if "_" in subgroup:
            location, device = subgroup.split("_", 1)
        else:
            location = subgroup
            device = "Unknown"

        rows.append(
            {
                "dataset": dataset_name,
                "subgroup": subgroup,
                "location": location,
                "device": device,
                "n_total": n_total,
                "n_tb_positive": n_positive,
                "n_tb_negative": n_negative,
                "tb_prevalence": prevalence,
            }
        )


results = pd.DataFrame(rows)
results = results.sort_values(["dataset", "subgroup"])

csv_path = OUTPUT_DIR / "device_site_summary.csv"
results.to_csv(csv_path, index=False)

print("\nDevice/site summary:")
print(results.to_string(index=False))


plot_df = results.copy()

plot_df["display_name"] = (
    plot_df["dataset"]
    + "\n"
    + plot_df["location"]
    + "\n"
    + plot_df["device"]
)

dataset_colors = {
    "CODA": "#2563EB",
    "TBscreen": "#16A34A",
    "Zambia": "#7C3AED",
}

bar_colors = [
    dataset_colors[dataset]
    for dataset in plot_df["dataset"]
]

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    plot_df["display_name"],
    plot_df["tb_prevalence"] * 100,
    color=bar_colors,
)

ax.set_title("TB prevalence by dataset, location and recording device")
ax.set_xlabel("Dataset, location and recording device")
ax.set_ylabel("TB-positive prevalence (%)")
ax.set_ylim(0, 55)
ax.tick_params(axis="x", rotation=25)

for bar, prevalence, sample_size in zip(
    bars,
    plot_df["tb_prevalence"],
    plot_df["n_total"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{prevalence * 100:.1f}%\n(n={sample_size})",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

png_path = OUTPUT_DIR / "device_site_prevalence.png"
pdf_path = OUTPUT_DIR / "device_site_prevalence.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.close(fig)

print(f"\nSaved CSV: {csv_path}")
print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")

print(
    "\nImportant limitation: in CODA, location and device are "
    "confounded because each location uses a different device."
)