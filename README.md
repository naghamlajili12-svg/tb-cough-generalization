# Why ML-based cough models do not generalize

**A systematic cross-dataset evaluation for tuberculosis screening.**

Companion code and reproducibility materials for the paper of the same name.

> Zhang, W., Teijeiro, T., Thevenot, J., Atienza, D. *Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening.*
> Embedded Systems Laboratory, EPFL; Basque Center for Applied Mathematics.
> *2026 — manuscript under review; venue and DOI will be added upon publication.*

---

## Overview

Cough acoustics are a promising signal for non-invasive tuberculosis (TB) screening, but it is unclear whether models learn disease-related acoustic patterns or artifacts of how the audio was collected. This repository contains the code to reproduce a systematic **cross-dataset** evaluation of cough-based TB classifiers across three public datasets — **CODA** (multi-country), **TBscreen** (Kenya), and the **Zambia / CIDRZ** dataset (HeAR) — under a consistent external-validation protocol.

The study covers four analyses:

1. **Cross-dataset generalization** of a classical ML pipeline and a deep-learning (DL) pipeline.
2. A **clinical-variable baseline** for context.
3. **Acquisition-related bias in CODA** (location/device shortcuts vs disease signal).
4. **Device generalization in Zambia** (single-device vs multi-device training).


⚠️ **The headline finding** ⚠️ : 
1. Poor cross-dataset transfer: 
AUC frequently drops below 0.6 across both pipelines.
2. Clinical baseline is more robust: 
Generalizes more consistently; points to audio-specific variability
3. Device/location shortcuts: 
Model associates site-specific features with disease prior.
4. Device gap limits transfer: 
Cross-device ROC-AUC drops; multi-device training helps

⚠️ **Implications and Prospectives** ⚠️ : 
1. Rigorous validation:
Always use external validation and stratified analysis by device, site, and population
2. Device characterization:
Identify hardware properties suitable for cough-based TB screening applications
3. Device-diverse datasets:
Balance device, site, and disease status by design to avoid confounding

---

## Data

This repository **does not redistribute any data**. The three datasets are public but must be obtained from their original providers under their respective terms, then placed under `data/raw/<dataset>/`. Access details and the exact subsets used (subjects, cough counts, TB-positive rates, train/validation roles) are documented in [`data/README.md`](data/README.md).

A few preparation details that the pipeline applies (see paper Methods):

- **CODA** — Vietnam, Madagascar, Tanzania only (single-device countries), solicited coughs.
- **TBscreen** — passive vs forced cough kept separate; passive for training (capped at 100 coughs/subject), forced for validation only.
- **Zambia-CIDRZ** — Chainda-South dropped (only 3.03% TB+ rate); audio recorder used for validation only; custom session-level cough extraction; cross-device cross-correlation to detect and fix mislabeled subjects.

---

## Evaluation protocol (shared across pipelines)

- **External validation by dataset:** model selection happens *only* within a source dataset; the other datasets are held out for external evaluation.
- **Subject-level splitting:** all folds split by subject, never by cough event, to avoid leakage.
- **Nested CV:** 5 outer × 4 inner folds, repeated twice → 10 outer estimates; report mean ± std.
- **Subgroup-balanced resampling:** upsample to balance TB label × device × location where metadata exists.
- **Metric:** cough-level probabilities are averaged to subject level, then **subject-level ROC-AUC**.


---

## Repository structure

```
tb-cough-generalization/
├── README.md                  # this file
├── LICENSE                    # MIT
├── requirements.txt           # pip dependencies (pinned)
├── environment.yml            # conda environment
├── .gitignore
│
├── data/                      # NO data re-distributed (folders ship empty)
│   ├── raw/  interim/  processed/   #   git-ignored; populate locally
│   ├── pytorch_vggish.pth     #   user-provided VGGish weights (git-ignored; see Installation)
│   ├── zambia_change_log.csv  #   Zambia subject re-labeling table (used by step 1)
│   └── README.md              # data availability & access
│
├── scripts/                   # self-contained, numbered entry points + shared utils
│   ├── 01_prepare_data.py
│   ├── 02_clinical_baseline.py
│   ├── 03a_extract_features.py
│   ├── 03b_classical_ml_pipeline.py
│   ├── 03c_representation_analysis.py
│   ├── 04a_dl_pipeline.py   04b_dl_analysis.py
│   ├── 05_coda_bias_analysis.py
│   ├── 06a_zambia_device_pipeline.py   06b_zambia_device_analysis.py
│   ├── utils_feature_extraction.py     # handcrafted time/freq features
│   ├── utils_audio_aug.py              # waveform augmentations
│   └── utils_dl.py                     # VGGish/ResNet, training, mel-spectrograms
│
└── results/                   # metrics & figures (git-ignored; folders ship empty)
    ├── figures/
    └── metrics/
```

Each script keeps all parameters in a `CONFIGURATION` block at the top (no external
config files) and reads/writes under `data/` and `results/`.

---

## Installation

**Python 3.13** is used for the paper's results, which is recommended for reproducibility. The code is compatible with Python 3.10+.

```bash
git clone <repo-url> tb-cough-generalization
cd tb-cough-generalization

# option A: conda  (Python 3.13 is pinned in environment.yml)
conda env create -f environment.yml
conda activate tbcough

# option B: pip / venv  (create the venv with a Python 3.13 interpreter)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The scripts are run directly (no package install needed). Pretrained backbones are
obtained separately:

- **CLAP** is downloaded automatically by the `msclap` package on first use.
- **VGGish** weights are **not redistributed here.** Download the standard AudioSet
  VGGish network ported to PyTorch — e.g. the `vggish-10086976.pth` checkpoint from the
  [torchvggish](https://github.com/harritaylor/torchvggish) project (`torch.hub.load('harritaylor/torchvggish', 'vggish')`
  also caches this file) — and save it as **`data/pytorch_vggish.pth`**, the path each DL
  script's `VGGISH_WEIGHTS` already points to. Weights are matched by name and shape
  (`strict=False`), so any standard VGGish PyTorch state dict works. If the file is
  absent, the backbone is randomly initialised and only the classifier head is trained.

---

## Reproducing the results

Run the numbered scripts in order. Each is self-contained; adjust the paths in its
`CONFIGURATION` block to point at your local data first.

```bash
# 1. Prepare data: segment coughs, validate/relabel subjects, build subsets
python scripts/01_prepare_data.py

# 2. Clinical-variable baseline
python scripts/02_clinical_baseline.py

# 3a. Extract features for classical ML pipeline (handcrafted, CLAP embeddings)
python scripts/03a_extract_features.py

# 3b. Classical ML pipeline (cross-dataset)
python scripts/03b_classical_ml_pipeline.py

# 3c. Representation analysis (MMD/MDS, t-SNE)
python scripts/03c_representation_analysis.py

# 4. Deep-learning pipeline (cross-dataset): train + save predictions, then score + plot
python scripts/04a_dl_pipeline.py
python scripts/04b_dl_analysis.py

# 5. CODA location/device bias analysis
python scripts/05_coda_bias_analysis.py

# 6. Zambia device-generalization (single + multi device): train + save predictions, then score + plot
python scripts/06a_zambia_device_pipeline.py
python scripts/06b_zambia_device_analysis.py
```

#### Mapping experiments → paper figures

| Script | Produces (paper figure/table) |
|---|---|
| `02_clinical_baseline.py` | Clinical-variable coefficients & permutation importance |
| `03b_classical_ml_pipeline.py` | Best classical-ML performance heatmap |
| `03c_representation_analysis.py` | MMD/MDS and t-SNE of CLAP and time/frequency features |
| `04a_dl_pipeline.py` + `04b_dl_analysis.py` | Best DL performance ; stratified within-dataset performance |
| `05_coda_bias_analysis.py` | CODA predicted-probability vs prevalence (replication & conservative) |
| `06a_zambia_device_pipeline.py` + `06b_zambia_device_analysis.py` | Single-device and multi-device generalization heatmaps |

---

## Citation

```bibtex
@unpublished{zhang_tbcough_generalization,
  title  = {Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening},
  author = {Zhang, Wensi and Teijeiro, Tomas and Thevenot, J\'er\^ome and Atienza, David},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
