"""
scripts/03c_representation_analysis.py

Representation analysis of the acoustic feature spaces.

Examines whether learned representations are organised more strongly by acquisition
factors (dataset, recording device, location) or by TB status, for both feature
families:
    - CLAP embeddings        (1024-dim)
    - time/frequency summary (136-dim: 68 frame features x [mean, std])

Two visualisations per feature type and per grouping factor (device / location /
gender / TB):
    - MMD distances between subgroups, embedded in 2-D with MDS;
    - t-SNE of individual samples coloured by dataset and by the grouping factor.

Unlike the training pipeline, ALL available coughs are used in their original form
(no resampling, no augmentation), including every CODA country and every Zambia
device/location. Raw features are extracted once and cached under CACHE_DIR.

All parameters are defined in the CONFIGURATION section below.

Usage:
    python scripts/03c_representation_analysis.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils_feature_extraction import extract_time_freq_summary

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000
FEATURE_TYPES = ["clap", "timefreq"]
CLAP_VERSION  = "2023"

CACHE_DIR  = Path("data/processed/representation")    # raw per-sample features + metadata
OUTPUT_DIR = Path("results/figures/representation")

# time/frequency feature parameters (match 03a)
TF_WIN_S, TF_STEP_S, TF_FREQ_LIMITS = 0.02, 0.01, (0, 8000)

CLAP_BATCH = 64
RANDOM_SEED = 42

# Tractability caps (the paper uses all samples; these bound memory/time).
MMD_MAX_PER_GROUP   = 2000   # subsample each subgroup before the RBF kernel
TSNE_MAX_PER_DATASET = 3000  # subsample each dataset before t-SNE

# Grouping factors to visualise.
LABEL_OPTIONS = ["device", "location", "gender", "TB"]

# Fixed order -> consistent colours/markers across plots.
REPR_DATASETS = ["Zambia", "TBScreen_Forced", "TBScreen_Passive", "CODA"]

# --- interim audio + metadata paths (see 03a for the repo-relative defaults) ---
ZAMBIA_INTERIM  = Path("data/interim/zambia")
ZAMBIA_METADATA = Path(
    "data/raw/zambia/Metadata and Codebook/"
    "Metadata and Codebook/GHAI_Final_Data_2023.csv"
)
TBSCREEN_INTERIM = Path("data/interim/tbscreen")
TBSCREEN_RAW     = Path("data/raw/tbscreen")
CODA_INTERIM   = Path("data/interim/coda")
CODA_CLINICAL  = Path("data/raw/coda/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv")
CODA_EXTRA     = Path("data/raw/coda/meta_data/Clinical/CODA_TB_additional_variables_train.csv")
CODA_SOLICITED = Path("data/raw/coda/meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv")

DEVICES   = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]
LOCATIONS = ["Chainda", "Chawama", "Kanyama"]
TBSCREEN_DEVICE_IDS = {"pixel": 0, "yeti": 1, "codec": 2}
TBSCREEN_PASSIVE_MAX_COUGHS = 100   # cap coughs per subject (passive only); seeded subsample

# Offsets keep device / location ids unique across datasets (for plot colouring).
TBSCREEN_DEVICE_OFFSET = 4
CODA_DEVICE_OFFSET     = 7
CODA_LOCATION_OFFSET   = 3


# =============================================================================
# SAMPLE INDEX (raw, full coverage, with metadata)
# =============================================================================

@dataclass
class SampleIndex:
    paths: list[str] = field(default_factory=list)
    tb: list[int] = field(default_factory=list)
    device: list[int] = field(default_factory=list)
    location: list[int] = field(default_factory=list)
    gender: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)

    def add(self, path, tb, device, location, gender, group) -> None:
        self.paths.append(path); self.tb.append(tb); self.device.append(device)
        self.location.append(location); self.gender.append(gender); self.groups.append(group)

    def __len__(self) -> int:
        return len(self.paths)

    def subset(self, positions) -> "SampleIndex":
        out = SampleIndex()
        for i in positions:
            out.add(self.paths[i], self.tb[i], self.device[i],
                    self.location[i], self.gender[i], self.groups[i])
        return out


def _cap_coughs_per_subject(idx: SampleIndex, max_coughs: int, seed: int = RANDOM_SEED) -> SampleIndex:
    """Keep at most `max_coughs` clips per subject (seeded random subsample)."""
    rng = np.random.default_rng(seed)
    groups = np.array(idx.groups)
    keep = []
    for subject in np.unique(groups):
        positions = np.where(groups == subject)[0]
        if len(positions) > max_coughs:
            positions = rng.choice(positions, size=max_coughs, replace=False)
        keep.extend(positions.tolist())
    return idx.subset(sorted(keep))


def build_zambia_repr() -> SampleIndex:
    df = pd.read_csv(ZAMBIA_METADATA)
    label_by = dict(zip(df["barcode"].astype(str), df["ground_truth_tb"].astype(str)))
    gender_by = {str(b): (1 if g == "m" else 0) for b, g in zip(df["barcode"], df["sex"])}

    idx = SampleIndex()
    for device in DEVICES:
        for location in LOCATIONS:
            folder = ZAMBIA_INTERIM / f"{device}_{location}"
            if not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav")):
                subject = wav.stem.split("_")[0]
                raw = label_by.get(subject, "nan")
                if raw not in ("pos", "neg"):
                    continue
                idx.add(str(wav), 1 if raw == "pos" else 0,
                        DEVICES.index(device), LOCATIONS.index(location),
                        gender_by.get(subject, 0), subject)
    return idx


def build_tbscreen_repr(cough_type: str, max_coughs_per_subject: int | None = None) -> SampleIndex:
    meta = pd.read_csv(TBSCREEN_RAW / cough_type / f"{cough_type}.csv").dropna(subset=["Label", "subject", "path"])
    meta = meta[meta["Label"].astype(str).isin(["TB", "NTB"])]
    gender_by = {str(s): (1 if str(g) in ("0", "0.0") else 0)
                 for s, g in zip(meta["subject"], meta.get("gndr_a", pd.Series(dtype=str)))}

    idx = SampleIndex()
    for _, row in meta.iterrows():
        stem = str(row["path"])
        wav = TBSCREEN_INTERIM / cough_type / f"{stem}.wav"
        if not wav.exists():
            continue
        dev = next((d for key, d in TBSCREEN_DEVICE_IDS.items() if key in stem), None)
        if dev is None:
            continue
        subject = str(row["subject"])
        idx.add(str(wav), 1 if str(row["Label"]) == "TB" else 0,
                dev + TBSCREEN_DEVICE_OFFSET, -1, gender_by.get(subject, 0), subject)

    if max_coughs_per_subject is not None:
        idx = _cap_coughs_per_subject(idx, max_coughs_per_subject)
    return idx


def build_coda_repr() -> SampleIndex:
    clinical = pd.read_csv(CODA_CLINICAL)
    extra = pd.read_csv(CODA_EXTRA)
    solicited = pd.read_csv(CODA_SOLICITED)

    label_by = dict(zip(clinical["participant"].astype(str), clinical["tb_status"]))
    gender_by = {str(p): (1 if str(s) == "Male" else 0)
                 for p, s in zip(clinical["participant"], clinical["sex"])}
    country_by = dict(zip(extra["participant"].astype(str), extra["Country"].astype(str)))
    audio_by_name = {p.name: p for p in CODA_INTERIM.rglob("*.wav")}
    countries = sorted({c for c in country_by.values() if isinstance(c, str)})

    idx = SampleIndex()
    for _, row in solicited.iterrows():
        subject = str(row["participant"])
        country = country_by.get(subject)
        if country not in countries:
            continue
        wav = audio_by_name.get(Path(str(row["filename"])).name)
        label = label_by.get(subject)
        if wav is None or label is None or pd.isna(label):
            continue
        c_idx = countries.index(country)
        idx.add(str(wav), int(label),
                c_idx + CODA_DEVICE_OFFSET, c_idx + CODA_LOCATION_OFFSET,
                gender_by.get(subject, 0), subject)
    return idx


DATASET_BUILDERS = {
    "Zambia":           build_zambia_repr,
    "TBScreen_Forced":  lambda: build_tbscreen_repr("Forced_coughs"),
    "TBScreen_Passive": lambda: build_tbscreen_repr("Passive_coughs", TBSCREEN_PASSIVE_MAX_COUGHS),
    "CODA":             build_coda_repr,
}


# =============================================================================
# RAW FEATURE EXTRACTION (cached)
# =============================================================================

def _extract_clap(paths: list[str], clap_model) -> np.ndarray:
    feats = []
    for start in tqdm(range(0, len(paths), CLAP_BATCH), desc="    CLAP", leave=False):
        batch = paths[start:start + CLAP_BATCH]
        feats.append(clap_model.get_audio_embeddings(batch).detach().numpy())
    return np.vstack(feats)


def _extract_timefreq(paths: list[str]) -> np.ndarray:
    feats = []
    for path in tqdm(paths, desc="    time/freq", leave=False):
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        feats.append(extract_time_freq_summary(audio, SAMPLE_RATE, TF_WIN_S, TF_STEP_S, TF_FREQ_LIMITS))
    return np.vstack(feats)


def extract_or_load(dataset: str) -> dict:
    """Extract (and cache) raw features + metadata for one representation dataset.

    The sample index is rebuilt every run (cheap: CSV + globbing) so metadata always
    reflects the current configuration. A cached feature array is reused only if its
    length matches the index; otherwise it is re-extracted, keeping features and
    metadata consistent (e.g. after changing the per-subject cap).
    """
    idx = DATASET_BUILDERS[dataset]()
    n = len(idx)
    print(f"  [{dataset}] {n} clips")
    out = {k: np.array(getattr(idx, k)) for k in ("tb", "device", "location", "gender", "groups")}

    for feature_type in FEATURE_TYPES:
        fpath = CACHE_DIR / feature_type / f"{dataset}.npy"
        if fpath.exists():
            feats = np.load(fpath)
            if len(feats) == n:
                print(f"    [{feature_type}] loaded cached features ({n})")
                out[feature_type] = feats
                continue
            print(f"    [{feature_type}] cache size {len(feats)} != {n} clips; re-extracting")

        fpath.parent.mkdir(parents=True, exist_ok=True)
        if feature_type == "clap":
            from msclap import CLAP
            feats = _extract_clap(idx.paths, CLAP(version=CLAP_VERSION, use_cuda=False))
        else:
            feats = _extract_timefreq(idx.paths)
        np.save(fpath, feats)
        out[feature_type] = feats

    return out


# =============================================================================
# MMD + MDS + t-SNE
# =============================================================================

def compute_mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        stacked = np.vstack([X, Y])
        gamma = 1.0 / (2 * np.median(cdist(stacked, stacked)) ** 2 + 1e-8)
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    m, n = X.shape[0], Y.shape[0]
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2 * Kxy.mean())


def build_groups(X_list, label_list, dataset_names, rng, max_per_group=MMD_MAX_PER_GROUP):
    """Split each dataset into subgroups by label value; return (groups, names)."""
    groups, names = [], []
    for X, labels, name in zip(X_list, label_list, dataset_names):
        for cls in np.unique(labels):
            idx = np.where(labels == cls)[0]
            if len(idx) <= 1:
                continue
            if len(idx) > max_per_group:
                idx = rng.choice(idx, max_per_group, replace=False)
            groups.append(X[idx])
            names.append(f"{name}_C{int(cls)}")
    return groups, names


def compute_mmd_matrix(groups):
    n = len(groups)
    D = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in tqdm(pairs, desc="    MMD matrix", leave=False):
        D[i, j] = D[j, i] = compute_mmd_rbf(groups[i], groups[j])
    return D


def _domain_of(name: str, dataset_names: list[str]) -> str:
    return next(d for d in dataset_names if name.startswith(d + "_C"))


def plot_mmd_embedding(D, names, dataset_names, label_name, out_path, show_class=False):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED).fit_transform(D)
    colors = plt.cm.tab10(np.arange(len(dataset_names)))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"][:len(dataset_names)]

    plt.figure(figsize=(5, 4))
    for i, name in enumerate(names):
        d = dataset_names.index(_domain_of(name, dataset_names))
        plt.scatter(coords[i, 0], coords[i, 1], marker=markers[d], c=[colors[d]], s=100, alpha=0.7)
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.xticks(np.linspace(*xlim, 5)); plt.yticks(np.linspace(*ylim, 5))
    if show_class:
        for i, name in enumerate(names):
            d = dataset_names.index(_domain_of(name, dataset_names))
            cls = name[name.rfind("_C") + 2:]
            plt.text(coords[i, 0] + 0.015 * (xlim[1] - xlim[0]),
                     coords[i, 1] + 0.02 * (ylim[1] - ylim[0]), cls, color=colors[d], fontsize=9)
    plt.title(f"MMD-based MDS Embedding by {label_name}" if label_name else "MMD-based MDS Embedding",
              fontsize=14)
    plt.xlabel("Dim 1", fontsize=14); plt.ylabel("Dim 2", fontsize=14)
    plt.legend(handles=[plt.Line2D([0], [0], marker=markers[i], color="w", label=dataset_names[i],
                                   markerfacecolor=colors[i], markersize=10)
                        for i in range(len(dataset_names))], loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path); plt.close()


def plot_sample_tsne(X_list, label_list, dataset_names, label_name, out_path, rng):
    # subsample each dataset for tractability
    subs_X, subs_lab, dataset_ids = [], [], []
    for i, (X, labels) in enumerate(zip(X_list, label_list)):
        sel = np.arange(len(X))
        if len(X) > TSNE_MAX_PER_DATASET:
            sel = rng.choice(len(X), TSNE_MAX_PER_DATASET, replace=False)
        subs_X.append(X[sel]); subs_lab.append(np.asarray(labels)[sel])
        dataset_ids.append(np.full(len(sel), i))
    X = np.vstack(subs_X)
    labels = np.concatenate(subs_lab)
    dataset_ids = np.concatenate(dataset_ids)

    coords = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X)
    colors_domain = plt.cm.tab10(np.arange(len(X_list)))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"][:len(X_list)]
    uniq_labels = np.unique(labels)
    palette = plt.cm.Set1(np.arange(9))   # 9 base colors; rotate from the start beyond that
    label_to_color = {lab: palette[k % len(palette)] for k, lab in enumerate(uniq_labels)}

    fig = plt.figure(figsize=(12, 8))
    # panel 1: coloured by dataset
    ax = fig.add_subplot(2, 3, 1)
    for i in range(len(X_list)):
        m = dataset_ids == i
        ax.scatter(coords[m, 0], coords[m, 1], c=[colors_domain[i]], s=5, alpha=0.5, marker=markers[i])
    ax.legend(handles=[plt.Line2D([0], [0], marker=markers[i], color="w", label=dataset_names[i],
                                  markerfacecolor=colors_domain[i], markersize=10)
                       for i in range(len(X_list))], loc="best")
    ax.set_title("t-SNE by dataset"); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # panels 2..: coloured by label, one per dataset
    panel_slots = [2, 3, 5, 6]
    for i in range(len(X_list)):
        ax = fig.add_subplot(2, 3, panel_slots[i % len(panel_slots)])
        m = dataset_ids == i
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[label_to_color[v] for v in labels[m]], s=5, alpha=0.5, marker=markers[i])
        ax.legend(handles=[plt.Line2D([0], [0], marker=markers[i], color="w",
                                      label=f"{label_name} {int(v)}",
                                      markerfacecolor=label_to_color[v], markersize=10)
                           for v in np.unique(labels[m])], loc="best")
        ax.set_title(f"t-SNE by {label_name} -- {dataset_names[i]}")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    data = {name: extract_or_load(name) for name in REPR_DATASETS}

    for feature_type in FEATURE_TYPES:
        print(f"\n=== Representation analysis: {feature_type} ===")
        out_dir = OUTPUT_DIR / feature_type
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(RANDOM_SEED)

        X_list = [data[name][feature_type].astype(float) for name in REPR_DATASETS]
        if feature_type == "timefreq":
            # pooled standardisation + clipping (raw time/freq features have disparate scales)
            scaler = StandardScaler().fit(np.nan_to_num(np.vstack(X_list), posinf=0.0, neginf=0.0))
            X_list = [np.clip(scaler.transform(np.nan_to_num(X, posinf=0.0, neginf=0.0)), -5, 5)
                      for X in X_list]

        mmd_dir = CACHE_DIR / "mmd"
        mmd_dir.mkdir(parents=True, exist_ok=True)

        for label_name in LABEL_OPTIONS:
            print(f"  grouping by {label_name}")
            label_list = [data[name][label_name if label_name != "TB" else "tb"] for name in REPR_DATASETS]

            # MMD matrices are expensive: load from cache if present, else compute and save.
            mmd_file = mmd_dir / f"{feature_type}_{label_name}.npz"
            if mmd_file.exists():
                cached = np.load(mmd_file, allow_pickle=True)
                D, names = cached["D"], list(cached["names"])
            else:
                groups, names = build_groups(X_list, label_list, REPR_DATASETS, rng)
                D = compute_mmd_matrix(groups)
                np.savez(mmd_file, D=D, names=np.array(names, dtype=object))

            plot_mmd_embedding(
                D, names, REPR_DATASETS, label_name,
                out_dir / f"mmd_embedding_{label_name}.pdf",
                show_class=label_name not in ("device", "location"),
            )
            plot_sample_tsne(
                X_list, label_list, REPR_DATASETS, label_name,
                out_dir / f"tsne_{label_name}.pdf", rng,
            )
    print(f"\nDone. Figures written under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
