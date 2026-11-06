"""
scripts/03a_extract_features.py

Feature extraction for the classical ML pipeline.

For each dataset subset, this script:
  1. links every segmented cough clip (produced by 01_prepare_data.py) to its
     subject id, TB label, and acquisition subgroup (device / location);
  2. balances the data by upsampling acquisition-defined subgroups
     (device x location x TB label) to equal size;
  3. applies waveform augmentation to every (resampled) clip;
  4. extracts two feature representations from the SAME augmented waveform:
       - CLAP embeddings        (1024-dim)
       - time/frequency summary (136-dim: 68 frame features x [mean, std])
  5. saves per-dataset arrays consumed by 03b_classical_ml_pipeline.py.

Outputs, per feature type {clap, timefreq} and dataset key
{zambia, zambia_AR, tbscreen_passive, tbscreen_forced, coda}:
    OUTPUT_DIR/<feature_type>/<key>_features.npy
    OUTPUT_DIR/<feature_type>/<key>_labels.npy
    OUTPUT_DIR/<feature_type>/<key>_groups.npy
    OUTPUT_DIR/<feature_type>/<key>_device_location_label.npy

All parameters are defined in the CONFIGURATION section below.

Usage:
    python scripts/03a_extract_features.py
"""

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence CLAP tokenizer fork warning

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from utils_audio_aug import aug_brownian_tape_speed, aug_pitch_shift
from utils_feature_extraction import extract_time_freq_summary

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000
OUTPUT_DIR  = Path("data/processed/features")

FEATURE_TYPES = ["clap", "timefreq"]
CLAP_VERSION  = "2023"

# time/frequency feature parameters
TF_WIN_S       = 0.02
TF_STEP_S      = 0.01
TF_FREQ_LIMITS = (0, 8000)

# augmentation: one option is chosen at random per (resampled) clip
AUGMENTATIONS = [
    (aug_pitch_shift,         {"max_steps": 2.0, "p_apply": 1.0}),
    (aug_brownian_tape_speed, {"max_rate_perturb": 0.1, "n_walk_steps": 5, "p_apply": 0.8}),
]

RANDOM_SEED = 42
MIN_CLIP_S  = 0.1   # pad (tile) clips shorter than this so feature extractors stay valid

# --- interim (segmented) audio + raw metadata paths -------------------------
# NOTE: data is NOT included in this repo.

ZAMBIA_INTERIM  = Path("data/interim/zambia")
ZAMBIA_METADATA = Path("data/raw/zambia/Metadata and Codebook/Metadata and Codebook/GHAI_Final_Data_2023.csv")

TBSCREEN_INTERIM = Path("data/interim/tbscreen")
TBSCREEN_RAW = Path("data/raw/tbscreen")     # holds Passive_coughs/Passive_coughs.csv etc.

CODA_INTERIM = Path("data/interim/coda")
CODA_CLINICAL  = Path("data/raw/coda/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv")
CODA_EXTRA     = Path("data/raw/coda/meta_data/Clinical/CODA_TB_additional_variables_train.csv")
CODA_SOLICITED = Path("data/raw/coda/meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv")

# --- subgroup definitions ---------------------------------------------------
DEVICES   = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]
LOCATIONS = ["Chainda", "Chawama", "Kanyama"]
ZAMBIA_PHONE_DEVICES = ["Phone A", "Phone B", "Phone C"]
ZAMBIA_TRAIN_LOCATIONS = ["Chawama", "Kanyama"]

TBSCREEN_DEVICE_IDS = {"pixel": 0, "yeti": 1, "codec": 2}
TBSCREEN_PASSIVE_MAX_COUGHS = 100   # cap coughs per subject (passive only)

CODA_COUNTRIES = ["MG", "TZ", "VN"]   # single-device countries used in this study


# =============================================================================
# SAMPLE INDEX
# =============================================================================

@dataclass
class SampleIndex:
    """Parallel arrays describing the clips of one dataset subset."""
    paths: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=list)   # device(_location)_label stratify key

    def add(self, path: str, label: int, group: str, subgroup: str) -> None:
        self.paths.append(path)
        self.labels.append(label)
        self.groups.append(group)
        self.subgroups.append(subgroup)

    def __len__(self) -> int:
        return len(self.paths)


def build_zambia(audio_recorder: bool) -> SampleIndex:
    """Zambia subsets. Phone subset (train/val) or audio-recorder subset (val only)."""
    devices = ["Audio-Recorder"] if audio_recorder else ZAMBIA_PHONE_DEVICES
    df = pd.read_csv(ZAMBIA_METADATA)
    label_by_subject = dict(zip(df["barcode"].astype(str), df["ground_truth_tb"].astype(str)))

    idx = SampleIndex()
    for device in devices:
        for location in ZAMBIA_TRAIN_LOCATIONS:
            folder = ZAMBIA_INTERIM / f"{device}_{location}"
            if not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav")):
                subject = wav.stem.split("_")[0]
                raw_label = label_by_subject.get(subject, "nan")
                if raw_label not in ("pos", "neg"):
                    continue
                label = 1 if raw_label == "pos" else 0
                dev_id = DEVICES.index(device)
                loc_id = LOCATIONS.index(location)
                idx.add(str(wav), label, subject, f"{dev_id}_{loc_id}_{label}")
    return idx


def build_tbscreen(cough_type: str, max_coughs_per_subject: int | None) -> SampleIndex:
    """TBScreen passive or forced. Labels/subjects from the released CSV; device from filename."""
    meta = pd.read_csv(TBSCREEN_RAW / cough_type / f"{cough_type}.csv")
    meta = meta.dropna(subset=["Label", "subject", "path"])
    meta = meta[meta["Label"].astype(str).isin(["TB", "NTB"])]

    idx = SampleIndex()
    for _, row in meta.iterrows():
        stem = str(row["path"])
        wav = TBSCREEN_INTERIM / cough_type / f"{stem}.wav"
        if not wav.exists():
            continue
        dev_id = next((d for key, d in TBSCREEN_DEVICE_IDS.items() if key in stem), None)
        if dev_id is None:
            continue
        label = 1 if str(row["Label"]) == "TB" else 0
        idx.add(str(wav), label, str(row["subject"]), f"{dev_id}_{label}")

    if max_coughs_per_subject is not None:
        idx = _cap_coughs_per_subject(idx, max_coughs_per_subject)
    return idx


def build_coda() -> SampleIndex:
    """CODA (Madagascar/Tanzania/Vietnam). Links solicited clips -> participant -> label/country."""
    clinical = pd.read_csv(CODA_CLINICAL)
    extra = pd.read_csv(CODA_EXTRA)
    solicited = pd.read_csv(CODA_SOLICITED)

    label_by_subject = dict(zip(clinical["participant"].astype(str), clinical["tb_status"]))
    country_by_subject = dict(zip(extra["participant"].astype(str), extra["Country"].astype(str)))
    audio_by_name = {p.name: p for p in CODA_INTERIM.rglob("*.wav")}   # robust to flat or per-country layout

    idx = SampleIndex()
    for _, row in solicited.iterrows():
        subject = str(row["participant"])
        country = country_by_subject.get(subject)
        if country not in CODA_COUNTRIES:
            continue
        wav = audio_by_name.get(Path(str(row["filename"])).name)
        if wav is None:
            continue
        label = label_by_subject.get(subject)
        if label is None or pd.isna(label):
            continue
        label = int(label)
        idx.add(str(wav), label, subject, f"{country}_{label}")
    return idx


def _cap_coughs_per_subject(idx: SampleIndex, max_coughs: int) -> SampleIndex:
    """Randomly keep at most `max_coughs` clips per subject."""
    rng = np.random.default_rng(RANDOM_SEED)
    groups = np.array(idx.groups)
    keep = np.zeros(len(idx), dtype=bool)
    for subject in np.unique(groups):
        positions = np.where(groups == subject)[0]
        if len(positions) > max_coughs:
            positions = rng.choice(positions, size=max_coughs, replace=False)
        keep[positions] = True

    capped = SampleIndex()
    for i in np.where(keep)[0]:
        capped.add(idx.paths[i], idx.labels[i], idx.groups[i], idx.subgroups[i])
    return capped


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Computes the requested feature representations from an augmented waveform."""

    def __init__(self, feature_types: list[str]):
        self.feature_types = feature_types
        self._clap = None
        if "clap" in feature_types:
            from msclap import CLAP
            self._clap = CLAP(version=CLAP_VERSION, use_cuda=False)
            print("CLAP model loaded.")

    def __call__(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if "clap" in self.feature_types:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, audio, SAMPLE_RATE)
                out["clap"] = self._clap.get_audio_embeddings([tmp_path]).detach().numpy()[0]
            finally:
                os.unlink(tmp_path)
        if "timefreq" in self.feature_types:
            out["timefreq"] = extract_time_freq_summary(
                audio, SAMPLE_RATE, TF_WIN_S, TF_STEP_S, TF_FREQ_LIMITS
            )
        return out


def _ensure_min_length(audio: np.ndarray) -> np.ndarray:
    min_len = int(MIN_CLIP_S * SAMPLE_RATE)
    if len(audio) >= min_len or len(audio) == 0:
        return audio
    reps = int(np.ceil(min_len / len(audio)))
    return np.tile(audio, reps)[:min_len]


def _apply_random_augmentation(audio: np.ndarray) -> np.ndarray:
    func, params = AUGMENTATIONS[np.random.choice(len(AUGMENTATIONS))]
    return _ensure_min_length(func(audio, SAMPLE_RATE, **params))


def process_dataset(key: str, idx: SampleIndex, extractor: FeatureExtractor) -> None:
    """Subgroup-balanced resampling + augmentation + feature extraction; saves arrays."""
    if len(idx) == 0:
        print(f"  [{key}] no samples found - skipping.")
        return

    subgroups = np.array(idx.subgroups)
    unique_sg = np.unique(subgroups)
    max_count = max(int(np.sum(subgroups == sg)) for sg in unique_sg)
    seeds = np.random.default_rng(RANDOM_SEED).integers(0, 10_000, size=len(unique_sg))
    print(f"  [{key}] {len(idx)} clips | {len(unique_sg)} subgroups | upsampling each to {max_count}")

    features = {ft: [] for ft in extractor.feature_types}
    labels, groups, dll = [], [], []
    audio_cache: dict[str, np.ndarray] = {}

    for sg, seed in zip(tqdm(unique_sg, desc=key, leave=False), seeds):
        positions = np.where(subgroups == sg)[0]
        n = len(positions)
        # oversample (with replacement) to the largest subgroup size
        np.random.seed(int(seed))
        import random as _random
        _random.seed(int(seed))
        if n < max_count:
            extra = np.random.choice(n, max_count - n, replace=True)
            selection = np.concatenate([np.arange(n), extra])
        else:
            selection = np.arange(max_count)

        for occ in selection:
            pos = positions[occ]
            path = idx.paths[pos]
            if path not in audio_cache:
                audio_cache[path], _ = librosa.load(path, sr=SAMPLE_RATE)
            augmented = _apply_random_augmentation(audio_cache[path])
            feats = extractor(augmented)
            for ft in extractor.feature_types:
                features[ft].append(feats[ft])
            labels.append(idx.labels[pos])
            groups.append(idx.groups[pos])
            dll.append(idx.subgroups[pos])

    # sort by subject so a participant's clips stay together
    order = np.argsort(np.array(groups, dtype=str))
    labels = np.array(labels)[order]
    groups = np.array(groups, dtype=str)[order]
    dll = np.array(dll, dtype=str)[order]

    for ft in extractor.feature_types:
        feats = np.vstack(features[ft])[order]
        out_dir = OUTPUT_DIR / ft
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{key}_features.npy", feats)
        np.save(out_dir / f"{key}_labels.npy", labels)
        np.save(out_dir / f"{key}_groups.npy", groups)
        np.save(out_dir / f"{key}_device_location_label.npy", dll)
    print(f"  [{key}] saved {len(labels)} samples for {extractor.feature_types}")


# =============================================================================
# MAIN
# =============================================================================

DATASET_BUILDERS = {
    "zambia":           lambda: build_zambia(audio_recorder=False),
    "zambia_AR":        lambda: build_zambia(audio_recorder=True),
    "tbscreen_passive": lambda: build_tbscreen("Passive_coughs", TBSCREEN_PASSIVE_MAX_COUGHS),
    "tbscreen_forced":  lambda: build_tbscreen("Forced_coughs", None),
    "coda":             build_coda,
}


def main() -> None:
    extractor = FeatureExtractor(FEATURE_TYPES)
    for key, builder in DATASET_BUILDERS.items():
        print(f"\n=== {key} ===")
        process_dataset(key, builder(), extractor)
    print(f"\nDone. Features written under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
