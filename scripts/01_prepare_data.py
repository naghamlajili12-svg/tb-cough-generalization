"""
scripts/01_prepare_data.py

Segment individual cough events from raw recordings for three datasets:

  TBscreen  : energy-based trimming of pre-segmented 1-s clips
  CODA      : energy-based trimming of pre-segmented 0.5-s clips
  Zambia    : CLAP-based coarse detection + cross-device temporal alignment +
              energy-based fine segmentation with cross-device consensus masking

All parameters are defined in the CONFIGURATION section below.
Raw data is read from SOURCE paths; segmented clips are written to OUTPUT paths.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# TBscreen  (place the downloaded dataset under data/raw/tbscreen/)
# ---------------------------------------------------------------------------
TBSCREEN_RAW_DIR  = Path("data/raw/tbscreen")
TBSCREEN_OUT_DIR  = Path("data/interim/tbscreen")
TBSCREEN_SUBTYPES = ["Passive_coughs", "Forced_coughs"]

# ---------------------------------------------------------------------------
# CODA  (same energy-based trimming as TBscreen, applied to 0.5-s pre-cut clips)
# Place the downloaded released clips under data/raw/coda/
# ---------------------------------------------------------------------------
CODA_RAW_DIR   = Path("data/raw/coda")
CODA_OUT_DIR   = Path("data/interim/coda")
CODA_COUNTRIES = ["Vietnam", "Madagascar", "Tanzania"]  # single-device countries only

# ---------------------------------------------------------------------------
# Zambia / CIDRZ  (HeAR benchmark; place the downloaded dataset under data/raw/zambia/)
# ---------------------------------------------------------------------------
ZAMBIA_RAW_DIR   = Path("data/raw/zambia")
ZAMBIA_OUT_DIR   = Path("data/interim/zambia")
ZAMBIA_CHANGE_LOG = Path("data/zambia_change_log.csv")
ZAMBIA_METADATA  = Path(
    "data/raw/zambia/Metadata and Codebook/Metadata and Codebook/GHAI_Final_Data_2023.csv"
)
ZAMBIA_DEVICES   = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]
ZAMBIA_LOCATIONS = ["Chainda", "Chawama", "Kanyama"]

# Subject ID prefix encodes collection site
ZAMBIA_PREFIX_TO_LOCATION = {
    "01-": "Kanyama",
    "02-": "Chawama",
    "03-": "Chainda",
}

# CLAP classification
ZAMBIA_CLAP_LABELS    = ["white noise", "people talking", "coughing sound"]
ZAMBIA_CLAP_COUGH_IDX = 2                  # index of "coughing sound" in CLAP_LABELS
ZAMBIA_CLAP_WIN_S     = 2.0                 # sliding window duration (s)
ZAMBIA_CLAP_SKIP_S    = 8.0                 # suppress cough predictions in first N seconds
ZAMBIA_ALIGN_SR       = 4_000               # downsample rate for cross-correlation (speed)

# Cough segmentation
ZAMBIA_N_SINGLE_COUGHS      = 3    # first N events are individual coughs
ZAMBIA_MAX_CLIP_S           = 1.0  # discard clips longer than this (segmentation errors)
ZAMBIA_CONSENSUS_FRAC       = 0.5  # fraction of devices that must agree on a segment

# ---------------------------------------------------------------------------
# Energy-based trimming parameters (shared across all three datasets)
# ---------------------------------------------------------------------------
ENERGY_WINDOW_S   = 0.05   # moving-average window for energy envelope (s)
ENERGY_THRESHOLD  = 0.05   # activity threshold as fraction of peak energy
ENERGY_PRE_PAD_S  = 0.05   # silence kept before activity onset (s)
ENERGY_POST_PAD_S = 0.25   # silence kept after activity offset (s)

# Peak detection within consecutive-cough segments (Zambia only)
PEAK_THRESH_FRAC = 0.30    # minimum peak height as fraction of segment peak
PEAK_DIST_S      = 0.50    # minimum inter-peak distance (s)
MIN_BASELINE_S   = 0.10    # minimum silence duration defining a cough boundary (s)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_and_normalize(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    clip, _ = librosa.load(str(path), sr=sr)
    peak = np.abs(clip).max()
    return clip / peak if peak > 0 else clip


def trim_by_energy(
    clip: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_s: float = ENERGY_WINDOW_S,
    threshold: float = ENERGY_THRESHOLD,
    pre_pad_s: float = ENERGY_PRE_PAD_S,
    post_pad_s: float = ENERGY_POST_PAD_S,
) -> np.ndarray:
    """Trim a clip to its active-cough region using a moving-average energy envelope."""
    win = int(window_s * sr)
    energy_ma = np.convolve(clip ** 2, np.ones(win) / win, mode="same")
    baseline = threshold * energy_ma.max()
    active = np.where(energy_ma >= baseline)[0]
    if len(active) == 0:
        return clip
    start = max(0, int(active[0]) - int(pre_pad_s * sr))
    end   = min(len(clip), int(active[-1]) + int(post_pad_s * sr))
    return clip[start:end]


def _nonzero_bounds(clip: np.ndarray, tol: float = 1e-6) -> tuple[int, int]:
    """Return the first and last non-silent sample indices.

    TBscreen clips have hard-zero padding at both ends from prior processing;
    without this, the energy trimmer would treat those zeros as valid signal.
    """
    nz = np.where(np.abs(clip) > tol)[0]
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) > 0 else (0, len(clip))


# =============================================================================
# ZAMBIA RE-LABELING
# =============================================================================

def _load_relabeling(
    change_log_path: Path,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Load the subject re-labeling table and return two lookup dicts.

    Returns
    -------
    forward : (folder_name, old_stem) -> canonical_subject_id
        Used when scanning files on disk to map a mislabeled filename to the
        correct subject ID.
    reverse : (folder_name, canonical_subject_id) -> old_stem
        Used when loading audio for a known subject ID to find the actual
        filename stored on disk.
    """
    df = pd.read_csv(change_log_path)
    df = df.dropna(subset=["from_subject_id", "to_subject_id"])
    df["path"]            = df["path"].str.strip()
    df["from_subject_id"] = df["from_subject_id"].str.strip()
    df["to_subject_id"]   = df["to_subject_id"].str.strip()
    # drop no-op rows (from == to)
    df = df[df["from_subject_id"] != df["to_subject_id"]]

    forward: dict[tuple[str, str], str] = {}
    reverse: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        folder   = row["path"]
        old_stem = row["from_subject_id"]
        new_stem = row["to_subject_id"]
        forward[(folder, old_stem)] = new_stem
        reverse[(folder, new_stem)] = old_stem

    return forward, reverse


# =============================================================================
# TBSCREEN & CODA  —  energy-based trimming of pre-segmented clips
# =============================================================================

def _process_clip_directory(
    src_dir: Path,
    dst_dir: Path,
    sr: int = SAMPLE_RATE,
    clip_has_zero_padding: bool = True,
) -> int:
    """Trim all WAV files in src_dir and write results to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        f for f in src_dir.iterdir()
        if not f.name.startswith(".") and f.suffix.lower() == ".wav"
    )
    for path in tqdm(files, desc=src_dir.name, leave=False):
        clip = load_and_normalize(path, sr)
        if clip_has_zero_padding:
            z0, z1 = _nonzero_bounds(clip)
            trimmed = trim_by_energy(clip[z0:z1], sr)
        else:
            trimmed = trim_by_energy(clip, sr)
        sf.write(dst_dir / path.name, trimmed, sr)
    return len(files)


def process_tbscreen(
    raw_dir: Path = TBSCREEN_RAW_DIR,
    out_dir: Path = TBSCREEN_OUT_DIR,
) -> None:
    print("=== TBscreen ===")
    for subtype in TBSCREEN_SUBTYPES:
        n = _process_clip_directory(
            src_dir=raw_dir / subtype / 'Audio_files',
            dst_dir=out_dir / subtype,
            clip_has_zero_padding=True,
        )
        print(f"  {subtype}: {n} clips")


def process_coda(
    raw_dir: Path = CODA_RAW_DIR,
    out_dir: Path = CODA_OUT_DIR,
) -> None:
    print("=== CODA ===")
    n = _process_clip_directory(
        src_dir=raw_dir,
        dst_dir=out_dir,
        clip_has_zero_padding=False,
    )
    print(f"  {n} clips")


# =============================================================================
# ZAMBIA  —  CLAP detection + cross-device alignment + fine segmentation
# =============================================================================

def _align_to_reference(
    audios: list[np.ndarray], sr: int = SAMPLE_RATE
) -> list[np.ndarray]:
    """Time-align all recordings to the first one via cross-correlation."""
    ref = audios[0]
    ref_ds = librosa.resample(ref, orig_sr=sr, target_sr=ZAMBIA_ALIGN_SR)
    aligned = [ref]
    for audio in audios[1:]:
        aud_ds = librosa.resample(audio, orig_sr=sr, target_sr=ZAMBIA_ALIGN_SR)
        corr = np.correlate(ref_ds, aud_ds, mode="full")
        lag = int((np.argmax(corr) - len(aud_ds) + 1) * sr / ZAMBIA_ALIGN_SR)
        aligned_audio = np.pad(audio, (lag, 0)) if lag > 0 else audio[-lag:]
        aligned.append(aligned_audio)
    min_len = min(len(a) for a in aligned)
    return [a[:min_len] for a in aligned]


def _clap_classify_windows(
    audio: np.ndarray,
    sr: int,
    clap_model,
    text_embeddings,
) -> list[int]:
    """Classify fixed-length sliding windows with CLAP.

    Windows in the first ZAMBIA_CLAP_SKIP_S seconds or with amplitude below
    the 99th-percentile peak level are forced to class 0 (noise) regardless of
    the model prediction — this suppresses false detections during recording setup.
    """
    win = int(ZAMBIA_CLAP_WIN_S * sr)
    skip_wins = int(ZAMBIA_CLAP_SKIP_S / ZAMBIA_CLAP_WIN_S)
    peak_level = np.percentile(np.abs(audio), 99.0)
    predictions = []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        for i, start in enumerate(range(0, len(audio), win)):
            clip = audio[start : min(start + win, len(audio))]
            sf.write(tmp_path, clip, sr)
            emb = clap_model.get_audio_embeddings([tmp_path])
            sim = clap_model.compute_similarity(emb, text_embeddings)
            pred = int(np.argmax(sim.detach().numpy()))
            # suppress uncertain / early-window predictions
            if i > skip_wins and np.max(np.abs(clip)) >= peak_level:
                predictions.append(pred)
            else:
                predictions.append(0)
    finally:
        os.unlink(tmp_path)
    return predictions


def _energy_trim_within_window(
    audio: np.ndarray, sr: int, win_start: int, win_end: int
) -> tuple[int, int]:
    """Apply energy trimming to a coarse window; return absolute start/end indices."""
    clip = audio[win_start:win_end]
    win = int(ENERGY_WINDOW_S * sr)
    energy_ma = np.convolve(clip ** 2, np.ones(win) / win, mode="same")
    baseline = ENERGY_THRESHOLD * energy_ma.max()
    active = np.where(energy_ma >= baseline)[0]
    if len(active) == 0:
        return win_start, win_end
    start = max(0, int(active[0])  - int(ENERGY_PRE_PAD_S  * sr)) + win_start
    end   = min(len(clip), int(active[-1]) + int(ENERGY_POST_PAD_S * sr)) + win_start
    return start, end


def _extract_individual_coughs(
    aligned: list[np.ndarray],
    cough_preds: list[int],
    sr: int,
    win: int,
) -> tuple[list[list[np.ndarray]], int]:
    """Extract the first ZAMBIA_N_SINGLE_COUGHS individual-cough events.

    Returns per-device clip lists and the pointer into cough_preds marking
    the start of the consecutive-cough portion.
    """
    # Group consecutive cough windows into runs (= individual events)
    events: list[list[int]] = []
    pointer = 0
    while pointer < len(cough_preds) and len(events) < ZAMBIA_N_SINGLE_COUGHS:
        run = [cough_preds[pointer]]
        pointer += 1
        while pointer < len(cough_preds) and cough_preds[pointer] - cough_preds[pointer - 1] == 1:
            run.append(cough_preds[pointer])
            pointer += 1
        events.append(run)

    # If the last run is long it's likely the start of the consecutive section
    if len(events) == ZAMBIA_N_SINGLE_COUGHS and len(events[-1]) >= 3:
        pointer -= len(events[-1])
        events.pop()

    clips_per_device: list[list[np.ndarray]] = [[] for _ in aligned]
    for event in events:
        starts_dev, ends_dev = [], []
        for audio in aligned:
            coarse_s = max(0, event[0] * win - win // 2)
            coarse_e = min(len(audio), (event[-1] + 1) * win + win // 2)
            s, e = _energy_trim_within_window(audio, sr, coarse_s, coarse_e)
            starts_dev.append(s)
            ends_dev.append(e)
        # use median start/end across devices
        s_med = int(np.median(starts_dev))
        e_med = int(np.median(ends_dev))
        for dev_i, audio in enumerate(aligned):
            clip = audio[s_med:e_med]
            if 0 < len(clip) / sr < ZAMBIA_MAX_CLIP_S:
                clips_per_device[dev_i].append(clip)

    return clips_per_device, pointer


def _extract_consecutive_coughs(
    aligned: list[np.ndarray],
    cough_preds: list[int],
    pointer: int,
    sr: int,
    win: int,
) -> list[list[np.ndarray]]:
    """Segment individual coughs from the consecutive-cough portion of the session.

    Uses energy peak detection per device, then takes the consensus across devices
    so that only segments agreed upon by >= ZAMBIA_CONSENSUS_FRAC of devices are kept.
    """
    remaining = cough_preds[pointer:]
    remaining_set = set(remaining)
    min_baseline = int(MIN_BASELINE_S * sr)
    peak_dist    = int(PEAK_DIST_S * sr)

    coarse_per_device = []
    starts_per_device = []
    ends_per_device   = []

    for audio in aligned:
        # Build a coarse clip covering all remaining cough windows,
        # adding a half-window buffer at the edges of each cough run.
        chunks = []
        for p in remaining:
            s = max(0, p * win - win // 2) if (p - 1) not in remaining_set else p * win
            e = min(len(audio), p * win + win + win // 2) if (p + 1) not in remaining_set else p * win + win
            chunks.append(audio[s:e])
        coarse = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        coarse_per_device.append(coarse)

        # Detect energy peaks within the coarse clip
        win_e = int(ENERGY_WINDOW_S * sr)
        energy_ma = np.convolve(coarse ** 2, np.ones(win_e) / win_e, mode="same")
        baseline_level = ENERGY_THRESHOLD * energy_ma.max()
        peak_thresh    = PEAK_THRESH_FRAC * energy_ma.max()
        peaks, _ = find_peaks(energy_ma, height=peak_thresh, distance=peak_dist)

        # Merge peaks that are closer than peak_dist
        merged = [peaks[0]]
        for pk in peaks[1:]:
            if pk - merged[-1] < peak_dist:
                if energy_ma[pk] > energy_ma[merged[-1]]:
                    merged[-1] = pk
            else:
                merged.append(pk)
        peaks = merged

        # Find start of each cough: last long baseline crossing before the peak
        dev_starts = []
        for peak in peaks:
            seg_start = max(0, peak - 2 * sr)
            seg = energy_ma[seg_start:peak]
            idx = np.where(seg <= baseline_level)[0]
            if len(idx) == 0:
                start = seg_start
            else:
                runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                long_runs = [r for r in runs if len(r) >= min_baseline]
                ref = long_runs[-1] if long_runs else [idx[-1]]
                start = max(0, seg_start + ref[-1] - int(0.05 * sr))
            dev_starts.append(start)

        # Find end of each cough: first long baseline crossing after the peak
        dev_ends = []
        for peak, next_start in zip(peaks, dev_starts[1:] + [len(coarse) - 1]):
            seg = energy_ma[peak:next_start]
            idx = np.where(seg <= baseline_level)[0]
            if len(idx) == 0:
                end = next_start
            else:
                runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                long_runs = [r for r in runs if len(r) >= min_baseline]
                ref = long_runs[0] if long_runs else [idx[0]]
                end = min(len(coarse), peak + ref[0] + int(0.10 * sr))
            dev_ends.append(end)

        starts_per_device.append(dev_starts)
        ends_per_device.append(dev_ends)

    # Cross-device consensus: keep segments agreed by >= ZAMBIA_CONSENSUS_FRAC
    coarse_len = min(len(c) for c in coarse_per_device)
    mask = np.zeros((len(aligned), coarse_len))
    for dev_i, (dev_starts, dev_ends) in enumerate(zip(starts_per_device, ends_per_device)):
        for s, e in zip(dev_starts, dev_ends):
            s, e = min(s, coarse_len), min(e, coarse_len)
            mask[dev_i, s:e] = 1
    consensus = mask.mean(axis=0) >= ZAMBIA_CONSENSUS_FRAC
    # ensure diff detects both rising and falling edges
    consensus[[0, -1]] = False
    diff = np.diff(consensus.astype(int))
    seg_starts = np.where(diff ==  1)[0] + 1
    seg_ends   = np.where(diff == -1)[0] + 1

    clips_per_device: list[list[np.ndarray]] = [[] for _ in aligned]
    for dev_i, coarse in enumerate(coarse_per_device):
        for s, e in zip(seg_starts, seg_ends):
            clip = coarse[int(s):int(e)]
            if 0 < len(clip) / sr < ZAMBIA_MAX_CLIP_S:
                clips_per_device[dev_i].append(clip)

    return clips_per_device


def _load_subject_audios(
    subject_id: str,
    raw_dir: Path,
    devices: list[str],
    sr: int,
    reverse_relabeling: dict[tuple[str, str], str] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Load all available device recordings for a subject.

    If reverse_relabeling is provided, files that were stored on disk under a
    different (mislabeled) name are still found via the lookup table.
    """
    audios, device_names = [], []
    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        matched = [d for d in devices if d in folder.name]
        if not matched:
            continue
        # resolve the actual filename: may differ from subject_id if mislabeled
        actual_stem = subject_id
        if reverse_relabeling is not None:
            actual_stem = reverse_relabeling.get((folder.name, subject_id), subject_id)
        for ext in (".WAV", ".wav"):
            path = folder / (actual_stem + ext)
            if path.exists():
                y, _ = librosa.load(str(path), sr=sr)
                audios.append(y)
                device_names.append(matched[0])
                break
    return audios, device_names


def process_zambia(
    raw_dir: Path = ZAMBIA_RAW_DIR,
    out_dir: Path = ZAMBIA_OUT_DIR,
    metadata_path: Path = ZAMBIA_METADATA,
    change_log_path: Path = ZAMBIA_CHANGE_LOG,
) -> None:
    print("=== Zambia ===")

    for device in ZAMBIA_DEVICES:
        for location in ZAMBIA_LOCATIONS:
            (out_dir / f"{device}_{location}").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    known_ids = set(df["barcode"].tolist())

    forward_relabeling, reverse_relabeling = _load_relabeling(change_log_path)
    print(f"  Loaded {len(forward_relabeling)} re-labeling entries")

    # Collect canonical subject IDs from disk, applying forward relabeling so
    # mislabeled filenames are mapped to their correct subject ID before lookup.
    count_modification = 0
    subject_ids: set[str] = set()
    for folder in raw_dir.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            stem = f.stem.strip()
            canonical = forward_relabeling.get((folder.name, stem), stem)
            if canonical != stem:
                count_modification += 1
            if canonical in known_ids:
                subject_ids.add(canonical)
    print(f"  {len(subject_ids)} subjects with recordings")
    print(f"  {count_modification} filenames were modified due to re-labeling")

    # Lazy import: CLAP is an optional heavy dependency
    from msclap import CLAP
    clap_model = CLAP(version="2023", use_cuda=False)
    text_embeddings = clap_model.get_text_embeddings(ZAMBIA_CLAP_LABELS)
    win_samples = int(ZAMBIA_CLAP_WIN_S * SAMPLE_RATE)

    for subject_id in tqdm(sorted(subject_ids), desc="subjects"):
        location = next(
            (loc for prefix, loc in ZAMBIA_PREFIX_TO_LOCATION.items()
             if subject_id.startswith(prefix)),
            None,
        )
        if location is None:
            continue

        audios, audio_devices = _load_subject_audios(
            subject_id, raw_dir, ZAMBIA_DEVICES, SAMPLE_RATE,
            reverse_relabeling=reverse_relabeling,
        )
        if not audios:
            continue

        aligned = _align_to_reference(audios, SAMPLE_RATE)
        predictions = _clap_classify_windows(aligned[0], SAMPLE_RATE, clap_model, text_embeddings)
        cough_preds = [i for i, p in enumerate(predictions) if p == ZAMBIA_CLAP_COUGH_IDX]
        if not cough_preds:
            continue

        try:
            clips_indiv, pointer = _extract_individual_coughs(aligned, cough_preds, SAMPLE_RATE, win_samples)
        except Exception as exc:
            print(f"  {subject_id}: individual cough extraction failed — {exc}")
            clips_indiv = [[] for _ in aligned]
            pointer = 0

        try:
            clips_consec = _extract_consecutive_coughs(aligned, cough_preds, pointer, SAMPLE_RATE, win_samples)
        except Exception as exc:
            print(f"  {subject_id}: consecutive cough extraction failed — {exc}")
            clips_consec = [[] for _ in aligned]

        for dev_i, device in enumerate(audio_devices):
            save_dir = out_dir / f"{device}_{location}"
            all_clips = clips_indiv[dev_i] + clips_consec[dev_i]
            for clip_idx, clip in enumerate(all_clips, start=1):
                sf.write(save_dir / f"{subject_id}_{clip_idx}.wav", clip, SAMPLE_RATE)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    process_tbscreen()
    process_coda()
    process_zambia()


if __name__ == "__main__":
    main()
