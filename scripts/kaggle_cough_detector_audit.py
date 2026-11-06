"""
===================================================================================
KAGGLE NOTEBOOK CELL: TB COUGH DETECTOR CROSS-DEVICE VALIDATION & AUDIT
===================================================================================
Instructions:
  1. Open your Kaggle Notebook where the dataset 'google-health-ai' is attached.
  2. Paste this entire script into a single code cell.
  3. Run the cell (Shift + Enter).
  4. The script will:
     - Automatically locate the audio files for the 20 sampled patients across all 4 devices
       (Audio-Recorder, Phone A, Phone B, Phone C).
     - Run the 'median + 6*MAD' detector and 'relative dB' detector.
     - Calculate event counts per device and cross-device Coefficient of Variation (CV).
     - Save publication-quality plots, summary tables, and listening manifests.
     - Automatically create a download link for 'results_cough_detector_audit.zip'.
===================================================================================
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_SR = 16000
WIN_MS = 50
HOP_MS = 10
SMOOTH_FRAMES = 3
HP_CUTOFF_HZ = 80.0
MIN_EVENT_DUR_S = 0.08
MAX_EVENT_DUR_S = 1.20
MERGE_GAP_S = 0.15

DEVICES = ["Audio-Recorder", "Phone A", "Phone B", "Phone C"]

# Candidate Kaggle paths
CANDIDATE_SEARCH_PATHS = [
    Path("/kaggle/input/google-health-ai"),
    Path("/kaggle/input/google-health-ai/GHAI"),
    Path("/kaggle/input/google-health-ai/zambia"),
    Path("/kaggle/input/hear-dataset"),
    Path("/kaggle/input/zambia"),
    Path("/kaggle/working/data/raw/zambia"),
    Path("data/raw/zambia"),
    Path("."),
]

OUTPUT_DIR = Path("/kaggle/working/results/cough_detector_audit") if Path("/kaggle/working").exists() else Path("results/cough_detector_audit")

# 20 Sampled patients with has_ABC == True (seed 42)
SAMPLED_MANIFEST = [
    # Barcode, Device, Folder, File, Rel_Path, Facility
    ("01-399-0826", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0826.WAV", "Audio-Recorder-Kanyama/01-399-0826.WAV", "Kan"),
    ("01-399-0826", "Phone A", "Kanyama Phone A", "01-399-0826.wav", "Kanyama Phone A/01-399-0826.wav", "Kan"),
    ("01-399-0826", "Phone B", "Kanyama Phone B", "01-399-0826.wav", "Kanyama Phone B/01-399-0826.wav", "Kan"),
    ("01-399-0826", "Phone C", "Kanyama Phone C", "01-399-0826.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0826.wav", "Kan"),
    ("01-399-0827", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0827.WAV", "Audio-Recorder-Kanyama/01-399-0827.WAV", "Kan"),
    ("01-399-0827", "Phone A", "Kanyama Phone A", "01-399-0827.wav", "Kanyama Phone A/01-399-0827.wav", "Kan"),
    ("01-399-0827", "Phone B", "Kanyama Phone B", "01-399-0827.wav", "Kanyama Phone B/01-399-0827.wav", "Kan"),
    ("01-399-0827", "Phone C", "Kanyama Phone C", "01-399-0827.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0827.wav", "Kan"),
    ("01-399-0837", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0837.WAV", "Audio-Recorder-Kanyama/01-399-0837.WAV", "Kan"),
    ("01-399-0837", "Phone A", "Kanyama Phone A", "01-399-0837.wav", "Kanyama Phone A/01-399-0837.wav", "Kan"),
    ("01-399-0837", "Phone B", "Kanyama Phone B", "01-399-0837.wav", "Kanyama Phone B/01-399-0837.wav", "Kan"),
    ("01-399-0837", "Phone C", "Kanyama Phone C", "01-399-0837.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0837.wav", "Kan"),
    ("01-399-0852", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0852.WAV", "Audio-Recorder-Kanyama/01-399-0852.WAV", "Kan"),
    ("01-399-0852", "Phone A", "Kanyama Phone A", "01-399-0852.wav", "Kanyama Phone A/01-399-0852.wav", "Kan"),
    ("01-399-0852", "Phone B", "Kanyama Phone B", "01-399-0852.wav", "Kanyama Phone B/01-399-0852.wav", "Kan"),
    ("01-399-0852", "Phone C", "Kanyama Phone C", "01-399-0852.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0852.wav", "Kan"),
    ("01-399-0917", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0917.WAV", "Audio-Recorder-Kanyama/01-399-0917.WAV", "Kan"),
    ("01-399-0917", "Phone A", "Kanyama Phone A", "01-399-0917.wav", "Kanyama Phone A/01-399-0917.wav", "Kan"),
    ("01-399-0917", "Phone B", "Kanyama Phone B", "01-399-0917.wav", "Kanyama Phone B/01-399-0917.wav", "Kan"),
    ("01-399-0917", "Phone C", "Kanyama Phone C", "01-399-0917.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0917.wav", "Kan"),
    ("01-399-0918", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0918.WAV", "Audio-Recorder-Kanyama/01-399-0918.WAV", "Kan"),
    ("01-399-0918", "Phone A", "Kanyama Phone A", "01-399-0918.wav", "Kanyama Phone A/01-399-0918.wav", "Kan"),
    ("01-399-0918", "Phone B", "Kanyama Phone B", "01-399-0918.wav", "Kanyama Phone B/01-399-0918.wav", "Kan"),
    ("01-399-0918", "Phone C", "Kanyama Phone C", "01-399-0918.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0918.wav", "Kan"),
    ("01-399-0928", "Audio-Recorder", "Audio-Recorder-Kanyama", "01-399-0928.WAV", "Audio-Recorder-Kanyama/01-399-0928.WAV", "Kan"),
    ("01-399-0928", "Phone A", "Kanyama Phone A", "01-399-0928.wav", "Kanyama Phone A/01-399-0928.wav", "Kan"),
    ("01-399-0928", "Phone B", "Kanyama Phone B", "01-399-0928.wav", "Kanyama Phone B/01-399-0928.wav", "Kan"),
    ("01-399-0928", "Phone C", "Kanyama Phone C", "01-399-0928.wav", "Kanyama Phone C/Kanyama Phone C/01-399-0928.wav", "Kan"),
    ("02-399-0697", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0697.WAV", "Audio-Recorder-Chawama/02-399-0697.WAV", "Cha"),
    ("02-399-0697", "Phone A", "Kanyama Phone A", "02-399-0697.wav", "Kanyama Phone A/02-399-0697.wav", "Cha"),
    ("02-399-0697", "Phone B", "Kanyama Phone B", "02-399-0697.wav", "Kanyama Phone B/02-399-0697.wav", "Cha"),
    ("02-399-0697", "Phone C", "Kanyama Phone C", "02-399-0697.wav", "Kanyama Phone C/Kanyama Phone C/02-399-0697.wav", "Cha"),
    ("02-399-0732", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0732.WAV", "Audio-Recorder-Chawama/02-399-0732.WAV", "Cha"),
    ("02-399-0732", "Phone A", "Chawama Phone A", "02-399-0732.wav", "Chawama Phone A/Chawama Phone A/02-399-0732.wav", "Cha"),
    ("02-399-0732", "Phone B", "Chawama Phone B", "02-399-0732.wav", "Chawama Phone B/Chawama Phone B/02-399-0732.wav", "Cha"),
    ("02-399-0732", "Phone C", "Chawama Phone C", "02-399-0732.wav", "Chawama Phone C/Chawama Phone C/02-399-0732.wav", "Cha"),
    ("02-399-0748", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0748.WAV", "Audio-Recorder-Chawama/02-399-0748.WAV", "Cha"),
    ("02-399-0748", "Phone A", "Chawama Phone A", "02-399-0748.wav", "Chawama Phone A/Chawama Phone A/02-399-0748.wav", "Cha"),
    ("02-399-0748", "Phone B", "Chawama Phone B", "02-399-0748.wav", "Chawama Phone B/Chawama Phone B/02-399-0748.wav", "Cha"),
    ("02-399-0748", "Phone C", "Chawama Phone C", "02-399-0748.wav", "Chawama Phone C/Chawama Phone C/02-399-0748.wav", "Cha"),
    ("02-399-0760", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0760.WAV", "Audio-Recorder-Chawama/02-399-0760.WAV", "Cha"),
    ("02-399-0760", "Phone A", "Chawama Phone A", "02-399-0760.wav", "Chawama Phone A/Chawama Phone A/2-399-0760.wav", "Cha"),
    ("02-399-0760", "Phone B", "Chawama Phone B", "02-399-0760.wav", "Chawama Phone B/Chawama Phone B/02-399-0760.wav", "Cha"),
    ("02-399-0760", "Phone C", "Chawama Phone C", "02-399-0760.wav", "Chawama Phone C/Chawama Phone C/02-399-0760.wav", "Cha"),
    ("02-399-0762", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0762.WAV", "Audio-Recorder-Chawama/02-399-0762.WAV", "Cha"),
    ("02-399-0762", "Phone A", "Chawama Phone A", "02-399-0762.wav", "Chawama Phone A/Chawama Phone A/02-399-0762.wav", "Cha"),
    ("02-399-0762", "Phone B", "Chawama Phone B", "02-399-0762.wav", "Chawama Phone B/Chawama Phone B/02-399-0762.wav", "Cha"),
    ("02-399-0762", "Phone C", "Chawama Phone C", "02-399-0762.wav", "Chawama Phone C/Chawama Phone C/02-399-0762.wav", "Cha"),
    ("02-399-0767", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0767.WAV", "Audio-Recorder-Chawama/02-399-0767.WAV", "Cha"),
    ("02-399-0767", "Phone A", "Chawama Phone A", "02-399-0767.wav", "Chawama Phone A/Chawama Phone A/02-399-0767.wav", "Cha"),
    ("02-399-0767", "Phone B", "Chawama Phone B", "02-399-0767.wav", "Chawama Phone B/Chawama Phone B/02-399-0767.wav", "Cha"),
    ("02-399-0767", "Phone C", "Chawama Phone C", "02-399-0767.wav", "Chawama Phone C/Chawama Phone C/02-399-0767.wav", "Cha"),
    ("02-399-0776", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0776.WAV", "Audio-Recorder-Chawama/02-399-0776.WAV", "Cha"),
    ("02-399-0776", "Phone A", "Chawama Phone A", "02-399-0776.wav", "Chawama Phone A/Chawama Phone A/02-399-0776.wav", "Cha"),
    ("02-399-0776", "Phone B", "Chawama Phone B", "02-399-0776.wav", "Chawama Phone B/Chawama Phone B/02-399-0776.wav", "Cha"),
    ("02-399-0776", "Phone C", "Chawama Phone C", "02-399-0776.wav", "Chawama Phone C/Chawama Phone C/02-399-0776.wav", "Cha"),
    ("02-399-0785", "Audio-Recorder", "Audio-Recorder-Chawama", "02-399-0785.WAV", "Audio-Recorder-Chawama/02-399-0785.WAV", "Cha"),
    ("02-399-0785", "Phone A", "Chawama Phone A", "02-399-0785.wav", "Chawama Phone A/Chawama Phone A/02-399-0785.wav", "Cha"),
    ("02-399-0785", "Phone B", "Chawama Phone B", "02-399-0785.wav", "Chawama Phone B/Chawama Phone B/02-399-0785.wav", "Cha"),
    ("02-399-0785", "Phone C", "Chawama Phone C", "02-399-0785.wav", "Chawama Phone C/Chawama Phone C/02-399-0785.wav", "Cha"),
    ("03-399-0301", "Audio-Recorder", "Audio-Recorder-Chainda-South", "03-399-0301.WAV", "Audio-Recorder-Chainda-South/03-399-0301.WAV", "Chai"),
    ("03-399-0301", "Phone A", "Chainda South Phone A", "03-399-0301.wav", "Chainda South Phone A/Chainda South Phone A/03-399-0301.wav", "Chai"),
    ("03-399-0301", "Phone B", "Chainda South Phone B", "03-399-0301.wav", "Chainda South Phone B/Chainda South Phone B/03-399-0301.wav", "Chai"),
    ("03-399-0301", "Phone C", "Chainda South Phone C", "03-399-0301.wav", "Chainda South Phone C/Chainda South Phone C/03-399-0301.wav", "Chai"),
    ("03-399-0319", "Audio-Recorder", "Audio-Recorder-Chainda-South", "03-399-0319.WAV", "Audio-Recorder-Chainda-South/03-399-0319.WAV", "Chai"),
    ("03-399-0319", "Phone A", "Chainda South Phone A", "03-399-0319.wav", "Chainda South Phone A/Chainda South Phone A/03-399-0319.wav", "Chai"),
    ("03-399-0319", "Phone B", "Chainda South Phone B", "03-399-0319.wav", "Chainda South Phone B/Chainda South Phone B/03-399-0319.wav", "Chai"),
    ("03-399-0319", "Phone C", "Chainda South Phone C", "03-399-0319.wav", "Chainda South Phone C/Chainda South Phone C/03-399-0319.wav", "Chai"),
    ("03-399-0320", "Audio-Recorder", "Audio-Recorder-Chainda-South", "03-399-0320.WAV", "Audio-Recorder-Chainda-South/03-399-0320.WAV", "Chai"),
    ("03-399-0320", "Phone A", "Chainda South Phone A", "03-399-0320.wav", "Chainda South Phone A/Chainda South Phone A/03-399-0320.wav", "Chai"),
    ("03-399-0320", "Phone B", "Chainda South Phone B", "03-399-0320.wav", "Chainda South Phone B/Chainda South Phone B/03-399-0320.wav", "Chai"),
    ("03-399-0320", "Phone C", "Chainda South Phone C", "03-399-0320.wav", "Chainda South Phone C/Chainda South Phone C/03-399-0320.wav", "Chai"),
    ("03-399-0328", "Audio-Recorder", "Audio-Recorder-Chainda-South", "03-399-0328.WAV", "Audio-Recorder-Chainda-South/03-399-0328.WAV", "Chai"),
    ("03-399-0328", "Phone A", "Chainda South Phone A", "03-399-0328.wav", "Chainda South Phone A/Chainda South Phone A/03-399-0328.wav", "Chai"),
    ("03-399-0328", "Phone B", "Chainda South Phone B", "03-399-0328.wav", "Chainda South Phone B/Chainda South Phone B/03-399-0328.wav", "Chai"),
    ("03-399-0328", "Phone C", "Chainda South Phone C", "03-399-0328.wav", "Chainda South Phone C/Chainda South Phone C/03-399-0328.wav", "Chai"),
    ("03-399-0330", "Audio-Recorder", "Audio-Recorder-Chainda-South", "03-399-0330.WAV", "Audio-Recorder-Chainda-South/03-399-0330.WAV", "Chai"),
    ("03-399-0330", "Phone A", "Chainda South Phone A", "03-399-0330.wav", "Chainda South Phone A/Chainda South Phone A/03-399-0330.wav", "Chai"),
    ("03-399-0330", "Phone B", "Chainda South Phone B", "03-399-0330.wav", "Chainda South Phone B/Chainda South Phone B/03-399-0330.wav", "Chai"),
    ("03-399-0330", "Phone C", "Chainda South Phone C", "03-399-0330.wav", "Chainda South Phone C/Chainda South Phone C/03-399-0330.wav", "Chai"),
]


# -----------------------------------------------------------------------------
# Core DSP & Detection Logic
# -----------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio file cleanly across librosa, soundfile, or scipy."""
    path = str(path)
    try:
        import soundfile as sf
        data, native_sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if native_sr != target_sr:
            num_samples = int(len(data) * target_sr / native_sr)
            data = signal.resample(data, num_samples)
        return data.astype(np.float32), target_sr
    except Exception:
        pass

    try:
        import librosa
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def highpass_filter(x: np.ndarray, cutoff: float = HP_CUTOFF_HZ, fs: int = TARGET_SR, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, x).astype(np.float32)


def compute_frame_rms_envelope(x: np.ndarray, sr: int = TARGET_SR, win_ms: float = WIN_MS, hop_ms: float = HOP_MS) -> Tuple[np.ndarray, np.ndarray]:
    win_len = int(win_ms * sr / 1000)
    hop_len = int(hop_ms * sr / 1000)
    pad_len = win_len // 2
    x_padded = np.pad(x, pad_len, mode="reflect")
    n_frames = (len(x_padded) - win_len) // hop_len + 1

    from numpy.lib.stride_tricks import as_strided
    frames = as_strided(
        x_padded,
        shape=(n_frames, win_len),
        strides=(x_padded.strides[0] * hop_len, x_padded.strides[0]),
    )
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    kernel = np.ones(SMOOTH_FRAMES) / float(SMOOTH_FRAMES)
    rms = np.convolve(rms, kernel, mode="same")
    t_axis = np.arange(n_frames) * (hop_ms / 1000.0)
    return rms, t_axis


def extract_events_from_mask(mask: np.ndarray, time_axis: np.ndarray) -> List[Tuple[float, float, float]]:
    if not np.any(mask):
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    if len(starts) == 0:
        return []

    intervals = []
    for s, e in zip(starts, ends):
        t_start = float(time_axis[s])
        t_end = float(time_axis[min(e, len(time_axis) - 1)])
        intervals.append([t_start, t_end])

    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        if cur[0] - prev[1] <= MERGE_GAP_S:
            prev[1] = max(prev[1], cur[1])
        else:
            merged.append(cur)

    final_events = []
    for s, e in merged:
        dur = e - s
        if MIN_EVENT_DUR_S <= dur <= MAX_EVENT_DUR_S:
            final_events.append((round(s, 3), round(e, 3), round(dur, 3)))
    return final_events


def detect_coughs_mad(rms: np.ndarray, t_axis: np.ndarray, k: float = 6.0) -> List[Tuple[float, float, float]]:
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)))
    thresh = med + k * mad
    return extract_events_from_mask(rms >= thresh, t_axis)


def detect_coughs_relative_db(rms: np.ndarray, t_axis: np.ndarray, delta_db: float = 20.0, mode: str = "noise_floor") -> List[Tuple[float, float, float]]:
    rms_db = 20.0 * np.log10(rms + 1e-7)
    if mode == "noise_floor":
        nf = float(np.percentile(rms_db, 10.0))
        thresh = nf + delta_db
    elif mode == "peak":
        pk = float(np.max(rms_db))
        thresh = pk - delta_db
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return extract_events_from_mask(rms_db >= thresh, t_axis)


# -----------------------------------------------------------------------------
# Main Execution in Kaggle
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("RUNNING KAGGLE COUGH DETECTOR AUDIT ON 20 PATIENTS (80 AUDIO FILES)")
    print("=" * 75)

    # 1. Locate audio files
    manifest_df = pd.DataFrame(SAMPLED_MANIFEST, columns=["barcode", "device", "folder", "file", "rel_path", "facility_code"])
    resolved_paths = []

    for _, row in manifest_df.iterrows():
        found_p = None
        for base in CANDIDATE_SEARCH_PATHS:
            cand = base / row["rel_path"]
            if cand.is_file():
                found_p = cand
                break
            if not base.is_dir():
                continue
            matches = list(base.rglob(row["file"]))
            if matches:
                found_p = matches[0]
                break
        resolved_paths.append(str(found_p) if found_p else None)

    manifest_df["resolved_path"] = resolved_paths
    n_found = manifest_df["resolved_path"].notna().sum()
    print(f"Files Found: {n_found} / {len(manifest_df)}")

    if n_found < len(manifest_df):
        print("\nWARNING: Some audio files were not found. Inspecting attached Kaggle datasets:")
        if Path("/kaggle/input").exists():
            print("Contents of /kaggle/input:")
            for p in Path("/kaggle/input").iterdir():
                print(f"  - {p}")
        else:
            print("Not running in standard /kaggle/input path. Set CANDIDATE_SEARCH_PATHS appropriately.")
        return

    # Detector configs
    detector_configs = [
        ("MAD_k4", lambda r, t: detect_coughs_mad(r, t, k=4.0)),
        ("MAD_k5", lambda r, t: detect_coughs_mad(r, t, k=5.0)),
        ("MAD_k6 (PRIMARY)", lambda r, t: detect_coughs_mad(r, t, k=6.0)),
        ("MAD_k7", lambda r, t: detect_coughs_mad(r, t, k=7.0)),
        ("MAD_k8", lambda r, t: detect_coughs_mad(r, t, k=8.0)),
        ("RelDB_NF_+12dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=12.0, mode="noise_floor")),
        ("RelDB_NF_+15dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=15.0, mode="noise_floor")),
        ("RelDB_NF_+20dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=20.0, mode="noise_floor")),
        ("RelDB_NF_+25dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=25.0, mode="noise_floor")),
        ("RelDB_Peak_-15dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=15.0, mode="peak")),
        ("RelDB_Peak_-20dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=20.0, mode="peak")),
        ("RelDB_Peak_-25dB", lambda r, t: detect_coughs_relative_db(r, t, delta_db=25.0, mode="peak")),
    ]

    patient_records = []
    listening_records = []
    barcodes = sorted(manifest_df["barcode"].unique().tolist())

    for idx, bcode in enumerate(barcodes, 1):
        p_sub = manifest_df[manifest_df["barcode"] == bcode]
        fac = p_sub["facility_code"].iloc[0]
        print(f"[{idx:02d}/20] Processing {bcode} ({fac})...")

        # Load and compute envelopes once per device
        dev_envelopes = {}
        for dev in DEVICES:
            path_str = p_sub[p_sub["device"] == dev]["resolved_path"].iloc[0]
            audio, _ = load_audio(path_str)
            audio_hp = highpass_filter(audio)
            rms, t_axis = compute_frame_rms_envelope(audio_hp)
            dev_envelopes[dev] = (rms, t_axis)

        # Evaluate detectors
        for d_name, d_func in detector_configs:
            counts = {}
            for dev in DEVICES:
                rms, t = dev_envelopes[dev]
                evs = d_func(rms, t)
                counts[dev] = len(evs)

                if d_name == "MAD_k6 (PRIMARY)":
                    for ev_i, (s, e, dur) in enumerate(evs, 1):
                        listening_records.append({
                            "barcode": bcode,
                            "facility": fac,
                            "device": dev,
                            "event_idx": ev_i,
                            "start_s": s,
                            "end_s": e,
                            "dur_s": dur,
                        })

            cnt_vals = [counts[d] for d in DEVICES]
            m_cnt = float(np.mean(cnt_vals))
            s_cnt = float(np.std(cnt_vals, ddof=1)) if len(cnt_vals) > 1 else 0.0
            cv = float(s_cnt / m_cnt) if m_cnt > 0 else 0.0

            patient_records.append({
                "barcode": bcode,
                "facility": fac,
                "detector": d_name,
                "count_Recorder": counts["Audio-Recorder"],
                "count_Phone_A": counts["Phone A"],
                "count_Phone_B": counts["Phone B"],
                "count_Phone_C": counts["Phone C"],
                "mean_count": round(m_cnt, 2),
                "std_count": round(s_cnt, 2),
                "cv_across_devices": round(cv, 4),
            })

    # Save CSVs
    df_patient = pd.DataFrame(patient_records)
    df_listening = pd.DataFrame(listening_records)
    df_patient.to_csv(OUTPUT_DIR / "per_patient_device_counts.csv", index=False)
    df_listening.to_csv(OUTPUT_DIR / "listening_validation_manifest.csv", index=False)

    # Summary
    summary_list = []
    for d_name, _ in detector_configs:
        sub = df_patient[df_patient["detector"] == d_name]
        mean_cv = sub["cv_across_devices"].mean()
        med_cv = sub["cv_across_devices"].median()
        avg_cnt = sub["mean_count"].mean()
        avg_rec = sub["count_Recorder"].mean()
        avg_pa = sub["count_Phone_A"].mean()
        avg_pb = sub["count_Phone_B"].mean()
        avg_pc = sub["count_Phone_C"].mean()

        dev_cols = ["count_Recorder", "count_Phone_A", "count_Phone_B", "count_Phone_C"]
        zero_rate = float(np.mean((sub[dev_cols] == 0).values))

        summary_list.append({
            "Detector": d_name,
            "Mean_CV": round(mean_cv, 4),
            "Median_CV": round(med_cv, 4),
            "Avg_Count": round(avg_cnt, 1),
            "Rec_Avg": round(avg_rec, 1),
            "PhoneA_Avg": round(avg_pa, 1),
            "PhoneB_Avg": round(avg_pb, 1),
            "PhoneC_Avg": round(avg_pc, 1),
            "Zero_Failure_Rate": round(zero_rate, 3),
        })

    df_summary = pd.DataFrame(summary_list).sort_values("Mean_CV")
    df_summary.to_csv(OUTPUT_DIR / "detector_comparison_summary.csv", index=False)

    # Diagnostic Plots
    plt.figure(figsize=(12, 6))
    detectors = [d[0] for d in detector_configs]
    data_to_plot = [df_patient[df_patient["detector"] == d]["cv_across_devices"].values for d in detectors]
    plt.boxplot(data_to_plot, labels=detectors, showmeans=True)
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Coefficient of Variation across Devices (CV)")
    plt.title("Cough Detector Cross-Device Stability Comparison (Lower is Better)")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "detector_cv_comparison.png", dpi=300)
    plt.close()

    # Package Zip
    zip_f = OUTPUT_DIR.parent / "results_cough_detector_audit.zip"
    with zipfile.ZipFile(zip_f, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                abs_f = Path(root) / file
                zf.write(abs_f, arcname=str(abs_f.relative_to(OUTPUT_DIR.parent)))

    print("\n" + "=" * 75)
    print("AUDIT RESULTS TABLE:")
    print(df_summary.to_string(index=False))
    print("=" * 75)
    print(f"\n[Done] All plots, CSVs, and audit reports saved to: {OUTPUT_DIR}")
    print(f"[Done] Downloadable zip archive created: {zip_f}")

if __name__ == "__main__":
    main()

