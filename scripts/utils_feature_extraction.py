"""
scripts/utils_feature_extraction.py

Handcrafted time/frequency-domain acoustic feature extraction.

`stFeatureExtraction` computes 68 short-time frame-level descriptors (zero-crossing
rate, frame energy/entropy, spectral statistics, log band powers, MFCCs, harmonic
ratio/f0, chroma) over an audio signal. `extract_time_freq_summary` summarises those
frame-level features into a single fixed-length vector by concatenating the per-feature
mean and std across frames (68 * 2 = 136 dimensions).

Ported from the authors' feature-extraction utilities (translated in part from the
pyAudioAnalysis routines of Giannakopoulos & Pikrakis).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, lfilter, periodogram, welch
from scipy.signal import medfilt
from scipy.signal.windows import get_window
from scipy.stats import tmean, tstd

EPS = 2 ** (-52)
N_FRAME_FEATURES = 68


# =============================================================================
# FEATURE NAMES
# =============================================================================

def get_feature_names() -> list[str]:
    names = [
        "ZCR", "frame_energy", "energy_entropy", "wentropy", "RMS", "frame_std",
        "spectral_kurtosis", "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_entropy", "spectral_flux", "spectral_rolloff", "spectral_crest_factor",
        "spectral_decrease", "spectral_flatness", "spectral_slope", "spectral_skewness",
        "mean_freq", "welch_mean", "welch_max",
    ]
    names += [f"log_bandpower_{i}" for i in range(20)]
    names += [f"mfcc_{i}" for i in range(13)]
    names += ["harmonic_hr", "harmonic_f0"]
    names += [f"chroma_vector_{i}" for i in range(12)]
    names += ["chroma_vector_1_medfilt"]
    return names


# =============================================================================
# FRAME-LEVEL FEATURE PRIMITIVES
# =============================================================================

def feature_zcr(frame: np.ndarray) -> float:
    return np.sum(np.abs(np.diff(frame > 0))) / len(frame)


def feature_energy_entropy(frame: np.ndarray, n_blocks: int) -> float:
    energy = np.sum(np.abs(frame) ** 2)
    sub_len = int(len(frame) / n_blocks)
    frame_cut = frame[:sub_len * n_blocks].copy()
    sub_frames = np.reshape(frame_cut, [n_blocks, sub_len])
    s = np.sum(sub_frames ** 2, axis=1) / (energy + EPS)
    return -np.sum(s * np.log2(s + EPS))


def feature_spectral_centroid(frame_fft: np.ndarray, fs: int) -> tuple[float, float]:
    frame_len = len(frame_fft)
    m = (fs / (2 * frame_len)) * np.arange(1, frame_len + 1)
    frame_fft = frame_fft / np.max(frame_fft)
    mean = np.sum(m * frame_fft) / (np.sum(frame_fft + EPS))
    std = (np.sum(((m - mean) ** 2) * frame_fft) / (np.sum(frame_fft) + EPS)) ** 0.5
    return mean, std


def feature_spectral_kurtosis(frame_fft: np.ndarray, fs: int) -> float:
    f = np.linspace(0, fs / 2, frame_fft.shape[0])
    sum_frame = np.sum(frame_fft, axis=0)
    if sum_frame == 0:
        return 0
    mu, std = feature_spectral_centroid(frame_fft, fs)
    return np.sum(((f - mu) ** 4) * frame_fft) / ((std ** 4) * sum_frame * frame_fft.shape[0]) - 3


def feature_spectral_skewness(frame_fft: np.ndarray, fs: int) -> float:
    f = np.linspace(0, fs / 2, frame_fft.shape[0])
    sum_frame = np.sum(frame_fft, axis=0)
    if sum_frame == 0:
        return 0
    mu, std = feature_spectral_centroid(frame_fft, fs)
    return np.sum(((f - mu) ** 3) * frame_fft) / ((std ** 3) * sum_frame * frame_fft.shape[0])


def feature_spectral_entropy(frame_fft: np.ndarray, n_blocks: int) -> float:
    return feature_energy_entropy(frame_fft, n_blocks)


def feature_spectral_flux(frame_fft: np.ndarray, frame_fft_prev: np.ndarray) -> float:
    frame_fft = frame_fft / np.sum(frame_fft)
    with np.errstate(invalid="ignore"):
        frame_fft_prev = frame_fft_prev / np.sum(frame_fft_prev)
    return np.sum((frame_fft - frame_fft_prev) ** 2)


def feature_spectral_rolloff(frame_fft: np.ndarray, c: float) -> float:
    total_energy = np.sum(frame_fft ** 2)
    cur_energy = 0.0
    count = 0
    frame_len = len(frame_fft)
    while (cur_energy <= c * total_energy) and (count <= frame_len):
        cur_energy += frame_fft[count] ** 2
        count += 1
    return (count - 1) / frame_len


def feature_spectral_crest_factor(frame_fft: np.ndarray) -> float:
    sum_frame = np.sum(frame_fft, axis=0)
    return 0 if sum_frame == 0 else np.max(frame_fft, axis=0) / sum_frame


def feature_spectral_decrease(frame_fft: np.ndarray) -> float:
    sum_frame = np.sum(frame_fft[1:], axis=0)
    k = np.arange(0, frame_fft.shape[0])
    k[0] = 1
    return 0 if sum_frame == 0 else np.dot(frame_fft - frame_fft[0], 1 / k) / sum_frame


def feature_spectral_flatness(frame_fft: np.ndarray) -> float:
    log_fft = np.log(frame_fft + 1e-20)
    return 0 if np.sum(frame_fft) == 0 else np.exp(np.mean(log_fft, axis=0)) / np.mean(frame_fft, axis=0)


def feature_spectral_slope(frame_fft: np.ndarray) -> float:
    mu = np.mean(np.abs(frame_fft), axis=0)
    kmu = np.arange(0, frame_fft.shape[0]) - frame_fft.shape[0] / 2
    frame_fft = frame_fft - mu
    return np.dot(kmu, frame_fft) / np.dot(kmu, kmu)


def bandpower(signal: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    f, pxx = periodogram(signal, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(pxx[ind_min:ind_max], f[ind_min:ind_max])


def meanfreq(signal: np.ndarray, fs: int) -> float:
    f, pxx = periodogram(signal, fs)
    pxx = np.reshape(pxx, (1, -1))
    width = np.tile(f[1] - f[0], (1, pxx.shape[1]))
    f = np.reshape(f, (1, -1))
    p = pxx * width
    pwr = np.sum(p)
    return np.dot(p, f.T) / pwr


def feature_harmonic(frame: np.ndarray, fs: int) -> tuple[float, float]:
    m_max = int(np.round(0.016 * fs))
    r = correlate(frame, frame)
    g = r[len(frame) - 1]
    r = r[len(frame):]

    i, m0 = 1, len(r) + 1
    while i < len(r):
        if (r[i] < 0) and (r[i - 1] >= 0):
            m0 = i
            break
        i += 1

    m_max = min(m_max, len(r))
    gamma = np.zeros([m_max])
    csum = np.cumsum(frame ** 2)
    gamma[m0:m_max] = r[m0:m_max] / (
        (g * csum[np.arange(len(csum) - m0 - 2, len(csum) - m_max - 2, -1).astype(int)]) ** 0.5 + EPS
    )
    z = feature_zcr(gamma)

    if z > 0.15:
        return 0, 0
    if len(gamma) == 0:
        return 1, 0
    hr = np.max(gamma)
    blag = np.argmax(gamma)
    f0 = fs / (blag + 1)
    if f0 > 5000:
        f0 = 0
    if hr < 0.1:
        f0 = 0
    return hr, f0


def feature_chroma_vector(frame: np.ndarray, fs: int) -> np.ndarray:
    frame = frame / np.max(np.abs(frame))
    tone_analysis, num_bins = 12, 12
    window_length = len(frame)
    freqs = np.arange(0, int(window_length / 2) * fs / window_length, fs / window_length, dtype=int)

    f0, i, f = 55, 0, []
    while True:
        f.append(f0 * 2 ** (i / tone_analysis))
        if f[i] > freqs[-1]:
            break
        i += 1
    f = np.array(f[:i])

    fft_mag = np.abs(np.fft.fft(frame))[:int(window_length / 2)]
    if np.max(fft_mag) <= EPS:
        return np.zeros(num_bins)

    dead = np.where((freqs < f[0]) | (freqs > 2000))[0]
    fft_mag[dead] = 0

    c1 = fft_mag - np.concatenate([[0], fft_mag[:-1]])
    c2 = fft_mag - np.concatenate([fft_mag[1:], [0]])
    dead = np.where(~((c1 > 0) & (c2 > 0)))[0]
    fft_mag[dead] = 0

    nonzero = np.where(fft_mag > 0)[0]
    if len(nonzero) == 0:
        return np.zeros(num_bins)

    ytemp, ctemp = np.zeros(num_bins), np.zeros(num_bins)
    for k in nonzero:
        imin = np.argmin(np.abs(freqs[k] - f))
        h = (imin + 1) % num_bins
        if h == 0:
            h = num_bins
        ytemp[h - 1] += fft_mag[k]
        ctemp[h - 1] += 1
    return ytemp / (ctemp + 1)


def get_dft(signal: np.ndarray, fs: int) -> np.ndarray:
    n = len(signal)
    fft = np.abs(np.fft.fft(signal)) / n
    return fft[:int(np.ceil(n / 2))]


# =============================================================================
# MFCC INITIALISATION / COMPUTATION
# =============================================================================

def feature_mfccs_init(window_length: int, fs: int) -> dict:
    p = {
        "cepstralCoefficients": 13,
        "fftSize": round(window_length / 2),
        "lowestFrequency": 133.3333,
        "linearFilters": 13,
        "linearSpacing": 66.66666666,
        "logFilters": 27,
        "logSpacing": 1.0711703,
    }
    p["totalFilters"] = p["linearFilters"] + p["logFilters"]
    freqs = p["lowestFrequency"] + np.arange(0, p["linearFilters"]) * p["linearSpacing"]
    freqs = np.concatenate(
        [freqs, freqs[p["linearFilters"] - 1] * p["logSpacing"] ** np.arange(1, p["logFilters"] + 3)]
    )
    p["lower"] = freqs[:p["totalFilters"]]
    p["center"] = freqs[1:p["totalFilters"] + 1]
    p["upper"] = freqs[2:p["totalFilters"] + 2]
    p["mfccFilterWeights"] = np.zeros([p["totalFilters"], p["fftSize"]])
    p["triangleHeight"] = 2.0 / (p["upper"] - p["lower"])
    p["fftFreqs"] = np.arange(0, p["fftSize"]) / p["fftSize"] * fs

    for chan in range(p["totalFilters"]):
        p["mfccFilterWeights"][chan, :] = (
            (p["fftFreqs"] > p["lower"][chan]) & (p["fftFreqs"] <= p["center"][chan])
        ) * p["triangleHeight"][chan] * (p["fftFreqs"] - p["lower"][chan]) / (
            p["center"][chan] - p["lower"][chan]
        ) + (
            (p["fftFreqs"] > p["center"][chan]) & (p["fftFreqs"] < p["upper"][chan])
        ) * p["triangleHeight"][chan] * (p["upper"][chan] - p["fftFreqs"]) / (
            p["upper"][chan] - p["center"][chan]
        )

    p["mfccDCTMatrix"] = (
        1 / np.sqrt(p["totalFilters"] / 2)
        * np.cos(
            np.matmul(
                np.arange(0, p["cepstralCoefficients"]).reshape([-1, 1]),
                (2 * np.arange(0, p["totalFilters"]) + 1).reshape([1, -1]),
            )
            * np.pi / (2 * p["totalFilters"])
        )
    )
    p["mfccDCTMatrix"][0, :] *= np.sqrt(2) / 2
    return p


def feature_mfccs(window_fft: np.ndarray, mfcc_params: dict, eps: float = EPS) -> np.ndarray:
    weights = mfcc_params["mfccFilterWeights"]
    if weights.shape[1] != window_fft.shape[0]:
        if weights.shape[1] > window_fft.shape[0]:
            weights = weights[:, :window_fft.shape[0]]
        else:
            window_fft = window_fft[:weights.shape[1]]
    ear_mag = np.log10(np.matmul(weights, window_fft) + eps)
    return np.matmul(mfcc_params["mfccDCTMatrix"], ear_mag)


# =============================================================================
# SHORT-TERM FEATURE EXTRACTION
# =============================================================================

def stFeatureExtraction(  # noqa: N802 (kept for compatibility with original API)
    signal: np.ndarray,
    fs: int,
    win: float,
    step: float,
    frequency_limits: tuple[float, float],
) -> np.ndarray:
    """Compute the 68 frame-level features; returns array of shape (n_frames, 68)."""
    if signal.ndim > 1:
        signal = np.sum(signal, axis=1) / 2  # to mono

    window_length = round(win * fs)
    step = round(step * fs)
    n_frames = int(np.floor((len(signal) - window_length) / step) + 1)
    features = np.zeros((N_FRAME_FEATURES, n_frames))
    ham = get_window("hamming", window_length)
    mfcc_params = feature_mfccs_init(window_length, fs)

    cur_pos = 0
    frame_fft_prev = None
    for i in range(n_frames):
        frame = signal[cur_pos:cur_pos + window_length] * ham
        frame_fft = get_dft(frame, fs)
        if frame_fft_prev is None:
            frame_fft_prev = frame_fft

        if np.sum(np.abs(frame)) > EPS:
            features[0, i] = feature_zcr(frame)
            features[1, i] = np.sum(np.abs(frame ** 2)) / len(frame)
            features[2, i] = feature_energy_entropy(frame, 10)
            features[3, i] = 0  # wentropy (not implemented)
            features[4, i] = (np.sum(np.abs(frame) ** 2) / len(frame)) ** 0.5
            features[5, i] = tstd(frame)
            features[6, i] = feature_spectral_kurtosis(frame_fft, fs)
            features[7, i], features[8, i] = feature_spectral_centroid(frame_fft, fs)
            features[9, i] = feature_spectral_entropy(frame_fft, 10)
            features[10, i] = feature_spectral_flux(frame_fft, frame_fft_prev)
            features[11, i] = feature_spectral_rolloff(frame_fft, 0.90)
            features[12, i] = feature_spectral_crest_factor(frame_fft)
            features[13, i] = feature_spectral_decrease(frame_fft)
            features[14, i] = feature_spectral_flatness(frame_fft)
            features[15, i] = feature_spectral_slope(frame_fft)
            features[16, i] = feature_spectral_skewness(frame_fft, fs)
            features[17, i] = meanfreq(frame, fs)
            features[18, i] = tmean(welch(frame, nperseg=int(frame.shape[0] / 8), detrend=False)[1] * 1000)
            features[19, i] = np.max(welch(frame, nperseg=int(frame.shape[0] / 8), detrend=False)[1] * 1000)

            band = np.floor((frequency_limits[1] - frequency_limits[0]) / 40)
            for j in range(20):
                features[20 + j, i] = np.log(
                    bandpower(frame, fs, frequency_limits[0] + j * band, frequency_limits[0] + (j + 1) * band)
                )

            features[40:53, i] = feature_mfccs(frame_fft, mfcc_params)
            features[53, i], features[54, i] = feature_harmonic(frame, fs)
            features[55:67, i] = feature_chroma_vector(frame, fs)
        # else: leave zeros

        cur_pos += step
        frame_fft_prev = frame_fft

    features[67, :] = medfilt(features[66, :], 3)
    return features.T


def extract_time_freq_summary(
    audio: np.ndarray,
    fs: int,
    win: float = 0.02,
    step: float = 0.01,
    frequency_limits: tuple[float, float] = (0, 8000),
) -> np.ndarray:
    """Summarise frame-level features into a 136-dim vector (mean | std)."""
    frames = stFeatureExtraction(audio, fs, win, step, frequency_limits)  # (n_frames, 68)
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
