import numpy as np
import librosa
import random

# -----------------------------
# 1) Crop & Pad
# -----------------------------
def aug_crop_pad(y, sr, p_apply = 1.0, min_frac = 0.5, max_frac = 1.0):
    """
    Random crop (or pad) to target_duration seconds.
    Params:
        target_duration: seconds
        p_apply: probability to apply
    """
    if random.random() > p_apply:
        return y
    original_len = len(y)
    
    frac = random.uniform(min_frac, max_frac)
    start = random.randint(0, max(0, original_len - int(original_len * frac)))
    y = y[start:start + int(original_len * frac)]
    # pad
    pad_left = random.randint(0, original_len - len(y))
    pad_right = original_len - len(y) - pad_left

    return np.pad(y, (pad_left, pad_right), mode='constant')

# -------------------------------------------------
# 2) Add Noise (Gaussian)
# -------------------------------------------------
def aug_add_noise(y, sr, snr_db=20.0, p_apply=1.0):
    """
    Add Gaussian noise to reach desired SNR (dB).
    Higher snr_db = less noise.
    """
    if random.random() > p_apply:
        return y
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    snr = 10 ** (snr_db / 20.0)
    noise_rms = rms / snr
    noise = np.random.normal(0, noise_rms, len(y))
    return y + noise

# -------------------------------------------------
# 3) Brownian Tape Speed
# -------------------------------------------------
def aug_brownian_tape_speed(y, sr, max_rate_perturb=0.1, n_walk_steps=3, p_apply=1.0):
    """
    Simulate small random speed fluctuations via piecewise resampling.
    max_rate_perturb: maximum relative speed change (e.g. 0.1 -> +/-10%)
    n_walk_steps: how many segments to split into (>=1)
    """
    if random.random() > p_apply or n_walk_steps <= 1:
        return y
    n = len(y)
    steps = np.linspace(0, n, n_walk_steps + 1, dtype=int)
    segs = []
    for i in range(n_walk_steps):
        seg = y[steps[i]:steps[i+1]]
        if len(seg) < 2:
            continue
        factor = 1.0 + np.random.uniform(-max_rate_perturb, max_rate_perturb)
        seg_rs = librosa.resample(seg.astype(float), orig_sr = sr, target_sr = int(sr * factor))
        segs.append(seg_rs)
    if not segs:
        return y
    return np.concatenate(segs)

# -------------------------------------------------
# 4) Scaling (Gain)
# -------------------------------------------------
def aug_scaling(y, sr, min_gain=0.25, max_gain=1.75, p_apply=1.0):
    """
    Random gain between min and max.
    """
    if random.random() > p_apply:
        return y
    g = np.random.uniform(min_gain, max_gain)
    return y * g

# -------------------------------------------------
# 5) Pitch Shift
# -------------------------------------------------
def aug_pitch_shift(y, sr, max_steps=2.0, p_apply=1.0):
    """
    Pitch shift by +/- max_steps semitones (uniform).
    Note: librosa.effects.pitch_shift uses phase vocoder; works on mono.
    """
    if random.random() > p_apply:
        return y
    steps = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr = sr, n_steps=steps)

# -------------------------------------------------
# 6) Time Stretch
# -------------------------------------------------
def aug_time_stretch(y, sr, min_rate=0.9, max_rate=1.1, p_apply=1.0):
    """
    Speed up / slow down audio by factor in [min_rate, max_rate].
    Uses librosa.effects.time_stretch on STFT phase-vocoder (mono).
    Note: time_stretch expects a fixed windowed STFT; very short signals may be problematic.
    """
    if random.random() > p_apply:
        return y
    rate = np.random.uniform(min_rate, max_rate)
    if rate <= 0:
        return y
    try:
        return librosa.effects.time_stretch(y, rate)
    except:
        # fallback: resample version
        yr = librosa.resample(y, orig_sr = sr, target_sr = int(sr * rate))
        return librosa.resample(yr, orig_sr = int(sr * rate), target_sr = sr)

# -------------------------------------------------
# 7) Circular Time Shift
# -------------------------------------------------
def aug_circular_shift(y, sr, p_apply=1.0):
    """
    Circularly roll the signal by a random amount up to max_shift_sec seconds.
    """
    if random.random() > p_apply:
        return y
    shift = random.randint(-len(y), len(y))
    return np.roll(y, shift)

# -------------------------------------------------
# 8) SpecAugment
# -------------------------------------------------
def aug_spec(spec,
    p_apply=1.0,
    time_mask_max_frames=10,
    time_mask_count=5,
    freq_mask_max_bins=10,
    freq_mask_count=5,):
    
    if np.random.rand() > p_apply:
        return spec

    augmented = spec.copy()
    _, num_bins, num_frames = augmented.shape

    for _ in range(time_mask_count):
        t = np.random.randint(1, time_mask_max_frames + 1)
        t0 = np.random.randint(0, max(1, num_frames - t))
        augmented[:,:,t0:t0+t] = 0

    for _ in range(freq_mask_count):
        f = np.random.randint(1, freq_mask_max_bins + 1)
        f0 = np.random.randint(0, max(1, num_bins - f))
        augmented[:, f0:f0+f,:] = 0

    return augmented
