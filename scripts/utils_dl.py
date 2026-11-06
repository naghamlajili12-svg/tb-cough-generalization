"""
scripts/utils_dl.py

Shared deep-learning utilities for the DL pipeline (04) and the Zambia device
generalization study (06):

  * model definitions: VGGish (optionally with MixStyle), ResNet-18/34, classifier head;
  * pretrained-weight loading by shape matching;
  * training loop with class-weighting, mixup, label smoothing, LR scheduling, early stop;
  * subject-level nested-CV helpers;
  * mel-spectrogram / spectrogram feature extraction and audio padding.

Ported from the authors' training scripts with light cleanup; numeric behaviour is
preserved.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models, transforms

# --- audio defaults ---------------------------------------------------------
SR = 16_000
WIN = 0.025
HOP = 0.010
AUDIO_DURATION = 0.96
N_MEL = 64

MAX_BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

@torch.no_grad()
def safe_forward(model, X, input_adapter=None, max_batch_size: int = MAX_BATCH_SIZE):
    """Run forward in chunks to avoid OOM.

    `X` may be a numpy array or CPU tensor; each chunk is moved to the device and
    passed through `input_adapter` (channel-repeat / transpose / resize / normalise)
    just before the model, so the full model-ready tensor is never materialised.
    """
    outputs = []
    for i in range(0, len(X), max_batch_size):
        xb = X[i:i + max_batch_size]
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(np.asarray(xb), dtype=torch.float32)
        xb = xb.to(DEVICE).float()
        if input_adapter is not None:
            xb = input_adapter(xb)
        outputs.append(model(xb))
    return torch.cat(outputs, dim=0)


def load_partial_weights(model, checkpoint_path, verbose: bool = True):
    """Load weights from a checkpoint by matching tensor shapes (order-preserving)."""
    pre_state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in pre_state.items():
        for k1 in model_state:
            if v.shape == model_state[k1].shape:
                model_state[k1] = v.clone()
                loaded.append((k, k1))
                break
        else:
            skipped.append(k)
    model.load_state_dict(model_state, strict=False)
    if verbose:
        print(f"  partial weights: loaded {len(loaded)}, skipped {len(skipped)}")
    return model


class MixStyle(nn.Module):
    """Mix instance-wise feature statistics for domain generalization (Zhou et al.)."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p, self.alpha = p, alpha

    def forward(self, x):
        if not self.training or torch.rand(1) > self.p:
            return x
        N, C, H, W = x.size()
        xr = x.view(N, C, -1)
        mean = xr.mean(dim=2)
        std = torch.sqrt(xr.var(dim=2, unbiased=False) + 1e-6)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((N,)).to(x.device).view(N, 1)
        idx = torch.randperm(N)
        mixed_mean = lam * mean + (1 - lam) * mean[idx]
        mixed_std = lam * std + (1 - lam) * std[idx]
        x_norm = (xr - mean.view(N, C, 1)) / std.view(N, C, 1)
        return (x_norm * mixed_std.view(N, C, 1) + mixed_mean.view(N, C, 1)).view(N, C, H, W)


class VGGish(nn.Module):
    def __init__(self, if_mixstyle: bool = False, mixstyle_p: float = 0.5, mixstyle_alpha: float = 0.1):
        super().__init__()
        ms = lambda: MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        block = []

        def conv(cin, cout):
            return [nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True)]

        if not if_mixstyle:
            block = [
                *conv(1, 64), nn.MaxPool2d(2, 2),
                *conv(64, 128), nn.MaxPool2d(2, 2),
                *conv(128, 256), *conv(256, 256), nn.MaxPool2d(2, 2),
                *conv(256, 512), *conv(512, 512), nn.MaxPool2d(2, 2),
            ]
        else:
            block = [
                *conv(1, 64), ms(), nn.MaxPool2d(2, 2),
                *conv(64, 128), ms(), nn.MaxPool2d(2, 2),
                *conv(128, 256), ms(), *conv(256, 256), ms(), nn.MaxPool2d(2, 2),
                *conv(256, 512), ms(), *conv(512, 512), ms(), nn.MaxPool2d(2, 2),
            ]
        self.features = nn.Sequential(*block)
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        return self.fc(x.view(x.size(0), -1))


class AddLayer(nn.Module):
    """Classification head producing a single logit."""

    def __init__(self, hidden_nodes: int | None = 64, dropout_rate: float = 0.3):
        super().__init__()
        if hidden_nodes is None:
            self.classifier = nn.Linear(128, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128, hidden_nodes), nn.ReLU(),
                nn.Dropout(dropout_rate), nn.Linear(hidden_nodes, 1),
            )

    def forward(self, x):
        return self.classifier(x)


def build_model(model_name, hidden_nodes=64, dropout_rate=0.3, if_mixstyle=False,
                mixstyle_p=0.5, mixstyle_alpha=0.1, vggish_weights=None):
    if model_name in ("Resnet18", "Resnet34"):
        ctor = models.resnet18 if model_name == "Resnet18" else models.resnet34
        weights = models.ResNet18_Weights.DEFAULT if model_name == "Resnet18" else models.ResNet34_Weights.DEFAULT
        backbone = ctor(weights=weights, progress=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, 128)
    elif model_name == "VGGish16":
        backbone = VGGish(if_mixstyle=if_mixstyle, mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_alpha)
        if vggish_weights is not None:
            backbone = load_partial_weights(backbone, vggish_weights)
        else: # put an error
            raise ValueError("VGGish weights must be provided for VGGish model.")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return nn.Sequential(backbone, AddLayer(hidden_nodes=hidden_nodes, dropout_rate=dropout_rate))


# =============================================================================
# TRAINING
# =============================================================================

def mixup(x, y, alpha=0.1):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def smooth_labels(y, eps=0.05):
    return y * (1 - eps) + 0.5 * eps


def train_from_arrays(model, X_train, y_train, X_val, y_val, *,
                      input_adapter=None,
                      epochs=40, batch_size=32, lr_backbone=1e-4, lr_head=1e-5,
                      weight_decay=0, if_mixup=False, if_label_smoothing=False,
                      patience=5, mixup_alpha=0.1, verbose=False):
    """Train a binary classifier from arrays (BCE w/ pos-weight, early stopping on val AUC).

    `X_train`/`X_val` stay as (1-channel) numpy arrays; `input_adapter` adapts each
    mini-batch to the model's expected input on the device, so the heavy
    channel-repeated / resized tensor is only ever built one batch at a time.
    """
    model = model.to(DEVICE)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    y_val = np.asarray(y_val)

    head_lr_differs = lr_backbone != lr_head
    opt_cls = torch.optim.AdamW if weight_decay else torch.optim.Adam
    if head_lr_differs:
        groups = [
            {"params": model[0].parameters(), "lr": lr_backbone},
            {"params": model[1].parameters(), "lr": lr_head},
        ]
        optimizer = opt_cls(groups, weight_decay=weight_decay) if weight_decay else opt_cls(groups)
    else:
        optimizer = opt_cls(model.parameters(), lr=lr_backbone, weight_decay=weight_decay) \
            if weight_decay else opt_cls(model.parameters(), lr=lr_backbone)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val, best_state, no_improve = float("inf"), None, 0
    n_train = len(X_train)
    n_batches = max(n_train // batch_size, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train).numpy()
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.as_tensor(X_train[idx], dtype=torch.float32).to(DEVICE)
            if input_adapter is not None:
                xb = input_adapter(xb)
            yb = y_train[idx].to(DEVICE)
            if if_mixup:
                xb, yb = mixup(xb, yb, alpha=mixup_alpha)
            if if_label_smoothing:
                yb = smooth_labels(yb)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds_val = safe_forward(model, X_val, input_adapter).squeeze(1)
            val_auc = roc_auc_score(y_val, torch.sigmoid(preds_val).cpu().numpy())
        val_loss = -val_auc
        scheduler.step(val_loss)
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs} | train {total_loss / n_batches:.4f} | val AUC {val_auc:.4f}")

        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                model.load_state_dict(best_state)
                break
    return model


@torch.no_grad()
def model_eval_roc(model, X, y, input_adapter=None):
    """Return (AUC, predicted probabilities) for inputs X with labels y."""
    model.eval()
    logits = safe_forward(model, X, input_adapter).squeeze(1)
    preds = torch.sigmoid(logits).cpu().numpy()
    return roc_auc_score(np.array(y), preds), preds


@torch.no_grad()
def extract_vggish_embedding(model, X, input_adapter=None, max_batch_size=MAX_BATCH_SIZE):
    """Return the 128-d backbone embedding for each input (per-batch adaptation)."""
    backbone = model[0].eval()
    outputs = []
    for i in range(0, len(X), max_batch_size):
        xb = X[i:i + max_batch_size]
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(np.asarray(xb), dtype=torch.float32)
        xb = xb.to(DEVICE).float()
        if input_adapter is not None:
            xb = input_adapter(xb)
        outputs.append(backbone(xb))
    return torch.cat(outputs, dim=0).cpu().numpy()


# =============================================================================
# NESTED-CV HELPERS (subject-level)
# =============================================================================

def random_stratified_kfold(X, y, group, n_splits=10, random_state=None):
    """Return coarse sample-index groups for subject-level stratified k folds."""
    rs = random_state if random_state is not None else np.random.randint(0, 10000)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    subject_groups = [list(np.unique(group[val_idx])) for _, val_idx in gkf.split(X, y, group)]
    coarse = [[] for _ in range(n_splits)]
    for i in range(len(y)):
        for g in range(n_splits):
            if group[i] in subject_groups[g]:
                coarse[g].append(i)
                break
    return coarse


def prep_X_y(X, y, coarse_indices, test_fold):
    """Split into train/test given coarse fold indices; also return inner CV index lists."""
    X_train = y_train = X_test = y_test = None
    sizes = []
    for g, idx in enumerate(coarse_indices):
        idx = np.array(idx)
        if g == test_fold:
            X_test, y_test = X[idx], y[idx]
        elif X_train is None:
            X_train, y_train = X[idx], y[idx]
            sizes.append(len(idx))
        else:
            X_train = np.concatenate([X_train, X[idx]], axis=0)
            y_train = np.concatenate([y_train, y[idx]])
            sizes.append(len(idx))

    cumulative = np.cumsum([0] + sizes)
    cv_train_idx, cv_val_idx = [], []
    for i in range(len(sizes)):
        val = np.arange(cumulative[i], cumulative[i + 1])
        cv_val_idx.append(val)
        cv_train_idx.append(np.setdiff1d(np.arange(sum(sizes)), val))
    return X_train, y_train, X_test, y_test, cv_train_idx, cv_val_idx


# =============================================================================
# AUDIO -> SPECTROGRAM
# =============================================================================

def pad_audio(waveform, sr=SR, audio_duration=AUDIO_DURATION, pad_by="repeating"):
    n = int(audio_duration * sr)
    if len(waveform) < n:
        if pad_by == "zeros":
            return np.pad(waveform, (0, n - len(waveform)))
        if pad_by == "repeating":
            return np.tile(waveform, int(np.ceil(n / len(waveform))))[:n]
        raise ValueError("pad_by must be 'zeros' or 'repeating'")
    if len(waveform) == n:
        return waveform
    start = np.random.randint(0, len(waveform) - n + 1)
    return waveform[start:start + n]


def prep_mel_spectrogram(y, fmin=10, fmax=8000, sr=SR, n_mels=N_MEL, win=WIN, hop=HOP):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=int(win * sr), hop_length=int(hop * sr),
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel + 1e-2, ref=np.max)
    return np.expand_dims(mel_db, axis=0)  # [1, n_mels, time]


def prep_spectrogram(y, fmin=10, fmax=8000, sr=SR, win=WIN, hop=HOP):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=int(win * sr), hop_length=int(hop * sr), window="hann")) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=int(win * sr))
    idx = np.where((freqs >= (fmin or 0)) & (freqs <= (fmax or freqs[-1])))[0]
    return np.expand_dims(S_db[idx, :], axis=0)  # [1, freq_bins, time]


# =============================================================================
# PER-BATCH INPUT ADAPTERS
#
# Applied to a mini-batch [B, 1, H, W] on the device, just before the model, so the
# (channel-repeated / resized) model-ready tensor is never built for the whole set.
# =============================================================================

_RESIZE_VGGISH = transforms.Resize((96, 64))
_RESIZE_RESNET = transforms.Resize((112, 112))
_NORM_RESNET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def adapt_vggish(xb):
    """[B,1,mel,time] -> [B,1,time,mel] resized to 96x64 (VGGish single channel)."""
    return _RESIZE_VGGISH(xb.transpose(2, 3))


def adapt_resnet(xb):
    """[B,1,H,W] -> 3-channel, ImageNet-normalised, resized to 112x112 (ResNet)."""
    return _RESIZE_RESNET(_NORM_RESNET(xb.repeat(1, 3, 1, 1)))


def get_input_adapter(model_name: str):
    """Return the per-batch input adapter for a backbone."""
    return adapt_resnet if model_name in ("Resnet18", "Resnet34") else adapt_vggish
