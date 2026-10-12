"""Gap filling and zero-phase filters on NaN-gapped time series."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt


def runs(mask: np.ndarray):
    """[(start, stop)] of consecutive True runs in a 1-D bool array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int | None):
    """Shape-preserving (PCHIP) fill of interior gaps up to max_gap frames.

    x [T,...] with valid [T]; None fills every interior gap. Edges are not
    extrapolated. Returns (filled x, bool [T] of frames that were filled).
    """
    x = x.copy()
    valid = np.asarray(valid, bool) & np.isfinite(x.reshape(len(x), -1)).all(1)
    filled = np.zeros(len(x), bool)
    idx = np.nonzero(valid)[0]
    if len(idx) < 2:
        return x, filled
    interp = PchipInterpolator(idx, x[idx], axis=0)
    for a, b in runs(~valid):
        if a == 0 or b == len(x) or (max_gap is not None and b - a > max_gap):
            continue
        x[a:b] = interp(np.arange(a, b))
        filled[a:b] = True
    return x, filled


def hold_edges(x: np.ndarray):
    """Repeat the first/last finite sample over leading/trailing NaNs."""
    x = x.copy()
    ok = np.isfinite(x.reshape(len(x), -1)).all(1)
    idx = np.nonzero(ok)[0]
    if len(idx):
        x[: idx[0]] = x[idx[0]]
        x[idx[-1] + 1:] = x[idx[-1]]
    return x


def _filtfilt_segments(x: np.ndarray, sos) -> np.ndarray:
    out = x.copy()
    ok = np.isfinite(x)
    for a, b in runs(ok):
        n = b - a
        if n < 7:
            continue
        out[a:b] = sosfiltfilt(sos, x[a:b], padlen=min(3 * 2 * len(sos), n - 1))
    return out


def lowpass(x: np.ndarray, fps: float, cutoff: float, order: int = 2) -> np.ndarray:
    """Zero-phase (lag-free) Butterworth low-pass along axis 0, per finite segment."""
    if cutoff >= fps / 2:
        return x.copy()
    sos = butter(order, cutoff, "low", fs=fps, output="sos")
    flat = x.reshape(len(x), -1)
    out = np.stack([_filtfilt_segments(flat[:, k], sos) for k in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def bandpass(x: np.ndarray, fps: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    hi = min(hi, 0.45 * fps)
    sos = butter(order, [lo, hi], "band", fs=fps, output="sos")
    flat = x.reshape(len(x), -1)
    out = np.stack([_filtfilt_segments(flat[:, k], sos) for k in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def spikes(track: np.ndarray, valid: np.ndarray, min_jump: float, k_mad: float = 6.0) -> np.ndarray:
    """Single-sample outliers in a [T,D] track: far from the local median."""
    T = len(track)
    out = np.zeros(T, bool)
    idx = np.nonzero(valid & np.isfinite(track).all(1))[0]
    if len(idx) < 5:
        return out
    v = track[idx]
    med = median_filter(v, size=(5, 1), mode="nearest")
    res = np.linalg.norm(v - med, axis=1)
    mad = np.median(np.abs(res - np.median(res))) + 1e-9
    out[idx[res > max(min_jump, k_mad * 1.4826 * mad)]] = True
    return out


def mode_filter(labels: np.ndarray, size: int) -> np.ndarray:
    """Majority vote over a sliding window (for per-frame activity labels)."""
    labels = np.asarray(labels)
    vals = np.unique(labels)
    h = size // 2
    pad = np.pad(labels, h, mode="edge")
    counts = np.stack([np.convolve(pad == v, np.ones(size), "valid") for v in vals])
    return vals[counts.argmax(0)]
