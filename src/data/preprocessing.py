from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt


def bandpass(signal: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    clean = np.asarray(signal, dtype=float)
    if not np.isfinite(clean).all():
        clean = np.nan_to_num(clean, nan=np.nanmedian(clean), posinf=0.0, neginf=0.0)
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, clean)


def zscore_from_train(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def maybe_resample(windows: np.ndarray, length: int | None) -> np.ndarray:
    if length is None or windows.shape[1] == length:
        return windows
    return resample(windows, length, axis=1)

