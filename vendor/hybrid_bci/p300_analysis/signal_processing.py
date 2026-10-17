"""Baseline correction, фильтрация и детекция плохих каналов для ERP."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def bandpass_filter(
    X: np.ndarray,
    fs: float,
    lo: float = 0.5,
    hi: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Полосовой фильтр Баттерворта (SOS, нулевой сдвиг фазы).

    Использует sosfiltfilt: численно устойчиво для широкого диапазона fs
    (например, 5000 Гц у NeuronSpectrum), в отличие от формы (b,a)+filtfilt,
    которая на 5 кГц + 0.5–20 Гц + order=4 даёт NaN/overflow.
    X: (n_samples,) или (n_samples, n_channels)
    fs: частота дискретизации в Гц
    Возвращает массив той же формы.
    Если длина сигнала слишком мала — возвращает X без изменений.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return X

    n = X.shape[0]
    min_len = 3 * (order + 1) * 2
    if n < min_len:
        return X

    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 1.0 - 1e-5)
    if lo_n >= hi_n:
        return X

    # SOS-форма численно устойчива при больших fs/order и узких полосах
    # (Butterworth(4, 0.5-20) на 5 kHz через (b,a)+filtfilt даёт NaN/overflow).
    sos = butter(order, [lo_n, hi_n], btype="band", output="sos")
    if X.ndim == 1:
        return sosfiltfilt(sos, X).astype(X.dtype)
    return sosfiltfilt(sos, X, axis=0).astype(X.dtype)


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """Common Average Reference (CAR): вычитает среднее по каналам из каждого отсчёта.

    X: (n_samples, n_channels)
    Убирает общий дрейф/шум, присутствующий на всех каналах одновременно
    (движение головы, дыхание, помехи от провода питания).
    Усиливает локальные сигналы (P300 на теменно-затылочных каналах).
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return X
    return X - X.mean(axis=1, keepdims=True)


def normalize_channels(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Нормализация по каналу: делит каждый канал на его std.

    X: (n_samples, n_channels)  — нормализует по каждому столбцу.
        (n_channels,)           — нормализует скаляром.
    Каналы с std < eps не изменяются (защита от плоской линии).
    """
    if X.ndim == 1:
        s = float(np.std(X))
        return X / s if s > eps else X.copy()
    stds = np.std(X, axis=0, keepdims=True)  # (1, n_ch)
    stds = np.where(stds < eps, 1.0, stds)
    return X / stds


def detect_bad_channels(
    X: np.ndarray,
    std_thresh: float = 4.0,
    abs_thresh: float = 3.0,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Обнаруживает каналы с аномальным шумом.

    X: (n_samples, n_channels)
    Возвращает:
      bad_indices  — список индексов плохих каналов (0-based)
      abs_means    — среднее |x| по каждому каналу
      stds         — std по каждому каналу
    Критерий: канал считается плохим, если его std > std_thresh * median(stds)
              ИЛИ abs_mean > abs_thresh * median(abs_means).
    """
    if X.ndim != 2 or X.shape[1] == 0:
        return [], np.array([]), np.array([])

    abs_means = np.mean(np.abs(X), axis=0)
    stds = np.std(X, axis=0)

    med_abs = float(np.median(abs_means))
    med_std = float(np.median(stds))

    bad_std = stds > std_thresh * med_std if med_std > 0 else np.zeros(X.shape[1], dtype=bool)
    bad_abs = abs_means > abs_thresh * med_abs if med_abs > 0 else np.zeros(X.shape[1], dtype=bool)
    bad_mask = bad_std | bad_abs

    bad_indices = [int(i) for i in np.where(bad_mask)[0]]
    return bad_indices, abs_means, stds


def baseline_correction(raw: np.ndarray, time_ms: np.ndarray, baseline_ms: int) -> np.ndarray:
    """Baseline correction: corrected = raw - median(raw[:baseline_idx]).

    Использует median вместо mean для устойчивости к артефактам в pre-stimulus периоде.
    raw: shape (..., n_time)
    """
    if raw.ndim < 1:
        raise ValueError("raw must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be a 1D array")
    if raw.shape[-1] != time_ms.shape[0]:
        raise ValueError("raw and time_ms length mismatch")

    baseline_start_idx = int(np.searchsorted(time_ms, -float(baseline_ms), side="left"))
    baseline_end_idx = int(np.searchsorted(time_ms, 0.0, side="left"))

    if baseline_end_idx > baseline_start_idx:
        baseline_slice = raw[..., baseline_start_idx:baseline_end_idx]
    else:
        dt_ms = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
        baseline_idx = int(round(float(baseline_ms) / dt_ms))
        baseline_idx = max(1, min(baseline_idx, time_ms.shape[0]))
        baseline_slice = raw[..., :baseline_idx]

    baseline_val = np.median(baseline_slice, axis=-1, keepdims=True)
    return raw - baseline_val


def time_window_to_indices(
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[int, int]:
    """Преобразует окно в мс в [start, end) индексы по реальной оси времени."""
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be 1D array")
    if time_ms.size == 0:
        raise ValueError("time_ms must not be empty")

    wx = float(window_x_ms)
    wy = float(window_y_ms)
    if wy <= wx:
        wy = wx + 1.0

    x_idx = int(np.searchsorted(time_ms, wx, side="left"))
    y_idx = int(np.searchsorted(time_ms, wy, side="right"))

    x_idx = max(0, min(x_idx, time_ms.shape[0] - 1))
    y_idx = max(x_idx + 1, min(y_idx, time_ms.shape[0]))
    return x_idx, y_idx


def integrated_cumsum(
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
) -> tuple:
    """Интеграция ERP по модулю: cumsum(abs(corrected[x_idx:y_idx]))."""
    if corrected.ndim < 1:
        raise ValueError("corrected must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be 1D array")
    if corrected.shape[-1] != time_ms.shape[0]:
        raise ValueError("corrected and time_ms length mismatch")

    x_idx, y_idx = time_window_to_indices(time_ms, window_x_ms, window_y_ms)
    segment = corrected[..., x_idx:y_idx]
    integrated = np.cumsum(np.abs(segment), axis=-1)
    time_crop = time_ms[x_idx:y_idx]
    return integrated, time_crop
