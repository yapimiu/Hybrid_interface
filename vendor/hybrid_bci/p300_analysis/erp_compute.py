"""Усреднение эпох и подготовка данных для отображения победителя."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.constants import SAFE_MIN_EPOCHS_TO_DECIDE
from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit
from p300_analysis.signal_processing import (
    baseline_correction,
    integrated_cumsum,
    normalize_channels,
    time_window_to_indices,
)
from p300_analysis.winner_selection import (
    WINNER_MODE_AUC,
    WINNER_MODE_SIGNED_MEAN,
    WINNER_MODE_TEMPLATE_CORR,
)


def artifact_reject_epochs(
    epochs: List[np.ndarray],
    threshold_uv: float,
) -> Tuple[List[np.ndarray], int]:
    """Отбрасывает эпохи, в которых амплитуда превышает порог.

    epochs: список массивов (epoch_len,) или (epoch_len, n_ch).
    np.max(np.abs) работает для обоих вариантов.
    """
    if threshold_uv <= 0:
        return epochs, 0
    clean: List[np.ndarray] = []
    rejected = 0
    for ep in epochs:
        if np.max(np.abs(ep)) <= threshold_uv:
            clean.append(ep)
        else:
            rejected += 1
    return clean, rejected


def build_averaged_erp(
    epochs_data: Dict[str, List[np.ndarray]],
    epoch_len: int,
    artifact_threshold_uv: Optional[float] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """Усредняет эпохи по каждому стимулу.

    Поддерживает два формата эпох:
    - 1D (epoch_len,)           — legacy / single-channel
    - 2D (epoch_len, n_ch)      — per-channel (новый формат)

    Для 2D каждая эпоха нормализуется по каналу (÷ std), затем каналы
    усредняются → выравнивает влияние шумных каналов.

    Возвращает (stim_keys, raw_averaged, rejected_counts),
    где raw_averaged.shape = (n_stim, epoch_len) — совместимо с downstream.
    """
    stim_keys = [k for k, v in epochs_data.items() if v]
    stim_keys.sort(key=stim_key_sort_key)
    n_stim = len(stim_keys)
    raw_averaged = np.zeros((n_stim, epoch_len), dtype=np.float64)
    rejected_counts: Dict[str, int] = {}

    for i, key in enumerate(stim_keys):
        epochs = epochs_data.get(key, [])
        if not epochs:
            continue
        if artifact_threshold_uv is not None and artifact_threshold_uv > 0:
            epochs, n_rej = artifact_reject_epochs(epochs, artifact_threshold_uv)
            rejected_counts[key] = n_rej
        if not epochs:
            rejected_counts[key] = rejected_counts.get(key, 0)
            continue

        if epochs[0].ndim == 2:
            # Per-channel path: normalize each epoch by channel std, then average
            normed = [normalize_channels(ep[:epoch_len]) for ep in epochs]
            stack = np.stack(normed, axis=0)          # (n_ep, epoch_len, n_ch)
            mean_ch_erp = np.mean(stack, axis=0)      # (epoch_len, n_ch)
            raw_averaged[i, :] = np.mean(mean_ch_erp, axis=-1)  # (epoch_len,)
        else:
            stack = np.stack([e[:epoch_len] for e in epochs], axis=0)
            raw_averaged[i, :] = np.mean(stack, axis=0)

    return stim_keys, raw_averaged, rejected_counts


def compute_corrected_and_integrated(
    raw_averaged: np.ndarray,
    time_ms: np.ndarray,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    wy = window_y_ms if window_y_ms > window_x_ms else window_x_ms + 1
    corrected = baseline_correction(raw_averaged, time_ms, baseline_ms=baseline_ms)
    integrated, time_crop = integrated_cumsum(
        corrected,
        time_ms,
        window_x_ms=window_x_ms,
        window_y_ms=wy,
    )
    return corrected, integrated, time_crop, window_x_ms, wy


def stim_epoch_count_stats(
    stim_keys: List[str],
    epochs_data: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Per-class epoch counts at decision time (all accumulated epochs are used downstream)."""
    counts = {k: len(epochs_data.get(k, [])) for k in stim_keys}
    vals = list(counts.values())
    if not vals:
        return {
            "min_per_class": 0,
            "max_per_class": 0,
            "total_epochs": 0,
            "epochs_used_for_decision": {},
            "epoch_count_at_decision": {},
        }
    return {
        "min_per_class": int(min(vals)),
        "max_per_class": int(max(vals)),
        "total_epochs": int(sum(vals)),
        "epochs_used_for_decision": dict(counts),
        "epoch_count_at_decision": dict(counts),
    }


def check_can_decide(stim_keys: List[str], epochs_data: Dict[str, List[np.ndarray]]) -> Tuple[bool, int]:
    """Allow decision when every active class has at least SAFE_MIN epochs (use all stored epochs)."""
    if not stim_keys:
        return False, 0
    stats = stim_epoch_count_stats(stim_keys, epochs_data)
    min_n = int(stats["min_per_class"])
    can = min_n >= int(SAFE_MIN_EPOCHS_TO_DECIDE)
    return can, min_n


def compute_winner_metrics(
    stim_keys: List[str],
    raw_averaged: np.ndarray,
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
    winner_mode: str = WINNER_MODE_AUC,
    *,
    template_window: Optional[np.ndarray] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    """Возвращает winner_idx, mode_used, data для debug_ndjson (winner_compare).

    Дополнительно вычисляет margin = (top1 - top2) / top1 для индикатора уверенности.
    """
    dt_m = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    xi0, xi1 = time_window_to_indices(time_ms, window_x_ms, window_y_ms)
    corr_win = corrected[:, xi0:xi1]
    abs_auc_values = np.sum(np.abs(corr_win), axis=1) * dt_m
    signed_mean_values = np.mean(corr_win, axis=1) if corr_win.size else np.zeros(len(stim_keys))
    positive_peak_values = np.max(corr_win, axis=1) if corr_win.size else np.zeros(len(stim_keys))

    template_corr_values = None
    if winner_mode == WINNER_MODE_TEMPLATE_CORR:
        # Template correlation in the decision window. If template is missing or degenerate,
        # gracefully fall back to AUC mode.
        tw = None if template_window is None else np.asarray(template_window, dtype=np.float64).ravel()
        if tw is None or tw.size != corr_win.shape[1]:
            final_metric_values = abs_auc_values
            mode_used = WINNER_MODE_AUC
        else:
            t = tw.copy()
            t = t - float(np.mean(t)) if t.size else t
            t_std = float(np.std(t, ddof=1)) if t.size > 1 else 0.0
            if not np.isfinite(t_std) or t_std < 1e-12:
                final_metric_values = abs_auc_values
                mode_used = WINNER_MODE_AUC
            else:
                t /= t_std
                # Per-stimulus Pearson correlation with z-scored template
                template_corr_values = np.zeros(len(stim_keys), dtype=np.float64)
                for i in range(len(stim_keys)):
                    x = np.asarray(corr_win[i], dtype=np.float64).ravel()
                    x = x - float(np.mean(x)) if x.size else x
                    x_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
                    if not np.isfinite(x_std) or x_std < 1e-12:
                        template_corr_values[i] = 0.0
                        continue
                    x /= x_std
                    denom = float(max(1, x.size - 1))
                    template_corr_values[i] = float(np.sum(x * t) / denom) if x.size else 0.0
                final_metric_values = template_corr_values
                mode_used = WINNER_MODE_TEMPLATE_CORR
    elif winner_mode == WINNER_MODE_SIGNED_MEAN:
        final_metric_values = signed_mean_values
        mode_used = WINNER_MODE_SIGNED_MEAN
    else:
        final_metric_values = abs_auc_values
        mode_used = WINNER_MODE_AUC

    winner_idx = int(np.argmax(final_metric_values))
    auc_winner_idx = int(np.argmax(abs_auc_values))

    # Margin: уверенность решения (0..1). 0 — невозможно выбрать, 1 — явный лидер.
    sorted_vals = np.sort(final_metric_values)[::-1]
    top1 = float(sorted_vals[0]) if sorted_vals.size > 0 else 0.0
    top2 = float(sorted_vals[1]) if sorted_vals.size > 1 else 0.0
    margin = (top1 - top2) / abs(top1) if abs(top1) > 1e-9 else 0.0

    abs_max_values = np.max(np.abs(corr_win), axis=1) if corr_win.size else np.zeros(len(stim_keys))
    debug_payload = {
        "winner_rule": mode_used,
        "chosen_winner_idx": winner_idx,
        "chosen_winner_key": stim_keys[winner_idx],
        "stim_keys": stim_keys,
        "final_metric_values": [float(x) for x in final_metric_values],
        "signed_mean_final": [float(x) for x in signed_mean_values],
        "abs_auc_values": [float(x) for x in abs_auc_values],
        "template_corr_values": (
            [float(x) for x in template_corr_values] if template_corr_values is not None else None
        ),
        "auc_winner_idx": auc_winner_idx,
        "auc_winner_key": stim_keys[auc_winner_idx],
        "positive_peak_values": [float(x) for x in positive_peak_values],
        "margin": margin,
        "window_index": [xi0, xi1],
        "dt_ms": float(dt_m),
        "window_ms": [window_x_ms, window_y_ms],
        "window_time_ms_actual": [float(time_ms[xi0]), float(time_ms[xi1 - 1])],
        "corr_abs_max": [float(x) for x in abs_max_values],
        "corr_mean_in_window": [float(x) for x in signed_mean_values],
        "corr_window_shape": [int(corr_win.shape[0]), int(corr_win.shape[1])],
    }
    return winner_idx, mode_used, debug_payload


def winner_display_lines(
    winner_key: str,
    mode_short: str,
    lsl_cue_target_id: Optional[int],
    margin: Optional[float] = None,
) -> Tuple[List[str], int, bool]:
    win_digit = stim_key_to_tile_digit(winner_key)
    lines = ["РЕЗУЛЬТАТ:", f"ПЛИТКА {win_digit}", f"режим: {mode_short}"]
    if margin is not None:
        pct = int(round(margin * 100))
        confidence = "высокая" if pct >= 30 else ("средняя" if pct >= 12 else "низкая ⚠")
        lines.append(f"уверенность: {pct}% ({confidence})")
    if lsl_cue_target_id is not None:
        lines.append(f"цель LSL: {lsl_cue_target_id}")
    match_lsl = lsl_cue_target_id is None or win_digit == lsl_cue_target_id
    return lines, win_digit, match_lsl
