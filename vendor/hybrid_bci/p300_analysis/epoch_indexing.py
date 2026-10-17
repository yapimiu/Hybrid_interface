"""Привязка маркера LSL к индексам эпохи в буфере ЭЭГ (без Qt, для тестов)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# Доля уникальных меток времени на хвосте буфера; ниже — не используем fallback
# (иначе при грубом шаге 1 с несколько вспышек получают один start_idx).
FALLBACK_MIN_UNIQUE_TS_FRACTION = 0.12

FALLBACK_LOOKBACK_SAMPLES = 2000


def eeg_timestamps_sufficient_for_fallback(
    time_arr: np.ndarray, *, buf_len: int, lookback: int = FALLBACK_LOOKBACK_SAMPLES
) -> bool:
    """True, если по хвосту time_arr видно «достаточно уникальных» меток для безопасного fallback."""
    if time_arr.ndim != 1 or time_arr.size != buf_len or buf_len == 0:
        return False
    n = int(min(lookback, buf_len))
    sl = time_arr[-n:]
    frac = float(np.unique(sl).size) / float(sl.size)
    return frac >= FALLBACK_MIN_UNIQUE_TS_FRACTION


def resolve_epoch_indices_for_marker(
    *,
    marker_ts: float,
    buf_len: int,
    srate: float,
    epoch_len: int,
    lsl_ref: float,
    time_arr: np.ndarray,
    marker_eeg_offset: Optional[float],
    compute_start_index: Callable[[np.ndarray, float], Optional[int]],
    pre_event_s: float = 0.0,
) -> Tuple[Optional[int], Optional[int], bool]:
    """Возвращает (start_idx, end_idx, wait_more).

    ``lsl_ref`` — время **последнего отсчёта ЭЭГ** в той же шкале, что и ``marker_ts`` после
    калибровки (передавайте ``float(eeg_times[-1])``). Не ``pylsl.local_clock()`` на приёмнике,
    если штампы ЭЭГ (Neurospectrum и т.п.) в другой оси, чем local_clock.

    ``marker_eeg_offset`` — сдвиг маркера в шкалу времени ЭЭГ (см. time_alignment_calibrated в GUI).

    wait_more=True — нужно дождаться ещё данных в буфере (эпоха ещё не помещается).
    (None, None, False) — маркер нельзя надёжно извлечь (отбросить).
    """
    mt_raw = float(marker_ts)
    t_mark = (
        mt_raw + float(marker_eeg_offset) if marker_eeg_offset is not None else mt_raw
    )
    t_start = t_mark - max(0.0, float(pre_event_s))
    ref = float(lsl_ref)

    seconds_back = ref - t_start
    start_idx = int(round(buf_len - 1 - seconds_back * srate))
    end_idx = int(start_idx + epoch_len)

    if 0 <= start_idx and end_idx <= buf_len:
        return start_idx, end_idx, False

    direct_needs_wait = end_idx > buf_len
    start_past_buffer = start_idx < 0

    ta = np.asarray(time_arr, dtype=np.float64).reshape(-1)
    # Если t_mark всё ещё «новее» ref — только прямой индекс; fallback по time_arr
    # с сырым mt_raw даст ложный хвост, если маркер вне шкалы ЭЭГ.
    use_fallback = (t_start <= ref) and eeg_timestamps_sufficient_for_fallback(ta, buf_len=buf_len)

    if use_fallback:
        candidates: list[float] = [t_start]
        for t_eff in candidates:
            fb_start = compute_start_index(ta, t_eff)
            if fb_start is None:
                continue
            fb_end = int(fb_start) + int(epoch_len)
            if 0 <= fb_start and fb_end <= buf_len:
                return int(fb_start), fb_end, False
            if fb_end > buf_len:
                return None, None, True

    if direct_needs_wait:
        return None, None, True

    # Маркер «в прошлом» относительно буфера или отказались от ненадёжного fallback.
    if start_past_buffer:
        return None, None, False

    return None, None, False
