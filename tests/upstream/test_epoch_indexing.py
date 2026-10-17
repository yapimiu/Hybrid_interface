"""Тесты привязки маркеров к эпохам (epoch_indexing).

Без импорта EpochGeometry/pylsl — тот же алгоритм, что в epoch_geometry.compute_start_index.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from p300_analysis.epoch_indexing import (
    eeg_timestamps_sufficient_for_fallback,
    resolve_epoch_indices_for_marker,
)

DT_MS = 4.0
EPOCH_LEN = 50


def _compute_start_index_like_geom(time_arr: np.ndarray, t_eff: float) -> Optional[int]:
    """Копия логики EpochGeometry.compute_start_index (без pylsl)."""
    if DT_MS <= 0:
        return None
    n = int(time_arr.shape[0])
    el = int(EPOCH_LEN)
    if n < el:
        return None
    dt_s = DT_MS / 1000.0
    t0 = float(time_arr[0])
    i_nom = int(np.round((float(t_eff) - t0) / dt_s))
    i_nom = max(0, min(i_nom, n - el))
    return i_nom


@pytest.fixture
def cs() -> object:
    return lambda ta, te: _compute_start_index_like_geom(ta, te)


def test_marker_newer_than_lsl_ref_waits(cs) -> None:
    """Маркер заметно «впереди» ref: индекс уезжает за хвост буфера — ждём (как на main)."""
    buf_len = 500
    ta = np.linspace(0.0, (buf_len - 1) / 250.0, buf_len, dtype=np.float64)
    s, e, wait = resolve_epoch_indices_for_marker(
        marker_ts=100.0,
        buf_len=buf_len,
        srate=250.0,
        epoch_len=50,
        lsl_ref=99.0,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    assert (s, e) == (None, None)
    assert wait is True


def test_marker_slightly_ahead_of_ref_waits_if_epoch_beyond_buffer(cs) -> None:
    """Небольшой mt > ref: эпоха ещё не влезает в буфер — wait_more."""
    ref = 10.0
    mt = ref + 0.0025
    buf_len = 400
    ta = np.linspace(0.0, (buf_len - 1) / 250.0, buf_len, dtype=np.float64)
    _, _, wait = resolve_epoch_indices_for_marker(
        marker_ts=mt,
        buf_len=buf_len,
        srate=250.0,
        epoch_len=50,
        lsl_ref=ref,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    assert wait is True


def test_direct_indexing_distinct_starts_for_close_markers(cs) -> None:
    """Две близкие вспышки (~20 мс) дают разные start_idx при нормальном ref."""
    buf_len = 800
    srate = 250.0
    el = 50
    lsl_ref = 5.0
    ta = np.linspace(0.0, (buf_len - 1) / srate, buf_len, dtype=np.float64)
    m1, m2 = 4.50, 4.52
    a1, b1, w1 = resolve_epoch_indices_for_marker(
        marker_ts=m1,
        buf_len=buf_len,
        srate=srate,
        epoch_len=el,
        lsl_ref=lsl_ref,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    a2, b2, w2 = resolve_epoch_indices_for_marker(
        marker_ts=m2,
        buf_len=buf_len,
        srate=srate,
        epoch_len=el,
        lsl_ref=lsl_ref,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    assert not w1 and not w2
    assert a1 is not None and a2 is not None
    assert abs((a2 or 0) - (a1 or 0)) >= 4  # ~20 мс * 250 ≈ 5 отсчётов


def test_pre_event_offset_moves_epoch_start_earlier(cs) -> None:
    buf_len = 800
    srate = 250.0
    el = 50
    lsl_ref = 5.0
    marker_ts = 4.5
    ta = np.linspace(0.0, (buf_len - 1) / srate, buf_len, dtype=np.float64)

    s0, e0, w0 = resolve_epoch_indices_for_marker(
        marker_ts=marker_ts,
        buf_len=buf_len,
        srate=srate,
        epoch_len=el,
        lsl_ref=lsl_ref,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    s1, e1, w1 = resolve_epoch_indices_for_marker(
        marker_ts=marker_ts,
        buf_len=buf_len,
        srate=srate,
        epoch_len=el,
        lsl_ref=lsl_ref,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
        pre_event_s=0.1,
    )

    assert not w0 and not w1
    assert s0 is not None and s1 is not None and e0 is not None and e1 is not None
    assert s1 == s0 - 25
    assert e1 == e0 - 25


def test_coarse_eeg_timestamps_disable_fallback() -> None:
    """Грубые метки (мало уникальных значений) — fallback отключён."""
    buf_len = 400
    ta = np.zeros(buf_len, dtype=np.float64)
    ta[200:] = 1.0
    assert eeg_timestamps_sufficient_for_fallback(ta, buf_len=buf_len) is False


def test_fine_eeg_timestamps_enable_fallback() -> None:
    ta = np.linspace(0.0, 1.0, 500, dtype=np.float64)
    assert eeg_timestamps_sufficient_for_fallback(ta, buf_len=500) is True


def test_coarse_timestamps_no_false_positive_same_window_two_markers(cs) -> None:
    """При грубых time_arr два разных маркера не должны оба получить валидный fallback
    с одним и тем же окном из-за одинакового округления (типичный баг 1 блок)."""
    buf_len = 400
    el = 50
    srate = 250.0
    lsl_ref = 3.0
    ta = np.zeros(buf_len, dtype=np.float64)
    ta[:] = 1.0

    results = []
    for mt in (1.05, 1.08, 1.11, 1.14):
        s, e, w = resolve_epoch_indices_for_marker(
            marker_ts=mt,
            buf_len=buf_len,
            srate=srate,
            epoch_len=el,
            lsl_ref=lsl_ref,
            time_arr=ta,
            marker_eeg_offset=None,
            compute_start_index=cs,
        )
        results.append((s, e, w))

    valid = [r for r in results if r[0] is not None and r[1] is not None and not r[2]]
    if len(valid) >= 2:
        starts = [v[0] for v in valid]
        assert len(set(starts)) == len(starts), "одинаковый start_idx для разных маркеров — регрессия"


def test_marker_eeg_offset_aligns_domains(cs) -> None:
    """Разные абсолютные шкалы marker_ts и конца буфера ref — offset переводит маркер в ref."""
    buf_len = 500
    srate = 250.0
    el = 50
    ref = 1000.0
    mt_raw = 500.0
    off = 499.8  # t_mark = 999.8 → 0.2 с до ref → ~50 отсчётов от хвоста
    ta = np.linspace(ref - (buf_len - 1) / srate, ref, buf_len, dtype=np.float64)
    s, e, w = resolve_epoch_indices_for_marker(
        marker_ts=mt_raw,
        buf_len=buf_len,
        srate=srate,
        epoch_len=el,
        lsl_ref=ref,
        time_arr=ta,
        marker_eeg_offset=off,
        compute_start_index=cs,
    )
    assert not w
    assert s is not None and e is not None
    assert int(e) - int(s) == el


def test_end_beyond_buffer_waits(cs) -> None:
    """Эпоха ещё не помещается в хвост буфера — wait_more (без грубого fallback)."""
    buf_len = 55
    el = 50
    ta = np.ones(buf_len, dtype=np.float64)
    s, e, wait = resolve_epoch_indices_for_marker(
        marker_ts=9.98,
        buf_len=buf_len,
        srate=250.0,
        epoch_len=el,
        lsl_ref=10.0,
        time_arr=ta,
        marker_eeg_offset=None,
        compute_start_index=cs,
    )
    assert wait is True
    assert s is None
