"""P300 decision gating: safe minimum per class, use all accumulated epochs."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

if "pylsl" not in sys.modules:
    _pylsl = ModuleType("pylsl")
    _pylsl.local_clock = lambda: 0.0  # type: ignore[attr-defined]
    _pylsl.StreamInfo = MagicMock  # type: ignore[attr-defined]
    _pylsl.StreamInlet = MagicMock  # type: ignore[attr-defined]
    _pylsl.StreamOutlet = MagicMock  # type: ignore[attr-defined]
    _pylsl.resolve_byprop = MagicMock(return_value=[])  # type: ignore[attr-defined]
    _pylsl.resolve_streams = MagicMock(return_value=[])  # type: ignore[attr-defined]
    sys.modules["pylsl"] = _pylsl

from p300_analysis.constants import IDEAL_EPOCHS_PER_CLASS, SAFE_MIN_EPOCHS_TO_DECIDE
from p300_analysis.erp_compute import check_can_decide, stim_epoch_count_stats
from p300_analysis.online_engine import P300EngineParams, P300OnlineEngine
from p300_analysis.winner_selection import WINNER_MODE_AUC


def test_safe_min_is_five_ideal_twelve() -> None:
    assert SAFE_MIN_EPOCHS_TO_DECIDE == 5
    assert IDEAL_EPOCHS_PER_CLASS == 12


def test_check_can_decide_allows_eleven_when_one_class_short_of_twelve() -> None:
    """Session-like counts: min=11 must still allow decision (not all(c>=12))."""
    stim_keys = [f"стимул_{i}" for i in range(9)]
    epochs = {k: [np.zeros(10) for _ in range(12 if i % 3 else 11)] for i, k in enumerate(stim_keys)}
    can, min_n = check_can_decide(stim_keys, epochs)
    assert min_n == 11
    assert can is True


def test_check_can_decide_rejects_below_safe_min() -> None:
    stim_keys = ["стимул_1", "стимул_2"]
    epochs = {
        "стимул_1": [np.zeros(10) for _ in range(4)],
        "стимул_2": [np.zeros(10) for _ in range(6)],
    }
    can, min_n = check_can_decide(stim_keys, epochs)
    assert can is False
    assert min_n == 4


def test_stim_epoch_count_stats_totals() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    epochs = {
        "стимул_0": [np.zeros(10) for _ in range(11)],
        "стимул_1": [np.zeros(10) for _ in range(12)],
    }
    stats = stim_epoch_count_stats(stim_keys, epochs)
    assert stats["min_per_class"] == 11
    assert stats["max_per_class"] == 12
    assert stats["total_epochs"] == 23
    assert stats["epochs_used_for_decision"]["стимул_0"] == 11


def test_online_engine_decides_with_eleven_epochs_per_class() -> None:
    engine = P300OnlineEngine()
    engine.reset(params=P300EngineParams(fs_hz=250.0))
    engine.ingest_eeg_chunk(
        eeg_chunk=np.zeros((200, 2), dtype=np.float64),
        eeg_ts=[float(i) / 250.0 for i in range(200)],
        lsl_local_clock_now=1.0,
    )
    n = int(engine._epoch_geom.epoch_len or 200)
    for _ in range(11):
        engine.epochs_data.setdefault("стимул_1", []).append(np.ones((n, 1)))
        engine.epochs_data.setdefault("стимул_2", []).append(np.ones((n, 1)) * 0.1)
    dec = engine.compute_decision(winner_mode=WINNER_MODE_AUC)
    assert dec.can_decide is True
    assert dec.winner_key is not None
    dbg = dec.debug or {}
    assert dbg.get("min_per_class") == 11
    assert dbg.get("total_epochs") == 22
