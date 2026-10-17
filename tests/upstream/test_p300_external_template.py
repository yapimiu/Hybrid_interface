"""P300 external template construction (Stage 2.1)."""

from __future__ import annotations

import numpy as np

from p300_analysis.erp_compute import build_averaged_erp, compute_corrected_and_integrated
from p300_analysis.epoch_geometry import EpochGeometry
from p300_analysis.online_engine import P300EngineParams, P300OnlineEngine
from p300_analysis.signal_processing import time_window_to_indices


def _setup_engine(*, n_epochs: int = 12, n_ch: int = 4) -> P300OnlineEngine:
    geom = EpochGeometry()
    fs_hz = 500.0
    dt_ms = 1000.0 / fs_hz
    n_time = 200
    eeg_times = (np.arange(n_time, dtype=np.float64) * (dt_ms / 1000.0)).tolist()
    geom.ensure_template(None, eeg_times, baseline_ms=100, fs_hz=fs_hz)
    assert geom.epoch_len is not None and geom.time_ms_template is not None

    eng = P300OnlineEngine()
    eng.params = P300EngineParams(
        baseline_ms=100,
        window_x_ms=550,
        window_y_ms=725,
        fs_hz=fs_hz,
        artifact_threshold_uv=0.0,
    )
    eng._epoch_geom = geom
    eng.current_cue_target_id = 4
    key = "стимул_4"
    epoch_len = int(geom.epoch_len)
    rng = np.random.default_rng(0)
    eng.epochs_data[key] = [
        rng.standard_normal((epoch_len, n_ch), dtype=np.float64) for _ in range(n_epochs)
    ]
    return eng


def _legacy_broken_template_length(eng: P300OnlineEngine) -> int:
    """Reproduce pre-fix path: mean stack then avg.T before compute_corrected_and_integrated."""
    key = "стимул_4"
    epochs = eng.epochs_data[key]
    el = int(eng._epoch_geom.epoch_len)
    time_ms = eng._epoch_geom.time_ms_template
    assert time_ms is not None
    stack = np.stack([np.asarray(e, dtype=np.float64) for e in epochs[:12]], axis=0)
    avg = np.mean(stack, axis=0)
    raw_averaged = np.expand_dims(avg.T, axis=0)
    corrected, _, _, wx, wy = compute_corrected_and_integrated(
        raw_averaged,
        time_ms,
        int(eng.params.baseline_ms),
        int(eng.params.window_x_ms),
        int(eng.params.window_y_ms),
    )
    xi0, xi1 = time_window_to_indices(time_ms, int(wx), int(wy))
    return int(np.asarray(corrected[0, xi0:xi1], dtype=np.float64).ravel().size)


def test_external_template_matches_compute_decision_pipeline() -> None:
    eng = _setup_engine()
    assert eng.try_build_external_template_from_epochs(min_epochs=12)
    tw = np.asarray(eng.external_template_window, dtype=np.float64).ravel()

    stim_keys, raw_averaged, _ = build_averaged_erp(
        {"стимул_4": eng.epochs_data["стимул_4"]},
        int(eng._epoch_geom.epoch_len),
    )
    corrected, _, _, wx, wy = compute_corrected_and_integrated(
        raw_averaged,
        eng._epoch_geom.time_ms_template,
        int(eng.params.baseline_ms),
        int(eng.params.window_x_ms),
        int(eng.params.window_y_ms),
    )
    idx = stim_keys.index("стимул_4")
    xi0, xi1 = time_window_to_indices(eng._epoch_geom.time_ms_template, int(wx), int(wy))
    expected = np.asarray(corrected[idx, xi0:xi1], dtype=np.float64).ravel()

    assert tw.size == expected.size
    assert tw.size > 1
    np.testing.assert_allclose(tw, expected)


def test_legacy_transpose_mismatched_template_length_for_multichannel() -> None:
    eng = _setup_engine(n_ch=4)
    broken_len = _legacy_broken_template_length(eng)
    assert eng.try_build_external_template_from_epochs(min_epochs=12)
    fixed_len = int(eng.external_template_window.size)
    assert broken_len != fixed_len
    assert fixed_len > 10
