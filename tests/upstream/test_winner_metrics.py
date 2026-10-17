from __future__ import annotations

import numpy as np

from p300_analysis.erp_compute import (
    artifact_reject_epochs,
    build_averaged_erp,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.signal_processing import (
    baseline_correction,
    common_average_reference,
    integrated_cumsum,
    normalize_channels,
    time_window_to_indices,
)
from p300_analysis.winner_selection import (
    WINNER_MODE_AUC,
    WINNER_MODE_SIGNED_MEAN,
    WINNER_MODE_TEMPLATE_CORR,
)


def test_auc_mode_uses_absolute_integral_and_matches_integrated_plot() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    raw = np.array(
        [
            [0.0, 3.0, -8.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw,
        time_ms,
        baseline_ms=100,
        window_x_ms=100,
        window_y_ms=300,
    )
    winner_idx, mode_used, dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged=raw,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=wx,
        window_y_ms=wy,
        winner_mode=WINNER_MODE_AUC,
    )

    assert mode_used == WINNER_MODE_AUC
    assert winner_idx == 0
    np.testing.assert_allclose(dbg["abs_auc_values"], integrated[:, -1] * 100.0)
    assert dbg["chosen_winner_key"] == "стимул_0"


def test_signed_mean_mode_remains_available_for_offline_comparison() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    corrected = np.array(
        [
            [0.0, 12.0, -10.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    winner_idx, mode_used, _ = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=0,
        window_y_ms=300,
        winner_mode=WINNER_MODE_SIGNED_MEAN,
    )

    assert mode_used == WINNER_MODE_SIGNED_MEAN
    assert winner_idx == 1


def test_template_corr_mode_prefers_shape_match_in_window() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    # Two candidates have similar energy, but only stim_1 matches the template shape.
    corrected = np.array(
        [
            [0.0, 1.0, -1.0, 0.0],   # anti-shape
            [0.0, 1.0, 1.0, 0.0],    # matches
        ],
        dtype=np.float64,
    )
    template_window = np.array([1.0, 1.0, 0.0], dtype=np.float64)  # indices 1..4 (100,200,300)

    winner_idx, mode_used, dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=100,
        window_y_ms=300,
        winner_mode=WINNER_MODE_TEMPLATE_CORR,
        template_window=template_window,
    )

    assert mode_used == WINNER_MODE_TEMPLATE_CORR
    assert winner_idx == 1
    assert dbg["template_corr_values"] is not None


def test_build_averaged_erp_per_channel_noisy_channel_does_not_dominate() -> None:
    """Шумный канал с амплитудой в 10x не должен доминировать после нормализации.

    stim_0 содержит реальный P300-пик на тихом канале.
    stim_1 содержит случайный выброс на шумном канале, но тихие каналы плоские.
    Без нормализации stim_1 выиграет (шум > сигнал).
    С нормализацией (÷ std) stim_0 должен выиграть.
    """
    rng = np.random.default_rng(42)
    epoch_len = 50

    # stim_0: тихий канал ch0 имеет пик +5, шумный ch1 ~0
    ep0_ch0 = np.zeros(epoch_len); ep0_ch0[25] = 5.0          # P300-пик
    ep0_ch1 = rng.normal(0, 0.1, epoch_len)                    # почти 0
    ep0 = np.column_stack([ep0_ch0, ep0_ch1])                  # (50, 2)

    # stim_1: тихий канал ch0 ~0, шумный ch1 имеет большой случайный шум
    ep1_ch0 = rng.normal(0, 0.1, epoch_len)
    ep1_ch1 = rng.normal(0, 50.0, epoch_len)                   # шум 50x больше

    ep1 = np.column_stack([ep1_ch0, ep1_ch1])

    epochs_data = {
        "стимул_0": [ep0.copy() for _ in range(5)],
        "стимул_1": [ep1.copy() for _ in range(5)],
    }
    stim_keys, raw_averaged, _ = build_averaged_erp(epochs_data, epoch_len)

    # После нормализации stim_0 должен иметь бо́льший сигнал в пиковой зоне
    assert stim_keys == ["стимул_0", "стимул_1"]
    peak_0 = float(np.max(raw_averaged[0]))
    peak_1 = float(np.max(np.abs(raw_averaged[1])))
    assert peak_0 > peak_1, (
        f"stim_0 peak {peak_0:.3f} должен быть больше stim_1 |peak| {peak_1:.3f} "
        "после нормализации каналов"
    )


def test_normalize_channels_equalizes_amplitudes() -> None:
    """normalize_channels делит каждый столбец на его std."""
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    Xn = normalize_channels(X)
    # После нормализации std каждого столбца ≈ 1
    np.testing.assert_allclose(np.std(Xn, axis=0), [1.0, 1.0], atol=1e-9)


def test_common_average_reference_removes_common_drift() -> None:
    """CAR вычитает среднее по каналам из каждого отсчёта."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1.0, (100, 4))
    drift = rng.normal(0, 10.0, (100, 1))  # общий дрейф
    X = signal + drift
    X_car = common_average_reference(X)
    # После CAR строковое среднее должно быть ≈ 0
    np.testing.assert_allclose(X_car.mean(axis=1), np.zeros(100), atol=1e-10)
    # Форма сохраняется
    assert X_car.shape == X.shape


def test_artifact_reject_epochs_2d() -> None:
    """artifact_reject_epochs работает с 2D эпохами (epoch_len, n_ch)."""
    good = np.ones((50, 3)) * 10.0
    bad = np.ones((50, 3)) * 10.0
    bad[25, 1] = 200.0  # выброс на одном канале

    clean, n_rej = artifact_reject_epochs([good, bad], threshold_uv=150.0)
    assert n_rej == 1
    assert len(clean) == 1
    assert np.array_equal(clean[0], good)


def test_time_window_to_indices_respects_negative_time_axis() -> None:
    time_ms = np.array([-100.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    x_idx, y_idx = time_window_to_indices(time_ms, 200, 300)
    assert (x_idx, y_idx) == (3, 5)


def test_integrated_cumsum_uses_actual_time_values_with_baseline_offset() -> None:
    time_ms = np.array([-100.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    corrected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)

    integrated, time_crop = integrated_cumsum(corrected, time_ms, window_x_ms=200, window_y_ms=300)

    np.testing.assert_allclose(time_crop, np.array([200.0, 300.0]))
    np.testing.assert_allclose(integrated, np.array([[4.0, 9.0]]))


def test_baseline_correction_uses_negative_time_window_when_available() -> None:
    time_ms = np.array([-100.0, 0.0, 100.0, 200.0], dtype=np.float64)
    raw = np.array([[5.0, 100.0, 100.0, 100.0]], dtype=np.float64)

    corrected = baseline_correction(raw, time_ms, baseline_ms=100)

    np.testing.assert_allclose(corrected, np.array([[0.0, 95.0, 95.0, 95.0]]))


def test_winner_metrics_uses_actual_time_values_with_baseline_offset() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([-100.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    corrected = np.array(
        [
            [0.0, 0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    winner_idx, _, dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=200,
        window_y_ms=300,
        winner_mode=WINNER_MODE_AUC,
    )

    assert winner_idx == 1
    np.testing.assert_allclose(dbg["abs_auc_values"], [0.0, 600.0])
    assert dbg["window_index"] == [3, 5]
    assert dbg["window_time_ms_actual"] == [200.0, 300.0]
