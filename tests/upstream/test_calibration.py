from __future__ import annotations

import numpy as np

from p300_analysis.calibration import CalibrationExample, evaluate_configuration, search_best_configuration


def _make_example(expected: int, amp: float) -> CalibrationExample:
    time_ms = np.arange(-100.0, 801.0, 100.0, dtype=np.float64)
    epoch_target = np.zeros((time_ms.size, 2), dtype=np.float64)
    epoch_other = np.zeros((time_ms.size, 2), dtype=np.float64)

    late_mask = (time_ms >= 600.0) & (time_ms <= 800.0)
    early_mask = (time_ms >= 200.0) & (time_ms <= 300.0)

    # Channel 0 carries the useful late positive response.
    target_idx = expected
    other_idx = 1 - expected
    if target_idx == 0:
        epoch_target[late_mask, 0] = amp
    else:
        epoch_other[late_mask, 0] = amp

    # Channel 1 is misleading earlier noise and should be excluded by calibration.
    if target_idx == 0:
        epoch_other[early_mask, 1] = amp * 1.2
    else:
        epoch_target[early_mask, 1] = amp * 1.2

    return CalibrationExample(
        file=f"run_target_{expected}.csv",
        path=f"/tmp/run_target_{expected}.csv",
        expected=expected,
        epochs_data={
            "стимул_0": (epoch_target.copy(),),
            "стимул_1": (epoch_other.copy(),),
        },
        time_ms=time_ms,
        channel_names=("ch_1", "ch_2"),
        fs_hz=10.0,
        artifact_uv=60.0,
    )


def test_evaluate_configuration_uses_selected_channels() -> None:
    examples = [_make_example(0, 10.0), _make_example(1, 12.0)]

    bad = evaluate_configuration(
        examples,
        baseline_ms=100,
        window_x_ms=200,
        window_y_ms=300,
        channels_0idx=(1,),
    )
    good = evaluate_configuration(
        examples,
        baseline_ms=100,
        window_x_ms=600,
        window_y_ms=800,
        channels_0idx=(0,),
    )

    assert bad.correct == 0
    assert good.correct == 2
    assert good.accuracy_pct == 100.0


def test_evaluate_configuration_rejects_artifacts_after_roi_selection() -> None:
    time_ms = np.arange(-100.0, 801.0, 100.0, dtype=np.float64)
    target_epoch = np.zeros((time_ms.size, 2), dtype=np.float64)
    other_epoch = np.zeros((time_ms.size, 2), dtype=np.float64)

    target_epoch[(time_ms >= 600.0) & (time_ms <= 800.0), 0] = 8.0
    other_epoch[(time_ms >= 600.0) & (time_ms <= 800.0), 0] = 1.0
    target_epoch[5, 1] = 200.0

    example = CalibrationExample(
        file="artifact.csv",
        path="/tmp/artifact.csv",
        expected=0,
        epochs_data={
            "стимул_0": (target_epoch,),
            "стимул_1": (other_epoch,),
        },
        time_ms=time_ms,
        channel_names=("ch_1", "ch_2"),
        fs_hz=10.0,
        artifact_uv=60.0,
    )

    result = evaluate_configuration(
        [example],
        baseline_ms=100,
        window_x_ms=600,
        window_y_ms=800,
        channels_0idx=(0,),
    )

    assert result.correct == 1
    assert result.predictions[0].predicted == 0


def test_search_best_configuration_prefers_late_informative_channel() -> None:
    examples = [_make_example(0, 10.0), _make_example(1, 12.0), _make_example(0, 11.0)]

    results = search_best_configuration(
        examples,
        baseline_ms=100,
        x_values=[200, 600],
        y_values=[300, 800],
        max_subset_size=2,
        top_k=3,
    )

    assert results
    best = results[0]
    assert best.correct == 3
    assert best.window_x_ms == 600
    assert best.window_y_ms == 800
    assert best.channels_0idx == (0,)
