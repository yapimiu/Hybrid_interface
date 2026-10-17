from __future__ import annotations

import numpy as np
import pytest

from p300_analysis.constants import EPOCH_DURATION_MS
pytest.importorskip("pylsl")
from p300_analysis.epoch_geometry import EpochGeometry


def test_epoch_geometry_includes_pre_stim_baseline_in_template() -> None:
    geom = EpochGeometry()
    eeg_times = np.arange(200, dtype=np.float64) * 0.002

    geom.ensure_template(None, eeg_times.tolist(), baseline_ms=100)

    assert geom.dt_ms == pytest.approx(2.0)
    assert geom.epoch_len == int(round((EPOCH_DURATION_MS + 100) / 2.0)) + 1
    assert geom.time_ms_template is not None
    assert geom.time_ms_template[0] == pytest.approx(-100.0)
    assert geom.time_ms_template[-1] == pytest.approx(float(EPOCH_DURATION_MS))
    assert geom.baseline_ms == 100
