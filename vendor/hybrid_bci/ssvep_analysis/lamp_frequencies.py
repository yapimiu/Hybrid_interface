"""Частоты ламп мигалки и дефолты MSI (без Qt — для protocol_runner_gui и ssvep_analyzer)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

MSI_DEFAULT_FS = 5000.0
MSI_DEFAULT_WINDOW_SEC = 3.0
MSI_DEFAULT_SAMPLES = 15000
LAMP1_DEFAULT_HZ = 1000.0 / 117.0  # ≈ 8.547008547 Гц
CHANNEL_CB_COLUMNS = 4

# First four Migalka lamps (1000/i Hz), shared by ssvep_analyzer and hybrid protocol.
DEFAULT_LAMP_FREQS_HZ: Tuple[float, ...] = (
    LAMP1_DEFAULT_HZ,
    1000.0 / 99.0,
    1000.0 / 87.0,
    1000.0 / 76.0,
)

# Canonical hybrid-BCI SSVEP montage: channels 9, 10, 21 (1-based) → 0-based.
CANONICAL_SSVEP_ROI_0IDX: Tuple[int, ...] = (8, 9, 20)
SSVEP_PROTOCOL_ROI_0IDX: Tuple[int, ...] = CANONICAL_SSVEP_ROI_0IDX

_LAMP_FREQ_CHOICES: List[Tuple[str, float]] = []


def lamp_frequency_choices() -> List[Tuple[str, float]]:
    """500 дискретных частот: 1000/i Гц, i = 1..500; подпись с десятичной точкой."""
    if not _LAMP_FREQ_CHOICES:
        for i in range(1, 501):
            v = 1000.0 / float(i)
            s = f"{v}".replace(",", ".")
            _LAMP_FREQ_CHOICES.append((s, v))
    return _LAMP_FREQ_CHOICES


def lamp_frequency_closest_index(target_hz: float) -> int:
    arr = np.array([v for _, v in lamp_frequency_choices()], dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(target_hz))))
