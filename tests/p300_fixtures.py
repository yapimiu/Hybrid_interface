"""Синтетические сигналы исключительно для автоматизированных проверок."""

import numpy as np

from app.p300 import P300Config, Stimulus
from app.p300.processing import Epoch


def config(**kwargs):
    return P300Config(**{"sfreq": 100.0, "channel_names": ("Cz", "Pz"), **kwargs})


def event(index=0, timestamp=4.0, target=True, block="b0", trial="t0", obj=None):
    return Stimulus(str(index), obj or ("A" if target else "B"), trial, block,
                    ("A", "B"), timestamp, timestamp - 0.25, 0.25, target)


def raw_signal(end=12.0, sfreq=100.0):
    times = np.arange(0, end, 1 / sfreq)
    signal = np.column_stack([3 * np.sin(2 * np.pi * 7 * times),
                              2 * np.sin(2 * np.pi * 9 * times + 0.3)])
    signal += 5 * np.exp(-((times - 4.35) / 0.1) ** 2)[:, None]
    return signal, times


def calibration_epochs():
    rng = np.random.default_rng(42)
    times = np.arange(-10, 41) / 50
    bump = np.exp(-((times - 0.35) / 0.09) ** 2)
    result = []
    for block in range(3):
        for repeat in range(6):
            for target in (False, True):
                index = len(result)
                timestamp = 10 + index * 4
                data = rng.normal(0, 1e-6, (2, len(times)))
                if target:
                    data += np.array([[6e-6], [9e-6]]) * bump
                marker = event(index, timestamp, target, block=f"b{block}", trial=f"t{repeat}")
                result.append(Epoch(data, times, marker, (timestamp - 1.2, timestamp + 1.8), 0.0))
    return result
