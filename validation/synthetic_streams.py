"""Deterministic synthetic LSL inlets for live ProtocolRunner validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MarkerEvent:
    lsl_time: float
    value: str


class SyntheticEEGInlet:
    """Minimal pylsl.StreamInlet stand-in for ProtocolRunner._pull_lsl."""

    def __init__(
        self,
        *,
        fs_hz: float = 250.0,
        n_channels: int = 21,
        seed: int = 0,
        ssvep_hz: Optional[float] = None,
        ssvep_roi_0idx: Sequence[int] = (9, 10, 20),
        noise_uv: float = 2.0,
        ssvep_amp_uv: float = 25.0,
    ) -> None:
        self._fs = float(fs_hz)
        self._n_ch = int(n_channels)
        self._rng = np.random.default_rng(int(seed))
        self._ssvep_hz = float(ssvep_hz) if ssvep_hz is not None else None
        self._roi = tuple(int(c) for c in ssvep_roi_0idx)
        self._noise = float(noise_uv) * 1e-6
        self._ssvep_amp = float(ssvep_amp_uv) * 1e-6
        self._lsl_t = 0.0
        self._closed = False

    def close_stream(self) -> None:
        self._closed = True

    def pull_chunk(
        self, timeout: float = 0.0, max_samples: int = 32
    ) -> Tuple[List[List[float]], List[float]]:
        if self._closed:
            return [], []
        n = max(1, min(int(max_samples), 64))
        t0 = self._lsl_t
        ts = [t0 + i / self._fs for i in range(n)]
        data = self._rng.standard_normal((n, self._n_ch)) * self._noise
        if self._ssvep_hz is not None:
            phase = 2.0 * np.pi * self._ssvep_hz * (np.arange(n) / self._fs + t0)
            wave = np.sin(phase) * self._ssvep_amp
            for ch in self._roi:
                if 0 <= ch < self._n_ch:
                    data[:, ch] += wave
        self._lsl_t = ts[-1] + 1.0 / self._fs
        return data.tolist(), ts


class SyntheticMarkerInlet:
    """Releases queued markers when pull_chunk is called (LSL time order)."""

    def __init__(self, events: Sequence[MarkerEvent] | None = None) -> None:
        self._events: List[MarkerEvent] = sorted(events or [], key=lambda e: e.lsl_time)
        self._idx = 0
        self._closed = False

    def arm(self, events: Sequence[MarkerEvent]) -> None:
        self._events = sorted(events, key=lambda e: e.lsl_time)
        self._idx = 0

    def close_stream(self) -> None:
        self._closed = True

    def pull_chunk(
        self, timeout: float = 0.0, max_samples: int = 32
    ) -> Tuple[List[str], List[float]]:
        if self._closed or self._idx >= len(self._events):
            return [], []
        batch = self._events[self._idx : self._idx + max(1, int(max_samples))]
        self._idx += len(batch)
        return [e.value for e in batch], [e.lsl_time for e in batch]


def p300_trial_markers(
    *,
    target_tile: int,
    t_start: float,
    n_flashes: int = 6,
    flash_gap_s: float = 0.35,
) -> List[MarkerEvent]:
    """Synthetic P300 flash sequence for one trial."""
    events: List[MarkerEvent] = []
    t = float(t_start)
    events.append(MarkerEvent(t, f"-1|trial_start|target={int(target_tile)}"))
    t += 0.05
    for i in range(int(n_flashes)):
        tid = int(target_tile) if i % 2 == 0 else (int(target_tile) + 1) % 9
        events.append(MarkerEvent(t, f"{tid}|on"))
        t += 0.05
        events.append(MarkerEvent(t, f"{tid}|off"))
        t += float(flash_gap_s)
    events.append(MarkerEvent(t + 0.2, "-2|trial_end"))
    return events


def erp_like_epochs(
    *,
    n_channels: int,
    n_samples: int,
    target_ch: int,
    amp_uv: float = 40.0,
    seed: int = 0,
) -> np.ndarray:
    """Single epoch (n_samples, n_channels) with Gaussian ERP bump on target."""
    rng = np.random.default_rng(seed)
    ep = rng.standard_normal((n_samples, n_channels)) * 2e-6
    center = n_samples // 3
    width = max(4, n_samples // 10)
    bump = np.exp(-0.5 * ((np.arange(n_samples) - center) / width) ** 2)
    ep[:, target_ch] += bump * float(amp_uv) * 1e-6
    return ep
