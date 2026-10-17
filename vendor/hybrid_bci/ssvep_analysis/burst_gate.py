"""
Прослойка пакетного SSVEP: по LSL-маркерам Migalka решает, когда вызывать MSI.

Постоянный режим: classify_allowed всегда True (если буфер полон).
Пакетный: MSI только если в окне анализа достаточно времени с активной стимуляцией.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_LED_RE = re.compile(
    r"^LED\s+(\d+)\s+(\d+)\s+(ON|OFF)\s*$",
    re.IGNORECASE,
)

_BURST_DONE_RE = re.compile(
    r"^BURST\s+DONE\s+wall=(\d+)(.*)$",
    re.IGNORECASE,
)
_BURST_LAMP_RE = re.compile(r"L(\d+)=(\d+)", re.IGNORECASE)
_BURST_TOTAL_RE = re.compile(r"total=(\d+)", re.IGNORECASE)
_BURST_TARGET_GLOBAL_RE = re.compile(r"target=(\d+)", re.IGNORECASE)
_BURST_PAUSE_RE = re.compile(r"pause=(\d+)", re.IGNORECASE)


def parse_burst_done_line(line: str) -> Optional[Dict[str, object]]:
    """``BURST DONE wall=8000 total=6001 target=6000 L0=2000 ... pause=1200``."""
    m = _BURST_DONE_RE.match(line.strip())
    if not m:
        return None
    tail = m.group(2) or ""
    per_lamp: Dict[int, int] = {}
    for lm in _BURST_LAMP_RE.finditer(tail):
        per_lamp[int(lm.group(1))] = int(lm.group(2))
    pause_m = _BURST_PAUSE_RE.search(tail)
    total_m = _BURST_TOTAL_RE.search(tail)
    target_m = _BURST_TARGET_GLOBAL_RE.search(tail)
    wall_ms = int(m.group(1))
    on_vals = [v for v in per_lamp.values() if v > 0]
    total_on = int(total_m.group(1)) if total_m else int(sum(on_vals))
    global_target = int(target_m.group(1)) if target_m else None
    return {
        "wall_ms": wall_ms,
        "pause_ms": int(pause_m.group(1)) if pause_m else 0,
        "global_target_on_ms": global_target,
        "total_on_ms": total_on,
        "accumulated_on_ms_by_lamp_0idx": per_lamp,
        "on_ms_min": min(on_vals) if on_vals else 0,
        "on_ms_max": max(on_vals) if on_vals else 0,
        "on_ms_spread": (max(on_vals) - min(on_vals)) if len(on_vals) > 1 else 0,
        "stim_utilization": (float(total_on) / float(global_target))
        if global_target and global_target > 0
        else None,
        "duty_cycle": (float(total_on) / wall_ms) if wall_ms > 0 else None,
    }


def parse_burst_target_ack(line: str) -> Optional[int]:
    """``BURST TARGET 6000`` → global ON budget (ms), shared across lamps."""
    s = line.strip()
    if not s.upper().startswith("BURST TARGET"):
        return None
    parts = s.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def parse_led_serial_line(line: str) -> Optional[Tuple[int, bool]]:
    """Строка прошивки: ``LED 2 12345 ON`` → (lamp_index, is_on)."""
    m = _LED_RE.match(line.strip())
    if not m:
        return None
    lamp = int(m.group(1))
    is_on = m.group(3).upper() == "ON"
    return lamp, is_on


def parse_lsl_marker(value: object) -> Optional[Tuple[int, bool]]:
    """``102|on`` / ``102|off`` → (lamp_index, is_on)."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    s = value.strip()
    if "|" not in s:
        return None
    left, right = s.split("|", 1)
    try:
        raw_id = int(left.strip())
    except ValueError:
        return None
    if raw_id < 100:
        return None
    lamp = raw_id - 100
    if not (0 <= lamp <= 8):
        return None
    ev = right.strip().lower()
    if ev == "on":
        return lamp, True
    if ev == "off":
        return lamp, False
    return None


@dataclass
class BurstGateConfig:
    window_sec: float = 2.0
    min_on_fraction: float = 0.55
    """Доля окна (по времени), когда хотя бы одна из active_lamps в ON."""
    min_on_sec: float = 1.0
    """Минимум секунд суммарного ON в окне (для вспышки ~2 с)."""
    grace_sec: float = 2.0
    """После старта subblock — смягчённые пороги (первые classify-окна)."""
    min_on_fraction_grace: float = 0.45
    min_on_sec_grace: float = 0.8


@dataclass
class _Interval:
    lamp: int
    t_on: float
    t_off: Optional[float] = None


@dataclass
class BurstGate:
    config: BurstGateConfig = field(default_factory=BurstGateConfig)
    active_lamps: Tuple[int, ...] = ()
    _intervals: List[_Interval] = field(default_factory=list)
    _open: Dict[int, _Interval] = field(default_factory=dict)
    _max_intervals: int = 512
    _block_start_t: Optional[float] = None

    def set_active_lamps(self, n_lamps: int) -> None:
        self.active_lamps = tuple(range(max(0, int(n_lamps))))

    def set_block_start(self, t: float) -> None:
        """Начало пакетного subblock (для grace period гейта)."""
        self._block_start_t = float(t)

    def clear_block_start(self) -> None:
        self._block_start_t = None

    def ingest_marker(self, lsl_time: float, value: object) -> None:
        parsed = parse_lsl_marker(value)
        if parsed is None:
            return
        lamp, is_on = parsed
        self._apply_lamp_state(lamp, is_on, lsl_time)

    def ingest_led_line(self, line: str, lsl_time: float) -> None:
        parsed = parse_led_serial_line(line)
        if parsed is None:
            return
        lamp, is_on = parsed
        self._apply_lamp_state(lamp, is_on, lsl_time)

    def _apply_lamp_state(self, lamp: int, is_on: bool, t: float) -> None:
        t = float(t)
        if is_on:
            if lamp in self._open:
                return
            iv = _Interval(lamp=lamp, t_on=t)
            self._open[lamp] = iv
            self._intervals.append(iv)
        else:
            iv = self._open.pop(lamp, None)
            if iv is not None:
                iv.t_off = t
        if len(self._intervals) > self._max_intervals:
            drop = len(self._intervals) - self._max_intervals
            self._intervals = self._intervals[drop:]
            self._open = {iv.lamp: iv for iv in self._intervals if iv.t_off is None}

    def _lamp_on_at(self, lamp: int, t: float) -> bool:
        if lamp not in self.active_lamps:
            return False
        for iv in self._intervals:
            if iv.lamp != lamp:
                continue
            t_end = iv.t_off if iv.t_off is not None else float("inf")
            if iv.t_on <= t <= t_end:
                return True
        return False

    def prune_intervals_before(self, t_min: float) -> int:
        """Удалить закрытые интервалы старше t_min (снижает «хвосты» чужих ламп)."""
        t_min = float(t_min)
        before = len(self._intervals)
        kept: List[_Interval] = []
        for iv in self._intervals:
            t_end = iv.t_off if iv.t_off is not None else float("inf")
            if iv.t_off is None or t_end >= t_min:
                kept.append(iv)
        self._intervals = kept
        self._open = {iv.lamp: iv for iv in self._intervals if iv.t_off is None}
        return before - len(self._intervals)

    def window_diagnostics(
        self,
        buf_times: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> Dict[str, object]:
        """Снимок окна для отладки пакетного режима."""
        if buf_times.size == 0:
            return {"error": "empty_buf_times"}
        t_end = float(now if now is not None else buf_times[-1])
        t_start = t_end - self.config.window_sec
        mask = (buf_times >= t_start) & (buf_times <= t_end)
        times = buf_times[mask]
        if times.size == 0:
            return {"t_start": t_start, "t_end": t_end, "n_samples": 0}

        per_lamp: Dict[int, float] = {}
        for lamp in self.active_lamps:
            on_n = sum(1 for t in times if self._lamp_on_at(lamp, float(t)))
            per_lamp[int(lamp)] = on_n / float(times.size)

        on_at_end = [
            int(lamp)
            for lamp in self.active_lamps
            if self._lamp_on_at(lamp, t_end)
        ]
        open_iv = [
            {"lamp": iv.lamp, "t_on": iv.t_on, "t_off": iv.t_off}
            for iv in self._intervals
            if iv.t_off is None
        ]
        return {
            "t_start": t_start,
            "t_end": t_end,
            "n_samples": int(times.size),
            "dt_sec": float(times[-1] - times[0]) if times.size > 1 else 0.0,
            "on_fraction_by_lamp_0idx": per_lamp,
            "lamps_on_at_end_0idx": on_at_end,
            "n_lamps_on_at_end": len(on_at_end),
            "open_intervals": open_iv,
            "n_intervals_total": len(self._intervals),
        }

    def _gate_thresholds(self, t_end: float) -> Tuple[float, float, bool]:
        min_frac = float(self.config.min_on_fraction)
        min_sec = float(self.config.min_on_sec)
        grace = False
        if self._block_start_t is not None:
            elapsed = float(t_end) - float(self._block_start_t)
            if elapsed < float(self.config.grace_sec):
                grace = True
                min_frac = min(min_frac, float(self.config.min_on_fraction_grace))
                min_sec = min(min_sec, float(self.config.min_on_sec_grace))
        return min_frac, min_sec, grace

    def on_sample_mask(self, buf_times: np.ndarray) -> np.ndarray:
        """True там, где хотя бы одна active_lamp в ON (длина = buf_times)."""
        if buf_times.size == 0 or not self.active_lamps:
            return np.zeros(buf_times.size, dtype=bool)
        out = np.zeros(int(buf_times.size), dtype=bool)
        for i, t in enumerate(buf_times):
            if any(self._lamp_on_at(lamp, float(t)) for lamp in self.active_lamps):
                out[i] = True
        return out

    def classify_allowed(
        self,
        buf_times: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        allowed, reason, _detail = self.evaluate_gate(buf_times, now=now)
        return allowed, reason

    def evaluate_gate(
        self,
        buf_times: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, object]]:
        """
        buf_times: метки LSL для каждого сэмпла в rolling-буфере (длина = EEG).
        Возвращает (allowed, reason, diagnostics) с on_fraction и gate_reason.
        """
        if buf_times.size == 0:
            detail = {
                "allowed": False,
                "gate_reason": "нет меток времени EEG",
                "on_fraction": 0.0,
                "on_sec": 0.0,
            }
            return False, detail["gate_reason"], detail

        t_end = float(now if now is not None else buf_times[-1])
        self.prune_intervals_before(t_end - self.config.window_sec * 4.0)
        t_start = t_end - self.config.window_sec
        mask = (buf_times >= t_start) & (buf_times <= t_end)
        times = buf_times[mask]
        if times.size < 8:
            detail = {
                "allowed": False,
                "gate_reason": "мало сэмплов в окне",
                "on_fraction": 0.0,
                "on_sec": 0.0,
                "t_start": t_start,
                "t_end": t_end,
            }
            return False, detail["gate_reason"], detail

        if not self.active_lamps:
            detail = {
                "allowed": False,
                "gate_reason": "не заданы активные лампы",
                "on_fraction": 0.0,
                "on_sec": 0.0,
            }
            return False, detail["gate_reason"], detail

        on_samples = 0
        for t in times:
            if any(self._lamp_on_at(lamp, float(t)) for lamp in self.active_lamps):
                on_samples += 1

        frac = on_samples / float(times.size)
        dt = float(times[-1] - times[0]) if times.size > 1 else 0.0
        on_sec = frac * max(dt, self.config.window_sec * 0.99)
        min_frac, min_sec, grace = self._gate_thresholds(t_end)

        if frac < min_frac:
            reason = f"стимул {frac:.0%} окна (нужно ≥{min_frac:.0%})"
            detail = {
                "allowed": False,
                "gate_reason": reason,
                "on_fraction": round(frac, 4),
                "on_sec": round(on_sec, 4),
                "min_on_fraction": min_frac,
                "min_on_sec": min_sec,
                "grace_active": grace,
                "t_start": t_start,
                "t_end": t_end,
            }
            return False, reason, detail
        if on_sec < min_sec:
            reason = f"ON {on_sec:.2f} с (нужно ≥{min_sec:.1f} с)"
            detail = {
                "allowed": False,
                "gate_reason": reason,
                "on_fraction": round(frac, 4),
                "on_sec": round(on_sec, 4),
                "min_on_fraction": min_frac,
                "min_on_sec": min_sec,
                "grace_active": grace,
                "t_start": t_start,
                "t_end": t_end,
            }
            return False, reason, detail

        reason = f"стимул {frac:.0%} окна, ON≈{on_sec:.2f} с"
        detail = {
            "allowed": True,
            "gate_reason": reason,
            "on_fraction": round(frac, 4),
            "on_sec": round(on_sec, 4),
            "min_on_fraction": min_frac,
            "min_on_sec": min_sec,
            "grace_active": grace,
            "t_start": t_start,
            "t_end": t_end,
        }
        return True, reason, detail

    def intervals_in_range(
        self,
        t_min: float,
        t_max: float,
    ) -> List[Tuple[int, float, float]]:
        """Интервалы ON для диаграммы Ганта: (lamp_index, t_start, t_end)."""
        if t_max <= t_min:
            return []
        out: List[Tuple[int, float, float]] = []
        for iv in self._intervals:
            if iv.lamp not in self.active_lamps:
                continue
            t_end = iv.t_off if iv.t_off is not None else t_max
            t_start = max(float(iv.t_on), t_min)
            t_end_clip = min(float(t_end), t_max)
            if t_start < t_end_clip:
                out.append((iv.lamp, t_start, t_end_clip))
        return out


def build_on_only_msi_window(
    buf: np.ndarray,
    buf_times: np.ndarray,
    gate: BurstGate,
    n_samples: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Собрать окно MSI (n_samples, n_ch) только из сэмплов, где active_lamps в ON.

    Идёт от конца буфера назад; если ON-сэмплов меньше n_samples — возвращает
    все доступные ON (вызывающий код должен отклонить classify).
    """
    n_samples = int(n_samples)
    arr = np.asarray(buf, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    bt = np.asarray(buf_times, dtype=np.float64)
    meta: Dict[str, object] = {
        "msi_window_mode": "on_only",
        "n_samples_requested": n_samples,
        "n_samples_returned": 0,
        "on_fraction_in_buffer": 0.0,
        "sufficient_on_samples": False,
    }
    if arr.shape[0] != bt.size or n_samples <= 0:
        meta["error"] = "shape_mismatch_or_bad_n"
        return np.zeros((0, arr.shape[1]), dtype=np.float64), meta

    on_mask = gate.on_sample_mask(bt)
    on_idx = np.flatnonzero(on_mask)
    meta["on_fraction_in_buffer"] = round(float(on_mask.sum()) / float(bt.size), 4) if bt.size else 0.0
    if on_idx.size == 0:
        meta["error"] = "no_on_samples"
        return np.zeros((0, arr.shape[1]), dtype=np.float64), meta

    pick = on_idx[-n_samples:] if on_idx.size >= n_samples else on_idx
    win = np.ascontiguousarray(arr[pick, :], dtype=np.float64)
    meta["n_samples_returned"] = int(win.shape[0])
    meta["sufficient_on_samples"] = bool(win.shape[0] >= n_samples)
    if win.shape[0] >= 2:
        meta["win_t_start"] = float(bt[pick[0]])
        meta["win_t_end"] = float(bt[pick[-1]])
        meta["win_duration_sec"] = float(bt[pick[-1]] - bt[pick[0]])
    return win, meta


def append_chunk_timestamps(
    prev_t: np.ndarray,
    chunk_ts: Sequence[float],
    n_new: int,
    fs: float,
    *,
    last_t: Optional[float] = None,
) -> np.ndarray:
    """Build per-sample LSL times for one new EEG chunk (length exactly ``n_new``).

    Contract:
    - Returns only timestamps for the new rows; callers append to the rolling buffer.
    - ``last_t`` (or last element of ``prev_t``) is the previous buffer tail for continuity.
    - With a single inlet timestamp (LSL anchor / 1 Hz quantisation), the grid is uniform
      with step ``1/fs``, strictly increasing after ``last_t``, and ending near ``chunk_ts[0]``
      when that is newer than the natural continuation.
    - Does not prepend ``prev_t`` (legacy prepend caused duplicate boundary samples and
      ``buf_t`` longer than ``buf`` in ``SSVEPOnlineEngine``).
    """
    if n_new <= 0:
        return np.zeros(0, dtype=np.float64)

    step = 1.0 / max(float(fs), 1.0)
    anchor = float(last_t) if last_t is not None else (float(prev_t[-1]) if prev_t.size else None)
    ts = [float(x) for x in chunk_ts]

    if len(ts) == int(n_new):
        new_t = np.asarray(ts, dtype=np.float64)
    elif len(ts) == 1:
        t_end = float(ts[0])
        if anchor is None:
            t0 = t_end - (n_new - 1) * step
            new_t = t0 + step * np.arange(n_new, dtype=np.float64)
        else:
            t_start = float(anchor) + step
            t_end_natural = t_start + (n_new - 1) * step
            if t_end >= t_end_natural:
                new_t = t_start + step * np.arange(n_new, dtype=np.float64)
            else:
                t0 = max(t_start, t_end - (n_new - 1) * step)
                if t0 <= anchor:
                    t0 = float(anchor) + step
                new_t = t0 + step * np.arange(n_new, dtype=np.float64)
    elif len(ts) > 1:
        new_t = np.linspace(float(ts[0]), float(ts[-1]), int(n_new), dtype=np.float64)
    else:
        t0 = (float(anchor) + step) if anchor is not None else 0.0
        new_t = t0 + step * np.arange(n_new, dtype=np.float64)

    if anchor is not None and new_t.size and new_t[0] <= anchor:
        shift = float(anchor) + step - float(new_t[0])
        new_t = new_t + shift

    if new_t.size >= 2:
        d = np.diff(new_t)
        if np.any(d <= 0):
            t0 = float(new_t[0]) if anchor is None else float(anchor) + step
            new_t = t0 + step * np.arange(n_new, dtype=np.float64)

    return np.asarray(new_t, dtype=np.float64)
