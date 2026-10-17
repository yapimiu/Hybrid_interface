"""Live MSI decision trace (observability only — does not affect classification)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def coef_s_from_sparse(coef: Any, winner_0idx: Optional[int]) -> Optional[float]:
    if winner_0idx is None or not isinstance(coef, list):
        return None
    i = int(winner_0idx)
    if 0 <= i < len(coef) and coef[i] is not None:
        return float(coef[i])
    return None


def _winner_1based_to_0idx(winner_1based: Optional[int]) -> Optional[int]:
    if winner_1based is None:
        return None
    return int(winner_1based) - 1


def resolve_ssvep_block_winners(
    points: Sequence[Dict[str, Any]],
    *,
    final_winner_0idx: Optional[int] = None,
    final_winner_1based: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Statistical block winner from throttled MSI trace (majority vote).

  ``winner_last_*`` — последняя точка trace, иначе финальный classify().
  При ничьей по частоте ``winner_majority_*`` = None (undecided).
    """
    winners_1 = [int(p["winner_1based"]) for p in points if p.get("winner_1based") is not None]
    last_1: Optional[int] = None
    last_0: Optional[int] = None
    if winners_1:
        last_1 = int(winners_1[-1])
        last_0 = _winner_1based_to_0idx(last_1)
    elif final_winner_1based is not None:
        last_1 = int(final_winner_1based)
        last_0 = (
            int(final_winner_0idx)
            if final_winner_0idx is not None
            else _winner_1based_to_0idx(last_1)
        )
    elif final_winner_0idx is not None:
        last_0 = int(final_winner_0idx)
        last_1 = int(last_0) + 1

    histogram: Dict[str, int] = {}
    majority_1: Optional[int] = None
    majority_0: Optional[int] = None
    majority_count: Optional[int] = None
    majority_tied = False
    if winners_1:
        hist = Counter(winners_1)
        histogram = {str(k): int(v) for k, v in sorted(hist.items())}
        max_count = max(hist.values())
        top = [lamp for lamp, count in hist.items() if count == max_count]
        if len(top) == 1:
            majority_1 = int(top[0])
            majority_0 = _winner_1based_to_0idx(majority_1)
            majority_count = int(max_count)
        else:
            majority_tied = True
    elif last_1 is not None:
        majority_1 = int(last_1)
        majority_0 = last_0
        majority_count = 1
        histogram = {str(majority_1): 1}

    return {
        "winner_majority_1based": majority_1,
        "winner_majority_0idx": majority_0,
        "winner_majority_count": majority_count,
        "winner_majority_tied": bool(majority_tied),
        "winner_last_1based": last_1,
        "winner_last_0idx": last_0,
        "decision_histogram": histogram,
    }


def summarize_msi_decision_trace(
    points: Sequence[Dict[str, Any]],
    *,
    target_lamp_0idx: Optional[int] = None,
    final_winner_0idx: Optional[int] = None,
    block_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate stability / switches / lock-on from throttled live decisions."""
    winners_1 = [int(p["winner_1based"]) for p in points if p.get("winner_1based") is not None]
    n = len(winners_1)
    compact = " ".join(str(w) for w in winners_1)
    winner_stats = resolve_ssvep_block_winners(
        points,
        final_winner_0idx=final_winner_0idx,
        final_winner_1based=(int(final_winner_0idx) + 1 if final_winner_0idx is not None else None),
    )

    if n == 0:
        return {
            "compact_stream": compact,
            "n_decisions": 0,
            "stability_pct": None,
            "switch_count": 0,
            "switch_rate_per_sec": None,
            "avg_lock_duration_sec": None,
            "false_transition_count": 0,
            "lock_on_sec": None,
            "dominant_winner_1based": winner_stats.get("winner_majority_1based"),
            "final_matches_dominant": None,
            **winner_stats,
        }

    switches = sum(1 for i in range(1, n) if winners_1[i] != winners_1[i - 1])
    dur = float(block_duration_sec) if block_duration_sec and block_duration_sec > 0 else None
    if dur is None and len(points) >= 2:
        t0 = points[0].get("elapsed_s")
        t1 = points[-1].get("elapsed_s")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and float(t1) > float(t0):
            dur = float(t1) - float(t0)
    switch_rate = (float(switches) / dur) if dur and dur > 0 else None

    # Run lengths in wall-time (elapsed_s between first/last point of each constant winner).
    lock_durations: List[float] = []
    run_start = 0
    for i in range(1, n):
        if winners_1[i] != winners_1[i - 1]:
            lock_durations.append(_run_duration_sec(points, run_start, i - 1))
            run_start = i
    lock_durations.append(_run_duration_sec(points, run_start, n - 1))
    avg_lock = sum(lock_durations) / len(lock_durations) if lock_durations else None

    dominant = winner_stats.get("winner_majority_1based")
    if dominant is None and not winner_stats.get("winner_majority_tied"):
        dominant = max(set(winners_1), key=winners_1.count)
    stability_pct = (
        round(100.0 * winners_1.count(int(dominant)) / n, 2) if dominant is not None else None
    )

    target_1 = int(target_lamp_0idx) + 1 if target_lamp_0idx is not None else None
    false_trans = 0
    lock_on_sec: Optional[float] = None
    if target_1 is not None:
        stability_pct = round(100.0 * sum(1 for w in winners_1 if w == target_1) / n, 2)
        for i in range(1, n):
            prev_ok = winners_1[i - 1] == target_1
            curr_ok = winners_1[i] == target_1
            if prev_ok and not curr_ok:
                false_trans += 1
        for p in points:
            if p.get("winner_1based") == target_1:
                el = p.get("elapsed_s")
                if isinstance(el, (int, float)):
                    lock_on_sec = round(float(el), 4)
                break

    last_0 = winner_stats.get("winner_last_0idx")
    majority_0 = winner_stats.get("winner_majority_0idx")
    final_dom_match = None
    if last_0 is not None and majority_0 is not None:
        final_dom_match = int(last_0) == int(majority_0)
    elif final_winner_0idx is not None and dominant is not None:
        final_dom_match = int(final_winner_0idx) + 1 == int(dominant)

    out = {
        "compact_stream": compact,
        "n_decisions": n,
        "stability_pct": stability_pct,
        "switch_count": int(switches),
        "switch_rate_per_sec": round(switch_rate, 4) if switch_rate is not None else None,
        "avg_lock_duration_sec": round(avg_lock, 4) if avg_lock is not None else None,
        "false_transition_count": int(false_trans),
        "lock_on_sec": lock_on_sec,
        "dominant_winner_1based": int(dominant) if dominant is not None else None,
        "final_matches_dominant": final_dom_match,
        **winner_stats,
    }
    return out


def _run_duration_sec(points: Sequence[Dict[str, Any]], i0: int, i1: int) -> float:
    t_a = points[i0].get("elapsed_s")
    t_b = points[i1].get("elapsed_s")
    if isinstance(t_a, (int, float)) and isinstance(t_b, (int, float)):
        return max(0.0, float(t_b) - float(t_a))
    return max(1.0, float(i1 - i0 + 1))


@dataclass
class MsiDecisionTraceCollector:
    """Throttled live MSI winner stream for one SSVEP block."""

    min_interval_sec: float = 2.0
    _points: List[Dict[str, Any]] = field(default_factory=list)
    _last_wall_t: Optional[float] = None

    def reset(self) -> None:
        self._points.clear()
        self._last_wall_t = None

    def _due(self, wall_t: float, *, force: bool) -> bool:
        if force:
            return True
        if self._last_wall_t is None:
            return True
        return (float(wall_t) - float(self._last_wall_t)) >= float(self.min_interval_sec)

    def record(
        self,
        dec: Any,
        *,
        wall_t: float,
        elapsed_s: Optional[float] = None,
        target_lamp_0idx: Optional[int] = None,
        mode: str = "",
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Append one trace point if throttle allows; returns point or None."""
        if not self._due(wall_t, force=force):
            return None

        w1 = getattr(dec, "winner_1based", None)
        w0 = getattr(dec, "winner_0idx", None)
        coef = getattr(dec, "coef", None)
        debug = getattr(dec, "debug", None) or {}
        if not isinstance(debug, dict):
            debug = {}

        point: Dict[str, Any] = {
            "t_wall": round(float(wall_t), 6),
            "elapsed_s": round(float(elapsed_s), 4) if elapsed_s is not None else None,
            "winner_1based": int(w1) if w1 is not None else None,
            "winner_0idx": int(w0) if w0 is not None else None,
            "winner_label": f"L{int(w1)}" if w1 is not None else None,
            "coefS": coef_s_from_sparse(coef, w0),
            "classify_allowed": bool(getattr(dec, "classify_allowed", False)),
            "ssvep_mode": str(mode),
            "target_lamp_0idx": target_lamp_0idx,
            "burst_gate_reason": debug.get("gate_reason"),
            "msi_exec_ms": debug.get("msi_exec_ms"),
        }
        self._points.append(point)
        self._last_wall_t = float(wall_t)
        return point

    def finalize(
        self,
        *,
        target_lamp_0idx: Optional[int] = None,
        final_winner_0idx: Optional[int] = None,
        block_duration_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        summary = summarize_msi_decision_trace(
            self._points,
            target_lamp_0idx=target_lamp_0idx,
            final_winner_0idx=final_winner_0idx,
            block_duration_sec=block_duration_sec,
        )
        return {
            "points": list(self._points),
            "summary": summary,
            "compact_stream": summary.get("compact_stream", ""),
        }
