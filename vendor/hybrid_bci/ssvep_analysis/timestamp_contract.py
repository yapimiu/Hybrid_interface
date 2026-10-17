"""SSVEP EEG timestamp contract (Stage 2.3)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def timestamps_need_lsl_anchor(ts: Sequence[float]) -> bool:
    """True when inlet timestamps are too coarse for per-sample SSVEP alignment."""
    if not ts:
        return True
    t_last = float(ts[-1])
    if t_last > 1e6:
        return True
    if len(ts) >= 2:
        med = float(np.median(np.diff(np.asarray(ts, dtype=np.float64))))
        return med > 0.05
    return len(ts) == 1


def validate_timestamp_stream(
    buf_t: np.ndarray,
    *,
    expected_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Diagnostics for replay/debug (monotonicity, duplicates, gaps)."""
    t = np.asarray(buf_t, dtype=np.float64).ravel()
    n = int(t.size)
    out: Dict[str, Any] = {"n": n, "expected_len": expected_len, "len_match": expected_len == n if expected_len is not None else None}
    if n < 2:
        return out
    d = np.diff(t)
    out["dt_min"] = float(np.min(d))
    out["dt_max"] = float(np.max(d))
    out["dt_median"] = float(np.median(d))
    out["non_monotonic_steps"] = int(np.sum(d <= 0))
    out["duplicate_steps"] = int(np.sum(d == 0))
    out["max_backward_s"] = float(np.min(d)) if np.any(d < 0) else 0.0
    return out
