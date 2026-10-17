"""Подробный построчный NDJSON-лог для каждого обследования (один файл на запись «Начать анализ»).

Формат строки: JSON с полями schema, event, wall_time_iso, unix_ms, monotonic_ns, run_seq, data.
Не должен ломать анализатор при ошибках записи.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

LOG_SCHEMA = "p300_exam_detail/v1"


class ExamSessionDetailLogger:
    """Один файл на сессию записи эпох; flush после каждой записи."""

    __slots__ = ("_path", "_fh", "_run_seq", "_closed")

    def __init__(self, path: Path, fh: TextIO, run_seq: int) -> None:
        self._path = path
        self._fh = fh
        self._run_seq = int(run_seq)
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def open_new(
        cls,
        *,
        run_seq: int,
        exam_start_data: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> "ExamSessionDetailLogger":
        root = Path(__file__).resolve().parent.parent
        out_dir = output_dir if output_dir is not None else (root / "data" / "examination_logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pid = f"{int(time.time() * 1000) % 1_000_000:06d}"
        path = out_dir / f"exam_run{int(run_seq):04d}_{stamp}_{pid}.ndjson"
        fh = open(path, "a", encoding="utf-8", buffering=1)
        self = cls(path, fh, run_seq)
        self.write("exam_start", exam_start_data)
        return self

    def write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        rec = {
            "schema": LOG_SCHEMA,
            "event": event,
            "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "unix_ms": int(time.time() * 1000),
            "monotonic_ns": time.monotonic_ns(),
            "run_seq": self._run_seq,
            "data": data or {},
        }
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:
            pass


def pending_snapshot_for_log(pending: List[Tuple[float, str]], max_each_side: int = 8) -> Dict[str, Any]:
    n = len(pending)
    if n == 0:
        return {"n": 0, "head": [], "tail": []}
    head = [{"marker_ts": float(ts), "stim_key": sk} for ts, sk in pending[:max_each_side]]
    tail = [{"marker_ts": float(ts), "stim_key": sk} for ts, sk in pending[-max_each_side:]]
    return {"n": n, "head": head, "tail": tail, "same_head_tail": n <= max_each_side * 2}


def summarize_eeg_chunk(arr_2d: Any, eeg_ts: List[float]) -> Dict[str, Any]:
    """Сжатая статистика чанка ЭЭГ без записи всех сырых отсчётов в лог."""
    import numpy as np

    a = np.asarray(arr_2d, dtype=np.float64)
    if a.size == 0:
        return {"empty": True}
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    ts = np.asarray(eeg_ts, dtype=np.float64)
    out: Dict[str, Any] = {
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "n_samples": int(a.shape[0]),
        "n_channels": int(a.shape[1]),
        "ts_first": float(ts[0]) if ts.size else None,
        "ts_last": float(ts[-1]) if ts.size else None,
        "ts_span_s": float(ts[-1] - ts[0]) if ts.size > 1 else 0.0,
        "ts_unique_in_chunk": int(np.unique(ts).size) if ts.size else 0,
    }
    if ts.size > 1:
        d = np.diff(ts)
        d = d[d > 0]
        if d.size:
            out["ts_median_dt_ms"] = float(np.median(d) * 1000.0)
            out["ts_min_dt_ms"] = float(np.min(d) * 1000.0)
            out["ts_max_dt_ms"] = float(np.max(d) * 1000.0)
    # По каналам: mean, std, min, max (первые до 32 каналов)
    max_ch = min(a.shape[1], 32)
    ch_stats = []
    for c in range(max_ch):
        col = a[:, c]
        ch_stats.append(
            {
                "ch": c,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }
        )
    out["per_channel_stats"] = ch_stats
    out["global_mean_abs"] = float(np.mean(np.abs(a)))
    return out


def epoch_roi_summary(epoch_1d: Any) -> Dict[str, Any]:
    import numpy as np

    e = np.asarray(epoch_1d, dtype=np.float64).ravel()
    n = int(e.size)
    out: Dict[str, Any] = {
        "len": n,
        "mean": float(np.mean(e)) if n else None,
        "std": float(np.std(e)) if n else None,
        "min": float(np.min(e)) if n else None,
        "max": float(np.max(e)) if n else None,
    }
    # Полные отсчёты усреднённого ROI-ряда (обычно ~201 отсчёт при 800 мс @ 250 Гц)
    max_keep = 512
    if n <= max_keep:
        out["epoch_samples_roi_mean"] = [float(x) for x in e.tolist()]
    else:
        out["epoch_samples_roi_mean_head"] = [float(x) for x in e[:32].tolist()]
        out["epoch_samples_roi_mean_tail"] = [float(x) for x in e[-32:].tolist()]
    return out
