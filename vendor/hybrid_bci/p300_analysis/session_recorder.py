"""Append-only recorder for reproducible offline P300 debugging."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class SessionRecorder:
    """Writes all online analyzer runs into one NDJSON file."""

    def __init__(self, *, output_path: Optional[Path] = None, enabled: bool = False) -> None:
        root = Path(__file__).resolve().parent.parent
        self._path = output_path or (root / "data" / "p300_run_history" / "all_runs.ndjson")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._active_run_id: Optional[str] = None
        self._enabled = bool(enabled)

    @property
    def output_path(self) -> Path:
        return self._path

    def start_run(self, metadata: Dict[str, Any]) -> str:
        if not self._enabled:
            return ""
        run_id = f"run-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        self._active_run_id = run_id
        self._write("run_start", metadata, run_id=run_id)
        return run_id

    def stop_run(self, *, reason: str, summary: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        if self._active_run_id is None:
            return
        self._write("run_end", {"reason": reason, **summary}, run_id=self._active_run_id)
        self._active_run_id = None

    def log_markers(self, *, marker_chunk: Any, marker_ts: Any) -> None:
        if not self._enabled:
            return
        if self._active_run_id is None:
            return
        normalized = []
        for sample, ts in zip(marker_chunk, marker_ts):
            if isinstance(sample, (list, tuple)):
                marker_value = sample[0] if sample else ""
            else:
                marker_value = sample
            normalized.append({"ts": float(ts), "value": str(marker_value)})
        self._write("markers_chunk", {"markers": normalized}, run_id=self._active_run_id)

    def log_eeg_chunk(self, *, eeg_chunk: np.ndarray, eeg_ts: Any) -> None:
        if not self._enabled:
            return
        if self._active_run_id is None:
            return
        arr = np.asarray(eeg_chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr2d = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            arr2d = arr
        else:
            arr2d = arr.reshape(arr.shape[0], -1)
        payload = {
            "ts": [float(t) for t in eeg_ts],
            "shape": [int(x) for x in arr2d.shape],
            "samples": arr2d.tolist(),
        }
        self._write("eeg_chunk", payload, run_id=self._active_run_id)

    def log_winner(self, payload: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        if self._active_run_id is None:
            return
        self._write("winner_update", payload, run_id=self._active_run_id)

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Generic detailed event logger for offline replay/debug."""
        if not self._enabled:
            return
        if self._active_run_id is None:
            return
        self._write(event, payload, run_id=self._active_run_id)

    def _write(self, event: str, data: Dict[str, Any], *, run_id: str) -> None:
        try:
            line = json.dumps(
                {
                    "timestamp_ms": int(time.time() * 1000),
                    "run_id": run_id,
                    "event": event,
                    "data": data,
                },
                ensure_ascii=False,
            )
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Recorder must never break online analysis loop.
            pass
