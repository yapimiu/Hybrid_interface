"""
Подробный лог SSVEP-эксперимента: NDJSON-события + полный поток EEG (NPZ) + manifest.

Каталог по умолчанию: <repo>/ssvep_experiment_logs/run_YYYYMMDD_HHMMSS_<id>/
  - events.ndjson   — все события (построчный JSON)
  - manifest.json   — снимок параметров и итоги
  - eeg.npz         — times (n,), eeg (n, ch), channel_labels
  - plots/          — PNG скриншоты (опционально)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

import numpy as np

LOG_SCHEMA = "ssvep_experiment/v1"


def _msi_coef_length(c: Any) -> Optional[int]:
    """Длина коллекции Coef: IList.Count, массив .Length или len()."""
    cnt = getattr(c, "Count", None)
    if cnt is not None:
        return int(cnt)
    length = getattr(c, "Length", None)
    if length is not None:
        return int(length)
    try:
        return int(len(c))
    except TypeError:
        return None


def coef_values(
    msi: Any,
    n_freqs: int,
    *,
    winner_1based: Optional[int] = None,
    winner_coef: Optional[float] = None,
) -> List[Optional[float]]:
    """Sparse MSI score vector for logging (one entry per lamp frequency).

    MSIController exposes ``Coef`` as a single ``System.Double`` (often 0.0) and does
    **not** provide a per-frequency coefficient array after ``MSIExec(Signal)``.
    Pass ``winner_1based`` + ``winner_coef`` from ``MSIExec(Signal, out coefS)``.

    Legacy IList/Array ``Coef`` (if ever returned by a future DLL) is still supported.
    Scalar ``Coef`` alone is **not** mapped to index 0 (avoids fake ``[0.0, null, ...]``).
    """
    out: List[Optional[float]] = [None] * max(0, n_freqs)
    if winner_1based is not None and winner_coef is not None:
        idx = int(winner_1based) - 1
        if 0 <= idx < n_freqs:
            out[idx] = float(winner_coef)
        return out
    try:
        c = msi.Coef
    except Exception:
        return out
    if c is None:
        return out
    n_read = _msi_coef_length(c)
    if n_read is not None and n_read > 0:
        for i in range(min(int(n_read), n_freqs)):
            try:
                out[i] = float(c[i])
            except Exception:
                out[i] = None
        return out
    return out


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return repr(obj)


class SSVEPExperimentLogger:
    """Один каталог на запуск «Начать анализ»; events пишутся сразу, EEG — в память до finalize."""

    __slots__ = (
        "_dir",
        "_events_path",
        "_fh",
        "_run_id",
        "_seq",
        "_closed",
        "_eeg_times",
        "_eeg_data",
        "_n_events",
    )

    def __init__(self, session_dir: Path, fh: TextIO, run_id: str) -> None:
        self._dir = session_dir
        self._events_path = session_dir / "events.ndjson"
        self._fh = fh
        self._run_id = run_id
        self._seq = 0
        self._closed = False
        self._eeg_times: List[float] = []
        self._eeg_data: List[np.ndarray] = []
        self._n_events = 0

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def events_path(self) -> Path:
        return self._events_path

    @classmethod
    def open_new(
        cls,
        *,
        output_root: Path,
        start_payload: Dict[str, Any],
    ) -> "SSVEPExperimentLogger":
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pid = f"{int(time.time() * 1000) % 1_000_000:06d}"
        run_id = f"run_{stamp}_{pid}"
        session_dir = output_root / run_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "plots").mkdir(exist_ok=True)
        events_path = session_dir / "events.ndjson"
        fh = open(events_path, "a", encoding="utf-8", buffering=1)
        logger = cls(session_dir, fh, run_id)
        logger.write("experiment_start", start_payload)
        return logger

    def write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        self._seq += 1
        rec = {
            "schema": LOG_SCHEMA,
            "seq": self._seq,
            "event": event,
            "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "unix_ms": int(time.time() * 1000),
            "monotonic_ns": time.monotonic_ns(),
            "run_id": self._run_id,
            "data": _json_safe(data or {}),
        }
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._n_events += 1
        except Exception:
            pass

    def append_eeg_chunk(
        self,
        times: Sequence[float],
        samples: np.ndarray,
        *,
        lsl_local_clock: Optional[float] = None,
        write_chunk_event: bool = True,
        max_total_samples: Optional[int] = None,
    ) -> None:
        """Накопить EEG (samples, channels) с LSL-временами."""
        if self._closed or samples.size == 0:
            return
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ts = np.asarray(times, dtype=np.float64)
        if ts.size != arr.shape[0]:
            if ts.size == 1:
                fs_guess = 250.0
                t0 = float(ts[0]) - (arr.shape[0] - 1) / fs_guess
                ts = t0 + np.arange(arr.shape[0], dtype=np.float64) / fs_guess
            elif ts.size > 1 and arr.shape[0] > 1:
                ts = np.linspace(float(ts[0]), float(ts[-1]), arr.shape[0])
            else:
                return
        self._eeg_times.extend(ts.tolist())
        self._eeg_data.append(arr)
        if max_total_samples is not None:
            self.trim_eeg_samples(int(max_total_samples))
        if write_chunk_event:
            self.write(
                "eeg_chunk",
                {
                    "n_samples": int(arr.shape[0]),
                    "n_channels": int(arr.shape[1]),
                    "t_first": float(ts[0]),
                    "t_last": float(ts[-1]),
                    "lsl_local_clock": lsl_local_clock,
                    "total_eeg_samples": len(self._eeg_times),
                },
            )

    def trim_eeg_samples(self, max_samples: int) -> None:
        """Оставить только последние max_samples точек EEG в RAM."""
        max_samples = max(0, int(max_samples))
        n = len(self._eeg_times)
        if n <= max_samples:
            return
        drop = n - max_samples
        self._eeg_times = self._eeg_times[drop:]
        remaining = max_samples
        trimmed: List[np.ndarray] = []
        for ch in self._eeg_data:
            if remaining <= 0:
                break
            if ch.shape[0] <= remaining:
                trimmed.append(ch)
                remaining -= int(ch.shape[0])
            else:
                trimmed.append(ch[-remaining:])
                remaining = 0
        self._eeg_data = trimmed

    def finalize(
        self,
        *,
        stop_payload: Dict[str, Any],
        channel_labels: Optional[Sequence[str]] = None,
    ) -> Path:
        """Записать manifest.json и eeg.npz, закрыть events."""
        if not self._closed:
            self.write("experiment_stop", stop_payload)
        manifest: Dict[str, Any] = {
            "schema": LOG_SCHEMA,
            "run_id": self._run_id,
            "session_dir": str(self._dir),
            "events_file": str(self._events_path),
            "n_events": self._n_events,
            "stop": _json_safe(stop_payload),
        }
        if self._eeg_times and self._eeg_data:
            eeg = np.vstack(self._eeg_data)
            times = np.asarray(self._eeg_times, dtype=np.float64)
            n = min(len(times), eeg.shape[0])
            times = times[:n]
            eeg = eeg[:n]
            labels = list(channel_labels or [])
            if len(labels) < eeg.shape[1]:
                labels.extend(f"Ch{i + 1}" for i in range(len(labels), eeg.shape[1]))
            npz_path = self._dir / "eeg.npz"
            np.savez_compressed(
                npz_path,
                times=times,
                eeg=eeg,
                channel_labels=np.array(labels, dtype=object),
            )
            manifest["eeg_npz"] = str(npz_path)
            manifest["eeg_samples"] = int(eeg.shape[0])
            manifest["eeg_channels"] = int(eeg.shape[1])
            manifest["eeg_t_span_sec"] = float(times[-1] - times[0]) if n > 1 else 0.0
        else:
            manifest["eeg_npz"] = None
        manifest_path = self._dir / "manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        self.close()
        return self._dir

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:
            pass

    def plots_dir(self) -> Path:
        return self._dir / "plots"
