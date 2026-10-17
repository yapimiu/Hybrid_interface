"""
Автолог пакетного SSVEP: events.ndjson + eeg.npz + summary.json для разбора ошибок MSI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ssvep_analysis.experiment_logger import SSVEPExperimentLogger

# ~90 с @ 250 Hz — достаточно для разбора, без раздувания RAM
BURST_DEBUG_MAX_EEG_SAMPLES = 250 * 90


def expected_msi_lamp_from_diag(diag: Dict[str, object]) -> Optional[int]:
    """Одна лампа ON на конце окна (0-based) → номер MSI (1-based)."""
    on_end = diag.get("lamps_on_at_end_0idx")
    if not isinstance(on_end, list) or len(on_end) != 1:
        return None
    try:
        return int(on_end[0]) + 1
    except (TypeError, ValueError):
        return None


def summarize_burst_trace(
    events_path: Path,
    *,
    target_lamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Пост-обработка events.ndjson: доля совпадений MSI vs маркеры / цель."""
    if not events_path.is_file():
        return {"error": "no_events_file", "path": str(events_path)}

    msi_rows: List[Dict[str, Any]] = []
    with open(events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "burst_msi":
                continue
            msi_rows.append(rec.get("data") or {})

    n = len(msi_rows)
    if n == 0:
        return {"n_burst_msi": 0}

    vs_marker = 0
    vs_target = 0
    multi_on = 0
    for d in msi_rows:
        w = d.get("winner")
        exp = d.get("expected_from_markers")
        if d.get("n_lamps_on_at_end", 0) != 1:
            multi_on += 1
        if w is not None and exp is not None and int(w) == int(exp):
            vs_marker += 1
        tgt = target_lamp if target_lamp is not None else d.get("target_lamp")
        if w is not None and tgt is not None and int(w) == int(tgt):
            vs_target += 1

    return {
        "n_burst_msi": n,
        "match_marker_fraction": vs_marker / n,
        "match_target_fraction": vs_target / n if target_lamp else None,
        "multi_lamp_at_end_count": multi_on,
        "target_lamp": target_lamp,
    }


class BurstDebugSession:
    """Обёртка над SSVEPExperimentLogger для ssvep_burst_debug/."""

    def __init__(self, inner: SSVEPExperimentLogger) -> None:
        self._inner = inner

    @property
    def session_dir(self) -> Path:
        return self._inner.session_dir

    @classmethod
    def open_new(cls, output_root: Path, start_payload: Dict[str, Any]) -> "BurstDebugSession":
        output_root.mkdir(parents=True, exist_ok=True)
        inner = SSVEPExperimentLogger.open_new(
            output_root=output_root,
            start_payload={"log_kind": "burst_debug", **start_payload},
        )
        return cls(inner)

    def write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._inner.write(event, data)

    def append_eeg_chunk(
        self,
        times: Sequence[float],
        samples: Any,
        *,
        lsl_local_clock: Optional[float] = None,
    ) -> None:
        self._inner.append_eeg_chunk(
            times,
            samples,
            lsl_local_clock=lsl_local_clock,
            write_chunk_event=False,
            max_total_samples=BURST_DEBUG_MAX_EEG_SAMPLES,
        )

    def finalize(
        self,
        *,
        stop_payload: Dict[str, Any],
        channel_labels: Optional[Sequence[str]] = None,
        target_lamp: Optional[int] = None,
    ) -> Path:
        summary = summarize_burst_trace(
            self._inner.events_path,
            target_lamp=target_lamp,
        )
        log_dir = self._inner.finalize(
            stop_payload={**stop_payload, "burst_summary": summary},
            channel_labels=channel_labels,
        )
        summary_path = log_dir / "burst_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return log_dir
