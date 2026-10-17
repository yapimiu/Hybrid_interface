"""Logging contracts for runtime-core (implementations stay in experiment_protocol for 3.4a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class AnalysisWriterProtocol(Protocol):
    """CSV analysis/ bundle under session_dir/analysis/."""

    analysis_dir: Path

    def finish_experiment(self, flat_row: Dict[str, Any]) -> None: ...

    def finalize(self, *, stop_reason: str) -> None: ...


@runtime_checkable
class ExperimentSessionLogger(Protocol):
    """Session-scoped NDJSON + manifest + optional EEG archive."""

    session_id: str
    session_dir: Path

    @classmethod
    def allocate_session_dir(
        cls,
        *,
        output_root: Path,
        subject_id: str,
    ) -> Path: ...

    @classmethod
    def open_new(
        cls,
        *,
        output_root: Path,
        subject_id: str,
        protocol_plan: Dict[str, Any],
        start_payload: Dict[str, Any],
        session_dir: Optional[Path] = None,
    ) -> "ExperimentSessionLogger": ...

    def attach_analysis(self, writer: AnalysisWriterProtocol) -> None: ...

    def write(self, event: str, payload: Dict[str, Any]) -> None: ...

    def write_tech(self, event: str, payload: Dict[str, Any]) -> None: ...

    def append_experiment(self, row: Dict[str, Any]) -> int: ...

    def set_eeg_channel_labels(self, labels: Any) -> None: ...

    def append_eeg_chunk(self, chunk: Any, timestamps: Any) -> None: ...

    def finalize(self, *, stop_payload: Dict[str, Any]) -> Path: ...
