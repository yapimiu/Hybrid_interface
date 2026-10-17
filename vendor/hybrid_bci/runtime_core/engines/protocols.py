"""Online engine contracts (implementations remain in p300_analysis / ssvep_analysis)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class P300OnlineEngineProtocol(Protocol):
    """Headless P300 epoch buffer + AUC/template decision (protocol path)."""

    params: Any
    epochs_data: Dict[str, List[Any]]
    pending_markers: List[Any]
    external_template_window: Any
    external_template_target_id: Optional[int]
    current_cue_target_id: Optional[int]

    def reset(self, *, params: Optional[Any] = None) -> None: ...

    def set_eeg_inlet(self, inlet: Any) -> None: ...

    def ingest_marker_chunk(
        self,
        *,
        marker_chunk: Any,
        marker_ts: Any,
        lsl_local_clock_now: float,
    ) -> None: ...

    def ingest_eeg_chunk(
        self,
        *,
        eeg_chunk: Any,
        eeg_ts: Any,
        lsl_local_clock_now: float,
    ) -> None: ...

    def extract_ready_epochs(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    def try_build_external_template_from_epochs(
        self,
        *,
        target_tile_id: int,
        min_epochs: int,
    ) -> bool: ...

    def compute_decision(self, *, winner_mode: str) -> Any: ...


@runtime_checkable
class SSVEPOnlineEngineProtocol(Protocol):
    """Headless SSVEP buffer + burst gate + MSI classify."""

    params: Any

    def reset(self, *, params: Optional[Any] = None) -> None: ...

    def ingest_eeg_chunk(
        self,
        *,
        times: Any,
        samples: Any,
        lsl_clock_now: float,
    ) -> None: ...

    def ingest_migalka_marker(self, *, lsl_time: float, value: Any) -> None: ...

    def can_classify(self) -> bool: ...

    def classify(self) -> Any: ...
