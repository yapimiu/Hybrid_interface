"""Adapter contracts (implementations remain in analysis / experiment_protocol packages)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class StimControlAdapter(Protocol):
    def write_paused(
        self,
        session_dir: Path,
        *,
        reason: str = "",
        message: str = "",
        experiment_index: int = 0,
        experiment_total: int = 0,
    ) -> None: ...

    def write_trial_request(
        self,
        session_dir: Path,
        *,
        target_tile_id: int,
        experiment_index: int,
        experiment_total: int,
        label: str = "P300",
    ) -> None: ...

    def write_stim_done(self, session_dir: Path) -> None: ...

    def read_control(self, session_dir: Path) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class MigalkaControllerAdapter(Protocol):
    def open_and_start(self, cfg: Any) -> None: ...

    def stop_and_close(self, *, join_timeout_s: float = 2.0) -> None: ...

    def is_open(self) -> bool: ...

    def drain_pending_events(self) -> Any: ...

    def pending_event_count(self) -> int: ...
