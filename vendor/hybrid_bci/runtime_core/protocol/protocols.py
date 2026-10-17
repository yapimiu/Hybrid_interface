"""Hybrid protocol runtime contract (implementation in experiment_protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from runtime_core.protocol.display_state import ProtocolDisplayView


@runtime_checkable
class HybridProtocolRuntime(Protocol):
    """Headless hybrid protocol FSM (driven from GUI main thread)."""

    state: str
    status_text: str
    cfg: Any

    @property
    def session_ready(self) -> bool: ...

    @property
    def block_running(self) -> bool: ...

    @property
    def eeg_stream_live(self) -> bool: ...

    @property
    def eeg_inlet_open(self) -> bool: ...

    def display_view(self) -> ProtocolDisplayView: ...

    def dismiss_operator_ssvep_overlay(self) -> None: ...

    def set_ssvep_display_clear_callback(self, cb: Optional[Callable[[], None]]) -> None: ...

    def start(self) -> None: ...

    def tick(self) -> None: ...

    def stop(self, *, reason: str = "user_stop") -> None: ...

    def start_p300_block(self) -> bool: ...

    def start_ssvep_block(self, *, mode: str) -> bool: ...
