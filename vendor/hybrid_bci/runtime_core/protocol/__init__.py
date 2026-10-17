"""Protocol FSM facade — avoid circular import via runner.py submodule."""

from __future__ import annotations

from runtime_core.protocol.display_state import ProtocolDisplayView
from runtime_core.protocol.operator_surface import (
    OPERATOR_RUNTIME_CALLS,
    block_running,
    eeg_indicator_state,
    session_ready,
)
from runtime_core.protocol.protocols import HybridProtocolRuntime

__all__ = [
    "HybridProtocolRuntime",
    "OPERATOR_RUNTIME_CALLS",
    "ProtocolDisplayView",
    "block_running",
    "eeg_indicator_state",
    "session_ready",
]
