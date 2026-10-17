"""Runtime shutdown ownership contract (Phase 3.3)."""

from __future__ import annotations

from typing import Any, Optional

# Ownership: who may close/release each resource class.
RESOURCE_OWNERSHIP = {
    "lsl_eeg_inlet": "ProtocolRunner (GUI thread tick)",
    "lsl_marker_inlet": "ProtocolRunner",
    "lsl_migalka_marker_inlet": "ProtocolRunner",
    "migalka_serial": "MigalkaSerialController (open on main; reader daemon enqueue-only)",
    "migalka_event_queue": "ProtocolRunner drain on tick/stop/finalize",
    "psychoPy_subprocess": "protocol_runner_gui (not ProtocolRunner)",
    "qt_timers": "protocol_runner_gui",
    "unified_logger": "ProtocolRunner finalize once per session",
}


def close_lsl_inlet(inlet: Any) -> None:
    """Best-effort close_stream; safe to call repeatedly."""
    if inlet is None:
        return
    try:
        inlet.close_stream()
    except Exception:
        pass
