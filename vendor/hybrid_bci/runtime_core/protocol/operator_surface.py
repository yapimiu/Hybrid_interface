"""Explicit operator/GUI contract for HybridProtocolRuntime (Phase 3.4e)."""

from __future__ import annotations

from runtime_core.protocol.display_state import ProtocolDisplayView

# Procedural API the GUI may invoke on HybridProtocolRuntime (no cfg/engines/inlets).
OPERATOR_RUNTIME_CALLS = frozenset(
    {
        "start",
        "stop",
        "tick",
        "display_view",
        "dismiss_operator_ssvep_overlay",
        "set_ssvep_display_clear_callback",
        "start_p300_block",
        "start_ssvep_block",
    }
)

_IDLE_STATES = frozenset({"idle", "stopped"})


def session_ready(view: ProtocolDisplayView | None) -> bool:
    return view is not None and bool(view.session_ready)


def block_running(view: ProtocolDisplayView | None) -> bool:
    return view is not None and bool(view.block_running)


def eeg_indicator_state(view: ProtocolDisplayView | None) -> str | None:
    """off | stale | live — None means use GUI local test inlet."""
    if view is None or view.state in _IDLE_STATES:
        return None
    if not view.eeg_inlet_open:
        return "stale"
    return "live" if view.eeg_stream_live else "stale"
