"""Read-only view of protocol state for GUI (Phase 3.4c)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolDisplayView:
    """Snapshot for operator UI — no engine internals."""

    state: str
    status_text: str
    session_ready: bool
    block_running: bool
    pause_active: bool
    eeg_inlet_open: bool
    eeg_stream_live: bool
    ssvep_blackout_visible: bool
    ssvep_cue_visible: bool
    ssvep_cue_exp_index: int
    ssvep_cue_exp_total: int
    ssvep_cue_lamp_1based: int
    ssvep_cue_freq_hz: float
    ssvep_cue_mode_label: str
    ssvep_cue_group_index: int
