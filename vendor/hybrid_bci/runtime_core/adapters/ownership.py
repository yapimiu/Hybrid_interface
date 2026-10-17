"""I/O adapter ownership map (Phase 3.4d)."""

from __future__ import annotations

ADAPTER_OWNERSHIP = {
    "lsl_eeg_discovery": "runtime_core.adapters.lsl — GUI preflight + ProtocolRunner preflight",
    "lsl_eeg_inlet": "ProtocolRunner — open/close on main thread",
    "lsl_stim_markers": "ProtocolRunner — P300 block",
    "lsl_migalka_markers": "ProtocolRunner — SSVEP burst optional",
    "migalka_com": "MigalkaSerialController — ProtocolRunner main thread open/close",
    "migalka_event_queue": "ProtocolRunner — drain on tick/stop",
    "migalka_lsl_mirror": "MigalkaSerialController on drain (main thread)",
    "stim_control_json": "ProtocolRunner writes — PsychoPy child reads (gui/gui.py)",
    "psychoPy_subprocess": "protocol_runner_gui — not an adapter module",
}
