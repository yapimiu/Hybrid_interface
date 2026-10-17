"""PsychoPy stim_control file IPC (implementation in experiment_protocol.stim_control)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from experiment_protocol.stim_control import (
    CONTROL_FILENAME,
    control_path,
    read_control,
    write_control,
    write_paused,
    write_stim_done,
    write_trial_request,
)

def read_stim_state(session_dir: Path) -> Optional[str]:
    """Current stim_control.json state field, or None if missing/unreadable."""
    cmd = read_control(session_dir)
    if cmd is None:
        return None
    return str(cmd.get("state") or "") or None


def is_stim_session_done(session_dir: Path) -> bool:
    """True when PsychoPy reported block completion via stim_control."""
    return read_stim_state(session_dir) == "done"


__all__ = [
    "CONTROL_FILENAME",
    "control_path",
    "is_stim_session_done",
    "read_control",
    "read_stim_state",
    "write_control",
    "write_paused",
    "write_stim_done",
    "write_trial_request",
]
