"""Hardware / IPC adapters facade (3.4d)."""

from __future__ import annotations

from runtime_core.adapters import lsl, migalka, stim
from runtime_core.adapters.ownership import ADAPTER_OWNERSHIP

__all__ = [
    "ADAPTER_OWNERSHIP",
    "lsl",
    "migalka",
    "stim",
]
