"""Headless hybrid-BCI runtime (extraction from experiment_protocol + GUI).

Phase 3.4 slices (dependency order):
  3.4a — logging facades + protocols (this package root)
  3.4b — engines (P300/SSVEP online)
  3.4c — protocol FSM (ProtocolRunner)
  3.4d — I/O adapters (LSL, Migalka, stim_control)
  3.4e — thin protocol_runner_gui
"""

from __future__ import annotations

__all__ = ["adapters", "engines", "logging", "protocol", "shutdown"]
