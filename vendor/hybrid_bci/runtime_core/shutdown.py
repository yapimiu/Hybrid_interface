"""Shutdown contract re-export (3.4a — physical module still in experiment_protocol)."""

from __future__ import annotations

from experiment_protocol.shutdown_contract import (
    RESOURCE_OWNERSHIP,
    close_lsl_inlet,
)

__all__ = ["RESOURCE_OWNERSHIP", "close_lsl_inlet"]
