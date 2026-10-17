"""Lifecycle event contract re-export (3.4c)."""

from __future__ import annotations

from experiment_protocol.lifecycle import (
    EVENT_P300_BLOCK_START,
    EVENT_SESSION_START,
    EVENT_SESSION_STOP,
    EVENT_SSVEP_BLOCK_END,
    EVENT_SSVEP_BLOCK_START,
    EVENT_SSVEP_QUEUE_START,
    LIFECYCLE_STATES,
    audit_lifecycle_events,
)

__all__ = [
    "EVENT_P300_BLOCK_START",
    "EVENT_SESSION_START",
    "EVENT_SESSION_STOP",
    "EVENT_SSVEP_BLOCK_END",
    "EVENT_SSVEP_BLOCK_START",
    "EVENT_SSVEP_QUEUE_START",
    "LIFECYCLE_STATES",
    "audit_lifecycle_events",
]
