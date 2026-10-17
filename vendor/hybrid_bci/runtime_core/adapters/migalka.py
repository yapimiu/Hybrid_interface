"""Migalka COM + event queue (implementations in ssvep_analysis)."""

from __future__ import annotations

from ssvep_analysis.migalka_event_queue import MigalkaEventQueue
from ssvep_analysis.migalka_serial_controller import (
    LampEvent,
    MigalkaConfig,
    MigalkaSerialController,
)

__all__ = [
    "LampEvent",
    "MigalkaConfig",
    "MigalkaEventQueue",
    "MigalkaSerialController",
]
