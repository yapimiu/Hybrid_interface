"""Session logger facade — minimal CSV layout only (legacy name retained)."""

from __future__ import annotations

from experiment_protocol.minimal_session_logger import (
    LOG_SCHEMA,
    MinimalSessionLogger,
    SessionPaths,
    build_p300_averages_export,
    build_ssvep_averages_export,
)

UnifiedExperimentLogger = MinimalSessionLogger

__all__ = [
    "LOG_SCHEMA",
    "MinimalSessionLogger",
    "SessionPaths",
    "UnifiedExperimentLogger",
    "build_p300_averages_export",
    "build_ssvep_averages_export",
]
