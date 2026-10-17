"""Runtime engine facade — math stays in p300_analysis / ssvep_analysis (3.4b)."""

from __future__ import annotations

from p300_analysis.online_engine import (
    P300Decision,
    P300EngineParams,
    P300OnlineEngine,
)
from ssvep_analysis.online_engine import (
    SSVEPDecision,
    SSVEPOnlineEngine,
    SSVEPParams,
)

from runtime_core.engines.protocols import (
    P300OnlineEngineProtocol,
    SSVEPOnlineEngineProtocol,
)

__all__ = [
    "P300Decision",
    "P300EngineParams",
    "P300OnlineEngine",
    "P300OnlineEngineProtocol",
    "SSVEPDecision",
    "SSVEPOnlineEngine",
    "SSVEPOnlineEngineProtocol",
    "SSVEPParams",
]
