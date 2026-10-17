"""Protocol lifecycle contract and validation (Stage 2.5)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

# FSM states (ProtocolRunner)
LIFECYCLE_STATES = (
    "idle",
    "preflight",
    "ready",
    "main",
    "ssvep_continuous",
    "ssvep_burst",
    "finalize",
    "stopped",
)

# Primary session events (events.ndjson)
EVENT_SESSION_START = "session_start"
EVENT_SESSION_STOP = "session_stop"
EVENT_SSVEP_QUEUE_START = "ssvep_queue_start"
EVENT_SSVEP_BLOCK_START = "ssvep_block_start"
EVENT_SSVEP_BLOCK_END = "ssvep_block_end"
EVENT_P300_BLOCK_START = "p300_block_start"


def audit_lifecycle_events(events: List[Dict[str, Any]], manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic lifecycle audit from sorted events.ndjson rows."""
    counts = Counter(str(e.get("event")) for e in events if e.get("event"))
    ssvep_start = int(counts.get(EVENT_SSVEP_BLOCK_START, 0))
    ssvep_end = int(counts.get(EVENT_SSVEP_BLOCK_END, 0))
    ssvep_queue = int(counts.get(EVENT_SSVEP_QUEUE_START, 0))
    # Legacy sessions used ssvep_block_start for queue plan (extra +1 per mode batch).
    legacy_queue_as_block = int(counts.get(EVENT_SSVEP_BLOCK_START, 0)) - ssvep_end
    if legacy_queue_as_block < 0:
        legacy_queue_as_block = 0

    missing_stop = int(counts.get(EVENT_SESSION_STOP, 0)) == 0 and not (manifest or {}).get("stop")
    orphan_blocks = max(0, ssvep_start - ssvep_end)
    return {
        "event_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "session_start": int(counts.get(EVENT_SESSION_START, 0)),
        "session_stop": int(counts.get(EVENT_SESSION_STOP, 0)),
        "missing_session_stop": bool(missing_stop),
        "ssvep_block_start": ssvep_start,
        "ssvep_block_end": ssvep_end,
        "ssvep_queue_start": ssvep_queue,
        "ssvep_start_end_delta": int(ssvep_start - ssvep_end),
        "ssvep_orphan_starts": int(orphan_blocks),
        "legacy_extra_ssvep_block_start": int(legacy_queue_as_block),
    }
