"""Thread-safe buffer for Migalka COM lamp events (Phase 3.2)."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

LampEvent = Tuple[int, bool, str]


@dataclass(frozen=True)
class QueuedLampEvent:
    enqueued_at: float
    lamp: int
    is_on: bool
    raw: str

    def as_tuple(self) -> LampEvent:
        return (int(self.lamp), bool(self.is_on), str(self.raw))


class MigalkaEventQueue:
    """Producer: MigalkaSerialReader daemon. Consumer: GUI / ProtocolRunner.tick only."""

    def __init__(self, *, maxsize: int = 8192) -> None:
        self._maxsize = int(maxsize)
        self._q: queue.Queue[QueuedLampEvent] = queue.Queue(maxsize=self._maxsize)
        self._enqueued_total: int = 0
        self._dropped_total: int = 0
        self._drained_total: int = 0

    def enqueue(self, ev: LampEvent) -> bool:
        lamp, is_on, raw = ev
        item = QueuedLampEvent(
            enqueued_at=float(time.monotonic()),
            lamp=int(lamp),
            is_on=bool(is_on),
            raw=str(raw),
        )
        try:
            self._q.put_nowait(item)
            self._enqueued_total += 1
            return True
        except queue.Full:
            self._dropped_total += 1
            return False

    def drain(self) -> Tuple[List[LampEvent], Dict[str, float | int]]:
        """Drain all pending events (main thread only)."""
        items: List[QueuedLampEvent] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        now = float(time.monotonic())
        latencies_ms: List[float] = []
        events: List[LampEvent] = []
        for it in items:
            events.append(it.as_tuple())
            latencies_ms.append(max(0.0, (now - float(it.enqueued_at)) * 1000.0))
        self._drained_total += len(events)
        stats: Dict[str, float | int] = {
            "batch": len(events),
            "pending_after": int(self._q.qsize()),
            "enqueued_total": int(self._enqueued_total),
            "drained_total": int(self._drained_total),
            "dropped_total": int(self._dropped_total),
        }
        if latencies_ms:
            stats["max_latency_ms"] = round(max(latencies_ms), 3)
            stats["mean_latency_ms"] = round(sum(latencies_ms) / len(latencies_ms), 3)
        else:
            stats["max_latency_ms"] = 0.0
            stats["mean_latency_ms"] = 0.0
        return events, stats

    def pending_count(self) -> int:
        return int(self._q.qsize())
