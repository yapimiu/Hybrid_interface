"""NDJSON-лог для отладки (опционально)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def debug_ndjson(payload: Dict[str, Any]) -> None:
    try:
        p = Path(__file__).resolve().parent.parent / ".cursor" / "debug-5ea034.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"sessionId": "5ea034", "timestamp": int(time.time() * 1000), **payload},
            ensure_ascii=False,
        ) + "\n"
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
