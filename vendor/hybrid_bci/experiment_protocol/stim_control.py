"""Файловый обмен протокол ↔ PsychoPy: когда запускать следующий P300 trial."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


CONTROL_FILENAME = "stim_control.json"


def control_path(session_dir: Path) -> Path:
    return Path(session_dir) / CONTROL_FILENAME


def _write_text_with_retries(path: Path, text: str, *, attempts: int = 8) -> None:
    last_err: Optional[OSError] = None
    for attempt in range(attempts):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError as exc:
            last_err = exc
            time.sleep(0.05 * (attempt + 1))
    if last_err is not None:
        raise last_err


def write_control(session_dir: Path, payload: Dict[str, Any]) -> None:
    p = control_path(session_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["unix_ms"] = int(time.time() * 1000)
    text = json.dumps(body, ensure_ascii=False)
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    except OSError:
        # Windows: replace fails if stim_control.json is open (PsychoPy poll, AV).
        _write_text_with_retries(p, text)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_paused(
    session_dir: Path,
    *,
    reason: str = "",
    message: str = "",
    experiment_index: int = 0,
    experiment_total: int = 0,
) -> None:
    write_control(
        session_dir,
        {
            "state": "paused",
            "reason": str(reason),
            "message": str(message),
            "experiment_index": int(experiment_index),
            "experiment_total": int(experiment_total),
        },
    )


def write_trial_request(
    session_dir: Path,
    *,
    target_tile_id: int,
    experiment_index: int,
    experiment_total: int,
    label: str = "P300",
) -> None:
    write_control(
        session_dir,
        {
            "state": "trial",
            "target_tile_id": int(target_tile_id),
            "experiment_index": int(experiment_index),
            "experiment_total": int(experiment_total),
            "label": str(label),
        },
    )


def write_stim_done(session_dir: Path) -> None:
    write_control(session_dir, {"state": "done"})


def read_control(session_dir: Path) -> Optional[Dict[str, Any]]:
    p = control_path(session_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
