"""Verbose console logging for clinical protocol runner (stdout, flushed)."""

from __future__ import annotations

import sys
import time
from typing import Any


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, *, level: str = "INFO") -> None:
    line = f"[{_ts()}] [protocol] [{level}] {msg}"
    print(line, flush=True)


def info(msg: str) -> None:
    log(msg, level="INFO")


def warn(msg: str) -> None:
    log(msg, level="WARN")


def error(msg: str) -> None:
    log(msg, level="ERROR")


def exc(context: str, err: BaseException) -> None:
    error(f"{context}: {type(err).__name__}: {err}")


def state_change(old: str, new: str, *, detail: str = "") -> None:
    extra = f" — {detail}" if detail else ""
    info(f"STATE {old} -> {new}{extra}")


def event(name: str, **fields: Any) -> None:
    parts = ", ".join(f"{k}={v!r}" for k, v in fields.items())
    info(f"{name}: {parts}" if parts else name)
