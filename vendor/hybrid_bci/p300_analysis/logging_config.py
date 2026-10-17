"""Настройка логгера p300_analyzer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging() -> Path:
    """Файл рядом со скриптом запуска + stderr; только логгер p300_analyzer."""
    log_path = Path(__file__).resolve().parent.parent / "scripts" / "p300_analyzer.log"
    lg = logging.getLogger("p300_analyzer")
    if lg.handlers:
        return log_path
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    return log_path
