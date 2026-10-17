"""Разбор строковых маркеров LSL (плитки, trial_start)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np


def decode_stim_tile_id(raw_id: int) -> Optional[int]:
    """Decode tile id from LSL marker payload.

    Supported formats:
    - New: 100..108 -> 0..8
    - Legacy: 0..8 -> 0..8
    """
    if raw_id < 0:
        return None
    tile_id = raw_id - 100 if raw_id >= 100 else raw_id
    if 0 <= tile_id <= 8:
        return tile_id
    return None


def marker_value_to_stim_key(marker_value: Any) -> Optional[str]:
    """
    Ключ класса эпохи, например «стимул_3».

    GUI шлёт ``f"{tile_id}|{event}"``: ``5|on``, ``5|off``,
    а также ``-1|trial_start|target=...``, ``-2|trial_end``.

    Для P300 берём только вспышку ``|on``; ``|off`` и служебные id<0 пропускаем.
    """
    mv = marker_value

    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]

    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")

    if isinstance(mv, (int, np.integer)):
        tile_id = decode_stim_tile_id(int(mv))
        return f"стимул_{tile_id}" if tile_id is not None else None

    if isinstance(mv, (float, np.floating)):
        tile_id = decode_stim_tile_id(int(round(float(mv))))
        return f"стимул_{tile_id}" if tile_id is not None else None

    if isinstance(mv, str):
        s = mv.strip()
        if not s:
            return None
        if "|" in s:
            left, right = s.split("|", 1)
            left, right = left.strip(), right.strip()
            try:
                raw_id = int(left)
            except ValueError:
                raw_id = None
            if raw_id is not None and raw_id < 0:
                return None
            first_seg = right.split("|", 1)[0].strip()
            if raw_id is not None:
                tile_id = decode_stim_tile_id(raw_id)
                if tile_id is None:
                    return None
                if first_seg == "on":
                    return f"стимул_{tile_id}"
                if first_seg == "off":
                    return None
                if right.startswith("trial_start") or right.startswith("trial_end"):
                    return None
                return None
        m = re.search(r"(\d+)", s)
        if m:
            return f"стимул_{int(m.group(1))}"
        return s

    return str(mv)


def parse_trial_target_tile_id(marker_value: Any) -> Optional[int]:
    """Из маркера ``-1|trial_start|target=N`` извлекает N (id плитки 0..8)."""
    mv = marker_value
    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]
    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")
    if not isinstance(mv, str):
        return None
    s = mv.strip()
    if "trial_start" not in s:
        return None
    m = re.search(r"target[=:](\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def parse_trial_end(marker_value: Any) -> bool:
    """True если маркер соответствует окончанию trial: ``-2|trial_end``."""
    mv = marker_value
    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]
    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")
    if not isinstance(mv, str):
        return False
    s = mv.strip()
    return "trial_end" in s


def parse_trial_config_payload(marker_value: Any) -> Optional[Dict[str, str]]:
    """Parse marker ``-3|trial_config|k=v;...`` into a dict.

    Returns None if marker is not a trial_config marker.
    """
    mv = marker_value
    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]
    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")
    if not isinstance(mv, str):
        return None
    s = mv.strip()
    if "trial_config|" not in s:
        return None
    payload = s.split("trial_config|", 1)[1].strip()
    if not payload:
        return {}
    out: Dict[str, str] = {}
    for part in payload.split(";"):
        p = part.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def stim_key_sort_key(stim_key: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", stim_key)
    if m:
        return int(m.group(1)), stim_key
    return 10**9, stim_key


def stim_key_to_tile_digit(stim_key: str) -> int:
    m = re.search(r"(\d+)", stim_key)
    return int(m.group(1)) if m else -1
