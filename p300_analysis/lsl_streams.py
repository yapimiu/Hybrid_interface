"""Поиск потоков LSL и создание StreamInlet с совместимым буфером."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Set, Tuple

from pylsl import StreamInfo, StreamInlet, resolve_byprop, resolve_streams

from p300_analysis.constants import EEG_STREAM_TYPES, NEUROSPECTR_MARKER, SIMULATOR_NAME, SIMULATOR_SOURCE_ID

# PsychoPy / gui (core.lsl.LslMarkerSender)
BCI_STIM_MARKER_STREAM_NAME = "BCI_StimMarkers"
MIGALKA_MARKER_STREAM_NAME = "MigalkaStimMarkers"


def _is_allowed_stream(info: StreamInfo) -> bool:
    try:
        name = (info.name() or "").strip().lower()
        sid = (info.source_id() or "").strip().lower()
    except Exception:
        return False
    if name == SIMULATOR_NAME.lower() or SIMULATOR_SOURCE_ID in sid:
        return True
    if NEUROSPECTR_MARKER in name or NEUROSPECTR_MARKER in sid:
        return True
    return False


def find_allowed_eeg_streams(timeout: float = 3.0) -> List[StreamInfo]:
    all_streams: List[StreamInfo] = []
    for stream_type in EEG_STREAM_TYPES:
        try:
            streams = resolve_byprop("type", stream_type, timeout=timeout)
            all_streams.extend(streams)
        except Exception:
            pass
    return [s for s in all_streams if _is_allowed_stream(s)]


def discover_all_eeg_streams(timeout: float = 1.0) -> List[StreamInfo]:
    """Все потоки типа EEG/Signal (без фильтра устройства) — для выбора в GUI."""
    merged: List[StreamInfo] = []
    seen: Set[Tuple[str, str]] = set()
    for stream_type in EEG_STREAM_TYPES:
        try:
            batch = list(resolve_byprop("type", stream_type, timeout=float(timeout)))
        except Exception:
            batch = []
        _append_unique_streams(merged, batch, seen)
    return merged


def discover_eeg_streams(timeout: float = 1.0) -> List[StreamInfo]:
    """Разрешённые устройства + остальные EEG/Signal (для списка в операторском GUI)."""
    merged: List[StreamInfo] = []
    seen: Set[Tuple[str, str]] = set()
    _append_unique_streams(merged, find_allowed_eeg_streams(timeout=timeout), seen)
    _append_unique_streams(merged, discover_all_eeg_streams(timeout=timeout), seen)
    return merged


def stream_display_label(info: StreamInfo) -> str:
    try:
        name = info.name() or "?"
        ch = int(info.channel_count())
        fs = float(info.nominal_srate() or 0.0)
        stype = info.type() or "?"
    except Exception:
        return "?"
    fs_s = f"{fs:g} Гц" if fs > 1.0 else "Гц ?"
    return f"{name} ({ch} кан., {fs_s}, {stype})"


def select_eeg_stream(
    streams: List[StreamInfo],
    *,
    name: str,
    session_id: str = "",
) -> Optional[StreamInfo]:
    if not streams:
        return None
    want_name = (name or "").strip()
    want_sid = (session_id or "").strip()
    if not want_name:
        return streams[0]
    for s in streams:
        try:
            if (s.name() or "") == want_name and (not want_sid or (s.session_id() or "") == want_sid):
                return s
        except Exception:
            continue
    for s in streams:
        try:
            if (s.name() or "") == want_name:
                return s
        except Exception:
            continue
    # NeuronSpectrum и др.: имя потока часто с датой/временем — допускаем префикс.
    if want_name:
        prefix = want_name.lower()
        for s in streams:
            try:
                n = (s.name() or "").strip()
                if n.lower().startswith(prefix):
                    return s
            except Exception:
                continue
    return None


def _marker_like_stream(info: StreamInfo) -> bool:
    """Эвристика: поток похож на маркеры стимуляции (не только type=Markers)."""
    try:
        stype = (info.type() or "").strip().lower()
        name = (info.name() or "").strip().lower()
    except Exception:
        return False
    if stype == "markers":
        return True
    if "marker" in stype or "stim" in name or "bci" in name:
        return True
    return False


def _append_unique_streams(
    target: List[StreamInfo],
    batch: List[StreamInfo],
    seen: Set[Tuple[str, str]],
) -> None:
    for s in batch:
        try:
            key = (s.name() or "", s.session_id() or "")
        except Exception:
            key = (str(s), "")
        if key not in seen:
            seen.add(key)
            target.append(s)


def select_migalka_marker_stream(streams: List[StreamInfo]) -> Optional[StreamInfo]:
    """Поток MigalkaStimMarkers (мигалка SSVEP), как в ssvep_analyzer."""
    if not streams:
        return None
    for s in streams:
        try:
            if (s.name() or "") == MIGALKA_MARKER_STREAM_NAME:
                return s
        except Exception:
            continue
    for s in streams:
        try:
            name = (s.name() or "").lower()
        except Exception:
            continue
        if MIGALKA_MARKER_STREAM_NAME.lower() in name or "migalka" in name:
            return s
    return None


def select_stimulus_marker_stream(streams: List[StreamInfo]) -> Optional[StreamInfo]:
    """Поток маркеров плиток (PsychoPy), не мигалка SSVEP."""
    if not streams:
        return None
    for s in streams:
        try:
            if (s.name() or "") == BCI_STIM_MARKER_STREAM_NAME:
                return s
        except Exception:
            continue
    for s in streams:
        try:
            name = (s.name() or "").lower()
        except Exception:
            continue
        if MIGALKA_MARKER_STREAM_NAME.lower() in name or "migalka" in name:
            continue
        if "bci" in name and "stim" in name:
            return s
        if "stim" in name and "marker" in (s.type() or "").lower():
            return s
    return None


def probe_stimulus_marker_stream(
    *,
    timeout: float = 0.7,
) -> Tuple[Optional[StreamInfo], List[StreamInfo]]:
    """Non-blocking LSL probe for BCI_StimMarkers (one resolve pass, no sleep)."""
    streams = resolve_marker_streams(timeout=float(timeout), attempts=1)
    return select_stimulus_marker_stream(streams), streams


def wait_for_stimulus_marker_stream(
    *,
    max_wait_sec: float = 20.0,
    poll_interval_sec: float = 0.4,
) -> Tuple[Optional[StreamInfo], List[StreamInfo]]:
    """Ждёт появления BCI_StimMarkers (стимулятор PsychoPy), не MigalkaStimMarkers."""
    import time

    deadline = time.time() + float(max_wait_sec)
    last: List[StreamInfo] = []
    while time.time() < deadline:
        last = resolve_marker_streams(timeout=0.7, attempts=1)
        picked = select_stimulus_marker_stream(last)
        if picked is not None:
            return picked, last
        time.sleep(float(poll_interval_sec))
    return select_stimulus_marker_stream(last), last


def resolve_marker_streams(
    timeout: float = 5.0,
    *,
    attempts: int = 2,
) -> List[StreamInfo]:
    """Поиск потоков маркеров (LSL discovery).

    Сначала несколько попыток resolve_byprop(type=Markers). Если пусто —
    resolve_streams (все потоки) и фильтр по типу/имени: на части LAN второй
    путь иногда находит поток, который не отвечает на узкий resolve.
    """
    merged: List[StreamInfo] = []
    seen: Set[Tuple[str, str]] = set()
    for _ in range(max(1, int(attempts))):
        try:
            batch = list(resolve_byprop("type", "Markers", timeout=float(timeout)))
        except Exception:
            batch = []
        _append_unique_streams(merged, batch, seen)
        if merged:
            return merged

    try:
        broad = list(resolve_streams(wait_time=float(timeout)))
    except Exception:
        broad = []
    markerish = [s for s in broad if _marker_like_stream(s)]
    _append_unique_streams(merged, markerish, seen)
    return merged


def stream_channel_labels(info: StreamInfo, count: int) -> List[str]:
    """Подписи каналов из LSL StreamInfo (как в p300_analysis.qt_window)."""
    labels: List[str] = []
    try:
        channels = info.desc().child("channels")
        ch = channels.child("channel")
        for i in range(count):
            if ch is None:
                break
            label = (
                ch.child_value("label")
                or ch.child_value("name")
                or ch.child_value("channel")
                or ""
            )
            label = str(label).strip()
            labels.append(label if label else f"Канал {i + 1}")
            nxt = ch.next_sibling()
            if nxt is None:
                break
            ch = nxt
    except Exception:
        labels = []
    if len(labels) < count:
        try:
            root = ET.fromstring(info.as_xml())
            for ch_el in root.findall(".//channels/channel"):
                if len(labels) >= count:
                    break
                label = (
                    (ch_el.findtext("label") or "").strip()
                    or (ch_el.findtext("name") or "").strip()
                    or (ch_el.findtext("channel") or "").strip()
                )
                labels.append(label if label else f"Канал {len(labels) + 1}")
        except Exception:
            pass
    if len(labels) < count:
        labels.extend([f"Канал {i + 1}" for i in range(len(labels), count)])
    return labels[:count]


def unwrap_combo_userdata(data: Any) -> Any:
    """QComboBox.itemData иногда отдаёт QVariant; pylsl ждёт «сырой» StreamInfo."""
    if data is None:
        return None
    try:
        from PyQt5.QtCore import QVariant

        if isinstance(data, QVariant):
            return data.value()
    except Exception:
        pass
    return data


def stream_inlet_with_buffer(info: StreamInfo, buffer_seconds: int) -> StreamInlet:
    """Создаёт inlet; разные сборки pylsl знают max_buffered или max_buflen."""
    try:
        return StreamInlet(info, max_buffered=buffer_seconds)
    except TypeError:
        pass
    try:
        return StreamInlet(info, max_buflen=buffer_seconds)
    except TypeError:
        pass
    return StreamInlet(info)
