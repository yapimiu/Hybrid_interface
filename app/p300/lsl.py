"""Отдельный приёмник нерегулярных JSON-маркеров визуальных стимулов."""

import threading

from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop

from .markers import Stimulus


def stream_metadata(info):
    return {"name": info.name(), "type": info.type(), "source_id": info.source_id(),
            "uid": info.uid(), "nominal_srate": info.nominal_srate(),
            "channel_count": info.channel_count(), "xml": info.as_xml()}


class MarkerReceiver(QThread):
    marker_received = pyqtSignal(object)
    state_changed = pyqtSignal(bool, object)
    diagnostic = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._revision = 0
        self._lock = threading.Lock()

    def select_stream(self, name):
        with self._lock:
            self._name = name.strip()
            self._revision += 1

    def _selection(self):
        with self._lock:
            return self._name, self._revision

    def run(self):
        self.state_changed.emit(False, {})
        last_diagnostic = None
        while not self.isInterruptionRequested():
            inlet = None
            name, revision = self._selection()
            try:
                streams = []
                for kind in ("Markers", "Marker"):
                    if self.isInterruptionRequested():
                        return
                    streams.extend(resolve_byprop("type", kind, timeout=0.25))
                streams = list({s.uid(): s for s in streams if not name or s.name() == name}.values())
                if len(streams) != 1:
                    message = ("Нет LSL-потока Markers/Marker" if not streams else
                               "Найдено несколько потоков маркеров; укажите точное имя")
                    if message != last_diagnostic:
                        self.diagnostic.emit(message)
                        last_diagnostic = message
                    continue
                inlet = StreamInlet(streams[0], max_buflen=60, recover=False)
                inlet.open_stream(timeout=1.0)
                info = inlet.info(timeout=1.0)
                if info.channel_count() != 1:
                    raise ValueError("Ожидается один строковый канал JSON-маркеров")
                inlet.time_correction(timeout=1.0)
                metadata = stream_metadata(info)
                self.state_changed.emit(True, metadata)
                self.diagnostic.emit(f"Маркеры: {info.name()} (часы синхронизированы)")
                last_diagnostic = None
                while not self.isInterruptionRequested() and self._selection()[1] == revision:
                    sample, timestamp = inlet.pull_sample(timeout=0.1)
                    if sample is None:
                        continue
                    try:
                        offset = inlet.time_correction(timeout=0.1)
                        marker = Stimulus.parse(sample[0], float(timestamp), float(offset))
                    except (ValueError, TypeError, KeyError) as exc:
                        self.diagnostic.emit(f"Отклонён LSL-маркер: {exc}")
                        continue
                    self.marker_received.emit(marker)
            except Exception as exc:
                message = f"Маркеры недоступны: {exc}"
                if message != last_diagnostic:
                    self.diagnostic.emit(message)
                    last_diagnostic = message
                self.msleep(100)
            finally:
                if inlet:
                    try:
                        inlet.close_stream()
                    except Exception:
                        pass
                self.state_changed.emit(False, {})
