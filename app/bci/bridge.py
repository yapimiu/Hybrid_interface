"""Transport adapters only; upstream timestamps and numerical algorithms are preserved."""
from collections import deque
import threading

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop
from p300_analysis.lsl_streams import probe_stimulus_marker_stream


class ObservedMarkerInlet:
    """Observe exactly the marker batches delivered to the original analyzer."""
    def __init__(self, inlet, observer):
        self.inlet, self.observer = inlet, observer

    def info(self, *args, **kwargs):
        return self.inlet.info(*args, **kwargs)

    def pull_chunk(self, *args, **kwargs):
        samples, timestamps = self.inlet.pull_chunk(*args, **kwargs)
        for sample, timestamp in zip(samples, timestamps):
            value = sample[0] if isinstance(sample, (list, tuple)) else sample
            self.observer(str(value), float(timestamp))
        return samples, timestamps

    def close_stream(self):
        self.inlet.close_stream()


class SharedEEGInlet:
    """The upstream inlet interface backed by the viewer's existing EEG receiver."""
    def __init__(self, info, scale=1.0):
        self._info = info
        self.scale = float(scale)
        self.limit = max(4096, int(max(1, info.nominal_srate()) * 10))
        self._chunks = deque()
        self._count = 0
        self._closed = False
        self._overflow = False
        self._lock = threading.Lock()

    def info(self, timeout=None):
        return self._info

    def push(self, samples, timestamps):
        with self._lock:
            if self._closed:
                return
            if self._count + len(timestamps) > self.limit:
                self._overflow = True
                return
            self._chunks.append((samples, timestamps))
            self._count += len(timestamps)

    def pull_chunk(self, timeout=0.0, max_samples=2048):
        arrays, times = [], []
        with self._lock:
            if self._overflow:
                raise RuntimeError('Переполнение буфера P300. Анализ остановлен; исходная CSV-запись независима.')
            remaining = int(max_samples)
            while self._chunks and remaining > 0 and not self._closed:
                samples, timestamps = self._chunks.popleft()
                take = min(len(timestamps), remaining)
                arrays.append(np.asarray(samples[:take], dtype=float) * self.scale)
                times.extend(np.asarray(timestamps[:take], dtype=float).tolist())
                if take < len(timestamps):
                    self._chunks.appendleft((samples[take:], timestamps[take:]))
                self._count -= take
                remaining -= take
        return (np.concatenate(arrays).tolist(), times) if arrays else ([], [])

    def close_stream(self):
        with self._lock:
            self._closed = True
            self._chunks.clear()
            self._count = 0


class ConnectMarkers(QThread):
    connected = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name.strip()

    def run(self):
        inlet = None
        try:
            if self.name:
                matches = resolve_byprop('name', self.name, timeout=1)
                matches = [s for s in matches if s.channel_count() == 1]
                if len(matches) != 1:
                    raise ValueError('Нужен один поток маркеров с именем ' + self.name)
                info = matches[0]
            else:
                info, _ = probe_stimulus_marker_stream(timeout=.7)
                if info is None:
                    raise ValueError('Не найден поток плиток BCI_StimMarkers')
            if self.isInterruptionRequested():
                return
            inlet = StreamInlet(info, max_buflen=60, recover=False)
            inlet.open_stream(timeout=1)
            if self.isInterruptionRequested():
                inlet.close_stream()
            else:
                self.connected.emit(inlet)
                inlet = None  # ownership transfers to the GUI
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if inlet is not None:
                inlet.close_stream()
