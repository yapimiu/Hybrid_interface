"""Ограниченный кольцевой буфер samples × channels в локальном времени LSL."""

import numpy as np


class WaitingForEEG(Exception):
    pass


class EpochRejected(ValueError):
    pass


class EEGBuffer:
    def __init__(self, sfreq, channel_count, seconds):
        self.sfreq = sfreq
        self.capacity = int(np.ceil(sfreq * seconds))
        self.samples = np.empty((self.capacity, channel_count), dtype=np.float64)
        self.timestamps = np.empty(self.capacity, dtype=np.float64)
        self.count = self.position = 0
        self.last_timestamp = None

    def append(self, samples, timestamps):
        samples, timestamps = np.asarray(samples, float), np.asarray(timestamps, float)
        if (samples.ndim != 2 or samples.shape[1] != self.samples.shape[1]
                or timestamps.shape != (len(samples),)):
            raise ValueError("Неверная форма EEG/timestamps")
        if not len(samples):
            return
        if (not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0)
                or (self.last_timestamp is not None and timestamps[0] <= self.last_timestamp)):
            raise ValueError("EEG timestamps должны строго возрастать")
        self.last_timestamp = float(timestamps[-1])
        samples, timestamps = samples[-self.capacity:], timestamps[-self.capacity:]
        n = len(samples)
        first = min(n, self.capacity - self.position)
        self.samples[self.position:self.position + first] = samples[:first]
        self.timestamps[self.position:self.position + first] = timestamps[:first]
        if first < n:
            self.samples[:n - first] = samples[first:]
            self.timestamps[:n - first] = timestamps[first:]
        self.position = (self.position + n) % self.capacity
        self.count = min(self.capacity, self.count + n)

    def snapshot(self):
        start = (self.position - self.count) % self.capacity
        indices = (start + np.arange(self.count)) % self.capacity
        return self.samples[indices].copy(), self.timestamps[indices].copy()

    def window(self, timestamp, first_offset, last_offset, max_gap=1.5, max_jitter=0.5):
        data, times = self.snapshot()
        if not len(times) or times[-1] < timestamp + last_offset / self.sfreq:
            raise WaitingForEEG()
        if times[0] > timestamp + first_offset / self.sfreq + 0.5 / self.sfreq:
            raise EpochRejected("Недостаточно предыстории/маркер опоздал за пределы буфера")
        insertion = int(np.searchsorted(times, timestamp))
        candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(times)]
        onset = min(candidates, key=lambda i: abs(times[i] - timestamp))
        if abs(times[onset] - timestamp) > 0.75 / self.sfreq:
            raise EpochRejected("Нет EEG-отсчёта вблизи стимула")
        start, stop = onset + first_offset, onset + last_offset + 1
        if start < 0:
            raise EpochRejected("Недостаточно предыстории")
        if stop > len(times):
            raise WaitingForEEG()
        selected = times[start:stop]
        expected = times[onset] + np.arange(first_offset, last_offset + 1) / self.sfreq
        if (np.any(np.diff(selected) > max_gap / self.sfreq)
                or np.max(np.abs(selected - expected)) > max_jitter / self.sfreq):
            raise EpochRejected("Разрыв/нерегулярная частота EEG внутри эпохи")
        return data[start:stop], selected, float(times[onset] - timestamp)
