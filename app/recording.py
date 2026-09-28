"""Запись сырых отсчётов LSL в CSV без накопления всей записи в памяти."""

import csv
from pathlib import Path

import numpy as np


class CsvRecording:
    def __init__(self):
        self.path: Path | None = None
        self.sample_count = 0
        self.channel_count = 0
        self._file = None
        self._writer = None

    @property
    def active(self) -> bool:
        return self._file is not None

    def start(self, path: str, channel_names: list[str]):
        if self.active:
            raise RuntimeError("Запись уже идёт")
        file = open(path, "w", newline="", encoding="utf-8-sig")
        try:
            writer = csv.writer(file)
            writer.writerow(["lsl_timestamp_s", *channel_names])
            file.flush()
        except Exception:
            file.close()
            raise
        self.path = Path(path)
        self.sample_count = 0
        self.channel_count = len(channel_names)
        self._file = file
        self._writer = writer

    def append(self, samples: np.ndarray, timestamps: np.ndarray):
        if not self.active:
            return
        if samples.ndim != 2 or samples.shape[1] != self.channel_count:
            raise ValueError("Изменилось количество каналов LSL")
        if timestamps.ndim != 1 or len(timestamps) != len(samples):
            raise ValueError("Количество временных меток не совпадает с числом отсчётов")
        self._writer.writerows(
            [float(timestamp), *sample] for timestamp, sample in zip(timestamps, samples)
        )
        self._file.flush()
        self.sample_count += len(samples)

    def stop(self):
        file = self._file
        self._file = None
        self._writer = None
        if file is not None:
            file.close()
