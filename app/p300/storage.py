"""Потоковая запись эксперимента и переносимый датасет принятых эпох."""

from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np

from .config import P300Config
from .markers import Stimulus
from .model import software_versions
from .processing import Epoch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_epochs(path, epochs, config):
    shape = epochs[0].data.shape if epochs else (len(config.channel_names), 0)
    with Path(path).open("wb") as file:
        np.savez_compressed(file, data=np.stack([e.data for e in epochs]) if epochs else np.empty((0, *shape)),
                            times=epochs[0].times if epochs else np.empty(0),
                            metadata=json.dumps({"schema": 1, "config": config.to_dict(),
                                "epochs": [{"event": e.event.to_dict(), "source_interval": e.source_interval,
                                            "alignment_error_s": e.alignment_error_s} for e in epochs]},
                                ensure_ascii=False))


def load_epochs(path, config=None):
    with np.load(path, allow_pickle=False) as file:
        meta = json.loads(str(file["metadata"]))
        saved_config = P300Config.from_dict(meta["config"])
        if meta["schema"] != 1 or (config is not None and saved_config != config):
            raise ValueError("Несовместимый датасет/конфигурация")
        data, times = file["data"], file["times"]
        if (data.ndim != 3 or data.shape != (len(meta["epochs"]), len(saved_config.channel_names), len(times))
                or len(data) > saved_config.max_calibration_epochs or not np.all(np.isfinite(data))):
            raise ValueError("Повреждённый датасет")
        epochs = [Epoch(array.copy(), times.copy(), Stimulus(**item["event"]),
                        tuple(item["source_interval"]), item["alignment_error_s"])
                  for array, item in zip(data, meta["epochs"])]
    return epochs, saved_config


class ExperimentStore:
    def __init__(self, path, config, stream_metadata=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.handles = []
        self.sample_count = 0
        self.closed = False
        try:
            self.metadata = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                             "config": config.to_dict(), "versions": software_versions(),
                             "stream": stream_metadata or {}, "complete": False}
            write_json(self.path / "session.json", self.metadata)
            self.eeg = self._open("eeg.csv", newline="")
            self.writer = csv.writer(self.eeg)
            self.writer.writerow(["source_timestamp_s", "local_timestamp_s", "clock_offset_s", *config.channel_names])
            self.events = self._open("events.jsonl")
            self.results = self._open("results.jsonl")
            self.eeg.flush()
        except Exception:
            self.close(complete=False)
            raise

    def _open(self, name, **kwargs):
        file = (self.path / name).open("w", encoding="utf-8", **kwargs)
        self.handles.append(file)
        return file

    def append_eeg(self, samples, source_timestamps, offset):
        self.writer.writerows([float(t), float(t + offset), float(offset), *sample]
                              for sample, t in zip(samples, source_timestamps))
        self.eeg.flush()
        self.sample_count += len(samples)

    def event(self, event):
        self.events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.events.flush()

    def result(self, value):
        self.results.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.results.flush()

    def close(self, complete=True):
        if self.closed:
            return
        self.closed = True
        errors = []
        for handle in self.handles:
            try:
                handle.close()
            except OSError as exc:
                errors.append(exc)
        if hasattr(self, "metadata"):
            self.metadata.update(complete=complete and not errors, sample_count=self.sample_count,
                                 ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(self.path / "session.json", self.metadata)
        if errors:
            raise errors[0]
