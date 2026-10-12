"""Воспроизводимая обработка реальных архивов эксперимента, без Qt и LSL."""

import csv
import json
from pathlib import Path

import numpy as np

from .config import P300Config
from .engine import P300Engine
from .markers import Stimulus


def preprocess_session(path):
    """Выделить все явно размеченные эпохи из сохранённых raw EEG и событий."""
    root = Path(path)
    metadata = json.loads((root / "session.json").read_text(encoding="utf-8"))
    config = P300Config.from_dict(metadata["config"])
    events = [Stimulus(**json.loads(line)) for line in
              (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    events = sorted((e for e in events if e.target is not None), key=lambda e: e.timestamp)
    engine = P300Engine(config)
    engine.set_marker_available(True)
    engine.start_calibration()
    index = 0

    def ingest(rows):
        nonlocal index
        values = np.asarray(rows, float)
        corrected = values[:, 1]
        if not np.allclose(values[:, 0] + values[:, 2], corrected, rtol=0, atol=1e-7):
            raise ValueError("Несогласованные raw/corrected timestamps в архиве")
        while index < len(events) and events[index].timestamp <= corrected[-1]:
            engine.ingest_marker(events[index])
            index += 1
        # Reuse the original offset per row; split where inlet correction changed.
        boundaries = np.flatnonzero(np.diff(values[:, 2]) != 0) + 1
        for chunk in np.split(values, boundaries):
            engine.ingest_eeg(chunk[:, 3:], chunk[:, 0], float(chunk[0, 2]))

    with (root / "eeg.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        expected = ["source_timestamp_s", "local_timestamp_s", "clock_offset_s", *config.channel_names]
        if next(reader) != expected:
            raise ValueError("Неверные столбцы EEG-архива")
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= min(4096, max(1, int(config.sfreq))):
                ingest(rows)
                rows = []
        if rows:
            ingest(rows)
    for event in events[index:]:
        engine.ingest_marker(event)
    engine.close()
    return engine.epochs, config, {"accepted": engine.accepted, "rejected": engine.rejected,
                                   "last_diagnostic": engine.last_diagnostic}
