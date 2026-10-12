"""Последовательный координатор ядра. Все методы вызываются одним worker-потоком."""

from collections import deque
import math

import numpy as np

from .buffer import EEGBuffer, EpochRejected, WaitingForEEG
from .model import CalibratedModel, train_models
from .processing import EpochProcessor, ERPAverage
from .scoring import ScoreAccumulator
from .storage import ExperimentStore, load_epochs, save_epochs, write_json


class P300Engine:
    def __init__(self, config):
        self.config = config
        self.buffer = EEGBuffer(config.sfreq, len(config.channel_names), config.buffer_s)
        self.processor = EpochProcessor(config)
        self.erp = ERPAverage()
        self.scores = ScoreAccumulator(config.min_repetitions, config.score_margin)
        self.pending = []
        self.seen = set()
        self.seen_order = deque()
        self.epochs = []
        self.models = {}
        self.algorithm = "shrinkage_lda"
        self.calibrating = False
        self.marker_available = False
        self.store = None
        self.accepted = self.rejected = 0
        self.latest_result = None
        self.last_diagnostic = "Ожидание LSL-маркеров"
        self.report = None
        self.calibration_revision = 0

    @property
    def status(self):
        if not self.marker_available:
            return "Нет потока LSL-маркеров; классификация приостановлена"
        if self.calibrating:
            targets = sum(e.event.target is True for e in self.epochs)
            return f"Калибровка: {len(self.epochs)} эпох (target: {targets}), ожидают EEG: {len(self.pending)}"
        if self.algorithm not in self.models:
            return "Нет откалиброванной модели; доступен просмотр ERP"
        return "Модель загружена; ожидание/обработка стимулов (исследовательский результат)"

    def set_marker_available(self, available):
        self.marker_available = available
        if not available:
            for event, _ in self.pending:
                self._reject(event, "Поток маркеров отключён")
            self.pending.clear()
            self.scores = ScoreAccumulator(self.config.min_repetitions, self.config.score_margin)
            self.latest_result = None

    def start_experiment(self, path, metadata=None):
        if self.store is not None:
            raise ValueError("Эксперимент уже записывается")
        self.store = ExperimentStore(path, self.config, metadata)
        self.calibration_revision += 1
        # A saved dataset must correspond only to this recorded session.
        self.epochs.clear()
        self.erp = ERPAverage()
        self.calibrating = False
        self.scores = ScoreAccumulator(self.config.min_repetitions, self.config.score_margin)
        self.latest_result = None
        for event, _ in self.pending:
            self._reject(event, "Начата новая запись эксперимента")
        self.pending.clear()

    def stop_experiment(self, complete=True):
        if self.store is None:
            return
        store, self.store = self.store, None
        try:
            save_epochs(store.path / "epochs.npz", self.epochs, self.config)
            if self.report is not None:
                write_json(store.path / "calibration_report.json", self.report)
            for name, model in self.models.items():
                model.save(store.path / f"{name}.npz")
            snapshot = self.erp.snapshot()
            with (store.path / "erp.npz").open("wb") as file:
                np.savez_compressed(file, times=self.erp.times if self.erp.times is not None else [],
                                    objects=np.array(list(snapshot)),
                                    counts=np.array([v["count"] for v in snapshot.values()]),
                                    averages=np.array([v["data"] for v in snapshot.values()]))
        except Exception:
            store.close(complete=False)
            raise
        store.close(complete=complete)

    def start_calibration(self):
        self.calibration_revision += 1
        for event, _ in self.pending:
            self._reject(event, "Начат новый этап калибровки")
        self.pending.clear()
        self.epochs.clear()
        self.models.clear()
        self.report = None
        self.erp = ERPAverage()
        self.scores = ScoreAccumulator(self.config.min_repetitions, self.config.score_margin)
        self.latest_result = None
        self.calibrating = True
        if self.store:
            self.store.result({"type": "calibration_started"})

    def train(self, algorithm="shrinkage_lda"):
        if self.pending:
            raise ValueError("Есть незавершённые эпохи: остановите стимулы и дождитесь EEG с padding")
        models, report = train_models(self.epochs, self.config)
        self.apply_training(models, report, algorithm)
        return report

    def apply_training(self, models, report, algorithm):
        self.models, self.report = models, report
        self.select_algorithm(algorithm)
        self.calibrating = False
        if self.store:
            write_json(self.store.path / "calibration_report.json", report)
            save_epochs(self.store.path / "epochs.npz", self.epochs, self.config)
            for name, model in models.items():
                model.save(self.store.path / f"{name}.npz")
            self.store.result({"type": "trained", "algorithm": algorithm,
                               "dataset_sha256": report["dataset_sha256"]})

    def select_algorithm(self, algorithm):
        from .model import ALGORITHMS
        if algorithm not in ALGORITHMS:
            raise ValueError("Неизвестный алгоритм")
        self.algorithm = algorithm
        self.scores = ScoreAccumulator(self.config.min_repetitions, self.config.score_margin)
        self.latest_result = None
        if self.store:
            self.store.result({"type": "algorithm_selected", "algorithm": algorithm})

    def load_model(self, path):
        model = CalibratedModel.load(path, self.config)
        self.calibration_revision += 1
        self.models[model.algorithm] = model
        self.report = model.metadata
        self.select_algorithm(model.algorithm)
        self.calibrating = False
        if self.store:
            self.store.result({"type": "model_loaded", "algorithm": model.algorithm,
                               "report": model.metadata})

    def load_calibration(self, path):
        if self.store:
            raise ValueError("Импорт калибровки доступен вне текущей записи эксперимента")
        epochs, _ = load_epochs(path, self.config)
        self.start_calibration()
        self.epochs = epochs
        for epoch in epochs:
            self.erp.add(epoch)

    def ingest_eeg(self, samples, source_timestamps, clock_offset):
        if not math.isfinite(clock_offset):
            raise ValueError("Нет достоверной коррекции часов EEG")
        source_timestamps = np.asarray(source_timestamps, float)
        self.buffer.append(samples, source_timestamps + clock_offset)
        if self.store:
            self.store.append_eeg(samples, source_timestamps, clock_offset)
        self._drain()

    def ingest_marker(self, event):
        if self.store:
            self.store.event(event.to_dict())
        if not self.marker_available:
            self._reject(event, "Нет подключённого потока маркеров")
            return
        if event.event_id in self.seen:
            self._reject(event, "Повторный event_id")
            return
        self.seen.add(event.event_id)
        self.seen_order.append(event.event_id)
        if len(self.seen_order) > 100000:
            self.seen.remove(self.seen_order.popleft())
        if len(self.pending) >= self.config.max_pending:
            self._reject(event, "Переполнена очередь маркеров")
            return
        if self.calibrating and event.target is None:
            self._reject(event, "В калибровке требуется явная target-метка")
            return
        self.pending.append((event, self.calibrating))
        self.pending.sort(key=lambda item: item[0].timestamp)
        self._drain()

    def _reject(self, event, reason):
        self.rejected += 1
        self.last_diagnostic = reason
        if self.store:
            self.store.result({"type": "rejected", "event_id": event.event_id, "reason": reason})

    def _drain(self):
        remaining = []
        for event, calibration in self.pending:
            try:
                epoch = self.processor.extract(self.buffer, event)
            except WaitingForEEG:
                remaining.append((event, calibration))
                continue
            except EpochRejected as exc:
                self._reject(event, str(exc))
                continue
            if event.object_id not in self.erp.sums and len(self.erp.sums) >= 256:
                self._reject(event, "Превышен предел 256 объектов ERP")
                continue
            if calibration:
                if len(self.epochs) >= self.config.max_calibration_epochs:
                    self._reject(event, "Достигнут предел калибровочных эпох")
                    continue
                self.epochs.append(epoch)
            self.accepted += 1
            self.erp.add(epoch)
            self.last_diagnostic = f"Принята эпоха {event.event_id}"
            if self.store:
                self.store.result({"type": "accepted", "event_id": event.event_id,
                                   "calibration": calibration, "source_interval": epoch.source_interval,
                                   "alignment_error_s": epoch.alignment_error_s})
            if not calibration and self.marker_available and self.algorithm in self.models:
                score = float(self.models[self.algorithm].decision_function(epoch.data)[0])
                self.latest_result = self.scores.add(event, score)
                if self.store:
                    self.store.result({"type": "prediction", "algorithm": self.algorithm,
                                       **self.latest_result})
        self.pending = remaining

    def snapshot(self):
        return {"status": self.status, "diagnostic": self.last_diagnostic,
                "accepted": self.accepted, "rejected": self.rejected, "pending": len(self.pending),
                "calibration_count": len(self.epochs), "calibrating": self.calibrating,
                "recording": str(self.store.path) if self.store else None,
                "algorithm": self.algorithm, "models": list(self.models),
                "report": self.report,
                "result": self.latest_result, "erp": self.erp.snapshot(), "times": self.erp.times}

    def close(self):
        for event, _ in self.pending:
            self._reject(event, "Эксперимент завершён до получения полной эпохи")
        self.pending.clear()
        self.stop_experiment()
