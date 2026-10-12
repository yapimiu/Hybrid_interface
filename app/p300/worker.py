"""Одна ограниченная очередь: вычисления/диск выполняются вне главного потока Qt."""

from queue import Empty, Full, Queue
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from PyQt5.QtCore import QThread, pyqtSignal


class P300Worker(QThread):
    updated = pyqtSignal(object)
    diagnostic = pyqtSignal(str)
    trained = pyqtSignal(object)
    busy = pyqtSignal(bool)

    def __init__(self, parent=None, queue_size=256):
        super().__init__(parent)
        self.commands = Queue(maxsize=queue_size)
        self.overflow = threading.Event()
        self.stopping = threading.Event()
        self.engine = None
        self.indices = None
        self.marker_available = False
        self.stream_metadata = {}
        self.faulted = False
        self.marker_metadata = {}
        self.generation = 0
        self._queue_lock = threading.Lock()

    def submit(self, action, *args):
        if self.stopping.is_set():
            return False
        with self._queue_lock:
            try:
                self.commands.put_nowait((self.generation, action, args))
            except Full:
                self.generation += 1
                self.overflow.set()
                return False
        return True

    def submit_eeg(self, samples, timestamps, offset):
        # LSL owns new arrays for each block; no filtering or disk I/O here.
        self.submit("eeg", samples, timestamps, offset)

    def submit_marker(self, event):
        self.submit("marker", event)

    def marker_state(self, available, metadata):
        self.submit("marker_state", available, metadata)

    def shutdown(self):
        self.stopping.set()

    def _fault(self, message):
        self.faulted = True
        self.diagnostic.emit(message + ". P300 остановлен; нажмите «Применить» для нового запуска.")
        if self.engine:
            self.engine.calibration_revision += 1
            try:
                self.engine.stop_experiment(complete=False)
            except Exception as exc:
                self.diagnostic.emit(f"Ошибка завершения экспериментальной записи: {exc}")
            self.engine.set_marker_available(False)

    def run(self):
        from threadpoolctl import threadpool_limits
        from .engine import P300Engine
        from .model import train_models

        def fit(epochs, config):
            with threadpool_limits(limits=1):
                return train_models(epochs, config)

        last_update = 0.0
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p300-calibration")
        training = None
        training_origin = None
        stop_after_training = None
        retired_engines = set()

        def retire_engine():
            if self.engine is None:
                return
            if training is not None and training_origin[0] is self.engine:
                self.engine.set_marker_available(False)
                retired_engines.add(self.engine)
            else:
                self.engine.close()
        try:
            while not self.stopping.is_set() or not self.commands.empty() or training is not None:
                if self.overflow.is_set():
                    self.overflow.clear()
                    self._fault("Переполнение очереди: потеря данных, непрерывность не гарантирована")
                try:
                    generation, action, args = self.commands.get(timeout=0.05)
                    if generation != self.generation:
                        action, args = None, ()
                except Empty:
                    action, args = None, ()
                try:
                    if action == "configure":
                        retire_engine()
                        c, self.indices, self.stream_metadata = args
                        self.engine = P300Engine(c)
                        self.engine.set_marker_available(self.marker_available)
                        self.faulted = False
                    elif action == "disconnect":
                        retire_engine()
                        self.engine = None
                        self.updated.emit(None)
                        self.diagnostic.emit("Нет EEG-потока. После подключения примените параметры P300")
                    elif action == "marker_state":
                        self.marker_available, self.marker_metadata = args
                        if self.engine and not self.faulted:
                            self.engine.set_marker_available(self.marker_available)
                            if self.engine.store:
                                self.engine.store.result({"type": "marker_connection", "connected": args[0],
                                                          "stream": self.marker_metadata})
                    elif action == "diagnostic":
                        self.diagnostic.emit(args[0])
                        if self.engine and self.engine.store:
                            self.engine.store.result({"type": "diagnostic", "message": args[0]})
                    elif action == "sync_error":
                        if self.engine:
                            self._fault(args[0])
                        else:
                            self.diagnostic.emit(args[0])
                    elif self.engine is not None and not self.faulted:
                        if action == "eeg":
                            samples, timestamps, offset = args
                            self.engine.ingest_eeg(samples[:, self.indices], timestamps, offset)
                        elif action == "marker":
                            self.engine.ingest_marker(args[0])
                        elif action == "start_experiment":
                            self.engine.start_experiment(args[0], {"eeg": self.stream_metadata,
                                                                    "markers": self.marker_metadata})
                        elif action == "stop_experiment":
                            if training is not None and training_origin[0] is self.engine:
                                stop_after_training = self.engine
                                self.diagnostic.emit("Эксперимент будет сохранён после завершения обучения")
                            else:
                                self.engine.close()
                        elif action == "start_calibration":
                            self.engine.start_calibration()
                        elif action == "train":
                            if training is not None:
                                raise ValueError("Обучение уже выполняется")
                            if self.engine.pending:
                                raise ValueError("Остановите стимулы и дождитесь завершения эпох с padding")
                            self.diagnostic.emit("Обучение и проверка двух моделей в фоновом потоке…")
                            self.engine.select_algorithm(args[0])
                            training_origin = (self.engine, self.engine.calibration_revision, args[0])
                            training = executor.submit(fit, list(self.engine.epochs), self.engine.config)
                            self.engine.calibrating = False
                            self.busy.emit(True)
                        elif action == "algorithm":
                            self.engine.select_algorithm(*args)
                        elif action == "load_model":
                            self.engine.load_model(*args)
                        elif action == "load_calibration":
                            self.engine.load_calibration(*args)
                    elif action not in (None, "eeg", "marker"):
                        self.diagnostic.emit("Сначала подключите EEG и примените параметры P300")
                except Exception as exc:
                    if action in {"eeg", "marker"} or isinstance(exc, OSError):
                        self._fault(f"Ошибка P300: {exc}")
                    else:
                        self.diagnostic.emit(f"P300: {exc}")
                if training is not None and training.done():
                    origin, revision, algorithm = training_origin
                    try:
                        models, report = training.result()
                        if origin.calibration_revision == revision:
                            origin.apply_training(models, report, algorithm)
                            if self.engine is origin and not self.faulted:
                                self.trained.emit(report)
                                self.diagnostic.emit("Обе модели обучены. Доступны онлайн-оценки.")
                    except Exception as exc:
                        if isinstance(exc, OSError) and self.engine is origin:
                            self._fault(f"Ошибка сохранения обученной модели: {exc}")
                        elif self.engine is origin and origin.calibration_revision == revision:
                            origin.calibrating = True
                        self.diagnostic.emit(f"Обучение P300 не завершено: {exc}")
                    training = None
                    self.busy.emit(False)
                    if origin in retired_engines:
                        try:
                            origin.close()
                        except Exception as exc:
                            self.diagnostic.emit(f"Ошибка сохранения прежнего эксперимента: {exc}")
                        retired_engines.remove(origin)
                    if stop_after_training is not None:
                        try:
                            stop_after_training.close()
                        except Exception as exc:
                            self._fault(f"Ошибка завершения эксперимента: {exc}")
                        stop_after_training = None
                now = time.monotonic()
                if now - last_update >= 0.25:
                    if self.engine:
                        snapshot = self.engine.snapshot()
                        if self.faulted:
                            snapshot["status"] = "P300 остановлен после ошибки; примените параметры заново"
                        self.updated.emit(snapshot)
                    last_update = now
        finally:
            executor.shutdown(wait=True)
            for engine in retired_engines:
                try:
                    engine.close()
                except Exception as exc:
                    self.diagnostic.emit(f"Ошибка сохранения прежнего эксперимента: {exc}")
            if self.engine:
                try:
                    self.engine.close()
                except Exception as exc:
                    self.diagnostic.emit(f"Ошибка сохранения при закрытии P300: {exc}")
