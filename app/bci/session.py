"""Asynchronous, append-only research session archive; no signal processing here."""
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from app.recording import CsvRecording
from .erp_sums import save_sums


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class SessionWriter(QThread):
    ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, directory, metadata, parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self.metadata = dict(metadata)
        self.queue = Queue(maxsize=256)
        self.overflow = threading.Event()
        self.closing = threading.Event()
        self.finish_info = {'reason': 'interrupted', 'protocol_complete': False}

    def submit(self, kind, payload):
        if self.closing.is_set():
            return
        try:
            self.queue.put_nowait((kind, payload))
        except Full:
            self.overflow.set()

    def eeg(self, samples, timestamps):
        self.submit('eeg', (samples, timestamps))

    def finish(self, reason, protocol_complete=False):
        self.finish_info = {'reason': str(reason), 'protocol_complete': bool(protocol_complete)}
        self.closing.set()

    def _metadata(self, recorder, complete, error=''):
        data = {**self.metadata, **self.finish_info, 'recording_complete': complete,
                'error': error, 'eeg_samples': recorder.sample_count,
                'finished_utc': datetime.now(timezone.utc).isoformat() if self.closing.is_set() else None}
        (self.directory / 'session.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')

    def run(self):
        recorder = CsvRecording()
        events = None
        error = ''
        created = False
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            created = True
            recorder.start(str(self.directory / 'eeg.csv'), self.metadata['channels'])
            events = (self.directory / 'events.jsonl').open('w', encoding='utf-8')
            self._metadata(recorder, False)
            self.ready.emit()
            while True:
                if self.overflow.is_set():
                    raise RuntimeError('Переполнена очередь записи эксперимента; архив неполный')
                try:
                    kind, payload = self.queue.get(timeout=.1)
                except Empty:
                    if self.closing.is_set():
                        break
                    continue
                if kind == 'eeg':
                    recorder.append(*payload)
                elif kind == 'trial':
                    number, run, result, (sums, window) = payload
                    folder = self.directory / f'trial_{number:03d}'
                    folder.mkdir(exist_ok=False)
                    (folder / 'upstream_run.json').write_text(json.dumps(run, ensure_ascii=False, default=json_value), encoding='utf-8')
                    (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')
                    epochs = run.get('epochs_data', {})
                    arrays, objects = [], []
                    for key, values in epochs.items():
                        for epoch in values:
                            arrays.append(np.asarray(epoch, float))
                            objects.append(str(key))
                    if arrays:
                        np.savez_compressed(folder / 'epochs.npz', epochs=np.stack(arrays),
                                            objects=np.array(objects), time_ms=np.asarray(run.get('epoch_time_ms', [])))
                    if len(sums.objects):
                        save_sums(sums, folder, window)
                else:
                    events.write(json.dumps({'kind': kind, 'payload': payload}, ensure_ascii=False, default=json_value) + '\n')
                    events.flush()
        except Exception as exc:
            error = str(exc)
            self.closing.set()
            self.failed.emit(error)
        finally:
            try:
                recorder.stop()
                if events:
                    events.close()
                if created:
                    self._metadata(recorder, not bool(error) and self.closing.is_set(), error)
            except Exception as exc:
                self.failed.emit('Ошибка завершения архива: ' + str(exc))
