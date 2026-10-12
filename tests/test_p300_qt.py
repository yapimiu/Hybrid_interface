"""Интеграционные проверки Qt/LSL без сети, оборудования и GUI-дисплея."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from pylsl import StreamInfo

from app.p300.lsl import MarkerReceiver
from app.p300.worker import P300Worker
from app.viewer import LSLReceiver, SignalViewerWindow
from p300_fixtures import calibration_epochs, config, event, raw_signal


class LSLContractTests(unittest.TestCase):
    def test_eeg_raw_contract_and_separate_clock_offset(self):
        receiver = LSLReceiver()
        info = StreamInfo("test", "EEG", 2, 100, "float32", "test-eeg")
        data, times = raw_signal(end=0.1)
        data[0, 0] = np.nan
        raw, p300 = [], []
        receiver.chunk_received.connect(lambda x, t: raw.append((x, t)), Qt.DirectConnection)
        receiver.p300_chunk_received.connect(lambda x, t, o: p300.append((x, t, o)), Qt.DirectConnection)
        class Inlet:
            def __init__(self, *args, **kwargs): pass
            def open_stream(self, **kwargs): pass
            def info(self, **kwargs): return info
            def pull_chunk(self, **kwargs):
                receiver._running = False
                return data.tolist(), times.tolist()
            def time_correction(self, **kwargs): return 0.25
            def close_stream(self): pass
        with patch("app.viewer.resolve_byprop", return_value=[info]), patch("app.viewer.StreamInlet", Inlet):
            receiver.run()
        np.testing.assert_array_equal(raw[0][0], data)
        np.testing.assert_array_equal(raw[0][1], times)
        np.testing.assert_array_equal(p300[0][1], times)
        self.assertEqual(p300[0][2], 0.25)

    def test_marker_receiver_corrects_its_own_clock(self):
        receiver = MarkerReceiver()
        info = StreamInfo("stimuli", "Markers", 1, 0, "string", "test-markers")
        marker, states = [], []
        def receive(value):
            marker.append(value)
            receiver.requestInterruption()
        receiver.marker_received.connect(receive, Qt.DirectConnection)
        receiver.state_changed.connect(lambda state, _: states.append(state), Qt.DirectConnection)
        payload = json.dumps(dict(event_id="real-event-protocol-test", object_id="A",
                                  objects=["A", "B"], trial_id="t1", block_id="b1", target=True))
        class Inlet:
            def __init__(self, *args, **kwargs): pass
            def open_stream(self, **kwargs): pass
            def info(self, **kwargs): return info
            def time_correction(self, **kwargs): return 0.5
            def pull_sample(self, **kwargs): return [payload], 10.0
            def close_stream(self): pass
        with patch("app.p300.lsl.resolve_byprop", side_effect=lambda _, value, **kw:
                   [info] if value == "Markers" else []), patch("app.p300.lsl.StreamInlet", Inlet):
            receiver.start()
            self.assertTrue(receiver.wait(5000))
        self.assertEqual(marker[0].timestamp, 10.5)
        self.assertEqual(marker[0].source_timestamp, 10.0)
        self.assertIn(True, states)
        self.assertFalse(states[-1])


class QtIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            QTest.qWait(10)
        self.assertTrue(predicate(), "Qt operation timed out")

    def test_existing_viewer_recording_export_and_erp_panel(self):
        with patch.object(LSLReceiver, "start"), patch.object(MarkerReceiver, "start"):
            window = SignalViewerWindow()
        try:
            info = StreamInfo("fixture", "EEG", 2, 100, "float32", "qt-fixture")
            channels = info.desc().append_child("channels")
            for name in ("Cz", "Pz"):
                channels.append_child("channel").append_child_value("label", name)
            window._on_connected(info)
            panel = window.p300_panel
            panel.apply_config()
            panel.worker.submit("marker_state", True, {})
            data, times = raw_signal()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "original.csv"
                window.recording.start(path, ["Cz", "Pz"])
                window._on_chunk(data, times)
                panel.worker.submit_marker(event())
                panel.worker.submit_eeg(data, times, 0)
                self.wait_until(lambda: panel.last_snapshot is not None
                                and panel.last_snapshot["accepted"] == 1)
                window._update_array_view()
                np.testing.assert_array_equal(window.signal_array, data)
                np.testing.assert_array_equal(window.timestamps_array, times)
                self.assertEqual(window.recording.sample_count, len(data))
                self.assertEqual(len(panel._curves), 1)
                self.assertIsNone(panel.last_snapshot["result"])
                window.show()
                self.application.processEvents()
                png = Path(folder) / "graphs.png"
                with patch("app.viewer.QFileDialog.getSaveFileName", return_value=(str(png), "PNG")):
                    window._save_graphs()
                self.assertTrue(png.is_file())
                window._on_disconnected()
                self.assertFalse(window.recording.active)
                self.assertFalse(window.record_button.isEnabled())
                self.assertEqual(panel.result_label.text(), "Нет текущего результата")
        finally:
            window.close()
            self.wait_until(lambda: panel.stopped())
            window.close()

    def test_filtering_and_training_run_outside_gui_and_ingestion_continues(self):
        worker = P300Worker()
        snapshots, reports = [], []
        worker.updated.connect(lambda value: snapshots.append(value))
        worker.trained.connect(lambda value: reports.append(value))
        started, release = threading.Event(), threading.Event()
        thread_ids = []
        from app.p300.model import train_models
        def slow_fit(epochs, c):
            thread_ids.append(threading.get_ident())
            started.set()
            release.wait(5)
            return train_models(epochs, c)
        worker.start()
        try:
            worker.submit("configure", config(), [0, 1], {})
            self.wait_until(lambda: worker.engine is not None)
            # Inject the test fixture in the owner thread via the normal NPZ import command.
            from app.p300.storage import save_epochs
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "calibration.npz"
                save_epochs(path, calibration_epochs(), config())
                worker.submit("load_calibration", path)
                self.wait_until(lambda: worker.engine.calibrating)
                with patch("app.p300.model.train_models", side_effect=slow_fit):
                    # run() imports train_models on startup; patch its captured implementation
                    # by restarting the worker while the patch is active.
                    worker.shutdown()
                    self.assertTrue(worker.wait(5000))
                    worker = P300Worker()
                    worker.trained.connect(lambda value: reports.append(value))
                    worker.start()
                    worker.submit("configure", config(), [0, 1], {})
                    worker.submit("load_calibration", path)
                    worker.submit("train", "shrinkage_lda")
                    self.wait_until(started.is_set)
                    data, times = raw_signal()
                    worker.submit_eeg(data, times, 0)
                    self.wait_until(lambda: worker.engine.buffer.count == len(data))
                    self.assertFalse(reports)
                    self.assertNotEqual(thread_ids[0], threading.get_ident())
                    release.set()
                    self.wait_until(lambda: len(reports) == 1)
                    # A completed calibration from a disconnected stream must
                    # be archived for its origin, never activated on a new stream.
                    worker.submit("load_calibration", path)
                    self.wait_until(lambda: worker.engine.calibrating)
                    origin = worker.engine
                    started.clear()
                    release.clear()
                    worker.submit("train", "shrinkage_lda")
                    self.wait_until(started.is_set)
                    worker.submit("disconnect")
                    worker.submit("configure", config(), [0, 1], {})
                    self.wait_until(lambda: worker.engine is not None and worker.engine is not origin)
                    release.set()
                    self.wait_until(lambda: origin.report is not None)
                    self.assertFalse(worker.engine.models)
                    self.assertEqual(len(reports), 1)
        finally:
            release.set()
            worker.shutdown()
            self.assertTrue(worker.wait(10000))

    def test_queue_overflow_requires_fresh_configuration(self):
        worker = P300Worker(queue_size=2)
        worker.submit("configure", config(), [0, 1], {})
        worker.submit("marker_state", True, {})
        self.assertFalse(worker.submit("marker", event()))
        worker.start()
        try:
            self.wait_until(lambda: worker.faulted)
            self.assertIsNone(worker.engine)
            worker.submit("configure", config(), [0, 1], {})
            self.wait_until(lambda: worker.engine is not None and not worker.faulted)
        finally:
            worker.shutdown()
            self.assertTrue(worker.wait(5000))


if __name__ == "__main__":
    unittest.main()
