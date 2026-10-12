import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.p300 import P300Config, Stimulus
from app.p300.buffer import EEGBuffer, EpochRejected, WaitingForEEG
from app.p300.engine import P300Engine
from app.p300.model import CalibratedModel, make_pipeline, purged_group_splits, train_models
from app.p300.processing import EpochProcessor, ERPAverage
from app.p300.scoring import ScoreAccumulator
from app.p300.storage import load_epochs, save_epochs
from app.p300.offline import preprocess_session
from app.p300.__main__ import main as cli
from p300_fixtures import calibration_epochs, config, event, raw_signal


class ConfigurationAndMarkerTests(unittest.TestCase):
    def test_config_roundtrip_and_invalid_values(self):
        c = config()
        self.assertEqual(P300Config.from_dict(json.loads(json.dumps(c.to_dict()))), c)
        for kwargs in ({"sfreq": 0}, {"input_unit": "unknown"}, {"channel_names": ("Cz", "Cz")},
                       {"h_freq": 50}, {"buffer_s": 1}, {"baseline": (0, 0.1)},
                       {"reject_uv": float("nan")}, {"feature_sfreq": 30}, {"cv_folds": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config(**kwargs)

    def test_strict_marker_and_clock_correction(self):
        payload = dict(event_id="e1", object_id="A", trial_id="trial1", block_id="block1",
                       objects=["A", "B"], target=True)
        marker = Stimulus.parse(json.dumps(payload), 10, 0.25)
        self.assertEqual(marker.timestamp, 10.25)
        self.assertEqual(marker.source_timestamp, 10)
        for changes in ({"target": "false"}, {"object_id": "C"}, {"objects": ["A", "A"]},
                        {"event_id": ""}, {"timestamp": 999}, {"objects": "AB"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Stimulus.parse(json.dumps({**payload, **changes}), 10, 0)


class BufferAndEpochTests(unittest.TestCase):
    def test_ring_rollover_and_append_atomicity(self):
        buffer = EEGBuffer(10, 2, 1)
        data = np.arange(30).reshape(15, 2)
        buffer.append(data[:7], np.arange(7) / 10)
        buffer.append(data[7:], np.arange(7, 15) / 10)
        actual, times = buffer.snapshot()
        np.testing.assert_array_equal(actual, data[-10:])
        np.testing.assert_allclose(times, np.arange(5, 15) / 10)
        with self.assertRaises(ValueError):
            buffer.append(np.ones((2, 2)), [1.4, 1.5])
        np.testing.assert_array_equal(buffer.snapshot()[0], actual)
        buffer.append(data, np.arange(15, 30) / 10)
        np.testing.assert_array_equal(buffer.snapshot()[0], data[-10:])
        self.assertEqual(buffer.count, 10)

    def test_delayed_epochs_and_gap_rejection(self):
        c = config()
        processor = EpochProcessor(c)
        buffer = EEGBuffer(c.sfreq, 2, c.buffer_s)
        data, times = raw_signal()
        buffer.append(data[:500], times[:500])
        with self.assertRaises(WaitingForEEG):
            processor.extract(buffer, event())
        buffer.append(data[500:], times[500:])
        epoch = processor.extract(buffer, event())
        self.assertEqual(epoch.data.shape, (2, 51))
        np.testing.assert_allclose(epoch.data[:, epoch.times <= 0].mean(axis=1), 0, atol=1e-15)
        self.assertAlmostEqual(epoch.source_interval[0], 2.8)
        self.assertAlmostEqual(epoch.source_interval[1], 5.8)
        self.assertAlmostEqual(epoch.alignment_error_s, 0)
        buffer = EEGBuffer(c.sfreq, 2, c.buffer_s)
        buffer.append(np.delete(data, 400, axis=0), np.delete(times, 400))
        with self.assertRaises(EpochRejected):
            processor.extract(buffer, event())

    def test_nonfinite_flat_artifact_old_marker_and_units(self):
        c = config()
        data, times = raw_signal()
        for kind in ("nan", "flat", "artifact"):
            bad = data.copy()
            if kind == "nan":
                bad[400, 0] = np.nan
            elif kind == "flat":
                bad[:, 0] = 1
            else:
                bad[400, 0] = 1000
            buffer = EEGBuffer(c.sfreq, 2, c.buffer_s)
            buffer.append(bad, times)
            with self.subTest(kind=kind), self.assertRaises(EpochRejected):
                EpochProcessor(c).extract(buffer, event())
        epochs = []
        for unit, scale in (("uV", 1), ("V", 1e-6), ("mV", 1e-3)):
            buffer = EEGBuffer(c.sfreq, 2, c.buffer_s)
            buffer.append(data * scale, times)
            epochs.append(EpochProcessor(config(input_unit=unit)).extract(buffer, event()))
        np.testing.assert_allclose(epochs[0].data, epochs[1].data)
        np.testing.assert_allclose(epochs[0].data, epochs[2].data)
        buffer = EEGBuffer(c.sfreq, 2, 4)
        buffer.append(data, times)
        with self.assertRaises(EpochRejected):
            EpochProcessor(c).extract(buffer, event())

    def test_erp_is_mean_per_object(self):
        epochs = calibration_epochs()
        erp = ERPAverage()
        for epoch in epochs:
            erp.add(epoch)
        snapshot = erp.snapshot()
        self.assertEqual(snapshot["A"]["count"], 18)
        np.testing.assert_allclose(snapshot["A"]["data"],
                                   np.mean([e.data for e in epochs if e.event.target], axis=0))


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.epochs = calibration_epochs()
        cls.models, cls.report = train_models(cls.epochs, config())

    def test_both_models_and_portable_serialization(self):
        X = np.stack([e.data for e in self.epochs])
        with tempfile.TemporaryDirectory() as folder:
            for name, model in self.models.items():
                self.assertTrue(np.all(np.isfinite(model.decision_function(X))))
                path = Path(folder) / f"{name}.npz"
                model.save(path)
                loaded = CalibratedModel.load(path, config())
                np.testing.assert_allclose(loaded.decision_function(X), model.decision_function(X))
                with self.assertRaises(ValueError):
                    CalibratedModel.load(path, config(input_unit="V"))
        self.assertIn("not_independent_validation", self.report["validation"])
        self.assertEqual(len(self.report["algorithms"]), 2)

    def test_splits_have_no_shared_blocks_or_source_samples(self):
        for fold in self.report["folds"]:
            self.assertFalse(set(fold["train_blocks"]) & set(fold["test_blocks"]))
            self.assertFalse(set(fold["train_event_ids"]) & set(fold["test_event_ids"]))
            for train in fold["train_indices"]:
                for test in fold["test_indices"]:
                    a, b = self.epochs[train].source_interval, self.epochs[test].source_interval
                    self.assertTrue(a[1] < b[0] or b[1] < a[0])
        groups = ["a", "a", "b", "b", "c", "c"]
        intervals = [[0, 3], [10, 13], [2, 5], [20, 23], [30, 33], [40, 43]]
        splits = list(purged_group_splits(groups, intervals, 3))
        self.assertTrue(any(len(train) < 4 for train, _ in splits))
        for train, test in splits:
            for a in train:
                for b in test:
                    self.assertTrue(intervals[a][1] < intervals[b][0] or intervals[b][1] < intervals[a][0])

    def test_every_pipeline_fit_receives_only_its_train_partition(self):
        fitted = []
        original = make_pipeline
        def factory(algorithm, c):
            pipeline = original(algorithm, c)
            fit = pipeline.fit
            def record_fit(X, y, **kwargs):
                fitted.append((algorithm, X.copy(), y.copy()))
                return fit(X, y, **kwargs)
            pipeline.fit = record_fit
            return pipeline
        with patch("app.p300.model.make_pipeline", side_effect=factory):
            _, report = train_models(self.epochs, config())
        X = np.stack([e.data for e in self.epochs])
        for offset in (0, 4):
            for index, fold in enumerate(report["folds"]):
                np.testing.assert_array_equal(fitted[offset + index][1], X[fold["train_indices"]])
            np.testing.assert_array_equal(fitted[offset + 3][1], X)

    def test_missing_labels_insufficient_groups_and_inconsistent_targets(self):
        no_labels = [replace(e, event=replace(e.event, target=None)) for e in self.epochs]
        with self.assertRaises(ValueError):
            train_models(no_labels, config())
        with self.assertRaises(ValueError):
            train_models(self.epochs[:12], config())
        bad = self.epochs.copy()
        bad[0] = replace(bad[0], event=replace(bad[0].event, target=True))
        with self.assertRaises(ValueError):
            train_models(bad, config())

    def test_reproducible_and_epoch_dataset_roundtrip(self):
        models, report = train_models(self.epochs, config())
        self.assertEqual(report, self.report)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epochs.npz"
            save_epochs(path, self.epochs, config())
            loaded, c = load_epochs(path, config())
            self.assertEqual(c, config())
            for original, restored in zip(self.epochs, loaded):
                np.testing.assert_array_equal(original.data, restored.data)
                self.assertEqual(original.event, restored.event)
            with self.assertRaises(ValueError):
                load_epochs(path, config(input_unit="V"))

    def test_offline_cli_trains_and_evaluates_without_refitting_test_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training_path = root / "training.npz"
            save_epochs(training_path, self.epochs, config())
            cli(["train", str(training_path), str(root / "models")])
            model_path = root / "models" / "shrinkage_lda.npz"
            output = root / "evaluation.json"
            with self.assertRaises(SystemExit):
                cli(["evaluate", str(model_path), str(training_path), str(output)])
            heldout = [replace(e, data=e.data + 1e-8, event=replace(e.event,
                        event_id="external-" + e.event.event_id, block_id="external-" + e.event.block_id))
                       for e in calibration_epochs()]
            heldout_path = root / "heldout.npz"
            save_epochs(heldout_path, heldout, config())
            before = model_path.read_bytes()
            with patch("app.p300.model.make_pipeline", side_effect=AssertionError("test data fit")):
                cli(["evaluate", str(model_path), str(heldout_path), str(output)])
            self.assertEqual(before, model_path.read_bytes())
            self.assertEqual(len(json.loads(output.read_text())["scores"]), len(heldout))


class EngineIntegrationTests(unittest.TestCase):
    def test_real_pipeline_chunk_order_late_marker_duplicates_and_no_model(self):
        c = config()
        data, times = raw_signal()
        first, second = P300Engine(c), P300Engine(c)
        for engine in (first, second):
            engine.set_marker_available(True)
        first.ingest_marker(event())
        for start in range(0, len(data), 37):
            first.ingest_eeg(data[start:start + 37], times[start:start + 37] - 0.5, 0.5)
        second.ingest_eeg(data, times - 0.5, 0.5)
        second.ingest_marker(event())
        np.testing.assert_allclose(first.erp.snapshot()["A"]["data"], second.erp.snapshot()["A"]["data"])
        self.assertIsNone(first.latest_result)
        self.assertIn("Нет откалиброванной", first.status)
        first.ingest_marker(event())
        self.assertEqual(first.accepted, 1)
        self.assertEqual(first.rejected, 1)
        first.set_marker_available(False)
        self.assertIn("Нет потока", first.status)

    def test_raw_recording_labels_rejections_and_session_files(self):
        data, times = raw_signal()
        engine = P300Engine(config())
        engine.set_marker_available(True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment"
            engine.start_experiment(path, {"name": "test-only synthetic fixture"})
            engine.start_calibration()
            engine.ingest_marker(event())
            engine.ingest_marker(replace(event(1), target=None))
            engine.ingest_eeg(data, times - 0.5, 0.5)
            self.assertEqual(len(engine.epochs), 1)
            engine.close()
            saved = np.loadtxt(path / "eeg.csv", delimiter=",", skiprows=1)
            np.testing.assert_allclose(saved[:, 0], times - 0.5)
            np.testing.assert_allclose(saved[:, 1], times)
            np.testing.assert_array_equal(saved[:, 3:], data)
            self.assertTrue(json.loads((path / "session.json").read_text())["complete"])
            self.assertEqual(len((path / "events.jsonl").read_text().splitlines()), 2)
            self.assertIn("rejected", (path / "results.jsonl").read_text())
            epochs, _ = load_epochs(path / "epochs.npz")
            self.assertEqual(epochs[0].event, event())
            restored, restored_config, report = preprocess_session(path)
            self.assertEqual(restored_config, config())
            self.assertEqual(report["accepted"], 1)
            np.testing.assert_allclose(restored[0].data, epochs[0].data)

    def test_online_uses_model_and_never_target_label(self):
        epochs = calibration_epochs()
        models, _ = train_models(epochs, config())
        engine = P300Engine(config(min_repetitions=1))
        engine.set_marker_available(True)
        # Model must match all parameters, including accumulation policy, when loaded.
        engine.models = models
        data, times = raw_signal()
        engine.ingest_eeg(data, times, 0)
        engine.ingest_marker(replace(event(), target=None))
        self.assertIsNotNone(engine.latest_result)
        self.assertEqual(engine.latest_result["counts"], {"A": 1, "B": 0})
        self.assertIsNone(engine.latest_result["candidate"])

    def test_object_scores_require_all_objects_and_equal_repetitions(self):
        accumulator = ScoreAccumulator(2)
        a, b = event(), event(1, target=False)
        accumulator.add(a, 4)
        accumulator.add(a, 2)
        result = accumulator.add(b, -1)
        self.assertIsNone(result["candidate"])
        result = accumulator.add(b, 0)
        self.assertEqual(result["candidate"], "A")
        self.assertEqual(result["mean_scores"], {"A": 3, "B": -0.5})
        result = accumulator.add(replace(a, trial_id="new"), 9)
        self.assertEqual(result["counts"], {"A": 1, "B": 0})


if __name__ == "__main__":
    unittest.main()
