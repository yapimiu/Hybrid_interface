import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.recording import CsvRecording


class CsvRecordingTests(unittest.TestCase):
    def test_all_chunks_and_raw_values_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "signal.csv"
            recorder = CsvRecording()
            names = ['Канал, "1"', "Канал 2"]
            recorder.start(str(path), names)
            first = np.array([[1.2345678901234567, 300000.0], [np.nan, np.inf]])
            second = np.array([[-np.inf, -2.5]])
            recorder.append(first, np.array([10.25, 10.5]))
            recorder.append(second, np.array([11.0]))
            # Each received block must already be readable before Stop is pressed.
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            recorder.stop()
            self.assertEqual(rows[0], ["lsl_timestamp_s", *names])
            values = np.asarray(rows[1:], dtype=float)
            np.testing.assert_array_equal(values[:, 0], [10.25, 10.5, 11.0])
            np.testing.assert_array_equal(values[:, 1:], np.vstack([first, second]))
            self.assertEqual(recorder.sample_count, 3)
            self.assertFalse(recorder.active)

    def test_invalid_chunk_does_not_write_misaligned_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "signal.csv"
            recorder = CsvRecording()
            recorder.start(str(path), ["A", "B"])
            with self.assertRaises(ValueError):
                recorder.append(np.ones((2, 2)), np.array([1.0]))
            with self.assertRaises(ValueError):
                recorder.append(np.ones((2, 3)), np.array([1.0, 2.0]))
            recorder.stop()
            with path.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(len(list(csv.reader(file))), 1)

    def test_failed_open_can_be_retried_and_sessions_stay_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = CsvRecording()
            with self.assertRaises(OSError):
                recorder.start(str(Path(folder) / "missing" / "signal.csv"), ["A"])
            self.assertFalse(recorder.active)
            for index in range(2):
                recorder.start(str(Path(folder) / f"signal{index}.csv"), ["A"])
                self.assertEqual(recorder.sample_count, 0)
                recorder.append(np.array([[index]]), np.array([float(index)]))
                recorder.stop()
                recorder.stop()
                self.assertEqual(recorder.sample_count, 1)
            recorder.append(np.array([[99.0]]), np.array([99.0]))
            self.assertEqual(recorder.sample_count, 1)


if __name__ == "__main__":
    unittest.main()
