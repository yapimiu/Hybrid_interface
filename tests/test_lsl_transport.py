"""Optional network smoke test: original synthetic source over actual local LSL."""
import os
import time
import uuid
from unittest.mock import patch

import numpy as np
import pytest
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, local_clock
from PyQt5.QtWidgets import QApplication
from validation.synthetic_streams import SyntheticEEGInlet, p300_trial_markers

pytestmark = pytest.mark.skipif(os.environ.get('RUN_LSL_NETWORK') != '1', reason='Set RUN_LSL_NETWORK=1 for real local LSL sockets')


def test_original_source_over_lsl_into_both_tabs(tmp_path):
    from app.viewer import SignalViewerWindow
    app = QApplication.instance() or QApplication([])
    token = uuid.uuid4().hex
    eeg_name, marker_name = 'EEG_TEST_' + token, 'BCI_TEST_' + token
    eeg = StreamOutlet(StreamInfo(eeg_name, 'EEG', 21, 250, 'double64', token))
    markers = StreamOutlet(StreamInfo(marker_name, 'Markers', 1, 0, 'string', token + '_m'))
    def wait(predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        assert predicate()
    def resolve_test(*args, **kwargs):
        return resolve_byprop('name', eeg_name, timeout=.5)
    with patch('app.viewer.resolve_byprop', side_effect=resolve_test):
        window = SignalViewerWindow()
        try:
            wait(lambda: window._connected)
            panel = window.p300_panel
            panel.marker_name.setText(marker_name)
            panel._on_connect_clicked()
            wait(lambda: panel._inlet_markers is not None)
            wait(lambda: eeg.have_consumers() and markers.have_consumers())
            panel._on_start_analysis_clicked()
            window.recording.start(str(tmp_path / 'raw.csv'), [f'Ch {i}' for i in range(21)])
            source = SyntheticEEGInlet(n_channels=21, fs_hz=250, seed=7)
            anchor = local_clock()
            events = p300_trial_markers(target_tile=4, t_start=anchor + .3, n_flashes=2)
            sent = []
            for _ in range(25):
                samples, timestamps = source.pull_chunk(max_samples=32)
                for sample, stamp in zip(samples, timestamps):
                    stamp += anchor
                    eeg.push_sample(sample, timestamp=stamp)
                    sent.append([stamp, *sample])
                until = time.monotonic() + 32 / 250
                while time.monotonic() < until:
                    while events and events[0].lsl_time <= local_clock():
                        event = events.pop(0)
                        markers.push_sample([event.value], timestamp=event.lsl_time)
                    app.processEvents()
                    time.sleep(.005)
            wait(lambda: window.recording.sample_count == len(sent))
            wait(lambda: bool(panel.epochs_data))
            panel._on_stop_analysis_clicked()
            assert panel._last_run_export_data['epoch_segments']
            actual = np.loadtxt(tmp_path / 'raw.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
            np.testing.assert_array_equal(actual, sent)
        finally:
            window.close()
            wait(lambda: not window.receiver.isRunning() and window.p300_panel.stopped() and window.component_controls.stopped())
            window.close()
