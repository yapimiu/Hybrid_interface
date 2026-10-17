"""Host-integration tests, including the unchanged upstream synthetic battery."""
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pylsl import StreamInfo
from PyQt5.QtWidgets import QApplication

from app.bci.bridge import SharedEEGInlet
from app.bci.controls import bundled_source, prepare_runtime
from validation.p300_synthetic import run_p300_synthetic_battery
from validation.synthetic_streams import SyntheticEEGInlet, SyntheticMarkerInlet, p300_trial_markers


@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_for(app, predicate, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


def test_bridge_keeps_timestamps_order_and_partial_reads():
    info = StreamInfo('test', 'EEG', 2, 250, 'float32', 'test')
    bridge = SharedEEGInlet(info, scale=1e6)
    samples = np.arange(24).reshape(12, 2) * 1e-6
    times = np.arange(12) / 250 + 100
    bridge.push(samples[:7], times[:7])
    bridge.push(samples[7:], times[7:])
    first, t1 = bridge.pull_chunk(max_samples=9)
    second, t2 = bridge.pull_chunk(max_samples=9)
    np.testing.assert_array_equal(np.array(first + second), samples * 1e6)
    np.testing.assert_array_equal(t1 + t2, times)
    assert bridge.info() is info
    np.testing.assert_array_equal(samples, np.arange(24).reshape(12, 2) * 1e-6)
    bridge.close_stream()
    bridge.push(samples, times)
    assert bridge.pull_chunk() == ([], [])


def test_overflow_is_reported_instead_of_silent_epoch_corruption():
    info = StreamInfo('test', 'EEG', 1, 250, 'float32', 'test')
    bridge = SharedEEGInlet(info)
    bridge.push(np.zeros((5000, 1)), np.arange(5000) / 250)
    with pytest.raises(RuntimeError, match='Переполнение'):
        bridge.pull_chunk()


def test_original_synthetic_battery_is_used():
    result = run_p300_synthetic_battery(seed=0)
    assert result['schema'] == 'hybrid_bci_p300_synthetic_validation/v1'
    assert result['summary'] == {'trials': 2, 'both_decide': 2, 'stable_rate': 1.0}


def test_all_copied_code_is_byte_identical():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'p300_analysis/UPSTREAM.json').read_text())
    for path, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path


def test_runtime_copy_preserves_source_and_mutable_settings(tmp_path):
    root = prepare_runtime(tmp_path)
    assert (root / 'run_app.py').read_bytes() == (bundled_source() / 'run_app.py').read_bytes()
    (root / 'settings.json').write_text('{"test":"saved"}')
    assert prepare_runtime(tmp_path) == root
    assert json.loads((root / 'settings.json').read_text())['test'] == 'saved'


class MarkerStub(SyntheticMarkerInlet):
    def info(self):
        return StreamInfo('BCI_StimMarkers', 'Markers', 1, 0, 'string', 'test')


def test_original_viewer_csv_and_embedded_analyzer_share_samples(app, tmp_path):
    from app.viewer import SignalViewerWindow
    from p300_analysis.qt_window import P300AnalyzerWindow
    with patch('app.viewer.LSLReceiver.start'):
        window = SignalViewerWindow()
    try:
        info = StreamInfo('test', 'EEG', 21, 250, 'double64', 'integration')
        window._on_connected(info)
        panel = window.p300_panel
        assert isinstance(panel, P300AnalyzerWindow)
        assert window.tabs.count() == 4
        marker = MarkerStub(p300_trial_markers(target_tile=4, t_start=.1, n_flashes=12))
        panel._attach_markers(marker, panel.connection_revision)
        panel._timer.stop()
        panel._on_start_analysis_clicked()
        window.recording.start(str(tmp_path / 'original.csv'), [f'Ch {i}' for i in range(21)])
        source = SyntheticEEGInlet(n_channels=21, fs_hz=250, seed=0)
        saved_samples, saved_times = [], []
        # Original synthetic source drives the host receiver boundary; no new generator.
        for tick in range(50):
            samples, times = source.pull_chunk(max_samples=32)
            array, timestamps = np.array(samples), np.array(times)
            saved_samples.extend(samples)
            saved_times.extend(times)
            window._on_chunk(array, timestamps)
            panel.submit_eeg(array, timestamps)
            with patch('p300_analysis.qt_window.lsl_local_clock', return_value=times[-1]):
                panel._update_loop()
        panel._on_stop_analysis_clicked()
        assert panel._last_run_export_data is not None
        assert panel._last_run_export_data['markers']
        assert panel.epochs_data
        actual = np.loadtxt(tmp_path / 'original.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
        np.testing.assert_array_equal(actual[:, 0], saved_times)
        np.testing.assert_array_equal(actual[:, 1:], saved_samples)
        np.testing.assert_array_equal(panel._last_run_export_data['eeg_samples'], saved_samples)
        window._update_plots()
        assert len(window.channel_widgets) == 21
    finally:
        window.close()
        wait_for(app, lambda: window.p300_panel.stopped() and window.component_controls.stopped())
        window.close()


def test_transport_error_stops_p300_without_touching_viewer(app):
    from app.viewer import SignalViewerWindow
    with patch('app.viewer.LSLReceiver.start'):
        window = SignalViewerWindow()
    try:
        info = StreamInfo('test', 'EEG', 1, 250, 'double64', 'overflow')
        window._on_connected(info)
        panel = window.p300_panel
        panel._attach_markers(MarkerStub(), panel.connection_revision)
        panel.submit_eeg(np.zeros((5000, 1)), np.arange(5000) / 250)
        panel._update_loop()
        assert panel._inlet_eeg is None
        assert window._connected
        assert window.record_button.isEnabled()
    finally:
        window.close()


def test_disconnected_stream_cannot_finish_stale_marker_connection(app):
    from app.bci.panel import P300Panel
    panel = P300Panel()
    marker = MarkerStub()
    try:
        panel.set_stream(StreamInfo('a', 'EEG', 1, 250), ['A'])
        revision = panel.connection_revision
        panel.disconnected()
        panel._attach_markers(marker, revision)
        assert panel._inlet_markers is None
        assert marker._closed
    finally:
        panel.begin_shutdown()


def test_launcher_stops_owned_child_processes(app, tmp_path):
    import os
    import sys
    from PyQt5.QtCore import QProcess
    from app.bci.controls import ProcessControls
    root = tmp_path / 'runtime'
    root.mkdir()
    # Dummy component stands in for a GUI; never run a stimulus in unit tests.
    (root / 'run_app.py').write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path('child.pid').write_text(str(p.pid))\n"
        "Path('arguments.txt').write_text(' '.join(sys.argv))\n"
        "time.sleep(30)\n", encoding='utf-8')
    with patch('app.bci.controls.QSettings'), patch('app.bci.controls.prepare_runtime', return_value=root):
        controls = ProcessControls()
        controls.python.setText(sys.executable)
        try:
            controls.launch('stimulus')
            wait_for(app, lambda: (root / 'child.pid').exists() and (root / 'arguments.txt').exists())
            process = controls.processes['stimulus']
            assert process.state() == QProcess.Running
            assert '--no-analyzer' in (root / 'arguments.txt').read_text()
            child_pid = int((root / 'child.pid').read_text())
            if sys.platform == 'win32':
                import ctypes as c
                from ctypes import wintypes as w
                kernel = c.WinDLL('kernel32')
                kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
                kernel.OpenProcess.restype = w.HANDLE
                kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
                kernel.CloseHandle.argtypes = [w.HANDLE]
                child = kernel.OpenProcess(0x00100000, False, child_pid)
                assert child
            controls.begin_shutdown()
            wait_for(app, controls.stopped)
            if sys.platform == 'win32':
                try:
                    wait_for(app, lambda: kernel.WaitForSingleObject(child, 0) == 0)
                finally:
                    kernel.CloseHandle(child)
        finally:
            controls.begin_shutdown()
            wait_for(app, controls.stopped)
