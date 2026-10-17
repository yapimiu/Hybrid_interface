"""Session orchestration contracts, using unchanged upstream synthetic inlets."""
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel
from pylsl import StreamInfo

from app.bci.experiment import ExperimentController, ExperimentPlan
from app.bci.panel import P300Panel
from app.bci.session import SessionWriter
from app.bci.upstream import protocol_helper
from validation.synthetic_streams import SyntheticEEGInlet, SyntheticMarkerInlet, p300_trial_markers
from validation.synthetic_streams import erp_like_epochs, MarkerEvent


@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait(app, predicate, seconds=8):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


class Components(QObject):
    component_finished = pyqtSignal(str, int)
    def __init__(self):
        super().__init__()
        self.status = QLabel()
        self.launched = []
        self.managed_session = False
        self.stops = 0
    def any_running(self):
        return False
    def validate_python(self):
        return 'test-only'
    def launch(self, kind, extra_args=None):
        self.launched.append((kind, extra_args))
        return object()
    def stop_processes(self):
        self.stops += 1


class Markers(SyntheticMarkerInlet):
    def info(self):
        return StreamInfo('BCI_StimMarkers', 'Markers', 1, 0, 'string', 'session-test')


def setup(app, tmp_path, trials=1):
    panel = P300Panel()
    panel.set_stream(StreamInfo('test', 'EEG', 21, 250, 'double64', 'test-only'), [f'Ch{i}' for i in range(21)])
    components = Components()
    controller = ExperimentController(panel, components)
    controller.last_eeg_at = time.monotonic()
    controller.start(ExperimentPlan('synthetic-software-test', trials=trials), tmp_path)
    wait(app, lambda: controller.state == 'connecting')
    controller.timer.stop()
    return panel, components, controller


def cleanup(app, panel, controller):
    controller.begin_shutdown()
    wait(app, controller.stopped)
    panel.begin_shutdown()
    wait(app, panel.stopped)


def test_plan_uses_original_queue_exactly():
    plan = ExperimentPlan('S01', trials=19, seed=17)
    queue, _ = protocol_helper('experiment_queue').build_p300_block_queue(
        p300_calib_trials=0, calib_target_tile_id=4, p300_main_trials=19, shuffle_seed=17)
    assert plan.targets() == [item.target_tile_id for item in queue]
    assert plan.targets() == plan.targets()
    assert len(plan.targets()) == 19


def test_preflight_requires_samples_not_only_discovery(app, tmp_path):
    panel = P300Panel()
    panel.set_stream(StreamInfo('test', 'EEG', 2, 250), ['Cz', 'Pz'])
    controller = ExperimentController(panel, Components())
    try:
        with pytest.raises(ValueError, match='свежие отсчёты'):
            controller.start(ExperimentPlan('S01'), tmp_path)
        assert panel.channel_checkboxes  # visible before marker connection
        assert not list(tmp_path.iterdir())
    finally:
        panel.begin_shutdown()


def test_full_session_preserves_raw_and_cannot_guess_missing_classes(app, tmp_path):
    panel, components, controller = setup(app, tmp_path)
    try:
        assert '--stim-control-dir' in components.launched[0][1]
        assert json.loads((controller.directory / 'stim_control.json').read_text())['state'] == 'paused'
        target = controller.targets[0]
        panel._attach_markers(Markers(p300_trial_markers(target_tile=target, t_start=.1, n_flashes=12)), panel.connection_revision)
        panel._timer.stop()
        assert controller.state == 'trial'
        assert panel.combo_winner_mode.currentData() == 'auc'
        assert not panel.spin_x.isEnabled()
        source = SyntheticEEGInlet(n_channels=21, fs_hz=250, seed=2)
        expected = []
        for _ in range(60):
            samples, timestamps = source.pull_chunk(max_samples=32)
            array, times = np.array(samples), np.array(timestamps)
            controller.eeg(array, times)
            panel.submit_eeg(array, times)
            expected.extend([[stamp, *sample] for stamp, sample in zip(timestamps, samples)])
            with patch('p300_analysis.qt_window.lsl_local_clock', return_value=timestamps[-1]):
                panel._update_loop()
        assert controller.state == 'draining'
        controller.end_received_at = time.monotonic() - 2
        controller.last_flash_at = time.monotonic() - 2
        controller.tick()
        assert controller.state == 'pause'
        assert controller.results[0]['winner'] is None  # source sim flashes two classes, not nine
        assert controller.results[0]['status'] == 'insufficient_evidence'
        controller.deadline = 0
        controller.tick()
        wait(app, controller.stopped)
        assert controller.state == 'completed'
        actual = np.loadtxt(controller.directory / 'eeg.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
        np.testing.assert_array_equal(actual, expected)
        meta = json.loads((controller.directory / 'session.json').read_text())
        assert meta['recording_complete'] and meta['protocol_complete']
        result = json.loads((controller.directory / 'trial_001/result.json').read_text())
        assert result['target'] == target and result['winner'] is None
        with np.load(controller.directory / 'trial_001/epochs.npz', allow_pickle=False) as archive:
            assert archive['epochs'].shape[1] == 226
        events = [json.loads(line) for line in (controller.directory / 'events.jsonl').read_text().splitlines()]
        assert any(event['kind'] == 'marker' for event in events)
    finally:
        cleanup(app, panel, controller)


def test_dropout_stops_stimulus_and_marks_protocol_incomplete(app, tmp_path):
    panel, components, controller = setup(app, tmp_path)
    try:
        controller.last_eeg_at = time.monotonic() - 4
        controller.tick()
        wait(app, controller.stopped)
        assert controller.state == 'failed'
        assert components.stops == 1
        assert not json.loads((controller.directory / 'session.json').read_text())['protocol_complete']
        assert json.loads((controller.directory / 'stim_control.json').read_text())['state'] == 'done'
        assert panel.spin_x.isEnabled()
    finally:
        cleanup(app, panel, controller)


def test_wrong_trial_cue_is_rejected(app, tmp_path):
    panel, components, controller = setup(app, tmp_path)
    try:
        panel._attach_markers(Markers(), panel.connection_revision)
        controller._marker(f'-1|trial_start|target={(controller.targets[0] + 1) % 9}', .1)
        wait(app, controller.stopped)
        assert controller.state == 'failed'
        assert not controller.results
    finally:
        cleanup(app, panel, controller)


def test_stimulus_crash_closes_session(app, tmp_path):
    panel, components, controller = setup(app, tmp_path)
    try:
        components.component_finished.emit('stimulus', 1)
        wait(app, controller.stopped)
        assert controller.state == 'failed'
        assert 'Стимулятор' in json.loads((controller.directory / 'session.json').read_text(encoding='utf-8'))['reason']
    finally:
        cleanup(app, panel, controller)


def test_writer_does_not_overwrite_existing_session(app, tmp_path):
    original = tmp_path / 'session.json'
    original.write_text('preserved')
    writer = SessionWriter(tmp_path, {'channels': ['A']})
    failures = []
    writer.failed.connect(failures.append)
    writer.start()
    wait(app, lambda: not writer.isRunning())
    assert failures
    assert original.read_text() == 'preserved'


def test_two_trials_use_native_decisions_and_reset_epochs(app, tmp_path):
    panel, components, controller = setup(app, tmp_path, trials=2)
    try:
        markers = Markers()
        panel._attach_markers(markers, panel.connection_revision)
        panel._timer.stop()
        for number in range(2):
            target = controller.targets[number]
            stamps = np.arange(300) / 250 + number * 10
            markers.arm([MarkerEvent(float(stamps[0]), f'-1|trial_start|target={target}')])
            panel.submit_eeg(np.zeros((300, 21)), stamps)
            controller.eeg(np.zeros((300, 21)), stamps)
            with patch('p300_analysis.qt_window.lsl_local_clock', return_value=stamps[-1]):
                panel._update_loop()
            assert panel.epochs_data == {}
            length = len(panel._epoch_geom.time_ms_template)
            # Original numerical generator, confined to tests; no human accuracy claim.
            panel.epochs_data = {f'стимул_{obj}': [erp_like_epochs(
                n_channels=6, n_samples=length, target_ch=0, amp_uv=25 if obj == target else 5,
                seed=number * 1000 + obj * 10 + rep) * 1e6 for rep in range(5)] for obj in range(9)}
            panel._redraw_from_epochs()
            native_winner = panel.last_decision['winner_digit']
            controller._marker('-2|trial_end', float(stamps[-1]))
            controller.end_received_at = time.monotonic() - 2
            controller.tick()
            assert controller.results[number]['winner'] == native_winner
            assert controller.results[number]['status'] == 'decision'
            assert min(controller.results[number]['clean_counts'].values()) == 5
            controller.deadline = 0
            controller.tick()
        wait(app, controller.stopped)
        assert controller.state == 'completed'
        assert (controller.directory / 'trial_001/result.json').is_file()
        assert (controller.directory / 'trial_002/result.json').is_file()
    finally:
        cleanup(app, panel, controller)


def test_manual_stop_saves_interrupted_trial(app, tmp_path):
    panel, components, controller = setup(app, tmp_path)
    try:
        panel._attach_markers(Markers(), panel.connection_revision)
        panel._timer.stop()
        assert panel._recording_epochs
        controller.stop()
        wait(app, controller.stopped)
        result = json.loads((controller.directory / 'trial_001/result.json').read_text(encoding='utf-8'))
        assert result['status'] == 'interrupted'
        assert result['winner'] is None
    finally:
        cleanup(app, panel, controller)
