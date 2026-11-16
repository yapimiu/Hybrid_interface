"""P300 event discovery and selection, without LSL network dependencies."""
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtWidgets import QApplication

from app.bci.bridge import ConnectMarkers
from app.bci.marker_streams import discover_marker_streams
from app.bci.panel import P300Panel


def stream(name='ExternalEvents', kind='Events', channels=1, uid='external'):
    info = MagicMock()
    info.name.return_value = name
    info.type.return_value = kind
    info.channel_count.return_value = channels
    info.uid.return_value = uid
    info.hostname.return_value = 'device-pc'
    info.source_id.return_value = uid
    return info


def test_discovery_finds_vendor_events_and_excludes_eeg_ssvep_and_multichannel():
    events = stream()
    markers = stream('DeviceMarkers', 'markers', uid='markers')
    triggers = stream('Triggers', 'Triggers', uid='triggers')
    sources = [events, markers, events, triggers, stream('EEGMarkers', 'EEG'),
               stream('StimSignal', 'Signal'), stream('MigalkaStimMarkers', 'Markers'),
               stream(channels=2), stream('HeartRate', 'HR')]
    with patch('app.bci.marker_streams.resolve_streams', return_value=sources):
        assert discover_marker_streams() == [markers, events, triggers]


def test_auto_connect_uses_sole_external_event_stream():
    info = stream()
    worker = ConnectMarkers('')
    connected, failures = [], []
    worker.connected.connect(connected.append)
    worker.failed.connect(failures.append)
    with patch('app.bci.bridge.discover_marker_streams', return_value=[info]), \
            patch('app.bci.bridge.StreamInlet') as factory:
        worker.run()
        factory.assert_called_once_with(info, max_buflen=60, recover=False)
        assert connected == [factory.return_value]
        assert not failures


@pytest.mark.parametrize('sources', [[], [stream(), stream('OtherMarkers', 'Markers', uid='other')]])
def test_auto_connect_never_guesses_or_opens_an_inlet_when_ambiguous(sources):
    worker = ConnectMarkers('')
    failures, found = [], []
    worker.failed.connect(failures.append)
    worker.found.connect(found.append)
    with patch('app.bci.bridge.discover_marker_streams', return_value=sources), \
            patch('app.bci.bridge.StreamInlet') as factory:
        worker.run()
    assert found == [sources]
    assert failures
    factory.assert_not_called()


def test_manual_name_rejects_duplicates_but_selected_identity_connects():
    first, second = stream(), stream(uid='second')
    worker = ConnectMarkers('ExternalEvents')
    failures = []
    worker.failed.connect(failures.append)
    with patch('app.bci.bridge.resolve_byprop', return_value=[first, second]) as resolve, \
            patch('app.bci.bridge.StreamInlet') as factory:
        worker.run()
        assert resolve.call_args.kwargs['minimum'] == 1000
        factory.assert_not_called()
        assert failures
        selected = ConnectMarkers('ExternalEvents', stream_info=second)
        selected.run()
        factory.assert_called_once_with(second, max_buflen=60, recover=False)


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = P300Panel()
    yield widget
    widget.begin_shutdown()
    if widget.marker_discovery:
        assert widget.marker_discovery.wait(3000)
    if widget.connector:
        assert widget.connector.wait(3000)
    app.processEvents()


def test_ui_lists_multiple_streams_and_preserves_manual_name(panel):
    first, second = stream(), stream(uid='second')
    panel._marker_streams_found([first, second])
    assert panel.marker_streams.count() == 3
    assert not panel.marker_name.text()
    panel._select_marker_stream(2)
    assert panel._selected_marker_info is second
    assert panel.marker_name.text() == 'ExternalEvents'
    panel.marker_name.setText('ManualName')
    assert panel._selected_marker_info is None
    panel._marker_streams_found([first])
    assert panel.marker_name.text() == 'ManualName'


def test_ui_selects_sole_stream_and_locks_selection_during_experiment(panel):
    info = stream()
    panel._marker_streams_found([info])
    assert panel._selected_marker_info is info
    panel.set_experiment_owned(True)
    assert not panel.btn_find_markers.isEnabled()
    assert not panel.marker_streams.isEnabled()
    panel._marker_streams_found([stream('LateEvents')])
    assert panel._selected_marker_info is info
    panel.set_experiment_owned(False)
    assert panel.btn_find_markers.isEnabled()
    panel._select_marker_stream(0)
    assert panel.marker_name.text() == ''
    assert panel._selected_marker_info is None


def test_shutdown_waits_for_background_discovery(panel):
    import threading
    entered, release = threading.Event(), threading.Event()
    def blocked_search():
        entered.set()
        release.wait(2)
        return [stream()]
    with patch('app.bci.bridge.discover_marker_streams', side_effect=blocked_search):
        panel._find_marker_streams()
        assert entered.wait(1)
        panel.begin_shutdown()
        assert not panel.stopped()
        release.set()
        assert panel.marker_discovery.wait(2000)
        assert panel.stopped()
        QApplication.processEvents()
        assert not panel.marker_name.text()
