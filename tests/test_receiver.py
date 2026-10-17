"""LSL receiver stream choice: NeuroSpectr leaves a silent outlet for every stopped recording."""
import os
import time
import uuid
from unittest.mock import patch

import numpy as np
import pytest
from pylsl import StreamInfo, StreamOutlet, resolve_byprop
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.viewer import LSLReceiver


class Info:
    def __init__(self, uid, created_at):
        self._uid, self._created_at = uid, created_at

    def uid(self):
        return self._uid

    def created_at(self):
        return self._created_at


def test_choose_newest_stream():
    receiver = LSLReceiver()
    old, new = Info('old', 10.0), Info('new', 20.0)
    assert receiver._choose_stream([old, new]) is new


def test_skip_stream_that_sent_nothing():
    receiver = LSLReceiver()
    silent, live = Info('silent', 20.0), Info('live', 10.0)
    receiver._silent_uids.add('silent')
    assert receiver._choose_stream([silent, live]) is live


def test_retry_when_every_stream_was_silent():
    receiver = LSLReceiver()
    only = Info('only', 10.0)
    receiver._silent_uids.add('only')
    assert receiver._choose_stream([only]) is only
    assert not receiver._silent_uids


@pytest.mark.skipif(os.environ.get('RUN_LSL_NETWORK') != '1', reason='Set RUN_LSL_NETWORK=1 for real local LSL sockets')
def test_receiver_leaves_silent_stream_for_live_one():
    app = QApplication.instance() or QApplication([])
    prefix = 'RECEIVER_TEST_' + uuid.uuid4().hex
    silent = StreamOutlet(StreamInfo(prefix + '_silent', 'EEG', 4, 250, 'float32', prefix + '_silent'))
    names, chunks, live = [], [], None

    def own_streams(prop, value, **kwargs):
        return [s for s in resolve_byprop(prop, value, **kwargs) if s.name().startswith(prefix)]

    with patch('app.viewer.resolve_byprop', side_effect=own_streams):
        receiver = LSLReceiver()
        receiver.connected.connect(lambda info: names.append(info.name()), Qt.DirectConnection)
        receiver.chunk_received.connect(lambda data, stamps: chunks.append(len(data)), Qt.DirectConnection)
        receiver.start()
        try:
            deadline = time.monotonic() + 5
            while not names and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert names == [prefix + '_silent']
            live = StreamOutlet(StreamInfo(prefix + '_live', 'EEG', 4, 250, 'float32', prefix + '_live'))
            deadline = time.monotonic() + 15
            while not chunks and time.monotonic() < deadline:
                live.push_chunk(np.zeros((25, 4)).tolist())
                app.processEvents()
                time.sleep(.1)
            assert names[-1] == prefix + '_live'
            assert chunks
        finally:
            receiver.stop()
            receiver.wait(5000)
    del silent, live
