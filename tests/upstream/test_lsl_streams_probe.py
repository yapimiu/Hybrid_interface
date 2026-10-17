"""Non-blocking LSL probe helper (Phase 3.1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from p300_analysis.lsl_streams import BCI_STIM_MARKER_STREAM_NAME, probe_stimulus_marker_stream


@patch("p300_analysis.lsl_streams.resolve_marker_streams")
def test_probe_stimulus_marker_stream_picks_bci(mock_resolve: MagicMock) -> None:
    bci = MagicMock()
    bci.name.return_value = BCI_STIM_MARKER_STREAM_NAME
    mig = MagicMock()
    mig.name.return_value = "MigalkaStimMarkers"
    mock_resolve.return_value = [mig, bci]

    picked, streams = probe_stimulus_marker_stream(timeout=0.1)

    assert picked is bci
    assert streams == [mig, bci]
    mock_resolve.assert_called_once()


@patch("p300_analysis.lsl_streams.resolve_marker_streams")
def test_probe_stimulus_marker_stream_none_when_missing(mock_resolve: MagicMock) -> None:
    mock_resolve.return_value = []

    picked, streams = probe_stimulus_marker_stream(timeout=0.1)

    assert picked is None
    assert streams == []
