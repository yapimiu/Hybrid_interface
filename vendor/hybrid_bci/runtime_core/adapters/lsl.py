"""LSL discovery/inlet helpers (implementations in p300_analysis.lsl_streams)."""

from __future__ import annotations

from p300_analysis.lsl_streams import (
    BCI_STIM_MARKER_STREAM_NAME,
    discover_eeg_streams,
    probe_stimulus_marker_stream,
    resolve_marker_streams,
    select_eeg_stream,
    select_migalka_marker_stream,
    select_stimulus_marker_stream,
    stream_channel_labels,
    stream_display_label,
    stream_inlet_with_buffer,
    wait_for_stimulus_marker_stream,
)
from runtime_core.shutdown import close_lsl_inlet

__all__ = [
    "BCI_STIM_MARKER_STREAM_NAME",
    "close_lsl_inlet",
    "discover_eeg_streams",
    "probe_stimulus_marker_stream",
    "resolve_marker_streams",
    "select_eeg_stream",
    "select_migalka_marker_stream",
    "select_stimulus_marker_stream",
    "stream_channel_labels",
    "stream_display_label",
    "stream_inlet_with_buffer",
    "wait_for_stimulus_marker_stream",
]
