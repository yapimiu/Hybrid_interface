"""Drive ProtocolRunner with synthetic LSL/Migalka (live runtime path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner, ProtocolState
from validation.synthetic_streams import MarkerEvent, SyntheticEEGInlet, SyntheticMarkerInlet, p300_trial_markers


def _fake_stream_info(name: str, stype: str) -> MagicMock:
    info = MagicMock()
    info.name = MagicMock(return_value=name)
    info.type = MagicMock(return_value=stype)
    info.channel_count = MagicMock(return_value=21)
    return info


def run_synthetic_protocol_session(
    tmp_path: Path,
    *,
    max_ticks: int = 8000,
    ssvep_hz: float = 10.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Minimal hybrid queue through real ProtocolRunner.tick() with mocked adapters."""
    session_dir = tmp_path / "sess_synth"
    session_dir.mkdir(parents=True, exist_ok=True)

    p300_markers: List[MarkerEvent] = []
    t = 10.0
    for tile in (4, 5):
        p300_markers.extend(p300_trial_markers(target_tile=tile, t_start=t, n_flashes=5))
        t += 3.0

    eeg = SyntheticEEGInlet(fs_hz=250.0, n_channels=21, seed=seed, ssvep_hz=ssvep_hz)
    mk = SyntheticMarkerInlet()
    mig_mk = SyntheticMarkerInlet([])

    eeg_info = _fake_stream_info("SynthEEG", "EEG")
    mk_info = _fake_stream_info("BCI_StimMarkers", "Markers")

    inlet_queue = [eeg, mk, mig_mk]
    tick_metrics: List[Dict[str, Any]] = []

    def _stream_inlet(info: Any, buffer_seconds: float = 20) -> Any:
        if inlet_queue:
            return inlet_queue.pop(0)
        return MagicMock(pull_chunk=MagicMock(return_value=([], [])))

    import runtime_core.adapters.lsl as lsl_mod

    orig = {
        "discover": lsl_mod.discover_eeg_streams,
        "wait_mk": lsl_mod.wait_for_stimulus_marker_stream,
        "inlet": lsl_mod.stream_inlet_with_buffer,
        "select_eeg": lsl_mod.select_eeg_stream,
        "select_mig": lsl_mod.select_migalka_marker_stream,
        "labels": lsl_mod.stream_channel_labels,
    }

    lsl_mod.discover_eeg_streams = lambda timeout=1.5: [eeg_info]  # type: ignore
    lsl_mod.wait_for_stimulus_marker_stream = lambda max_wait_sec=3.0: (mk_info, [mk_info])  # type: ignore
    lsl_mod.stream_inlet_with_buffer = _stream_inlet  # type: ignore
    lsl_mod.select_eeg_stream = lambda streams, name="", session_id="": eeg_info  # type: ignore
    lsl_mod.select_migalka_marker_stream = lambda streams: None  # type: ignore
    lsl_mod.stream_channel_labels = lambda info, n: [f"Ch{i+1}" for i in range(n)]  # type: ignore

    import ssvep_analysis.migalka_serial_controller as msc

    ser = MagicMock()
    ser.is_open = True
    orig_serial = msc.serial.Serial
    msc.serial.Serial = lambda *a, **k: ser  # type: ignore

    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="synthetic",
        session_dir=session_dir,
        com_port="COM_SYNTH",
        eeg_stream_name="SynthEEG",
        p300_calib_trials=0,
        p300_main_trials=2,
        ssvep_blocks_per_mode=1,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=0.05,
        shuffle_seed=0,
        ssvep_freqs_hz=(8.0, 10.0, 12.0, 15.0),
    )
    runner = ProtocolRunner(cfg)
    try:
        runner.start()
        for _ in range(500):
            runner.tick()
            if runner.state == ProtocolState.Ready:
                break
        mk.arm(p300_markers)
        runner.start_p300_block()
        ticks_used = 0
        for i in range(int(max_ticks)):
            runner.tick()
            ticks_used = i + 1
            if i % 500 == 0 and runner._logger is not None:
                snap = runner._runtime_metrics_snapshot()
                tick_metrics.append(
                    {
                        "tick": i,
                        "state": str(runner.state),
                        "p300_epochs": snap.get("p300_epoch_classes"),
                        "ssvep_buf": snap.get("ssvep_buf_samples"),
                    }
                )
            if runner.state == ProtocolState.Stopped:
                break
        if runner.state != ProtocolState.Stopped:
            runner.stop(reason="validation_harness")
            for _ in range(100):
                runner.tick()
                if runner.state == ProtocolState.Stopped:
                    break

        view = runner.display_view()
        exp_path = session_dir / "experiments.ndjson"
        n_exp = 0
        if exp_path.is_file():
            n_exp = sum(1 for _ in exp_path.open(encoding="utf-8") if _.strip())

        return {
            "final_state": str(runner.state),
            "session_ready_was": view.session_ready,
            "experiments_logged": n_exp,
            "ticks_used": ticks_used,
            "tick_samples": len(tick_metrics),
            "metrics_tail": tick_metrics[-3:],
            "migalka_serial_opens": 1,
        }
    finally:
        lsl_mod.discover_eeg_streams = orig["discover"]  # type: ignore
        lsl_mod.wait_for_stimulus_marker_stream = orig["wait_mk"]  # type: ignore
        lsl_mod.stream_inlet_with_buffer = orig["inlet"]  # type: ignore
        lsl_mod.select_eeg_stream = orig["select_eeg"]  # type: ignore
        lsl_mod.select_migalka_marker_stream = orig["select_mig"]  # type: ignore
        lsl_mod.stream_channel_labels = orig["labels"]  # type: ignore
        msc.serial.Serial = orig_serial  # type: ignore
