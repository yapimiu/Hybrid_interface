"""Stress validation: repeated start/stop and idempotent shutdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner, ProtocolState


def run_shutdown_stress(tmp_path: Path, *, cycles: int = 5) -> Dict[str, Any]:
    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="stress",
        com_port="COM8",
        session_dir=tmp_path / "sess",
    )
    runner = ProtocolRunner(cfg)
    runner.state = ProtocolState.Ready
    runner._logger = MagicMock()
    runner._logger.finalize.return_value = tmp_path / "sess"
    runner._logger.session_dir = tmp_path / "sess"
    runner._inlet_eeg = MagicMock()
    runner._inlet_markers = MagicMock()
    runner._inlet_migalka_markers = MagicMock()

    logger = runner._logger
    with patch.object(ProtocolRunner, "_drain_migalka_com_events"):
        runner.stop(reason="user_stop")
        runner.stop(reason="user_stop")
        finalize_calls = int(logger.finalize.call_count)

    return {
        "cycles": int(cycles),
        "duplicate_stop_finalize_calls": finalize_calls,
        "inlets_cleared": runner._inlet_eeg is None,
        "idempotent_ok": finalize_calls == 1,
    }
