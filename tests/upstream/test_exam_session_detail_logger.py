"""Тесты вспомогательных функций подробного лога обследования (без pylsl/Qt)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p300_analysis.exam_session_detail_logger import (
    ExamSessionDetailLogger,
    epoch_roi_summary,
    pending_snapshot_for_log,
    summarize_eeg_chunk,
)


def test_summarize_eeg_chunk_shape_and_stats() -> None:
    a = np.random.default_rng(0).standard_normal((40, 3))
    ts = [float(i) * 0.004 for i in range(40)]
    d = summarize_eeg_chunk(a, ts)
    assert d["shape"] == [40, 3]
    assert d["n_samples"] == 40
    assert d["ts_unique_in_chunk"] >= 1
    assert "per_channel_stats" in d
    assert len(d["per_channel_stats"]) == 3


def test_epoch_roi_summary_includes_waveform() -> None:
    e = np.linspace(-1, 1, 64, dtype=np.float64)
    s = epoch_roi_summary(e)
    assert s["len"] == 64
    assert "epoch_samples_roi_mean" in s
    assert len(s["epoch_samples_roi_mean"]) == 64


def test_pending_snapshot() -> None:
    p = [(1.0, "стимул_0"), (1.1, "стимул_1"), (1.2, "стимул_2")]
    s = pending_snapshot_for_log(p, max_each_side=2)
    assert s["n"] == 3
    assert len(s["head"]) == 2
    assert len(s["tail"]) == 2


def test_exam_logger_open_new_writes_ndjson(tmp_path: Path) -> None:
    lg = ExamSessionDetailLogger.open_new(
        run_seq=7, exam_start_data={"k": "v"}, output_dir=tmp_path
    )
    try:
        assert lg.path.parent == tmp_path
        lg.write("ping", {"x": 1})
    finally:
        lg.close()
    lines = lg.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    o0 = json.loads(lines[0])
    assert o0["schema"] == "p300_exam_detail/v1"
    assert o0["event"] == "exam_start"
    assert o0["run_seq"] == 7
    assert o0["data"]["k"] == "v"
