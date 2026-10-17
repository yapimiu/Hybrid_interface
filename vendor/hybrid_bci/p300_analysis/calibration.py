"""Offline calibration helpers for subject-specific P300 latency and ROI search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import csv
import numpy as np

from p300_analysis.constants import EPOCH_DURATION_MS
from p300_analysis.erp_compute import build_averaged_erp, compute_winner_metrics
from p300_analysis.marker_parsing import decode_stim_tile_id, stim_key_to_tile_digit
from p300_analysis.signal_processing import (
    bandpass_filter,
    baseline_correction,
    common_average_reference,
)


@dataclass
class CalibrationExample:
    file: str
    path: str
    expected: int
    epochs_data: Dict[str, Tuple[np.ndarray, ...]]
    time_ms: np.ndarray
    channel_names: Tuple[str, ...]
    fs_hz: float
    artifact_uv: float


@dataclass
class PreparedCalibrationExample:
    file: str
    expected: int
    stim_keys: Tuple[str, ...]
    raw_averaged: np.ndarray
    corrected: np.ndarray
    time_ms: np.ndarray


@dataclass
class CalibrationPrediction:
    file: str
    expected: int
    predicted: int
    correct: bool
    margin: float


@dataclass
class CalibrationResult:
    accuracy_pct: float
    correct: int
    total: int
    window_x_ms: int
    window_y_ms: int
    channels_0idx: Tuple[int, ...]
    average_margin_pct: float
    predictions: List[CalibrationPrediction]


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
    delim = ";" if first.startswith("sep=") else ","
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            if row:
                rows.append(row)
    if rows and rows[0] and rows[0][0].startswith("sep="):
        rows = rows[1:]
    if not rows:
        raise RuntimeError(f"Empty file: {path}")
    return rows[0], rows[1:]


def _parse_num(s: str) -> Optional[float]:
    try:
        return float(str(s).strip().replace(",", "."))
    except Exception:
        return None


def load_calibration_example(
    path: Path,
    *,
    baseline_ms: int = 100,
    artifact_uv: float = 60.0,
    use_car: bool = False,
) -> CalibrationExample:
    header, data_rows = _read_csv(path)
    idx = {c: i for i, c in enumerate(header)}
    channel_cols = [c for c in header if c.startswith("ch_")]
    if not channel_cols:
        raise RuntimeError(f"No ch_* columns in {path.name}")
    if "t_rel_s" not in idx or "marker" not in idx:
        raise RuntimeError(f"Missing t_rel_s/marker in {path.name}")

    t_rel: List[float] = []
    marker_vals: List[int] = []
    signal_rows: List[np.ndarray] = []
    target_ids: List[int] = []

    for row in data_rows:
        if len(row) < len(header):
            continue
        tr = _parse_num(row[idx["t_rel_s"]])
        mv = _parse_num(row[idx["marker"]])
        if tr is None or mv is None:
            continue
        vals: List[float] = []
        for ch_name in channel_cols:
            col_idx = idx[ch_name]
            if col_idx >= len(row):
                vals = []
                break
            v = _parse_num(row[col_idx])
            if v is None:
                vals = []
                break
            vals.append(v)
        if not vals:
            continue
        t_rel.append(float(tr))
        marker_vals.append(int(round(float(mv))))
        signal_rows.append(np.array(vals, dtype=np.float64))
        if "target_tile_id" in idx:
            tgt = _parse_num(row[idx["target_tile_id"]])
            target_ids.append(int(round(float(tgt))) if tgt is not None else -1)

    if len(t_rel) < 100:
        raise RuntimeError(f"Too few samples ({len(t_rel)}) in {path.name}")

    valid_targets = [t for t in target_ids if t >= 0]
    if not valid_targets:
        raise RuntimeError(f"No target_tile_id in {path.name}")
    expected = Counter(valid_targets).most_common(1)[0][0]

    dt_s = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 0.002
    fs_hz = 1.0 / dt_s if dt_s > 0 else 500.0

    sig_2d = bandpass_filter(np.stack(signal_rows), fs_hz)
    if use_car:
        sig_2d = common_average_reference(sig_2d)

    post_stim_ms = int(EPOCH_DURATION_MS)
    epoch_len = int(round((baseline_ms + post_stim_ms) / (dt_s * 1000.0))) + 1
    pre_samples = int(round(float(baseline_ms) / (dt_s * 1000.0)))

    epochs_data: Dict[str, List[np.ndarray]] = {}
    prev_tile: Optional[int] = None
    for i, m in enumerate(marker_vals):
        tile_id = decode_stim_tile_id(int(m))
        if int(m) == 0:
            tile_id = None
        if tile_id is not None and (prev_tile is None or prev_tile != tile_id):
            start = i - pre_samples
            end = start + epoch_len
            if start >= 0 and end <= sig_2d.shape[0]:
                stim_key = f"стимул_{tile_id}"
                epochs_data.setdefault(stim_key, []).append(sig_2d[start:end, :].copy())
        prev_tile = tile_id

    if not any(epochs_data.values()):
        raise RuntimeError(f"No ERP data after artifact rejection in {path.name}")

    time_ms = np.arange(epoch_len, dtype=np.float64) * (dt_s * 1000.0) - baseline_ms
    return CalibrationExample(
        file=path.name,
        path=str(path),
        expected=int(expected),
        epochs_data={k: tuple(v) for k, v in epochs_data.items()},
        time_ms=time_ms,
        channel_names=tuple(channel_cols),
        fs_hz=float(fs_hz),
        artifact_uv=float(artifact_uv),
    )


def load_examples_from_paths(
    paths: Sequence[Path],
    *,
    baseline_ms: int = 100,
    artifact_uv: float = 60.0,
    use_car: bool = False,
) -> List[CalibrationExample]:
    examples: List[CalibrationExample] = []
    for path in paths:
        try:
            examples.append(
                load_calibration_example(
                    path,
                    baseline_ms=baseline_ms,
                    artifact_uv=artifact_uv,
                    use_car=use_car,
                )
            )
        except Exception:
            continue
    return examples


def iter_channel_subsets(
    n_channels: int,
    *,
    max_subset_size: Optional[int] = None,
) -> Iterable[Tuple[int, ...]]:
    if n_channels <= 0:
        return
    limit = int(max_subset_size) if max_subset_size is not None else n_channels
    limit = max(1, min(limit, n_channels))
    for size in range(1, limit + 1):
        for combo in combinations(range(n_channels), size):
            yield combo


def _slice_epochs_for_channels(
    epochs_data: Dict[str, Tuple[np.ndarray, ...]],
    channels_0idx: Sequence[int],
) -> Dict[str, List[np.ndarray]]:
    selected_epochs: Dict[str, List[np.ndarray]] = {}
    for stim_key, epochs in epochs_data.items():
        kept: List[np.ndarray] = []
        for ep in epochs:
            if ep.ndim == 1:
                if 0 in channels_0idx:
                    kept.append(ep.copy())
                continue
            valid = [ch for ch in channels_0idx if 0 <= int(ch) < ep.shape[1]]
            if valid:
                kept.append(ep[:, valid].copy())
        if kept:
            selected_epochs[stim_key] = kept
    return selected_epochs


def _prepare_examples_for_channels(
    examples: Sequence[CalibrationExample],
    *,
    baseline_ms: int,
    channels_0idx: Sequence[int],
) -> List[PreparedCalibrationExample]:
    prepared: List[PreparedCalibrationExample] = []
    for ex in examples:
        selected_epochs = _slice_epochs_for_channels(ex.epochs_data, channels_0idx)
        if not selected_epochs:
            continue
        stim_keys, raw_averaged, _ = build_averaged_erp(
            selected_epochs,
            int(ex.time_ms.size),
            artifact_threshold_uv=ex.artifact_uv if ex.artifact_uv > 0 else None,
        )
        if not stim_keys:
            continue
        corrected = baseline_correction(raw_averaged, ex.time_ms, baseline_ms=baseline_ms)
        prepared.append(
            PreparedCalibrationExample(
                file=ex.file,
                expected=ex.expected,
                stim_keys=tuple(stim_keys),
                raw_averaged=raw_averaged,
                corrected=corrected,
                time_ms=ex.time_ms,
            )
        )
    return prepared


def _evaluate_prepared_configuration(
    prepared_examples: Sequence[PreparedCalibrationExample],
    *,
    window_x_ms: int,
    window_y_ms: int,
    channels_0idx: Sequence[int],
) -> CalibrationResult:
    predictions: List[CalibrationPrediction] = []

    for ex in prepared_examples:
        winner_idx, _mode_used, debug = compute_winner_metrics(
            stim_keys=list(ex.stim_keys),
            raw_averaged=ex.raw_averaged,
            corrected=ex.corrected,
            time_ms=ex.time_ms,
            window_x_ms=int(window_x_ms),
            window_y_ms=int(window_y_ms),
        )
        predicted = stim_key_to_tile_digit(ex.stim_keys[winner_idx])
        margin = float(debug.get("margin") or 0.0)
        predictions.append(
            CalibrationPrediction(
                file=ex.file,
                expected=ex.expected,
                predicted=int(predicted),
                correct=int(predicted) == int(ex.expected),
                margin=margin,
            )
        )

    total = len(predictions)
    correct = sum(1 for p in predictions if p.correct)
    avg_margin = (sum(p.margin for p in predictions) / total) if total else 0.0
    accuracy_pct = 100.0 * correct / total if total else 0.0
    return CalibrationResult(
        accuracy_pct=accuracy_pct,
        correct=correct,
        total=total,
        window_x_ms=int(window_x_ms),
        window_y_ms=int(window_y_ms),
        channels_0idx=tuple(int(c) for c in channels_0idx),
        average_margin_pct=100.0 * avg_margin,
        predictions=predictions,
    )


def evaluate_configuration(
    examples: Sequence[CalibrationExample],
    *,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
    channels_0idx: Sequence[int],
) -> CalibrationResult:
    prepared = _prepare_examples_for_channels(
        examples,
        baseline_ms=baseline_ms,
        channels_0idx=channels_0idx,
    )
    return _evaluate_prepared_configuration(
        prepared,
        window_x_ms=window_x_ms,
        window_y_ms=window_y_ms,
        channels_0idx=channels_0idx,
    )


def search_best_configuration(
    examples: Sequence[CalibrationExample],
    *,
    baseline_ms: int = 100,
    x_values: Optional[Sequence[int]] = None,
    y_values: Optional[Sequence[int]] = None,
    max_subset_size: Optional[int] = None,
    top_k: int = 10,
) -> List[CalibrationResult]:
    if not examples:
        return []

    n_channels = len(examples[0].channel_names)
    if x_values is None:
        x_values = list(range(0, 801, 25))
    if y_values is None:
        y_values = list(range(100, 801, 25))

    results: List[CalibrationResult] = []
    for channels_0idx in iter_channel_subsets(n_channels, max_subset_size=max_subset_size):
        prepared = _prepare_examples_for_channels(
            examples,
            baseline_ms=baseline_ms,
            channels_0idx=channels_0idx,
        )
        if not prepared:
            continue
        for window_x_ms in x_values:
            for window_y_ms in y_values:
                if int(window_y_ms) <= int(window_x_ms):
                    continue
                results.append(
                    _evaluate_prepared_configuration(
                        prepared,
                        window_x_ms=int(window_x_ms),
                        window_y_ms=int(window_y_ms),
                        channels_0idx=channels_0idx,
                    )
                )

    results.sort(
        key=lambda r: (
            r.accuracy_pct,
            r.correct,
            r.average_margin_pct,
            -len(r.channels_0idx),
            -(r.window_y_ms - r.window_x_ms),
        ),
        reverse=True,
    )
    return results[: max(1, int(top_k))]
