#!/usr/bin/env python3
"""Regression test for P300 offline analysis pipeline.

Reads one or more *_continuous.csv files, runs the same pipeline as the
analyzer (bandpass filter → epoch extraction → AUC winner selection) and
prints a table showing expected vs. actual tile for each session.

Usage
-----
    python scripts/regression_test.py path/to/*.csv
    python scripts/regression_test.py /path/to/data_dir/

The target tile is read from the ``target_tile_id`` column (added by the
new exporter) OR inferred from trial_start|target=N markers embedded inside
the ``marker`` column value if the column is absent/all-negative.

Parameters (can be tweaked via CLI flags):
    --baseline-ms   Pre-stimulus baseline window (default 100 ms)
    --x-ms          AUC window start after stimulus (default 550 ms)
    --y-ms          AUC window end after stimulus   (default 725 ms)
    --artifact-uv   Epoch artifact rejection threshold µV (default 60, 0=off)
    --channels      Comma-separated 1-based channel indices to use, e.g. 1,2,4
                    Default: channel 4 from the recommended global profile.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from p300_analysis.marker_parsing import decode_stim_tile_id
from p300_analysis.analysis_profiles import ANALYSIS_PROFILE_GENERAL, format_channels_1idx, get_analysis_profile
from p300_analysis.constants import EPOCH_DURATION_MS
from p300_analysis.signal_processing import bandpass_filter, common_average_reference, detect_bad_channels
from p300_analysis.erp_compute import (
    build_averaged_erp,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.winner_selection import WINNER_MODE_AUC


DEFAULT_PROFILE = get_analysis_profile(ANALYSIS_PROFILE_GENERAL)
DEFAULT_CHANNELS_1BASED = [int(ch) + 1 for ch in DEFAULT_PROFILE.roi_channels_0idx]
DEFAULT_CHANNELS_TEXT = format_channels_1idx(DEFAULT_PROFILE.roi_channels_0idx)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_file(
    path: Path,
    *,
    baseline_ms: int = DEFAULT_PROFILE.baseline_ms,
    x_ms: int = DEFAULT_PROFILE.window_x_ms,
    y_ms: int = DEFAULT_PROFILE.window_y_ms,
    artifact_uv: float = 60.0,
    channel_indices: Optional[List[int]] = None,
    use_car: bool = False,
) -> Dict:
    """Run P300 pipeline on one continuous CSV and return a result dict."""
    header, data_rows = _read_csv(path)
    idx = {c: i for i, c in enumerate(header)}

    required = {"t_rel_s", "marker"}
    missing = required - set(idx)
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {path.name}")

    channel_cols = [c for c in header if c.startswith("ch_")]
    if not channel_cols:
        raise RuntimeError(f"No ch_* columns in {path.name}")

    # Determine channels to use
    if channel_indices:
        selected = [ch - 1 for ch in channel_indices if 1 <= ch <= len(channel_cols)]
    else:
        selected = list(range(len(channel_cols)))
    if not selected:
        raise RuntimeError("No valid channels selected.")

    has_target_col = "target_tile_id" in idx

    t_rel: List[float] = []
    marker_vals: List[int] = []
    signal_rows: List[np.ndarray] = []   # per-sample, all selected channels
    target_ids: List[int] = []

    for row in data_rows:
        if len(row) < len(header):
            continue
        tr = _parse_num(row[idx["t_rel_s"]])
        mv = _parse_num(row[idx["marker"]])
        if tr is None or mv is None:
            continue
        vals: List[float] = []
        for ch_i in selected:
            col = idx.get(f"ch_{ch_i + 1}")
            if col is None or col >= len(row):
                continue
            v = _parse_num(row[col])
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        t_rel.append(float(tr))
        marker_vals.append(int(round(float(mv))))
        signal_rows.append(np.array(vals, dtype=np.float64))
        if has_target_col:
            tgt = _parse_num(row[idx["target_tile_id"]])
            target_ids.append(int(round(float(tgt))) if tgt is not None else -1)

    if len(t_rel) < 100:
        raise RuntimeError(f"Too few samples ({len(t_rel)}) in {path.name}")

    # Estimate sampling rate from t_rel
    dt = float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else 0.002
    fs = 1.0 / dt if dt > 0 else 500.0

    # Build 2-D signal array (T, C)
    sig_2d = np.stack(signal_rows)  # (T, C)

    # Bandpass filter + CAR
    sig_2d = bandpass_filter(sig_2d, fs)
    if use_car:
        sig_2d = common_average_reference(sig_2d)

    # Compute epoch length
    post_stim_ms = max(int(y_ms), int(EPOCH_DURATION_MS))
    epoch_len = int(round((baseline_ms + post_stim_ms) / (dt * 1000.0))) + 1
    pre_samples = int(round(float(baseline_ms) / (dt * 1000.0)))

    # Determine expected target from target_tile_id column
    expected_tile: Optional[int] = None
    if has_target_col and target_ids:
        valid_targets = [t for t in target_ids if t >= 0]
        if valid_targets:
            # Most frequent non-negative target in the file
            from collections import Counter
            expected_tile = Counter(valid_targets).most_common(1)[0][0]

    # Extract per-channel epochs (epoch_len, n_ch)
    epochs_data: Dict[str, List[np.ndarray]] = {}
    prev_tile: Optional[int] = None
    onset_count = 0
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
                onset_count += 1
        prev_tile = tile_id

    if onset_count == 0:
        raise RuntimeError(f"No onset markers found in {path.name}")

    # Build averaged ERP
    stim_keys, raw_averaged, rejected = build_averaged_erp(
        epochs_data,
        epoch_len,
        artifact_threshold_uv=artifact_uv if artifact_uv > 0 else None,
    )
    if not stim_keys:
        raise RuntimeError(f"No ERP data after artifact rejection in {path.name}")

    # Time axis: from -baseline_ms to epoch end
    time_ms = np.arange(epoch_len, dtype=np.float64) * (dt * 1000.0) - baseline_ms

    # Baseline correction + integration
    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw_averaged, time_ms, baseline_ms, x_ms, y_ms
    )

    # Winner selection
    winner_idx, mode_used, debug = compute_winner_metrics(
        stim_keys=stim_keys,
        raw_averaged=raw_averaged,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=wx,
        window_y_ms=wy,
        winner_mode=WINNER_MODE_AUC,
    )

    winner_key = stim_keys[winner_idx] if stim_keys else None
    winner_digit: Optional[int] = None
    if winner_key:
        try:
            winner_digit = int(winner_key.split("_")[-1])
        except Exception:
            pass

    margin = debug.get("margin")

    # Bad-channel report
    bad_idx, abs_means, stds = detect_bad_channels(sig_2d)
    bad_channels = [f"ch_{i+1}" for i in bad_idx]

    epochs_per_stim = {k: len(v) for k, v in epochs_data.items()}

    return {
        "file": path.name,
        "expected": expected_tile,
        "result": winner_digit,
        "correct": (expected_tile is not None and winner_digit == expected_tile),
        "margin_pct": round(float(margin) * 100, 1) if margin is not None else None,
        "n_onsets": onset_count,
        "n_classes": len(stim_keys),
        "rejected": dict(rejected) if rejected else {},
        "bad_channels": bad_channels,
        "epochs_per_stim": epochs_per_stim,
        "fs": round(fs, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*_continuous.csv")))
        elif pp.exists():
            files.append(pp)
        else:
            print(f"[WARN] not found: {pp}", file=sys.stderr)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CSV files or directories")
    parser.add_argument("--baseline-ms", type=int, default=DEFAULT_PROFILE.baseline_ms, metavar="MS")
    parser.add_argument("--x-ms", type=int, default=DEFAULT_PROFILE.window_x_ms, metavar="MS")
    parser.add_argument("--y-ms", type=int, default=DEFAULT_PROFILE.window_y_ms, metavar="MS")
    parser.add_argument("--artifact-uv", type=float, default=60.0, metavar="UV",
                        help="Epoch rejection threshold µV (0 = off). Default 60 µV for hardware-filtered (LPF 35 Hz) data")
    parser.add_argument("--channels", type=str, default="",
                        help=f"Comma-separated 1-based channel indices, e.g. 1,2,4. Default: {DEFAULT_CHANNELS_TEXT}")
    parser.add_argument("--all-channels", action="store_true",
                        help="Use all channels instead of the recommended default ROI.")
    parser.add_argument("--car", action="store_true",
                        help="Enable Common Average Reference (CAR). Default is OFF.")
    parser.add_argument("--no-car", action="store_true",
                        help="Disable Common Average Reference (CAR). Default is already OFF.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    channel_indices: Optional[List[int]] = list(DEFAULT_CHANNELS_1BASED)
    if args.all_channels:
        channel_indices = None
    elif args.channels:
        try:
            channel_indices = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
        except ValueError:
            print("[ERROR] --channels must be comma-separated integers", file=sys.stderr)
            sys.exit(1)

    files = _collect_files(args.paths)
    if not files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        try:
            r = analyse_file(
                f,
                baseline_ms=args.baseline_ms,
                x_ms=args.x_ms,
                y_ms=args.y_ms,
                artifact_uv=args.artifact_uv,
                channel_indices=channel_indices,
                use_car=(args.car and not args.no_car),
            )
            results.append(r)
        except Exception as e:
            results.append({"file": f.name, "error": str(e)})

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    SEP = "─" * 90
    print(SEP)
    print(f"{'Файл':<40} {'Цель':>5} {'Рез':>5} {'OK':>4} {'Margin':>8} {'Эпох':>6} {'BadCh'}")
    print(SEP)

    n_ok = 0
    n_with_expected = 0
    for r in results:
        if "error" in r:
            print(f"{'[ERR] ' + r['file']:<40}  {r['error']}")
            continue
        exp = str(r["expected"]) if r["expected"] is not None else "?"
        res = str(r["result"]) if r["result"] is not None else "?"
        ok = "✓" if r["correct"] else ("?" if r["expected"] is None else "✗")
        margin = f"{r['margin_pct']:+.1f}%" if r["margin_pct"] is not None else "  n/a"
        n_ep = r["n_onsets"]
        bad = ",".join(r["bad_channels"]) if r["bad_channels"] else "—"
        print(f"{r['file']:<40} {exp:>5} {res:>5} {ok:>4} {margin:>8} {n_ep:>6}  {bad}")
        if r["expected"] is not None:
            n_with_expected += 1
            if r["correct"]:
                n_ok += 1

        if args.verbose:
            print(f"    Fs={r['fs']} Hz  classes={r['n_classes']}")
            if r["rejected"]:
                print(f"    Отбраковано эпох: {r['rejected']}")
            if r["epochs_per_stim"]:
                print(f"    Эпох по классам: {r['epochs_per_stim']}")

    print(SEP)
    if n_with_expected > 0:
        accuracy = 100.0 * n_ok / n_with_expected
        print(f"Точность: {n_ok}/{n_with_expected}  ({accuracy:.1f}%)")
    else:
        print("Нет файлов с известной целевой плиткой (target_tile_id). "
              "Для автоматической проверки нужна колонка target_tile_id >= 0.")
    print()
    channels_info = "all" if channel_indices is None else ",".join(str(ch) for ch in channel_indices)
    print("Параметры анализа: "
          f"baseline={args.baseline_ms} мс, AUC=[{args.x_ms}–{args.y_ms}] мс, "
          f"artifact={args.artifact_uv} мкВ, channels={channels_info}")


if __name__ == "__main__":
    main()
