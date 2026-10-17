#!/usr/bin/env python3
"""Search subject-specific P300 window and ROI channels on labeled CSV runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from p300_analysis.calibration import load_examples_from_paths, search_best_configuration


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
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="CSV files or directories")
    parser.add_argument("--baseline-ms", type=int, default=100, metavar="MS")
    parser.add_argument("--artifact-uv", type=float, default=60.0, metavar="UV")
    parser.add_argument("--car", action="store_true", help="Enable CAR during calibration.")
    parser.add_argument("--latest-n", type=int, default=0, metavar="N",
                        help="Use only the latest N files after sorting by filename.")
    parser.add_argument("--step-ms", type=int, default=25, metavar="MS",
                        help="Grid step for latency search.")
    parser.add_argument("--max-subset-size", type=int, default=6, metavar="N",
                        help="Maximum ROI subset size to search.")
    parser.add_argument("--top", type=int, default=10, metavar="N")
    parser.add_argument("--save-json", type=str, default="", metavar="PATH")
    args = parser.parse_args()

    files = _collect_files(args.paths)
    if args.latest_n > 0:
        files = files[-int(args.latest_n):]
    if not files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    examples = load_examples_from_paths(
        files,
        baseline_ms=int(args.baseline_ms),
        artifact_uv=float(args.artifact_uv),
        use_car=bool(args.car),
    )
    if not examples:
        print("No calibration examples could be loaded.", file=sys.stderr)
        sys.exit(1)

    step_ms = max(1, int(args.step_ms))
    x_values = list(range(0, 801, step_ms))
    y_values = list(range(100, 801, step_ms))

    results = search_best_configuration(
        examples,
        baseline_ms=int(args.baseline_ms),
        x_values=x_values,
        y_values=y_values,
        max_subset_size=int(args.max_subset_size),
        top_k=int(args.top),
    )
    if not results:
        print("No calibration result.", file=sys.stderr)
        sys.exit(1)

    best = results[0]
    print(f"Loaded runs: {len(examples)}")
    print()
    print("Top candidates:")
    for idx, res in enumerate(results, start=1):
        channels_1idx = ",".join(str(c + 1) for c in res.channels_0idx)
        print(
            f"{idx:>2}. acc={res.accuracy_pct:5.1f}% ({res.correct}/{res.total})  "
            f"window=[{res.window_x_ms}-{res.window_y_ms}] ms  "
            f"channels={channels_1idx}  avg_margin={res.average_margin_pct:.1f}%"
        )

    print()
    print("Best profile:")
    print(f"  baseline_ms = {args.baseline_ms}")
    print(f"  window_x_ms = {best.window_x_ms}")
    print(f"  window_y_ms = {best.window_y_ms}")
    print(f"  channels_1idx = {','.join(str(c + 1) for c in best.channels_0idx)}")
    print()
    print(
        "Regression command:"
        f" python3 scripts/regression_test.py {' '.join(str(p) for p in files)}"
        f" --x-ms {best.window_x_ms} --y-ms {best.window_y_ms}"
        f" --channels {','.join(str(c + 1) for c in best.channels_0idx)}"
    )
    print()
    print("Per-file predictions for best profile:")
    for pred in best.predictions:
        ok = "✓" if pred.correct else "✗"
        print(
            f"  {pred.file:<36} target={pred.expected} pred={pred.predicted} "
            f"{ok} margin={pred.margin * 100:.1f}%"
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_ms": int(args.baseline_ms),
            "window_x_ms": int(best.window_x_ms),
            "window_y_ms": int(best.window_y_ms),
            "channels_0idx": [int(c) for c in best.channels_0idx],
            "channels_1idx": [int(c) + 1 for c in best.channels_0idx],
            "accuracy_pct": float(best.accuracy_pct),
            "correct": int(best.correct),
            "total": int(best.total),
            "average_margin_pct": float(best.average_margin_pct),
            "files": [str(p) for p in files],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Saved profile: {out_path}")


if __name__ == "__main__":
    main()
