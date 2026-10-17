#!/usr/bin/env python3
"""Step-by-step forensic for one p300_analyzer all_stims export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.p300_single_trial_forensic import (
    render_markdown,
    run_stepwise_forensic,
    write_forensic_plots,
)
from validation.report import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv_path",
        type=Path,
        help="Path to *_all_stims.csv export",
    )
    ap.add_argument("--optical-tile", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "docs" / "replay_reports")
    args = ap.parse_args()

    report = run_stepwise_forensic(args.csv_path, optical_tile=int(args.optical_tile))
    stem = Path(args.csv_path).stem
    json_p = args.out_dir / f"p300_single_trial_{stem}.json"
    md_p = args.out_dir / f"p300_single_trial_{stem}.md"
    plot_dir = args.out_dir / f"p300_single_trial_{stem}_plots"
    write_json(json_p, report)
    md_p.write_text(render_markdown(report), encoding="utf-8")
    plots = write_forensic_plots(report, plot_dir)
    print(f"Wrote {md_p}")
    print(f"AUC winner: {report.get('auc_winner_tile')} optical: {args.optical_tile}")
    for row in report.get("ranking_evolution") or []:
        print(
            f"  {row['stage']}: leader={row['leader_tile']} "
            f"rank_opt={row.get('optical_tile_rank', row.get('optical_tile_7_rank'))}"
        )
    if plots:
        print("Plots:", plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
