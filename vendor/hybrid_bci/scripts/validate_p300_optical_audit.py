#!/usr/bin/env python3
"""P300 simulator audit with manual optical (photodiode) ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.p300_optical_ground_truth_audit import (
    audit_session_optical,
    render_optical_audit_markdown,
    write_optical_audit_plots,
)
from validation.report import write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="P300 optical ground-truth audit")
    ap.add_argument("session_path", type=Path)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "docs" / "replay_reports",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    report = audit_session_optical(args.session_path)
    name = Path(report["session_path"]).name
    json_path = args.out_dir / f"p300_optical_audit_{name}.json"
    md_path = args.out_dir / f"p300_optical_audit_{name}.md"
    plot_dir = args.out_dir / f"p300_optical_audit_{name}_plots"

    write_json(json_path, report)
    md_path.write_text(render_optical_audit_markdown(report), encoding="utf-8")
    if not args.no_plots:
        write_optical_audit_plots(report, plot_dir)

    agg = report.get("aggregate") or {}
    print(f"Wrote {md_path}")
    print(
        f"template vs OPTICAL: {agg.get('template_vs_optical')} | "
        f"vs cue: {agg.get('template_vs_protocol_cue')}"
    )
    print(
        f"AUC vs OPTICAL: {agg.get('auc_vs_optical')} | vs cue: {agg.get('auc_vs_protocol_cue')}"
    )
    mo = agg.get("main_only") or {}
    print(f"MAIN template vs optical: {mo.get('template_vs_optical')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
