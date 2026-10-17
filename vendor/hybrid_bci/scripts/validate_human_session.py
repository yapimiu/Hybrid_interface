#!/usr/bin/env python3
"""CLI: post-session human EEG audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from validation.human_session_audit import render_markdown, run_human_session_audit, write_audit_plots
from validation.report import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "docs" / "replay_reports")
    args = ap.parse_args()

    report = run_human_session_audit(args.session_dir)
    stem = Path(args.session_dir).name
    json_p = args.out_dir / f"human_session_{stem}.json"
    md_p = args.out_dir / f"human_session_{stem}.md"
    plot_dir = args.out_dir / f"human_session_{stem}_plots"
    write_json(json_p, report)
    md_p.write_text(render_markdown(report), encoding="utf-8")
    plots = write_audit_plots(report, plot_dir)
    print(f"Wrote {md_p}")
    print(f"Wrote {json_p}")
    if plots:
        print(f"Plots: {plot_dir}")
    v = report.get("verdict") or {}
    print("Verdict:", v.get("honest_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
