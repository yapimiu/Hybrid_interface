#!/usr/bin/env python3
"""Deep audit for P300 photodetector / generator sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.p300_simulator_audit import (
    audit_session,
    render_p300_simulator_markdown,
    write_audit_plots,
)
from validation.report import write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="P300 simulator / photodetector session audit")
    ap.add_argument(
        "session_path",
        type=Path,
        help="Session folder (with analysis/) or analysis/ directly",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "docs" / "replay_reports",
        help="Report output directory",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    report = audit_session(args.session_path)
    session_name = Path(report["session_path"]).name
    out_dir = Path(args.out_dir)
    json_path = out_dir / f"p300_simulator_audit_{session_name}.json"
    md_path = out_dir / f"p300_simulator_audit_{session_name}.md"
    plot_dir = out_dir / f"p300_simulator_audit_{session_name}_plots"

    write_json(json_path, report)
    md_path.write_text(render_p300_simulator_markdown(report), encoding="utf-8")
    plots: list[str] = []
    if not args.no_plots:
        plots = write_audit_plots(report, plot_dir)

    agg = report.get("aggregate") or {}
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if plots:
        print(f"Plots: {plot_dir}")
    print(
        f"Logged AUC acc={agg.get('logged_auc_accuracy')} "
        f"template={agg.get('logged_template_accuracy')}"
    )
    print(
        f"Re-run  AUC acc={agg.get('rerun_auc_accuracy')} "
        f"template={agg.get('rerun_template_accuracy')} "
        f"(matches logged: {agg.get('rerun_matches_logged_auc')}/{agg.get('n_trials')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
