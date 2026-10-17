#!/usr/bin/env python3
"""Forensic P300 regression: dc897a1 vs current pipeline on same session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.p300_forensic_regression import render_forensic_markdown, run_forensic_regression
from validation.report import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "docs" / "replay_reports")
    args = ap.parse_args()

    report = run_forensic_regression(args.session_path)
    name = Path(report["session_path"]).name
    json_p = args.out_dir / f"p300_forensic_{name}.json"
    md_p = args.out_dir / f"p300_forensic_{name}.md"
    write_json(json_p, report)
    md_p.write_text(render_forensic_markdown(report), encoding="utf-8")
    print(f"Wrote {md_p}")
    for k, st in (report.get("variant_summary") or {}).items():
        print(f"  {k}: AUC optical={st.get('auc_vs_optical')}")
    print(f"Strongest: {report.get('strongest_auc_variant')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
