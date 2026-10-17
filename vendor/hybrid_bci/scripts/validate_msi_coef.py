#!/usr/bin/env python3
"""Focused MSI/coef validation (DLL probe + runtime trace + historical scan)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.msi_coef_audit import (  # noqa: E402
    build_full_audit_report,
    render_audit_markdown,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="MSI/coef pipeline audit (validation-only)")
    ap.add_argument(
        "--experiment-runs",
        type=Path,
        default=_REPO / "experiment_runs",
        help="Root with session_* dirs for historical scan",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_REPO / "docs/replay_reports/msi_coef_runtime_validation.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=_REPO / "docs/replay_reports/msi_coef_runtime_validation.md",
    )
    args = ap.parse_args()

    report = build_full_audit_report(experiment_runs_root=args.experiment_runs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_audit_markdown(report), encoding="utf-8")

    verdict = report.get("verdict") or {}
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"current_runtime_path_ok={verdict.get('current_runtime_path_ok')}")
    print(f"historical_suspicious_rows={verdict.get('historical_suspicious_rows')}")
    return 0 if verdict.get("current_runtime_path_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
