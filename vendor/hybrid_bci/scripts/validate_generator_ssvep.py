#!/usr/bin/env python3
"""Validate SSVEP/MSI on generator ground-truth sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.generator_ssvep_audit import (  # noqa: E402
    build_generator_validation_report,
    render_generator_markdown,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generator-based SSVEP validation")
    ap.add_argument(
        "--experiment-runs",
        type=Path,
        default=_REPO / "experiment_runs",
    )
    ap.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Single session dir (optional; default: all generator sessions)",
    )
    ap.add_argument(
        "--no-rerun-msi",
        action="store_true",
        help="Skip MSI re-run on recorded EEG",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_REPO / "docs/replay_reports/generator_ssvep_validation.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=_REPO / "docs/replay_reports/generator_ssvep_validation.md",
    )
    args = ap.parse_args()

    if args.session is not None:
        from validation.generator_ssvep_audit import analyze_generator_session

        report = {
            "schema": "hybrid_bci_generator_ssvep_validation/v1",
            "sessions": [
                analyze_generator_session(args.session, rerun_msi=not args.no_rerun_msi)
            ],
        }
        n = len(report["sessions"][0].get("blocks") or [])
        acc = report["sessions"][0].get("accuracy")
        report["aggregate"] = {"n_sessions": 1, "n_blocks": n, "accuracy": acc}
        report["verdict"] = {
            "current_msi_path_trustworthy_on_generator": acc == 1.0,
            "confidence_level": "high" if acc == 1.0 else "medium",
        }
    else:
        report = build_generator_validation_report(
            experiment_runs_root=args.experiment_runs,
            rerun_msi=not args.no_rerun_msi,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_generator_markdown(report), encoding="utf-8")

    agg = report.get("aggregate") or {}
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"blocks={agg.get('n_blocks')} accuracy={agg.get('accuracy')}")
    if args.session:
        ok = report["sessions"][0].get("accuracy") == 1.0
    else:
        ok = (report.get("verdict") or {}).get("current_msi_path_trustworthy_on_generator")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
