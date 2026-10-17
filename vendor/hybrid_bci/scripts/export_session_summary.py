#!/usr/bin/env python3
"""Export all_results.* for an existing session directory (offline, no runtime impact)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_protocol.session_summary import export_session_summary, summary_matches_experiments


def main() -> int:
    p = argparse.ArgumentParser(description="Build all_results summary for a session folder")
    p.add_argument("session_dir", type=Path, help="Path to experiment_runs/session_xxx")
    p.add_argument("--check", action="store_true", help="Verify summary matches experiments.ndjson")
    args = p.parse_args()
    session_dir = args.session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        print(f"Not a directory: {session_dir}", file=sys.stderr)
        return 1
    paths = export_session_summary(session_dir)
    if not paths:
        print("No experiments found; nothing written.", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.check:
        ok, issues = summary_matches_experiments(session_dir)
        if not ok:
            for issue in issues:
                print(f"CHECK: {issue}", file=sys.stderr)
            return 3
        print("CHECK: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
