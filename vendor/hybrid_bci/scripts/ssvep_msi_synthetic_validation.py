#!/usr/bin/env python3
"""Run Stage 2.2 synthetic MSI validation and write JSON + markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ssvep_analysis.msi_validation import run_synthetic_battery, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="SSVEP MSI synthetic validation harness")
    parser.add_argument(
        "--output-json",
        default="docs/replay_reports/stage2_2_synthetic_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/replay_reports/stage2_2_synthetic_validation.md",
    )
    args = parser.parse_args()
    payload = run_synthetic_battery()
    write_report(payload, Path(args.output_json), Path(args.output_md))
    rate = payload["summary"]["winner_match_rate"]
    print(
        f"Synthetic MSI validation: {payload['summary']['winner_match']}/"
        f"{payload['summary']['cases']} winner matches (rate={rate})"
    )
    return 0 if rate >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
