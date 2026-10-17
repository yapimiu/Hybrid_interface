#!/usr/bin/env python3
"""Replay/regression CLI for historical hybrid BCI sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol.replay import (  # noqa: E402
    DEFAULT_ANALYSIS_VERSION,
    DEFAULT_PIPELINE_VERSION,
    REPLAY_MODE_DETERMINISTIC_LOG,
    build_baseline,
    compare_baselines,
    discover_sessions,
    load_baseline,
    render_markdown_report,
    render_structured_regression_report,
    write_json,
    write_markdown,
)


def _resolve_sessions(args: argparse.Namespace) -> List[Path]:
    if args.sessions:
        return [Path(p) for p in args.sessions]
    return discover_sessions(Path(args.root))


def _baseline_kwargs(args: argparse.Namespace) -> dict:
    deterministic = str(args.replay_mode) == REPLAY_MODE_DETERMINISTIC_LOG
    return {
        "label": args.label,
        "replay_mode": args.replay_mode,
        "pipeline_version": args.pipeline_version,
        "analysis_version": args.analysis_version,
        "deterministic": deterministic,
    }


def cmd_snapshot(args: argparse.Namespace) -> int:
    sessions = _resolve_sessions(args)
    if not sessions:
        print("No historical sessions found.", file=sys.stderr)
        return 2
    baseline = build_baseline(sessions, **_baseline_kwargs(args))
    write_json(Path(args.output), baseline)
    print(
        f"Wrote replay snapshot: {args.output} ({len(sessions)} sessions, "
        f"mode={args.replay_mode}, pipeline={args.pipeline_version})"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    sessions = _resolve_sessions(args)
    if not sessions:
        print("No historical sessions found.", file=sys.stderr)
        return 2
    expected = load_baseline(Path(args.baseline))
    actual = build_baseline(sessions, **_baseline_kwargs(args))
    diff = compare_baselines(expected, actual)
    if args.actual_output:
        write_json(Path(args.actual_output), actual)
    if args.diff_output:
        write_json(Path(args.diff_output), diff)
    if args.report:
        write_markdown(Path(args.report), render_markdown_report(diff, actual))
    if args.structured_report:
        write_markdown(
            Path(args.structured_report),
            render_structured_regression_report(diff, actual),
        )
    status = diff.get("status")
    print(
        f"Replay compare: {status} "
        f"(unexpected={diff.get('unexpected_diff_count', 0)}, "
        f"total={diff.get('diff_count', 0)}, "
        f"expected_fixes={diff.get('expected_fix_count', 0)})"
    )
    return 0 if status == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and compare deterministic replay snapshots from historical sessions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="Create a golden/current replay snapshot.")
    snap.add_argument("sessions", nargs="*", help="Session directories. Defaults to --root/session_*.")
    snap.add_argument("--root", default="experiment_runs", help="Historical sessions root.")
    snap.add_argument("--label", default="historical", help="Snapshot label.")
    snap.add_argument(
        "--replay-mode",
        default=REPLAY_MODE_DETERMINISTIC_LOG,
        help="Replay mode (deterministic_log uses log timestamps/seq only).",
    )
    snap.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    snap.add_argument("--analysis-version", default=DEFAULT_ANALYSIS_VERSION)
    snap.add_argument("--output", required=True, help="Output JSON path.")
    snap.set_defaults(func=cmd_snapshot)

    def _add_replay_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--replay-mode",
            default=REPLAY_MODE_DETERMINISTIC_LOG,
            help="Replay mode (deterministic_log uses log timestamps/seq only).",
        )
        p.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
        p.add_argument("--analysis-version", default=DEFAULT_ANALYSIS_VERSION)

    cmp_p = sub.add_parser("compare", help="Compare current replay snapshot with a golden baseline.")
    cmp_p.add_argument("sessions", nargs="*", help="Session directories. Defaults to --root/session_*.")
    cmp_p.add_argument("--root", default="experiment_runs", help="Historical sessions root.")
    cmp_p.add_argument("--label", default="current", help="Actual snapshot label.")
    cmp_p.add_argument("--baseline", required=True, help="Golden baseline JSON path.")
    cmp_p.add_argument("--actual-output", help="Optional actual snapshot JSON path.")
    cmp_p.add_argument("--diff-output", help="Optional diff JSON path.")
    cmp_p.add_argument("--report", help="Optional markdown report path.")
    cmp_p.add_argument("--structured-report", help="Optional structured markdown report path.")
    _add_replay_flags(cmp_p)
    cmp_p.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

