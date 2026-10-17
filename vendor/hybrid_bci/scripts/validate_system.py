#!/usr/bin/env python3
"""Full-system validation CLI (stable baseline a9b8b07+)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from validation.historical import build_historical_validation_report, confidence_from_historical
from validation.p300_synthetic import run_p300_synthetic_battery
from validation.report import render_master_markdown, write_json
from validation.runtime_harness import run_synthetic_protocol_session
from validation.ssvep_synthetic import run_ssvep_msi_battery, run_ssvep_online_smoke
from validation.stress import run_shutdown_stress


def _run_replay_gate(baseline: Path, label: str) -> dict:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "replay_sessions.py"),
        "compare",
        "--root",
        str(_ROOT / "experiment_runs"),
        "--baseline",
        str(baseline),
        "--pipeline-version",
        label,
        "--report",
        str(_ROOT / "docs" / "replay_reports" / f"{label}_replay.md"),
        "--diff-output",
        str(_ROOT / "docs" / "replay_reports" / f"{label}_replay_diff.json"),
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    status = "pass" if proc.returncode == 0 else "fail"
    unexpected = 0
    for line in (proc.stdout or "").splitlines():
        if "unexpected=" in line:
            try:
                unexpected = int(line.split("unexpected=")[1].split(",")[0])
            except ValueError:
                pass
    return {
        "status": status,
        "returncode": proc.returncode,
        "unexpected_diff_count": unexpected,
        "stdout_tail": (proc.stdout or "")[-500:],
    }


def cmd_historical(args: argparse.Namespace) -> int:
    report = build_historical_validation_report(
        root=Path(args.root),
        baseline_path=Path(args.baseline),
        pipeline_version=args.pipeline_version,
    )
    out_json = Path(args.output_json)
    write_json(out_json, report)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.get("per_session_table") or []
    lines = [
        "# Historical validation report",
        "",
        f"- Replay compare: **{report['replay_compare']['status']}** "
        f"(unexpected={report['replay_compare']['unexpected_diff_count']})",
        f"- Sessions: {len(rows)}",
        "",
        "## Per session",
        "",
        "| session | p300 | ssvep | known_issues | auc_margin_mean | tpl_margin_mean | degenerate |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('session')} | {r.get('p300_trials')} | {r.get('ssvep_blocks')} | "
            f"{r.get('known_issues')} | {r.get('auc_margin_mean')} | {r.get('template_margin_mean')} | "
            f"{r.get('ssvep_degenerate')} |"
        )
    lines.extend(["", "## Aggregate P300", "", "```json",])
    import json

    lines.append(json.dumps(report.get("aggregate", {}).get("p300_decisions"), ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json} and {md_path}")
    return 0 if report["replay_compare"]["status"] == "pass" else 1


def cmd_all(args: argparse.Namespace) -> int:
    gate = _run_replay_gate(Path(args.baseline), args.pipeline_version)
    hist = build_historical_validation_report(
        root=Path(args.root),
        baseline_path=Path(args.baseline),
        pipeline_version=args.pipeline_version,
    )
    p300 = run_p300_synthetic_battery()
    ssvep_msi = run_ssvep_msi_battery()
    ssvep_smoke = run_ssvep_online_smoke()
    with tempfile.TemporaryDirectory() as td:
        runtime = run_synthetic_protocol_session(Path(td))
        stress = run_shutdown_stress(Path(td))
    weak = []
    if int(gate.get("unexpected_diff_count") or 0) > 0:
        weak.append("Replay gate regression against golden baseline")
    if float((p300.get("summary") or {}).get("stable_rate") or 0) < 0.5:
        weak.append("P300 synthetic battery: low dual-decision stability")
    if float((ssvep_msi.get("summary") or {}).get("winner_match_rate") or 0) < 0.85:
        weak.append("SSVEP MSI synthetic battery below 0.85 winner match rate")
    if not ssvep_smoke.get("winner_match"):
        weak.append("SSVEP online smoke: winner mismatch on injected sine")
    if str(runtime.get("final_state")) != "stopped":
        weak.append(f"Synthetic runtime ended in state {runtime.get('final_state')!r}")
    if not stress.get("idempotent_ok"):
        weak.append("Shutdown stress: finalize not idempotent")

    master = {
        "schema": "hybrid_bci_system_validation_master/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stable_commit": "a9b8b07",
        "replay_gate": gate,
        "historical": hist,
        "synthetic": {
            "p300_battery": p300,
            "ssvep_msi_battery": ssvep_msi,
            "ssvep_online_smoke": ssvep_smoke,
            "runtime_protocol": runtime,
            "shutdown_stress": stress,
        },
        "confidence_summary": confidence_from_historical(hist),
        "weak_spots": weak,
        "research_limitations": [
            "Historical validation re-reads NDJSON logs; does not reprocess raw EEG files.",
            "Synthetic runtime patches adapters only; COM/LSL hardware not exercised.",
            "MSI path may require pythonnet/CLR on host (skipped or failed if unavailable).",
            "PsychoPy/subprocess paths not covered by synthetic runtime harness.",
        ],
    }
    out = Path(args.output_json)
    write_json(out, master)
    md = Path(args.output_md)
    md.write_text(render_master_markdown(master), encoding="utf-8")
    print(f"Wrote {out} and {md}")
    if weak:
        print("Weak spots:", "; ".join(weak))
    return 0 if not weak and gate.get("status") == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid BCI system validation harness")
    sub = p.add_subparsers(dest="command", required=True)
    h = sub.add_parser("historical", help="Re-summarize historical sessions vs golden baseline")
    h.add_argument("--root", default="experiment_runs")
    h.add_argument(
        "--baseline",
        default="tests/fixtures/replay_baselines/historical_replay_baseline.json",
    )
    h.add_argument("--pipeline-version", default="validation-historical")
    h.add_argument("--output-json", default="docs/replay_reports/validation_historical.json")
    h.add_argument("--output-md", default="docs/replay_reports/validation_historical.md")
    h.set_defaults(func=cmd_historical)

    a = sub.add_parser("all", help="Historical + synthetic + stress + master report")
    a.add_argument("--root", default="experiment_runs")
    a.add_argument(
        "--baseline",
        default="tests/fixtures/replay_baselines/historical_replay_baseline.json",
    )
    a.add_argument("--pipeline-version", default="validation-full")
    a.add_argument("--output-json", default="docs/replay_reports/validation_system_master.json")
    a.add_argument("--output-md", default="docs/replay_reports/validation_system_master.md")
    a.set_defaults(func=cmd_all)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
