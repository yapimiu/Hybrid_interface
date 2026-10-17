"""Markdown/JSON report helpers for validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_section_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_no data_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_master_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Current system validation report",
        "",
        f"- **Stable baseline commit:** `{report.get('stable_commit', 'a9b8b07')}`",
        f"- **Generated:** {report.get('generated_at', '')}",
        f"- **Replay gate:** {report.get('replay_gate', {}).get('status', 'unknown')} "
        f"(unexpected={report.get('replay_gate', {}).get('unexpected_diff_count', '?')})",
        "",
        "## Confidence (real experiments)",
        "",
        str(report.get("confidence_summary", "")),
        "",
        "## Historical sessions (logged outputs)",
        "",
    ]
    hist = report.get("historical") or {}
    agg = hist.get("aggregate") or {}
    lines.append(f"- Sessions analyzed: **{agg.get('session_count', 0)}**")
    lines.append(f"- P300 trials: **{agg.get('p300_trials_total', 0)}**")
    lines.append(f"- SSVEP blocks: **{agg.get('ssvep_blocks_total', 0)}**")
    lines.append(f"- Sessions with known issues: **{agg.get('sessions_with_known_issues', 0)}**")
    lines.append("")
    lines.append("### P300 decision summary")
    lines.append("")
    p300 = agg.get("p300_decisions") or {}
    for k, v in sorted(p300.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### SSVEP coef summary")
    lines.append("")
    ssvep = agg.get("ssvep_coef") or {}
    for k, v in sorted(ssvep.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Lifecycle / timing")
    lines.append("")
    life = agg.get("lifecycle") or {}
    for k, v in sorted(life.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    syn = report.get("synthetic") or {}
    if syn:
        lines.extend(["## Synthetic validation", ""])
        for section, payload in syn.items():
            lines.append(f"### {section}")
            lines.append("")
            if isinstance(payload, dict):
                for k, v in sorted(payload.items()):
                    lines.append(f"- {k}: {v}")
            lines.append("")
    weak = report.get("weak_spots") or []
    if weak:
        lines.extend(["## Remaining weak spots", ""])
        for item in weak:
            lines.append(f"- {item}")
        lines.append("")
    lim = report.get("research_limitations") or []
    if lim:
        lines.extend(["## Research limitations", ""])
        for item in lim:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)
