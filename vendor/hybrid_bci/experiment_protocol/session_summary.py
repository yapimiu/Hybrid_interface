"""Session-level aggregated summary (all_results.*) from existing logs only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUMMARY_SCHEMA = "hybrid_protocol/session_summary/v1"

P300_CSV_FIELDS = [
    "experiment_index",
    "phase",
    "target_tile",
    "auc_winner",
    "template_winner",
    "final_winner",
    "auc_correct",
    "template_correct",
    "agreement",
    "auc_margin",
    "template_margin",
    "epoch_count",
]

SSVEP_CSV_FIELDS = [
    "experiment_index",
    "ssvep_mode",
    "target_lamp",
    "majority_winner",
    "last_winner",
    "correct",
    "stability_pct",
    "switch_count",
    "coefS",
    "msi_trace_compact",
    "group_index",
    "block_in_group",
    "group_size",
]


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _margin_from_debug(debug: Any) -> str:
    if not isinstance(debug, dict):
        return ""
    m = debug.get("margin")
    if m is None:
        return ""
    try:
        return f"{float(m):.4f}"
    except (TypeError, ValueError):
        return ""


def _epoch_count(rec: Dict[str, Any]) -> int:
    counts = rec.get("epoch_counts_by_stim") or {}
    if not isinstance(counts, dict):
        return 0
    return int(sum(int(v) for v in counts.values()))


def _trace_summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    trace = rec.get("msi_decision_trace") or {}
    if not isinstance(trace, dict):
        return {}
    summ = trace.get("summary")
    return summ if isinstance(summ, dict) else {}


def _coef_compact(coef: Any) -> str:
    if not isinstance(coef, (list, tuple)):
        return ""
    parts: List[str] = []
    for i, v in enumerate(coef):
        if v is None:
            continue
        try:
            parts.append(f"L{i + 1}:{float(v):.4g}")
        except (TypeError, ValueError):
            continue
    return ",".join(parts)


def _exp_index(rec: Dict[str, Any], fallback: int) -> int:
    for key in ("experiment_record_id", "experiment_number", "experiment_index"):
        v = rec.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return int(fallback)


def build_p300_summary_row(rec: Dict[str, Any], *, experiment_index: int) -> Dict[str, Any]:
    auc = rec.get("auc") or {}
    tpl = rec.get("template_corr") or {}
    auc_w = auc.get("winner_tile_id")
    tpl_w = tpl.get("winner_tile_id")
    agree = rec.get("methods_agree")
    final_w = ""
    if agree is True and auc_w is not None:
        final_w = auc_w
    elif agree is False:
        final_w = ""

    def _bool_cell(v: Any) -> str:
        if v is None:
            return ""
        return "1" if bool(v) else "0"

    return {
        "experiment_index": int(experiment_index),
        "phase": str(rec.get("phase") or ""),
        "target_tile": rec.get("cue_target_tile_id", ""),
        "auc_winner": auc_w if auc_w is not None else "",
        "template_winner": tpl_w if tpl_w is not None else "",
        "final_winner": final_w,
        "auc_correct": _bool_cell(rec.get("correct_auc")),
        "template_correct": _bool_cell(rec.get("correct_template")),
        "agreement": _bool_cell(agree),
        "auc_margin": _margin_from_debug(auc.get("debug")),
        "template_margin": _margin_from_debug(tpl.get("debug")),
        "epoch_count": _epoch_count(rec),
    }


def build_ssvep_summary_row(rec: Dict[str, Any], *, experiment_index: int) -> Dict[str, Any]:
    summ = _trace_summary(rec)
    lamp = rec.get("cue_target_lamp_1based")
    if lamp is None and rec.get("cue_target_lamp_0idx") is not None:
        lamp = int(rec["cue_target_lamp_0idx"]) + 1
    maj = rec.get("winner_majority_1based", rec.get("winner_1based"))
    last = rec.get("winner_last_1based", rec.get("winner_1based"))
    switch = summ.get("false_transition_count", summ.get("switch_count", ""))
    stability = summ.get("stability_pct", "")
    compact = summ.get("compact_stream", "")
    if not compact:
        trace = rec.get("msi_decision_trace") or {}
        if isinstance(trace, dict):
            compact = trace.get("compact_stream", "")

    def _bool_cell(v: Any) -> str:
        if v is None:
            return ""
        return "1" if bool(v) else "0"

    return {
        "experiment_index": int(experiment_index),
        "ssvep_mode": str(rec.get("ssvep_mode") or ""),
        "target_lamp": lamp if lamp is not None else "",
        "majority_winner": maj if maj is not None else "",
        "last_winner": last if last is not None else "",
        "correct": _bool_cell(rec.get("correct")),
        "stability_pct": stability,
        "switch_count": switch,
        "coefS": _coef_compact(rec.get("coef")),
        "msi_trace_compact": compact,
        "group_index": rec.get("group_index", ""),
        "block_in_group": rec.get("block_in_group", ""),
        "group_size": rec.get("group_size", ""),
    }


def _accuracy(rows: Sequence[Dict[str, Any]], correct_key: str) -> Optional[float]:
    vals = [r.get(correct_key) for r in rows if r.get(correct_key) != ""]
    if not vals:
        return None
    hits = sum(1 for v in vals if str(v) == "1")
    return hits / len(vals)


def compute_totals(
    p300_rows: Sequence[Dict[str, Any]], ssvep_rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    p300_tot: Dict[str, Any] = {"n_experiments": len(p300_rows)}
    ssvep_tot: Dict[str, Any] = {"n_blocks": len(ssvep_rows)}
    auc_acc = _accuracy(p300_rows, "auc_correct")
    tpl_acc = _accuracy(p300_rows, "template_correct")
    agree_rate = _accuracy(p300_rows, "agreement")
    if auc_acc is not None:
        p300_tot["auc_accuracy"] = round(auc_acc, 4)
    if tpl_acc is not None:
        p300_tot["template_accuracy"] = round(tpl_acc, 4)
    if agree_rate is not None:
        p300_tot["methods_agreement_rate"] = round(agree_rate, 4)
    ssvep_acc = _accuracy(ssvep_rows, "correct")
    if ssvep_acc is not None:
        ssvep_tot["majority_accuracy"] = round(ssvep_acc, 4)
    cont = [r for r in ssvep_rows if str(r.get("ssvep_mode")) == "continuous"]
    burst = [r for r in ssvep_rows if str(r.get("ssvep_mode")) == "burst"]
    if cont:
        acc = _accuracy(cont, "correct")
        if acc is not None:
            ssvep_tot["continuous_accuracy"] = round(acc, 4)
    if burst:
        acc = _accuracy(burst, "correct")
        if acc is not None:
            ssvep_tot["burst_accuracy"] = round(acc, 4)
    return {"p300": p300_tot, "ssvep": ssvep_tot}


def build_session_summary_payload(
    experiments: Sequence[Dict[str, Any]], *, session_id: str = ""
) -> Dict[str, Any]:
    p300_rows: List[Dict[str, Any]] = []
    ssvep_rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(experiments, start=1):
        kind = str(rec.get("kind") or "").lower()
        exp_idx = _exp_index(rec, i)
        if kind == "p300":
            p300_rows.append(build_p300_summary_row(rec, experiment_index=exp_idx))
        elif kind == "ssvep":
            ssvep_rows.append(build_ssvep_summary_row(rec, experiment_index=exp_idx))
    return {
        "schema": SUMMARY_SCHEMA,
        "session_id": session_id,
        "p300": p300_rows,
        "ssvep": ssvep_rows,
        "totals": compute_totals(p300_rows, ssvep_rows),
    }


ALL_RESULTS_CSV_FIELDS = (
    ["section", "metric", "value", "experiment_index"]
    + [f for f in P300_CSV_FIELDS if f != "experiment_index"]
    + [f for f in SSVEP_CSV_FIELDS if f not in ("experiment_index",)]
)


def _write_all_results_csv(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_RESULTS_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in payload.get("p300") or []:
            w.writerow({"section": "p300", **{k: row.get(k, "") for k in P300_CSV_FIELDS}})
        for row in payload.get("ssvep") or []:
            w.writerow({"section": "ssvep", **{k: row.get(k, "") for k in SSVEP_CSV_FIELDS}})
        totals = payload.get("totals") or {}
        for block, prefix in (("p300", "p300"), ("ssvep", "ssvep")):
            block_tot = totals.get(block) or {}
            if not isinstance(block_tot, dict):
                continue
            for metric, value in block_tot.items():
                w.writerow(
                    {
                        "section": f"total_{prefix}",
                        "metric": metric,
                        "value": value,
                        "experiment_index": "",
                    }
                )


def format_pretty_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    sid = payload.get("session_id") or ""
    if sid:
        lines.append(f"Session: {sid}")
    lines.append(f"Schema: {payload.get('schema', '')}")
    lines.append("")

    def _table(title: str, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        if not rows:
            lines.append("(none)")
            lines.append("")
            return
        widths = [len(h) for h in headers]
        str_rows: List[List[str]] = []
        for row in rows:
            cells = [str(row.get(h, "")) for h in headers]
            str_rows.append(cells)
            widths = [max(w, len(c)) for w, c in zip(widths, cells)]
        fmt = "  ".join(f"{{:{w}}}" for w in widths)
        lines.append(fmt.format(*headers))
        for cells in str_rows:
            lines.append(fmt.format(*cells))
        lines.append("")

    p300 = payload.get("p300") or []
    ssvep = payload.get("ssvep") or []
    _table("P300", P300_CSV_FIELDS, p300)
    _table("SSVEP", SSVEP_CSV_FIELDS, ssvep)

    totals = payload.get("totals") or {}
    lines.append("TOTALS")
    lines.append("------")
    for block in ("p300", "ssvep"):
        block_tot = totals.get(block) or {}
        if not isinstance(block_tot, dict) or not block_tot:
            continue
        lines.append(f"[{block.upper()}]")
        for k, v in block_tot.items():
            lines.append(f"  {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def export_session_summary(session_dir: Path, *, session_id: str = "") -> Dict[str, Path]:
    """Write all_results.json/csv and all_results_pretty.txt into session_dir."""
    session_dir = Path(session_dir)
    experiments = _read_ndjson(session_dir / "experiments.ndjson")
    if not experiments:
        experiments = _load_experiments_from_analysis(session_dir / "analysis")
    if not session_id:
        manifest_path = session_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                session_id = str(manifest.get("session_id") or "")
            except Exception:
                session_id = session_dir.name
        else:
            session_id = session_dir.name

    payload = build_session_summary_payload(experiments, session_id=session_id)
    if not payload["p300"] and not payload["ssvep"]:
        return {}

    json_path = session_dir / "all_results.json"
    csv_path = session_dir / "all_results.csv"
    pretty_path = session_dir / "all_results_pretty.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_all_results_csv(csv_path, payload)

    pretty_path.write_text(format_pretty_summary(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "pretty": pretty_path}


def _load_experiments_from_analysis(analysis_dir: Path) -> List[Dict[str, Any]]:
    """Fallback when experiments.ndjson is missing (unit tests with analysis only)."""
    if not analysis_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for results_path in sorted(analysis_dir.glob("exp_*/results.csv")):
        try:
            with results_path.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                flat = next(reader, None)
            if not flat:
                continue
            kind = str(flat.get("kind") or "")
            rec: Dict[str, Any] = {
                "kind": kind,
                "phase": flat.get("phase"),
                "experiment_record_id": flat.get("exp_id"),
            }
            if kind == "p300":
                rec.update(
                    {
                        "cue_target_tile_id": _int_or_none(flat.get("target_tile")),
                        "auc": {
                            "winner_tile_id": _int_or_none(flat.get("auc_winner")),
                            "debug": {"margin": flat.get("auc_margin") or None},
                        },
                        "template_corr": {
                            "winner_tile_id": _int_or_none(flat.get("template_winner")),
                            "debug": {"margin": flat.get("template_margin") or None},
                        },
                        "methods_agree": _bool_or_none(flat.get("methods_agree")),
                        "correct_auc": _bool_or_none(flat.get("correct_auc")),
                        "correct_template": _bool_or_none(flat.get("correct_template")),
                    }
                )
            elif kind == "ssvep":
                rec.update(
                    {
                        "ssvep_mode": flat.get("ssvep_mode"),
                        "cue_target_lamp_1based": _int_or_none(flat.get("target_lamp")),
                        "winner_majority_1based": _int_or_none(flat.get("msi_winner_majority")),
                        "winner_last_1based": _int_or_none(flat.get("msi_winner_last")),
                        "winner_1based": _int_or_none(flat.get("msi_winner")),
                        "correct": _bool_or_none(flat.get("msi_correct")),
                        "coef": [
                            flat.get(f"msi_L{i}") for i in range(1, 7)
                        ],
                        "msi_decision_trace": {
                            "compact_stream": flat.get("msi_trace_compact", ""),
                            "summary": {
                                "stability_pct": flat.get("msi_trace_stability_pct"),
                                "false_transition_count": flat.get("msi_trace_false_transitions"),
                                "compact_stream": flat.get("msi_trace_compact", ""),
                            },
                        },
                    }
                )
            rows.append(rec)
        except Exception:
            continue
    return rows


def _int_or_none(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _bool_or_none(v: Any) -> Optional[bool]:
    if v is None or v == "":
        return None
    return str(v) == "1"


def summary_matches_experiments(
    session_dir: Path, payload: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """Validate summary rows against experiments.ndjson (for tests)."""
    experiments = _read_ndjson(Path(session_dir) / "experiments.ndjson")
    if payload is None:
        json_path = Path(session_dir) / "all_results.json"
        if not json_path.is_file():
            return False, ["missing all_results.json"]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    built = build_session_summary_payload(experiments)
    issues: List[str] = []
    if len(built.get("p300") or []) != len(payload.get("p300") or []):
        issues.append("p300 row count mismatch")
    if len(built.get("ssvep") or []) != len(payload.get("ssvep") or []):
        issues.append("ssvep row count mismatch")
    for key in ("p300", "ssvep"):
        for i, (a, b) in enumerate(zip(built.get(key) or [], payload.get(key) or [])):
            for field in (P300_CSV_FIELDS if key == "p300" else SSVEP_CSV_FIELDS):
                if str(a.get(field, "")) != str(b.get(field, "")):
                    issues.append(f"{key}[{i}].{field}: {a.get(field)!r} != {b.get(field)!r}")
                    break
    return (len(issues) == 0, issues)
