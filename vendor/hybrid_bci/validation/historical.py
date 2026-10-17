"""Historical session validation via deterministic log replay summaries."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiment_protocol.replay import (
    REPLAY_MODE_DETERMINISTIC_LOG,
    build_baseline,
    compare_baselines,
    discover_sessions,
    load_baseline,
    summarize_session,
)


def _p300_decision_stats(p300_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    auc_modes: Counter[str] = Counter()
    tpl_modes: Counter[str] = Counter()
    margins_auc: List[float] = []
    margins_tpl: List[float] = []
    agree = 0
    can_auc = 0
    can_tpl = 0
    for row in p300_rows:
        auc = row.get("auc") or {}
        tpl = row.get("template_corr") or {}
        if auc.get("can_decide"):
            can_auc += 1
            mu = auc.get("mode_used")
            if mu:
                auc_modes[str(mu)] += 1
            m = auc.get("margin")
            if isinstance(m, (int, float)):
                margins_auc.append(float(m))
        if tpl.get("can_decide"):
            can_tpl += 1
            mu = tpl.get("mode_used")
            if mu:
                tpl_modes[str(mu)] += 1
            m = tpl.get("margin")
            if isinstance(m, (int, float)):
                margins_tpl.append(float(m))
        if row.get("methods_agree") is True:
            agree += 1

    def _mstats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": round(sum(vals) / len(vals), 6),
        }

    n = len(p300_rows)
    return {
        "trials": n,
        "auc_can_decide": can_auc,
        "template_can_decide": can_tpl,
        "methods_agree": agree,
        "methods_agree_rate": round(agree / n, 4) if n else None,
        "auc_mode_used": dict(auc_modes),
        "template_mode_used": dict(tpl_modes),
        "auc_margin": _mstats(margins_auc),
        "template_margin": _mstats(margins_tpl),
    }


def _aggregate_summaries(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    p300_all: List[Dict[str, Any]] = []
    ssvep_all: List[Dict[str, Any]] = []
    known_issue_sessions = 0
    timing_unstable = 0
    for s in summaries:
        out = s.get("outputs") or {}
        p300_all.extend(out.get("p300") or [])
        ssvep_all.extend(out.get("ssvep") or [])
        if s.get("known_issues"):
            known_issue_sessions += 1
        ts = s.get("timing_summaries") or {}
        if (ts.get("unpaired_p300_starts") or 0) > 0 or (ts.get("unpaired_ssvep_starts") or {}):
            timing_unstable += 1

    coef = {"degenerate_blocks": 0, "blocks": 0, "correct": 0, "classify_allowed": 0}
    for row in ssvep_all:
        coef["blocks"] += 1
        sh = row.get("coef_shape") or {}
        if sh.get("state") == "missing" or sh.get("all_zero_or_null"):
            coef["degenerate_blocks"] += 1
        if row.get("correct") is True:
            coef["correct"] += 1
        if row.get("classify_allowed") is True:
            coef["classify_allowed"] += 1

    return {
        "session_count": len(summaries),
        "p300_trials_total": len(p300_all),
        "ssvep_blocks_total": len(ssvep_all),
        "sessions_with_known_issues": known_issue_sessions,
        "sessions_timing_anomalies": timing_unstable,
        "p300_decisions": _p300_decision_stats(p300_all),
        "ssvep_coef": {
            **coef,
            "correct_rate": round(coef["correct"] / coef["blocks"], 4)
            if coef["blocks"]
            else None,
            "degenerate_rate": round(coef["degenerate_blocks"] / coef["blocks"], 4)
            if coef["blocks"]
            else None,
        },
        "lifecycle": {
            "missing_session_stop": sum(
                1 for s in summaries if (s.get("lifecycle") or {}).get("missing_session_stop")
            ),
            "ssvep_mismatch_sessions": sum(
                1
                for s in summaries
                if (s.get("lifecycle") or {}).get("ssvep_start_end_mismatches")
            ),
        },
    }


def _session_paths_from_baseline(
    baseline: Dict[str, Any], root: Path
) -> List[Path]:
    root = Path(root)
    paths: List[Path] = []
    for row in baseline.get("sessions") or []:
        name = str(row.get("session_dir_name") or row.get("session_id") or "")
        if not name:
            continue
        p = root / name
        if p.is_dir():
            paths.append(p)
    if paths:
        return sorted(paths)
    return discover_sessions(root)


def build_historical_validation_report(
    *,
    root: Path,
    baseline_path: Path,
    pipeline_version: str = "validation-stable-a9b8b07",
) -> Dict[str, Any]:
    """Re-summarize historical logs and compare to frozen golden baseline."""
    root = Path(root)
    expected = load_baseline(Path(baseline_path))
    sessions = _session_paths_from_baseline(expected, root)
    summaries = [
        summarize_session(p, deterministic=True, replay_mode=REPLAY_MODE_DETERMINISTIC_LOG)
        for p in sessions
    ]
    actual = build_baseline(
        sessions,
        label="validation-current",
        pipeline_version=pipeline_version,
        analysis_version="replay-compact-v2",
        deterministic=True,
    )
    diff = compare_baselines(expected, actual)

    per_session_rows: List[Dict[str, Any]] = []
    for s in summaries:
        per_session_rows.append(
            {
                "session": s.get("session_dir_name"),
                "p300_trials": (s.get("counts") or {}).get("p300_trials"),
                "ssvep_blocks": (s.get("counts") or {}).get("ssvep_blocks"),
                "known_issues": ", ".join(s.get("known_issues") or []) or "none",
                "auc_margin_mean": (s.get("template_metrics") or {})
                .get("auc_margin", {})
                .get("mean"),
                "template_margin_mean": (s.get("template_metrics") or {})
                .get("template_margin", {})
                .get("mean"),
                "ssvep_degenerate": (s.get("coef_summaries") or {}).get("degenerate_blocks"),
            }
        )

    drift = {
        "status": diff.get("status"),
        "unexpected_diff_count": diff.get("unexpected_diff_count"),
        "expected_fix_count": diff.get("expected_fix_count"),
        "diff_count": diff.get("diff_count"),
        "classification": "no_drift"
        if int(diff.get("unexpected_diff_count") or 0) == 0
        else "unexpected_drift_detected",
    }

    aggregate = _aggregate_summaries(summaries)
    return {
        "schema": "hybrid_bci_historical_validation/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stable_commit": "a9b8b07",
        "note": (
            "Deterministic log re-summarization at stable architecture; "
            "does not re-run live LSL/COM or recompute classifiers from raw EEG."
        ),
        "session_paths": [str(p) for p in sessions],
        "replay_compare": drift,
        "diff_full": diff,
        "per_session_table": per_session_rows,
        "summaries": summaries,
        "aggregate": aggregate,
        "roi_montage": {
            "p300_roi_0idx": [5, 6, 7, 8, 19, 20],
            "ssvep_roi_0idx": [9, 10, 21],
            "verification": "logged outputs only; manifest channel counts per session",
        },
    }


def confidence_from_historical(report: Dict[str, Any]) -> str:
    gate = (report.get("replay_compare") or {}).get("status")
    unexpected = int((report.get("replay_compare") or {}).get("unexpected_diff_count") or 0)
    agg = report.get("aggregate") or {}
    issues = int(agg.get("sessions_with_known_issues") or 0)
    n = int(agg.get("session_count") or 0)
    if gate != "pass" or unexpected > 0:
        return (
            "**Low** for pipeline regression — replay gate failed; "
            "do not treat as stable for new science until resolved."
        )
    if issues >= n and n > 0:
        return (
            "**Medium** — replay summaries match golden baseline, but all historical "
            "sessions carry known legacy lifecycle/coef issues; use for regression only."
        )
    if issues > 0:
        return (
            "**Medium–High** — replay gate pass; some sessions have documented legacy "
            "anomalies (missing stop, SSVEP mismatch, degenerate coef). "
            "Suitable for controlled lab replication with checklist."
        )
    return (
        "**High** for regression stability — replay pass and clean lifecycle summaries; "
        "still subject to online hardware and operator procedure."
    )
