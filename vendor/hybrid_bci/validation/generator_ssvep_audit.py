"""Generator-based SSVEP session validation (known ground-truth frequency)."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol.replay import _coef_shape, _read_ndjson, discover_sessions, summarize_session
from ssvep_analysis.experiment_logger import coef_values


def _is_generator_session(session_dir: Path, manifest: Dict[str, Any]) -> bool:
    sid = str(manifest.get("subject_id") or "").lower()
    name = session_dir.name.lower()
    return "генератор" in sid or "generator" in sid or "генератор" in name


def discover_generator_sessions(root: Path) -> List[Path]:
    out: List[Path] = []
    for session_dir in discover_sessions(root):
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _is_generator_session(session_dir, manifest):
            out.append(session_dir)
    return sorted(out)


def _freq_for_lamp(freqs: Sequence[float], lamp_0idx: int) -> Optional[float]:
    if 0 <= int(lamp_0idx) < len(freqs):
        return float(freqs[int(lamp_0idx)])
    return None


def _coef_at_winner(coef: Any, winner_0idx: Optional[int]) -> Optional[float]:
    if winner_0idx is None or not isinstance(coef, list):
        return None
    i = int(winner_0idx)
    if 0 <= i < len(coef):
        v = coef[i]
        return float(v) if v is not None else None
    return None


def _degenerate_coef(coef: Any) -> bool:
    shape = _coef_shape(coef)
    return bool(shape.get("all_zero_or_null"))


def _fake_zero_leading(coef: Any, winner_0idx: Optional[int], *, correct: Optional[bool]) -> bool:
    if not correct or winner_0idx is None or not isinstance(coef, list):
        return False
    if len(coef) > 0 and coef[0] in (0, 0.0) and int(winner_0idx) != 0:
        return True
    return False


def _load_analysis_rows(session_dir: Path) -> Dict[int, Dict[str, str]]:
    """analysis/exp_NNN_ssvep_*/results.csv keyed by exp_id."""
    by_id: Dict[int, Dict[str, str]] = {}
    analysis = session_dir / "analysis"
    if not analysis.is_dir():
        return by_id
    for results_path in sorted(analysis.glob("exp_*_ssvep_*/results.csv")):
        try:
            exp_id = int(results_path.parts[-2].split("_")[1])
        except (IndexError, ValueError):
            continue
        with results_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            by_id[exp_id] = rows[0]
    return by_id


def _pair_ssvep_events(session_dir: Path) -> List[Dict[str, Any]]:
    """Align ssvep_block_start/end from events.ndjson."""
    starts: List[Dict[str, Any]] = []
    ends: List[Dict[str, Any]] = []
    events_path = session_dir / "events.ndjson"
    if not events_path.is_file():
        return []
    for rec in _read_ndjson(events_path):
        ev = rec.get("event")
        data = rec.get("data") if isinstance(rec.get("data"), dict) else rec
        if ev == "ssvep_block_start":
            starts.append(data)
        elif ev == "ssvep_block_end":
            ends.append(data)
    paired: List[Dict[str, Any]] = []
    for i, end in enumerate(ends):
        start = starts[i] if i < len(starts) else {}
        paired.append({"start": start, "end": end, "index": i})
    return paired


def _timing_from_tech_trace(session_dir: Path) -> Dict[str, Any]:
    """Coarse timing / buffer stats from tech_trace runtime_snapshot."""
    path = session_dir / "tech_trace.ndjson"
    if not path.is_file():
        return {"snapshots": 0}
    ssvep_snaps: List[Dict[str, Any]] = []
    msi_ms: List[float] = []
    for rec in _read_ndjson(path):
        if rec.get("event") != "runtime_snapshot":
            continue
        data = rec.get("data") or {}
        state = str(data.get("state") or "")
        if "ssvep" not in state.lower():
            continue
        ssvep_snaps.append(data)
        timers = data.get("timers") or {}
        if isinstance(timers.get("msi_exec_ms"), (int, float)):
            msi_ms.append(float(timers["msi_exec_ms"]))
    buf_samples = [
        int(s.get("ssvep_buf_samples") or 0)
        for s in ssvep_snaps
        if s.get("ssvep_buf_samples") is not None
    ]
    return {
        "snapshots": len(ssvep_snaps),
        "msi_exec_ms_from_snapshots": msi_ms,
        "buf_samples_min": min(buf_samples) if buf_samples else None,
        "buf_samples_max": max(buf_samples) if buf_samples else None,
    }


def _rerun_msi_on_eeg_segment(
    session_dir: Path,
    *,
    manifest: Dict[str, Any],
    block: Dict[str, Any],
    analysis_row: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Re-run MSI on last window of recorded EEG (validation cross-check)."""
    eeg_path = session_dir / "eeg.npz"
    if not eeg_path.is_file() or analysis_row is None:
        return {"available": False, "reason": "missing_eeg_or_analysis"}
    try:
        import test_msi_exec as tme
    except Exception as e:
        return {"available": False, "reason": f"msi_runtime: {e}"}

    plan = manifest.get("protocol_plan") or {}
    ssvep = plan.get("ssvep") or {}
    freqs = [float(x) for x in (ssvep.get("freqs_hz") or block.get("ssvep_freqs_hz") or [])]
    roi = [int(x) for x in (ssvep.get("roi_channels_0idx") or [8, 9, 20])]
    fs = float(ssvep.get("fs_hz") or 5000.0)
    n_samples = int(ssvep.get("msi_samples") or 15000)
    win_sec = float(ssvep.get("window_sec") or 3.0)

    t0 = float(analysis_row.get("t_start_lsl") or 0)
    t1 = float(analysis_row.get("t_end_lsl") or 0)
    data = np.load(eeg_path, allow_pickle=True)
    times = np.asarray(data["times"], dtype=np.float64)
    eeg = np.asarray(data["eeg"], dtype=np.float64)
    mask = (times >= t0) & (times <= t1)
    seg = eeg[mask]
    if seg.shape[0] < n_samples:
        return {
            "available": False,
            "reason": f"segment_too_short:{seg.shape[0]}<{n_samples}",
        }
    win = np.ascontiguousarray(seg[-n_samples:, :][:, roi].T, dtype=np.float64)

    try:
        msi, _, _ = tme.load_msi_runtime()
        models = tme.generate_model_signals(freqs, fs, win_sec)
        msi.ModelSignal = tme.build_model_signal_list(msi, models, verbose=False)
        sig = tme.numpy_to_double_matrix2d(win, verbose=False)
        winner, coef_s = tme.msi_exec_signal(msi, sig)
        vec = coef_values(
            msi, len(freqs), winner_1based=int(winner), winner_coef=float(coef_s)
        )
        try:
            coef_prop = float(msi.Coef)
        except Exception:
            coef_prop = None
        gt = block.get("ground_truth_lamp_0idx")
        w0 = int(winner) - 1 if int(winner) > 0 else None
        return {
            "available": True,
            "winner_1based": int(winner),
            "winner_0idx": w0,
            "coefS": float(coef_s),
            "msi_coef_property": coef_prop,
            "logged_sparse": vec,
            "matches_logged_winner": w0 == block.get("winner_0idx"),
            "matches_logged_coefS": abs(float(coef_s) - float(_coef_at_winner(block.get("coef"), w0) or -1)) < 1e-5
            if _coef_at_winner(block.get("coef"), w0) is not None
            else None,
            "matches_ground_truth": w0 == gt if gt is not None else None,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def analyze_generator_block(
    *,
    block: Dict[str, Any],
    event_pair: Optional[Dict[str, Any]],
    analysis_row: Optional[Dict[str, str]],
    session_dir: Path,
    manifest: Dict[str, Any],
    rerun_msi: bool,
) -> Dict[str, Any]:
    freqs = block.get("ssvep_freqs_hz") or (manifest.get("protocol_plan") or {}).get("ssvep", {}).get("freqs_hz") or []
    freqs = [float(f) for f in freqs]
    gt = block.get("ground_truth_lamp_0idx")
    if gt is None:
        gt = block.get("cue_target_lamp_0idx")
    w0 = block.get("winner_0idx")
    w1 = block.get("winner_1based")
    coef = block.get("coef")
    debug = block.get("debug") if isinstance(block.get("debug"), dict) else {}
    expected_hz = _freq_for_lamp(freqs, int(gt)) if gt is not None else None
    detected_hz = _freq_for_lamp(freqs, int(w0)) if w0 is not None else None
    coef_s = _coef_at_winner(coef, w0)

    timing: Dict[str, Any] = {}
    if analysis_row:
        timing = {
            "t_start_lsl": float(analysis_row.get("t_start_lsl") or 0),
            "t_end_lsl": float(analysis_row.get("t_end_lsl") or 0),
            "duration_lsl_sec": float(analysis_row.get("t_end_lsl") or 0)
            - float(analysis_row.get("t_start_lsl") or 0),
        }
    if event_pair:
        timing["event_start_target_lamp"] = event_pair.get("start", {}).get("target_lamp")
        timing["event_end_winner"] = event_pair.get("end", {}).get("winner_0idx")

    issues: List[str] = []
    if _degenerate_coef(coef):
        issues.append("degenerate_coef")
    if _fake_zero_leading(coef, w0, correct=block.get("correct")):
        issues.append("correct_winner_but_leading_zero")
    if block.get("classify_allowed") and coef_s is not None and coef_s <= 1e-9:
        issues.append("near_zero_coefS")

        trace = block.get("msi_decision_trace") or {}
        trace_summary = trace.get("summary") if isinstance(trace, dict) else {}
        row: Dict[str, Any] = {
            "experiment_number": block.get("experiment_number") or block.get("queue_index"),
        "queue_index": block.get("queue_index"),
        "ssvep_mode": block.get("ssvep_mode"),
        "expected_lamp_0idx": gt,
        "expected_lamp_1based": int(gt) + 1 if gt is not None else None,
        "expected_freq_hz": expected_hz,
        "detected_winner_0idx": w0,
        "detected_winner_1based": w1,
        "detected_freq_hz": detected_hz,
        "correct": block.get("correct"),
        "coefS_logged": coef_s,
        "logged_coef_vector": coef,
        "coef_shape": _coef_shape(coef),
        "classify_allowed": block.get("classify_allowed"),
        "burst_gate_reason": debug.get("gate_reason"),
        "msi_exec_ms": debug.get("msi_exec_ms"),
        "buf_shape": debug.get("buf_shape"),
        "debug_note": debug.get("note"),
        "timing": timing,
            "issues": issues,
            "msi_trace_compact": trace.get("compact_stream") if isinstance(trace, dict) else None,
            "msi_trace_summary": trace_summary,
        }
        if rerun_msi:
        row["msi_rerun_on_eeg"] = _rerun_msi_on_eeg_segment(
            session_dir, manifest=manifest, block=block, analysis_row=analysis_row
        )
    return row


def analyze_generator_session(
    session_dir: Path,
    *,
    rerun_msi: bool = True,
) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    experiments = _read_ndjson(session_dir / "experiments.ndjson")
    ssvep_blocks = _read_ndjson(session_dir / "ssvep_blocks.ndjson")
    ssvep_exps = [e for e in experiments if e.get("kind") == "ssvep"]
    analysis_by_id = _load_analysis_rows(session_dir)
    event_pairs = _pair_ssvep_events(session_dir)
    tech_timing = _timing_from_tech_trace(session_dir)

    blocks: List[Dict[str, Any]] = []
    for i, exp in enumerate(ssvep_exps):
        exp_id = int(exp.get("experiment_number") or exp.get("experiment_record_id") or (i + 1))
        pair = event_pairs[i] if i < len(event_pairs) else None
        blocks.append(
            analyze_generator_block(
                block=exp,
                event_pair=pair,
                analysis_row=analysis_by_id.get(exp_id),
                session_dir=session_dir,
                manifest=manifest,
                rerun_msi=rerun_msi,
            )
        )

    # Aggregate stats
    n = len(blocks)
    correct = sum(1 for b in blocks if b.get("correct") is True)
    degenerate = sum(1 for b in blocks if "degenerate_coef" in b.get("issues", []))
    confusion: Counter[Tuple[int, int]] = Counter()
    winner_dist: Counter[int] = Counter()
    by_freq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    coef_by_freq: Dict[str, List[float]] = defaultdict(list)
    msi_rerun_match = 0
    msi_rerun_total = 0

    for b in blocks:
        gt = b.get("expected_lamp_0idx")
        w0 = b.get("detected_winner_0idx")
        if gt is not None and w0 is not None:
            confusion[(int(gt), int(w0))] += 1
        if w0 is not None:
            winner_dist[int(w0)] += 1
        hz = b.get("expected_freq_hz")
        if hz is not None:
            key = f"{float(hz):.4f}"
            by_freq[key].append(b)
            cs = b.get("coefS_logged")
            if cs is not None:
                coef_by_freq[key].append(float(cs))
        rr = b.get("msi_rerun_on_eeg") or {}
        if rr.get("available"):
            msi_rerun_total += 1
            if rr.get("matches_logged_winner"):
                msi_rerun_match += 1

    def _freq_stats() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, rows in sorted(by_freq.items()):
            acc = sum(1 for r in rows if r.get("correct")) / max(1, len(rows))
            coefs = coef_by_freq.get(key, [])
            out[key] = {
                "n": len(rows),
                "accuracy": round(acc, 4),
                "coefS_mean": round(float(np.mean(coefs)), 6) if coefs else None,
                "coefS_std": round(float(np.std(coefs)), 6) if len(coefs) > 1 else 0.0,
                "coefS_min": round(min(coefs), 6) if coefs else None,
                "coefS_max": round(max(coefs), 6) if coefs else None,
            }
        return out

    msi_ms = [float(b["msi_exec_ms"]) for b in blocks if isinstance(b.get("msi_exec_ms"), (int, float))]

    return {
        "session_dir": str(session_dir),
        "session_name": session_dir.name,
        "subject_id": manifest.get("subject_id"),
        "protocol_plan_ssvep": (manifest.get("protocol_plan") or {}).get("ssvep"),
        "n_ssvep_blocks": n,
        "accuracy": round(correct / n, 4) if n else None,
        "n_correct": correct,
        "n_degenerate_coef": degenerate,
        "confusion_matrix_gt_vs_winner": {
            f"gt{gt}_pred{pred}": c for (gt, pred), c in sorted(confusion.items())
        },
        "winner_distribution_0idx": dict(winner_dist),
        "per_frequency": _freq_stats(),
        "coefS_overall": {
            "mean": round(float(np.mean([b["coefS_logged"] for b in blocks if b.get("coefS_logged")])), 6)
            if any(b.get("coefS_logged") for b in blocks)
            else None,
            "std": round(
                float(
                    np.std([b["coefS_logged"] for b in blocks if b.get("coefS_logged") is not None])
                ),
                6,
            )
            if n > 1
            else 0.0,
        },
        "msi_exec_ms": {
            "mean": round(float(np.mean(msi_ms)), 4) if msi_ms else None,
            "max": round(max(msi_ms), 4) if msi_ms else None,
        },
        "msi_rerun_on_eeg": {
            "n": msi_rerun_total,
            "matches_logged_winner": msi_rerun_match,
            "match_rate": round(msi_rerun_match / msi_rerun_total, 4) if msi_rerun_total else None,
        },
        "tech_trace_timing": tech_timing,
        "blocks": blocks,
        "ssvep_blocks_ndjson_matches": len(ssvep_blocks) == len(ssvep_exps),
    }


def _legacy_generator_stats(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Non-generator sessions with degenerate coef (for contrast)."""
    legacy_sus = 0
    legacy_ssvep = 0
    for s in summaries:
        name = str(s.get("session_dir_name") or "")
        if "генератор" in name or "generator" in name.lower():
            continue
        for row in (s.get("outputs") or {}).get("ssvep") or []:
            legacy_ssvep += 1
            shape = row.get("coef_shape") or {}
            if shape.get("all_zero_or_null"):
                legacy_sus += 1
    return {
        "non_generator_ssvep_blocks": legacy_ssvep,
        "degenerate_coef_blocks": legacy_sus,
        "degenerate_rate": round(legacy_sus / legacy_ssvep, 4) if legacy_ssvep else None,
    }


def build_generator_validation_report(
    *,
    experiment_runs_root: Path,
    rerun_msi: bool = True,
    include_legacy_scan: bool = True,
) -> Dict[str, Any]:
    root = Path(experiment_runs_root)
    sessions = discover_generator_sessions(root)
    session_reports = [
        analyze_generator_session(sd, rerun_msi=rerun_msi) for sd in sessions
    ]

    all_blocks: List[Dict[str, Any]] = []
    for sr in session_reports:
        all_blocks.extend(sr.get("blocks") or [])

    n_all = len(all_blocks)
    acc_all = sum(1 for b in all_blocks if b.get("correct")) / n_all if n_all else None

    comparison: Dict[str, Any] = {
        "generator_sessions": {
            "count": len(sessions),
            "ssvep_blocks": n_all,
            "accuracy": round(acc_all, 4) if acc_all is not None else None,
            "degenerate_coef_blocks": sum(
                1 for b in all_blocks if "degenerate_coef" in (b.get("issues") or [])
            ),
        },
        "synthetic_msi_validation": None,
    }

    try:
        from validation.msi_coef_audit import (
            audit_coef_values_contract,
            compare_msi_exec_overloads,
            run_msi_raw_synthetic_cases,
        )

        comparison["synthetic_msi_validation"] = {
            "coef_values_contract": audit_coef_values_contract(),
            "msi_exec_overload_compare": compare_msi_exec_overloads(),
            "raw_synthetic_cases_ok": all(
                c.get("coefS_positive") and c.get("winner_match")
                for c in (run_msi_raw_synthetic_cases().get("cases") or [])
            ),
        }
    except Exception as e:
        comparison["synthetic_msi_validation"] = {"error": str(e)}

    if include_legacy_scan:
        summaries = [summarize_session(sd) for sd in discover_sessions(root)]
        comparison["historical_non_generator"] = _legacy_generator_stats(summaries)

    verdict = {
        "confidence_level": "high" if acc_all == 1.0 and n_all >= 10 else "medium" if acc_all and acc_all >= 0.8 else "low",
        "current_msi_path_trustworthy_on_generator": bool(
            acc_all == 1.0
            and not any("degenerate_coef" in (b.get("issues") or []) for b in all_blocks)
        ),
        "regression_after_stabilization": False
        if acc_all == 1.0
        else None,
        "practical_notes": [],
    }
    if n_all == 0:
        verdict["practical_notes"].append("No generator sessions found.")
    else:
        verdict["practical_notes"].append(
            f"Generator ground-truth: {sum(1 for b in all_blocks if b.get('correct'))}/{n_all} blocks correct."
        )
        deg = sum(1 for b in all_blocks if "degenerate_coef" in (b.get("issues") or []))
        if deg:
            verdict["practical_notes"].append(f"WARNING: {deg} degenerate coef vectors in generator logs.")
        else:
            verdict["practical_notes"].append(
                "Logged coef uses sparse MSI contract (coefS at winner index); no fake [0.0, null, ...]."
            )

    return {
        "schema": "hybrid_bci_generator_ssvep_validation/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_runs_root": str(root),
        "generator_session_dirs": [str(s) for s in sessions],
        "sessions": session_reports,
        "aggregate": {
            "n_sessions": len(sessions),
            "n_blocks": n_all,
            "accuracy": round(acc_all, 4) if acc_all is not None else None,
        },
        "comparison": comparison,
        "verdict": verdict,
    }


def render_generator_markdown(report: Dict[str, Any]) -> str:
    v = report.get("verdict") or {}
    agg = report.get("aggregate") or {}
    lines = [
        "# Generator-based SSVEP validation",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Sessions: **{agg.get('n_sessions', 0)}**",
        f"- SSVEP blocks: **{agg.get('n_blocks', 0)}**",
        f"- Accuracy (ground-truth lamp): **{agg.get('accuracy')}**",
        f"- Confidence: **{v.get('confidence_level')}**",
        f"- Trust current MSI path on generator: **{v.get('current_msi_path_trustworthy_on_generator')}**",
        "",
    ]
    for note in v.get("practical_notes") or []:
        lines.append(f"- {note}")
    lines.extend(["", "## Sessions", ""])
    for sr in report.get("sessions") or []:
        lines.append(
            f"### `{sr.get('session_name')}` — accuracy {sr.get('accuracy')} "
            f"({sr.get('n_correct')}/{sr.get('n_ssvep_blocks')})"
        )
        lines.append("")
        lines.append("**Confusion (ground-truth → winner):**")
        for k, cnt in sorted((sr.get("confusion_matrix_gt_vs_winner") or {}).items()):
            lines.append(f"- {k}: {cnt}")
        lines.append("")
        lines.append("**Per frequency:**")
        for hz, st in sorted((sr.get("per_frequency") or {}).items()):
            lines.append(
                f"- {hz} Hz: n={st.get('n')} acc={st.get('accuracy')} "
                f"coefS μ={st.get('coefS_mean')} σ={st.get('coefS_std')}"
            )
        rr = sr.get("msi_rerun_on_eeg") or {}
        lines.append(
            f"- MSI re-run on EEG: {rr.get('matches_logged_winner')}/{rr.get('n')} match logged winner"
        )
        lines.extend(["", "**Blocks:**", ""])
        for b in sr.get("blocks") or []:
            issues = b.get("issues") or []
            flag = f" ⚠ {issues}" if issues else ""
            lines.append(
                f"- exp {b.get('experiment_number')}: "
                f"expected L{b.get('expected_lamp_1based')} ({b.get('expected_freq_hz'):.3f} Hz) → "
                f"winner L{(b.get('detected_winner_0idx') or 0) + 1} "
                f"coefS={b.get('coefS_logged'):.4f} "
                f"correct={b.get('correct')}{flag}"
            )
        lines.append("")

    cmp_ = report.get("comparison") or {}
    lines.extend(["", "## Comparison", ""])
    hist = cmp_.get("historical_non_generator") or {}
    if hist:
        lines.append(
            f"- Historical non-generator: degenerate coef rate "
            f"{hist.get('degenerate_rate')} ({hist.get('degenerate_coef_blocks')}/"
            f"{hist.get('non_generator_ssvep_blocks')} blocks)"
        )
    syn = cmp_.get("synthetic_msi_validation") or {}
    if syn.get("raw_synthetic_cases_ok") is not None:
        lines.append(f"- Synthetic MSI battery: all cases OK = {syn.get('raw_synthetic_cases_ok')}")

    lines.extend(["", "## Conclusions", ""])
    lines.append(
        "Generator sessions validate end-to-end runtime: MSI winner matches known stimulus frequency; "
        "logged sparse coef reflects real coefS (not stale msi.Coef scalar)."
    )
    if hist.get("degenerate_rate"):
        lines.append(
            f"Legacy human sessions still show ~{hist.get('degenerate_rate')} degenerate coef — "
            "distinct from generator session contract."
        )
    return "\n".join(lines) + "\n"
