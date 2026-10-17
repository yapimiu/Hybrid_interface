#!/usr/bin/env python3
"""Полный forensic audit: per-session метрики P300/SSVEP/EEG + aggregate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.open("r", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _safe_mean(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return None
    return round(float(np.mean(xs)), 5)


def _safe_median(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return None
    return round(float(np.median(xs)), 5)


def audit_p300(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = [e for e in events if e.get("event") == "trial_decision_dual"]
    if not decisions:
        return {"present": False, "n_trials": 0}
    auc_correct = tpl_correct = union_correct = 0
    auc_decided = tpl_decided = 0
    minepc: List[int] = []
    auc_margins: List[float] = []
    tpl_margins: List[float] = []
    auc_winners: List[int] = []
    tpl_winners: List[int] = []
    cue_tids: List[int] = []
    epochs_used: List[int] = []
    no_decision = 0
    calib_trials = main_trials = 0
    for ev in decisions:
        d = ev.get("data", {})
        phase = str(d.get("phase", ""))
        if phase == "calib":
            calib_trials += 1
        else:
            main_trials += 1
        cue = d.get("cue_target_tile_id")
        cue_tids.append(int(cue) if cue is not None else -1)
        auc = d.get("auc", {})
        tpl = d.get("template", {})
        if auc.get("can_decide"):
            auc_decided += 1
            w = auc.get("winner_tile_id")
            auc_winners.append(int(w) if w is not None else -1)
            if w == cue:
                auc_correct += 1
            vals = auc.get("debug", {}).get("final_metric_values") or []
            if vals:
                sv = sorted([abs(float(x)) for x in vals], reverse=True)
                if len(sv) >= 2 and sv[0] > 0:
                    auc_margins.append((sv[0] - sv[1]) / sv[0])
            mpc = auc.get("min_epochs_per_class")
            if mpc is not None:
                minepc.append(int(mpc))
        else:
            no_decision += 1
            auc_winners.append(-1)
        if tpl.get("can_decide"):
            tpl_decided += 1
            w = tpl.get("winner_tile_id")
            tpl_winners.append(int(w) if w is not None else -1)
            if w == cue:
                tpl_correct += 1
            vals = tpl.get("debug", {}).get("template_corr_values") or []
            if vals:
                sv = sorted([float(x) for x in vals if x is not None], reverse=True)
                if len(sv) >= 2:
                    tpl_margins.append(float(sv[0] - sv[1]))
        else:
            tpl_winners.append(-1)
        if auc.get("can_decide") or tpl.get("can_decide"):
            if (auc.get("winner_tile_id") == cue) or (tpl.get("winner_tile_id") == cue):
                union_correct += 1
        e_used = d.get("epochs_used_for_decision")
        if e_used is not None:
            epochs_used.append(int(e_used))
    n = len(decisions)
    agreement = sum(1 for a, t in zip(auc_winners, tpl_winners) if a == t and a != -1)
    return {
        "present": True,
        "n_trials": n,
        "n_calib": calib_trials,
        "n_main": main_trials,
        "auc_decided": auc_decided,
        "tpl_decided": tpl_decided,
        "no_decision": no_decision,
        "auc_accuracy": round(auc_correct / n, 4) if n else None,
        "tpl_accuracy": round(tpl_correct / n, 4) if n else None,
        "union_accuracy": round(union_correct / n, 4) if n else None,
        "auc_tpl_agreement": round(agreement / n, 4) if n else None,
        "min_epochs_per_class_mean": _safe_mean(minepc),
        "min_epochs_per_class_min": int(min(minepc)) if minepc else None,
        "epochs_used_mean": _safe_mean(epochs_used),
        "auc_margin_norm_mean": _safe_mean(auc_margins),
        "tpl_margin_mean": _safe_mean(tpl_margins),
    }


def audit_ssvep(events: List[Dict[str, Any]], msi_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ("continuous", "burst"):
        block_starts = [
            e for e in events if e.get("event") == "ssvep_block_start" and e.get("data", {}).get("mode") == mode
        ]
        block_ends = [
            e for e in events if e.get("event") == "ssvep_block_end" and e.get("data", {}).get("mode") == mode
        ]
        n_correct = 0
        n_total = 0
        coefS: List[float] = []
        for ev in block_ends:
            d = ev.get("data", {})
            tgt = d.get("target_lamp")
            w0 = d.get("winner_0idx")
            if tgt is not None and w0 is not None:
                n_total += 1
                if int(w0) == int(tgt):
                    n_correct += 1
            coef = d.get("coef")
            if isinstance(coef, list) and w0 is not None and 0 <= int(w0) < len(coef):
                v = coef[int(w0)]
                if v is not None:
                    coefS.append(float(v))
        trace_mode = [r for r in msi_trace if r.get("ssvep_mode") == mode and r.get("winner_1based") is not None]
        winners = [int(r["winner_1based"]) for r in trace_mode]
        switches = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i - 1])
        trace_coefS = [float(r["coefS"]) for r in trace_mode if r.get("coefS") is not None]
        trace_correct = 0
        trace_allowed = 0
        for r in trace_mode:
            if r.get("classify_allowed") and r.get("target_lamp_0idx") is not None:
                trace_allowed += 1
                if int(r.get("winner_0idx", -9)) == int(r["target_lamp_0idx"]):
                    trace_correct += 1
        by_mode[mode] = {
            "n_block_start": len(block_starts),
            "n_block_end": len(block_ends),
            "block_accuracy": round(n_correct / n_total, 4) if n_total else None,
            "n_blocks_scored": n_total,
            "block_coefS_mean": _safe_mean(coefS),
            "trace_n": len(trace_mode),
            "trace_switch_rate": round(switches / max(1, len(winners) - 1), 4) if len(winners) > 1 else None,
            "trace_accuracy": round(trace_correct / trace_allowed, 4) if trace_allowed else None,
            "trace_coefS_mean": _safe_mean(trace_coefS),
        }
    return by_mode


def audit_eeg(session_dir: Path) -> Dict[str, Any]:
    npz = session_dir / "eeg.npz"
    if not npz.is_file():
        return {"present": False}
    try:
        data = np.load(npz, mmap_mode="r")
    except Exception as e:
        return {"present": True, "error": str(e)}
    if "eeg" not in data.files:
        return {"present": False, "error": "no_eeg_key"}
    sig = data["eeg"]
    times = data["times"] if "times" in data.files else None
    n = int(sig.shape[0])
    if n < 1000:
        return {"present": True, "n_samples": n, "too_small": True}
    n_ch = int(sig.shape[1]) if sig.ndim == 2 else 1
    step = max(1, n // 200_000)
    chunk = np.asarray(sig[::step], dtype=np.float64)
    fs = None
    if times is not None and times.size > 10:
        ts = np.asarray(times[: min(50_000, times.size)], dtype=np.float64)
        dt = float(np.median(np.diff(ts)))
        fs = round(1.0 / dt, 1) if dt > 0 else None
    per_ch = []
    for c in range(chunk.shape[1] if chunk.ndim == 2 else 1):
        x = chunk[:, c] if chunk.ndim == 2 else chunk
        std = float(np.std(x))
        ptp = float(np.ptp(x))
        mean = float(np.mean(x))
        clip = float(np.mean(np.abs(x - mean) > 500.0))
        flat = std < 1e-3
        try:
            x_d = x - np.mean(x)
            head = float(np.mean(x_d[: max(1, x_d.size // 10)]))
            tail = float(np.mean(x_d[-max(1, x_d.size // 10) :]))
            drift = abs(tail - head)
        except Exception:
            drift = 0.0
        per_ch.append(
            {
                "ch": c,
                "std_uv": round(std, 2),
                "ptp_uv": round(ptp, 2),
                "clipping_frac": round(clip, 4),
                "flat": bool(flat),
                "drift_uv": round(drift, 2),
            }
        )
    stds = [p["std_uv"] for p in per_ch]
    return {
        "present": True,
        "fs_hz": fs,
        "n_samples": n,
        "n_channels": n_ch,
        "duration_sec": round(n / fs, 1) if fs else None,
        "std_min": round(min(stds), 2) if stds else None,
        "std_median": round(statistics.median(stds), 2) if stds else None,
        "std_max": round(max(stds), 2) if stds else None,
        "flat_channels": [p["ch"] for p in per_ch if p["flat"]],
        "high_clip_channels": [p["ch"] for p in per_ch if p["clipping_frac"] > 0.01],
        "per_channel": per_ch,
    }


def audit_timing(events: List[Dict[str, Any]], tech: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(e.get("event")) for e in events)
    tech_counts = Counter(str(e.get("event")) for e in tech)
    burst_done_events = [
        e for e in events if e.get("event") == "migalka_burst_complete"
    ]
    walls = []
    utils = []
    duties = []
    for ev in burst_done_events:
        d = ev.get("data", {})
        if d.get("wall_ms") is not None:
            walls.append(int(d["wall_ms"]))
        if d.get("stim_utilization") is not None:
            utils.append(float(d["stim_utilization"]))
        if d.get("duty_cycle") is not None:
            duties.append(float(d["duty_cycle"]))
    ingest = [e for e in tech if e.get("event") == "ingest_pipeline"]
    eeg_buf_sizes = []
    for e in ingest[:5000]:
        d = e.get("data", {})
        if d.get("ssvep_buf_after_ingest") is not None:
            eeg_buf_sizes.append(int(d["ssvep_buf_after_ingest"]))
    msi_decs = [e for e in tech if e.get("event") == "msi_decision"]
    inter_classify_sec: List[float] = []
    if len(msi_decs) >= 2:
        for a, b in zip(msi_decs[:-1], msi_decs[1:]):
            ta = a.get("data", {}).get("t_wall")
            tb = b.get("data", {}).get("t_wall")
            if ta is not None and tb is not None:
                dt = float(tb) - float(ta)
                if 0 < dt < 30:
                    inter_classify_sec.append(dt)
    return {
        "n_migalka_led": counts.get("migalka_led", 0),
        "n_eeg_chunks": tech_counts.get("eeg_chunk", 0),
        "n_ingest_pipeline": tech_counts.get("ingest_pipeline", 0),
        "n_runtime_snapshot": tech_counts.get("runtime_snapshot", 0),
        "burst_complete_n": len(burst_done_events),
        "burst_wall_ms_mean": _safe_mean(walls),
        "burst_stim_util_mean": _safe_mean(utils),
        "burst_duty_cycle_mean": _safe_mean(duties),
        "msi_decision_n": len(msi_decs),
        "msi_inter_classify_sec_mean": _safe_mean(inter_classify_sec),
        "msi_inter_classify_sec_median": _safe_median(inter_classify_sec),
    }


def audit_session(session_dir: Path) -> Dict[str, Any]:
    events = _read_ndjson(session_dir / "events.ndjson")
    tech = _read_ndjson(session_dir / "tech_trace.ndjson")
    msi_trace = _read_ndjson(session_dir / "msi_trace.ndjson")
    eeg = audit_eeg(session_dir)
    return {
        "session": session_dir.name,
        "n_events": len(events),
        "n_tech": len(tech),
        "p300": audit_p300(events),
        "ssvep": audit_ssvep(events, msi_trace),
        "eeg": {k: v for k, v in eeg.items() if k != "per_channel"},
        "timing": audit_timing(events, tech),
    }


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    real_human = [r for r in reports if _looks_human(r)]
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in real_human:
        subj = _subject_key(r["session"])
        by_subject[subj].append(r)

    def _agg_p300(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        auc = [r["p300"].get("auc_accuracy") for r in rs if r["p300"].get("auc_accuracy") is not None]
        tpl = [r["p300"].get("tpl_accuracy") for r in rs if r["p300"].get("tpl_accuracy") is not None]
        uni = [r["p300"].get("union_accuracy") for r in rs if r["p300"].get("union_accuracy") is not None]
        return {
            "n_sessions": len(rs),
            "auc_accuracy_mean": _safe_mean(auc),
            "tpl_accuracy_mean": _safe_mean(tpl),
            "union_accuracy_mean": _safe_mean(uni),
        }

    def _agg_ssvep(rs: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        accs = [r["ssvep"].get(mode, {}).get("block_accuracy") for r in rs]
        accs = [a for a in accs if a is not None]
        coefs = [r["ssvep"].get(mode, {}).get("block_coefS_mean") for r in rs]
        coefs = [a for a in coefs if a is not None]
        sw = [r["ssvep"].get(mode, {}).get("trace_switch_rate") for r in rs]
        sw = [a for a in sw if a is not None]
        return {
            "n_sessions": len(rs),
            "block_accuracy_mean": _safe_mean(accs),
            "coefS_mean": _safe_mean(coefs),
            "switch_rate_mean": _safe_mean(sw),
        }

    return {
        "n_total_sessions": len(reports),
        "n_human_sessions": len(real_human),
        "subjects": {
            subj: {
                "sessions": [r["session"] for r in rs],
                "p300": _agg_p300(rs),
                "ssvep_continuous": _agg_ssvep(rs, "continuous"),
                "ssvep_burst": _agg_ssvep(rs, "burst"),
            }
            for subj, rs in by_subject.items()
        },
        "overall_human": {
            "p300": _agg_p300(real_human),
            "ssvep_continuous": _agg_ssvep(real_human, "continuous"),
            "ssvep_burst": _agg_ssvep(real_human, "burst"),
        },
    }


def _looks_human(r: Dict[str, Any]) -> bool:
    name = r["session"].lower()
    bad_tokens = (
        "генератор", "тест", "fsdf", "gf", "анти", "геныч", "тестик", "прошивка",
        "fgds", "mklnj", "ыва", "длор", "долргоп", "kek", "кек", "fds_", "апвып",
        "l_m_ml", "lkjhm", "бро_бро", "вывап", "цй_цй",
    )
    if any(t in name for t in bad_tokens):
        return False
    if r["p300"].get("n_trials", 0) >= 5:
        return True
    bb = r["ssvep"].get("burst", {}).get("n_block_end", 0)
    bc = r["ssvep"].get("continuous", {}).get("n_block_end", 0)
    return (bb + bc) >= 5


def _subject_key(name: str) -> str:
    base = name.split("_", 2)[-1].lower()
    for prefix in ("влад", "халибеков", "бородин", "максим_бород", "бородка", "пронин"):
        if base.startswith(prefix):
            return prefix
    return base.split("_")[0]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("experiment_runs"))
    ap.add_argument("-o", "--output", type=Path, default=Path("docs/replay_reports/forensic_audit_full.json"))
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args(argv)

    sessions = sorted(p for p in args.root.iterdir() if p.is_dir())
    if args.names:
        sessions = [p for p in sessions if any(n in p.name for n in args.names)]

    reports: List[Dict[str, Any]] = []
    for s in sessions:
        try:
            r = audit_session(s)
        except Exception as e:
            r = {"session": s.name, "error": str(e)}
        reports.append(r)
        ok = r.get("p300", {}).get("n_trials", 0) or r.get("ssvep", {}).get("burst", {}).get("n_block_end", 0)
        print(f"  audited {s.name:<70} p300={r.get('p300',{}).get('n_trials',0):>3} burst={r.get('ssvep',{}).get('burst',{}).get('n_block_end',0):>3}")

    agg = aggregate(reports)
    payload = {"per_session": reports, "aggregate": agg}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nSaved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
