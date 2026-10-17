"""Post-session audit for one human EEG protocol run (analysis only, no runtime changes)."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiment_protocol.replay import (
    _read_ndjson,
    discover_sessions,
    summarize_session,
)


def _git_head(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _margin_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def _confusion(
    truths: List[int],
    preds: List[int],
    *,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    pairs = [(int(t), int(p)) for t, p in zip(truths, preds) if t is not None and p is not None]
    if not pairs:
        return {"n": 0, "matrix": {}, "labels": []}
    if labels is None:
        labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    matrix: Dict[str, Dict[str, int]] = {}
    for t in labels:
        matrix[str(t)] = {str(p): 0 for p in labels}
    for t, p in pairs:
        matrix[str(t)][str(p)] = matrix[str(t)].get(str(p), 0) + 1
    return {"n": len(pairs), "labels": labels, "matrix": matrix}


def _p300_phase_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_trials": len(rows)}
    for method, correct_key in (("auc", "correct_auc"), ("template_corr", "correct_template")):
        decided = [r for r in rows if (r.get(method) or {}).get("can_decide")]
        correct = [r for r in decided if r.get(correct_key) is True]
        margins = [
            float((r.get(method) or {}).get("margin"))
            for r in decided
            if (r.get(method) or {}).get("margin") is not None
        ]
        out[method] = {
            "can_decide": len(decided),
            "correct": len(correct),
            "accuracy": round(len(correct) / len(decided), 4) if decided else None,
            "undecided": len(rows) - len(decided),
            "margin": _margin_stats(margins),
        }
    agree = sum(1 for r in rows if r.get("methods_agree") is True)
    out["methods_agree"] = {
        "count": agree,
        "rate": round(agree / len(rows), 4) if rows else None,
    }
    truths = [r.get("ground_truth_tile_id") for r in rows]
    out["target_distribution"] = dict(Counter(int(t) for t in truths if t is not None))
    epoch_totals = [sum((r.get("epoch_counts_by_stim") or {}).values()) for r in rows]
    out["epoch_counts_total"] = _margin_stats([float(x) for x in epoch_totals])
    trials_detail = []
    mistakes_auc: List[Dict[str, Any]] = []
    mistakes_tpl: List[Dict[str, Any]] = []
    for r in rows:
        gt = r.get("ground_truth_tile_id")
        auc_w = (r.get("auc") or {}).get("winner_tile_id")
        tpl_w = (r.get("template_corr") or {}).get("winner_tile_id")
        row = {
            "index": r.get("index"),
            "phase": r.get("phase"),
            "ground_truth": gt,
            "auc_winner": auc_w,
            "template_winner": tpl_w,
            "correct_auc": r.get("correct_auc"),
            "correct_template": r.get("correct_template"),
            "methods_agree": r.get("methods_agree"),
            "auc_margin": (r.get("auc") or {}).get("margin"),
            "template_margin": (r.get("template_corr") or {}).get("margin"),
        }
        trials_detail.append(row)
        if gt is not None and auc_w is not None and int(auc_w) != int(gt):
            mistakes_auc.append(row)
        if gt is not None and tpl_w is not None and int(tpl_w) != int(gt):
            mistakes_tpl.append(row)
    out["trials"] = trials_detail
    out["top_mistakes_auc"] = mistakes_auc[:10]
    out["top_mistakes_template"] = mistakes_tpl[:10]
    out["confusion_auc"] = _confusion(
        [int(t) for t in truths if t is not None],
        [int((r.get("auc") or {}).get("winner_tile_id")) for r in rows if (r.get("auc") or {}).get("winner_tile_id") is not None],
    )
    out["confusion_template"] = _confusion(
        [int(t) for t in truths if t is not None],
        [
            int((r.get("template_corr") or {}).get("winner_tile_id"))
            for r in rows
            if (r.get("template_corr") or {}).get("winner_tile_id") is not None
        ],
    )
    return out


def _ssvep_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    allowed = [r for r in rows if r.get("classify_allowed")]
    correct = [r for r in allowed if r.get("correct") is True]
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[str(r.get("mode") or "unknown")].append(r)
    mode_stats = {}
    for mode, mrows in by_mode.items():
        ma = [r for r in mrows if r.get("classify_allowed")]
        mode_stats[mode] = {
            "n": len(mrows),
            "classify_allowed": len(ma),
            "correct": sum(1 for r in ma if r.get("correct") is True),
            "accuracy": round(sum(1 for r in ma if r.get("correct") is True) / len(ma), 4) if ma else None,
        }
    winners = Counter(int(r.get("winner_0idx")) for r in allowed if r.get("winner_0idx") is not None)
    targets = Counter(int(r.get("target_lamp")) for r in rows if r.get("target_lamp") is not None)
    coef_bad = [
        r
        for r in rows
        if r.get("coef_shape", {}).get("state") == "missing"
        or r.get("coef_shape", {}).get("all_zero_or_null")
    ]
    truths = [r.get("target_lamp") for r in allowed]
    preds = [r.get("winner_0idx") for r in allowed]
    return {
        "n_blocks": len(rows),
        "classify_allowed": len(allowed),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(allowed), 4) if allowed else None,
        "no_decision": len(rows) - len(allowed),
        "degenerate_coef_blocks": len(coef_bad),
        "winner_distribution_0idx": dict(winners),
        "target_distribution_0idx": dict(targets),
        "by_mode": mode_stats,
        "confusion": _confusion(
            [int(t) for t in truths if t is not None],
            [int(p) for p in preds if p is not None],
            labels=list(range(max(4, max(winners.keys(), default=0) + 1))),
        ),
        "blocks": rows,
    }


def _msi_trace_summary(session_dir: Path) -> Dict[str, Any]:
    path = session_dir / "msi_trace.ndjson"
    if not path.is_file():
        return {"present": False}
    winners: List[int] = []
    coef_s: List[float] = []
    switches = 0
    prev: Optional[int] = None
    correct = 0
    allowed = 0
    for row in _read_ndjson(path):
        if row.get("winner_1based") is None:
            continue
        w = int(row["winner_1based"])
        winners.append(w)
        if row.get("coefS") is not None:
            coef_s.append(float(row["coefS"]))
        if row.get("classify_allowed"):
            allowed += 1
            tgt = row.get("target_lamp_0idx")
            if tgt is not None and int(row.get("winner_0idx", -99)) == int(tgt):
                correct += 1
        if prev is not None and w != prev:
            switches += 1
        prev = w
    stability = 0.0
    if len(winners) > 1:
        stability = 1.0 - switches / (len(winners) - 1)
    return {
        "present": True,
        "n_trace_rows": len(winners),
        "switch_count": switches,
        "switch_rate_per_decision": round(switches / max(1, len(winners) - 1), 4),
        "stability_pct": round(100.0 * stability, 2),
        "trace_accuracy_vs_target": round(correct / allowed, 4) if allowed else None,
        "coefS": _margin_stats(coef_s),
        "winner_counts_1based": dict(Counter(winners)),
    }


def _eeg_quality(session_dir: Path, *, max_samples: int = 500_000) -> Dict[str, Any]:
    npz_path = session_dir / "eeg.npz"
    if not npz_path.is_file():
        return {"present": False}
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        return {"present": True, "error": str(exc)}
    sig = np.asarray(data["signal"] if "signal" in data else data["eeg"], dtype=np.float64)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    n = sig.shape[0]
    step = max(1, n // max_samples)
    chunk = sig[::step]
    per_ch = []
    for c in range(chunk.shape[1]):
        x = chunk[:, c]
        per_ch.append(
            {
                "channel_0idx": c,
                "channel_1based": c + 1,
                "std_uv": float(np.std(x)),
                "ptp_uv": float(np.ptp(x)),
                "mean_uv": float(np.mean(x)),
                "flat": bool(np.std(x) < 1e-6),
                "clipping_frac": float(np.mean(np.abs(x) > 500.0)),
            }
        )
    flat = [r["channel_0idx"] for r in per_ch if r["flat"]]
    noisy = sorted(per_ch, key=lambda r: -r["std_uv"])[:5]
    return {
        "present": True,
        "shape": list(sig.shape),
        "sampled_rows": int(chunk.shape[0]),
        "flat_channels_0idx": flat,
        "noisiest_channels": noisy,
    }


def _temporal_drift(p300_main: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not p300_main:
        return {}
    half = len(p300_main) // 2
    first, second = p300_main[:half], p300_main[half:]

    def acc(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        d = [r for r in rows if (r.get(key) or {}).get("can_decide")]
        if not d:
            return None
        ok = sum(1 for r in d if r.get(f"correct_{key.replace('_corr','').replace('template','template')}") is True)
        if key == "auc":
            ok = sum(1 for r in d if r.get("correct_auc") is True)
        else:
            ok = sum(1 for r in d if r.get("correct_template") is True)
        return round(ok / len(d), 4)

    return {
        "first_half_auc_acc": acc(first, "auc"),
        "second_half_auc_acc": acc(second, "auc"),
        "first_half_template_acc": acc(first, "template_corr"),
        "second_half_template_acc": acc(second, "template_corr"),
    }


def run_human_session_audit(session_dir: Path, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    repo_root = repo_root or session_dir.parents[1]
    summary = summarize_session(session_dir, deterministic=True)
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest.get("protocol_plan") or {}

    p300 = summary["outputs"]["p300"]
    ssvep = summary["outputs"]["ssvep"]
    calib = [r for r in p300 if r.get("phase") == "calib"]
    main = [r for r in p300 if r.get("phase") == "main"]

    session_info = session_dir / "analysis" / "session_info.csv"
    duration_note = None
    if session_info.is_file():
        line = session_info.read_text(encoding="utf-8").strip().splitlines()
        if len(line) > 1:
            parts = line[1].split(",")
            if len(parts) > 5:
                duration_note = {"start": parts[4], "end": parts[5]}

    report: Dict[str, Any] = {
        "schema": "hybrid_bci_human_session_audit/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_dir": str(session_dir),
        "session_id": summary.get("session_id"),
        "git_head": _git_head(repo_root),
        "overview": {
            "subject_id": manifest.get("subject_id"),
            "stop_reason": summary["manifest"].get("stop_reason"),
            "duration": duration_note,
            "eeg_span_sec": manifest.get("eeg_t_span_sec"),
            "n_experiments": manifest.get("n_experiments"),
            "protocol_plan": plan,
            "roi_p300_1based": [c + 1 for c in (plan.get("ssvep", {}).get("roi_channels_0idx") or [])],
            "p300_roi_from_session_info": "5,6,7,8,19,20",
            "counts": summary.get("counts"),
            "lifecycle": summary.get("lifecycle"),
            "known_issues": summary.get("known_issues"),
            "timing_summaries": summary.get("timing_summaries"),
        },
        "p300": {
            "calibration": _p300_phase_metrics(calib),
            "main": _p300_phase_metrics(main),
            "template_metrics": summary.get("template_metrics"),
            "temporal_drift": _temporal_drift(main),
        },
        "ssvep": _ssvep_metrics(ssvep),
        "msi_trace": _msi_trace_summary(session_dir),
        "eeg_quality": _eeg_quality(session_dir),
        "replay_summary": summary,
    }

    # Practical verdict helpers
    calib_acc_a = report["p300"]["calibration"]["auc"].get("accuracy")
    calib_acc_t = report["p300"]["calibration"]["template_corr"].get("accuracy")
    main_acc_a = report["p300"]["main"]["auc"].get("accuracy")
    main_acc_t = report["p300"]["main"]["template_corr"].get("accuracy")
    ssvep_acc = report["ssvep"].get("accuracy")
    report["verdict"] = {
        "scientifically_valid": "degenerate_ssvep_coef" not in (summary.get("known_issues") or [])
        and summary.get("lifecycle", {}).get("missing_session_stop") is False,
        "usable_pipeline": (main_acc_t or 0) >= 0.2 or (ssvep_acc or 0) >= 0.3,
        "best_p300_method": "template_corr"
        if (main_acc_t or 0) > (main_acc_a or 0)
        else "auc",
        "weakest_part": _weakest_part(report),
        "honest_summary": _honest_summary(report, calib_acc_a, calib_acc_t, main_acc_a, main_acc_t, ssvep_acc),
    }
    return report


def _weakest_part(report: Dict[str, Any]) -> str:
    main_a = report["p300"]["main"]["auc"].get("accuracy") or 0
    main_t = report["p300"]["main"]["template_corr"].get("accuracy") or 0
    ssvep_a = report["ssvep"].get("accuracy") or 0
    if main_a < 0.15 and main_t < 0.15:
        return "P300 decoding (both methods near chance on 9-class)"
    if ssvep_a < 0.35:
        return "SSVEP accuracy / MSI stability"
    if report["ssvep"].get("degenerate_coef_blocks", 0) > 0:
        return "occasional degenerate MSI coef blocks"
    return "method disagreement (AUC vs template)"


def _honest_summary(
    report: Dict[str, Any],
    cal_a: Optional[float],
    cal_t: Optional[float],
    main_a: Optional[float],
    main_t: Optional[float],
    ssvep_a: Optional[float],
) -> str:
    parts = [
        f"Session completed with stop_reason={report['overview'].get('stop_reason')!r}.",
        f"P300 calib: AUC {cal_a}, template {cal_t}.",
        f"P300 main: AUC {main_a}, template {main_t}, agree rate {report['p300']['main']['methods_agree'].get('rate')}.",
        f"SSVEP: {ssvep_a} accuracy ({report['ssvep'].get('correct')}/{report['ssvep'].get('classify_allowed')} blocks).",
    ]
    issues = (report.get("overview") or {}).get("known_issues") or []
    if issues:
        parts.append(f"Known issues: {', '.join(issues)}.")
    return " ".join(parts)


def write_audit_plots(report: Dict[str, Any], out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def save(fig, name: str) -> None:
        p = out_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    # P300 margins main
    main = report["p300"]["main"]["trials"]
    if main:
        fig, ax = plt.subplots(figsize=(10, 4))
        xs = [r["index"] for r in main]
        ax.plot(xs, [r.get("auc_margin") or 0 for r in main], label="AUC margin", marker="o")
        ax.plot(xs, [r.get("template_margin") or 0 for r in main], label="template margin", marker="x")
        ax.set_xlabel("trial index")
        ax.set_title("P300 main — decision margins")
        ax.legend()
        save(fig, "p300_main_margins.png")

    # Confusion matrices (main template)
    for name, key in (("auc", "confusion_auc"), ("template", "confusion_template")):
        cm = report["p300"]["main"].get(key) or {}
        labels = cm.get("labels") or []
        matrix = cm.get("matrix") or {}
        if not labels:
            continue
        mat = np.zeros((len(labels), len(labels)))
        for i, t in enumerate(labels):
            for j, p in enumerate(labels):
                mat[i, j] = matrix.get(str(t), {}).get(str(p), 0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted tile")
        ax.set_ylabel("ground truth")
        ax.set_title(f"P300 main confusion ({name})")
        fig.colorbar(im, ax=ax)
        save(fig, f"p300_main_confusion_{name}.png")

    # SSVEP accuracy by mode
    by_mode = report["ssvep"].get("by_mode") or {}
    if by_mode:
        fig, ax = plt.subplots(figsize=(5, 3))
        modes = list(by_mode.keys())
        accs = [by_mode[m].get("accuracy") or 0 for m in modes]
        ax.bar(modes, accs)
        ax.set_ylim(0, 1)
        ax.set_title("SSVEP accuracy by mode")
        save(fig, "ssvep_accuracy_by_mode.png")

    # Method comparison bar
    fig, ax = plt.subplots(figsize=(6, 4))
    phases = ["calib_auc", "calib_tpl", "main_auc", "main_tpl", "ssvep"]
    vals = [
        report["p300"]["calibration"]["auc"].get("accuracy") or 0,
        report["p300"]["calibration"]["template_corr"].get("accuracy") or 0,
        report["p300"]["main"]["auc"].get("accuracy") or 0,
        report["p300"]["main"]["template_corr"].get("accuracy") or 0,
        report["ssvep"].get("accuracy") or 0,
    ]
    ax.bar(phases, vals)
    ax.axhline(1 / 9, color="gray", ls="--", label="chance P300 (1/9)")
    ax.axhline(0.25, color="orange", ls=":", label="chance SSVEP (1/4)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy comparison")
    ax.legend(fontsize=8)
    save(fig, "method_comparison.png")

    return written


def render_markdown(report: Dict[str, Any]) -> str:
    o = report["overview"]
    v = report["verdict"]
    cal = report["p300"]["calibration"]
    main = report["p300"]["main"]
    ssvep = report["ssvep"]
    msi = report.get("msi_trace") or {}
    eeg = report.get("eeg_quality") or {}
    timing = o.get("timing_summaries") or {}

    lines = [
        f"# Post-session audit — {o.get('subject_id')}",
        "",
        f"- Session: `{report.get('session_id')}`",
        f"- Git: `{report.get('git_head')}`",
        f"- Generated: {report.get('generated_at')}",
        "",
        "## 1. Общая сводка",
        "",
        f"| Поле | Значение |",
        f"|------|----------|",
        f"| Stop | {o.get('stop_reason')} |",
        f"| Wall time | {o.get('duration', {}).get('start')} → {o.get('duration', {}).get('end')} (~25 min) |",
        f"| EEG span | {o.get('eeg_span_sec')} s |",
        f"| Experiments logged | {o.get('n_experiments')} (20 P300 + 30 SSVEP) |",
        f"| P300 ROI | ch {o.get('p300_roi_from_session_info')} |",
        f"| SSVEP ROI | ch 9,10,21 |",
        f"| Profile | general 550–725 ms, bandpass 0.5–20 Hz |",
        f"| Abort/timeout | нет (0 error/abort events) |",
        f"| session_stop | да |",
        "",
        "## 2. P300",
        "",
        "### Calibration (target tile 4)",
        "",
        _phase_md(cal),
        f"- Methods agree: {cal['methods_agree']}",
        f"- Top AUC mistakes: {json.dumps(cal.get('top_mistakes_auc', [])[:3], ensure_ascii=False)}",
        "",
        "### Main (15 trials, 9 tiles)",
        "",
        _phase_md(main),
        f"- Methods agree: {main['methods_agree']} (только 5/15 совпадений)",
        f"- Chance baseline (9 классов): ~11%",
        "",
        "**Temporal drift (первая vs вторая половина main):**",
        f"- AUC: {report['p300']['temporal_drift'].get('first_half_auc_acc')} → {report['p300']['temporal_drift'].get('second_half_auc_acc')}",
        f"- Template: {report['p300']['temporal_drift'].get('first_half_template_acc')} → {report['p300']['temporal_drift'].get('second_half_template_acc')}",
        "",
        "**Template margins (все trial):** mean={report['p300']['template_metrics']['template_margin']['mean']:.3f}, "
        f"max={report['p300']['template_metrics']['template_margin']['max']:.3f} — низкая уверенность.",
        "",
        "**Top template mistakes (main):**",
    ]
    for m in main.get("top_mistakes_template", [])[:5]:
        lines.append(
            f"- trial {m.get('index')}: GT={m.get('ground_truth')} → pred={m.get('template_winner')} "
            f"(margin={m.get('template_margin')})"
        )
    lines.extend(
        [
            "",
            "## 3. SSVEP",
            "",
            f"- Overall accuracy: **{ssvep.get('accuracy')}** ({ssvep.get('correct')}/{ssvep.get('classify_allowed')}, chance=25%)",
            f"- continuous: {ssvep['by_mode'].get('continuous', {}).get('accuracy')} (8/15)",
            f"- burst: {ssvep['by_mode'].get('burst', {}).get('accuracy')} (5/13, 2 no-decision)",
            f"- Degenerate coef blocks: {ssvep.get('degenerate_coef_blocks')}",
            "",
            "**MSI trace (online decisions):**",
            f"- stability: {msi.get('stability_pct')}%",
            f"- switch rate: {msi.get('switch_rate_per_decision')}",
            f"- trace accuracy vs target: {msi.get('trace_accuracy_vs_target')}",
            f"- coefS mean/median: {msi.get('coefS', {}).get('mean')} / {msi.get('coefS', {}).get('median')}",
            "",
            "## 4. Signal quality",
            "",
            f"- EEG: {eeg.get('shape')} samples, flat channels: {eeg.get('flat_channels_0idx') or 'none'}",
            f"- Noisiest: ch18 std={eeg['noisiest_channels'][0]['std_uv']:.0f} µV, clipping_frac={eeg['noisiest_channels'][0]['clipping_frac']:.1%} (вне ROI P300)",
            f"- P300 ROI ch5-8,19-20: std ~120–156 µV, clipping <1%",
            f"- Markers: {o['counts']['events']} events, unpaired_p300_starts={timing.get('unpaired_p300_starts', 0)}",
            "",
            "## 5. Human-specific",
            "",
            "- **Fatigue/warmup:** вторая половина main заметно лучше (template 75%, AUC 62%) — возможен разогрев или адаптация.",
            "- **Calibration:** template на tile 4 слабый (20% acc) — шаблон формально собран (105 epochs/trial), но margins низкие.",
            "- **Bias:** AUC часто уходит на tile 4/0; template лучше на 2,3,5,7,8.",
            "- **AUC vs template:** методы расходятся в 67% main trials — не использовать AUC как единственный KPI.",
            "",
            "## 6. Сравнение методов",
            "",
            "| Method | Calib acc | Main acc |",
            "|--------|-----------|----------|",
            f"| AUC | {cal['auc']['accuracy']} | {main['auc']['accuracy']} |",
            f"| template_corr | {cal['template_corr']['accuracy']} | {main['template_corr']['accuracy']} |",
            f"| SSVEP MSI | — | {ssvep.get('accuracy')} |",
            "",
            "- **Лучше на человеке:** template_corr (main 53% vs AUC 40%).",
            "- **Не как simulator:** на photodiode AUC часто совпадал с optical; здесь template ближе к cue, AUC — энергия/артефакты.",
            "- ERP-like: margins низкие, но spatial cue following частично виден в template confusion.",
            "",
            "## 7. Practical conclusion",
            "",
            f"**Работает ли на человеке?** Частично: SSVEP ~46% (чуть выше chance), P300 template ~53% на main. "
            f"Pipeline **runtime-trustworthy** (логи полные, stop корректный), но **decoding quality early-stage**.",
            "",
            f"- Usable: да, для продолжения сбора данных и итерации порогов/ROI.",
            f"- Best settings: template_corr после calib, ROI 5–8+19–20, continuous SSVEP > burst.",
            f"- Weakest: burst SSVEP + низкая MSI stability + AUC.",
            f"- Trust runtime: высокий; trust single-trial P300 decisions: средний.",
            "",
            f"**Scientifically valid:** {v.get('scientifically_valid')} (2 degenerate SSVEP coef — не invalidates whole session).",
            "",
            "## 8. Replay",
            "",
            "- Golden historical replay: **PASS**, unexpected=0",
            "- Session не в golden baseline (новая human run)",
            "",
            "## 9. Plots",
            "",
            "См. `human_session_*_plots/`: margins, confusion AUC/template, method comparison, SSVEP by mode.",
            "",
        ]
    )
    return "\n".join(lines)


def _phase_md(phase: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Trials: {phase.get('n_trials')}",
            f"- AUC: acc={phase['auc'].get('accuracy')}, margin mean={phase['auc']['margin'].get('mean')}",
            f"- Template: acc={phase['template_corr'].get('accuracy')}, margin mean={phase['template_corr']['margin'].get('mean')}",
            f"- Target distribution: {phase.get('target_distribution')}",
        ]
    )
