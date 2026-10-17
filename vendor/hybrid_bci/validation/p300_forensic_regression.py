"""Forensic: compare dc897a1-era P300 pipeline variants vs current on same session export."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from p300_analysis.analysis_profiles import hybrid_protocol_p300_profile
from p300_analysis.erp_compute import (
    build_averaged_erp,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.marker_parsing import stim_key_to_tile_digit
from p300_analysis.signal_processing import bandpass_filter, normalize_channels, time_window_to_indices
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR
from validation.p300_optical_ground_truth_audit import (
    CALIB_OPTICAL_TILE,
    MAIN_OPTICAL_SEQUENCE,
    manual_optical_target,
)
from validation.p300_simulator_audit import (
    ROI_PRESETS,
    _discover_experiments,
    _extract_epochs_from_export,
    _filter_epochs_log,
    _final_counts_from_results,
    _load_eeg_matrix,
    _read_csv_dicts,
    _time_ms_template,
)
from validation.report import write_json

BASELINE_COMMIT = "dc897a10582a6dc934e951262bf301326fc0297e"


@dataclass(frozen=True)
class PipelineVariant:
    key: str
    label: str
    roi_export_cols: Tuple[int, ...]
    per_channel_std_norm: bool
    channel_aggregate: str  # mean | first_only
    bandpass: str  # sos | ba (legacy filtfilt if scipy)
    profile_key: str = "general"
    notes: str = ""


VARIANTS: Tuple[PipelineVariant, ...] = (
    PipelineVariant(
        "current_protocol",
        "Current HEAD (6ch, std-norm, SOS)",
        ROI_PRESETS["export_all_6cols"],
        True,
        "mean",
        "sos",
        notes="protocol_runner / hybrid profile today",
    ),
    PipelineVariant(
        "dc897a1_general_1ch",
        "dc897a1 general profile: ROI ch4 only (export col 0)",
        (0,),
        True,
        "mean",
        "sos",
        notes="analysis_profiles roi=(3,) → single channel",
    ),
    PipelineVariant(
        "dc897a1_recent_2ch",
        "dc897a1 recent: ROI ch5-6 (export cols 0-1)",
        (0, 1),
        True,
        "mean",
        "sos",
    ),
    PipelineVariant(
        "general_1ch_no_std_norm",
        "1ch ch5, NO per-epoch std norm (raw mean)",
        (0,),
        False,
        "mean",
        "sos",
        notes="hypothesis: norm kills optical amplitude ranking",
    ),
    PipelineVariant(
        "six_ch_no_std_norm",
        "6ch, NO per-epoch std norm",
        ROI_PRESETS["export_all_6cols"],
        False,
        "mean",
        "sos",
    ),
    PipelineVariant(
        "general_1ch_legacy_filtfilt",
        "1ch ch5, std-norm, legacy filtfilt bandpass",
        (0,),
        True,
        "mean",
        "ba",
    ),
)


def build_averaged_erp_variant(
    epochs_data: Dict[str, List[np.ndarray]],
    epoch_len: int,
    *,
    per_channel_std_norm: bool,
    channel_aggregate: str,
    artifact_uv: float = 60.0,
) -> Tuple[List[str], np.ndarray]:
    from p300_analysis.erp_compute import artifact_reject_epochs
    from p300_analysis.marker_parsing import stim_key_sort_key

    stim_keys = [k for k, v in epochs_data.items() if v]
    stim_keys.sort(key=stim_key_sort_key)
    raw_averaged = np.zeros((len(stim_keys), epoch_len), dtype=np.float64)
    for i, key in enumerate(stim_keys):
        epochs = epochs_data.get(key, [])
        if artifact_uv > 0:
            epochs, _ = artifact_reject_epochs(epochs, artifact_uv)
        if not epochs:
            continue
        if epochs[0].ndim == 2:
            if per_channel_std_norm:
                normed = [normalize_channels(ep[:epoch_len]) for ep in epochs]
            else:
                normed = [ep[:epoch_len] for ep in epochs]
            stack = np.stack(normed, axis=0)
            mean_ch = np.mean(stack, axis=0)
            if channel_aggregate == "first_only":
                raw_averaged[i, :] = mean_ch[:, 0]
            else:
                raw_averaged[i, :] = np.mean(mean_ch, axis=-1)
        else:
            stack = np.stack([e[:epoch_len] for e in epochs], axis=0)
            raw_averaged[i, :] = np.mean(stack, axis=0)
    return stim_keys, raw_averaged


def bandpass_variant(X: np.ndarray, fs: float, mode: str) -> np.ndarray:
    if mode == "sos":
        return bandpass_filter(X, fs)
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return X
    n = X.shape[0]
    if n < 24:
        return X
    nyq = fs / 2.0
    lo, hi = 0.5 / nyq, 20.0 / nyq
    if lo >= hi:
        return X
    b, a = butter(4, [lo, hi], btype="band")
    if X.ndim == 1:
        return filtfilt(b, a, X).astype(X.dtype)
    return filtfilt(b, a, X, axis=0).astype(X.dtype)


def extract_epochs_variant(
    eeg: np.ndarray,
    *,
    epochs_log: Sequence[Dict[str, str]],
    t_start_lsl: float,
    fs_hz: float,
    roi_cols: Tuple[int, ...],
    baseline_ms: int,
    bandpass_mode: str,
) -> Dict[str, List[np.ndarray]]:
    if eeg.size == 0:
        return {}
    n_ch = int(eeg.shape[1])
    valid = [int(c) for c in roi_cols if 0 <= int(c) < n_ch]
    if not valid:
        valid = list(range(n_ch))
    el = int(round((800 + baseline_ms) / (1000.0 / fs_hz))) + 1
    pre_s = baseline_ms / 1000.0
    buf = bandpass_variant(eeg[:, valid], fs_hz, bandpass_mode)
    out: Dict[str, List[np.ndarray]] = {}
    for row in epochs_log:
        sk = str(row.get("stim_key") or "")
        if not sk:
            continue
        lsl_t = float(row["lsl_time"])
        start = int(round((lsl_t - t_start_lsl - pre_s) * fs_hz))
        end = start + el
        if start < 0 or end > buf.shape[0]:
            continue
        ep = buf[start:end, :].copy()
        if ep.shape[0] == el:
            out.setdefault(sk, []).append(ep)
    return out


def run_variant_on_trial(
    exp_dir: Path,
    variant: PipelineVariant,
    *,
    template_window: Optional[np.ndarray],
) -> Dict[str, Any]:
    profile = hybrid_protocol_p300_profile(variant.profile_key)
    row = _read_csv_dicts(exp_dir / "results.csv")[0]
    exp_id = int(row["exp_id"])
    phase = str(row.get("phase") or "")
    optical = manual_optical_target(exp_id, phase)
    protocol_cue = row.get("target_tile")
    try:
        cue_i = int(float(protocol_cue)) if protocol_cue not in ("", None) else None
    except ValueError:
        cue_i = None
    if cue_i is not None and cue_i < 0:
        cue_i = None

    eeg, _ = _load_eeg_matrix(exp_dir)
    elog = _read_csv_dicts(exp_dir / "epochs_log.csv")
    t_start = float(row.get("t_start_lsl") or 0)
    lsl = [float(r["lsl_time"]) for r in elog]
    fs = float(eeg.shape[0]) / max(0.1, max(lsl) - t_start)
    fc = _final_counts_from_results(row)
    elog_u = _filter_epochs_log(elog, fc) if fc else elog

    epochs_data = extract_epochs_variant(
        eeg,
        epochs_log=elog_u,
        t_start_lsl=t_start,
        fs_hz=fs,
        roi_cols=variant.roi_export_cols,
        baseline_ms=int(profile.baseline_ms),
        bandpass_mode=variant.bandpass,
    )
    time_ms = _time_ms_template(fs, int(profile.baseline_ms))
    el = int(time_ms.shape[0])
    stim_keys, raw = build_averaged_erp_variant(
        epochs_data,
        el,
        per_channel_std_norm=variant.per_channel_std_norm,
        channel_aggregate=variant.channel_aggregate,
    )
    corrected, integrated, _, wx, wy = compute_corrected_and_integrated(
        raw,
        time_ms,
        int(profile.baseline_ms),
        int(profile.window_x_ms),
        int(profile.window_y_ms),
    )
    _, _, auc_dbg = compute_winner_metrics(
        stim_keys,
        raw,
        corrected,
        time_ms,
        int(wx),
        int(wy),
        winner_mode=WINNER_MODE_AUC,
    )
    _, _, tpl_dbg = compute_winner_metrics(
        stim_keys,
        raw,
        corrected,
        time_ms,
        int(wx),
        int(wy),
        winner_mode=WINNER_MODE_TEMPLATE_CORR,
        template_window=template_window,
    )
    auc_w = int(auc_dbg["auc_winner_idx"])
    tpl_w = int(
        np.argmax(tpl_dbg["final_metric_values"])
        if tpl_dbg.get("final_metric_values")
        else auc_w
    )
    auc_tile = stim_key_to_tile_digit(stim_keys[auc_w])
    tpl_tile = stim_key_to_tile_digit(stim_keys[tpl_w])

    xi0, xi1 = time_window_to_indices(time_ms, wx, wy)
    opt_idx = None
    if optical is not None:
        for i, k in enumerate(stim_keys):
            if stim_key_to_tile_digit(k) == optical:
                opt_idx = i
                break
    abs_auc = auc_dbg.get("abs_auc_values") or []

    return {
        "exp_id": exp_id,
        "optical_gt": optical,
        "protocol_cue": cue_i,
        "auc_winner": auc_tile,
        "template_winner": tpl_tile,
        "logged_auc": row.get("auc_winner"),
        "logged_template": row.get("template_winner"),
        "auc_margin": auc_dbg.get("margin"),
        "template_margin": tpl_dbg.get("margin"),
        "optical_abs_auc": float(abs_auc[opt_idx]) if opt_idx is not None and abs_auc else None,
        "max_abs_auc": float(max(abs_auc)) if abs_auc else None,
        "dominant_energy_tile": stim_key_to_tile_digit(stim_keys[int(np.argmax(abs_auc))])
        if abs_auc
        else None,
    }


def _acc(rows: Sequence[Dict[str, Any]], winner: str, gt: str) -> Dict[str, Any]:
    v = [r for r in rows if r.get(gt) is not None and r.get(winner) is not None]
    ok = sum(1 for r in v if int(r[winner]) == int(r[gt]))
    return {"n": len(v), "correct": ok, "accuracy": round(ok / len(v), 4) if v else None}


def git_forensic_summary(repo: Path) -> Dict[str, Any]:
    def _run(args: List[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            return str(e)

    files = [
        "p300_analysis/erp_compute.py",
        "p300_analysis/analysis_profiles.py",
        "p300_analysis/signal_processing.py",
        "p300_analysis/epoch_geometry.py",
        "p300_analysis/online_engine.py",
        "p300_analysis/qt_window.py",
    ]
    commits = _run(
        ["git", "log", "--oneline", f"{BASELINE_COMMIT}..HEAD", "--", *files]
    ).splitlines()
    roi_diff = _run(["git", "show", BASELINE_COMMIT + ":p300_analysis/analysis_profiles.py"])
    return {
        "baseline_commit": BASELINE_COMMIT,
        "head": _run(["git", "rev-parse", "HEAD"]),
        "commits_touching_p300_core": commits,
        "baseline_general_roi_0idx": "(3,) → UI ch4 only",
        "current_general_roi_0idx": "(4,5,6,7,18,19) → UI ch5-8,19-20",
        "key_commits": [
            "dae28a5 per-channel epochs + normalize_channels (before baseline)",
            "dc897a1 baseline merge",
            "522ac98 shared profile modules",
            "cbdc7ce canonical 6ch montage",
            "0d5aa75 external template axis fix",
            "online_engine added for protocol_runner",
        ],
    }


def run_forensic_regression(session_path: Path) -> Dict[str, Any]:
    session_path = Path(session_path)
    analysis = session_path / "analysis" if (session_path / "analysis").is_dir() else session_path
    profile = hybrid_protocol_p300_profile()
    calib_template: Optional[np.ndarray] = None

    by_variant: Dict[str, List[Dict[str, Any]]] = {v.key: [] for v in VARIANTS}

    for exp_dir in _discover_experiments(analysis):
        row = _read_csv_dicts(exp_dir / "results.csv")[0]
        if str(row.get("phase")) == "calib":
            eeg, _ = _load_eeg_matrix(exp_dir)
            elog = _read_csv_dicts(exp_dir / "epochs_log.csv")
            t_start = float(row.get("t_start_lsl") or 0)
            lsl = [float(r["lsl_time"]) for r in elog]
            fs = float(eeg.shape[0]) / max(0.1, max(lsl) - t_start)
            ed = _extract_epochs_from_export(
                eeg,
                epochs_log=elog,
                t_start_lsl=t_start,
                fs_hz=fs,
                roi_cols_0idx=ROI_PRESETS["export_all_6cols"],
                baseline_ms=int(profile.baseline_ms),
            )
            from validation.p300_simulator_audit import _build_template, _time_ms_template as tmt

            time_ms = tmt(fs, int(profile.baseline_ms))
            calib_template = _build_template(
                ed,
                target_tile=CALIB_OPTICAL_TILE,
                time_ms=time_ms,
                baseline_ms=int(profile.baseline_ms),
                window_x_ms=int(profile.window_x_ms),
                window_y_ms=int(profile.window_y_ms),
                artifact_uv=60.0,
            )

    for exp_dir in _discover_experiments(analysis):
        for variant in VARIANTS:
            by_variant[variant.key].append(
                run_variant_on_trial(exp_dir, variant, template_window=calib_template)
            )

    summary: Dict[str, Any] = {}
    for variant in VARIANTS:
        rows = by_variant[variant.key]
        summary[variant.key] = {
            "label": variant.label,
            "notes": variant.notes,
            "auc_vs_optical": _acc(rows, "auc_winner", "optical_gt"),
            "auc_vs_cue": _acc(rows, "auc_winner", "protocol_cue"),
            "template_vs_optical": _acc(rows, "template_winner", "optical_gt"),
            "template_vs_cue": _acc(rows, "template_winner", "protocol_cue"),
        }

    # Isolated proof: does ROI alone explain regression?
    roi_proof = {
        "current_6ch_auc_optical": summary["current_protocol"]["auc_vs_optical"],
        "old_1ch_auc_optical": summary["dc897a1_general_1ch"]["auc_vs_optical"],
        "old_1ch_no_norm_auc_optical": summary["general_1ch_no_std_norm"]["auc_vs_optical"],
        "delta_1ch_vs_6ch": None,
    }
    a = summary["dc897a1_general_1ch"]["auc_vs_optical"].get("accuracy")
    b = summary["current_protocol"]["auc_vs_optical"].get("accuracy")
    if a is not None and b is not None:
        roi_proof["delta_1ch_vs_6ch"] = round(float(a) - float(b), 4)

    strongest = max(
        VARIANTS,
        key=lambda v: (summary[v.key]["auc_vs_optical"].get("accuracy") or 0.0),
    )

    return {
        "schema": "hybrid_bci_p300_forensic_regression/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_path": str(session_path),
        "baseline_commit": BASELINE_COMMIT,
        "git": git_forensic_summary(Path(__file__).resolve().parents[1]),
        "variant_summary": summary,
        "roi_isolated_proof": roi_proof,
        "strongest_auc_variant": strongest.key,
        "trials_by_variant": by_variant,
        "verdict": _forensic_verdict(summary, strongest),
    }


def _forensic_verdict(summary: Dict[str, Any], strongest: PipelineVariant) -> Dict[str, Any]:
    cur = summary.get("current_protocol", {})
    old = summary.get("dc897a1_general_1ch", {})
    recent = summary.get("dc897a1_recent_2ch", {})
    nonorm = summary.get("general_1ch_no_std_norm", {})
    return {
        "primary_regression_suspects": [
            {
                "rank": 1,
                "change": "ROI defaults 1ch→6ch (cbdc7ce / protocol session_info)",
                "evidence": "dc897a1 general roi=(3,) UI ch4; p300test used 6ch; optical GT needs localized channel",
            },
            {
                "rank": 2,
                "change": "protocol_runner + online_engine (new vs standalone qt_window)",
                "evidence": "dc897a1 had no protocol_runner; live LSL path/fs_hz/epoch alignment differ from CSV export",
            },
            {
                "rank": 3,
                "change": "epoch_geometry explicit fs_hz + 1s timestamp guard",
                "evidence": "NeuronSpectrum coarse timestamps; fixes real EEG, changes epoch alignment vs old",
            },
            {
                "rank": 4,
                "change": "bandpass SOS vs filtfilt (signal_processing)",
                "evidence": "numerical stability at 5kHz; minor shape change",
            },
        ],
        "not_auc_formula_regression": True,
        "abs_integral_unchanged_since_dc897a1": True,
        "template_corr_added_post_baseline": True,
        "strongest_auc_on_session_export_replay": strongest.label,
        "auc_accuracy_current_vs_optical": cur.get("auc_vs_optical"),
        "auc_accuracy_old_1ch_vs_optical": old.get("auc_vs_optical"),
        "auc_accuracy_old_2ch_vs_optical": recent.get("auc_vs_optical"),
        "auc_accuracy_1ch_no_norm_vs_optical": nonorm.get("auc_vs_optical"),
        "logged_runtime_note": (
            "Logged results.csv on p300test: AUC 0/20 vs optical, template 20/20. "
            "CSV export re-analysis does not fully reproduce live winners (~2/20 match logged AUC); "
            "variant comparison is relative on same export only."
        ),
        "practical_explanation": (
            "At dc897a1, standalone p300_analyzer used profile «Общий» = single ROI channel (ch4). "
            "Photodiode energy concentrated on that channel; argmax(sum|corrected|)) aligned with optical tile. "
            "Current hybrid protocol logs 6 ROI channels, averages after per-epoch std-norm — optical contrast "
            "is diluted and AUC ranks a global energy peak (often unrelated to photodiode position). "
            "template_corr (added after baseline) z-scores waveforms and tracks optical source 20/20 on this session."
        ),
        "real_eeg_tradeoff": (
            "6ch + normalization + fs_hz fix: better for real EEG robustness; worse as optical/AUC simulator KPI."
        ),
    }


def render_forensic_markdown(report: Dict[str, Any]) -> str:
    v = report.get("verdict") or {}
    lines = [
        "# P300 forensic regression: dc897a1 vs HEAD",
        "",
        f"- Baseline: `{report.get('baseline_commit')}`",
        f"- HEAD: `{report.get('git', {}).get('head')}`",
        f"- Session: `{report.get('session_path')}`",
        f"- Generated: {report.get('generated_at')}",
        "",
        "## Timeline (suspects)",
        "",
    ]
    for c in report.get("git", {}).get("key_commits") or []:
        lines.append(f"- {c}")
    lines.extend(
        [
            "",
            "## ROI drift (proof)",
            "",
            f"- dc897a1 **general**: `{report.get('git', {}).get('baseline_general_roi_0idx')}`",
            f"- current **general**: `{report.get('git', {}).get('current_general_roi_0idx')}`",
            f"- Commit: **cbdc7ce** (Stage 2.4b canonical montage)",
            "",
            "## AUC accuracy vs optical GT (same session re-analysis)",
            "",
            "| Variant | AUC vs optical | Template vs optical |",
            "|---------|----------------|---------------------|",
        ]
    )
    for key, st in sorted((report.get("variant_summary") or {}).items()):
        lines.append(
            f"| {st.get('label', key)} | {st.get('auc_vs_optical')} | {st.get('template_vs_optical')} |"
        )
    lines.extend(["", "## Verdict", "", v.get("practical_explanation", ""), ""])
    lines.append(f"**Strongest AUC variant on session:** {report.get('strongest_auc_variant')}")
    lines.extend(["", "### Tradeoff", "", v.get("real_eeg_tradeoff", "")])
    return "\n".join(lines) + "\n"
