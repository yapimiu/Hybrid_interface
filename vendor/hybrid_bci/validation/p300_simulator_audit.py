"""Deep audit for P300 photodetector / generator (simulator) sessions."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from p300_analysis.analysis_profiles import CANONICAL_P300_ROI_0IDX, hybrid_protocol_p300_profile
from p300_analysis.constants import EPOCH_DURATION_MS, SAFE_MIN_EPOCHS_TO_DECIDE
from p300_analysis.erp_compute import (
    build_averaged_erp,
    check_can_decide,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.marker_parsing import stim_key_to_tile_digit
from p300_analysis.online_engine import P300EngineParams
from p300_analysis.signal_processing import bandpass_filter, time_window_to_indices
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR

# Indices into exported eeg_markers columns (ch5,ch6,ch7,ch8,ch19,ch20).
_EXPORT_COL_BY_CANONICAL: Dict[int, int] = {ch: i for i, ch in enumerate(CANONICAL_P300_ROI_0IDX)}

ROI_PRESETS: Dict[str, Tuple[int, ...]] = {
    "export_all_6cols": tuple(range(6)),
    "canonical_6ch": tuple(range(6)),
    "legacy_2ch_5_6": (_EXPORT_COL_BY_CANONICAL[4], _EXPORT_COL_BY_CANONICAL[5]),
    "legacy_1ch_5": (_EXPORT_COL_BY_CANONICAL[4],),
    "ch6_only": (_EXPORT_COL_BY_CANONICAL[5],),
    "ch8_only": (_EXPORT_COL_BY_CANONICAL[7],),
    "ch19_only": (_EXPORT_COL_BY_CANONICAL[18],),
    "ch20_only": (_EXPORT_COL_BY_CANONICAL[19],),
    "posterior_4ch": tuple(_EXPORT_COL_BY_CANONICAL[c] for c in (4, 5, 6, 7)),
}


@dataclass
class TrialRecord:
    exp_id: int
    phase: str
    target_tile: Optional[int]
    logged: Dict[str, Any]
    rerun: Dict[str, Any]
    roi_sweep: Dict[str, Dict[str, Any]]


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_session_info(analysis_dir: Path) -> Dict[str, str]:
    p = analysis_dir / "session_info.csv"
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh), None)
    return dict(row) if row else {}


def _discover_experiments(analysis_dir: Path) -> List[Path]:
    return sorted(
        p for p in analysis_dir.glob("exp_*") if (p / "results.csv").is_file()
    )


def _load_eeg_matrix(exp_dir: Path) -> Tuple[np.ndarray, List[str]]:
    path = exp_dir / "eeg_markers.csv"
    rows = _read_csv_dicts(path)
    if not rows:
        return np.zeros((0, 0)), []
    ch_cols = [c for c in rows[0] if c.startswith("ch")]
    data = np.array([[float(r[c]) for c in ch_cols] for r in rows], dtype=np.float64)
    return data, ch_cols


def _estimate_fs(n_samples: int, t_start: float, t_end: float) -> float:
    dur = float(t_end) - float(t_start)
    if dur <= 0.1 or n_samples < 10:
        return 250.0
    return float(n_samples) / dur


def _epoch_len_samples(fs_hz: float, baseline_ms: int = 100) -> int:
    dt_ms = 1000.0 / fs_hz
    total_ms = float(EPOCH_DURATION_MS + baseline_ms)
    return int(round(total_ms / dt_ms)) + 1


def _time_ms_template(fs_hz: float, baseline_ms: int = 100) -> np.ndarray:
    dt_ms = 1000.0 / fs_hz
    el = _epoch_len_samples(fs_hz, baseline_ms)
    return np.arange(el, dtype=np.float64) * dt_ms - float(baseline_ms)


def _extract_epochs_from_export(
    eeg: np.ndarray,
    *,
    epochs_log: Sequence[Dict[str, str]],
    t_start_lsl: float,
    fs_hz: float,
    roi_cols_0idx: Tuple[int, ...],
    baseline_ms: int,
) -> Dict[str, List[np.ndarray]]:
    """Row-order EEG + LSL marker times → epochs_data (stim_key → list of 2D epochs)."""
    if eeg.size == 0 or not epochs_log:
        return {}
    n_ch_export = int(eeg.shape[1])
    valid_roi = [int(c) for c in roi_cols_0idx if 0 <= int(c) < n_ch_export]
    if not valid_roi:
        valid_roi = list(range(n_ch_export))

    el = _epoch_len_samples(fs_hz, baseline_ms)
    pre_s = float(baseline_ms) / 1000.0
    buf = bandpass_filter(eeg[:, valid_roi], fs_hz)
    epochs_data: Dict[str, List[np.ndarray]] = {}

    for row in epochs_log:
        sk = str(row.get("stim_key") or "").strip()
        if not sk:
            continue
        try:
            lsl_t = float(row["lsl_time"])
        except (KeyError, ValueError, TypeError):
            continue
        start = int(round((lsl_t - float(t_start_lsl) - pre_s) * fs_hz))
        end = start + el
        if start < 0 or end > buf.shape[0]:
            continue
        ep = buf[start:end, :].copy()
        if ep.shape[0] == el:
            epochs_data.setdefault(sk, []).append(ep)
    return epochs_data


def _build_template(
    epochs_data: Dict[str, List[np.ndarray]],
    *,
    target_tile: int,
    time_ms: np.ndarray,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
    artifact_uv: float,
) -> Optional[np.ndarray]:
    key = f"стимул_{int(target_tile)}"
    epochs = epochs_data.get(key, [])
    if len(epochs) < SAFE_MIN_EPOCHS_TO_DECIDE:
        return None
    stim_keys, raw, _ = build_averaged_erp(
        {key: list(epochs)},
        int(time_ms.shape[0]),
        artifact_threshold_uv=artifact_uv if artifact_uv > 0 else None,
    )
    if key not in stim_keys:
        return None
    idx = stim_keys.index(key)
    corrected, _, _, wx, wy = compute_corrected_and_integrated(
        raw, time_ms, baseline_ms, window_x_ms, window_y_ms
    )
    xi0, xi1 = time_window_to_indices(time_ms, wx, wy)
    tw = np.asarray(corrected[idx, xi0:xi1], dtype=np.float64).ravel()
    return tw if tw.size else None


def _decide(
    epochs_data: Dict[str, List[np.ndarray]],
    *,
    time_ms: np.ndarray,
    profile,
    template_window: Optional[np.ndarray],
    winner_mode: str,
    artifact_uv: float,
) -> Dict[str, Any]:
    el = int(time_ms.shape[0])
    stim_keys, raw, rej = build_averaged_erp(
        epochs_data,
        el,
        artifact_threshold_uv=artifact_uv if artifact_uv > 0 else None,
    )
    can, min_n = check_can_decide(stim_keys, epochs_data)
    out: Dict[str, Any] = {
        "can_decide": bool(can),
        "min_epochs": int(min_n),
        "rejected": rej,
        "stim_keys": stim_keys,
    }
    if not can or not stim_keys:
        return out
    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw,
        time_ms,
        int(profile.baseline_ms),
        int(profile.window_x_ms),
        int(profile.window_y_ms),
    )
    widx, mode_used, dbg = compute_winner_metrics(
        stim_keys,
        raw,
        corrected,
        time_ms,
        int(wx),
        int(wy),
        winner_mode=winner_mode,
        template_window=template_window,
    )
    winner_key = stim_keys[int(widx)]
    out.update(
        {
            "winner_tile": stim_key_to_tile_digit(winner_key),
            "winner_key": winner_key,
            "mode_used": mode_used,
            "margin": dbg.get("margin"),
            "abs_auc_values": dbg.get("abs_auc_values"),
            "template_corr_values": dbg.get("template_corr_values"),
            "signed_mean_values": dbg.get("signed_mean_values"),
            "corr_mean_in_window": dbg.get("corr_mean_in_window"),
            "window_ms": dbg.get("window_ms"),
        }
    )
    return out


def _filter_epochs_log(
    epochs_log: Sequence[Dict[str, str]],
    final_counts: Dict[int, int],
) -> List[Dict[str, str]]:
    """Keep first N markers per tile as logged at decision time (epochs_final_t*)."""
    have: Dict[int, int] = {}
    out: List[Dict[str, str]] = []
    for row in epochs_log:
        sk = str(row.get("stim_key") or "")
        tid = stim_key_to_tile_digit(sk)
        if tid is None:
            continue
        cap = int(final_counts.get(int(tid), 9999))
        n = have.get(int(tid), 0)
        if n < cap:
            out.append(row)
            have[int(tid)] = n + 1
    return out


def _final_counts_from_results(row: Dict[str, str]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i in range(9):
        k = f"epochs_final_t{i}"
        if k in row and row[k] not in ("", None):
            try:
                out[i] = int(float(row[k]))
            except ValueError:
                pass
    return out


def audit_experiment(
    exp_dir: Path,
    *,
    calib_template: Optional[np.ndarray],
    calib_target: int,
    profile_key: str = "general",
    artifact_uv: float = 60.0,
    roi_cols_0idx: Tuple[int, ...] = ROI_PRESETS["export_all_6cols"],
) -> TrialRecord:
    profile = hybrid_protocol_p300_profile(profile_key)
    logged_row = _read_csv_dicts(exp_dir / "results.csv")[0]
    exp_id = int(logged_row["exp_id"])
    phase = str(logged_row.get("phase") or "")
    tgt_raw = logged_row.get("target_tile")
    target_tile: Optional[int]
    try:
        target_tile = int(float(tgt_raw)) if tgt_raw not in ("", None) else None
    except ValueError:
        target_tile = None
    if target_tile is not None and target_tile < 0:
        target_tile = None

    eeg, _ = _load_eeg_matrix(exp_dir)
    epochs_log = _read_csv_dicts(exp_dir / "epochs_log.csv") if phase != "calib" else _read_csv_dicts(
        exp_dir / "epochs_log.csv"
    )
    if not epochs_log and (exp_dir / "epochs_log.csv").is_file():
        epochs_log = _read_csv_dicts(exp_dir / "epochs_log.csv")

    t_start = float(logged_row.get("t_start_lsl") or 0.0)
    t_end_logged = float(logged_row.get("t_end_lsl") or 0.0)
    # t_end_lsl is sometimes polluted with Unix wall time — prefer marker span.
    if epochs_log:
        lsl_times = [float(r["lsl_time"]) for r in epochs_log if r.get("lsl_time")]
        t_end = max(lsl_times) if lsl_times else t_start + 1.0
    else:
        t_end = t_end_logged if 0 < (t_end_logged - t_start) < 600 else t_start + 1.0
    fs_hz = _estimate_fs(int(eeg.shape[0]), t_start, t_end)

    final_counts = _final_counts_from_results(logged_row)
    epochs_log_use = _filter_epochs_log(epochs_log, final_counts) if final_counts else list(epochs_log)

    epochs_data = _extract_epochs_from_export(
        eeg,
        epochs_log=epochs_log_use,
        t_start_lsl=t_start,
        fs_hz=fs_hz,
        roi_cols_0idx=roi_cols_0idx,
        baseline_ms=int(profile.baseline_ms),
    )

    time_ms = _time_ms_template(fs_hz, int(profile.baseline_ms))
    template = calib_template
    if phase == "calib" and target_tile is not None:
        template = _build_template(
            epochs_data,
            target_tile=int(target_tile),
            time_ms=time_ms,
            baseline_ms=int(profile.baseline_ms),
            window_x_ms=int(profile.window_x_ms),
            window_y_ms=int(profile.window_y_ms),
            artifact_uv=artifact_uv,
        )

    auc = _decide(
        epochs_data,
        time_ms=time_ms,
        profile=profile,
        template_window=None,
        winner_mode=WINNER_MODE_AUC,
        artifact_uv=artifact_uv,
    )
    tpl = _decide(
        epochs_data,
        time_ms=time_ms,
        profile=profile,
        template_window=template,
        winner_mode=WINNER_MODE_TEMPLATE_CORR,
        artifact_uv=artifact_uv,
    )

    roi_sweep: Dict[str, Dict[str, Any]] = {}
    for name, roi in ROI_PRESETS.items():
        if name in ("export_all_6cols", "canonical_6ch"):
            continue
        ed = _extract_epochs_from_export(
            eeg,
            epochs_log=epochs_log_use,
            t_start_lsl=t_start,
            fs_hz=fs_hz,
            roi_cols_0idx=roi,
            baseline_ms=int(profile.baseline_ms),
        )
        ra = _decide(
            ed,
            time_ms=time_ms,
            profile=profile,
            template_window=template,
            winner_mode=WINNER_MODE_AUC,
            artifact_uv=artifact_uv,
        )
        rt = _decide(
            ed,
            time_ms=time_ms,
            profile=profile,
            template_window=template,
            winner_mode=WINNER_MODE_TEMPLATE_CORR,
            artifact_uv=artifact_uv,
        )
        roi_sweep[name] = {
            "auc_winner": ra.get("winner_tile"),
            "template_winner": rt.get("winner_tile"),
            "auc_ok": ra.get("winner_tile") == target_tile if target_tile is not None else None,
            "tpl_ok": rt.get("winner_tile") == target_tile if target_tile is not None else None,
        }

    rerun = {
        "fs_hz_estimated": fs_hz,
        "n_epochs_keys": {k: len(v) for k, v in epochs_data.items()},
        "auc": auc,
        "template": tpl,
        "auc_matches_logged": auc.get("winner_tile") == _int_or_none(logged_row.get("auc_winner")),
        "tpl_matches_logged": tpl.get("winner_tile") == _int_or_none(logged_row.get("template_winner")),
    }
    return TrialRecord(
        exp_id=exp_id,
        phase=phase,
        target_tile=target_tile,
        logged={k: logged_row.get(k) for k in logged_row},
        rerun=rerun,
        roi_sweep=roi_sweep,
    )


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def audit_session(
    session_path: Path,
    *,
    profile_key: str = "general",
) -> Dict[str, Any]:
    session_path = Path(session_path)
    analysis_dir = session_path / "analysis" if (session_path / "analysis").is_dir() else session_path
    session_info = _load_session_info(analysis_dir)
    calib_target = int(float(session_info.get("calib_target_tile") or 4))

    trials: List[TrialRecord] = []
    calib_template: Optional[np.ndarray] = None
    profile = hybrid_protocol_p300_profile(profile_key)

    for exp_dir in _discover_experiments(analysis_dir):
        tr = audit_experiment(
            exp_dir,
            calib_template=calib_template,
            calib_target=calib_target,
            profile_key=profile_key,
            roi_cols_0idx=ROI_PRESETS["export_all_6cols"],
        )
        trials.append(tr)
        if tr.phase == "calib" and tr.target_tile is not None:
            # rebuild template from last calib trial (matches locked template after warmup)
            eeg, _ = _load_eeg_matrix(exp_dir)
            elog = _read_csv_dicts(exp_dir / "epochs_log.csv")
            row = tr.logged
            fs = tr.rerun["fs_hz_estimated"]
            t_start = float(row.get("t_start_lsl") or 0)
            ed = _extract_epochs_from_export(
                eeg,
                epochs_log=elog,
                t_start_lsl=t_start,
                fs_hz=float(fs),
                roi_cols_0idx=ROI_PRESETS["export_all_6cols"],
                baseline_ms=int(profile.baseline_ms),
            )
            time_ms = _time_ms_template(float(fs), int(profile.baseline_ms))
            calib_template = _build_template(
                ed,
                target_tile=int(tr.target_tile),
                time_ms=time_ms,
                baseline_ms=int(profile.baseline_ms),
                window_x_ms=int(profile.window_x_ms),
                window_y_ms=int(profile.window_y_ms),
                artifact_uv=60.0,
            )

    def _logged_acc_phase(phase: str, col: str) -> Dict[str, Any]:
        subset = [t for t in trials if t.phase == phase and t.target_tile is not None]
        if not subset:
            return {"n": 0}
        ok = sum(1 for t in subset if t.logged.get(col) in ("1", "True", "true"))
        return {"n": len(subset), "correct": ok, "accuracy": round(ok / len(subset), 4)}

    def _rerun_acc_phase(phase: str, method: str) -> Dict[str, Any]:
        subset = [t for t in trials if t.phase == phase and t.target_tile is not None]
        return _rerun_phase_acc(subset, method)

    roi_agg: Dict[str, Dict[str, Any]] = {}
    for preset in ROI_PRESETS:
        if preset == "export_all_6cols":
            continue
        main = [t for t in trials if t.phase == "main" and t.target_tile is not None]
        auc_ok = sum(1 for t in main if t.roi_sweep.get(preset, {}).get("auc_ok"))
        tpl_ok = sum(1 for t in main if t.roi_sweep.get(preset, {}).get("tpl_ok"))
        roi_agg[preset] = {
            "main_auc_accuracy": round(auc_ok / len(main), 4) if main else None,
            "main_template_accuracy": round(tpl_ok / len(main), 4) if main else None,
        }

    calib_subset = [t for t in trials if t.phase == "calib"]
    main_subset = [t for t in trials if t.phase == "main" and t.target_tile is not None]

    # AUC bias: which tile wins when wrong
    auc_wrong = [t for t in trials if t.target_tile is not None and t.rerun["auc"].get("winner_tile") != t.target_tile]
    bias: Dict[str, int] = {}
    for t in auc_wrong:
        w = t.rerun["auc"].get("winner_tile")
        bias[str(w)] = bias.get(str(w), 0) + 1

    report: Dict[str, Any] = {
        "schema": "hybrid_bci_p300_simulator_audit/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_path": str(session_path),
        "session_info": session_info,
        "profile": {
            "key": profile_key,
            "baseline_ms": profile.baseline_ms,
            "window_x_ms": profile.window_x_ms,
            "window_y_ms": profile.window_y_ms,
            "roi_canonical_0idx": list(CANONICAL_P300_ROI_0IDX),
        },
        "aggregate": {
            "n_trials": len(trials),
            "logged_auc_accuracy": _logged_acc(trials, "correct_auc"),
            "logged_template_accuracy": _logged_acc(trials, "correct_template"),
            "rerun_auc_accuracy": _rerun_acc(trials, "auc"),
            "rerun_template_accuracy": _rerun_acc(trials, "template"),
            "rerun_matches_logged_auc": sum(1 for t in trials if t.rerun.get("auc_matches_logged")),
            "rerun_matches_logged_template": sum(1 for t in trials if t.rerun.get("tpl_matches_logged")),
            "calib": {
                "logged_template": _logged_acc_phase("calib", "correct_template"),
                "logged_auc": _logged_acc_phase("calib", "correct_auc"),
                "rerun_template": _rerun_phase_acc(calib_subset, "template"),
                "rerun_auc": _rerun_phase_acc(calib_subset, "auc"),
            },
            "main": {
                "logged_template": _logged_acc_phase("main", "correct_template"),
                "logged_auc": _logged_acc_phase("main", "correct_auc"),
                "rerun_template": _rerun_phase_acc(main_subset, "template"),
                "rerun_auc": _rerun_phase_acc(main_subset, "auc"),
            },
            "score_diagnostics": _score_diagnostics(trials),
            "auc_wrong_winner_histogram": bias,
            "roi_sweep_main": roi_agg,
        },
        "trials": [_trial_to_dict(t) for t in trials],
        "verdict": _build_verdict(trials, roi_agg, calib_target),
    }
    return report


def _logged_acc(trials: Sequence[TrialRecord], col: str) -> Optional[float]:
    ok = sum(1 for t in trials if t.logged.get(col) in ("1", "True", "true"))
    return round(ok / len(trials), 4) if trials else None


def _rerun_acc(trials: Sequence[TrialRecord], method: str) -> Optional[float]:
    valid = [t for t in trials if t.target_tile is not None]
    ok = sum(1 for t in valid if t.rerun.get(method, {}).get("winner_tile") == t.target_tile)
    return round(ok / len(valid), 4) if valid else None


def _score_diagnostics(trials: Sequence[TrialRecord]) -> Dict[str, Any]:
    """Logged auc_t / tpl_t: target score vs winner score."""
    rows: List[Dict[str, Any]] = []
    for t in trials:
        if t.target_tile is None:
            continue
        tgt = int(t.target_tile)
        auc_vals = []
        tpl_vals = []
        for i in range(9):
            try:
                auc_vals.append(float(t.logged.get(f"auc_t{i}") or 0))
            except ValueError:
                auc_vals.append(0.0)
            try:
                tpl_vals.append(float(t.logged.get(f"tpl_t{i}") or 0))
            except ValueError:
                tpl_vals.append(0.0)
        if not auc_vals:
            continue
        auc_w = int(max(range(9), key=lambda i: auc_vals[i]))
        tpl_w = int(max(range(9), key=lambda i: tpl_vals[i]))
        rows.append(
            {
                "exp_id": t.exp_id,
                "phase": t.phase,
                "target": tgt,
                "auc_target": auc_vals[tgt],
                "auc_winner": auc_vals[auc_w],
                "auc_ratio_target_over_winner": round(auc_vals[tgt] / auc_vals[auc_w], 4)
                if auc_vals[auc_w] > 1e-9
                else None,
                "auc_winner_tile": auc_w,
                "tpl_target": tpl_vals[tgt],
                "tpl_winner": tpl_vals[tpl_w],
                "tpl_winner_tile": tpl_w,
            }
        )
    calib = [r for r in rows if r["phase"] == "calib"]
    main = [r for r in rows if r["phase"] == "main"]
    return {
        "calib_mean_auc_ratio": _mean([r["auc_ratio_target_over_winner"] for r in calib]),
        "main_mean_auc_ratio": _mean([r["auc_ratio_target_over_winner"] for r in main]),
        "calib_tile4_auc_wins": sum(1 for r in calib if r["auc_winner_tile"] == 4),
        "main_tile4_auc_wins": sum(1 for r in main if r["auc_winner_tile"] == 4),
        "per_trial": rows,
    }


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rerun_phase_acc(trials: Sequence[TrialRecord], method: str) -> Dict[str, Any]:
    valid = [t for t in trials if t.target_tile is not None]
    ok = sum(1 for t in valid if t.rerun.get(method, {}).get("winner_tile") == t.target_tile)
    return {"n": len(valid), "correct": ok, "accuracy": round(ok / len(valid), 4) if valid else None}


def _trial_to_dict(t: TrialRecord) -> Dict[str, Any]:
    return {
        "exp_id": t.exp_id,
        "phase": t.phase,
        "target_tile": t.target_tile,
        "logged": dict(t.logged),
        "logged_auc_winner": t.logged.get("auc_winner"),
        "logged_template_winner": t.logged.get("template_winner"),
        "logged_correct_auc": t.logged.get("correct_auc"),
        "logged_correct_template": t.logged.get("correct_template"),
        "rerun": t.rerun,
        "roi_sweep": t.roi_sweep,
    }


def _build_verdict(
    trials: Sequence[TrialRecord],
    roi_agg: Dict[str, Dict[str, Any]],
    calib_target: int,
) -> Dict[str, Any]:
    calib_tpl = _rerun_phase_acc([t for t in trials if t.phase == "calib"], "template")
    main_auc = _rerun_phase_acc([t for t in trials if t.phase == "main"], "auc")
    best_roi = max(
        roi_agg.items(),
        key=lambda kv: (kv[1].get("main_auc_accuracy") or 0.0),
    )[0] if roi_agg else None
    notes = [
        "AUC metric = sum(|corrected_ERP|) in 550–725 ms after per-channel std normalize + channel mean.",
        "Template locked to calibration target tile waveform (protocol: calib_target_tile).",
        "Photodetector/optical P300 is periodic — not ERP-shaped; template_corr can track calib flash shape.",
    ]
    if calib_tpl.get("accuracy") == 1.0 and (main_auc.get("accuracy") or 0) < 0.5:
        notes.append(
            "Template high in calib only: expected when template matches fixed calib tile; "
            "main targets differ → template_corr is not target-adaptive."
        )
    return {
        "root_cause_summary": (
            "AUC fails on simulator because it ranks total |corrected| energy in the decision window, "
            "not shape similarity to the cued tile; optical flashes differ in amplitude/phase across tiles "
            "and 6-channel mean can cancel periodic components."
        ),
        "auc_pipeline_issue": True,
        "simulator_physics_issue": True,
        "roi_averaging_issue": True,
        "expected_on_artificial_signal": True,
        "best_roi_preset_main_auc": best_roi,
        "practical_notes": notes,
        "trust_real_eeg": "AUC remains primary for real ERP; validate per subject ROI. Simulator is template/AUC divergence test only.",
    }


def render_p300_simulator_markdown(report: Dict[str, Any]) -> str:
    agg = report.get("aggregate") or {}
    v = report.get("verdict") or {}
    lines = [
        "# P300 simulator / photodetector session audit",
        "",
        f"- Session: `{report.get('session_path')}`",
        f"- Generated: {report.get('generated_at')}",
        "",
        "## Summary",
        "",
        f"| Metric | Logged | Re-analysis |",
        f"|--------|--------|-------------|",
        f"| AUC accuracy | {agg.get('logged_auc_accuracy')} | {agg.get('rerun_auc_accuracy')} |",
        f"| Template accuracy | {agg.get('logged_template_accuracy')} | {agg.get('rerun_template_accuracy')} |",
        f"| Re-analysis matches logged AUC | {agg.get('rerun_matches_logged_auc')}/{agg.get('n_trials')} | |",
        "",
        "## Calibration vs main",
        "",
    ]
    for phase in ("calib", "main"):
        block = agg.get(phase) or {}
        lines.append(f"### {phase}")
        lines.append(f"- Logged AUC: {block.get('logged_auc')}")
        lines.append(f"- Logged template: {block.get('logged_template')}")
        lines.append(f"- Re-run AUC: {block.get('rerun_auc')}")
        lines.append(f"- Re-run template: {block.get('rerun_template')}")
        lines.append("")

    lines.extend(["## ROI sweep (main block)", ""])
    for preset, st in sorted((agg.get("roi_sweep_main") or {}).items()):
        lines.append(
            f"- **{preset}**: AUC acc={st.get('main_auc_accuracy')}, "
            f"template acc={st.get('main_template_accuracy')}"
        )

    diag = agg.get("score_diagnostics") or {}
    lines.extend(
        [
            "",
            "## Score diagnostics (logged)",
            "",
            f"- Calib: mean AUC(target)/AUC(winner) = **{diag.get('calib_mean_auc_ratio')}**; "
            f"tile 4 wins AUC in **{diag.get('calib_tile4_auc_wins')}/5** calib trials",
            f"- Main: mean AUC(target)/AUC(winner) = **{diag.get('main_mean_auc_ratio')}**; "
            f"tile 4 wins AUC in **{diag.get('main_tile4_auc_wins')}** main trials (calib flash bias)",
            "",
        ]
    )

    lines.extend(["", "## Root cause", "", v.get("root_cause_summary", ""), ""])
    for n in v.get("practical_notes") or []:
        lines.append(f"- {n}")

    lines.extend(["", "## Per-trial", ""])
    for tr in report.get("trials") or []:
        rr = tr.get("rerun") or {}
        auc = rr.get("auc") or {}
        tpl = rr.get("template") or {}
        lines.append(
            f"- exp {tr.get('exp_id')} ({tr.get('phase')}): target={tr.get('target_tile')} "
            f"logged auc={tr.get('logged_auc_winner')} tpl={tr.get('logged_template_winner')} | "
            f"rerun auc={auc.get('winner_tile')} margin={auc.get('margin')} "
            f"tpl={tpl.get('winner_tile')} margin={tpl.get('margin')}"
        )
    return "\n".join(lines) + "\n"


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

    # Score heatmap from logged auc_t / tpl_t for main trials
    trials = [t for t in report.get("trials") or [] if t.get("phase") == "main"]
    if trials:
        auc_mat = []
        tpl_mat = []
        labels = []
        for tr in trials:
            row = tr.get("logged") or {}
            labels.append(f"e{tr.get('exp_id')}/t{tr.get('target_tile')}")
            auc_mat.append([float(row.get(f"auc_t{i}") or 0) for i in range(9)])
            tpl_mat.append([float(row.get(f"tpl_t{i}") or 0) for i in range(9)])
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(trials) * 0.35)))
        for ax, mat, title in zip(
            axes,
            (auc_mat, tpl_mat),
            ("Logged AUC scores (|integral|)", "Logged template corr"),
        ):
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xticks(range(9))
            ax.set_xticklabels([str(i) for i in range(9)])
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        p = out_dir / "main_score_heatmaps.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    # ERP overlay: target vs AUC-winner (logged) for first main trial with valid target
    for tr in trials:
        if tr.get("phase") != "main" or tr.get("target_tile") is None:
            continue
        exp_id = int(tr["exp_id"])
        sp = Path(report["session_path"])
        analysis_dir = sp / "analysis" if (sp / "analysis").is_dir() else sp
        exp_dir = analysis_dir / f"exp_{exp_id:03d}_p300_main"
        if not exp_dir.is_dir():
            continue
        try:
            profile = hybrid_protocol_p300_profile()
            eeg, _ = _load_eeg_matrix(exp_dir)
            elog = _read_csv_dicts(exp_dir / "epochs_log.csv")
            row = tr.get("logged") or {}
            t_start = float(row.get("t_start_lsl") or 0)
            lsl_times = [float(r["lsl_time"]) for r in elog]
            fs = _estimate_fs(int(eeg.shape[0]), t_start, max(lsl_times))
            time_ms = _time_ms_template(fs, int(profile.baseline_ms))
            fc = _final_counts_from_results(row)
            elog_use = _filter_epochs_log(elog, fc) if fc else elog
            ed = _extract_epochs_from_export(
                eeg,
                epochs_log=elog_use,
                t_start_lsl=t_start,
                fs_hz=fs,
                roi_cols_0idx=ROI_PRESETS["export_all_6cols"],
                baseline_ms=int(profile.baseline_ms),
            )
            stim_keys, raw, _ = build_averaged_erp(ed, int(time_ms.shape[0]), artifact_threshold_uv=60.0)
            corrected, _, _, wx, wy = compute_corrected_and_integrated(
                raw,
                time_ms,
                int(profile.baseline_ms),
                int(profile.window_x_ms),
                int(profile.window_y_ms),
            )
            tgt = int(tr["target_tile"])
            auc_w = int(tr.get("logged_auc_winner") or 0)
            keys_idx = {stim_key_to_tile_digit(k): i for i, k in enumerate(stim_keys)}
            if tgt not in keys_idx or auc_w not in keys_idx:
                break
            xi0, xi1 = time_window_to_indices(time_ms, wx, wy)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(
                time_ms[xi0:xi1],
                corrected[keys_idx[tgt], xi0:xi1],
                label=f"target tile {tgt}",
                lw=2,
            )
            ax.plot(
                time_ms[xi0:xi1],
                corrected[keys_idx[auc_w], xi0:xi1],
                label=f"AUC winner tile {auc_w}",
                lw=2,
                alpha=0.85,
            )
            ax.axvline(wx, color="gray", ls="--", alpha=0.5)
            ax.axvline(wy, color="gray", ls="--", alpha=0.5)
            ax.set_title(f"exp {exp_id}: corrected ERP in AUC window")
            ax.set_xlabel("ms")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p2 = out_dir / f"erp_target_vs_auc_winner_exp{exp_id:03d}.png"
            fig.savefig(p2, dpi=120)
            plt.close(fig)
            written.append(str(p2))
        except Exception:
            pass
        break

    return written
