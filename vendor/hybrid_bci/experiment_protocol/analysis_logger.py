"""CSV-экспорт в analysis/ (v3): session_info, eeg_markers, results, epochs_log."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from p300_analysis.marker_parsing import stim_key_to_tile_digit

ANALYSIS_SCHEMA = "hybrid_protocol/v3"
N_TILES = 9
N_LAMPS = 6

SESSION_INFO_FIELDS = [
    "subject_last_name",
    "subject_first_name",
    "subject_id",
    "session_id",
    "session_start",
    "session_end",
    "stop_reason",
    "com_port",
    "eeg_stream_name",
    "p300_calib_trials",
    "p300_main_trials",
    "ssvep_blocks_per_mode",
    "pause_between_experiments_s",
    "shuffle_seed",
    "calib_target_tile",
    "p300_channels_1based",
    "ssvep_channels_1based",
    "ssvep_freq_L1_hz",
    "ssvep_freq_L2_hz",
    "ssvep_freq_L3_hz",
    "ssvep_freq_L4_hz",
    "ssvep_freq_L5_hz",
    "ssvep_freq_L6_hz",
    "msi_fs_hz",
    "msi_samples",
    "ssvep_block_sec",
]

RESULTS_COMMON = [
    "exp_id",
    "kind",
    "phase",
    "block_name",
    "queue_index",
    "t_start_lsl",
    "t_end_lsl",
    "notes",
]

RESULTS_P300 = [
    "target_tile",
    "planned_tile",
    "auc_winner",
    "template_winner",
    "auc_can_decide",
    "template_can_decide",
    "correct_auc",
    "correct_template",
    "methods_agree",
] + [f"auc_t{i}" for i in range(N_TILES)] + [f"tpl_t{i}" for i in range(N_TILES)] + [
    f"epochs_final_t{i}" for i in range(N_TILES)
]

RESULTS_SSVEP = [
    "ssvep_mode",
    "target_lamp",
    "target_freq_hz",
    "msi_winner",
    "msi_winner_last",
    "msi_winner_majority",
    "msi_correct",
    "msi_correct_last",
    "msi_classify_allowed",
    "msi_trace_compact",
    "msi_trace_n",
    "msi_trace_stability_pct",
    "msi_trace_switch_rate_per_sec",
    "msi_trace_avg_lock_sec",
    "msi_trace_false_transitions",
    "msi_trace_lock_on_sec",
    "msi_trace_histogram",
] + [f"msi_L{i}" for i in range(1, N_LAMPS + 1)]

RESULTS_FIELDS = RESULTS_COMMON + RESULTS_P300 + RESULTS_SSVEP

EPOCHS_LOG_FIELDS = (
    ["lsl_time", "marker", "tile_id", "stim_key", "n_extracted"]
    + [f"epochs_t{i}" for i in range(N_TILES)]
)


def sanitize_subject_slug(last_name: str, first_name: str) -> str:
    raw = f"{last_name.strip()}_{first_name.strip()}".strip("_")
    if not raw:
        return "unknown"
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE)
    return safe.strip("_") or "unknown"


def experiment_folder_name(*, exp_id: int, kind: str, phase: str, ssvep_mode: Optional[str] = None) -> str:
    k = str(kind).lower()
    if k == "ssvep":
        mode = str(ssvep_mode or "ssvep").lower().replace(" ", "_")
        return f"exp_{int(exp_id):03d}_ssvep_{mode}"
    ph = str(phase or "main").lower()
    return f"exp_{int(exp_id):03d}_p300_{ph}"


def _tile_cols(prefix: str, stim_keys: Sequence[str], values: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {f"{prefix}_t{i}": "" for i in range(N_TILES)}
    for k, v in zip(stim_keys, values):
        tid = stim_key_to_tile_digit(str(k))
        if tid is not None and 0 <= int(tid) < N_TILES:
            out[f"{prefix}_t{int(tid)}"] = float(v)
    return out


def _epochs_final_cols(epoch_counts_by_stim: Dict[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {f"epochs_final_t{i}": "" for i in range(N_TILES)}
    for k, n in (epoch_counts_by_stim or {}).items():
        tid = stim_key_to_tile_digit(str(k))
        if tid is not None and 0 <= int(tid) < N_TILES:
            out[f"epochs_final_t{int(tid)}"] = int(n)
    return out


def build_results_row(
    rec: Dict[str, Any],
    *,
    exp_id: int,
    t_start_lsl: Optional[float],
    t_end_lsl: Optional[float],
    block_name: str = "",
) -> Dict[str, Any]:
    """Плоская строка results.csv из записи append_experiment."""
    kind = str(rec.get("kind") or "")
    row: Dict[str, Any] = {k: "" for k in RESULTS_FIELDS}
    row.update(
        {
            "exp_id": int(exp_id),
            "kind": kind,
            "phase": str(rec.get("phase") or "main"),
            "block_name": str(block_name or rec.get("block_name") or ""),
            "queue_index": rec.get("queue_index", ""),
            "t_start_lsl": "" if t_start_lsl is None else float(t_start_lsl),
            "t_end_lsl": "" if t_end_lsl is None else float(t_end_lsl),
            "notes": str(rec.get("notes") or ""),
        }
    )
    if kind == "p300":
        cue = rec.get("cue_target_tile_id")
        planned = rec.get("planned_target_tile_id")
        auc = rec.get("auc") or {}
        tpl = rec.get("template_corr") or {}
        auc_dbg = auc.get("debug") or {}
        tpl_dbg = tpl.get("debug") or {}
        row["target_tile"] = cue if cue is None else int(cue)
        row["planned_tile"] = planned if planned is None else int(planned)
        row["auc_winner"] = auc.get("winner_tile_id", "")
        row["template_winner"] = tpl.get("winner_tile_id", "")
        row["auc_can_decide"] = 1 if auc.get("can_decide") else 0
        row["template_can_decide"] = 1 if tpl.get("can_decide") else 0
        row["correct_auc"] = "" if rec.get("correct_auc") is None else int(bool(rec["correct_auc"]))
        row["correct_template"] = (
            "" if rec.get("correct_template") is None else int(bool(rec["correct_template"]))
        )
        row["methods_agree"] = "" if rec.get("methods_agree") is None else int(bool(rec["methods_agree"]))
        if auc_dbg.get("stim_keys") and auc_dbg.get("final_metric_values"):
            row.update(_tile_cols("auc", auc_dbg["stim_keys"], auc_dbg["final_metric_values"]))
        if tpl_dbg.get("stim_keys") and tpl_dbg.get("final_metric_values"):
            row.update(_tile_cols("tpl", tpl_dbg["stim_keys"], tpl_dbg["final_metric_values"]))
        row.update(_epochs_final_cols(rec.get("epoch_counts_by_stim") or {}))
    elif kind == "ssvep":
        lamp = rec.get("cue_target_lamp_1based")
        row["ssvep_mode"] = str(rec.get("ssvep_mode") or "")
        row["target_lamp"] = "" if lamp is None else int(lamp)
        freqs = rec.get("ssvep_freqs_hz") or []
        if lamp is not None and freqs:
            idx = int(lamp) - 1
            if 0 <= idx < len(freqs):
                row["target_freq_hz"] = float(freqs[idx])
        row["msi_winner"] = rec.get("winner_1based", "")
        row["msi_winner_last"] = rec.get("winner_last_1based", "")
        row["msi_winner_majority"] = rec.get("winner_majority_1based", "")
        row["msi_correct"] = "" if rec.get("correct") is None else int(bool(rec["correct"]))
        row["msi_correct_last"] = (
            "" if rec.get("correct_last_decision") is None else int(bool(rec["correct_last_decision"]))
        )
        row["msi_classify_allowed"] = (
            "" if rec.get("classify_allowed") is None else int(bool(rec["classify_allowed"]))
        )
        coef = rec.get("coef") or []
        for i in range(N_LAMPS):
            if i < len(coef) and coef[i] is not None:
                row[f"msi_L{i + 1}"] = float(coef[i])
            else:
                row[f"msi_L{i + 1}"] = ""
        trace = rec.get("msi_decision_trace") or {}
        summ = trace.get("summary") if isinstance(trace, dict) else {}
        if isinstance(summ, dict):
            row["msi_trace_compact"] = summ.get("compact_stream", trace.get("compact_stream", ""))
            row["msi_trace_n"] = summ.get("n_decisions", "")
            row["msi_trace_stability_pct"] = summ.get("stability_pct", "")
            row["msi_trace_switch_rate_per_sec"] = summ.get("switch_rate_per_sec", "")
            row["msi_trace_avg_lock_sec"] = summ.get("avg_lock_duration_sec", "")
            row["msi_trace_false_transitions"] = summ.get("false_transition_count", "")
            row["msi_trace_lock_on_sec"] = summ.get("lock_on_sec", "")
            row["msi_trace_histogram"] = summ.get("decision_histogram", "")
    return row


@dataclass
class _ActiveExperiment:
    exp_id: int
    kind: str
    phase: str
    ssvep_mode: Optional[str]
    folder: Path
    roi_0idx: Tuple[int, ...]
    ch_headers: List[str]
    t_start_lsl: Optional[float] = None
    t_end_lsl: Optional[float] = None
    eeg_fh: Optional[TextIO] = None
    eeg_writer: Any = None
    epochs_fh: Optional[TextIO] = None
    epochs_writer: Any = None
    marker_queue: List[Tuple[float, str, str]] = field(default_factory=list)


class AnalysisSessionWriter:
    """Пишет analysis/ внутри папки сессии."""

    def __init__(self, session_dir: Path, *, session_info: Dict[str, Any]) -> None:
        self._session_dir = Path(session_dir)
        self._analysis_dir = self._session_dir / "analysis"
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        self._session_info = dict(session_info)
        self._session_info.setdefault("session_start", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        self._p300_roi_0idx: Tuple[int, ...] = tuple(session_info.get("p300_roi_0idx") or ())
        self._ssvep_roi_0idx: Tuple[int, ...] = tuple(session_info.get("ssvep_roi_0idx") or ())
        self._active: Optional[_ActiveExperiment] = None
        self._n_experiments = 0
        self._session_end: str = ""
        self._stop_reason: str = ""

    @property
    def analysis_dir(self) -> Path:
        return self._analysis_dir

    def begin_experiment(
        self,
        *,
        kind: str,
        phase: str = "main",
        ssvep_mode: Optional[str] = None,
        t_start_lsl: Optional[float] = None,
    ) -> int:
        self._n_experiments += 1
        exp_id = int(self._n_experiments)
        folder = self._analysis_dir / experiment_folder_name(
            exp_id=exp_id, kind=kind, phase=phase, ssvep_mode=ssvep_mode
        )
        folder.mkdir(parents=True, exist_ok=True)
        roi = self._p300_roi_0idx if str(kind) == "p300" else self._ssvep_roi_0idx
        ch_headers = [f"ch{i + 1}" for i in roi]
        act = _ActiveExperiment(
            exp_id=exp_id,
            kind=str(kind),
            phase=str(phase),
            ssvep_mode=ssvep_mode,
            folder=folder,
            roi_0idx=roi,
            ch_headers=ch_headers,
            t_start_lsl=t_start_lsl,
        )
        eeg_path = folder / "eeg_markers.csv"
        act.eeg_fh = open(eeg_path, "w", encoding="utf-8", newline="")
        act.eeg_writer = csv.writer(act.eeg_fh)
        act.eeg_writer.writerow(["time", *ch_headers, "event", "tile_id"])
        if str(kind) == "p300":
            ep_path = folder / "epochs_log.csv"
            act.epochs_fh = open(ep_path, "w", encoding="utf-8", newline="")
            act.epochs_writer = csv.DictWriter(act.epochs_fh, fieldnames=EPOCHS_LOG_FIELDS, extrasaction="ignore")
            act.epochs_writer.writeheader()
        self._active = act
        return exp_id

    def _roi_indices(self, n_channels: int) -> List[int]:
        act = self._active
        if act is None:
            return []
        return [i for i in act.roi_0idx if 0 <= int(i) < int(n_channels)]

    def append_eeg_chunk(self, times: Sequence[float], samples: np.ndarray) -> None:
        act = self._active
        if act is None or act.eeg_writer is None:
            return
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ts = np.asarray(times, dtype=np.float64)
        if ts.size != arr.shape[0]:
            if ts.size > 1 and arr.shape[0] > 1:
                ts = np.linspace(float(ts[0]), float(ts[-1]), arr.shape[0])
            else:
                return
        roi_idx = self._roi_indices(int(arr.shape[1]))
        if act.t_start_lsl is None and ts.size:
            act.t_start_lsl = float(ts[0])
        act.t_end_lsl = float(ts[-1])
        for i in range(int(ts.size)):
            t = float(ts[i])
            event = ""
            tile_id = ""
            while act.marker_queue and float(act.marker_queue[0][0]) <= t + 0.002:
                _, event, tile_id = act.marker_queue.pop(0)
            vals = [float(arr[i, j]) for j in roi_idx]
            act.eeg_writer.writerow([t, *vals, event, tile_id])

    def record_marker(self, *, lsl_time: float, value: str, tile_id: str = "") -> None:
        act = self._active
        if act is None:
            return
        act.marker_queue.append((float(lsl_time), str(value), str(tile_id)))
        if act.t_start_lsl is None:
            act.t_start_lsl = float(lsl_time)
        act.t_end_lsl = float(lsl_time)

    def _flush_marker_only_rows(self) -> None:
        act = self._active
        if act is None or act.eeg_writer is None:
            return
        while act.marker_queue:
            t, event, tile_id = act.marker_queue.pop(0)
            act.eeg_writer.writerow([float(t), *[""] * len(act.ch_headers), event, tile_id])

    def append_epochs_log(
        self,
        *,
        lsl_time: float,
        marker: str,
        stim_key: str,
        n_extracted: int,
        epochs_counts: Dict[str, int],
    ) -> None:
        act = self._active
        if act is None or act.epochs_writer is None:
            return
        tid = stim_key_to_tile_digit(str(stim_key))
        row: Dict[str, Any] = {
            "lsl_time": float(lsl_time),
            "marker": str(marker),
            "tile_id": "" if tid is None else int(tid),
            "stim_key": str(stim_key),
            "n_extracted": int(n_extracted),
        }
        for i in range(N_TILES):
            row[f"epochs_t{i}"] = ""
        for k, n in epochs_counts.items():
            t = stim_key_to_tile_digit(str(k))
            if t is not None and 0 <= int(t) < N_TILES:
                row[f"epochs_t{int(t)}"] = int(n)
        act.epochs_writer.writerow(row)

    def finish_experiment(self, results_row: Dict[str, Any]) -> None:
        act = self._active
        if act is None:
            return
        self._flush_marker_only_rows()
        if act.t_start_lsl is not None:
            results_row["t_start_lsl"] = act.t_start_lsl
        if act.t_end_lsl is not None:
            results_row["t_end_lsl"] = act.t_end_lsl
        try:
            if act.eeg_fh is not None:
                act.eeg_fh.flush()
                act.eeg_fh.close()
        except Exception:
            pass
        try:
            if act.epochs_fh is not None:
                act.epochs_fh.flush()
                act.epochs_fh.close()
        except Exception:
            pass
        res_path = act.folder / "results.csv"
        with res_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: results_row.get(k, "") for k in RESULTS_FIELDS})
        self._active = None

    def finalize(self, *, stop_reason: str) -> None:
        if self._active is not None:
            self.finish_experiment({})
        self._session_end = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self._stop_reason = str(stop_reason)
        info = dict(self._session_info)
        info["session_end"] = self._session_end
        info["stop_reason"] = self._stop_reason
        path = self._analysis_dir / "session_info.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SESSION_INFO_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: info.get(k, "") for k in SESSION_INFO_FIELDS})
