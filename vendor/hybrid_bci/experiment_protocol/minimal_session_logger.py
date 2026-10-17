"""Minimal session artifact layout (CSV + per-experiment JSON only)."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from p300_analysis.marker_parsing import stim_key_to_tile_digit

LOG_SCHEMA = "hybrid_protocol/minimal_v1"
EEG_CSV_DECIMALS = 3

ALL_EXPERIMENTS_FIELDS = [
    "experiment_id",
    "experiment_type",
    "phase",
    "target_tile",
    "target_lamp",
    "ssvep_mode",
    "auc_winner",
    "template_winner",
    "auc_correct",
    "template_correct",
    "any_correct",
    "methods_agree",
    "majority_winner",
    "msi_trace",
    "msi_correct",
]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return repr(obj)


def _normalize_marker_value(sample: Any) -> str:
    if isinstance(sample, (list, tuple)) and sample:
        sample = sample[0]
    if isinstance(sample, (bytes, bytearray)):
        return sample.decode("utf-8", errors="replace")
    return str(sample)


@dataclass
class SessionPaths:
    session_dir: Path
    eeg_raw_path: Path
    eeg_with_markers_path: Path
    all_experiments_path: Path
    experiments_root: Path


@dataclass
class _MarkerEvent:
    lsl_time: float
    event_type: str
    event_value: str
    experiment_id: int
    experiment_type: str
    target: str


class MinimalSessionLogger:
    """Session logger: only eeg_raw, eeg_with_markers, all_experiments, experiments/."""

    session_id: str
    session_dir: Path

    def __init__(
        self,
        session_dir: Path,
        *,
        session_id: str,
        subject_id: str,
    ) -> None:
        self._dir = Path(session_dir)
        self._paths = SessionPaths(
            session_dir=self._dir,
            eeg_raw_path=self._dir / "eeg_raw.csv",
            eeg_with_markers_path=self._dir / "eeg_with_markers.csv",
            all_experiments_path=self._dir / "all_experiments.csv",
            experiments_root=self._dir / "experiments",
        )
        self.session_id = str(session_id)
        self.session_dir = self._dir
        self._subject_id = str(subject_id)
        self._closed = False

        self._eeg_times: List[float] = []
        self._eeg_data: List[np.ndarray] = []
        self._labels: List[str] = []
        self._marker_events: List[_MarkerEvent] = []
        self._experiment_rows: List[Dict[str, Any]] = []
        self._n_experiments = 0

        self._active_exp_id = 0
        self._active_exp_type = ""
        self._active_target = ""
        self._pending_msi_trace: List[int] = []

        self._paths.experiments_root.mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> SessionPaths:
        return self._paths

    @staticmethod
    def allocate_session_dir(*, output_root: Path, subject_id: str) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        subj = str(subject_id).strip("_") or "unknown"
        session_dir = output_root / f"session_{stamp}_{subj}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @classmethod
    def open_new(
        cls,
        *,
        output_root: Path,
        subject_id: str,
        protocol_plan: Dict[str, Any],
        start_payload: Dict[str, Any],
        session_dir: Optional[Path] = None,
    ) -> "MinimalSessionLogger":
        del protocol_plan, start_payload  # not persisted (minimal layout)
        if session_dir is None:
            session_dir = cls.allocate_session_dir(output_root=output_root, subject_id=subject_id)
        else:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)

        subj = str(subject_id).strip("_")
        suffix = f"_{subj}" if subj else ""
        if suffix and session_dir.name.endswith(suffix):
            session_id = session_dir.name[: -len(suffix)]
        else:
            session_id = session_dir.name

        return cls(session_dir, session_id=session_id, subject_id=subject_id)

    def attach_analysis(self, writer: Any) -> None:
        del writer  # legacy analysis bundle disabled

    def set_block_label(self, label: str) -> None:
        del label

    def set_eeg_channel_labels(self, labels: Sequence[str]) -> None:
        self._labels = [str(x) for x in labels]

    def write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        _ingest_protocol_event_from_write(self, str(event), data or {})

    def write_tech(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        del event, data  # no tech trace in minimal layout

    def append_eeg_chunk(
        self,
        times: Sequence[float],
        samples: np.ndarray,
        *,
        stream: str,
        lsl_clock_now: Optional[float] = None,
    ) -> None:
        del stream, lsl_clock_now
        if self._closed or samples is None:
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
        self._eeg_times.extend(ts.tolist())
        self._eeg_data.append(arr)

    def record_marker(self, *, lsl_time: float, sample: Any, tile_id: Any = None) -> None:
        if self._closed:
            return
        value = _normalize_marker_value(sample)
        ev_type, ev_val = _marker_to_event_type(value)
        tgt = str(tile_id) if tile_id is not None else ""
        self._marker_events.append(
            _MarkerEvent(
                lsl_time=float(lsl_time),
                event_type=ev_type,
                event_value=ev_val,
                experiment_id=int(self._active_exp_id),
                experiment_type=str(self._active_exp_type),
                target=tgt or self._active_target,
            )
        )

    def append_epochs_log(self, **kwargs: Any) -> None:
        del kwargs

    def mark_experiment_start(self, **meta: Any) -> None:
        kind = str(meta.get("kind") or "p300")
        self._active_exp_id = int(self._n_experiments) + 1
        self._active_exp_type = kind
        tgt = meta.get("cue_target_tile_id")
        if tgt is not None:
            self._active_target = str(int(tgt))
        elif meta.get("target_lamp_0idx") is not None:
            self._active_target = str(int(meta["target_lamp_0idx"]))
        else:
            self._active_target = ""
        self._pending_msi_trace = []

    def append_msi_decision(self, point: Dict[str, Any], *, experiment_number: Optional[int] = None) -> None:
        del experiment_number
        w = point.get("winner_1based")
        if w is not None:
            self._pending_msi_trace.append(int(w))

    def append_p300_trial(self, rec: Dict[str, Any]) -> None:
        del rec

    def append_ssvep_block(self, rec: Dict[str, Any]) -> None:
        del rec

    def append_experiment(self, rec: Dict[str, Any]) -> int:
        if self._closed:
            return self._n_experiments
        self._n_experiments += 1
        exp_id = int(self._n_experiments)
        row = dict(rec)
        kind = str(row.get("kind") or "p300")
        folder = self._experiment_folder(exp_id, kind, row)
        folder.mkdir(parents=True, exist_ok=True)

        if kind == "p300":
            self._write_p300_experiment(folder, exp_id, row)
        elif kind == "ssvep":
            self._write_ssvep_experiment(folder, exp_id, row)

        flat = _flat_experiment_row(exp_id, row, pending_msi_trace=self._pending_msi_trace)
        self._experiment_rows.append(flat)
        self._rewrite_all_experiments_csv()

        self._active_exp_id = 0
        self._active_exp_type = ""
        self._pending_msi_trace = []
        return exp_id

    def finalize(self, *, stop_payload: Dict[str, Any]) -> Path:
        if self._closed:
            return self._dir
        self._write_eeg_csv_files()
        self._rewrite_all_experiments_csv()
        self._closed = True
        del stop_payload
        return self._dir

    def close(self) -> None:
        self._closed = True

    def _experiment_folder(self, exp_id: int, kind: str, row: Dict[str, Any]) -> Path:
        del row
        return self._paths.experiments_root / f"exp_{exp_id:03d}_{kind}"

    def _write_p300_experiment(self, folder: Path, exp_id: int, row: Dict[str, Any]) -> None:
        cue = _resolve_p300_target_tile(row)
        auc = row.get("auc") or {}
        tpl = row.get("template_corr") or {}
        auc_w = auc.get("winner_tile_id")
        tpl_w = tpl.get("winner_tile_id")
        auc_ok = bool(auc.get("can_decide")) and auc_w == cue if cue is not None else False
        tpl_ok = bool(tpl.get("can_decide")) and tpl_w == cue if cue is not None else False
        summary = {
            "experiment_id": exp_id,
            "experiment_type": "p300",
            "phase": str(row.get("phase") or "main"),
            "target_tile": int(cue) if cue is not None else None,
            "auc_winner": int(auc_w) if auc_w is not None else None,
            "template_winner": int(tpl_w) if tpl_w is not None else None,
            "auc_correct": bool(auc_ok) if cue is not None else None,
            "template_correct": bool(tpl_ok) if cue is not None else None,
            "any_correct": bool(auc_ok or tpl_ok) if cue is not None else None,
            "methods_agree": bool(row.get("methods_agree")),
        }
        (folder / "experiment_summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        export = row.pop("_p300_averages_export", None)
        if isinstance(export, dict):
            _write_p300_averages_csv(folder / "p300_averages.csv", export)

    def _write_ssvep_experiment(self, folder: Path, exp_id: int, row: Dict[str, Any]) -> None:
        cue = row.get("cue_target_lamp_0idx")
        if cue is None:
            cue = row.get("target_lamp")
        trace = _ssvep_msi_trace_from_row(row, pending_msi_trace=self._pending_msi_trace)
        maj = row.get("winner_majority_1based")
        if maj is None:
            maj = row.get("winner_1based")
        msi_ok = (
            int(row.get("winner_majority_0idx", -99)) == int(cue)
            if cue is not None and row.get("winner_majority_0idx") is not None
            else row.get("correct")
        )
        msi_tr = row.get("msi_decision_trace") or {}
        summ = msi_tr.get("summary") if isinstance(msi_tr, dict) else {}
        if not isinstance(summ, dict):
            summ = {}
        subblock_sec = row.get("subblock_duration_sec")
        if subblock_sec is None and isinstance(msi_tr, dict):
            subblock_sec = msi_tr.get("subblock_duration_sec")
        group_size = row.get("group_size")
        if group_size is None and isinstance(msi_tr, dict):
            group_size = msi_tr.get("group_size")
        experiment_sec = row.get("experiment_duration_sec")
        if experiment_sec is None and isinstance(msi_tr, dict):
            experiment_sec = msi_tr.get("experiment_duration_sec")
        n_dec = summ.get("n_decisions")
        summary = {
            "experiment_id": exp_id,
            "experiment_type": "ssvep",
            "target_lamp": int(cue) + 1 if cue is not None else None,
            "ssvep_mode": str(row.get("ssvep_mode") or row.get("mode") or ""),
            "msi_trace": trace,
            "msi_n_decisions": int(n_dec) if n_dec is not None else None,
            "subblock_duration_sec": float(subblock_sec) if subblock_sec is not None else None,
            "group_size": int(group_size) if group_size is not None else None,
            "experiment_duration_sec": float(experiment_sec) if experiment_sec is not None else None,
            "majority_winner": int(maj) if maj is not None else None,
            "msi_correct": bool(msi_ok) if msi_ok is not None else None,
            "any_correct": bool(msi_ok) if msi_ok is not None else None,
        }
        (folder / "experiment_summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        export = row.pop("_ssvep_averages_export", None)
        if isinstance(export, dict):
            _write_ssvep_averages_csv(folder / "ssvep_averages.csv", export)

    def _rewrite_all_experiments_csv(self) -> None:
        path = self._paths.all_experiments_path
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ALL_EXPERIMENTS_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in self._experiment_rows:
                w.writerow({k: row.get(k, "") for k in ALL_EXPERIMENTS_FIELDS})

    def _eeg_csv_base_row(self, time_val: float, sample_idx: int, eeg_row: np.ndarray, n_ch: int) -> List[Any]:
        """timestamp_lsl + channels with fixed decimal places in CSV."""
        return [
            _fmt_eeg_csv_number(time_val),
            int(sample_idx),
            *[_fmt_eeg_csv_number(eeg_row[c]) for c in range(n_ch)],
        ]

    def _write_eeg_csv_files(self) -> None:
        if not self._eeg_times or not self._eeg_data:
            return
        eeg = np.vstack(self._eeg_data)
        times = np.asarray(self._eeg_times, dtype=np.float64)
        n = min(int(times.size), int(eeg.shape[0]))
        times = times[:n]
        eeg = eeg[:n]
        n_ch = int(eeg.shape[1])
        labels = list(self._labels)
        if len(labels) < n_ch:
            labels.extend(f"ch{i + 1}" for i in range(len(labels), n_ch))

        ch_headers = [f"ch{i + 1}" for i in range(n_ch)]
        raw_header = ["timestamp_lsl", "sample_index", *ch_headers]
        with self._paths.eeg_raw_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(raw_header)
            for i in range(n):
                w.writerow(self._eeg_csv_base_row(float(times[i]), i, eeg[i], n_ch))

        mk_header = [
            "timestamp_lsl",
            "sample_index",
            *ch_headers,
            "event_type",
            "event_value",
            "experiment_id",
            "experiment_type",
            "target",
        ]
        markers_by_idx = _markers_to_sample_indices(times, self._marker_events)
        with self._paths.eeg_with_markers_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(mk_header)
            for i in range(n):
                mk = markers_by_idx.get(i)
                row = self._eeg_csv_base_row(float(times[i]), i, eeg[i], n_ch)
                if mk is not None:
                    row.extend(
                        [
                            mk.event_type,
                            mk.event_value,
                            int(mk.experiment_id),
                            mk.experiment_type,
                            mk.target,
                        ]
                    )
                else:
                    row.extend(["", "", "", "", ""])
                w.writerow(row)


def _fmt_eeg_csv_number(v: float) -> str:
    return f"{float(v):.{EEG_CSV_DECIMALS}f}"


def _marker_to_event_type(value: str) -> Tuple[str, str]:
    s = str(value).strip()
    low = s.lower()
    if low.startswith("-1|trial_start"):
        return "trial_start", s
    if low.startswith("-1|trial_end"):
        return "trial_end", s
    if "|" in s and s.split("|", 1)[0].strip().lstrip("-").isdigit():
        parts = s.split("|", 1)
        return f"stimulus_{parts[0].strip()}", parts[1].strip() if len(parts) > 1 else ""
    if low.startswith("led"):
        return "migalka_led", s
    return "marker", s


def _ingest_protocol_event_from_write(logger: MinimalSessionLogger, event: str, data: Dict[str, Any]) -> None:
    """Map high-level protocol events into marker stream."""
    lsl_t = float(data.get("lsl_time") or data.get("t_lsl") or 0.0)
    if event == "protocol_trial_start_arm":
        tid = data.get("cue_target_tile_id")
        logger._marker_events.append(
            _MarkerEvent(
                lsl_time=lsl_t or _guess_time(logger),
                event_type="trial_start",
                event_value=str(tid) if tid is not None else "",
                experiment_id=int(logger._active_exp_id),
                experiment_type="p300",
                target=str(tid) if tid is not None else "",
            )
        )
    elif event == "trial_decision_dual":
        logger._marker_events.append(
            _MarkerEvent(
                lsl_time=_guess_time(logger),
                event_type="trial_end",
                event_value="dual_decision",
                experiment_id=int(logger._active_exp_id),
                experiment_type="p300",
                target=str(data.get("cue_target_tile_id") or ""),
            )
        )
    elif event == "ssvep_block_start":
        lamp = data.get("target_lamp")
        logger._marker_events.append(
            _MarkerEvent(
                lsl_time=_guess_time(logger),
                event_type="ssvep_start",
                event_value=str(data.get("mode") or ""),
                experiment_id=int(logger._active_exp_id),
                experiment_type="ssvep",
                target=str(lamp) if lamp is not None else "",
            )
        )
    elif event == "ssvep_block_end":
        logger._marker_events.append(
            _MarkerEvent(
                lsl_time=_guess_time(logger),
                event_type="ssvep_end",
                event_value=str(data.get("mode") or ""),
                experiment_id=int(logger._active_exp_id),
                experiment_type="ssvep",
                target=str(data.get("target_lamp") or ""),
            )
        )
    elif event == "migalka_led":
        lamp = data.get("lamp")
        on = data.get("on")
        logger._marker_events.append(
            _MarkerEvent(
                lsl_time=_guess_time(logger),
                event_type="stimulus_on" if on else "stimulus_off",
                event_value=f"L{int(lamp) + 1}" if lamp is not None else str(data.get("raw") or ""),
                experiment_id=int(logger._active_exp_id),
                experiment_type=str(logger._active_exp_type or "ssvep"),
                target=str(lamp) if lamp is not None else "",
            )
        )


def _guess_time(logger: MinimalSessionLogger) -> float:
    if logger._eeg_times:
        return float(logger._eeg_times[-1])
    return float(time.time())


# patch write to use mapper
def _markers_to_sample_indices(
    times: np.ndarray,
    markers: Sequence[_MarkerEvent],
    *,
    max_delta_s: float = 0.05,
) -> Dict[int, _MarkerEvent]:
    if times.size == 0 or not markers:
        return {}
    out: Dict[int, _MarkerEvent] = {}
    for mk in markers:
        idx = int(np.argmin(np.abs(times - float(mk.lsl_time))))
        if abs(float(times[idx]) - float(mk.lsl_time)) <= float(max_delta_s):
            out[idx] = mk
    return out


def _resolve_p300_target_tile(row: Dict[str, Any]) -> Optional[int]:
    """0-based tile id for logs; fallback to planned queue target if LSL cue missing."""
    cue = row.get("cue_target_tile_id")
    if cue is not None:
        try:
            c = int(cue)
            if c >= 0:
                return c
        except (TypeError, ValueError):
            pass
    planned = row.get("planned_target_tile_id")
    if planned is not None:
        try:
            p = int(planned)
            if p >= 0:
                return p
        except (TypeError, ValueError):
            pass
    return None


def _ssvep_msi_trace_from_row(
    row: Dict[str, Any],
    *,
    pending_msi_trace: Sequence[int] = (),
) -> str:
    msi_tr = row.get("msi_decision_trace") or {}
    trace = ""
    if isinstance(msi_tr, dict):
        trace = str(msi_tr.get("compact_stream") or "").strip()
        if not trace:
            pts = msi_tr.get("points")
            if isinstance(pts, list):
                trace = " ".join(
                    str(p.get("winner_1based"))
                    for p in pts
                    if p.get("winner_1based") is not None
                )
    if not trace and pending_msi_trace:
        trace = " ".join(str(x) for x in pending_msi_trace)
    return trace


def _flat_experiment_row(
    exp_id: int,
    row: Dict[str, Any],
    *,
    pending_msi_trace: Sequence[int] = (),
) -> Dict[str, Any]:
    kind = str(row.get("kind") or "")
    cue_tile = _resolve_p300_target_tile(row) if kind == "p300" else row.get("cue_target_tile_id")
    cue_lamp = row.get("cue_target_lamp_0idx")
    if cue_lamp is None:
        cue_lamp = row.get("target_lamp")
    auc = row.get("auc") or {}
    tpl = row.get("template_corr") or {}
    auc_w = auc.get("winner_tile_id")
    tpl_w = tpl.get("winner_tile_id")
    trace = _ssvep_msi_trace_from_row(row, pending_msi_trace=pending_msi_trace)
    maj = row.get("winner_majority_1based")
    if maj is None:
        maj = row.get("winner_1based")
    flat: Dict[str, Any] = {
        "experiment_id": exp_id,
        "experiment_type": kind,
        "phase": row.get("phase", ""),
        "target_tile": cue_tile if cue_tile is not None else "",
        "target_lamp": int(cue_lamp) + 1 if cue_lamp is not None else "",
        "ssvep_mode": row.get("ssvep_mode") or row.get("mode") or "",
        "auc_winner": auc_w if auc_w is not None else "",
        "template_winner": tpl_w if tpl_w is not None else "",
        "auc_correct": row.get("correct_auc", ""),
        "template_correct": row.get("correct_template", ""),
        "methods_agree": row.get("methods_agree", ""),
        "majority_winner": maj if maj is not None else "",
        "msi_trace": trace,
        "msi_correct": row.get("correct", ""),
    }
    auc_ok = row.get("correct_auc")
    tpl_ok = row.get("correct_template")
    flat["any_correct"] = bool(auc_ok or tpl_ok) if kind == "p300" else row.get("correct", "")
    return flat


def _write_p300_averages_csv(path: Path, export: Dict[str, Any]) -> None:
    time_ms = export.get("time_ms") or []
    tiles = export.get("tiles") or {}
    if not time_ms:
        return
    tile_ids = sorted(int(k) for k in tiles.keys())
    header = ["time_ms"] + [f"tile_{tid}" for tid in tile_ids]
    n = len(time_ms)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            row = [float(time_ms[i])]
            for tid in tile_ids:
                col = tiles.get(tid) or tiles.get(str(tid))
                if col is None:
                    row.append("")
                elif i < len(col):
                    row.append(float(col[i]))
                else:
                    row.append("")
            w.writerow(row)


def _write_ssvep_averages_csv(path: Path, export: Dict[str, Any]) -> None:
    time_ms = export.get("time_ms") or []
    lamps = export.get("lamps") or {}
    if not time_ms:
        return
    lamp_ids = sorted(int(k) for k in lamps.keys())
    header = ["time_ms"] + [f"lamp_{lid + 1}" for lid in lamp_ids]
    n = len(time_ms)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            row = [float(time_ms[i])]
            for lid in lamp_ids:
                col = lamps.get(lid) or lamps.get(str(lid)) or []
                row.append(float(col[i]) if i < len(col) else "")
            w.writerow(row)


def build_p300_averages_export(p300_engine: Any) -> Optional[Dict[str, Any]]:
    """Build averaged ERP columns from accumulated epochs (no algorithm change)."""
    from p300_analysis.erp_compute import build_averaged_erp
    from p300_analysis.signal_processing import baseline_correction

    geom = p300_engine._epoch_geom
    if geom.time_ms_template is None or not p300_engine.epochs_data:
        return None
    epoch_len = int(geom.epoch_len)
    stim_keys, raw, rejected_counts = build_averaged_erp(
        p300_engine.epochs_data,
        epoch_len,
        artifact_threshold_uv=p300_engine.params.artifact_threshold_uv,
    )
    time_ms = np.asarray(geom.time_ms_template, dtype=np.float64)
    corrected = baseline_correction(raw, time_ms, baseline_ms=int(geom.baseline_ms))
    tiles: Dict[int, List[float]] = {}
    for i, key in enumerate(stim_keys):
        tid = stim_key_to_tile_digit(str(key))
        if tid is None or not (0 <= int(tid) <= 8):
            continue
        n_epochs = len(p300_engine.epochs_data.get(key, []))
        n_rejected = int(rejected_counts.get(key, 0))
        if n_epochs == 0 or (n_rejected >= n_epochs and n_epochs > 0):
            continue
        row_vals = corrected[i, :]
        if not np.any(np.isfinite(row_vals)):
            continue
        # Column names tile_0..tile_8 match 0-based stimulator tile_id.
        tiles[int(tid)] = row_vals.astype(float).tolist()
    return {"time_ms": time_ms.astype(float).tolist(), "tiles": tiles}


def build_ssvep_averages_export(
    ssvep_engine: Any,
    *,
    roi_channels_0idx: Sequence[int],
    n_lamps: int,
    max_ms: float = 3000.0,
    n_points: int = 600,
) -> Optional[Dict[str, Any]]:
    """Per-lamp mean ROI EEG during ON intervals (read-only export)."""
    buf = ssvep_engine._buf
    buf_t = ssvep_engine._buf_t
    if buf.shape[0] < 32 or buf_t.size < 32:
        return None
    gate = ssvep_engine._burst_gate
    ch = int(roi_channels_0idx[0]) if roi_channels_0idx else 0
    if ch < 0 or ch >= buf.shape[1]:
        ch = 0
    time_ms = np.linspace(0.0, float(max_ms), int(n_points), dtype=np.float64)
    lamps: Dict[int, List[float]] = {}
    for lamp in range(max(1, int(n_lamps))):
        idx = [
            i
            for i, t in enumerate(buf_t)
            if gate._lamp_on_at(int(lamp), float(t))  # noqa: SLF001 — export-only read
        ]
        if not idx:
            lamps[int(lamp)] = [0.0] * int(n_points)
            continue
        seg = np.asarray([float(buf[i, ch]) for i in idx], dtype=np.float64)
        dur_ms = float(max_ms)
        if len(idx) > 1:
            dur_ms = float((buf_t[idx[-1]] - buf_t[idx[0]]) * 1000.0)
            dur_ms = min(float(max_ms), max(dur_ms, 1.0))
        src_t = np.linspace(0.0, dur_ms, seg.size, dtype=np.float64)
        lamps[int(lamp)] = np.interp(time_ms, src_t, seg).astype(float).tolist()
    return {"time_ms": time_ms.astype(float).tolist(), "lamps": lamps}
