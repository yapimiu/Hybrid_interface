"""Deterministic replay/regression snapshots for historical experiment sessions.

This module intentionally does not re-run signal-processing yet. Stage 0/1 freezes
the currently recorded outputs from logs so future signal-processing changes can be
compared against a stable baseline.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BASELINE_SCHEMA = "hybrid_bci_replay_baseline/v1"
SESSION_SUMMARY_SCHEMA = "hybrid_bci_session_replay_summary/v1"
REPLAY_DIFF_SCHEMA = "hybrid_bci_replay_diff/v2"

REPLAY_MODE_DETERMINISTIC_LOG = "deterministic_log"
DEFAULT_PIPELINE_VERSION = "megafix-roadmap-pre-stage2"
DEFAULT_ANALYSIS_VERSION = "replay-compact-v2"


@dataclass(frozen=True)
class ReplayDiff:
    session: str
    category: str
    path: str
    expected: Any
    actual: Any
    classification: str = "unexpected"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session,
            "category": self.category,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "classification": self.classification,
        }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _event_sort_key(event: Dict[str, Any], line_index: int) -> Tuple[int, int, int, str, int]:
    """Stable ordering from log fields only (no wall-clock reads)."""

    def _as_int(value: Any, default: int = -1) -> int:
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    return (
        _as_int(event.get("seq")),
        _as_int(event.get("unix_ms")),
        _as_int(event.get("monotonic_ns")),
        str(event.get("event") or ""),
        int(line_index),
    )


def _read_ndjson(path: Path, *, deterministic: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"_decode_error": True, "raw": line[:500]})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if deterministic and rows:
        rows = [row for _, row in sorted(enumerate(rows), key=lambda t: _event_sort_key(t[1], t[0]))]
    return rows


def _timing_summaries(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive block/trial durations from logged timestamps only."""
    ssvep_pending: Dict[str, List[int]] = {}
    ssvep_durations_ms: List[int] = []
    ssvep_by_mode: Dict[str, List[int]] = {}
    p300_pending: List[int] = []
    p300_durations_ms: List[int] = []

    def _unix_ms(event: Dict[str, Any], data: Dict[str, Any]) -> Optional[int]:
        for key in ("unix_ms",):
            if key in event:
                try:
                    return int(event[key])
                except (TypeError, ValueError):
                    pass
            if key in data:
                try:
                    return int(data[key])
                except (TypeError, ValueError):
                    pass
        return None

    for event in events:
        name = str(event.get("event") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        ts = _unix_ms(event, data)
        if ts is None:
            continue
        if name == "ssvep_block_start":
            mode = str(data.get("mode") or "unknown")
            ssvep_pending.setdefault(mode, []).append(ts)
        elif name == "ssvep_block_end":
            mode = str(data.get("mode") or "unknown")
            starts = ssvep_pending.get(mode) or []
            if starts:
                start_ts = starts.pop(0)
                dur = max(0, int(ts) - int(start_ts))
                ssvep_durations_ms.append(dur)
                ssvep_by_mode.setdefault(mode, []).append(dur)
        elif name in ("protocol_trial_start_arm", "p300_block_start"):
            p300_pending.append(ts)
        elif name in ("trial_decision_dual", "p300_trial_end"):
            if p300_pending:
                start_ts = p300_pending.pop(0)
                p300_durations_ms.append(max(0, int(ts) - int(start_ts)))

    def _stats(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "min_ms": None, "max_ms": None, "mean_ms": None}
        return {
            "count": len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": round(sum(values) / len(values), 2),
        }

    return {
        "ssvep_block_duration_ms": _stats(ssvep_durations_ms),
        "ssvep_block_duration_ms_by_mode": {
            mode: _stats(durs) for mode, durs in sorted(ssvep_by_mode.items())
        },
        "p300_trial_duration_ms": _stats(p300_durations_ms),
        "unpaired_ssvep_starts": {mode: len(v) for mode, v in sorted(ssvep_pending.items()) if v},
        "unpaired_p300_starts": len(p300_pending),
    }


def _coef_summaries(ssvep: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lengths: List[int] = []
    degenerate = 0
    length_mismatch = 0
    for row in ssvep:
        shape = row.get("coef_shape") if isinstance(row.get("coef_shape"), dict) else {}
        if shape.get("state") == "missing" or shape.get("all_zero_or_null"):
            degenerate += 1
        length = int(shape.get("length") or 0)
        lengths.append(length)
        expected = shape.get("expected_length")
        if expected is not None and length != int(expected):
            length_mismatch += 1
    return {
        "blocks": len(ssvep),
        "degenerate_blocks": degenerate,
        "length_mismatch_blocks": length_mismatch,
        "coef_lengths": lengths,
    }


def _template_metrics_summary(p300: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    auc_margins: List[float] = []
    tpl_margins: List[float] = []
    epoch_totals: List[int] = []
    for row in p300:
        auc_margin = (row.get("auc") or {}).get("margin")
        tpl_margin = (row.get("template_corr") or {}).get("margin")
        if isinstance(auc_margin, (int, float)):
            auc_margins.append(float(auc_margin))
        if isinstance(tpl_margin, (int, float)):
            tpl_margins.append(float(tpl_margin))
        counts = row.get("epoch_counts_by_stim")
        if isinstance(counts, dict):
            epoch_totals.append(int(sum(int(v) for v in counts.values())))
    def _margin_stats(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 6),
        }

    return {
        "trials": len(p300),
        "auc_margin": _margin_stats(auc_margins),
        "template_margin": _margin_stats(tpl_margins),
        "epoch_total": {
            "count": len(epoch_totals),
            "min": min(epoch_totals) if epoch_totals else None,
            "max": max(epoch_totals) if epoch_totals else None,
            "mean": round(sum(epoch_totals) / len(epoch_totals), 2) if epoch_totals else None,
        },
    }


def _known_issues(
    *,
    missing_stop: bool,
    ssvep_mismatches: Dict[str, Any],
    coef_bad: int,
) -> List[str]:
    issues: List[str] = []
    if missing_stop:
        issues.append("missing_session_stop")
    if ssvep_mismatches:
        issues.append("ssvep_start_end_mismatch")
    if coef_bad > 0:
        issues.append("degenerate_ssvep_coef")
    return issues


def _coef_shape(coef: Any) -> Dict[str, Any]:
    if coef is None:
        return {"state": "missing", "length": 0, "all_zero_or_null": False}
    if not isinstance(coef, list):
        return {"state": "scalar", "length": 1, "all_zero_or_null": False}
    all_zero_or_null = all(x is None or x == 0 or x == 0.0 for x in coef)
    return {
        "state": "vector",
        "length": len(coef),
        "all_zero_or_null": bool(all_zero_or_null),
    }


def _compact_p300(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    auc = row.get("auc") if isinstance(row.get("auc"), dict) else {}
    tpl = row.get("template_corr") if isinstance(row.get("template_corr"), dict) else {}
    auc_dbg = auc.get("debug") if isinstance(auc.get("debug"), dict) else {}
    tpl_dbg = tpl.get("debug") if isinstance(tpl.get("debug"), dict) else {}
    counts = row.get("epoch_counts_by_stim")
    if not isinstance(counts, dict):
        counts = {}
    return {
        "index": index,
        "phase": row.get("phase"),
        "queue_index": row.get("queue_index"),
        "cue_target_tile_id": row.get("cue_target_tile_id"),
        "ground_truth_tile_id": row.get("ground_truth_tile_id"),
        "auc": {
            "can_decide": auc.get("can_decide"),
            "winner_tile_id": auc.get("winner_tile_id"),
            "mode_used": auc.get("mode_used"),
            "min_epochs_per_class": auc.get("min_epochs_per_class"),
            "margin": auc_dbg.get("margin"),
            "dt_ms": auc_dbg.get("dt_ms"),
            "corr_window_shape": auc_dbg.get("corr_window_shape"),
        },
        "template_corr": {
            "can_decide": tpl.get("can_decide"),
            "winner_tile_id": tpl.get("winner_tile_id"),
            "mode_used": tpl.get("mode_used"),
            "min_epochs_per_class": tpl.get("min_epochs_per_class"),
            "margin": tpl_dbg.get("margin"),
            "dt_ms": tpl_dbg.get("dt_ms"),
            "corr_window_shape": tpl_dbg.get("corr_window_shape"),
        },
        "methods_agree": row.get("methods_agree"),
        "correct_auc": row.get("correct_auc"),
        "correct_template": row.get("correct_template"),
        "epoch_counts_by_stim": {str(k): int(v) for k, v in sorted(counts.items())},
        "trial_end_seen": row.get("trial_end_seen"),
        "template_target_id": row.get("template_target_id"),
    }


def _compact_ssvep(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    coef = row.get("coef")
    freqs = row.get("ssvep_freqs_hz")
    expected_coef_len = len(freqs) if isinstance(freqs, list) else None
    return {
        "index": index,
        "mode": row.get("mode") or row.get("ssvep_mode"),
        "target_lamp": row.get("target_lamp") if "target_lamp" in row else row.get("ground_truth_lamp_0idx"),
        "winner_1based": row.get("winner_1based"),
        "winner_0idx": row.get("winner_0idx"),
        "classify_allowed": row.get("classify_allowed"),
        "correct": row.get("correct"),
        "coef": coef,
        "coef_shape": {
            **_coef_shape(coef),
            "expected_length": expected_coef_len,
        },
        "debug_note": (row.get("debug") or {}).get("note") if isinstance(row.get("debug"), dict) else None,
        "gate_reason": (row.get("debug") or {}).get("gate_reason") if isinstance(row.get("debug"), dict) else None,
        "buf_shape": (row.get("debug") or {}).get("buf_shape") if isinstance(row.get("debug"), dict) else None,
    }


def _session_display_name(session_dir: Path, manifest: Dict[str, Any]) -> str:
    sid = str(manifest.get("session_id") or session_dir.name)
    return sid


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _is_minimal_session(session_dir: Path) -> bool:
    return (session_dir / "all_experiments.csv").is_file() and not (
        session_dir / "experiments.ndjson"
    ).is_file()


def _minimal_row_to_p300(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    cue = row.get("target_tile")
    try:
        cue_i = int(cue) if cue not in ("", None) else None
    except (TypeError, ValueError):
        cue_i = None
    auc_w = row.get("auc_winner")
    tpl_w = row.get("template_winner")
    return {
        "index": index,
        "phase": row.get("phase"),
        "cue_target_tile_id": cue_i,
        "ground_truth_tile_id": cue_i,
        "auc": {
            "can_decide": auc_w not in ("", None),
            "winner_tile_id": int(auc_w) if auc_w not in ("", None) else None,
            "mode_used": "auc",
        },
        "template_corr": {
            "can_decide": tpl_w not in ("", None),
            "winner_tile_id": int(tpl_w) if tpl_w not in ("", None) else None,
            "mode_used": "template_corr",
        },
        "methods_agree": row.get("methods_agree") in (True, "True", "true", "1"),
        "correct_auc": row.get("auc_correct") in (True, "True", "true", "1"),
        "correct_template": row.get("template_correct") in (True, "True", "true", "1"),
        "epoch_counts_by_stim": {},
    }


def _minimal_row_to_ssvep(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    lamp = row.get("target_lamp")
    try:
        lamp_1 = int(lamp) if lamp not in ("", None) else None
    except (TypeError, ValueError):
        lamp_1 = None
    maj = row.get("majority_winner")
    try:
        maj_1 = int(maj) if maj not in ("", None) else None
    except (TypeError, ValueError):
        maj_1 = None
    return {
        "index": index,
        "mode": row.get("ssvep_mode"),
        "target_lamp": lamp_1 - 1 if lamp_1 is not None else None,
        "winner_1based": maj_1,
        "winner_0idx": maj_1 - 1 if maj_1 is not None else None,
        "classify_allowed": maj_1 is not None,
        "correct": row.get("msi_correct") in (True, "True", "true", "1"),
        "coef": None,
        "coef_shape": _coef_shape(None),
    }


def _summarize_minimal_session(
    session_dir: Path,
    *,
    replay_mode: str = REPLAY_MODE_DETERMINISTIC_LOG,
) -> Dict[str, Any]:
    rows = _read_csv_rows(session_dir / "all_experiments.csv")
    marker_rows = _read_csv_rows(session_dir / "eeg_with_markers.csv")
    event_counts = Counter(
        str(r.get("event_type"))
        for r in marker_rows
        if str(r.get("event_type") or "").strip()
    )
    ssvep_start_mode = Counter()
    ssvep_end_mode = Counter()
    for r in marker_rows:
        et = str(r.get("event_type") or "")
        ev = str(r.get("event_value") or "")
        if et == "ssvep_start":
            ssvep_start_mode[ev or "unknown"] += 1
        elif et == "ssvep_end":
            ssvep_end_mode[ev or "unknown"] += 1

    p300_rows = [r for r in rows if str(r.get("experiment_type")) == "p300"]
    ssvep_rows = [r for r in rows if str(r.get("experiment_type")) == "ssvep"]
    p300 = [_minimal_row_to_p300(r, i + 1) for i, r in enumerate(p300_rows)]
    ssvep = [_minimal_row_to_ssvep(r, i + 1) for i, r in enumerate(ssvep_rows)]

    ssvep_mismatches = {
        mode: {
            "starts": int(ssvep_start_mode.get(mode, 0)),
            "ends": int(ssvep_end_mode.get(mode, 0)),
        }
        for mode in sorted(set(ssvep_start_mode) | set(ssvep_end_mode))
        if ssvep_start_mode.get(mode, 0) != ssvep_end_mode.get(mode, 0)
    }
    coef_bad = sum(
        1
        for row in ssvep
        if row["coef_shape"]["state"] == "missing" or row["coef_shape"]["all_zero_or_null"]
    )
    ssvep_correct = sum(1 for row in ssvep if row.get("correct") is True)
    known = _known_issues(
        missing_stop=False,
        ssvep_mismatches=ssvep_mismatches,
        coef_bad=coef_bad,
    )
    return {
        "schema": SESSION_SUMMARY_SCHEMA,
        "replay_mode": replay_mode,
        "session_id": session_dir.name,
        "session_dir_name": session_dir.name,
        "manifest": {
            "schema": "hybrid_protocol/minimal_v1",
            "subject_id": None,
            "n_events": int(sum(event_counts.values())),
            "n_experiments": len(rows),
            "eeg_samples": len(_read_csv_rows(session_dir / "eeg_raw.csv")),
            "eeg_channels": None,
            "stop_reason": None,
        },
        "counts": {
            "events": int(sum(event_counts.values())),
            "experiments": len(rows),
            "p300_trials": len(p300),
            "ssvep_blocks": len(ssvep),
            "session_start": 0,
            "session_stop": 0,
            "ssvep_block_start": int(sum(ssvep_start_mode.values())),
            "ssvep_block_end": int(sum(ssvep_end_mode.values())),
            "ssvep_no_decision": int(sum(1 for row in ssvep if row.get("classify_allowed") is False)),
            "ssvep_bad_coef": int(coef_bad),
            "ssvep_correct": int(ssvep_correct),
            "p300_auc_decisions": int(sum(1 for row in p300 if row["auc"]["can_decide"] is True)),
            "p300_template_decisions": int(
                sum(1 for row in p300 if row["template_corr"]["can_decide"] is True)
            ),
        },
        "lifecycle": {
            "event_counts": {str(k): int(v) for k, v in sorted(event_counts.items())},
            "ssvep_start_by_mode": {str(k): int(v) for k, v in sorted(ssvep_start_mode.items())},
            "ssvep_end_by_mode": {str(k): int(v) for k, v in sorted(ssvep_end_mode.items())},
            "missing_session_stop": False,
            "ssvep_start_end_mismatches": ssvep_mismatches,
        },
        "known_issues": known,
        "timing_summaries": {},
        "coef_summaries": _coef_summaries(ssvep),
        "template_metrics": _template_metrics_summary(p300),
        "outputs": {
            "p300_auc_winners": [row["auc"]["winner_tile_id"] for row in p300],
            "p300_template_winners": [row["template_corr"]["winner_tile_id"] for row in p300],
            "ssvep_winners_0idx": [row.get("winner_0idx") for row in ssvep],
            "p300": p300,
            "ssvep": ssvep,
        },
    }


def summarize_session(
    session_dir: Path,
    *,
    deterministic: bool = True,
    replay_mode: str = REPLAY_MODE_DETERMINISTIC_LOG,
) -> Dict[str, Any]:
    """Build a deterministic compact summary for one historical session."""
    session_dir = Path(session_dir)
    if _is_minimal_session(session_dir):
        return _summarize_minimal_session(session_dir, replay_mode=replay_mode)
    manifest = _read_json(session_dir / "manifest.json")
    events = _read_ndjson(session_dir / "events.ndjson", deterministic=deterministic)
    experiments = _read_ndjson(session_dir / "experiments.ndjson")
    p300_trials = _read_ndjson(session_dir / "p300_trials.ndjson")
    ssvep_blocks = _read_ndjson(session_dir / "ssvep_blocks.ndjson")

    event_counts = Counter(str(e.get("event")) for e in events if e.get("event") is not None)
    ssvep_start_mode = Counter()
    ssvep_end_mode = Counter()
    for event in events:
        name = event.get("event")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if name == "ssvep_block_start" and "mode" in data:
            ssvep_start_mode[str(data.get("mode"))] += 1
        elif name == "ssvep_block_end":
            ssvep_end_mode[str(data.get("mode"))] += 1

    p300_rows = [row for row in experiments if row.get("kind") == "p300"]
    if not p300_rows:
        p300_rows = p300_trials
    ssvep_rows = [row for row in experiments if row.get("kind") == "ssvep"]
    if not ssvep_rows:
        ssvep_rows = ssvep_blocks

    p300 = [_compact_p300(row, i + 1) for i, row in enumerate(p300_rows)]
    ssvep = [_compact_ssvep(row, i + 1) for i, row in enumerate(ssvep_rows)]

    missing_stop = event_counts.get("session_stop", 0) == 0 and not manifest.get("stop")
    ssvep_mismatches = {
        mode: {
            "starts": int(ssvep_start_mode.get(mode, 0)),
            "ends": int(ssvep_end_mode.get(mode, 0)),
        }
        for mode in sorted(set(ssvep_start_mode) | set(ssvep_end_mode))
        if ssvep_start_mode.get(mode, 0) != ssvep_end_mode.get(mode, 0)
    }
    coef_bad = sum(
        1
        for row in ssvep
        if row["coef_shape"]["state"] == "missing" or row["coef_shape"]["all_zero_or_null"]
    )
    ssvep_correct = sum(1 for row in ssvep if row.get("correct") is True)
    p300_auc_changed_payload = [
        row["auc"]["winner_tile_id"]
        for row in p300
    ]
    p300_tpl_changed_payload = [
        row["template_corr"]["winner_tile_id"]
        for row in p300
    ]
    ssvep_winners = [row.get("winner_0idx") for row in ssvep]

    known = _known_issues(
        missing_stop=missing_stop,
        ssvep_mismatches=ssvep_mismatches,
        coef_bad=coef_bad,
    )

    return {
        "schema": SESSION_SUMMARY_SCHEMA,
        "replay_mode": replay_mode,
        "session_id": _session_display_name(session_dir, manifest),
        "session_dir_name": session_dir.name,
        "manifest": {
            "schema": manifest.get("schema"),
            "subject_id": manifest.get("subject_id"),
            "n_events": manifest.get("n_events"),
            "n_experiments": manifest.get("n_experiments"),
            "eeg_samples": manifest.get("eeg_samples"),
            "eeg_channels": manifest.get("eeg_channels"),
            "stop_reason": (manifest.get("stop") or {}).get("reason") if isinstance(manifest.get("stop"), dict) else None,
        },
        "counts": {
            "events": len(events),
            "experiments": len(experiments),
            "p300_trials": len(p300),
            "ssvep_blocks": len(ssvep),
            "session_start": int(event_counts.get("session_start", 0)),
            "session_stop": int(event_counts.get("session_stop", 0)),
            "ssvep_block_start": int(sum(ssvep_start_mode.values())),
            "ssvep_block_end": int(sum(ssvep_end_mode.values())),
            "ssvep_no_decision": int(sum(1 for row in ssvep if row.get("classify_allowed") is False)),
            "ssvep_bad_coef": int(coef_bad),
            "ssvep_correct": int(ssvep_correct),
            "p300_auc_decisions": int(sum(1 for row in p300 if row["auc"]["can_decide"] is True)),
            "p300_template_decisions": int(sum(1 for row in p300 if row["template_corr"]["can_decide"] is True)),
        },
        "lifecycle": {
            "event_counts": {str(k): int(v) for k, v in sorted(event_counts.items())},
            "ssvep_start_by_mode": {str(k): int(v) for k, v in sorted(ssvep_start_mode.items())},
            "ssvep_end_by_mode": {str(k): int(v) for k, v in sorted(ssvep_end_mode.items())},
            "missing_session_stop": bool(missing_stop),
            "ssvep_start_end_mismatches": ssvep_mismatches,
        },
        "known_issues": known,
        "timing_summaries": _timing_summaries(events),
        "coef_summaries": _coef_summaries(ssvep),
        "template_metrics": _template_metrics_summary(p300),
        "outputs": {
            "p300_auc_winners": p300_auc_changed_payload,
            "p300_template_winners": p300_tpl_changed_payload,
            "ssvep_winners_0idx": ssvep_winners,
            "p300": p300,
            "ssvep": ssvep,
        },
    }


def discover_sessions(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("session_*") if (p / "events.ndjson").is_file())


def build_baseline(
    sessions: Sequence[Path],
    *,
    label: str = "historical",
    replay_mode: str = REPLAY_MODE_DETERMINISTIC_LOG,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    analysis_version: str = DEFAULT_ANALYSIS_VERSION,
    deterministic: bool = True,
) -> Dict[str, Any]:
    summaries = [
        summarize_session(Path(p), deterministic=deterministic, replay_mode=replay_mode)
        for p in sessions
    ]
    return {
        "schema": BASELINE_SCHEMA,
        "schema_version": 1,
        "pipeline_version": str(pipeline_version),
        "analysis_version": str(analysis_version),
        "replay_mode": str(replay_mode),
        "label": str(label),
        "sessions": summaries,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_baseline(path: Path) -> Dict[str, Any]:
    data = _read_json(Path(path))
    if data.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"unexpected baseline schema in {path}: {data.get('schema')!r}")
    data.setdefault("schema_version", 1)
    data.setdefault("pipeline_version", DEFAULT_PIPELINE_VERSION)
    data.setdefault("analysis_version", DEFAULT_ANALYSIS_VERSION)
    data.setdefault("replay_mode", REPLAY_MODE_DETERMINISTIC_LOG)
    return data


def _session_map(baseline: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in baseline.get("sessions", []):
        if isinstance(row, dict):
            out[str(row.get("session_dir_name") or row.get("session_id"))] = row
    return out


def _compare_value(
    diffs: List[ReplayDiff],
    *,
    session: str,
    category: str,
    path: str,
    expected: Any,
    actual: Any,
    expected_session: Optional[Dict[str, Any]] = None,
    actual_session: Optional[Dict[str, Any]] = None,
) -> None:
    if expected != actual:
        classification = _classify_diff(
            path,
            expected,
            actual,
            expected_session=expected_session,
            actual_session=actual_session,
        )
        diffs.append(
            ReplayDiff(session, category, path, expected, actual, classification=classification)
        )


def _classify_diff(
    path: str,
    expected: Any,
    actual: Any,
    *,
    expected_session: Optional[Dict[str, Any]],
    actual_session: Optional[Dict[str, Any]],
) -> str:
    exp_issues = set((expected_session or {}).get("known_issues") or [])
    act_issues = set((actual_session or {}).get("known_issues") or [])

    if path == "lifecycle.missing_session_stop":
        if expected is True and actual is False and "missing_session_stop" in exp_issues:
            return "expected_fix"
        if expected is False and actual is True:
            return "unexpected"
    if path == "lifecycle.ssvep_start_end_mismatches":
        if expected and not actual and "ssvep_start_end_mismatch" in exp_issues:
            return "expected_fix"
        if not expected and actual:
            return "unexpected"
    if path == "counts.ssvep_bad_coef":
        try:
            if int(expected or 0) > 0 and int(actual or 0) < int(expected or 0):
                if "degenerate_ssvep_coef" in exp_issues:
                    return "expected_fix"
            if int(actual or 0) > int(expected or 0):
                return "unexpected"
        except (TypeError, ValueError):
            pass
    if path in (
        "outputs.p300_auc_winners",
        "outputs.p300_template_winners",
        "outputs.ssvep_winners_0idx",
        "outputs.p300",
        "outputs.ssvep",
    ):
        return "unexpected"
    if exp_issues and act_issues and exp_issues != act_issues:
        fixed = exp_issues - act_issues
        if fixed and path.startswith("lifecycle."):
            return "expected_fix"
    return "unexpected"


def compare_baselines(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Compare compact behavior fields. Timestamps and raw event totals remain visible but
    only deterministic behavioral summaries are treated as regression gates.
    """
    exp = _session_map(expected)
    act = _session_map(actual)
    diffs: List[ReplayDiff] = []
    for key in sorted(set(exp) | set(act)):
        if key not in exp:
            diffs.append(ReplayDiff(key, "session", "session", None, "added"))
            continue
        if key not in act:
            diffs.append(ReplayDiff(key, "session", "session", "present", None))
            continue
        e = exp[key]
        a = act[key]
        for path in (
            ("counts", "p300_trials"),
            ("counts", "ssvep_blocks"),
            ("counts", "session_stop"),
            ("counts", "ssvep_block_start"),
            ("counts", "ssvep_block_end"),
            ("counts", "ssvep_no_decision"),
            ("counts", "ssvep_bad_coef"),
            ("counts", "ssvep_correct"),
            ("counts", "p300_auc_decisions"),
            ("counts", "p300_template_decisions"),
            ("lifecycle", "missing_session_stop"),
            ("lifecycle", "ssvep_start_end_mismatches"),
            ("outputs", "p300_auc_winners"),
            ("outputs", "p300_template_winners"),
            ("outputs", "ssvep_winners_0idx"),
        ):
            p0, p1 = path
            _compare_value(
                diffs,
                session=key,
                category=p0,
                path=f"{p0}.{p1}",
                expected=(e.get(p0) or {}).get(p1),
                actual=(a.get(p0) or {}).get(p1),
                expected_session=e,
                actual_session=a,
            )
        _compare_value(
            diffs,
            session=key,
            category="outputs",
            path="outputs.p300",
            expected=(e.get("outputs") or {}).get("p300"),
            actual=(a.get("outputs") or {}).get("p300"),
            expected_session=e,
            actual_session=a,
        )
        _compare_value(
            diffs,
            session=key,
            category="outputs",
            path="outputs.ssvep",
            expected=(e.get("outputs") or {}).get("ssvep"),
            actual=(a.get("outputs") or {}).get("ssvep"),
            expected_session=e,
            actual_session=a,
        )

    by_category = Counter(d.category for d in diffs)
    by_classification = Counter(d.classification for d in diffs)
    unexpected = [d for d in diffs if d.classification == "unexpected"]
    return {
        "schema": REPLAY_DIFF_SCHEMA,
        "schema_version": 2,
        "status": "pass" if not unexpected else "fail",
        "diff_count": len(diffs),
        "unexpected_diff_count": len(unexpected),
        "expected_fix_count": int(by_classification.get("expected_fix", 0)),
        "diffs_by_category": {str(k): int(v) for k, v in sorted(by_category.items())},
        "diffs_by_classification": {str(k): int(v) for k, v in sorted(by_classification.items())},
        "structured_report": build_structured_regression_report(expected, actual, diffs),
        "diffs": [d.to_dict() for d in diffs],
    }


def build_structured_regression_report(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    diffs: Sequence[ReplayDiff],
) -> Dict[str, Any]:
    exp = _session_map(expected)
    act = _session_map(actual)
    winners_changed: List[Dict[str, Any]] = []
    coef_drift: List[Dict[str, Any]] = []
    lifecycle_changes: List[Dict[str, Any]] = []
    timing_changes: List[Dict[str, Any]] = []

    for key in sorted(set(exp) & set(act)):
        e, a = exp[key], act[key]
        e_out, a_out = e.get("outputs") or {}, a.get("outputs") or {}
        if e_out.get("p300_auc_winners") != a_out.get("p300_auc_winners"):
            winners_changed.append(
                {
                    "session": key,
                    "metric": "p300_auc_winners",
                    "expected": e_out.get("p300_auc_winners"),
                    "actual": a_out.get("p300_auc_winners"),
                }
            )
        if e_out.get("p300_template_winners") != a_out.get("p300_template_winners"):
            winners_changed.append(
                {
                    "session": key,
                    "metric": "p300_template_winners",
                    "expected": e_out.get("p300_template_winners"),
                    "actual": a_out.get("p300_template_winners"),
                }
            )
        if e_out.get("ssvep_winners_0idx") != a_out.get("ssvep_winners_0idx"):
            winners_changed.append(
                {
                    "session": key,
                    "metric": "ssvep_winners_0idx",
                    "expected": e_out.get("ssvep_winners_0idx"),
                    "actual": a_out.get("ssvep_winners_0idx"),
                }
            )
        e_coef, a_coef = e.get("coef_summaries") or {}, a.get("coef_summaries") or {}
        if e_coef != a_coef:
            coef_drift.append({"session": key, "expected": e_coef, "actual": a_coef})
        e_lc, a_lc = e.get("lifecycle") or {}, a.get("lifecycle") or {}
        for field in ("missing_session_stop", "ssvep_start_end_mismatches"):
            if e_lc.get(field) != a_lc.get(field):
                lifecycle_changes.append(
                    {
                        "session": key,
                        "field": field,
                        "expected": e_lc.get(field),
                        "actual": a_lc.get(field),
                        "expected_known_issues": e.get("known_issues"),
                        "actual_known_issues": a.get("known_issues"),
                    }
                )
        e_tm, a_tm = e.get("timing_summaries") or {}, a.get("timing_summaries") or {}
        if e_tm != a_tm:
            timing_changes.append({"session": key, "expected": e_tm, "actual": a_tm})

    unexpected = [d.to_dict() for d in diffs if d.classification == "unexpected"]
    expected_fixes = [d.to_dict() for d in diffs if d.classification == "expected_fix"]

    return {
        "pipeline_version": {
            "expected": expected.get("pipeline_version"),
            "actual": actual.get("pipeline_version"),
        },
        "analysis_version": {
            "expected": expected.get("analysis_version"),
            "actual": actual.get("analysis_version"),
        },
        "replay_mode": {
            "expected": expected.get("replay_mode"),
            "actual": actual.get("replay_mode"),
        },
        "winners_changed": winners_changed,
        "winners_unchanged_sessions": sorted(
            k
            for k in set(exp) & set(act)
            if not any(w["session"] == k for w in winners_changed)
        ),
        "coef_drift": coef_drift,
        "lifecycle_changes": lifecycle_changes,
        "timing_changes": timing_changes,
        "unexpected_diffs": unexpected,
        "expected_fixes": expected_fixes,
    }


def render_structured_regression_report(diff: Dict[str, Any], actual: Dict[str, Any]) -> str:
    structured = diff.get("structured_report") if isinstance(diff.get("structured_report"), dict) else {}
    lines = [
        "# Structured Replay Regression Report",
        "",
        f"- Gate status: **{diff.get('status', 'unknown').upper()}**",
        f"- Total diffs: {diff.get('diff_count', 0)}",
        f"- Unexpected diffs: {diff.get('unexpected_diff_count', 0)}",
        f"- Expected fixes: {diff.get('expected_fix_count', 0)}",
        f"- Pipeline version: `{actual.get('pipeline_version')}`",
        f"- Analysis version: `{actual.get('analysis_version')}`",
        f"- Replay mode: `{actual.get('replay_mode')}`",
        "",
        "## Winners",
        "",
    ]
    winners = structured.get("winners_changed") or []
    if not winners:
        lines.append("No winner drift detected.")
    else:
        for row in winners:
            lines.append(
                f"- `{row.get('session')}` **{row.get('metric')}**: "
                f"expected `{row.get('expected')}` → actual `{row.get('actual')}`"
            )
    unchanged = structured.get("winners_unchanged_sessions") or []
    lines.extend(["", f"Unchanged winner sessions: {len(unchanged)}", ""])
    lines.extend(["## Coef drift", ""])
    coef = structured.get("coef_drift") or []
    if not coef:
        lines.append("No coef summary drift.")
    else:
        for row in coef[:50]:
            lines.append(f"- `{row.get('session')}`: expected `{row.get('expected')}` actual `{row.get('actual')}`")
    lines.extend(["", "## Lifecycle", ""])
    for row in structured.get("lifecycle_changes") or []:
        lines.append(
            f"- `{row.get('session')}` `{row.get('field')}`: "
            f"{row.get('expected')} → {row.get('actual')} "
            f"(known: {row.get('expected_known_issues')} → {row.get('actual_known_issues')})"
        )
    if not structured.get("lifecycle_changes"):
        lines.append("No lifecycle drift.")
    lines.extend(["", "## Timing (informational)", ""])
    for row in structured.get("timing_changes") or []:
        lines.append(f"- `{row.get('session')}` timing summary changed (not part of strict gate).")
    if not structured.get("timing_changes"):
        lines.append("No timing summary drift.")
    lines.extend(["", "## Unexpected vs expected fixes", ""])
    if structured.get("unexpected_diffs"):
        lines.append("### Unexpected")
        for item in structured["unexpected_diffs"][:100]:
            lines.append(
                f"- `{item.get('session')}` `{item.get('path')}`: "
                f"`{item.get('expected')}` → `{item.get('actual')}`"
            )
    else:
        lines.append("No unexpected diffs.")
    if structured.get("expected_fixes"):
        lines.extend(["", "### Expected fixes (improvements over known baseline issues)", ""])
        for item in structured["expected_fixes"][:100]:
            lines.append(
                f"- `{item.get('session')}` `{item.get('path')}`: "
                f"`{item.get('expected')}` → `{item.get('actual')}`"
            )
    return "\n".join(lines) + "\n"


def render_markdown_report(diff: Dict[str, Any], actual: Dict[str, Any]) -> str:
    lines = [
        "# Replay Regression Report",
        "",
        f"- Status: **{diff.get('status', 'unknown').upper()}**",
        f"- Diff count: {diff.get('diff_count', 0)}",
        f"- Unexpected diffs: {diff.get('unexpected_diff_count', 0)}",
        f"- Expected fixes: {diff.get('expected_fix_count', 0)}",
        f"- Sessions: {len(actual.get('sessions', []))}",
        f"- Pipeline version: `{actual.get('pipeline_version')}`",
        f"- Analysis version: `{actual.get('analysis_version')}`",
        f"- Replay mode: `{actual.get('replay_mode')}`",
        "",
        "## Sessions",
        "",
        "| Session | P300 | SSVEP | Stop | SSVEP bad coef | SSVEP correct | Known issues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in actual.get("sessions", []):
        counts = s.get("counts") or {}
        lifecycle = s.get("lifecycle") or {}
        issues = list(s.get("known_issues") or [])
        lines.append(
            "| {name} | {p300} | {ssvep_start}/{ssvep_end} | {stop} | {bad_coef} | {correct}/{ssvep} | {issues} |".format(
                name=s.get("session_dir_name") or s.get("session_id"),
                p300=counts.get("p300_trials", 0),
                ssvep_start=counts.get("ssvep_block_start", 0),
                ssvep_end=counts.get("ssvep_block_end", 0),
                stop=counts.get("session_stop", 0),
                bad_coef=counts.get("ssvep_bad_coef", 0),
                correct=counts.get("ssvep_correct", 0),
                ssvep=counts.get("ssvep_blocks", 0),
                issues=", ".join(issues) if issues else "none",
            )
        )
    lines.extend(["", "## Drift Analysis", ""])
    if diff.get("status") == "pass":
        lines.append("No behavior drift detected against the selected golden baseline.")
    else:
        for item in diff.get("diffs", [])[:200]:
            lines.append(
                f"- `{item.get('session')}` `{item.get('path')}`: "
                f"expected `{item.get('expected')}` actual `{item.get('actual')}`"
            )
    lines.extend(
        [
            "",
            "## Gate Scope",
            "",
            "- P300 winners and compact metrics are compared.",
            "- SSVEP winners, classify_allowed and coef shape/value summaries are compared.",
            "- Lifecycle counters and missing stop/start/end mismatches are compared.",
            "- Timing summaries, coef summaries and template metrics are recorded for golden snapshots but are informational unless explicitly promoted to gate fields.",
            "- Only `unexpected` diffs fail the gate; `expected_fix` improvements over known baseline issues are reported separately.",
            "- Raw EEG re-processing is intentionally not part of Stage 0/1; this report freezes recorded outputs before signal-processing changes.",
        ]
    )
    structured_md = render_structured_regression_report(diff, actual)
    return "\n".join(lines) + "\n\n" + structured_md


def write_markdown(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

