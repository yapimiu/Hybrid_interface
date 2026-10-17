#!/usr/bin/env python3
"""Offline before/after MSI burst replay on recorded human sessions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ssvep_analysis.burst_gate import (  # noqa: E402
    BurstGate,
    BurstGateConfig,
    append_chunk_timestamps,
    build_on_only_msi_window,
)
from ssvep_analysis.experiment_logger import coef_values  # noqa: E402
from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams  # noqa: E402
from scripts import test_msi_exec as tme  # noqa: E402

LEGACY_GATE = BurstGateConfig(
    window_sec=3.0,
    min_on_fraction=0.70,
    min_on_sec=1.4,
    grace_sec=0.0,
    min_on_fraction_grace=0.70,
    min_on_sec_grace=1.4,
)
FIXED_GATE = BurstGateConfig(window_sec=3.0)


@dataclass
class BurstBlock:
    target_lamp_0idx: int
    t_start: float
    t_end: float
    freqs_hz: Tuple[float, ...]


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_burst_blocks(events_path: Path) -> List[BurstBlock]:
    blocks: List[BurstBlock] = []
    pending: Optional[Dict[str, Any]] = None
    for ev in _read_ndjson(events_path):
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if name == "ssvep_block_start" and str(data.get("mode")) == "burst":
            pending = {"start_ms": int(ev["unix_ms"]), "data": data}
        elif name == "ssvep_block_end" and pending is not None:
            end_ms = int(ev.get("unix_ms", pending["start_ms"]))
            d = pending["data"]
            blocks.append(
                BurstBlock(
                    target_lamp_0idx=int(d.get("target_lamp", d.get("experiment_target_lamp_0idx", 0))),
                    t_start=float(pending["start_ms"]) / 1000.0,
                    t_end=float(end_ms) / 1000.0,
                    freqs_hz=tuple(float(x) for x in d.get("freqs_hz", (8.0, 10.0, 12.0, 14.0))),
                )
            )
            pending = None
    return blocks


def _load_led_markers(events_path: Path) -> List[Tuple[float, int, bool]]:
    out: List[Tuple[float, int, bool]] = []
    for ev in _read_ndjson(events_path):
        if str(ev.get("event")) != "migalka_led":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        out.append((float(ev["unix_ms"]) / 1000.0, int(data["lamp"]), bool(data["on"])))
    return out


def _infer_fs(times: np.ndarray) -> float:
    if times.size < 2:
        return 5000.0
    dt = float(np.median(np.diff(times[: min(10000, times.size)])))
    return 1.0 / dt if dt > 0 else 5000.0


def _classify_decisions(
    *,
    times: np.ndarray,
    eeg: np.ndarray,
    markers: Sequence[Tuple[float, int, bool]],
    block: BurstBlock,
    roi: Tuple[int, ...],
    fs: float,
    n_samples: int,
    gate_cfg: BurstGateConfig,
    on_only: bool,
    misaligned_markers: bool,
    classify_interval_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    gate = BurstGate(gate_cfg)
    gate.set_active_lamps(len(block.freqs_hz))
    gate.set_block_start(block.t_start)

    lookback = block.t_start - max(12.0, float(n_samples) / fs * 2.5)
    i0 = int(np.searchsorted(times, lookback, side="left"))
    i1 = int(np.searchsorted(times, block.t_end, side="right"))
    if i1 <= i0:
        return []

    msi, _, _ = tme.load_msi_runtime()
    win_sec = float(n_samples) / fs
    np_models = tme.generate_model_signals(list(block.freqs_hz), fs, win_sec)
    model_list = tme.build_model_signal_list(msi, np_models, verbose=False)
    msi.ModelSignal = model_list

    buf = np.zeros((0, eeg.shape[1]), dtype=np.float64)
    buf_t = np.zeros(0, dtype=np.float64)
    m_idx = 0
    decisions: List[Dict[str, Any]] = []
    last_classify_t: Optional[float] = None
    chunk = max(256, int(fs // 4))

    for i in range(i0, i1, chunk):
        j = min(i1, i + chunk)
        part = np.ascontiguousarray(eeg[i:j, :], dtype=np.float64)
        ts_chunk = [float(times[i])]
        new_t = append_chunk_timestamps(buf_t, ts_chunk, part.shape[0], fs, last_t=float(buf_t[-1]) if buf_t.size else None)
        buf = np.concatenate([buf, part], axis=0)
        buf_t = np.concatenate([buf_t, new_t])
        max_buf = int(n_samples * 3) + int(fs * 4)
        if buf.shape[0] > max_buf:
            drop = buf.shape[0] - max_buf
            buf = buf[drop:, :]
            buf_t = buf_t[drop:]

        while m_idx < len(markers) and markers[m_idx][0] <= float(buf_t[-1]):
            t_m, lamp, on = markers[m_idx]
            if misaligned_markers:
                t_m = float(buf_t[-1])
            gate.ingest_marker(t_m, f"{100 + lamp}|{'on' if on else 'off'}")
            m_idx += 1

        if buf.shape[0] < n_samples or buf_t.size < n_samples:
            continue
        t_end = float(buf_t[-1])
        if t_end < block.t_start or t_end > block.t_end:
            continue
        if last_classify_t is not None and (t_end - last_classify_t) < classify_interval_sec:
            continue

        allowed, reason, gate_detail = gate.evaluate_gate(buf_t, now=t_end)
        rec: Dict[str, Any] = {
            "t_end": t_end,
            "allowed": allowed,
            "gate_reason": reason,
            **gate_detail,
        }
        if not allowed:
            decisions.append(rec)
            last_classify_t = t_end
            continue

        if on_only:
            win_on, win_meta = build_on_only_msi_window(buf, buf_t, gate, n_samples)
            if not win_meta.get("sufficient_on_samples"):
                rec["note"] = "insufficient_on_samples"
                rec.update(win_meta)
                decisions.append(rec)
                last_classify_t = t_end
                continue
            win_full = win_on.T
            rec.update(win_meta)
        else:
            win_full = buf[-n_samples:, :].T
            rec["msi_window_mode"] = "contiguous_tail"
            off_frac = 1.0 - float(gate.on_sample_mask(buf_t[-n_samples:]).mean())
            rec["off_fraction_in_msi_window"] = round(off_frac, 4)

        ch_idx = [c for c in roi if 0 <= c < win_full.shape[0]]
        if not ch_idx:
            ch_idx = list(range(win_full.shape[0]))
        win = np.ascontiguousarray(win_full[ch_idx, :], dtype=np.float64)
        managed = tme.numpy_to_double_matrix2d(win, verbose=False)
        winner, winner_coef = tme.msi_exec_signal(msi, managed)
        coef = coef_values(msi, len(block.freqs_hz), winner_1based=int(winner), winner_coef=float(winner_coef))
        w0 = int(winner) - 1
        rec.update(
            {
                "winner_1based": int(winner),
                "winner_0idx": w0,
                "coefS": float(coef[w0]) if coef and 0 <= w0 < len(coef) and coef[w0] is not None else None,
                "correct": w0 == int(block.target_lamp_0idx),
            }
        )
        decisions.append(rec)
        last_classify_t = t_end

    return decisions


def _summarize(decisions: Sequence[Dict[str, Any]], *, target: int) -> Dict[str, Any]:
    allowed = [d for d in decisions if d.get("allowed")]
    classified = [d for d in allowed if d.get("winner_1based") is not None]
    denied = [d for d in decisions if not d.get("allowed")]
    coef_s = [float(d["coefS"]) for d in classified if d.get("coefS") is not None]
    winners = [int(d["winner_1based"]) for d in classified]
    switches = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i - 1])
    correct = sum(1 for d in classified if d.get("correct"))
    majority: Optional[int] = None
    if winners:
        from collections import Counter

        hist = Counter(winners)
        top = hist.most_common(1)[0]
        majority = int(top[0])
    return {
        "n_classify_attempts": len(decisions),
        "n_denied": len(denied),
        "n_classified": len(classified),
        "trace_accuracy": round(correct / len(classified), 4) if classified else None,
        "majority_winner_1based": majority,
        "majority_correct": majority == target + 1 if majority is not None else None,
        "switch_rate": round(switches / max(1, len(winners) - 1), 4) if len(winners) > 1 else 0.0,
        "coefS_mean": round(float(np.mean(coef_s)), 6) if coef_s else None,
        "coefS_median": round(float(np.median(coef_s)), 6) if coef_s else None,
    }


def evaluate_session(session_dir: Path, *, roi: Tuple[int, ...] = (8, 9, 20)) -> Dict[str, Any]:
    events = session_dir / "events.ndjson"
    npz_path = session_dir / "eeg.npz"
    blocks = _load_burst_blocks(events)
    markers = _load_led_markers(events)
    data = np.load(npz_path, mmap_mode="r")
    times = np.asarray(data["times"], dtype=np.float64)
    eeg = np.asarray(data["eeg"], dtype=np.float64)
    fs = _infer_fs(times)
    n_samples = int(round(3.0 * fs))

    per_block: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, Any]] = {}
    for label, on_only, gate_cfg, misalign in (
        ("legacy_raw_gate70", False, LEGACY_GATE, False),
        ("legacy_misaligned_markers", False, LEGACY_GATE, True),
        ("fixed_on_only_gate55", True, FIXED_GATE, False),
    ):
        summaries = []
        for bi, block in enumerate(blocks):
            decs = _classify_decisions(
                times=times,
                eeg=eeg,
                markers=markers,
                block=block,
                roi=roi,
                fs=fs,
                n_samples=n_samples,
                gate_cfg=gate_cfg,
                on_only=on_only,
                misaligned_markers=misalign,
            )
            summ = _summarize(decs, target=block.target_lamp_0idx)
            summaries.append(summ)
            per_block.append({"pipeline": label, "block_index": bi, "target": block.target_lamp_0idx, **summ})
        keys = ("n_denied", "n_classified", "trace_accuracy", "switch_rate", "coefS_mean")
        agg[label] = {
            k: round(float(np.mean([s[k] for s in summaries if s.get(k) is not None])), 4)
            if summaries and any(s.get(k) is not None for s in summaries)
            else None
            for k in keys
        }
        majority_hits = sum(1 for s in summaries if s.get("majority_correct"))
        agg[label]["majority_block_accuracy"] = round(majority_hits / len(summaries), 4) if summaries else None

    logged = _summarize_logged_trace(session_dir / "msi_trace.ndjson")
    return {
        "session": session_dir.name,
        "fs_hz": fs,
        "n_burst_blocks": len(blocks),
        "pipelines": agg,
        "production_logged_burst": logged,
        "per_block": per_block,
    }


def _summarize_logged_trace(path: Path) -> Dict[str, Any]:
    burst_rows = []
    for row in _read_ndjson(path):
        if str(row.get("ssvep_mode")) != "burst":
            continue
        if row.get("winner_1based") is None:
            continue
        burst_rows.append(row)
    if not burst_rows:
        return {"present": False}
    coef_s = [float(r["coefS"]) for r in burst_rows if r.get("coefS") is not None]
    winners = [int(r["winner_1based"]) for r in burst_rows]
    switches = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i - 1])
    correct = sum(
        1
        for r in burst_rows
        if r.get("classify_allowed")
        and r.get("target_lamp_0idx") is not None
        and int(r.get("winner_0idx", -9)) == int(r["target_lamp_0idx"])
    )
    allowed = sum(1 for r in burst_rows if r.get("classify_allowed"))
    return {
        "present": True,
        "n_trace": len(burst_rows),
        "trace_accuracy": round(correct / allowed, 4) if allowed else None,
        "switch_rate": round(switches / max(1, len(winners) - 1), 4),
        "coefS_mean": round(float(np.mean(coef_s)), 6) if coef_s else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", type=Path, nargs="+")
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args(argv)
    reports = [evaluate_session(d) for d in args.session_dir]
    out = {"reports": reports}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
