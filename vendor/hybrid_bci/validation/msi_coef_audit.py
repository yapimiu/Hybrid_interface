"""Focused MSI/coef pipeline audit (validation-only, no runtime changes)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol.replay import _coef_shape, discover_sessions, summarize_session
from ssvep_analysis.experiment_logger import coef_values
from ssvep_analysis.lamp_frequencies import DEFAULT_LAMP_FREQS_HZ, SSVEP_PROTOCOL_ROI_0IDX
from ssvep_analysis.msi_validation import run_synthetic_battery


def _msi_runtime():
    import test_msi_exec as tme

    msi, msi_res, dotnet_root = tme.load_msi_runtime()
    return msi, tme, msi_res, dotnet_root


def probe_msi_dll() -> Dict[str, Any]:
    """Reflection: MSIExec overloads and Coef property contract."""
    try:
        msi, tme, msi_res, dotnet_root = _msi_runtime()
    except Exception as e:
        return {"available": False, "error": str(e)}

    msi_type = msi.GetType()
    exec_methods: List[Dict[str, Any]] = []
    for meth in msi_type.GetMethods():
        if meth.Name != "MSIExec":
            continue
        ps = meth.GetParameters()
        exec_methods.append(
            {
                "signature": str(meth),
                "n_params": len(ps),
                "param_types": [str(p.ParameterType) for p in ps],
                "has_out_coef": len(ps) == 2 and bool(ps[1].IsOut),
            }
        )
    coef_prop = msi_type.GetProperty("Coef")
    coef_prop_info: Dict[str, Any] = {}
    if coef_prop is not None:
        coef_prop_info = {
            "property_type": str(coef_prop.PropertyType),
            "can_read": bool(coef_prop.CanRead),
        }
    return {
        "available": True,
        "msi_res": str(msi_res),
        "dotnet_root": str(dotnet_root),
        "msi_exec_overloads": exec_methods,
        "coef_property": coef_prop_info,
        "contract": {
            "production_path": "MSIExec(double[,] Signal, out double coefS) via scripts/test_msi_exec.msi_exec_signal",
            "coef_vector": "ssvep_analysis.experiment_logger.coef_values(winner_1based, winner_coef)",
            "msi.Coef_scalar": "NOT used for sparse vector (avoids fake [0.0, null, ...])",
        },
    }


def audit_coef_values_contract() -> Dict[str, Any]:
    """Unit-level coef_values behavior without DLL."""
    class ScalarCoef:
        Coef = 0.0

    class ArrCoef:
        class _C:
            Length = 3

            def __getitem__(self, i: int) -> float:
                return [0.1, 0.5, 0.9][i]

        Coef = _C()

    n = 4
    return {
        "scalar_only": coef_values(ScalarCoef(), n),
        "scalar_with_winner": coef_values(
            ScalarCoef(), n, winner_1based=2, winner_coef=0.77
        ),
        "dotnet_array_legacy": coef_values(ArrCoef(), n),
        "expected_sparse": [None, 0.77, None, None],
    }


def run_msi_raw_synthetic_cases(
    *,
    freqs: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    freqs = list(freqs if freqs is not None else DEFAULT_LAMP_FREQS_HZ[:4])
    """Direct DLL path: raw winner, coefS, Coef property, logged vector."""
    try:
        msi, tme, _, _ = _msi_runtime()
    except Exception as e:
        return {"available": False, "error": str(e)}

    fs, dur = 250.0, 2.0
    cases: List[Dict[str, Any]] = []
    msi.ModelSignal = tme.build_model_signal_list(
        msi, tme.generate_model_signals(freqs, fs, dur), verbose=False
    )

    for target_hz in freqs:
        eeg = tme.generate_sine_signal(float(target_hz), fs, dur, len(SSVEP_PROTOCOL_ROI_0IDX))
        sig = tme.numpy_to_double_matrix2d(np.ascontiguousarray(eeg, dtype=np.float64), verbose=False)
        winner, coef_s = tme.msi_exec_signal(msi, sig)
        try:
            coef_prop = float(msi.Coef)
        except Exception:
            coef_prop = None
        vec_correct = coef_values(
            msi, len(freqs), winner_1based=int(winner), winner_coef=float(coef_s)
        )
        vec_wrong = coef_values(msi, len(freqs))
        exp_idx = freqs.index(float(target_hz)) if float(target_hz) in freqs else -1
        cases.append(
            {
                "target_hz": float(target_hz),
                "expected_winner_1based": int(exp_idx) + 1 if exp_idx >= 0 else None,
                "msi_winner_1based": int(winner),
                "coefS_out_param": float(coef_s),
                "msi_coef_property_scalar": coef_prop,
                "logged_coef_sparse": vec_correct,
                "coef_if_ignored_winner_args": vec_wrong,
                "winner_match": int(winner) == int(exp_idx) + 1,
                "coefS_positive": float(coef_s) > 1e-6,
                "fake_zero_vector": vec_wrong == [0.0] + [None] * (len(freqs) - 1)
                if len(freqs) > 0
                else False,
            }
        )
    return {
        "available": True,
        "freqs_hz": [float(f) for f in freqs],
        "cases": cases,
        "battery": run_synthetic_battery(freqs=(8.0, 10.0, 12.0, 15.0)),
    }


def trace_online_engine_path(
    *,
    target_hz: float = 10.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Same code path as SSVEPOnlineEngine.classify (without ProtocolRunner)."""
    from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams
    from validation.synthetic_streams import SyntheticEEGInlet

    freqs = list(DEFAULT_LAMP_FREQS_HZ[:4])
    if target_hz not in freqs:
        target_hz = float(freqs[1])
    inlet = SyntheticEEGInlet(
        fs_hz=250.0,
        n_channels=21,
        seed=seed,
        ssvep_hz=target_hz,
        ssvep_roi_0idx=SSVEP_PROTOCOL_ROI_0IDX,
        ssvep_amp_uv=40.0,
    )
    engine = SSVEPOnlineEngine()
    engine.reset(
        params=SSVEPParams(
            fs_hz=250.0,
            freqs_hz=tuple(float(f) for f in freqs),
            roi_channels_0idx=SSVEP_PROTOCOL_ROI_0IDX,
            mode="continuous",
        )
    )
    for _ in range(1200):
        chunk, ts = inlet.pull_chunk(max_samples=32)
        if not ts:
            break
        engine.ingest_eeg_chunk(times=ts, samples=np.asarray(chunk, dtype=np.float64), lsl_clock_now=float(ts[-1]))
    dec = engine.classify()
    msi = engine._msi
    raw: Dict[str, Any] = {}
    if msi is not None and dec.classify_allowed:
        try:
            raw["msi_coef_property_after_classify"] = float(msi.Coef)
        except Exception:
            raw["msi_coef_property_after_classify"] = None
    return {
        "target_hz": target_hz,
        "classify_allowed": dec.classify_allowed,
        "winner_1based": dec.winner_1based,
        "winner_0idx": dec.winner_0idx,
        "logged_coef": dec.coef,
        "debug": dec.debug,
        "raw_msi": raw,
        "expected_winner_1based": freqs.index(target_hz) + 1 if target_hz in freqs else None,
    }


def compare_msi_exec_overloads(
    *,
    target_hz: float = 10.0,
    freqs: Sequence[float] = (8.0, 10.0, 12.0, 15.0),
) -> Dict[str, Any]:
    """Single-arg MSIExec vs out-param path on the same signal."""
    try:
        msi, tme, _, _ = _msi_runtime()
    except Exception as e:
        return {"available": False, "error": str(e)}

    fs, dur = 250.0, 2.0
    msi.ModelSignal = tme.build_model_signal_list(
        msi, tme.generate_model_signals(freqs, fs, dur), verbose=False
    )
    eeg = tme.generate_sine_signal(float(target_hz), fs, dur, 2)
    sig = tme.numpy_to_double_matrix2d(np.ascontiguousarray(eeg, dtype=np.float64), verbose=False)
    winner_single = int(msi.MSIExec(sig))
    try:
        coef_prop_after_single = float(msi.Coef)
    except Exception:
        coef_prop_after_single = None
    winner_out, coef_s = tme.msi_exec_signal(msi, sig)
    vec_single_path = coef_values(msi, len(freqs))
    vec_out_path = coef_values(
        msi, len(freqs), winner_1based=int(winner_out), winner_coef=float(coef_s)
    )
    return {
        "available": True,
        "target_hz": float(target_hz),
        "MSIExec_signal_only": {
            "winner_1based": winner_single,
            "msi_coef_property": coef_prop_after_single,
            "coef_values_without_winner": vec_single_path,
        },
        "MSIExec_signal_out_coefS": {
            "winner_1based": int(winner_out),
            "coefS": float(coef_s),
            "logged_sparse": vec_out_path,
        },
        "contract_ok": vec_out_path != [0.0] + [None] * (len(freqs) - 1)
        and (vec_out_path[int(winner_out) - 1] if winner_out > 0 else None) == float(coef_s),
    }


def run_runtime_ssvep_ndjson_trace(tmp_path: Path) -> Dict[str, Any]:
    """ProtocolRunner + synthetic LSL; MSI trace hook + experiments.ndjson."""
    from unittest.mock import MagicMock

    from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner, ProtocolState
    from validation.runtime_harness import _fake_stream_info
    from validation.synthetic_streams import SyntheticEEGInlet, SyntheticMarkerInlet

    import runtime_core.adapters.lsl as lsl_mod
    import scripts.test_msi_exec as tme_mod
    import ssvep_analysis.migalka_serial_controller as msc

    msi_traces: List[Dict[str, Any]] = []
    _orig_msi_exec = tme_mod.msi_exec_signal

    def _traced_msi_exec(msi: Any, sig_managed: Any) -> tuple[int, float]:
        winner, coef_s = _orig_msi_exec(msi, sig_managed)
        try:
            coef_prop = float(msi.Coef)
        except Exception:
            coef_prop = None
        msi_traces.append(
            {
                "winner_1based": int(winner),
                "coefS_out_param": float(coef_s),
                "msi_coef_property_scalar": coef_prop,
                "coef_property_maps_to_index0": coef_prop == 0.0,
            }
        )
        return winner, coef_s

    tme_mod.msi_exec_signal = _traced_msi_exec  # type: ignore[assignment]

    session_dir = tmp_path / "sess_msi_runtime"
    session_dir.mkdir(parents=True, exist_ok=True)
    ssvep_hz = 10.0
    eeg = SyntheticEEGInlet(
        fs_hz=250.0,
        n_channels=21,
        seed=7,
        ssvep_hz=ssvep_hz,
        ssvep_amp_uv=40.0,
    )
    mk = SyntheticMarkerInlet([])
    mig_mk = SyntheticMarkerInlet([])
    eeg_info = _fake_stream_info("SynthEEG", "EEG")
    mk_info = _fake_stream_info("BCI_StimMarkers", "Markers")
    inlet_queue = [eeg, mk, mig_mk]

    def _stream_inlet(info: Any, buffer_seconds: float = 20) -> Any:
        if inlet_queue:
            return inlet_queue.pop(0)
        return MagicMock(pull_chunk=MagicMock(return_value=([], [])))

    orig_lsl = {
        "discover": lsl_mod.discover_eeg_streams,
        "wait_mk": lsl_mod.wait_for_stimulus_marker_stream,
        "inlet": lsl_mod.stream_inlet_with_buffer,
        "select_eeg": lsl_mod.select_eeg_stream,
        "select_mig": lsl_mod.select_migalka_marker_stream,
        "labels": lsl_mod.stream_channel_labels,
    }
    lsl_mod.discover_eeg_streams = lambda timeout=1.5: [eeg_info]  # type: ignore
    lsl_mod.wait_for_stimulus_marker_stream = lambda max_wait_sec=3.0: (mk_info, [mk_info])  # type: ignore
    lsl_mod.stream_inlet_with_buffer = _stream_inlet  # type: ignore
    lsl_mod.select_eeg_stream = lambda streams, name="", session_id="": eeg_info  # type: ignore
    lsl_mod.select_migalka_marker_stream = lambda streams: None  # type: ignore
    lsl_mod.stream_channel_labels = lambda info, n: [f"Ch{i+1}" for i in range(n)]  # type: ignore

    ser = MagicMock()
    ser.is_open = True
    orig_serial = msc.serial.Serial
    msc.serial.Serial = lambda *a, **k: ser  # type: ignore

    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="msi_audit",
        session_dir=session_dir,
        com_port="COM_SYNTH",
        eeg_stream_name="SynthEEG",
        p300_calib_trials=0,
        p300_main_trials=0,
        ssvep_blocks_per_mode=1,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=4.0,
        shuffle_seed=0,
        ssvep_freqs_hz=(8.0, 10.0, 12.0, 15.0),
    )
    runner = ProtocolRunner(cfg)
    harness_out: Dict[str, Any] = {}
    try:
        runner.start()
        for _ in range(500):
            runner.tick()
            if runner.state == ProtocolState.Ready:
                break
        if not runner.start_ssvep_block(mode="continuous"):
            raise RuntimeError(f"start_ssvep_block failed: state={runner.state!r}")
        ticks_used = 0
        deadline = time.time() + 45.0
        while time.time() < deadline and runner.state != ProtocolState.Stopped:
            runner.tick()
            ticks_used += 1
        if runner.state != ProtocolState.Stopped:
            runner.stop(reason="msi_coef_audit")
            for _ in range(200):
                runner.tick()
                if runner.state == ProtocolState.Stopped:
                    break
        harness_out = {"final_state": str(runner.state), "ticks_used": ticks_used}
    finally:
        tme_mod.msi_exec_signal = _orig_msi_exec  # type: ignore[assignment]
        lsl_mod.discover_eeg_streams = orig_lsl["discover"]  # type: ignore
        lsl_mod.wait_for_stimulus_marker_stream = orig_lsl["wait_mk"]  # type: ignore
        lsl_mod.stream_inlet_with_buffer = orig_lsl["inlet"]  # type: ignore
        lsl_mod.select_eeg_stream = orig_lsl["select_eeg"]  # type: ignore
        lsl_mod.select_migalka_marker_stream = orig_lsl["select_mig"]  # type: ignore
        lsl_mod.stream_channel_labels = orig_lsl["labels"]  # type: ignore
        msc.serial.Serial = orig_serial  # type: ignore

    out = harness_out
    session_dirs = list(Path(tmp_path).glob("sess*"))
    session_dir = session_dirs[0] if session_dirs else None
    experiments: List[Dict[str, Any]] = []
    block_ends: List[Dict[str, Any]] = []
    if session_dir is not None:
        exp_path = session_dir / "experiments.ndjson"
        if exp_path.is_file():
            for line in exp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    experiments.append(json.loads(line))
        ev_path = session_dir / "events.ndjson"
        if ev_path.is_file():
            for line in ev_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event") == "ssvep_block_end":
                    block_ends.append(rec.get("data") or rec)
    ssvep_exps = [e for e in experiments if e.get("kind") == "ssvep"]
    runtime_issues: List[Dict[str, Any]] = []
    for exp in ssvep_exps:
        coef = exp.get("coef")
        shape = _coef_shape(coef)
        w0 = exp.get("winner_0idx")
        if shape.get("all_zero_or_null") and exp.get("classify_allowed"):
            runtime_issues.append(
                {
                    "issue": "classify_allowed_but_degenerate_coef",
                    "winner_0idx": w0,
                    "coef": coef,
                    "correct": exp.get("correct"),
                }
            )
    final_trace = msi_traces[-1] if msi_traces else None
    return {
        "harness": out,
        "session_dir": str(session_dir) if session_dir else None,
        "ssvep_target_hz": ssvep_hz,
        "msi_call_traces": msi_traces,
        "msi_call_count": len(msi_traces),
        "final_msi_trace": final_trace,
        "ssvep_experiments": ssvep_exps,
        "ssvep_block_end_events": block_ends,
        "runtime_suspicious": runtime_issues,
        "runtime_contract_ok": len(runtime_issues) == 0 and any(
            isinstance(e.get("coef"), list)
            and any(c is not None and float(c) > 1e-6 for c in e.get("coef") or [])
            for e in ssvep_exps
        ),
    }


def audit_historical_sessions(root: Path) -> Dict[str, Any]:
    """Scan logged SSVEP coef for suspicious patterns."""
    root = Path(root)
    suspicious: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for session_dir in discover_sessions(root):
        s = summarize_session(session_dir)
        summaries.append(
            {
                "session": s.get("session_dir_name"),
                "ssvep_blocks": (s.get("counts") or {}).get("ssvep_blocks"),
                "ssvep_bad_coef": (s.get("counts") or {}).get("ssvep_bad_coef"),
            }
        )
        for row in (s.get("outputs") or {}).get("ssvep") or []:
            coef = row.get("coef")
            shape = row.get("coef_shape") or _coef_shape(coef)
            w0 = row.get("winner_0idx")
            correct = row.get("correct")
            issues: List[str] = []
            if shape.get("all_zero_or_null"):
                issues.append("degenerate_coef")
            if isinstance(coef, list) and len(coef) == 1 and coef[0] in (0, 0.0):
                issues.append("length1_zero_scalar")
            if (
                w0 is not None
                and isinstance(coef, list)
                and len(coef) > int(w0)
                and coef[int(w0)] is None
                and any(c is not None and c != 0 for c in coef)
            ):
                issues.append("winner_index_null_but_other_nonnull")
            if (
                w0 is not None
                and isinstance(coef, list)
                and len(coef) > 0
                and coef[0] in (0, 0.0)
                and int(w0) != 0
                and correct is True
            ):
                issues.append("correct_winner_but_leading_zero")
            if issues:
                suspicious.append(
                    {
                        "session": s.get("session_dir_name"),
                        "index": row.get("index"),
                        "winner_0idx": w0,
                        "correct": correct,
                        "coef": coef,
                        "coef_shape": shape,
                        "issues": issues,
                    }
                )
    return {
        "sessions_scanned": len(summaries),
        "suspicious_cases": suspicious,
        "session_summaries": summaries,
    }


def build_full_audit_report(
    *,
    experiment_runs_root: Path,
    tmp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    import tempfile

    tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="msi_coef_audit_"))
    report: Dict[str, Any] = {
        "schema": "hybrid_bci_msi_coef_audit/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dll_probe": probe_msi_dll(),
        "msi_exec_overload_compare": compare_msi_exec_overloads(),
        "coef_values_contract": audit_coef_values_contract(),
        "msi_raw_synthetic": run_msi_raw_synthetic_cases(),
        "online_engine_trace": trace_online_engine_path(),
        "historical": audit_historical_sessions(experiment_runs_root),
    }
    try:
        report["runtime_ndjson"] = run_runtime_ssvep_ndjson_trace(tmp)
    except Exception as e:
        report["runtime_ndjson"] = {"error": str(e)}
    report["verdict"] = summarize_verdict(report)
    return report


def summarize_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """High-level pass/fail for current runtime MSI contract."""
    cmp_ = report.get("msi_exec_overload_compare") or {}
    rt = report.get("runtime_ndjson") or {}
    hist = report.get("historical") or {}
    raw = report.get("msi_raw_synthetic") or {}
    online = report.get("online_engine_trace") or {}

    current_runtime_ok = bool(
        cmp_.get("contract_ok")
        and rt.get("runtime_contract_ok")
        and not rt.get("runtime_suspicious")
    )
    historical_degenerate = len(hist.get("suspicious_cases") or [])
    return {
        "current_runtime_path_ok": current_runtime_ok,
        "historical_suspicious_rows": historical_degenerate,
        "note": (
            "Historical logs may show legacy [0.0] length-1 coef from pre-sparse contract; "
            "current ProtocolRunner + SSVEPOnlineEngine use MSIExec(out coefS) + sparse coef_values."
        ),
        "online_engine_classify_allowed": online.get("classify_allowed"),
        "online_logged_coef": online.get("logged_coef"),
        "raw_synthetic_all_positive_coefS": all(
            c.get("coefS_positive") for c in (raw.get("cases") or [])
        )
        if raw.get("available")
        else None,
    }


def render_audit_markdown(report: Dict[str, Any]) -> str:
    verdict = report.get("verdict") or summarize_verdict(report)
    lines = [
        "# MSI / coef pipeline audit",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- **Current runtime path OK:** {verdict.get('current_runtime_path_ok')}",
        f"- Historical suspicious rows: {verdict.get('historical_suspicious_rows')}",
        "",
        verdict.get("note", ""),
        "",
        "## DLL contract",
        "",
    ]
    dll = report.get("dll_probe") or {}
    if not dll.get("available"):
        lines.append(f"MSI unavailable: `{dll.get('error', '?')}`")
    else:
        for m in dll.get("msi_exec_overloads") or []:
            lines.append(f"- `{m.get('signature')}` out_coef={m.get('has_out_coef')}")
        lines.append(f"- Coef property: `{dll.get('coef_property')}`")
        for k, v in (dll.get("contract") or {}).items():
            lines.append(f"- {k}: {v}")
    cmp_ = report.get("msi_exec_overload_compare") or {}
    lines.extend(["", "## MSIExec overload compare", ""])
    if cmp_.get("available"):
        lines.append(f"- Single-arg winner: {cmp_.get('MSIExec_signal_only')}")
        lines.append(f"- Out-param path: {cmp_.get('MSIExec_signal_out_coefS')}")
        lines.append(f"- contract_ok: {cmp_.get('contract_ok')}")
    lines.extend(["", "## Historical suspicious cases", ""])
    hist = report.get("historical") or {}
    sus = hist.get("suspicious_cases") or []
    if not sus:
        lines.append("No suspicious historical patterns in replay scan.")
    else:
        lines.append(f"Found **{len(sus)}** suspicious SSVEP rows (legacy logging; see JSON).")
        for row in sus[:15]:
            lines.append(
                f"- `{row.get('session')}` idx={row.get('index')} "
                f"issues={row.get('issues')} winner_0idx={row.get('winner_0idx')} coef={row.get('coef')}"
            )
    lines.extend(["", "## Runtime NDJSON (synthetic ProtocolRunner)", ""])
    rt = report.get("runtime_ndjson") or {}
    if rt.get("error"):
        lines.append(f"Runtime trace error: `{rt['error']}`")
    else:
        lines.append(f"- MSI calls traced: {rt.get('msi_call_count')}")
        lines.append(f"- runtime_contract_ok: {rt.get('runtime_contract_ok')}")
        for exp in (rt.get("ssvep_experiments") or [])[:5]:
            lines.append(
                f"- exp coef={exp.get('coef')} winner_1based={exp.get('winner_1based')} "
                f"correct={exp.get('correct')} allowed={exp.get('classify_allowed')}"
            )
        for be in (rt.get("ssvep_block_end_events") or [])[:3]:
            lines.append(f"- ssvep_block_end coef={be.get('coef')} winner={be.get('winner_1based')}")
    lines.extend(["", "## Online engine (synthetic stream)", ""])
    on = report.get("online_engine_trace") or {}
    lines.append(
        f"- winner_1based={on.get('winner_1based')} coef={on.get('logged_coef')} "
        f"allowed={on.get('classify_allowed')}"
    )
    return "\n".join(lines) + "\n"
