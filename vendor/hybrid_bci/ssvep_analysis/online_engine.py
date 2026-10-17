"""Headless SSVEP runtime — та же логика, что в ``ssvep_analyzer.py`` (_push_eeg_chunk + _on_classify)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ssvep_analysis.burst_gate import (
    BurstGate,
    BurstGateConfig,
    append_chunk_timestamps,
    build_on_only_msi_window,
)
from ssvep_analysis.timestamp_contract import timestamps_need_lsl_anchor
from ssvep_analysis.experiment_logger import coef_values

# MSI-хелперы — те же, что в ssvep_analyzer.py / scripts/test_msi_exec.py
from scripts import test_msi_exec as tme


@dataclass(frozen=True)
class SSVEPParams:
    fs_hz: float = 250.0
    window_sec: float = 2.0
    msi_n_samples: int = 0  # если > 0 — длина окна MSI в отсчётах (приоритет над window_sec)
    freqs_hz: Tuple[float, ...] = ()
    mode: str = "continuous"  # "continuous" | "burst"
    roi_channels_0idx: Tuple[int, ...] = ()


@dataclass(frozen=True)
class SSVEPDecision:
    winner_1based: Optional[int]
    winner_0idx: Optional[int]
    coef: Optional[list]
    mode: str
    classify_allowed: bool
    debug: Dict[str, Any]


class SSVEPOnlineEngine:
    """
    Буфер EEG + BurstGate + MSIExec — повторяет ``ssvep_analyzer.SSVEPAnalyzerWindow``:
    ``_push_eeg_chunk`` (append_chunk_timestamps) и ``_on_classify`` (окно + MSIExec + coef_values).
    """

    def __init__(self) -> None:
        self.params = SSVEPParams()
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=float(self.params.window_sec)))
        self._msi = None
        self._models = None
        self._n_samples = 0
        self._max_buf = 0
        self._buf: np.ndarray = np.zeros((0, 1), dtype=np.float64)
        self._buf_t: np.ndarray = np.zeros(0, dtype=np.float64)

    def reset(self, *, params: Optional[SSVEPParams] = None) -> None:
        if params is not None:
            self.params = params
        fs = float(self.params.fs_hz)
        win_sec = float(self.params.window_sec)
        if int(self.params.msi_n_samples) > 0:
            self._n_samples = int(self.params.msi_n_samples)
            gate_sec = float(self._n_samples) / max(1.0, fs)
        else:
            gate_sec = win_sec
            self._n_samples = max(8, int(round(fs * gate_sec)))
        self._recompute_max_buf()
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=float(gate_sec)))
        n_lamps = len(self.params.freqs_hz)
        if n_lamps > 0:
            self._burst_gate.set_active_lamps(n_lamps)
        self._buf = np.zeros((0, max(1, 1)), dtype=np.float64)
        self._buf_t = np.zeros(0, dtype=np.float64)
        if self._msi is not None:
            self._apply_templates()

    def _recompute_max_buf(self) -> None:
        fs = max(1.0, float(self.params.fs_hz))
        need = int(self._n_samples) + max(8, int(fs * 2))
        if str(self.params.mode) == "burst":
            # Для ON-only окна нужен больший lookback при duty cycle < 100%.
            need = max(need, int(self._n_samples * 2) + int(fs * 4))
        cap = max(need, int(fs * 120))
        self._max_buf = max(need, min(need * 3, cap))

    def latest_buffer_lsl_time(self) -> Optional[float]:
        if self._buf_t.size == 0:
            return None
        return float(self._buf_t[-1])

    def set_burst_block_start(self, t: Optional[float] = None) -> None:
        """Grace period burst gate от времени EEG-буфера."""
        if t is None:
            t = self.latest_buffer_lsl_time()
        if t is not None:
            self._burst_gate.set_block_start(float(t))

    def ensure_msi_ready(self) -> None:
        if self._msi is not None:
            return
        msi, _msi_res, _dotnet_root = tme.load_msi_runtime()
        self._msi = msi
        self._apply_templates()

    def _apply_templates(self) -> None:
        """Как ssvep_analyzer._apply_model_signals_to_msi."""
        assert self._msi is not None
        fs = float(self.params.fs_hz)
        win_sec = float(self.params.window_sec)
        freqs = [float(x) for x in self.params.freqs_hz]
        np_models = tme.generate_model_signals(freqs, fs, win_sec)
        self._n_samples = (
            int(np_models[0].shape[1]) if np_models else max(8, int(round(fs * win_sec)))
        )
        if int(self.params.msi_n_samples) > 0:
            self._n_samples = int(self.params.msi_n_samples)
        self._recompute_max_buf()
        model_list = tme.build_model_signal_list(self._msi, np_models, verbose=False)
        self._msi.ModelSignal = model_list
        self._models = np_models
        gate_sec = float(self._n_samples) / max(1.0, fs)
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=gate_sec))
        if freqs:
            self._burst_gate.set_active_lamps(len(freqs))

    def ingest_eeg_chunk(
        self,
        *,
        times: Sequence[float],
        samples: np.ndarray,
        lsl_clock_now: Optional[float] = None,
    ) -> None:
        """Как ssvep_analyzer._push_eeg_chunk: append_chunk_timestamps + кольцевой срез."""
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        n_new = int(arr.shape[0])
        fs = float(self.params.fs_hz)

        ts_in = list(float(t) for t in times)
        if timestamps_need_lsl_anchor(ts_in) and lsl_clock_now is not None:
            ts_in = [float(lsl_clock_now)]

        last_t = float(self._buf_t[-1]) if self._buf_t.size else None
        new_t = append_chunk_timestamps(
            np.zeros(0, dtype=np.float64),
            ts_in,
            n_new,
            fs,
            last_t=last_t,
        )
        if self._buf_t.size and new_t.size:
            self._buf_t = np.concatenate([self._buf_t, new_t])
            self._buf = np.concatenate([self._buf, arr[:n_new, :]], axis=0)
        else:
            self._buf_t = new_t
            self._buf = np.ascontiguousarray(arr[:n_new, :], dtype=np.float64)

        if self._buf.shape[0] > self._max_buf:
            drop = int(self._buf.shape[0]) - int(self._max_buf)
            self._buf = self._buf[drop:, :]
            self._buf_t = self._buf_t[drop:]

    def ingest_migalka_marker(self, *, lsl_time: float, value: object) -> None:
        """Как ssvep_analyzer: burst_gate.ingest_marker(lsl_time, val)."""
        self._burst_gate.ingest_marker(float(lsl_time), value)

    def can_classify(self) -> bool:
        if int(self._buf.shape[0]) < int(self._n_samples):
            return False
        if str(self.params.mode) != "burst":
            return True
        if self._buf_t.size == 0:
            return False
        allowed, _reason = self._burst_gate.classify_allowed(self._buf_t)
        return bool(allowed)

    def classify(self) -> SSVEPDecision:
        """Как ssvep_analyzer._on_classify (без Qt)."""
        self.ensure_msi_ready()
        assert self._msi is not None

        if int(self._buf.shape[0]) < int(self._n_samples):
            return SSVEPDecision(
                winner_1based=None,
                winner_0idx=None,
                coef=None,
                mode=str(self.params.mode),
                classify_allowed=False,
                debug={
                    "note": "buffer_not_ready",
                    "buf_len": int(self._buf.shape[0]),
                    "n_samples": int(self._n_samples),
                },
            )

        gate_ok = True
        gate_reason = ""
        gate_detail: Dict[str, Any] = {"allowed": True, "gate_reason": ""}
        if str(self.params.mode) == "burst":
            allowed, gate_reason, gate_detail = self._burst_gate.evaluate_gate(self._buf_t)
            gate_ok = bool(allowed)
            gate_detail = {
                **gate_detail,
                "allowed": gate_ok,
                "gate_reason": gate_reason,
            }
            if not gate_ok:
                return SSVEPDecision(
                    winner_1based=None,
                    winner_0idx=None,
                    coef=None,
                    mode=str(self.params.mode),
                    classify_allowed=False,
                    debug={
                        "note": "burst_gate_closed",
                        "buf_len": int(self._buf.shape[0]),
                        "n_samples": int(self._n_samples),
                        **gate_detail,
                    },
                )

        roi = [int(c) for c in self.params.roi_channels_0idx]
        n_ch = int(self._buf.shape[1])
        ch_idx = [c for c in roi if 0 <= c < n_ch] if roi else list(range(n_ch))
        if not ch_idx:
            return SSVEPDecision(
                winner_1based=None,
                winner_0idx=None,
                coef=None,
                mode=str(self.params.mode),
                classify_allowed=False,
                debug={"note": "no_valid_roi_channels", "roi": list(roi), "n_ch": n_ch},
            )

        win_meta: Dict[str, Any] = {"msi_window_mode": "contiguous_tail"}
        if str(self.params.mode) == "burst":
            win_on, win_meta = build_on_only_msi_window(
                self._buf,
                self._buf_t,
                self._burst_gate,
                int(self._n_samples),
            )
            if not win_meta.get("sufficient_on_samples"):
                return SSVEPDecision(
                    winner_1based=None,
                    winner_0idx=None,
                    coef=None,
                    mode=str(self.params.mode),
                    classify_allowed=False,
                    debug={
                        "note": "insufficient_on_samples_for_msi",
                        "buf_len": int(self._buf.shape[0]),
                        "n_samples": int(self._n_samples),
                        **gate_detail,
                        **win_meta,
                    },
                )
            win_full = win_on.T.copy()
        else:
            win_full = self._buf[-int(self._n_samples) :].T.copy()
        win = np.ascontiguousarray(win_full[ch_idx, :], dtype=np.float64)
        sig_managed = tme.numpy_to_double_matrix2d(win, verbose=False)
        t0 = time.perf_counter()
        winner, winner_coef = tme.msi_exec_signal(self._msi, sig_managed)
        dt_ms = float((time.perf_counter() - t0) * 1000.0)
        coef = coef_values(
            self._msi,
            len(self.params.freqs_hz),
            winner_1based=int(winner),
            winner_coef=float(winner_coef),
        )
        w0 = winner - 1 if winner > 0 else None
        return SSVEPDecision(
            winner_1based=winner,
            winner_0idx=w0,
            coef=coef,
            mode=str(self.params.mode),
            classify_allowed=True,
            debug={
                "msi_exec_ms": dt_ms,
                "buf_shape": [int(win.shape[0]), int(win.shape[1])],
                "gate_reason": gate_detail.get("gate_reason", gate_reason),
                "on_fraction": gate_detail.get("on_fraction"),
                "allowed": gate_detail.get("allowed", gate_ok),
                **win_meta,
                **{k: v for k, v in gate_detail.items() if k not in ("gate_reason",)},
            },
        )
