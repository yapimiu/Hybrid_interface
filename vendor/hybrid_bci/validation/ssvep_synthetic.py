"""SSVEP synthetic validation (MSI battery + online engine smoke)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from ssvep_analysis.lamp_frequencies import DEFAULT_LAMP_FREQS_HZ, SSVEP_PROTOCOL_ROI_0IDX
from ssvep_analysis.msi_validation import run_synthetic_battery
from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams
from validation.synthetic_streams import SyntheticEEGInlet


def run_ssvep_msi_battery() -> Dict[str, Any]:
    return run_synthetic_battery()


def run_ssvep_online_smoke(*, target_hz: float = 10.0, seed: int = 0) -> Dict[str, Any]:
    freqs = list(DEFAULT_LAMP_FREQS_HZ[:4])
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
    target_0idx = min(
        range(len(freqs)),
        key=lambda i: abs(float(freqs[i]) - float(target_hz)),
    )

    for _ in range(1200):
        chunk, ts = inlet.pull_chunk(max_samples=32)
        if not ts:
            break
        arr = np.asarray(chunk, dtype=np.float64)
        engine.ingest_eeg_chunk(times=ts, samples=arr, lsl_clock_now=float(ts[-1]))

    allowed = engine.can_classify()
    decision = engine.classify()
    coef = decision.coef
    winner = decision.winner_0idx
    return {
        "target_hz": target_hz,
        "target_0idx": target_0idx,
        "classify_allowed": bool(allowed and decision.classify_allowed),
        "winner_0idx": winner,
        "winner_match": winner == target_0idx if winner is not None else False,
        "coef_len": len(coef) if isinstance(coef, list) else 0,
    }
