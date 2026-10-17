"""P300 online engine synthetic validation (no ProtocolRunner)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from p300_analysis.analysis_profiles import hybrid_protocol_p300_profile
from p300_analysis.online_engine import P300EngineParams, P300OnlineEngine
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR
from validation.synthetic_streams import erp_like_epochs


def run_p300_synthetic_battery(*, seed: int = 0) -> Dict[str, Any]:
    profile = hybrid_protocol_p300_profile()
    params = P300EngineParams(
        fs_hz=250.0,
        roi_channels_0idx=tuple(profile.roi_channels_0idx),
    )
    engine = P300OnlineEngine()
    engine.reset(params=params)

    target_ch = int(profile.roi_channels_0idx[0])
    engine.ingest_eeg_chunk(
        eeg_chunk=np.zeros((int(250 * 0.8), 21), dtype=np.float64),
        eeg_ts=[float(i) / 250.0 for i in range(int(250 * 0.8))],
        lsl_local_clock_now=1.0,
    )
    n_samples = int(engine._epoch_geom.epoch_len or int(250 * 0.8))
    cases: List[Dict[str, Any]] = []

    for trial, target_tile in enumerate((4, 6)):
        engine.epochs_data = {}
        key_tgt = f"стимул_{int(target_tile)}"
        key_distr = f"стимул_{(int(target_tile) + 1) % 9}"
        for rep in range(12):
            engine.epochs_data.setdefault(key_tgt, []).append(
                erp_like_epochs(
                    n_channels=21,
                    n_samples=n_samples,
                    target_ch=target_ch,
                    amp_uv=50.0 if rep % 2 == 0 else 35.0,
                    seed=seed + trial * 100 + rep,
                )
            )
            engine.epochs_data.setdefault(key_distr, []).append(
                erp_like_epochs(
                    n_channels=21,
                    n_samples=n_samples,
                    target_ch=target_ch,
                    amp_uv=8.0,
                    seed=seed + trial * 100 + rep + 50,
                )
            )
        engine.current_cue_target_id = int(target_tile)

        dec_auc = engine.compute_decision(winner_mode=WINNER_MODE_AUC)
        dec_tpl = engine.compute_decision(winner_mode=WINNER_MODE_TEMPLATE_CORR)
        cases.append(
            {
                "target_tile": int(target_tile),
                "auc_can_decide": dec_auc.can_decide,
                "auc_winner": dec_auc.winner_key,
                "auc_mode": dec_auc.mode_used,
                "template_can_decide": dec_tpl.can_decide,
                "template_winner": dec_tpl.winner_key,
                "template_mode": dec_tpl.mode_used,
                "margin_auc": (dec_auc.debug or {}).get("margin"),
                "margin_template": (dec_tpl.debug or {}).get("margin"),
            }
        )

    stable = sum(1 for c in cases if c["auc_can_decide"] and c["template_can_decide"])
    return {
        "schema": "hybrid_bci_p300_synthetic_validation/v1",
        "cases": cases,
        "summary": {
            "trials": len(cases),
            "both_decide": stable,
            "stable_rate": round(stable / len(cases), 4) if cases else 0.0,
        },
    }
