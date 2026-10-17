"""Synthetic MSI validation harness (Stage 2.2)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import test_msi_exec as tme  # noqa: E402
from ssvep_analysis.experiment_logger import coef_values  # noqa: E402


@dataclass
class SyntheticCaseResult:
    name: str
    freqs_hz: List[float]
    target_hz: float
    expected_winner_1based: int
    actual_winner_1based: int
    winner_match: bool
    coef_s: float
    coef_vector: List[Optional[float]]
    channels: int
    noise_std: float
    harmonic_weight: float
    repeat_std_winner: float
    repeat_std_coef: float


def _expected_index(freqs: Sequence[float], target_hz: float) -> int:
    for i, f in enumerate(freqs):
        if abs(float(f) - float(target_hz)) < 1e-6:
            return i + 1
    raise ValueError(f"target {target_hz} not in {freqs}")


def _run_once(
    msi: Any,
    *,
    freqs: Sequence[float],
    fs: float,
    duration: float,
    target_hz: float,
    channels: int,
    noise_std: float,
    harmonic_weight: float,
) -> tuple[int, float, List[Optional[float]]]:
    models = tme.generate_model_signals(freqs, fs, duration)
    msi.ModelSignal = tme.build_model_signal_list(msi, models, verbose=False)
    eeg = tme.generate_sine_signal(target_hz, fs, duration, channels)
    if harmonic_weight > 0:
        eeg = eeg + float(harmonic_weight) * tme.generate_sine_signal(
            2.0 * float(target_hz), fs, duration, channels
        )
    if noise_std > 0:
        eeg = eeg + float(noise_std) * np.random.default_rng(0).standard_normal(eeg.shape)
    sig = tme.numpy_to_double_matrix2d(np.ascontiguousarray(eeg, dtype=np.float64), verbose=False)
    winner, coef_s = tme.msi_exec_signal(msi, sig)
    vec = coef_values(
        msi,
        len(freqs),
        winner_1based=int(winner),
        winner_coef=float(coef_s),
    )
    return int(winner), float(coef_s), vec


def run_synthetic_battery(
    *,
    freqs: Sequence[float] = (8.0, 10.0, 12.0, 15.0),
    fs: float = 250.0,
    duration: float = 2.0,
    repeats: int = 3,
) -> Dict[str, Any]:
    msi, msi_res, dotnet_root = tme.load_msi_runtime()
    cases: List[SyntheticCaseResult] = []

    scenarios: List[Dict[str, Any]] = []
    for target in freqs:
        scenarios.append(
            {
                "name": f"pure_{target}Hz",
                "target_hz": float(target),
                "channels": 2,
                "noise_std": 0.0,
                "harmonic_weight": 0.0,
            }
        )
    for target in (10.0, 12.0):
        scenarios.append(
            {
                "name": f"harmonic_{target}Hz",
                "target_hz": float(target),
                "channels": 2,
                "noise_std": 0.0,
                "harmonic_weight": 0.3,
            }
        )
    scenarios.append(
        {
            "name": "noise_12Hz",
            "target_hz": 12.0,
            "channels": 2,
            "noise_std": 0.35,
            "harmonic_weight": 0.0,
        }
    )
    scenarios.append(
        {
            "name": "multichannel_12Hz",
            "target_hz": 12.0,
            "channels": 4,
            "noise_std": 0.0,
            "harmonic_weight": 0.0,
        }
    )

    for sc in scenarios:
        target_hz = float(sc["target_hz"])
        exp_w = _expected_index(freqs, target_hz)
        winners: List[int] = []
        coefs: List[float] = []
        vec: List[Optional[float]] = []
        for _ in range(max(1, int(repeats))):
            w, c, vec = _run_once(
                msi,
                freqs=freqs,
                fs=fs,
                duration=duration,
                target_hz=target_hz,
                channels=int(sc["channels"]),
                noise_std=float(sc["noise_std"]),
                harmonic_weight=float(sc["harmonic_weight"]),
            )
            winners.append(w)
            coefs.append(c)
        cases.append(
            SyntheticCaseResult(
                name=str(sc["name"]),
                freqs_hz=[float(x) for x in freqs],
                target_hz=target_hz,
                expected_winner_1based=exp_w,
                actual_winner_1based=int(winners[-1]),
                winner_match=bool(winners[-1] == exp_w),
                coef_s=float(coefs[-1]),
                coef_vector=vec,
                channels=int(sc["channels"]),
                noise_std=float(sc["noise_std"]),
                harmonic_weight=float(sc["harmonic_weight"]),
                repeat_std_winner=float(np.std(winners)) if len(winners) > 1 else 0.0,
                repeat_std_coef=float(np.std(coefs)) if len(coefs) > 1 else 0.0,
            )
        )

    passed = sum(1 for c in cases if c.winner_match)
    return {
        "schema": "ssvep_msi_synthetic_validation/v1",
        "msi_res": str(msi_res),
        "dotnet_root": str(dotnet_root),
        "msi_contract": {
            "coef_property": "System.Double (scalar, stale 0.0 after single-arg MSIExec)",
            "per_frequency_coef_vector": "not_supported_by_dll",
            "winner_score": "MSIExec(Signal, out double coefS)",
        },
        "params": {"freqs_hz": list(freqs), "fs_hz": fs, "duration_sec": duration, "repeats": repeats},
        "summary": {
            "cases": len(cases),
            "winner_match": passed,
            "winner_match_rate": round(passed / max(1, len(cases)), 4),
        },
        "cases": [asdict(c) for c in cases],
    }


def write_report(payload: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SSVEP MSI synthetic validation (Stage 2.2)",
        "",
        f"- Cases: {payload['summary']['cases']}",
        f"- Winner match: {payload['summary']['winner_match']}/{payload['summary']['cases']}",
        "",
        "## MSI API contract",
        "",
        f"- Coef property: `{payload['msi_contract']['coef_property']}`",
        f"- Per-frequency vector: **{payload['msi_contract']['per_frequency_coef_vector']}**",
        f"- Winner score: `{payload['msi_contract']['winner_score']}`",
        "",
        "## Cases",
        "",
        "| Case | Target | Expected | Actual | coefS | Match |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in payload["cases"]:
        lines.append(
            f"| {c['name']} | {c['target_hz']} | {c['expected_winner_1based']} | "
            f"{c['actual_winner_1based']} | {c['coef_s']:.4g} | {c['winner_match']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
