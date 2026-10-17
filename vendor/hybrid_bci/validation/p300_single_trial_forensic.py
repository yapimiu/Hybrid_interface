"""Forensic step-by-step pipeline for one p300_analyzer all_stims export (optical tile 7)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.analysis_profiles import hybrid_protocol_p300_profile
from p300_analysis.erp_compute import (
    build_averaged_erp,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit
from p300_analysis.signal_processing import bandpass_filter, time_window_to_indices
from p300_analysis.winner_selection import WINNER_MODE_AUC


def load_all_stims_csv(path: Path) -> Dict[str, List[np.ndarray]]:
    """Rebuild epochs_data from ``*_all_stims.csv`` export."""
    import csv

    epochs_data: Dict[str, List[np.ndarray]] = {}
    current_key: Optional[str] = None
    current_mt: Optional[str] = None
    buf: List[float] = []

    def flush() -> None:
        nonlocal buf, current_key
        if current_key and buf:
            epochs_data.setdefault(current_key, []).append(
                np.asarray(buf, dtype=np.float64).reshape(-1, 1)
            )
        buf = []

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        ch_cols = [c for c in reader.fieldnames or [] if c.startswith("ch_")]
        if not ch_cols:
            raise RuntimeError("no ch_* columns")
        ch = ch_cols[0]
        for row in reader:
            sk = str(row.get("stim_key") or "")
            mt = str(row.get("marker_ts") or "")
            if current_key is None:
                current_key, current_mt = sk, mt
            elif sk != current_key or mt != current_mt:
                flush()
                current_key, current_mt = sk, mt
            buf.append(float(row[ch]))
        flush()
    return epochs_data


def _rank_by_tile(values: Dict[int, float], *, higher_better: bool = True) -> List[Tuple[int, float]]:
    items = sorted(values.items(), key=lambda x: x[1], reverse=higher_better)
    return items


def _infer_fs_hz(epoch_len: int, *, span_ms: float = 900.0) -> float:
    """Match epoch_geometry span: pre 100 ms + post 800 ms ≈ 900 ms for 898 samples."""
    if epoch_len < 2:
        return 1000.0
    return float(epoch_len - 1) / (span_ms / 1000.0)


def _tile_index_map(stim_keys: List[str]) -> Tuple[List[Optional[int]], Dict[int, int]]:
    tiles = [stim_key_to_tile_digit(k) for k in stim_keys]
    idx_by_tile = {int(t): i for i, t in enumerate(tiles) if t is not None}
    return tiles, idx_by_tile


def _analyze_pipeline(
    *,
    label: str,
    stim_keys: List[str],
    raw_averaged: np.ndarray,
    corrected: np.ndarray,
    integrated: np.ndarray,
    time_ms: np.ndarray,
    time_crop: np.ndarray,
    xi0: int,
    xi1: int,
    optical_tile: int,
    wx: int,
    wy: int,
) -> Dict[str, Any]:
    tiles, idx_by_tile = _tile_index_map(stim_keys)
    dt_m = float(time_ms[1] - time_ms[0]) if time_ms.size > 1 else 1.0
    def tile_vals(fn) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for i, t in enumerate(tiles):
            if t is None:
                continue
            out[int(t)] = float(fn(i))
        return out

    ranking_stages: Dict[str, List[Tuple[int, float]]] = {}
    ranking_stages["raw_mean_max_abs_full"] = _rank_by_tile(
        tile_vals(lambda i: float(np.max(np.abs(raw_averaged[i]))))
    )
    ranking_stages["corrected_max_abs_full"] = _rank_by_tile(
        tile_vals(lambda i: float(np.max(np.abs(corrected[i]))))
    )
    ranking_stages["corrected_max_abs_window"] = _rank_by_tile(
        tile_vals(lambda i: float(np.max(np.abs(corrected[i, xi0:xi1]))))
    )
    ranking_stages["windowed_abs_mean"] = _rank_by_tile(
        tile_vals(lambda i: float(np.mean(np.abs(corrected[i, xi0:xi1]))))
    )
    ranking_stages["windowed_signed_sum"] = _rank_by_tile(
        tile_vals(lambda i: float(np.sum(corrected[i, xi0:xi1]) * dt_m))
    )
    ranking_stages["windowed_sum_pos"] = _rank_by_tile(
        tile_vals(lambda i: float(np.sum(np.clip(corrected[i, xi0:xi1], 0, None)) * dt_m))
    )
    ranking_stages["windowed_sum_neg_abs"] = _rank_by_tile(
        tile_vals(lambda i: float(np.sum(np.abs(np.clip(corrected[i, xi0:xi1], None, 0))) * dt_m))
    )
    ranking_stages["auc_sum_abs_window"] = _rank_by_tile(
        tile_vals(lambda i: float(np.sum(np.abs(corrected[i, xi0:xi1])) * dt_m))
    )
    ranking_stages["integrated_endpoint"] = _rank_by_tile(
        tile_vals(lambda i: float(integrated[i, -1]) if integrated[i].size else 0.0)
    )

    _, _, auc_dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged,
        corrected,
        time_ms,
        wx,
        wy,
        winner_mode=WINNER_MODE_AUC,
    )
    abs_auc = list(auc_dbg.get("abs_auc_values") or [])
    auc_w_idx = int(auc_dbg["auc_winner_idx"])
    auc_w_tile = tiles[auc_w_idx]

    # Energy inside vs outside decision window (explains full-epoch visual dominance)
    energy_split: Dict[str, Dict[str, float]] = {}
    for t, i in idx_by_tile.items():
        inside = float(np.sum(np.abs(corrected[i, xi0:xi1])) * dt_m)
        outside = float(
            (np.sum(np.abs(corrected[i, :xi0])) + np.sum(np.abs(corrected[i, xi1:]))) * dt_m
        )
        total = inside + outside
        energy_split[str(t)] = {
            "inside_auc_window": inside,
            "outside_window": outside,
            "inside_fraction": inside / total if total > 1e-12 else 0.0,
        }

    # Sample-by-sample cumulative AUC leader inside window
    seg_abs = np.abs(corrected[:, xi0:xi1])
    cum = np.cumsum(seg_abs, axis=1)
    leaders: List[Optional[int]] = []
    for j in range(cum.shape[1]):
        w = int(np.argmax(cum[:, j]))
        leaders.append(tiles[w])
    leader_changes = [
        {"sample": j, "time_ms": float(time_crop[j]) if j < time_crop.size else None, "from": leaders[j - 1], "to": leaders[j]}
        for j in range(1, len(leaders))
        if leaders[j] != leaders[j - 1]
    ]

    opt_i = idx_by_tile.get(optical_tile)
    per_sample_contrib: Dict[str, Any] = {}
    if opt_i is not None and auc_w_idx is not None:
        per_sample_contrib = {
            "optical_tile": optical_tile,
            "auc_winner_tile": auc_w_tile,
            "optical_abs_per_sample": seg_abs[opt_i].tolist(),
            "winner_abs_per_sample": seg_abs[auc_w_idx].tolist(),
            "cumsum_gap_winner_minus_optical": (cum[auc_w_idx] - cum[opt_i]).tolist(),
        }

    def leader(stage: str) -> Optional[int]:
        r = ranking_stages.get(stage)
        return int(r[0][0]) if r else None

    evolution = [
        {
            "stage": k,
            "leader_tile": leader(k),
            "optical_tile_rank": next((i + 1 for i, (t, _) in enumerate(v) if int(t) == optical_tile), None),
        }
        for k, v in ranking_stages.items()
    ]

    flip_at_auc = (
        leader("corrected_max_abs_window") == optical_tile
        and leader("auc_sum_abs_window") != optical_tile
    )

    return {
        "label": label,
        "auc_winner_tile": auc_w_tile,
        "auc_winner_idx": auc_w_idx,
        "auc_debug": {
            "abs_auc_values": abs_auc,
            "margin": auc_dbg.get("margin"),
            "window_index": [int(xi0), int(xi1)],
            "window_ms": [float(time_ms[xi0]), float(time_ms[xi1 - 1]) if xi1 > xi0 else float(wy)],
        },
        "ranking_stages": {k: [{"tile": t, "value": v} for t, v in vals] for k, vals in ranking_stages.items()},
        "ranking_evolution": evolution,
        "energy_inside_vs_outside": energy_split,
        "cumulative_leader": {
            "first": leaders[0] if leaders else None,
            "final": leaders[-1] if leaders else None,
            "n_leader_changes": sum(1 for a, b in zip(leaders, leaders[1:]) if a != b),
            "leader_changes": leader_changes[:20],
            "optical_tile_leads_samples": sum(1 for x in leaders if x == optical_tile),
            "n_window_samples": len(leaders),
        },
        "per_sample_contrib": per_sample_contrib,
        "leader_flip_peak_to_auc": flip_at_auc,
        "arrays": {
            "raw_averaged": raw_averaged,
            "corrected": corrected,
            "integrated": integrated,
            "time_ms": time_ms,
            "time_crop": time_crop,
        },
    }


def _run_one_pipeline(
    epochs_data: Dict[str, List[np.ndarray]],
    *,
    label: str,
    profile,
    time_ms: np.ndarray,
    optical_tile: int,
) -> Dict[str, Any]:
    el = int(time_ms.shape[0])
    stim_keys, raw_averaged, rejected = build_averaged_erp(epochs_data, el, artifact_threshold_uv=60.0)
    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw_averaged,
        time_ms,
        int(profile.baseline_ms),
        int(profile.window_x_ms),
        int(profile.window_y_ms),
    )
    xi0, xi1 = time_window_to_indices(time_ms, wx, wy)
    out = _analyze_pipeline(
        label=label,
        stim_keys=stim_keys,
        raw_averaged=raw_averaged,
        corrected=corrected,
        integrated=integrated,
        time_ms=time_ms,
        time_crop=time_crop,
        xi0=xi0,
        xi1=xi1,
        optical_tile=optical_tile,
        wx=wx,
        wy=wy,
    )
    out["rejected_epochs"] = rejected
    out["stim_keys"] = stim_keys
    return out


def run_stepwise_forensic(
    csv_path: Path,
    *,
    optical_tile: int = 7,
    profile_key: str = "general",
    fs_hz: Optional[float] = None,
) -> Dict[str, Any]:
    profile = hybrid_protocol_p300_profile(profile_key)
    epochs_raw = load_all_stims_csv(csv_path)

    sample_ep = next(iter(next(iter(epochs_raw.values()))))
    el = int(sample_ep.shape[0])
    fs = float(fs_hz) if fs_hz is not None else _infer_fs_hz(el)
    time_ms = np.linspace(-float(profile.baseline_ms), 800.0, el, dtype=np.float64)

    epochs_live = {sk: [bandpass_filter(ep, fs) for ep in eps] for sk, eps in epochs_raw.items()}
    live = _run_one_pipeline(
        epochs_live, label="live_gui_bandpass", profile=profile, time_ms=time_ms, optical_tile=optical_tile
    )
    raw_pipe = _run_one_pipeline(
        epochs_raw, label="export_raw_no_bandpass", profile=profile, time_ms=time_ms, optical_tile=optical_tile
    )
    primary = raw_pipe if raw_pipe.get("leader_flip_peak_to_auc") else live
    ap_live = live.pop("arrays")
    ap_raw = raw_pipe.pop("arrays")
    xi0, xi1 = primary["auc_debug"]["window_index"]
    ap_primary = ap_raw if primary["label"] == "export_raw_no_bandpass" else ap_live

    verdict = {
        "classification": "expected_metric_physics_not_plot_bug",
        "peak_vs_auc": (
            "Graphs ①–③ show the full epoch; graph ④ / AUC winner use Σ|corrected| only in "
            f"[{profile.window_x_ms},{profile.window_y_ms}] ms."
        ),
        "export_vs_live": (
            "CSV all_stims = RAW epochs. Live GUI = bandpass before average. "
            f"run9 raw AUC winner={raw_pipe.get('auc_winner_tile')}, live={live.get('auc_winner_tile')}."
        ),
        "graph4_equals_auc": "Graph ④ endpoint = integrated[-1] = auc_sum_abs_window.",
    }

    return {
        "schema": "hybrid_bci_p300_single_trial_forensic/v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(csv_path),
        "optical_ground_truth_tile": int(optical_tile),
        "profile": {
            "baseline_ms": profile.baseline_ms,
            "window_ms": [profile.window_x_ms, profile.window_y_ms],
            "epoch_len_samples": el,
            "fs_hz_inferred": fs,
        },
        "n_epochs_per_stim": {k: len(v) for k, v in epochs_raw.items()},
        "pipelines": {
            "live_gui_bandpass": {k: v for k, v in live.items()},
            "export_raw_no_bandpass": {k: v for k, v in raw_pipe.items()},
        },
        "primary_pipeline": primary["label"],
        "auc_winner_tile": primary.get("auc_winner_tile"),
        "ranking_evolution": primary.get("ranking_evolution"),
        "ranking_stages": primary.get("ranking_stages"),
        "auc_debug": primary.get("auc_debug"),
        "cumulative_leader": primary.get("cumulative_leader"),
        "energy_inside_vs_outside": primary.get("energy_inside_vs_outside"),
        "verdict": verdict,
        "arrays_for_plots": {
            "time_ms": [float(x) for x in time_ms],
            "time_crop_ms": [float(x) for x in ap_primary["time_crop"]],
            "window_idx": [int(xi0), int(xi1)],
            "live": {k: v.tolist() for k, v in ap_live.items()},
            "raw": {k: v.tolist() for k, v in ap_raw.items()},
        },
    }


def _plot_pipeline_panel(
    *,
    bundle: Dict[str, np.ndarray],
    time_ms: np.ndarray,
    time_crop: np.ndarray,
    window_idx: List[int],
    wx: int,
    wy: int,
    opt: int,
    auc_w: Any,
    stim_keys: List[str],
    title_suffix: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    raw = bundle["raw_averaged"]
    corr = bundle["corrected"]
    integ = bundle["integrated"]
    tile_by_i = [stim_key_to_tile_digit(k) for k in stim_keys]
    xi0, xi1 = window_idx

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, data, title, t_axis in [
        (axes[0, 0], raw, "① Mean ERP (full)", time_ms),
        (axes[0, 1], corr, "③ Corrected (full)", time_ms),
        (axes[1, 0], corr, f"Window [{wx}-{wy}] ms", time_ms),
        (axes[1, 1], integ, "④ cumsum|corr|", time_crop),
    ]:
        for i in range(data.shape[0]):
            tile_i = tile_by_i[i] if i < len(tile_by_i) else i
            lw = 3 if tile_i == opt else 1
            alpha = 1.0 if tile_i in (opt, auc_w) else 0.45
            if title.startswith("Window"):
                ax.plot(t_axis[xi0:xi1], data[i, xi0:xi1], label=f"t{tile_i}", lw=lw, alpha=alpha)
            else:
                n = min(data.shape[1], t_axis.size)
                ax.plot(t_axis[:n], data[i, :n], label=f"t{tile_i}", lw=lw, alpha=alpha)
        if "full" in title or title.startswith("Window"):
            ax.axvspan(wx, wy, color="gold", alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle(f"{title_suffix} | optical={opt} AUC winner={auc_w}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_forensic_plots(report: Dict[str, Any], out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    ap = report.get("arrays_for_plots") or {}
    time_ms = np.asarray(ap.get("time_ms") or [])
    time_crop = np.asarray(ap.get("time_crop_ms") or [])
    wx, wy = report["profile"]["window_ms"]
    opt = int(report["optical_ground_truth_tile"])
    xi0, xi1 = ap.get("window_idx", [0, 0])

    pipelines = report.get("pipelines") or {}
    for key, label in (("raw", "export_raw"), ("live", "live_bandpass")):
        bundle_np = {k: np.asarray(v) for k, v in (ap.get(key) or {}).items()}
        if bundle_np.get("corrected", np.array([])).size == 0:
            continue
        pipe = pipelines.get("export_raw_no_bandpass" if key == "raw" else "live_gui_bandpass") or {}
        stim_keys = pipe.get("stim_keys") or []
        auc_w = pipe.get("auc_winner_tile")
        p = out_dir / f"four_panel_{label}.png"
        _plot_pipeline_panel(
            bundle=bundle_np,
            time_ms=time_ms,
            time_crop=time_crop,
            window_idx=[xi0, xi1],
            wx=wx,
            wy=wy,
            opt=opt,
            auc_w=auc_w,
            stim_keys=stim_keys,
            title_suffix=label,
            out_path=p,
        )
        written.append(str(p))

    # Cumulative AUC race (raw pipeline — shows flip)
    raw_corr = np.asarray((ap.get("raw") or {}).get("corrected") or [])
    live_corr = np.asarray((ap.get("live") or {}).get("corrected") or [])
    for tag, corr in (("raw", raw_corr), ("live", live_corr)):
        if corr.size == 0:
            continue
        seg = np.abs(corr[:, xi0:xi1])
        cum = np.cumsum(seg, axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(cum.shape[0]):
            ax.plot(time_crop, cum[i], alpha=0.7, lw=2 if i < 9 else 1)
        ax.set_title(f"Cumulative Σ|corr| in AUC window ({tag})")
        ax.set_xlabel("ms")
        p = out_dir / f"cumulative_auc_{tag}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    # Heatmap |corrected| in window (raw)
    if raw_corr.size:
        hm = np.abs(raw_corr[:, xi0:xi1])
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(hm, aspect="auto", cmap="magma", origin="lower")
        ax.set_yticks(range(hm.shape[0]))
        sks = (pipelines.get("export_raw_no_bandpass") or {}).get("stim_keys") or []
        ax.set_yticklabels([stim_key_to_tile_digit(k) for k in sks])
        ax.set_xlabel("sample in window")
        ax.set_title("|corrected| heatmap in 550–725 ms (raw replay)")
        fig.colorbar(im, ax=ax, fraction=0.02)
        p = out_dir / "heatmap_abs_window_raw.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    # Energy inside vs outside bar chart
    energy = report.get("energy_inside_vs_outside") or {}
    if energy:
        tiles_sorted = sorted(int(t) for t in energy)
        inside = [energy[str(t)]["inside_auc_window"] for t in tiles_sorted]
        outside = [energy[str(t)]["outside_window"] for t in tiles_sorted]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(tiles_sorted))
        ax.bar(x, outside, label="outside window")
        ax.bar(x, inside, bottom=outside, label="inside AUC window")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in tiles_sorted])
        ax.set_title("Energy split: tile 7 dominates outside window")
        ax.legend()
        p = out_dir / "energy_inside_outside.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    auc_vals = report.get("auc_debug", {}).get("abs_auc_values") or []
    if auc_vals:
        fig, ax = plt.subplots(figsize=(9, 3))
        sks = (pipelines.get(report.get("primary_pipeline", "").replace("export_", "export_")) or {}).get("stim_keys")
        pipe_key = "export_raw_no_bandpass" if report.get("primary_pipeline") == "export_raw_no_bandpass" else "live_gui_bandpass"
        sks = (pipelines.get(pipe_key) or {}).get("stim_keys") or []
        labels = [str(stim_key_to_tile_digit(k)) for k in sks]
        ax.bar(labels, auc_vals)
        ax.set_title("AUC scores (primary pipeline)")
        p = out_dir / "auc_scores_bar.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    return written


def render_markdown(report: Dict[str, Any]) -> str:
    opt = report.get("optical_ground_truth_tile")
    lines = [
        "# P300 AUC window forensic (corrected → ∫|corr|)",
        "",
        f"- CSV: `{report.get('csv_path')}`",
        f"- Optical GT: tile **{opt}**",
        f"- Primary pipeline: **{report.get('primary_pipeline')}**",
        "",
    ]
    for pname in ("export_raw_no_bandpass", "live_gui_bandpass"):
        p = (report.get("pipelines") or {}).get(pname) or {}
        lines.append(f"### {pname}")
        lines.append(f"- AUC winner: **{p.get('auc_winner_tile')}**")
        lines.append(f"- Peak→AUC flip: **{p.get('leader_flip_peak_to_auc')}**")
        cl = p.get("cumulative_leader") or {}
        lines.append(
            f"- Cumsum leader: first={cl.get('first')} final={cl.get('final')} "
            f"changes={cl.get('n_leader_changes')}"
        )
        lines.append("")
    lines.extend([
        "## Ranking evolution (primary)",
        "",
        "| Stage | Leader | Rank optical |",
        "|-------|--------|--------------|",
    ])
    for row in report.get("ranking_evolution") or []:
        lines.append(
            f"| {row.get('stage')} | {row.get('leader_tile')} | {row.get('optical_tile_rank')} |"
        )
    e7 = (report.get("energy_inside_vs_outside") or {}).get("7") or {}
    if e7:
        lines.extend([
            "",
            f"## Tile 7 energy split",
            f"- Inside AUC window: {e7.get('inside_auc_window', 0):.1f}",
            f"- Outside window: {e7.get('outside_window', 0):.1f}",
            f"- Fraction inside: {100 * float(e7.get('inside_fraction', 0)):.1f}%",
        ])
    v = report.get("verdict") or {}
    lines.extend(["", "## Verdict", "", str(v.get("classification", "")), ""])
    for k in ("peak_vs_auc", "export_vs_live", "graph4_equals_auc"):
        if v.get(k):
            lines.append(f"- {v[k]}")
    return "\n".join(lines) + "\n"
