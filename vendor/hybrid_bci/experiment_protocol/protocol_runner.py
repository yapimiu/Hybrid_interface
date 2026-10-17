from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pylsl import StreamInlet, StreamInfo, local_clock as lsl_local_clock
except Exception:  # pragma: no cover
    StreamInlet = object  # type: ignore
    StreamInfo = object  # type: ignore

    def lsl_local_clock() -> float:  # type: ignore
        return time.time()

from experiment_protocol import protocol_log as plog
from experiment_protocol.experiment_queue import (
    default_ssvep_experiment_lamps,
    ensure_ssvep_experiment_lamp_permutation,
    normalize_ssvep_experiment_lamps,
    shuffle_ssvep_experiment_lamps,
    ssvep_physical_blocks_per_mode,
)
from experiment_protocol.msi_decision_trace import MsiDecisionTraceCollector
from experiment_protocol.ssvep_group_experiment import aggregate_ssvep_group_experiment
from runtime_core.protocol.queue import (
    QueueItem,
    build_main_queue,
    build_p300_block_queue,
    build_ssvep_mode_queue,
    queue_summary,
)
from runtime_core.adapters import lsl, stim as stim_ctl
from runtime_core.protocol.display_state import ProtocolDisplayView
from experiment_protocol.minimal_session_logger import (
    build_p300_averages_export,
    build_ssvep_averages_export,
)
from runtime_core.logging import UnifiedExperimentLogger
from runtime_core.logging.protocols import ExperimentSessionLogger
from p300_analysis.marker_parsing import stim_key_to_tile_digit
from p300_analysis.constants import EEG_PULL_MAX_SAMPLES, MARKERS_PULL_MAX_SAMPLES
from p300_analysis.marker_parsing import parse_trial_end, parse_trial_target_tile_id
from p300_analysis.analysis_profiles import hybrid_protocol_p300_profile
from runtime_core.engines import (
    P300Decision,
    P300EngineParams,
    P300OnlineEngine,
    P300OnlineEngineProtocol,
    SSVEPDecision,
    SSVEPOnlineEngine,
    SSVEPOnlineEngineProtocol,
    SSVEPParams,
)
from ssvep_analysis.lamp_frequencies import (
    DEFAULT_LAMP_FREQS_HZ,
    MSI_DEFAULT_FS,
    MSI_DEFAULT_SAMPLES,
    MSI_DEFAULT_WINDOW_SEC,
    SSVEP_PROTOCOL_ROI_0IDX,
)

_HYBRID_P300_PROFILE = hybrid_protocol_p300_profile()
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR
from ssvep_analysis.burst_gate import parse_lsl_marker
from runtime_core.adapters.migalka import (
    LampEvent,
    MigalkaConfig,
    MigalkaEventQueue,
    MigalkaSerialController,
)


@dataclass(frozen=True)
class ProtocolConfig:
    output_root: Path
    subject_id: str
    com_port: str
    subject_last_name: str = ""
    subject_first_name: str = ""
    # Пустые строки — первый найденный поток ЭЭГ при предстарте
    eeg_stream_name: str = ""
    eeg_stream_session_id: str = ""

    # Основной блок: 15 P300 (AUC+шаблон за один trial) + 15 SSVEP×2 (перемешано)
    p300_main_trials: int = 15
    ssvep_experiments_per_mode: int = 4
    ssvep_group_size: int = 4
    ssvep_blocks_per_mode: int = 0  # 0 = auto: experiments_per_mode × group_size
    ssvep_continuous_lamps_0idx: Tuple[int, ...] = (0, 1, 2, 3)
    ssvep_burst_lamps_0idx: Tuple[int, ...] = (0, 1, 2, 3)
    ssvep_auto_shuffle_on_prepare: bool = True
    ssvep_lamp_shuffle_seed: int = 0
    ssvep_inter_block_gap_s: float = 0.0
    ssvep_group_visual_continuous: bool = True
    pause_between_experiments_s: float = 2.0
    shuffle_seed: int = -1  # -1 = случайный seed при старте main

    # Калибровка P300 в начале (подряд, до main)
    p300_calib_trials: int = 5
    calib_target_tile_id: int = 4
    template_warmup_target_epochs: int = 12

    # SSVEP timing / MSI — defaults from ssvep_analysis.lamp_frequencies
    # continuous: wall-clock block duration; burst: global ON budget (shared, T ms to firmware)
    ssvep_block_sec: float = 8.0
    ssvep_burst_pause_margin_sec: float = 3.0  # wall cap ≈ ON_budget + pause margin
    ssvep_window_sec: float = MSI_DEFAULT_WINDOW_SEC
    ssvep_fs_hz: float = MSI_DEFAULT_FS
    ssvep_msi_samples: int = MSI_DEFAULT_SAMPLES
    ssvep_freqs_hz: Tuple[float, ...] = DEFAULT_LAMP_FREQS_HZ
    # P300 ROI / window — defaults from p300_analysis.analysis_profiles (general)
    roi_channels_0idx: Tuple[int, ...] = _HYBRID_P300_PROFILE.roi_channels_0idx
    ssvep_roi_channels_0idx: Tuple[int, ...] = SSVEP_PROTOCOL_ROI_0IDX

    # Папка сессии (если создана до старта стимулятора — общий stim_control.json)
    session_dir: Optional[Path] = None


def _ru_stop_reason(reason: str) -> str:
    return {"user_stop": "остановка оператором"}.get(str(reason), str(reason))


def _ru_p300_winner_mode(winner_mode: str) -> str:
    if winner_mode == WINNER_MODE_AUC:
        return "AUC"
    if winner_mode == WINNER_MODE_TEMPLATE_CORR:
        return "сравнение с шаблоном"
    return str(winner_mode)


def _ru_ssvep_stim_mode(mode: str) -> str:
    if mode == "continuous":
        return "непрерывный"
    if mode == "burst":
        return "пакетный"
    return str(mode)


class ProtocolState:
    Idle = "idle"
    Preflight = "preflight"
    Ready = "ready"  # сессия активна, оператор выбирает блок
    P300Calib = "p300_calib"  # legacy, не используется
    Main = "main"
    SSVEP_CONT = "ssvep_continuous"
    SSVEP_BURST = "ssvep_burst"
    Finalize = "finalize"
    Stopped = "stopped"


class ProtocolRunner:
    """Non-UI state machine: pulls LSL, controls Migalka, logs, produces trial/block results."""

    def __init__(self, cfg: ProtocolConfig) -> None:
        self.cfg = cfg
        self.state = ProtocolState.Idle
        self.status_text = "Ожидание"

        self._inlet_eeg: Optional[StreamInlet] = None
        self._inlet_markers: Optional[StreamInlet] = None
        self._inlet_migalka_markers: Optional[StreamInlet] = None
        self._last_eeg_data_wall_at: float = 0.0

        self._logger: Optional[ExperimentSessionLogger] = None
        self._p300: P300OnlineEngineProtocol = P300OnlineEngine()
        self._ssvep: SSVEPOnlineEngineProtocol = SSVEPOnlineEngine()
        self._migalka_event_queue = MigalkaEventQueue()
        self._migalka = MigalkaSerialController(
            mirror_lsl=True,
            event_queue=self._migalka_event_queue,
        )
        self._migalka_drain_last_stats: Dict[str, float | int] = {}
        self._runtime_resources_released: bool = False
        self._logger_finalized: bool = False

        self._p300_calib_trials_done = 0
        self._p300_main_trials_done = 0
        self._ssvep_block_count_cont = 0
        self._ssvep_block_count_burst = 0
        self._main_queue: List[QueueItem] = []
        self._main_queue_index = 0
        self._shuffle_seed_used: Optional[int] = None
        self._active_block_label: str = ""
        self._ssvep_any_continuous_done = False

        self._p300_trial_armed: bool = False
        self._p300_trial_end_seen: bool = False

        self._ssvep_block_started_at: Optional[float] = None
        self._ssvep_target_lamp: Optional[int] = None
        self._ssvep_last_msi_log_at: Optional[float] = None
        self._ssvep_msi_trace = MsiDecisionTraceCollector(min_interval_sec=2.0)
        self._ssvep_burst_device_done: bool = False
        self._ssvep_burst_stim_stats: Optional[Dict[str, Any]] = None
        self._ssvep_burst_block_start_pending: bool = False
        self._ssvep_group_subblocks: List[Dict[str, Any]] = []
        self._ssvep_experiment_count_cont: int = 0
        self._ssvep_experiment_count_burst: int = 0
        self._ssvep_committed_lamps: Dict[str, Tuple[int, ...]] = {}
        self._ssvep_mode_shuffle_seeds: Dict[str, int] = {}
        self._ssvep_cue_sequence_callback: Optional[Callable[[str, Tuple[int, ...]], None]] = None
        self._template_locked: bool = False

        # Неблокирующая пауза перед стартом следующей "единицы эксперимента" (trial/block).
        self._pause_until_wall: Optional[float] = None
        self._pause_status: Optional[str] = None
        self._pause_next_state: Optional[str] = None
        self._pause_next_ssvep_mode: Optional[str] = None
        self._last_state: str = ProtocolState.Idle
        self._eeg_nominal_fs_hz: float = 250.0
        self._ssvep_migalka_retry_at: float = 0.0
        self._ssvep_migalka_fail_streak: int = 0
        # Полноэкранная подсказка для оператора (рисует protocol_runner_gui)
        self.ssvep_cue_visible: bool = False
        self.ssvep_cue_exp_index: int = 0
        self.ssvep_cue_exp_total: int = 0
        self.ssvep_cue_lamp_1based: int = 1
        self.ssvep_cue_freq_hz: float = 0.0
        self.ssvep_cue_mode_label: str = ""
        self.ssvep_cue_group_index: int = -1
        self.ssvep_blackout_visible: bool = False
        self._on_ssvep_display_clear: Optional[Callable[[], None]] = None
        # Подсказки испытуемелю (читает protocol_runner_gui → ProtocolInstructionOverlay)
        self.participant_instruction: Optional[Dict[str, Any]] = None
        self._tech_last_snapshot_wall: float = 0.0
        self._tech_snapshot_interval_sec: float = 1.0

    def set_ssvep_cue_sequence_callback(
        self, cb: Optional[Callable[[str, Tuple[int, ...]], None]]
    ) -> None:
        """GUI: обновить combos после commit sequence при старте режима SSVEP."""
        self._ssvep_cue_sequence_callback = cb

    def set_ssvep_display_clear_callback(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_ssvep_display_clear = cb

    def display_view(self) -> ProtocolDisplayView:
        """Read-only operator UI snapshot (GUI should prefer this over field polling)."""
        return ProtocolDisplayView(
            state=str(self.state),
            status_text=str(self.status_text),
            session_ready=bool(self.session_ready),
            block_running=bool(self.block_running),
            pause_active=bool(self.pause_active),
            eeg_inlet_open=bool(self.eeg_inlet_open),
            eeg_stream_live=bool(self.eeg_stream_live),
            ssvep_blackout_visible=bool(self.ssvep_blackout_visible),
            ssvep_cue_visible=bool(self.ssvep_cue_visible),
            ssvep_cue_exp_index=int(self.ssvep_cue_exp_index),
            ssvep_cue_exp_total=int(self.ssvep_cue_exp_total),
            ssvep_cue_lamp_1based=int(self.ssvep_cue_lamp_1based),
            ssvep_cue_freq_hz=float(self.ssvep_cue_freq_hz),
            ssvep_cue_mode_label=str(self.ssvep_cue_mode_label),
            ssvep_cue_group_index=int(self.ssvep_cue_group_index),
        )

    def dismiss_operator_ssvep_overlay(self) -> None:
        """GUI: clear SSVEP overlay flags without full display teardown."""
        self.ssvep_blackout_visible = False
        self.ssvep_cue_visible = False

    def _runtime_metrics_snapshot(self) -> Dict[str, Any]:
        """Encapsulate engine internals for tech trace (not for GUI)."""
        marker_off = self._p300._marker_eeg_ts_offset
        buf = self._ssvep._buf
        buf_ch = int(buf.shape[1]) if getattr(buf, "ndim", 0) == 2 else 0
        return {
            "p300_pending_markers": int(len(self._p300.pending_markers)),
            "p300_epoch_classes": int(len(self._p300.epochs_data)),
            "p300_epoch_counts": {str(k): int(len(v)) for k, v in self._p300.epochs_data.items()},
            "p300_marker_offset": float(marker_off) if marker_off is not None else None,
            "ssvep_buf_samples": int(buf.shape[0]) if getattr(buf, "shape", None) else 0,
            "ssvep_buf_channels": buf_ch,
            "ssvep_n_samples": int(self._ssvep._n_samples),
            "ssvep_can_classify": bool(self._ssvep.can_classify()),
            "ssvep_mode": str(self._ssvep.params.mode),
            "ssvep_target_lamp": self._ssvep_target_lamp,
            "migalka_queue_pending": int(self._migalka.pending_event_count()),
            "migalka_drain_last": dict(self._migalka_drain_last_stats),
        }

    def _set_ssvep_blackout(self, visible: bool) -> None:
        self.ssvep_blackout_visible = bool(visible)
        if visible:
            self.ssvep_cue_visible = False
            self._update_participant_instruction(kind="blackout")

    def clear_ssvep_display(self) -> None:
        """Сброс флагов оверлея (чёрный экран / подсказка ССВП)."""
        self._set_ssvep_blackout(False)
        self._set_ssvep_cue_overlay(ssvep_mode=None)
        if self._on_ssvep_display_clear is not None:
            try:
                self._on_ssvep_display_clear()
            except Exception:
                pass

    @property
    def pause_active(self) -> bool:
        return self._pause_until_wall is not None

    @property
    def pause_before_ssvep(self) -> bool:
        return self._pause_until_wall is not None and self._pause_next_ssvep_mode is not None

    def _set_state(self, new_state: str, *, detail: str = "") -> None:
        if new_state != self._last_state:
            plog.state_change(self._last_state, new_state, detail=detail)
            self._last_state = str(new_state)
        self.state = str(new_state)

    def _ssvep_physical_blocks(self) -> int:
        return ssvep_physical_blocks_per_mode(
            int(self.cfg.ssvep_experiments_per_mode),
            group_size=int(self.cfg.ssvep_group_size),
            physical_blocks_override=int(self.cfg.ssvep_blocks_per_mode),
        )

    def _ssvep_lamps_from_config(self, mode: str) -> Tuple[int, ...]:
        n_lamps = max(1, len(self._active_ssvep_freqs_hz()))
        raw = (
            self.cfg.ssvep_continuous_lamps_0idx
            if str(mode) == "continuous"
            else self.cfg.ssvep_burst_lamps_0idx
        )
        return normalize_ssvep_experiment_lamps(
            raw,
            n_experiments=int(self.cfg.ssvep_experiments_per_mode),
            n_active_lamps=n_lamps,
        )

    def _ssvep_lamps_for_mode(self, mode: str) -> Tuple[int, ...]:
        """Committed cue sequence после старта режима; иначе — только нормализация config."""
        committed = self._ssvep_committed_lamps.get(str(mode))
        if committed is not None:
            return committed
        return self._ssvep_lamps_from_config(mode)

    def _commit_ssvep_mode_sequence(self, mode: str, *, force: bool = False) -> Tuple[int, ...]:
        """
        Cue permutation при старте режима (continuous / burst): 4 уникальные лампы, фиксируется до конца mode.
        """
        mode_s = str(mode)
        if not force and mode_s in self._ssvep_committed_lamps:
            return self._ssvep_committed_lamps[mode_s]

        n_lamps = max(1, len(self._active_ssvep_freqs_hz()))
        n_exp = int(self.cfg.ssvep_experiments_per_mode)
        seed = int(random.randrange(1, 1 << 31))
        if bool(self.cfg.ssvep_auto_shuffle_on_prepare):
            lamps = shuffle_ssvep_experiment_lamps(
                n_exp, n_lamps, random.Random(seed)
            )
        else:
            lamps = ensure_ssvep_experiment_lamp_permutation(
                self._ssvep_lamps_from_config(mode_s),
                n_experiments=n_exp,
                n_active_lamps=n_lamps,
                rng=random.Random(seed),
            )

        self._ssvep_committed_lamps[mode_s] = lamps
        self._ssvep_mode_shuffle_seeds[mode_s] = seed
        seq_1 = [int(x) + 1 for x in lamps]
        plog.info(
            f"SSVEP mode sequence committed: mode={mode_s!r} seed={seed} "
            f"sequence_L={seq_1} auto_shuffle={bool(self.cfg.ssvep_auto_shuffle_on_prepare)}"
        )
        if self._logger is not None:
            self._logger.write(
                "ssvep_mode_sequence_committed",
                {
                    "mode": mode_s,
                    "shuffle_seed": seed,
                    "experiment_target_sequence_0idx": list(lamps),
                    "experiment_target_sequence": seq_1,
                    "displayed_cue_sequence": seq_1,
                    "auto_shuffle": bool(self.cfg.ssvep_auto_shuffle_on_prepare),
                },
            )
        cb = self._ssvep_cue_sequence_callback
        if cb is not None:
            try:
                cb(mode_s, lamps)
            except Exception as e:
                plog.exc("ssvep_cue_sequence_callback", e)
        return lamps

    def _ensure_ssvep_mode_sequence_committed(self, mode: str) -> Tuple[int, ...]:
        mode_s = str(mode)
        if mode_s in self._ssvep_committed_lamps:
            return self._ssvep_committed_lamps[mode_s]
        lamps = self._commit_ssvep_mode_sequence(mode_s, force=True)
        if any(
            x.kind == "ssvep" and str(x.ssvep_mode or "") == mode_s for x in self._main_queue
        ):
            self._patch_main_queue_ssvep_lamps(mode_s, lamps)
        return lamps

    def _patch_main_queue_ssvep_lamps(self, mode: str, lamps: Sequence[int]) -> None:
        mode_s = str(mode)
        lamp_list = [int(x) for x in lamps]
        patched: List[QueueItem] = []
        for item in self._main_queue:
            if item.kind != "ssvep" or str(item.ssvep_mode or "") != mode_s:
                patched.append(item)
                continue
            gi = int(item.ssvep_group_index or 0)
            if gi < 0 or gi >= len(lamp_list):
                patched.append(item)
                continue
            patched.append(
                QueueItem(
                    kind=item.kind,
                    ssvep_mode=item.ssvep_mode,
                    target_lamp_0idx=int(lamp_list[gi]),
                    target_tile_id=item.target_tile_id,
                    phase=item.phase,
                    ssvep_group_index=item.ssvep_group_index,
                    ssvep_block_in_group=item.ssvep_block_in_group,
                    ssvep_group_size=item.ssvep_group_size,
                )
            )
        self._main_queue = patched

    def _configured_ssvep_sequences_log(self) -> Dict[str, Any]:
        cont = self._ssvep_lamps_for_mode("continuous")
        burst = self._ssvep_lamps_for_mode("burst")
        cont_1 = [int(x) + 1 for x in cont]
        burst_1 = [int(x) + 1 for x in burst]
        return {
            "configured_experiment_sequence_continuous": cont_1,
            "configured_experiment_sequence_burst": burst_1,
            "configured_experiment_sequence_continuous_0idx": list(cont),
            "configured_experiment_sequence_burst_0idx": list(burst),
            "continuous_sequence": cont_1,
            "burst_sequence": burst_1,
            "continuous_sequence_0idx": list(cont),
            "burst_sequence_0idx": list(burst),
            "experiment_target_sequence_continuous": cont_1,
            "experiment_target_sequence_burst": burst_1,
            "experiment_target_sequence_continuous_0idx": list(cont),
            "experiment_target_sequence_burst_0idx": list(burst),
            "displayed_cue_sequence_continuous": cont_1,
            "displayed_cue_sequence_burst": burst_1,
            "continuous_shuffle_seed": self._ssvep_mode_shuffle_seeds.get("continuous"),
            "burst_shuffle_seed": self._ssvep_mode_shuffle_seeds.get("burst"),
            "ssvep_lamp_shuffle_seed": int(self.cfg.ssvep_lamp_shuffle_seed),
            "ssvep_lamp_auto_shuffle": bool(self.cfg.ssvep_auto_shuffle_on_prepare),
            "experiment_target_lamp_sequence_continuous": cont_1,
            "experiment_target_lamp_sequence_burst": burst_1,
        }

    def _total_main_experiments(self) -> int:
        return (
            int(self.cfg.p300_main_trials)
            + int(self.cfg.ssvep_experiments_per_mode) * 2
        )

    def _ssvep_group_display(self, item: Optional[QueueItem]) -> Tuple[int, int, int, int]:
        """(experiment_index 1-based, experiment_total, subblock 1-based, subblocks_per_experiment)."""
        n_exp = max(1, int(self.cfg.ssvep_experiments_per_mode))
        gs = max(1, int(self.cfg.ssvep_group_size))
        if item is None or item.kind != "ssvep" or item.ssvep_group_index is None:
            return (1, n_exp, 1, gs)
        return (
            int(item.ssvep_group_index) + 1,
            n_exp,
            int(item.ssvep_block_in_group or 0) + 1,
            gs,
        )

    def _status_progress_label(self) -> str:
        prefix = f"{self._active_block_label}: " if self._active_block_label else ""
        if self.state == ProtocolState.Main and self._main_queue:
            item = self._current_queue_item()
            if item is not None and item.phase == "calib":
                n_calib = int(self.cfg.p300_calib_trials)
                return (
                    f"{prefix}Калибровка P300 {self._main_queue_index + 1}/{n_calib}"
                )
            return f"{prefix}{self._main_queue_index + 1}/{len(self._main_queue)}"
        if self.state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST):
            item = self._current_queue_item()
            exp_i, exp_n, sub_i, sub_n = self._ssvep_group_display(item)
            sm = _ru_ssvep_stim_mode(
                "continuous" if self.state == ProtocolState.SSVEP_CONT else "burst"
            )
            lamp1 = "?"
            if item is not None and item.target_lamp_0idx is not None:
                lamp1, _ = self._ssvep_lamp_display(int(item.target_lamp_0idx))
            return (
                f"{prefix}ССВП ({sm}) эксп. {exp_i}/{exp_n} — лампа L{lamp1}, "
                f"срез {sub_i}/{sub_n}"
            )
        return prefix.rstrip(": ") or "Протокол"

    @property
    def session_ready(self) -> bool:
        return self.state == ProtocolState.Ready

    @property
    def eeg_inlet_open(self) -> bool:
        return self._inlet_eeg is not None

    @property
    def eeg_stream_live(self) -> bool:
        """True, если в последние ~2.5 с приходили сэмплы ЭЭГ из LSL."""
        if self._inlet_eeg is None:
            return False
        if self._last_eeg_data_wall_at <= 0.0:
            return False
        return (time.time() - self._last_eeg_data_wall_at) < 2.5

    @property
    def block_running(self) -> bool:
        return self.state in (
            ProtocolState.Main,
            ProtocolState.SSVEP_CONT,
            ProtocolState.SSVEP_BURST,
        )

    def _queue_item_at(self, index: Optional[int] = None) -> Optional[QueueItem]:
        """Элемент очереди по индексу (без проверки FSM-состояния)."""
        idx = self._main_queue_index if index is None else int(index)
        if idx < 0 or idx >= len(self._main_queue):
            return None
        return self._main_queue[idx]

    def _current_queue_item(self) -> Optional[QueueItem]:
        if self.state not in (ProtocolState.Main, ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST):
            return None
        return self._queue_item_at()

    def _active_ssvep_freqs_hz(self) -> Tuple[float, ...]:
        active = tuple(float(x) for x in self.cfg.ssvep_freqs_hz if float(x) > 0.0)
        return active if active else tuple(float(x) for x in self.cfg.ssvep_freqs_hz)

    def _ssvep_target_lamp_0idx_for_item(self, item: QueueItem) -> int:
        if item.target_lamp_0idx is not None:
            return int(item.target_lamp_0idx)
        freqs = self._active_ssvep_freqs_hz()
        n_lamps = max(1, len(freqs))
        return int(self._ssvep_block_count_cont + self._ssvep_block_count_burst) % n_lamps

    def _resolve_ssvep_cue_lamp_0idx(
        self, item: Optional[QueueItem], *, event: str
    ) -> Tuple[int, int]:
        """Лампа cue: всегда из queue item (не stale runtime). Returns (lamp_0idx, group_index)."""
        gi = int(item.ssvep_group_index) if item is not None and item.ssvep_group_index is not None else -1
        if item is not None and item.kind == "ssvep" and item.target_lamp_0idx is not None:
            return int(item.target_lamp_0idx), gi
        if item is not None and item.kind == "ssvep":
            return int(self._ssvep_target_lamp_0idx_for_item(item)), gi
        stale = self._ssvep_target_lamp
        if stale is not None:
            plog.warn(f"SSVEP cue {event}: queue item без target_lamp — fallback runtime={stale}")
            return int(stale), gi
        return 0, gi

    def _log_ssvep_cue_overlay_trace(
        self,
        item: Optional[QueueItem],
        *,
        event: str,
        expected_lamp_0idx: int,
        group_index: int,
    ) -> None:
        if self._logger is None:
            return
        seq = self._ssvep_lamps_for_mode(
            str(item.ssvep_mode or "continuous") if item is not None else "continuous"
        )
        seq_1 = [int(x) + 1 for x in seq]
        expected_1 = int(expected_lamp_0idx) + 1
        gi = int(group_index)
        expected_from_seq = (
            int(seq[gi]) + 1 if 0 <= gi < len(seq) else None
        )
        payload: Dict[str, Any] = {
            "event": str(event),
            "group_index": gi,
            "expected_lamp_0idx": int(expected_lamp_0idx),
            "expected_lamp_1based": expected_1,
            "expected_lamp_from_sequence": expected_from_seq,
            "current_runtime_lamp_0idx": self._ssvep_target_lamp,
            "displayed_overlay_lamp_1based": int(self.ssvep_cue_lamp_1based),
            "displayed_overlay_group_index": int(self.ssvep_cue_group_index),
            "queue_index": int(self._main_queue_index),
            "block_in_group": int(item.ssvep_block_in_group or 0) if item else None,
            "mode": str(item.ssvep_mode) if item is not None else None,
            "final_sequence_1based": seq_1,
        }
        if expected_from_seq is not None and int(expected_from_seq) != expected_1:
            payload["sequence_mismatch"] = True
        if int(self.ssvep_cue_lamp_1based) != expected_1:
            payload["overlay_mismatch"] = True
        self._logger.write("ssvep_cue_overlay_trace", payload)
        plog.info(
            f"SSVEP cue trace [{event}]: group={gi} expected=L{expected_1} "
            f"displayed=L{self.ssvep_cue_lamp_1based} runtime={self._ssvep_target_lamp}"
        )

    def _ssvep_lamp_display(self, lamp_0idx: int) -> Tuple[int, float]:
        """Номер лампы для оператора (1-based) и частота."""
        freqs = self._active_ssvep_freqs_hz()
        lamp_1 = int(lamp_0idx) + 1
        hz = float(freqs[lamp_0idx]) if 0 <= int(lamp_0idx) < len(freqs) else 0.0
        return lamp_1, hz

    def _ssvep_cue_preview_seconds(self) -> float:
        """Минимум времени на информационный оверлей перед стартом мигалки."""
        return max(2.0, float(self.cfg.pause_between_experiments_s))

    def _ssvep_group_metadata(self, item: Optional[QueueItem]) -> Dict[str, Any]:
        if item is None or item.kind != "ssvep" or item.ssvep_group_index is None:
            return {}
        lamp = item.target_lamp_0idx
        meta: Dict[str, Any] = {
            "group_index": int(item.ssvep_group_index),
            "group_target": int(lamp) + 1 if lamp is not None else None,
            "group_target_lamp_0idx": int(lamp) if lamp is not None else None,
            "block_in_group": int(item.ssvep_block_in_group or 0),
            "group_size": int(item.ssvep_group_size),
            "experiment_target_lamp_0idx": int(lamp) if lamp is not None else None,
            "experiment_target_lamp_1based": int(lamp) + 1 if lamp is not None else None,
        }
        if self._ssvep_visual_groups_enabled():
            meta["group_visual_continuous"] = True
            meta["continuous_group_runtime"] = True
        return meta

    def _ssvep_visual_groups_enabled(self) -> bool:
        return bool(self.cfg.ssvep_group_visual_continuous)

    def _ssvep_continues_visual_group(self, item: Optional[QueueItem]) -> bool:
        """Следующий элемент очереди — тот же visual group (без паузы/blackout между блоками)."""
        if item is None or not self._ssvep_visual_groups_enabled():
            return False
        nxt_i = int(self._main_queue_index) + 1
        if nxt_i >= len(self._main_queue):
            return False
        nxt = self._main_queue[nxt_i]
        if item.kind != "ssvep" or nxt.kind != "ssvep":
            return False
        if str(item.ssvep_mode or "") != str(nxt.ssvep_mode or ""):
            return False
        if item.target_lamp_0idx != nxt.target_lamp_0idx:
            return False
        if item.ssvep_group_index is None or nxt.ssvep_group_index != item.ssvep_group_index:
            return False
        return int(nxt.ssvep_block_in_group or 0) == int(item.ssvep_block_in_group or 0) + 1

    def _schedule_ssvep_queue_item(self, item: QueueItem, *, after_previous: bool) -> None:
        """Пауза/оверлей: полный cue только при смене группы (block_in_group==0)."""
        mode = str(item.ssvep_mode or "continuous")
        self._ensure_ssvep_mode_sequence_committed(mode)
        item = self._current_queue_item() or item
        lamp0 = self._ssvep_target_lamp_0idx_for_item(item)
        self._ssvep_target_lamp = int(lamp0)
        st = ProtocolState.SSVEP_CONT if mode == "continuous" else ProtocolState.SSVEP_BURST
        label = self._status_progress_label()
        sm = _ru_ssvep_stim_mode(mode)
        if item.ssvep_group_start or not after_previous:
            self._sync_stim_control(
                item,
                message="Смотрите в указанном направлении",
            )
            self._set_ssvep_cue_overlay(
                ssvep_mode=mode, item=item, event="group_start_cue"
            )
            self._start_pause(
                seconds=self._ssvep_cue_preview_seconds(),
                status=f"{label}: смотрите на лампу — ССВП ({sm})",
                next_state=st,
                next_ssvep_mode=mode,
            )
            self._update_participant_instruction(kind="ssvep")
            return
        if after_previous:
            if (
                self._ssvep_visual_groups_enabled()
                and int(item.ssvep_block_in_group or 0) > 0
            ):
                self._set_state(st, detail="ssvep group continuation (scheduled)")
                try:
                    self._start_ssvep_block(mode=mode, lamp_0idx=lamp0, continuation=True)
                except Exception as e:
                    plog.exc("критическая ошибка _start_ssvep_block (continuation)", e)
                    self.status_text = f"ССВП: ошибка продолжения группы: {e}"
                return
            self.ssvep_cue_visible = False
            gap = max(0.0, float(self.cfg.ssvep_inter_block_gap_s))
            if gap > 0:
                self._start_pause(
                    seconds=gap,
                    status=f"{label}: ССВП ({sm}) — следующий блок",
                    next_state=st,
                    next_ssvep_mode=None,
                )
                self._update_participant_instruction(kind="pause", pause_message="", seconds_left=gap)
            else:
                self._set_state(st, detail="ssvep item (no gap)")
                try:
                    self._start_ssvep_block(mode=mode, lamp_0idx=lamp0, continuation=True)
                except Exception as e:
                    plog.exc("критическая ошибка _start_ssvep_block", e)
                    self.status_text = f"ССВП: ошибка запуска блока: {e}"
            return
        self._set_state(st, detail="ssvep item")
        try:
            self._start_ssvep_block(mode=mode, lamp_0idx=lamp0)
        except Exception as e:
            plog.exc("критическая ошибка _start_ssvep_block", e)
            self.status_text = f"ССВП: ошибка запуска блока: {e}"

    def _set_ssvep_cue_overlay(
        self,
        *,
        ssvep_mode: Optional[str],
        item: Optional[QueueItem] = None,
        event: str = "set_overlay",
    ) -> None:
        if ssvep_mode is None:
            self.ssvep_cue_visible = False
            self.ssvep_cue_group_index = -1
            return
        self.ssvep_blackout_visible = False
        cur = item if item is not None else self._current_queue_item()
        lamp0, gi = self._resolve_ssvep_cue_lamp_0idx(cur, event=str(event))
        self._ssvep_target_lamp = int(lamp0)
        lamp1, hz = self._ssvep_lamp_display(lamp0)
        self.ssvep_cue_visible = True
        exp_i, exp_n, _, _ = self._ssvep_group_display(cur)
        self.ssvep_cue_exp_index = int(exp_i)
        self.ssvep_cue_exp_total = int(exp_n)
        self.ssvep_cue_lamp_1based = int(lamp1)
        self.ssvep_cue_freq_hz = float(hz)
        self.ssvep_cue_mode_label = f"ССВП — {_ru_ssvep_stim_mode(str(ssvep_mode))}"
        self.ssvep_cue_group_index = int(gi)
        self._log_ssvep_cue_overlay_trace(
            cur, event=str(event), expected_lamp_0idx=int(lamp0), group_index=int(gi)
        )
        plog.info(
            f"оверлей ССВП: эксп. {self.ssvep_cue_exp_index}/{self.ssvep_cue_exp_total}, "
            f"группа {self.ssvep_cue_group_index}, лампа L{self.ssvep_cue_lamp_1based} "
            f"({self.ssvep_cue_freq_hz:g} Гц)"
        )

    def _start_pause(self, *, seconds: float, status: str, next_state: str, next_ssvep_mode: Optional[str] = None) -> None:
        sec = max(0.0, float(seconds))
        self._pause_until_wall = float(time.time()) + sec
        self._pause_status = str(status)
        self._pause_next_state = str(next_state)
        self._pause_next_ssvep_mode = str(next_ssvep_mode) if next_ssvep_mode is not None else None
        if next_ssvep_mode is not None:
            self._set_ssvep_cue_overlay(
                ssvep_mode=str(next_ssvep_mode),
                event="pause_before_ssvep",
            )
        else:
            self._set_ssvep_cue_overlay(ssvep_mode=None)
        plog.info(
            f"пауза {sec:.1f}s -> next_state={next_state}"
            + (f", ssvep_mode={next_ssvep_mode}" if next_ssvep_mode else "")
        )

    def _tick_pause(self) -> bool:
        """Returns True if we are in pause (and handled it)."""
        if self._pause_until_wall is None:
            return False
        now = float(time.time())
        rem = float(self._pause_until_wall) - now
        if rem > 0:
            base = self._pause_status or "Пауза"
            self.status_text = f"{base} ({rem:.1f} с)"
            self._update_participant_instruction(kind="pause", pause_message=base, seconds_left=rem)
            return True

        # Pause finished: apply scheduled transition once.
        next_state = self._pause_next_state
        next_mode = self._pause_next_ssvep_mode
        self._pause_until_wall = None
        self._pause_status = None
        self._pause_next_state = None
        self._pause_next_ssvep_mode = None
        self._set_ssvep_cue_overlay(ssvep_mode=None)
        if next_state:
            self._set_state(next_state, detail="pause finished")
            if next_state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST):
                item = self._current_queue_item()
                mode_run = str(
                    next_mode
                    or (item.ssvep_mode if item is not None else "continuous")
                )
                plog.info(f"пауза закончилась — запуск SSVEP migalka, mode={mode_run}")
                lamp = (
                    int(item.target_lamp_0idx)
                    if item is not None and item.target_lamp_0idx is not None
                    else self._ssvep_target_lamp
                )
                try:
                    self._start_ssvep_block(mode=str(mode_run), lamp_0idx=lamp)
                except Exception as e:
                    plog.exc("критическая ошибка _start_ssvep_block", e)
                    self.status_text = f"ССВП: ошибка запуска блока: {e}"
            elif next_state == ProtocolState.Main:
                item = self._current_queue_item()
                if item is not None and item.kind == "p300":
                    self._sync_stim_control(item)
                    self.status_text = f"{self._status_progress_label()}: P300 — смотрите на плитку…"
                    self._update_participant_instruction(kind="p300")
        return False

    @property
    def logger(self) -> Optional[ExperimentSessionLogger]:
        return self._logger

    def _session_dir(self) -> Optional[Path]:
        if self._logger is not None:
            return self._logger.session_dir
        if self.cfg.session_dir is not None:
            return Path(self.cfg.session_dir)
        return None

    def _request_stim_p300_trial(
        self,
        *,
        target_tile_id: int,
        experiment_index: int,
        experiment_total: int,
        label: str = "P300",
    ) -> None:
        sd = self._session_dir()
        if sd is None:
            return
        stim_ctl.write_trial_request(
            sd,
            target_tile_id=int(target_tile_id),
            experiment_index=int(experiment_index),
            experiment_total=int(experiment_total),
            label=str(label),
        )
        plog.info(
            f"stim_control trial: tile={int(target_tile_id)} "
            f"({experiment_index}/{experiment_total}) {label!r}"
        )

    def _sync_stim_control(self, item: Optional[QueueItem], *, message: str = "") -> None:
        sd = self._session_dir()
        if sd is None:
            return
        total = int(len(self._main_queue)) if self._main_queue else int(self._total_main_experiments())
        idx = int(self._main_queue_index) + 1
        if item is None:
            stim_ctl.write_stim_done(sd)
            return
        if item.kind == "p300":
            tid = int(item.target_tile_id) if item.target_tile_id is not None else 0
            stim_ctl.write_trial_request(
                sd,
                target_tile_id=tid,
                experiment_index=idx,
                experiment_total=total,
                label="P300",
            )
        else:
            stim_ctl.write_paused(
                sd,
                reason="ssvep",
                message=message or "Смотрите в указанном направлении",
                experiment_index=idx,
                experiment_total=total,
            )

    def _update_participant_instruction(
        self,
        *,
        kind: str,
        pause_message: str = "",
        seconds_left: Optional[float] = None,
    ) -> None:
        total_main = int(len(self._main_queue)) if self._main_queue else 0
        if self._pause_until_wall is not None:
            item_pause = self._current_queue_item()
            if (
                item_pause is not None
                and item_pause.kind == "ssvep"
                and not self.ssvep_cue_visible
                and int(item_pause.ssvep_block_in_group or 0) > 0
            ):
                self.participant_instruction = {
                    "type": "gap",
                    "seconds_left": seconds_left,
                }
                return
            self.participant_instruction = {
                "type": "pause",
                "message": pause_message or self._pause_status or "Пауза",
                "seconds_left": seconds_left,
            }
            return
        item = self._current_queue_item()
        if item is None:
            self.participant_instruction = None
            return
        idx = int(self._main_queue_index) + 1
        if kind == "blackout" or self.ssvep_blackout_visible:
            self.participant_instruction = {"type": "blackout"}
            return
        if item.kind == "p300" and item.phase == "calib":
            n_calib = int(self.cfg.p300_calib_trials)
            self.participant_instruction = {
                "type": "calib",
                "index": int(self._main_queue_index) + 1,
                "total": n_calib,
                "tile": int(self.cfg.calib_target_tile_id),
            }
            return
        if item.kind == "p300":
            tid = int(item.target_tile_id) if item.target_tile_id is not None else 0
            self.participant_instruction = {
                "type": "p300",
                "index": idx,
                "total": total_main,
                "tile": tid,
            }
            return
        if item.kind == "ssvep" and self.ssvep_cue_visible:
            lamp0 = self._ssvep_target_lamp_0idx_for_item(item)
            lamp1, hz = self._ssvep_lamp_display(lamp0)
            exp_i, exp_n, _, _ = self._ssvep_group_display(item)
            self.participant_instruction = {
                "type": "ssvep",
                "index": int(exp_i),
                "total": int(exp_n),
                "lamp_1based": int(lamp1),
                "freq_hz": float(hz),
                "mode_label": f"ССВП — {_ru_ssvep_stim_mode(str(item.ssvep_mode or ''))}",
            }
            return
        self.participant_instruction = None

    def start(self) -> None:
        ch_disp = (
            ",".join(str(c + 1) for c in self.cfg.roi_channels_0idx)
            if self.cfg.roi_channels_0idx
            else "ALL"
        )
        ssvep_ch_disp = (
            ",".join(str(c + 1) for c in self.cfg.ssvep_roi_channels_0idx)
            if self.cfg.ssvep_roi_channels_0idx
            else "ALL"
        )
        plog.info(
            f"=== PROTOCOL START subject={self.cfg.subject_id!r} "
            f"calib={self.cfg.p300_calib_trials} main_P300={self.cfg.p300_main_trials} "
            f"SSVEP={self.cfg.ssvep_experiments_per_mode}x{self.cfg.ssvep_group_size} "
            f"({self._ssvep_physical_blocks()} queue items/mode)x2 "
            f"shuffle_seed={self.cfg.shuffle_seed} "
            f"COM={self.cfg.com_port!r} P300_ch(1-based)={ch_disp} "
            f"SSVEP_ch(1-based)={ssvep_ch_disp} migalka_freqs_Hz={list(self.cfg.ssvep_freqs_hz)} ==="
        )
        self._runtime_resources_released = False
        self._logger_finalized = False
        self._set_state(ProtocolState.Preflight, detail="start()")
        self.status_text = "Предстарт: поиск потоков LSL и COM-порта мигалки…"
        plan = {
            "schema_version": 2,
            "calibration": {
                "p300_trials": int(self.cfg.p300_calib_trials),
                "target_tile_id": int(self.cfg.calib_target_tile_id),
                "template_epochs": int(self.cfg.template_warmup_target_epochs),
            },
            "main": {
                "p300_trials": int(self.cfg.p300_main_trials),
                "ssvep_experiments_per_mode": int(self.cfg.ssvep_experiments_per_mode),
                "ssvep_group_size": int(self.cfg.ssvep_group_size),
                "ssvep_continuous_blocks": int(self._ssvep_physical_blocks()),
                "ssvep_burst_blocks": int(self._ssvep_physical_blocks()),
                **self._configured_ssvep_sequences_log(),
                "shuffle_seed_cfg": int(self.cfg.shuffle_seed),
                "pause_sec": float(self.cfg.pause_between_experiments_s),
            },
            "p300_methods_per_trial": [WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR],
            "ssvep": {
                "block_sec": float(self.cfg.ssvep_block_sec),
                "window_sec": float(self.cfg.ssvep_window_sec),
                "msi_samples": int(self.cfg.ssvep_msi_samples),
                "fs_hz": float(self.cfg.ssvep_fs_hz),
                "freqs_hz": [float(x) for x in self.cfg.ssvep_freqs_hz],
                "roi_channels_0idx": [int(c) for c in self.cfg.ssvep_roi_channels_0idx],
            },
        }
        self._logger = UnifiedExperimentLogger.open_new(
            output_root=Path(self.cfg.output_root),
            subject_id=str(self.cfg.subject_id),
            protocol_plan=plan,
            start_payload={
                "note": "protocol_start",
                **self._configured_ssvep_sequences_log(),
            },
            session_dir=Path(self.cfg.session_dir) if self.cfg.session_dir is not None else None,
        )
        freqs = [float(x) for x in self.cfg.ssvep_freqs_hz]
        session_info = {
            "subject_last_name": str(self.cfg.subject_last_name),
            "subject_first_name": str(self.cfg.subject_first_name),
            "subject_id": str(self.cfg.subject_id),
            "session_id": self._logger.session_id,
            "com_port": str(self.cfg.com_port),
            "eeg_stream_name": str(self.cfg.eeg_stream_name),
            "p300_calib_trials": int(self.cfg.p300_calib_trials),
            "p300_main_trials": int(self.cfg.p300_main_trials),
            "ssvep_experiments_per_mode": int(self.cfg.ssvep_experiments_per_mode),
            "ssvep_group_size": int(self.cfg.ssvep_group_size),
            "ssvep_blocks_per_mode": int(self._ssvep_physical_blocks()),
            "ssvep_continuous_lamps_0idx": list(self.cfg.ssvep_continuous_lamps_0idx),
            "ssvep_burst_lamps_0idx": list(self.cfg.ssvep_burst_lamps_0idx),
            **self._configured_ssvep_sequences_log(),
            "pause_between_experiments_s": float(self.cfg.pause_between_experiments_s),
            "shuffle_seed": int(self.cfg.shuffle_seed),
            "calib_target_tile": int(self.cfg.calib_target_tile_id),
            "p300_channels_1based": ",".join(str(c + 1) for c in self.cfg.roi_channels_0idx),
            "ssvep_channels_1based": ",".join(str(c + 1) for c in self.cfg.ssvep_roi_channels_0idx),
            "msi_fs_hz": float(self.cfg.ssvep_fs_hz),
            "msi_samples": int(self.cfg.ssvep_msi_samples),
            "ssvep_block_sec": float(self.cfg.ssvep_block_sec),
            "p300_roi_0idx": tuple(int(c) for c in self.cfg.roi_channels_0idx),
            "ssvep_roi_0idx": tuple(int(c) for c in self.cfg.ssvep_roi_channels_0idx),
        }
        del session_info  # minimal layout: no analysis/ bundle
        sd = self._session_dir()
        if sd is not None:
            stim_ctl.write_paused(sd, reason="preflight", message="Подготовьтесь")

    def _abort_open_ssvep_block(self, *, reason: str) -> None:
        """Close orphan SSVEP block so start/end events stay paired on abrupt stop."""
        if self._ssvep_block_started_at is None or self._logger is None:
            return
        mode = (
            "continuous"
            if self.state == ProtocolState.SSVEP_CONT
            else "burst"
            if self.state == ProtocolState.SSVEP_BURST
            else "unknown"
        )
        rec = {
            "mode": str(mode),
            "aborted": True,
            "abort_reason": str(reason),
            "target_lamp": int(self._ssvep_target_lamp) if self._ssvep_target_lamp is not None else None,
        }
        self._logger.write("ssvep_block_end", rec)
        plog.warn(f"SSVEP block aborted ({reason}), mode={mode}")
        self._ssvep_block_started_at = None
        self._ssvep_target_lamp = None

    def _close_lsl_inlets(self) -> None:
        lsl.close_lsl_inlet(self._inlet_eeg)
        lsl.close_lsl_inlet(self._inlet_markers)
        lsl.close_lsl_inlet(self._inlet_migalka_markers)
        self._inlet_eeg = None
        self._inlet_markers = None
        self._inlet_migalka_markers = None
        self._p300.set_eeg_inlet(None)
        self._last_eeg_data_wall_at = 0.0

    def _release_runtime_resources(self) -> None:
        """Idempotent: LSL inlets + Migalka COM (main thread only)."""
        if self._runtime_resources_released:
            return
        self._close_lsl_inlets()
        try:
            self._migalka.stop_and_close()
        except Exception as e:
            plog.exc("migalka stop", e)
        self._runtime_resources_released = True

    def _finalize_logger_once(self, stop_payload: Dict[str, Any]) -> Optional[Path]:
        """Idempotent session_stop + manifest (UnifiedExperimentLogger also guards _closed)."""
        if self._logger_finalized or self._logger is None:
            return None
        try:
            out_dir = self._logger.finalize(stop_payload=dict(stop_payload))
        except Exception as e:
            plog.exc("logger finalize", e)
            out_dir = getattr(self._logger, "session_dir", None)
        self._logger = None
        self._logger_finalized = True
        return Path(out_dir) if out_dir is not None else None

    def stop(self, *, reason: str = "user_stop") -> None:
        if self.state == ProtocolState.Stopped and self._runtime_resources_released and self._logger_finalized:
            return
        plog.info(f"=== PROTOCOL STOP reason={reason!r} ===")
        self._drain_migalka_com_events()
        self._pause_until_wall = None
        self._main_queue = []
        self._main_queue_index = 0
        self._sync_stim_control(None)
        self.participant_instruction = None
        self.clear_ssvep_display()
        self._abort_open_ssvep_block(reason=str(reason))
        self._release_runtime_resources()
        self._set_state(ProtocolState.Stopped, detail=reason)
        self.status_text = f"Остановлено: {_ru_stop_reason(reason)}"
        self._finalize_logger_once({"reason": reason})
        self._active_block_label = ""

    def _enter_ready(self) -> None:
        """Блок завершён — ждём выбора оператора (P300 / SSVEP)."""
        assert self._logger is not None
        self._main_queue = []
        self._main_queue_index = 0
        self._pause_until_wall = None
        self._pause_status = None
        self._pause_next_state = None
        self._pause_next_ssvep_mode = None
        self._ssvep_block_started_at = None
        self._ssvep_target_lamp = None
        self._active_block_label = ""
        self.participant_instruction = None
        self.clear_ssvep_display()
        sd = self._logger.session_dir
        try:
            stim_ctl.write_paused(
                sd,
                reason="block_done",
                message="Эксперимент завершён",
            )
        except Exception as e:
            plog.exc("stim_control write_paused(block_done)", e)
        self._set_state(ProtocolState.Ready, detail="block done")
        self.status_text = (
            "Спасибо! Блок завершён. Можно запустить P300, ССВП пакетный или непрерывный."
        )
        plog.info("=== READY: блок завершён, ожидание выбора оператором ===")

    def ensure_marker_inlet(self, *, max_wait_sec: float = 20.0) -> bool:
        """Подключить LSL маркеров плиток (перед блоком P300)."""
        if self._inlet_markers is not None:
            return True
        info_mk, _ = lsl.wait_for_stimulus_marker_stream(max_wait_sec=float(max_wait_sec))
        if info_mk is None:
            return False
        self._inlet_markers = lsl.stream_inlet_with_buffer(info_mk, buffer_seconds=20)
        plog.info(f"маркеры P300 подключены: {info_mk.name()!r}")
        return True

    def start_p300_block(self) -> bool:
        """Калибровка + main P300 подряд (без перемешивания с ССВП)."""
        if self.state != ProtocolState.Ready:
            plog.warn(f"start_p300_block: state={self.state!r}")
            return False
        if not self.ensure_marker_inlet():
            self.status_text = (
                f"Нет потока «{lsl.BCI_STIM_MARKER_STREAM_NAME}». "
                "Запустите PsychoPy (галочка в настройках) и повторите."
            )
            return False
        assert self._logger is not None
        n_calib = int(self.cfg.p300_calib_trials)
        if n_calib > 0:
            self._p300_calib_trials_done = 0
            self._template_locked = False
            self._reset_for_new_p300_block()
        elif not self._template_locked:
            self._reset_for_new_p300_block()
        items, seed_used = build_p300_block_queue(
            p300_calib_trials=n_calib,
            calib_target_tile_id=int(self.cfg.calib_target_tile_id),
            p300_main_trials=int(self.cfg.p300_main_trials),
            shuffle_seed=None if int(self.cfg.shuffle_seed) < 0 else int(self.cfg.shuffle_seed),
        )
        self._shuffle_seed_used = int(seed_used)
        self._start_block_queue(
            items,
            block_label=f"P300 (калибр. {n_calib} + main {self.cfg.p300_main_trials})",
            event_name="p300_block_start",
        )
        return True

    def start_ssvep_block(self, *, mode: str) -> bool:
        """ССВП одного режима: continuous или burst (n блоков из настроек)."""
        if self.state != ProtocolState.Ready:
            plog.warn(f"start_ssvep_block: state={self.state!r}")
            return False
        if str(mode) not in ("continuous", "burst"):
            return False
        assert self._logger is not None
        mode_s = str(mode)
        self._ssvep_committed_lamps.pop(mode_s, None)
        self._ssvep_mode_shuffle_seeds.pop(mode_s, None)
        lamps = self._commit_ssvep_mode_sequence(mode_s, force=True)
        items = build_ssvep_mode_queue(
            mode=str(mode),
            experiment_lamps_0idx=lamps,
            blocks_per_group=int(self.cfg.ssvep_group_size),
        )
        if not items:
            return False
        label = (
            f"ССВП {_ru_ssvep_stim_mode(str(mode))} "
            f"({int(self.cfg.ssvep_experiments_per_mode)} эксп. × {int(self.cfg.ssvep_group_size)} среза)"
        )
        seq_log = self._configured_ssvep_sequences_log()
        self._start_block_queue(
            items,
            block_label=label,
            event_name="ssvep_queue_start",
            shuffle_seed=None,
        )
        self._logger.write(
            "ssvep_queue_configured",
            {
                "mode": str(mode),
                **seq_log,
                "experiment_lamps_0idx": list(lamps),
            },
        )
        return True

    def _start_block_queue(
        self,
        items: List[QueueItem],
        *,
        block_label: str,
        event_name: str,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        assert self._logger is not None
        self._main_queue = list(items)
        self._main_queue_index = 0
        self._active_block_label = str(block_label)
        plan_log: Dict[str, Any] = {
            "block": str(block_label),
            "summary": queue_summary(self._main_queue),
            "queue": [x.to_dict() for x in self._main_queue],
        }
        if shuffle_seed is not None:
            plan_log["shuffle_seed"] = int(shuffle_seed)
        self._logger.write(str(event_name), plan_log)
        self._logger.set_block_label(str(block_label))
        plog.info(f"=== BLOCK START {block_label!r}: {len(self._main_queue)} items ===")
        block_msg = "Калибровка" if "калибр" in str(block_label).lower() else "Подготовьтесь"
        stim_ctl.write_paused(
            self._logger.session_dir,
            reason="block_start",
            message=block_msg,
            experiment_index=1,
            experiment_total=len(self._main_queue),
        )
        self._advance_to_current_queue_item(start_pause=False)

    def tick(self) -> None:
        """One tick: pull streams, update engines, advance FSM."""
        if self.state in (ProtocolState.Idle, ProtocolState.Stopped):
            return
        self._drain_migalka_com_events()
        self._write_tech_snapshot(reason="tick")
        if self.state == ProtocolState.Preflight:
            self._preflight()
            return

        if self.state == ProtocolState.Ready:
            self._pull_lsl()
            # Пауза перед ССВП держит state=ready — обработать countdown и переход.
            if self._tick_pause():
                return
            return

        # Always pull EEG + markers while protocol runs (shared across modes).
        self._pull_lsl()

        # Pause between experiments (do not advance state while pausing).
        if self._tick_pause():
            return

        if self.state == ProtocolState.Main:
            self._run_main_queue_item()
        elif self.state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST):
            mode = "continuous" if self.state == ProtocolState.SSVEP_CONT else "burst"
            self._run_ssvep_blocks(mode=mode)

        # Finalize может быть выставлен внутри _run_ssvep_blocks — обработать в том же tick.
        if self.state == ProtocolState.Finalize:
            self._finalize()
        elif self.state == ProtocolState.Main and self._pause_until_wall is None:
            self._update_participant_instruction(kind="main")
        self._write_tech_snapshot(reason="tick_end")

    def _write_tech_snapshot(self, *, reason: str) -> None:
        lg = self._logger
        if lg is None:
            return
        now = float(time.time())
        if reason in {"tick", "tick_end"} and (now - float(self._tech_last_snapshot_wall)) < float(
            self._tech_snapshot_interval_sec
        ):
            return
        self._tech_last_snapshot_wall = now
        item = self._current_queue_item()
        metrics = self._runtime_metrics_snapshot()
        lg.write_tech(
            "runtime_snapshot",
            {
                "reason": str(reason),
                "state": str(self.state),
                "status_text": str(self.status_text),
                "queue_index": int(self._main_queue_index),
                "queue_len": int(len(self._main_queue)),
                "active_block_label": str(self._active_block_label),
                "pause_active": bool(self._pause_until_wall is not None),
                "pause_rem_s": (
                    float(self._pause_until_wall) - float(now)
                    if self._pause_until_wall is not None
                    else None
                ),
                "eeg_inlet_open": bool(self._inlet_eeg is not None),
                "marker_inlet_open": bool(self._inlet_markers is not None),
                "migalka_marker_inlet_open": bool(self._inlet_migalka_markers is not None),
                "eeg_stream_live": bool(self.eeg_stream_live),
                "migalka_open": bool(self._migalka.is_open()),
                "runtime_resources_released": bool(self._runtime_resources_released),
                "logger_finalized": bool(self._logger_finalized),
                "queue_item": item.to_dict() if item is not None else None,
                "timers": {
                    "ssvep_block_started_at": float(self._ssvep_block_started_at)
                    if self._ssvep_block_started_at is not None
                    else None,
                    "last_eeg_data_wall_at": float(self._last_eeg_data_wall_at)
                    if self._last_eeg_data_wall_at > 0
                    else None,
                },
                **metrics,
            },
        )

    def _preflight(self) -> None:
        assert self._logger is not None
        eeg_streams = lsl.discover_eeg_streams(timeout=1.5)
        info_mk, marker_streams = lsl.wait_for_stimulus_marker_stream(max_wait_sec=3.0)
        info_eeg = lsl.select_eeg_stream(
            eeg_streams,
            name=str(self.cfg.eeg_stream_name),
            session_id=str(self.cfg.eeg_stream_session_id),
        )
        if not eeg_streams:
            plog.error("Preflight: LSL EEG не найден")
            self.status_text = "Предстарт: не найден ни один поток LSL EEG (проверьте запись ЭЭГ)."
            return
        if info_eeg is None:
            want = self.cfg.eeg_stream_name or "?"
            plog.error(f"Preflight: выбранный поток ЭЭГ не найден: {want!r}")
            self.status_text = f"Предстарт: поток ЭЭГ «{want}» не найден. Нажмите «Обновить» в GUI и выберите поток."
            return
        if info_mk is None:
            names = []
            for s in marker_streams:
                try:
                    names.append(str(s.name() or "?"))
                except Exception:
                    names.append("?")
            plog.warn(
                f"Preflight: нет {lsl.BCI_STIM_MARKER_STREAM_NAME!r} (Markers: {names}). "
                "Блок P300 запустит стимулятор отдельно."
            )
            mk_name = "—"
        else:
            try:
                mk_name = info_mk.name()
            except Exception:
                mk_name = "?"
        try:
            eeg_name = info_eeg.name()
        except Exception:
            eeg_name = "?"
        try:
            fs_raw = float(info_eeg.nominal_srate() or 0.0)
        except Exception:
            fs_raw = 0.0
        self._eeg_nominal_fs_hz = fs_raw if fs_raw > 1.0 else float(self.cfg.ssvep_fs_hz)
        plog.info(
            f"Preflight OK: EEG={eeg_name!r} fs={self._eeg_nominal_fs_hz:g} Hz, "
            f"Markers={mk_name!r}, COM={self.cfg.com_port!r}"
        )
        self._inlet_eeg = lsl.stream_inlet_with_buffer(info_eeg, buffer_seconds=20)
        self._inlet_markers = (
            lsl.stream_inlet_with_buffer(info_mk, buffer_seconds=20) if info_mk is not None else None
        )
        info_mig = lsl.select_migalka_marker_stream(marker_streams)
        self._inlet_migalka_markers = (
            lsl.stream_inlet_with_buffer(info_mig, buffer_seconds=20) if info_mig is not None else None
        )
        self._p300.set_eeg_inlet(self._inlet_eeg)
        if info_mig is not None:
            try:
                plog.info(f"Preflight: Migalka LSL={info_mig.name()!r} (пакетный SSVEP, как ssvep_analyzer)")
            except Exception:
                plog.info("Preflight: Migalka LSL подключён (пакетный SSVEP)")
        else:
            plog.warn("Preflight: MigalkaStimMarkers не найден — burst-gate только с COM-событий")
        try:
            n_ch = max(1, int(info_eeg.channel_count()))
            labels = lsl.stream_channel_labels(info_eeg, n_ch)
            self._logger.set_eeg_channel_labels(labels)
        except Exception:
            pass
        self._logger.write(
            "preflight_ok",
            {
                "eeg_stream": {"name": getattr(info_eeg, "name", lambda: "")(), "type": getattr(info_eeg, "type", lambda: "")()},
                "markers_stream": {"name": getattr(info_mk, "name", lambda: "")(), "type": getattr(info_mk, "type", lambda: "")()},
                "com_port": str(self.cfg.com_port),
                "eeg_nominal_fs_hz": float(self._eeg_nominal_fs_hz),
            },
        )

        self._reset_for_new_p300_block()
        self._enter_ready()

    def _pull_lsl(self) -> None:
        if self._inlet_eeg is None or self._logger is None:
            return
        now_lc = float(lsl_local_clock())

        # markers (только если inlet подключён — блок P300)
        marker_chunk: list = []
        marker_ts: list = []
        if self._inlet_markers is not None:
            try:
                marker_chunk, marker_ts = self._inlet_markers.pull_chunk(
                    timeout=0.0, max_samples=MARKERS_PULL_MAX_SAMPLES
                )
            except TypeError:
                marker_chunk, marker_ts = self._inlet_markers.pull_chunk(timeout=0.0)
        if marker_ts:
            self._logger.write("markers_chunk", {"n": int(len(marker_ts))})
            self._logger.write_tech(
                "markers_chunk",
                {
                    "n": int(len(marker_ts)),
                    "ts_first": float(marker_ts[0]) if marker_ts else None,
                    "ts_last": float(marker_ts[-1]) if marker_ts else None,
                    "state": str(self.state),
                },
            )
            for sample, mk_ts in zip(marker_chunk, marker_ts):
                tid_log = parse_trial_target_tile_id(sample)
                if tid_log is None:
                    from p300_analysis.marker_parsing import marker_value_to_stim_key, stim_key_to_tile_digit

                    sk = marker_value_to_stim_key(sample)
                    if sk is not None:
                        tid_log = stim_key_to_tile_digit(sk)
                self._logger.record_marker(
                    lsl_time=float(mk_ts),
                    sample=sample,
                    tile_id="" if tid_log is None else int(tid_log),
                )
                if parse_trial_end(sample):
                    self._p300_trial_end_seen = True
                    plog.event("LSL trial_end", sample=str(sample)[:80])
                tid = parse_trial_target_tile_id(sample)
                if tid is not None:
                    self._p300_trial_armed = True
                    self._p300_trial_end_seen = False
                    plog.event("LSL trial_start", target=int(tid), sample=str(sample)[:80])
                    self._logger.write("protocol_trial_start_arm", {"cue_target_tile_id": int(tid)})
                    item = self._current_queue_item()
                    ph = item.phase if item is not None else "main"
                    self._logger.mark_experiment_start(
                        kind="p300",
                        phase=ph,
                        cue_target_tile_id=int(tid),
                        t_start_lsl=float(mk_ts),
                    )
                    mode_lbl = "P300, калибровка" if ph == "calib" else "P300, AUC + шаблон"
                    self.status_text = f"{self._status_progress_label()}: {mode_lbl}, целевая плитка={int(tid)}"
            if self.state == ProtocolState.Main:
                self._p300.ingest_marker_chunk(
                    marker_chunk=marker_chunk, marker_ts=marker_ts, lsl_local_clock_now=now_lc
                )

        # EEG
        try:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(timeout=0.0, max_samples=EEG_PULL_MAX_SAMPLES)
        except TypeError:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(timeout=0.0)
        if eeg_ts:
            self._last_eeg_data_wall_at = time.time()
            arr = np.asarray(eeg_chunk, dtype=np.float64)
            self._logger.append_eeg_chunk(eeg_ts, arr, stream="EEG", lsl_clock_now=now_lc)
            self._p300.ingest_eeg_chunk(eeg_chunk=arr, eeg_ts=eeg_ts, lsl_local_clock_now=now_lc)
            # SSVEP: как ssvep_analyzer._push_eeg_chunk (append_chunk_timestamps + LSL-якорь)
            self._ssvep.ingest_eeg_chunk(times=eeg_ts, samples=arr, lsl_clock_now=now_lc)
            self._maybe_arm_ssvep_burst_block_start()
            self._logger.write_tech(
                "ingest_pipeline",
                {
                    "state": str(self.state),
                    "eeg_samples_ingested": int(arr.shape[0]) if arr.ndim == 2 else int(arr.size),
                    "marker_pending_after_ingest": int(len(self._p300.pending_markers)),
                    "ssvep_buf_after_ingest": int(self._ssvep._buf.shape[0]),
                },
            )

        # В интегрированном протоколе используем COM-события Migalka для burst-gate.
        # Pull LSL MigalkaStimMarkers в главном tick может подвисать на "битом"
        # stream (Connection refused) и блокировать FSM целиком.
        # LSL inlet оставляем только как диагностический канал, без участия в тике.

        if self.state == ProtocolState.Main:
            extracted, extracted_meta = self._p300.extract_ready_epochs()
        else:
            extracted, extracted_meta = 0, []
        if extracted and self._logger is not None:
            self._logger.write("p300_epochs_extracted", {"n": int(extracted)})
            counts = {k: len(v) for k, v in self._p300.epochs_data.items()}
            for meta in extracted_meta:
                self._logger.append_epochs_log(
                    lsl_time=float(meta["lsl_time"]),
                    marker=str(meta.get("marker") or ""),
                    stim_key=str(meta.get("stim_key") or ""),
                    n_extracted=1,
                    epochs_counts=counts,
                )
            if not self._template_locked:
                if self._p300.try_build_external_template_from_epochs(
                    min_epochs=int(self.cfg.template_warmup_target_epochs)
                ):
                    self._logger.write(
                        "p300_template_ready",
                        {
                            "target_tile_id": int(self._p300.external_template_target_id)
                            if self._p300.external_template_target_id is not None
                            else None,
                            "n_epochs": int(self.cfg.template_warmup_target_epochs),
                            "phase": "calib",
                        },
                    )

    def _reset_for_new_p300_block(self) -> None:
        prof = P300EngineParams(
            baseline_ms=int(_HYBRID_P300_PROFILE.baseline_ms),
            window_x_ms=int(_HYBRID_P300_PROFILE.window_x_ms),
            window_y_ms=int(_HYBRID_P300_PROFILE.window_y_ms),
            fs_hz=float(self._eeg_nominal_fs_hz or 0.0),
            artifact_threshold_uv=60.0,
            use_car=False,
            roi_channels_0idx=tuple(int(c) for c in self.cfg.roi_channels_0idx),
        )
        self._p300.reset(params=prof)
        plog.info(
            f"P300 engine: roi_channels_0idx={prof.roi_channels_0idx or 'ALL'}, "
            f"window={prof.window_x_ms}-{prof.window_y_ms}ms, fs={prof.fs_hz:g} Hz"
        )
        self._p300_trial_armed = False
        self._p300_trial_end_seen = False

    def _p300_template_ready(self) -> bool:
        return self._p300.external_template_window is not None

    def _finish_p300_trial_armed(self) -> bool:
        """True если trial можно закрыть.

        В протоколе-объединителе закрываем P300 trial строго по marker `trial_end`,
        чтобы всегда отрабатывать полные `sequences` из trial_config (например 12),
        а не завершать досрочно по can_decide (решение — после trial_end + drain).
        """
        if not self._p300_trial_armed:
            return False
        if self._p300.current_cue_target_id is None:
            return False
        return bool(self._p300_trial_end_seen)

    def _p300_tile_label(self, tile_id: Optional[int]) -> str:
        if tile_id is None:
            return "—"
        return str(int(tile_id))

    def _p300_correct_label(self, ok: Optional[bool]) -> str:
        if ok is True:
            return "верно"
        if ok is False:
            return "неверно"
        return "нет решения"

    def _log_p300_per_tile_metrics(self, method_label: str, dec_dict: Dict[str, Any], cue_tid: int) -> None:
        dbg = dec_dict.get("debug") if isinstance(dec_dict.get("debug"), dict) else {}
        keys = dbg.get("stim_keys") or []
        vals = dbg.get("final_metric_values")
        if not keys or vals is None:
            return
        plog.info(f"  {method_label} — метрики по плиткам:")
        for k, v in zip(keys, vals):
            tid = stim_key_to_tile_digit(str(k))
            mark = " ← цель" if tid is not None and int(tid) == int(cue_tid) else ""
            plog.info(f"    плитка {tid}: {float(v):.4f}{mark}")

    def _log_p300_experiment_result(
        self,
        *,
        phase: str,
        cue_tid: int,
        planned_tid: int,
        auc_d: Dict[str, Any],
        tpl_d: Dict[str, Any],
        agree: bool,
        queue_index: Optional[int] = None,
    ) -> None:
        phase_ru = {"calib": "калибровка", "main": "основной блок"}.get(str(phase), str(phase))
        exp_tag = self._status_progress_label() if phase == "main" else f"Калибровка P300"
        plog.info(f"════ Итог P300 ({phase_ru}, {exp_tag}) ════")
        plog.info(f"Испытуемый смотрел на плитку: {cue_tid}")
        if planned_tid >= 0 and int(planned_tid) != int(cue_tid):
            plog.info(f"Плановая плитка в очереди: {int(planned_tid)}")
        auc_w = auc_d.get("winner_tile_id")
        tpl_w = tpl_d.get("winner_tile_id")
        auc_ok = auc_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None
        tpl_ok = tpl_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None
        plog.info(
            f"AUC → плитка {self._p300_tile_label(auc_w)} ({self._p300_correct_label(auc_ok)})"
        )
        plog.info(
            f"Шаблон → плитка {self._p300_tile_label(tpl_w)} ({self._p300_correct_label(tpl_ok)})"
        )
        plog.info(f"Методы совпали: {'да' if agree else 'нет'}")
        if cue_tid >= 0:
            self._log_p300_per_tile_metrics("AUC", auc_d, cue_tid)
            self._log_p300_per_tile_metrics("Шаблон", tpl_d, cue_tid)
        if queue_index is not None:
            plog.info(f"Индекс в очереди: {int(queue_index)}")

    def _emit_ssvep_msi_trace_point(
        self,
        dec: SSVEPDecision,
        *,
        mode: str,
        elapsed_s: Optional[float],
        final: bool = False,
    ) -> None:
        """Throttled live MSI trace → msi_trace.ndjson + tech_trace (observability only)."""
        if self._logger is None:
            return
        pt = self._ssvep_msi_trace.record(
            dec,
            wall_t=float(time.time()),
            elapsed_s=elapsed_s,
            target_lamp_0idx=self._ssvep_target_lamp,
            mode=str(mode),
            force=bool(final),
        )
        if pt is None:
            return
        exp_n = int(self._logger._n_experiments) + 1
        self._logger.append_msi_decision(pt, experiment_number=exp_n)
        self._logger.write_tech(
            "msi_decision",
            {**pt, "experiment_number": exp_n, "final": bool(final)},
        )

    def _log_ssvep_msi_result(
        self,
        *,
        dec: SSVEPDecision,
        mode: str,
        target_lamp_0idx: Optional[int],
        elapsed_s: Optional[float] = None,
        final: bool = False,
    ) -> None:
        freqs = list(self._active_ssvep_freqs_hz())
        tag = "итог блока" if final else "MSI (окно)"
        elapsed_sfx = f", t={float(elapsed_s):.1f} с" if elapsed_s is not None else ""
        plog.info(f"──── ССВП {_ru_ssvep_stim_mode(mode)} — {tag}{elapsed_sfx} ────")
        if target_lamp_0idx is not None:
            plog.info(
                f"Целевая лампа: L{int(target_lamp_0idx) + 1} "
                f"({freqs[int(target_lamp_0idx)]:.3f} Гц)"
                if 0 <= int(target_lamp_0idx) < len(freqs)
                else f"Целевая лампа (0-based): {int(target_lamp_0idx)}"
            )
        if not dec.classify_allowed:
            plog.info(f"MSI: классификация недоступна — {dec.debug}")
            return
        w1 = dec.winner_1based
        w0 = dec.winner_0idx
        ok = (
            int(w0) == int(target_lamp_0idx)
            if w0 is not None and target_lamp_0idx is not None
            else None
        )
        plog.info(
            f"MSI победитель: L{w1} (индекс {w0}) — {self._p300_correct_label(ok)}"
            if w1 is not None
            else "MSI победитель: не определён"
        )
        if dec.coef:
            plog.info("MSI Coef по лампам:")
            for i, c in enumerate(dec.coef):
                if c is None:
                    continue
                freq = float(freqs[i]) if i < len(freqs) else 0.0
                mark = ""
                if target_lamp_0idx is not None and int(i) == int(target_lamp_0idx):
                    mark = " ← цель"
                if w0 is not None and int(i) == int(w0):
                    mark += " ★выбор MSI"
                plog.info(f"  L{i + 1} ({freq:.3f} Гц): {float(c):.6f}{mark}")
        dbg = dec.debug or {}
        if dbg:
            plog.info(f"MSI debug: {dbg}")

    def _decision_to_dict(self, decision: P300Decision, *, method: str) -> Dict[str, Any]:
        winner_tile = (
            stim_key_to_tile_digit(str(decision.winner_key))
            if decision.winner_key
            else None
        )
        dbg = decision.debug or {}
        return {
            "method": str(method),
            "can_decide": bool(decision.can_decide),
            "winner_key": decision.winner_key,
            "winner_tile_id": winner_tile,
            "mode_used": decision.mode_used,
            "min_epochs_per_class": int(decision.min_epochs_per_class),
            "can_decide_after_epochs": bool(decision.can_decide),
            "decision_epoch_count": int(decision.min_epochs_per_class),
            "min_per_class": dbg.get("min_per_class"),
            "max_per_class": dbg.get("max_per_class"),
            "total_epochs": dbg.get("total_epochs"),
            "epochs_used_for_decision": dbg.get("epochs_used_for_decision"),
            "epoch_count_at_decision": dbg.get("epoch_count_at_decision"),
            "debug": decision.debug,
        }

    def _p300_finalize_epochs_before_decision(self) -> int:
        """Drain pending markers and extract any remaining epochs after trial_end."""
        drained = int(self._p300.drain_pending_epochs())
        if drained and self._logger is not None:
            self._logger.write(
                "p300_epochs_drained_before_decision",
                {
                    "n": int(drained),
                    "pending_markers_left": int(len(self._p300.pending_markers)),
                    "epoch_counts_by_stim": {k: len(v) for k, v in self._p300.epochs_data.items()},
                },
            )
        return drained

    def _record_dual_p300_experiment(self, *, phase: str, queue_index: Optional[int] = None) -> None:
        assert self._logger is not None
        cue_raw = self._p300.current_cue_target_id
        cue_tid = int(cue_raw) if cue_raw is not None else -1
        item = self._current_queue_item()
        planned_tid = (
            int(item.target_tile_id)
            if item is not None and item.target_tile_id is not None
            else cue_tid
        )
        record_tid = cue_tid
        if cue_tid < 0 and planned_tid >= 0:
            plog.warn(
                f"P300: LSL cue отсутствует, target из очереди: {int(planned_tid)} "
                f"(phase={phase}, queue_index={queue_index})"
            )
            record_tid = int(planned_tid)
        elif cue_tid < 0:
            plog.warn(
                f"P300: target неизвестен (нет LSL cue и плана в очереди), "
                f"phase={phase}, queue_index={queue_index}"
            )
        self._p300_finalize_epochs_before_decision()
        dec_auc = self._p300.compute_decision(winner_mode=WINNER_MODE_AUC)
        if not dec_auc.can_decide and self._p300_trial_end_seen:
            dec_auc = P300Decision(
                can_decide=False,
                min_epochs_per_class=int(dec_auc.min_epochs_per_class),
                winner_idx=None,
                winner_key=None,
                mode_used="no_decision",
                debug={**(dec_auc.debug or {}), "note": "trial_end_without_decision"},
            )
        dec_tpl = self._p300.compute_decision(winner_mode=WINNER_MODE_TEMPLATE_CORR)
        if not dec_tpl.can_decide and self._p300_trial_end_seen:
            dec_tpl = P300Decision(
                can_decide=False,
                min_epochs_per_class=int(dec_tpl.min_epochs_per_class),
                winner_idx=None,
                winner_key=None,
                mode_used="no_decision",
                debug={**(dec_tpl.debug or {}), "note": "trial_end_without_decision"},
            )

        auc_d = self._decision_to_dict(dec_auc, method=WINNER_MODE_AUC)
        tpl_d = self._decision_to_dict(dec_tpl, method=WINNER_MODE_TEMPLATE_CORR)
        agree = (
            auc_d.get("winner_tile_id") is not None
            and auc_d.get("winner_tile_id") == tpl_d.get("winner_tile_id")
        )
        exp_rec = {
            "phase": str(phase),
            "kind": "p300",
            "queue_index": queue_index,
            "cue_target_tile_id": record_tid,
            "cue_target_tile_id_lsl": cue_tid if cue_tid >= 0 else None,
            "planned_target_tile_id": int(planned_tid) if planned_tid >= 0 else None,
            "ground_truth_tile_id": int(planned_tid) if planned_tid >= 0 else None,
            "auc": auc_d,
            "template_corr": tpl_d,
            "methods_agree": bool(agree),
            "correct_auc": auc_d.get("winner_tile_id") == record_tid if record_tid >= 0 else None,
            "correct_template": tpl_d.get("winner_tile_id") == record_tid
            if record_tid >= 0
            else None,
            "epoch_counts_by_stim": {k: len(v) for k, v in self._p300.epochs_data.items()},
            "trial_end_seen": bool(self._p300_trial_end_seen),
            "template_target_id": int(self._p300.external_template_target_id)
            if self._p300.external_template_target_id is not None
            else None,
        }
        p300_avg = build_p300_averages_export(self._p300)
        if p300_avg is not None:
            exp_rec["_p300_averages_export"] = p300_avg
        self._logger.append_experiment(exp_rec)
        legacy = {
            "phase": phase,
            "cue_target_tile_id": record_tid,
            "auc": auc_d,
            "template_corr": tpl_d,
            "methods_agree": agree,
        }
        self._logger.append_p300_trial(legacy)
        self._logger.write("trial_decision_dual", legacy)
        self._log_p300_experiment_result(
            phase=str(phase),
            cue_tid=int(record_tid),
            planned_tid=int(planned_tid),
            auc_d=auc_d,
            tpl_d=tpl_d,
            agree=bool(agree),
            queue_index=queue_index,
        )
        self._p300.reset(params=self._p300.params)
        self._p300_trial_armed = False
        self._p300_trial_end_seen = False

    def _run_p300_calib(self) -> None:
        assert self._logger is not None
        if not self._p300_trial_armed:
            self.status_text = f"{self._status_progress_label()}: ждём trial_start (калибровка)…"
            return
        cue = self._p300.current_cue_target_id
        if cue is None:
            self.status_text = f"{self._status_progress_label()}: нет target в маркере…"
            return
        key = f"стимул_{int(cue)}"
        n = len(self._p300.epochs_data.get(key, []))
        self.status_text = (
            f"{self._status_progress_label()}: плитка {cue}, эпох {n}/"
            f"{self.cfg.template_warmup_target_epochs}"
        )
        if not self._finish_p300_trial_armed():
            return

        self._record_dual_p300_experiment(phase="calib")
        self._p300_calib_trials_done += 1
        template_ok = self._p300_template_ready()
        calib_done = (
            template_ok
            or int(self._p300_calib_trials_done) >= int(self.cfg.p300_calib_trials)
        )
        if not calib_done:
            nxt = int(self._p300_calib_trials_done) + 1
            self._request_stim_p300_trial(
                target_tile_id=int(self.cfg.calib_target_tile_id),
                experiment_index=nxt,
                experiment_total=int(self.cfg.p300_calib_trials),
                label="Калибровка P300",
            )
            self.status_text = (
                f"Калибровка: прогон {self._p300_calib_trials_done}/"
                f"{self.cfg.p300_calib_trials}, шаблон={'да' if template_ok else 'нет'} — "
                f"следующий прогон…"
            )
            return

        if not template_ok:
            plog.warn("калибровка завершена по числу trial, шаблон не собран — template_corr может быть слабым")
        self._template_locked = True
        self._logger.write(
            "calibration_done",
            {
                "trials": int(self._p300_calib_trials_done),
                "template_ready": bool(template_ok),
                "target_tile_id": int(self._p300.external_template_target_id or self.cfg.calib_target_tile_id),
            },
        )
        self._begin_main_phase()

    def _begin_main_phase(self) -> None:
        assert self._logger is not None
        self._ssvep_committed_lamps.clear()
        self._ssvep_mode_shuffle_seeds.clear()
        n_lamps = max(1, len(self._active_ssvep_freqs_hz()))
        n_exp = int(self.cfg.ssvep_experiments_per_mode)
        placeholder = default_ssvep_experiment_lamps(n_exp, n_lamps)
        seed_cfg = int(self.cfg.shuffle_seed)
        seed_arg: Optional[int] = None if seed_cfg < 0 else seed_cfg
        main_items, self._shuffle_seed_used = build_main_queue(
            p300_trials=int(self.cfg.p300_main_trials),
            ssvep_continuous_lamps=placeholder,
            ssvep_burst_lamps=placeholder,
            shuffle_seed=seed_arg,
            blocks_per_group=int(self.cfg.ssvep_group_size),
        )
        n_calib = int(self.cfg.p300_calib_trials)
        calib_items = [
            QueueItem(kind="p300", target_tile_id=int(self.cfg.calib_target_tile_id), phase="calib")
            for _ in range(n_calib)
        ]
        self._main_queue = calib_items + main_items
        self._main_queue_index = 0
        self._set_state(ProtocolState.Main, detail="calibration done")
        plan_log = {
            "shuffle_seed": int(self._shuffle_seed_used or 0),
            "summary": queue_summary(self._main_queue),
            "queue": [x.to_dict() for x in self._main_queue],
            **self._configured_ssvep_sequences_log(),
        }
        self._logger.write("main_queue_ready", plan_log)
        plog.info(f"=== MAIN очередь ({len(self._main_queue)}): seed={self._shuffle_seed_used} ===")
        stim_ctl.write_paused(
            self._logger.session_dir,
            reason="main_start",
            message="Ожидание",
            experiment_index=1,
            experiment_total=len(self._main_queue),
        )
        self._advance_to_current_queue_item(start_pause=False)

    def _advance_to_current_queue_item(self, *, start_pause: bool) -> None:
        """Перейти к текущему элементу очереди (пауза или немедленный старт SSVEP)."""
        item = self._queue_item_at()
        if item is None:
            self._enter_ready()
            return
        label = self._status_progress_label()
        if item.kind == "p300":
            if start_pause:
                stim_ctl.write_paused(
                    self._logger.session_dir,
                    reason="pause_before_p300",
                    message="Смотрите на указанную плитку",
                    experiment_index=int(self._main_queue_index) + 1,
                    experiment_total=len(self._main_queue),
                )
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"{label}: пауза перед P300",
                    next_state=ProtocolState.Main,
                )
                self._update_participant_instruction(kind="pause", pause_message="Подготовьтесь")
            else:
                self._set_state(ProtocolState.Main, detail="p300 item")
                self._sync_stim_control(item)
                self.status_text = f"{label}: P300 — смотрите на плитку {int(item.target_tile_id or 0)}…"
                self._update_participant_instruction(kind="p300")
            return
        self._schedule_ssvep_queue_item(item, after_previous=start_pause)

    def _complete_main_queue_item(self) -> None:
        self._main_queue_index += 1
        if self._main_queue_index >= len(self._main_queue):
            plog.info(f"=== BLOCK DONE {self._active_block_label!r} ===")
            try:
                self._migalka.stop_and_close()
            except Exception:
                pass
            self._enter_ready()
            return
        self._advance_to_current_queue_item(start_pause=True)

    def _run_main_queue_item(self) -> None:
        item = self._current_queue_item()
        if item is None:
            self._set_state(ProtocolState.Finalize, detail="empty queue")
            return
        if item.kind != "p300":
            return
        assert self._logger is not None
        if not self._p300_trial_armed:
            self.status_text = f"{self._status_progress_label()}: P300 — ждём trial_start…"
            return
        if self._p300.current_cue_target_id is None:
            self.status_text = f"{self._status_progress_label()}: P300 — нет target…"
            return
        if not self._finish_p300_trial_armed():
            self.status_text = (
                f"{self._status_progress_label()}: P300 — накопление эпох…"
            )
            return
        phase = item.phase if item is not None else "main"
        self._record_dual_p300_experiment(
            phase=phase,
            queue_index=int(self._main_queue_index),
        )
        if phase == "calib":
            self._p300_calib_trials_done += 1
            n_calib = int(self.cfg.p300_calib_trials)
            template_ok = self._p300_template_ready()
            calib_done = template_ok or int(self._p300_calib_trials_done) >= n_calib
            if calib_done and not self._template_locked:
                if not template_ok:
                    plog.warn("калибровка завершена по числу trial, шаблон не собран")
                self._template_locked = True
                self._logger.write(
                    "calibration_done",
                    {
                        "trials": int(self._p300_calib_trials_done),
                        "template_ready": bool(template_ok),
                        "target_tile_id": int(
                            self._p300.external_template_target_id or self.cfg.calib_target_tile_id
                        ),
                    },
                )
        else:
            self._p300_main_trials_done += 1
        self._complete_main_queue_item()

    def _pull_migalka_lsl_markers(self) -> None:
        """Как ssvep_analyzer._pull_markers → burst_gate (LSL MigalkaStimMarkers)."""
        if self._inlet_migalka_markers is None:
            return
        try:
            try:
                chunk, stamps = self._inlet_migalka_markers.pull_chunk(timeout=0.0, max_samples=256)
            except TypeError:
                chunk, stamps = self._inlet_migalka_markers.pull_chunk(timeout=0.0)
        except Exception as e:
            # При отвале MigalkaStimMarkers не держим "битый" inlet:
            # в burst автоматически включится fallback через COM-события
            # (COM burst events via _drain_migalka_com_events when LSL inlet unavailable).
            plog.warn(f"Migalka LSL pull failed, fallback to COM burst-gate: {e}")
            self._inlet_migalka_markers = None
            return
        for sample, ts in zip(chunk, stamps):
            try:
                val = sample[0] if sample else ""
            except (TypeError, IndexError):
                val = sample
            self._ssvep.ingest_migalka_marker(lsl_time=float(ts), value=val)

    def _drain_migalka_com_events(self) -> None:
        """Drain COM lamp events on the runtime owner thread (GUI tick / stop)."""
        events, stats = self._migalka.drain_pending_events()
        if stats.get("batch"):
            self._migalka_drain_last_stats = dict(stats)
        for ev in events:
            self._dispatch_migalka_event(ev)

    def _ssvep_burst_wall_limit_sec(self) -> float:
        """Wall safety cap: global ON budget + inter-burst pauses (not per-lamp × N)."""
        budget = float(self.cfg.ssvep_block_sec)
        return budget + float(self.cfg.ssvep_burst_pause_margin_sec)

    def _process_migalka_burst_status(self) -> None:
        for stats in self._migalka.drain_burst_status():
            self._ssvep_burst_stim_stats = dict(stats)
            self._ssvep_burst_device_done = True
            util = stats.get("stim_utilization")
            if util is not None:
                plog.info(
                    f"BURST DONE: wall={stats.get('wall_ms')}ms "
                    f"total_on={stats.get('total_on_ms')} "
                    f"target={stats.get('global_target_on_ms')} "
                    f"pause={stats.get('pause_ms')}ms util={float(util):.3f}"
                )
            else:
                plog.info(f"BURST DONE от мигалки: {stats}")
            if self._logger is not None:
                self._logger.write("migalka_burst_complete", dict(stats))
                self._logger.write_tech("migalka_burst_complete", dict(stats))

    def _maybe_arm_ssvep_burst_block_start(self) -> None:
        if not self._ssvep_burst_block_start_pending:
            return
        t_buf = self._ssvep.latest_buffer_lsl_time()
        if t_buf is None:
            return
        self._ssvep.set_burst_block_start(t_buf)
        self._ssvep_burst_block_start_pending = False

    def _dispatch_migalka_event(self, ev: LampEvent) -> None:
        """Main thread only: logger + SSVEP burst gate (never call from serial reader)."""
        lamp, is_on, raw = ev
        if self._logger is not None:
            self._logger.write("migalka_led", {"lamp": int(lamp), "on": bool(is_on), "raw": str(raw)})
        if self.state == ProtocolState.SSVEP_BURST:
            t_mark = self._ssvep.latest_buffer_lsl_time()
            if t_mark is None:
                t_mark = float(lsl_local_clock())
            self._ssvep.ingest_migalka_marker(
                lsl_time=float(t_mark),
                value=f"{100 + int(lamp)}|{'on' if is_on else 'off'}",
            )

    def _start_ssvep_block(
        self, *, mode: str, lamp_0idx: Optional[int] = None, continuation: bool = False
    ) -> None:
        assert self._logger is not None
        plog.info(
            f"=== SSVEP block START mode={mode!r} COM={self.cfg.com_port!r} "
            f"continuation={bool(continuation)} ==="
        )
        active_freqs = self._active_ssvep_freqs_hz()
        n_lamps = max(1, len(active_freqs))
        fs_hz = float(self._eeg_nominal_fs_hz or self.cfg.ssvep_fs_hz)
        # Use the intended window in seconds (from GUI: samples/configured_fs) and actual hardware fs.
        # This way a 3 s window at 2000 Hz gives 6000 samples, not the hardcoded 15000.
        win_sec = float(self.cfg.ssvep_window_sec) if self.cfg.ssvep_window_sec > 0 else (
            float(self.cfg.ssvep_msi_samples) / float(self.cfg.ssvep_fs_hz)
            if self.cfg.ssvep_msi_samples > 0 and self.cfg.ssvep_fs_hz > 0
            else 3.0
        )
        msi_n = int(round(fs_hz * win_sec)) if fs_hz > 0 and win_sec > 0 else int(self.cfg.ssvep_msi_samples)
        plog.info(f"SSVEP msi_n={msi_n} (fs={fs_hz} Hz × win={win_sec:.2f} s); cfg.ssvep_msi_samples={self.cfg.ssvep_msi_samples}")
        self._ssvep.reset(
            params=SSVEPParams(
                fs_hz=fs_hz,
                window_sec=float(win_sec),
                msi_n_samples=int(msi_n),
                freqs_hz=active_freqs,
                mode=str(mode),
                roi_channels_0idx=tuple(int(c) for c in self.cfg.ssvep_roi_channels_0idx),
            )
        )
        item = self._current_queue_item()
        if lamp_0idx is not None:
            self._ssvep_target_lamp = int(lamp_0idx)
        elif item is not None and item.kind == "ssvep":
            self._ssvep_target_lamp = int(self._ssvep_target_lamp_0idx_for_item(item))
        else:
            self._ssvep_target_lamp = 0
        freq_labels_list = [str(x).replace(",", ".") if float(x) > 0 else "0" for x in self.cfg.ssvep_freqs_hz]
        while len(freq_labels_list) < 6:
            freq_labels_list.append("0")
        freq_labels = tuple(freq_labels_list[:6])
        mig_mode = 0 if str(mode) == "continuous" else 1
        burst_on_ms = (
            int(round(float(self.cfg.ssvep_block_sec) * 1000.0))
            if str(mode) == "burst"
            else None
        )
        mcfg = MigalkaConfig(
            port=str(self.cfg.com_port),
            mode=int(mig_mode),
            freqs=freq_labels,
            burst_target_on_ms=burst_on_ms,
        )
        plog.info(
            f"SSVEP engine: lamps={n_lamps} freqs_Hz={list(active_freqs)} "
            f"roi={list(self.cfg.ssvep_roi_channels_0idx) or 'ALL'} "
            f"migalka_M={mig_mode} ({'постоянный' if mig_mode == 0 else 'пакетный'})"
        )
        try:
            if continuation and str(mode) == "continuous" and self._migalka.is_open():
                plog.info("continuous group: мигалка без stop/reopen (лампа продолжает)")
            elif continuation and str(mode) == "burst" and self._migalka.is_open():
                plog.info("burst group: reconfigure без stop/reopen")
                self._migalka.reconfigure(mcfg)
            elif str(mode) == "burst" and self._migalka.is_open() and self._ssvep_any_continuous_done:
                plog.info("пакетный: COM открыт после continuous — handoff apply")
                self._migalka.open_and_start(mcfg)
            else:
                try:
                    self._migalka.stop_and_close()
                except Exception:
                    pass
                self._migalka.open_and_start(mcfg)
            if not self._migalka.is_open():
                raise RuntimeError("порт открыт, но is_open() == False")
        except Exception as e:
            plog.exc(f"Migalka не запустилась на {self.cfg.com_port!r}", e)
            self._ssvep_block_started_at = None
            self._ssvep_migalka_fail_streak += 1
            self._ssvep_migalka_retry_at = float(time.time()) + 3.0
            sm = _ru_ssvep_stim_mode(mode)
            self.status_text = (
                f"ССВП ({sm}): ошибка мигалки {self.cfg.com_port}: {e}. "
                f"Повтор через 3 с (закройте migalka.py, если порт занят)."
            )
            self._logger.write("migalka_error", {"port": str(self.cfg.com_port), "error": str(e), "mode": str(mode)})
            return

        self._ssvep_migalka_fail_streak = 0
        self._ssvep_block_started_at = float(time.time())
        self._ssvep_last_msi_log_at = None
        self._ssvep_burst_device_done = False
        self._ssvep_burst_stim_stats = None
        self._ssvep_burst_block_start_pending = str(mode) == "burst"
        item_start = self._current_queue_item()
        is_group_start = (
            item_start is not None
            and int(item_start.ssvep_block_in_group or 0) == 0
            and not continuation
        )
        if is_group_start:
            # Один эксперимент = group_size суб-блоков; trace накапливаем на всю группу (~30 с).
            self._ssvep_msi_trace.reset()
            self._ssvep_group_subblocks = []
        if is_group_start or not continuation:
            self._logger.mark_experiment_start(
                kind="ssvep",
                mode=str(mode),
                target_lamp_0idx=int(self._ssvep_target_lamp) if self._ssvep_target_lamp is not None else 0,
            )
        if not continuation or not self.ssvep_blackout_visible:
            self._set_ssvep_blackout(True)
        plog.info(f"мигалка запущена, блок {self.cfg.ssvep_block_sec}s, лампа-цель={self._ssvep_target_lamp}")
        sm = _ru_ssvep_stim_mode(mode)
        self.status_text = (
            f"{self._status_progress_label()}: ССВП ({sm}), лампа {int(self._ssvep_target_lamp)} — "
            f"0/{self.cfg.ssvep_block_sec:.0f} с"
        )
        start_payload: Dict[str, Any] = {
            "mode": str(mode),
            "target_lamp": int(self._ssvep_target_lamp),
            "freqs_hz": [float(x) for x in self.cfg.ssvep_freqs_hz],
        }
        start_payload.update(self._ssvep_group_metadata(self._current_queue_item()))
        if continuation:
            start_payload["group_visual_continuation"] = True
            start_payload["is_subblock"] = True
        else:
            start_payload["is_subblock"] = int(item_start.ssvep_block_in_group or 0) > 0 if item_start else False
        if str(mode) == "burst":
            start_payload["burst_global_on_ms"] = int(round(float(self.cfg.ssvep_block_sec) * 1000.0))
            start_payload["burst_wall_limit_sec"] = float(self._ssvep_burst_wall_limit_sec())
            start_payload["burst_semantics"] = "global_on_budget_shared"
        self._logger.write("ssvep_block_start", start_payload)

    def _run_ssvep_blocks(self, *, mode: str) -> None:
        assert self._logger is not None
        if self._ssvep_block_started_at is None:
            if self._pause_until_wall is not None:
                return
            if self._ssvep_migalka_fail_streak >= 15:
                sm = _ru_ssvep_stim_mode(mode)
                self.status_text = (
                    f"ССВП ({sm}): мигалка на {self.cfg.com_port!r} не отвечает. "
                    "Проверьте COM, питание и что migalka.py не занял порт."
                )
                return
            now = float(time.time())
            if now < float(self._ssvep_migalka_retry_at):
                sm = _ru_ssvep_stim_mode(mode)
                rem = float(self._ssvep_migalka_retry_at) - now
                self.status_text = f"ССВП ({sm}): повтор подключения мигалки через {rem:.1f} с…"
                return
            try:
                item = self._current_queue_item()
                lamp = (
                    int(item.target_lamp_0idx)
                    if item is not None and item.target_lamp_0idx is not None
                    else None
                )
                self._start_ssvep_block(mode=str(mode), lamp_0idx=lamp)
            except Exception as e:
                plog.exc("критическая ошибка _start_ssvep_block", e)
                self.status_text = f"ССВП: ошибка запуска блока: {e}"
            return
        self._process_migalka_burst_status()
        elapsed = float(time.time()) - float(self._ssvep_block_started_at)
        sm = _ru_ssvep_stim_mode(mode)
        if str(mode) == "burst":
            wall_lim = float(self._ssvep_burst_wall_limit_sec())
            self.status_text = (
                f"{self._status_progress_label()}: ССВП ({sm}) — "
                f"wall {elapsed:.1f}/{wall_lim:.0f} с, ON total {self.cfg.ssvep_block_sec:.0f} с, "
                f"cue L{int(self._ssvep_target_lamp) + 1 if self._ssvep_target_lamp is not None else '?'}"
            )
            block_open = not self._ssvep_burst_device_done and elapsed < wall_lim
        else:
            self.status_text = (
                f"{self._status_progress_label()}: ССВП ({sm}) — "
                f"{elapsed:.1f}/{self.cfg.ssvep_block_sec:.0f} с, лампа {self._ssvep_target_lamp}"
            )
            block_open = elapsed < float(self.cfg.ssvep_block_sec)
        if block_open:
            if self._ssvep.can_classify():
                last = self._ssvep_last_msi_log_at
                if last is None or (float(time.time()) - float(last)) >= 2.0:
                    snap = self._ssvep.classify()
                    if snap.classify_allowed:
                        self._log_ssvep_msi_result(
                            dec=snap,
                            mode=str(mode),
                            target_lamp_0idx=self._ssvep_target_lamp,
                            elapsed_s=elapsed,
                            final=False,
                        )
                        self._emit_ssvep_msi_trace_point(
                            snap, mode=str(mode), elapsed_s=elapsed, final=False
                        )
                        self._ssvep_last_msi_log_at = float(time.time())
            return

        # End of logical block: MSI + logs; visual teardown only between groups.
        item = self._current_queue_item()
        continues_group = self._ssvep_continues_visual_group(item)
        if not continues_group:
            self._set_ssvep_blackout(False)
            has_burst_later = any(
                x.kind == "ssvep" and x.ssvep_mode == "burst"
                for x in self._main_queue[self._main_queue_index + 1 :]
            )
            cont_handoff_burst = (
                str(mode) == "continuous" and has_burst_later and self._migalka.is_open()
            )
            if cont_handoff_burst:
                self._ssvep_any_continuous_done = True
                plog.info("последний continuous -> halt + handoff M1 (COM остаётся открыт)")
                try:
                    self._migalka.halt_lamps()
                    self._migalka.prepare_burst_handoff()
                except Exception as e:
                    plog.exc("handoff continuous->burst", e)
            else:
                try:
                    self._migalka.stop_and_close()
                except Exception:
                    pass
        else:
            plog.info("SSVEP group continuation: мигалка/blackout без сброса до конца группы")
        dec = self._ssvep.classify()
        cue_lamp = int(self._ssvep_target_lamp) if self._ssvep_target_lamp is not None else None
        self._emit_ssvep_msi_trace_point(
            dec,
            mode=str(mode),
            elapsed_s=float(self.cfg.ssvep_block_sec),
            final=True,
        )
        item_sub = self._current_queue_item()
        group_size = max(1, int(item_sub.ssvep_group_size if item_sub is not None else self.cfg.ssvep_group_size))
        sub_bi = int(item_sub.ssvep_block_in_group or 0) if item_sub is not None else 0
        subblock_sec = float(self.cfg.ssvep_block_sec)
        experiment_sec = subblock_sec * float(group_size)
        trace_full = self._ssvep_msi_trace.finalize(
            target_lamp_0idx=cue_lamp,
            final_winner_0idx=dec.winner_0idx,
            block_duration_sec=experiment_sec if not continues_group else subblock_sec,
        )
        trace_summary = trace_full.get("summary") or {}
        trace_public = {
            "compact_stream": trace_full.get("compact_stream", ""),
            "summary": trace_summary,
            "subblock_index": int(sub_bi),
            "subblock_duration_sec": subblock_sec,
            "experiment_duration_sec": experiment_sec,
            "group_size": int(group_size),
        }
        maj_0 = trace_summary.get("winner_majority_0idx")
        maj_1 = trace_summary.get("winner_majority_1based")
        last_0 = trace_summary.get("winner_last_0idx")
        last_1 = trace_summary.get("winner_last_1based")
        if last_0 is None and dec.winner_0idx is not None:
            last_0 = int(dec.winner_0idx)
            last_1 = int(dec.winner_1based) if dec.winner_1based is not None else int(last_0) + 1
        stat_correct: Optional[bool] = None
        if cue_lamp is not None and maj_0 is not None and not trace_summary.get("winner_majority_tied"):
            stat_correct = int(maj_0) == int(cue_lamp)
        rec = {
            "mode": str(mode),
            "target_lamp": cue_lamp,
            "is_subblock": True,
            "subblock_index": sub_bi,
            **self._ssvep_group_metadata(item_sub),
            "winner_1based": maj_1,
            "winner_0idx": maj_0,
            "winner_last_1based": last_1,
            "winner_last_0idx": last_0,
            "winner_majority_1based": maj_1,
            "winner_majority_0idx": maj_0,
            "winner_majority_count": trace_summary.get("winner_majority_count"),
            "winner_majority_tied": bool(trace_summary.get("winner_majority_tied")),
            "decision_histogram": trace_summary.get("decision_histogram") or {},
            "coef": dec.coef,
            "classify_allowed": bool(dec.classify_allowed),
            "debug": dec.debug,
            "msi_decision_trace": trace_public,
            "correct": stat_correct,
            "correct_last_decision": (
                int(last_0) == int(cue_lamp)
                if cue_lamp is not None and last_0 is not None
                else None
            ),
        }
        self._logger.append_ssvep_block(rec)
        self._logger.write("ssvep_block_end", rec)
        self._logger.write(
            "msi_decision_trace",
            {
                **trace_public,
                "queue_index": int(self._main_queue_index),
                "ssvep_mode": str(mode),
                "target_lamp_0idx": cue_lamp,
                "is_subblock": True,
                "subblock_index": sub_bi,
            },
        )
        self._ssvep_group_subblocks.append(dict(rec))

        if not continues_group:
            group_meta = aggregate_ssvep_group_experiment(
                self._ssvep_group_subblocks,
                target_lamp_0idx=cue_lamp,
            )
            exp_rec = {
                "phase": "main",
                "kind": "ssvep",
                "queue_index": int(self._main_queue_index),
                **self._ssvep_group_metadata(item_sub),
                "ssvep_mode": str(mode),
                "cue_target_lamp_0idx": cue_lamp,
                "cue_target_lamp_1based": int(cue_lamp) + 1 if cue_lamp is not None else None,
                "ground_truth_lamp_0idx": cue_lamp,
                "winner_1based": group_meta.get("group_majority_1based", maj_1),
                "winner_0idx": maj_0,
                "winner_last_1based": last_1,
                "winner_last_0idx": last_0,
                "winner_majority_1based": group_meta.get("group_majority_1based", maj_1),
                "winner_majority_0idx": maj_0,
                "winner_majority_count": trace_summary.get("winner_majority_count"),
                "winner_majority_tied": bool(group_meta.get("group_majority_tied")),
                "decision_histogram": trace_summary.get("decision_histogram") or {},
                "correct": group_meta.get("group_correct", stat_correct),
                "correct_last_decision": (
                    int(last_0) == int(cue_lamp)
                    if cue_lamp is not None and last_0 is not None
                    else None
                ),
                "coef": dec.coef,
                "classify_allowed": bool(dec.classify_allowed),
                "debug": dec.debug,
                "ssvep_freqs_hz": [float(x) for x in self._active_ssvep_freqs_hz()],
                "msi_decision_trace": trace_full,
                "burst_stim_stats": self._ssvep_burst_stim_stats,
                "ssvep_experiment_semantics": "grouped_lamp_session",
                **group_meta,
            }
            ssvep_avg = build_ssvep_averages_export(
                self._ssvep,
                roi_channels_0idx=self.cfg.ssvep_roi_channels_0idx,
                n_lamps=len(self.cfg.ssvep_freqs_hz),
            )
            if ssvep_avg is not None:
                exp_rec["_ssvep_averages_export"] = ssvep_avg
            self._logger.append_experiment(exp_rec)
            self._ssvep_group_subblocks = []
            if mode == "continuous":
                self._ssvep_experiment_count_cont += 1
            else:
                self._ssvep_experiment_count_burst += 1

        self._log_ssvep_msi_result(
            dec=dec,
            mode=str(mode),
            target_lamp_0idx=cue_lamp,
            elapsed_s=float(self.cfg.ssvep_block_sec),
            final=True,
        )

        if mode == "continuous":
            self._ssvep_block_count_cont += 1
        else:
            self._ssvep_block_count_burst += 1

        if continues_group:
            self._main_queue_index += 1
            nxt = self._current_queue_item()
            lamp = (
                int(nxt.target_lamp_0idx)
                if nxt is not None and nxt.target_lamp_0idx is not None
                else self._ssvep_target_lamp
            )
            if nxt is not None and nxt.target_lamp_0idx is not None:
                self._ssvep_target_lamp = int(nxt.target_lamp_0idx)
            plog.info(
                f"SSVEP visual group: следующий logical block in-group "
                f"(idx={self._main_queue_index}, lamp_0idx={lamp})"
            )
            try:
                self._start_ssvep_block(mode=str(mode), lamp_0idx=lamp, continuation=True)
            except Exception as e:
                plog.exc("критическая ошибка continuation _start_ssvep_block", e)
                self.status_text = f"ССВП: ошибка продолжения группы: {e}"
                self._ssvep_block_started_at = None
                self._set_state(ProtocolState.Main, detail="ssvep continuation failed")
                self._complete_main_queue_item()
            return

        self._ssvep_block_started_at = None
        self._ssvep_target_lamp = None
        self._set_state(ProtocolState.Main, detail="ssvep block done")
        self._complete_main_queue_item()

    def _finalize(self) -> None:
        if self._logger_finalized and self._runtime_resources_released:
            return
        self._drain_migalka_com_events()
        self._sync_stim_control(None)
        self.participant_instruction = None
        self.clear_ssvep_display()
        self.status_text = "Завершение: сохранение логов…"
        self._abort_open_ssvep_block(reason="protocol_done")
        self._release_runtime_resources()
        out_dir = self._finalize_logger_once(
            {
                "reason": "protocol_done",
                "p300_calib_trials": int(self._p300_calib_trials_done),
                "p300_main_trials": int(self._p300_main_trials_done),
                "ssvep_cont_blocks": int(self._ssvep_block_count_cont),
                "ssvep_burst_blocks": int(self._ssvep_block_count_burst),
                "ssvep_cont_experiments": int(self._ssvep_experiment_count_cont),
                "ssvep_burst_experiments": int(self._ssvep_experiment_count_burst),
                "shuffle_seed": int(self._shuffle_seed_used or 0),
            }
        )
        self._set_state(ProtocolState.Stopped, detail="finalize")
        if out_dir is not None:
            self.status_text = f"Готово. Логи: {out_dir}"
            plog.info(f"=== PROTOCOL DONE logs={out_dir} ===")
        else:
            self.status_text = "Готово."
            plog.info("=== PROTOCOL DONE (already finalized) ===")

