#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Операторский GUI гибридного протокола v2 (калибровка → 45 экспериментов)."""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
import subprocess

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol.protocol_log import info as plog_info  # noqa: E402
from runtime_core.protocol import (  # noqa: E402
    HybridProtocolRuntime,
    ProtocolDisplayView,
    block_running,
    eeg_indicator_state,
    session_ready,
)
from runtime_core.adapters import stim as stim_adapter  # noqa: E402
from runtime_core.protocol.runner import ProtocolConfig, ProtocolRunner  # noqa: E402
from runtime_core.logging import UnifiedExperimentLogger, sanitize_subject_slug  # noqa: E402
from experiment_protocol.experiment_queue import (  # noqa: E402
    default_ssvep_experiment_lamps,
    ensure_ssvep_experiment_lamp_permutation,
    shuffle_ssvep_experiment_lamps,
)
from experiment_protocol.ssvep_cue_overlay import SsvepCueOverlay  # noqa: E402
from p300_analysis.analysis_profiles import (  # noqa: E402
    ANALYSIS_PROFILE_GENERAL,
    ANALYSIS_PROFILE_RECENT,
    get_analysis_profile,
)
from runtime_core.adapters.lsl import (  # noqa: E402
    BCI_STIM_MARKER_STREAM_NAME,
    discover_eeg_streams,
    probe_stimulus_marker_stream,
    select_eeg_stream,
    stream_channel_labels,
    stream_display_label,
    stream_inlet_with_buffer,
)
from ssvep_analysis.lamp_frequencies import (  # noqa: E402
    DEFAULT_LAMP_FREQS_HZ,
    SSVEP_PROTOCOL_ROI_0IDX,
    lamp_frequency_choices,
    lamp_frequency_closest_index,
)

_EEG_LED_STYLE_OFF = (
    "background-color:#b0b0b0; border:1px solid #888; border-radius:10px;"
    "min-width:20px; max-width:20px; min-height:20px; max-height:20px;"
)
_EEG_LED_STYLE_RED = (
    "background-color:#e74c3c; border:1px solid #c0392b; border-radius:10px;"
    "min-width:20px; max-width:20px; min-height:20px; max-height:20px;"
)
_EEG_LED_STYLE_GREEN = (
    "background-color:#27ae60; border:1px solid #1e8449; border-radius:10px;"
    "min-width:20px; max-width:20px; min-height:20px; max-height:20px;"
)
_EEG_LIVE_TIMEOUT_S = 2.5
_STIM_MARKER_WAIT_SEC = 35.0
_STIM_MARKER_POLL_MS = 250

_MAX_LAMPS = 6
_CHANNEL_COLUMNS = 4
_DEFAULT_COM_PORT = "COM8"

# Defaults aligned with analysis_profiles + lamp_frequencies (Stage 2.4).
_PROTOCOL_P300_ROI_0IDX = tuple(
    int(c) for c in get_analysis_profile(ANALYSIS_PROFILE_GENERAL).roi_channels_0idx
)
_PROTOCOL_SSVEP_ROI_0IDX = SSVEP_PROTOCOL_ROI_0IDX


def _list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports

        return sorted({p.device for p in list_ports.comports()})
    except Exception:
        return []


def _make_lamp_freq_combo(initial_hz: float) -> QComboBox:
    combo = QComboBox()
    combo.addItem("0 (выкл)", 0.0)
    for text, val in lamp_frequency_choices():
        combo.addItem(text, float(val))
    if float(initial_hz) <= 0.0:
        combo.setCurrentIndex(0)
    else:
        combo.setCurrentIndex(int(lamp_frequency_closest_index(float(initial_hz))) + 1)
    combo.setToolTip("Дискретные частоты 1000/i Гц (как в анализаторе SSVEP и migalka.py)")
    return combo


def _combo_freq_hz(combo: QComboBox) -> float:
    v = combo.currentData()
    return float(v) if v is not None else 0.0


def _eeg_stream_key(info) -> tuple[str, str]:
    try:
        return (str(info.name() or ""), str(info.session_id() or ""))
    except Exception:
        return ("", "")


def _clear_grid_layout(layout: QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _selected_channels_0idx(checkboxes: list[QCheckBox]) -> tuple[int, ...]:
    """Выбранные каналы (0-based). Пустой кортеж = все каналы отмечены."""
    if not checkboxes:
        return ()
    idx = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
    if not idx or len(idx) >= len(checkboxes):
        return ()
    return tuple(idx)


class ProtocolRunnerWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Гибридный протокол (P300×2 + SSVEP×2)")
        self.setMinimumWidth(620)
        self.resize(740, 260)

        self._runner: HybridProtocolRuntime | None = None
        self._stimulus_proc: subprocess.Popen[str] | None = None
        self._session_dir: Path | None = None
        self._last_stim_restart_attempt: float = 0.0
        self._last_status_printed: str = ""
        self._eeg_test_inlet = None
        self._ssvep_cue_overlay: SsvepCueOverlay | None = None
        self._ssvep_lamp_shuffle_seed: int = 0
        self._eeg_test_last_live_at: float = 0.0
        self._last_runner_state: str = ""
        self._stim_marker_wait_active: bool = False
        self._stim_marker_wait_deadline: float = 0.0
        self._stim_marker_wait_pending_p300: bool = False

        # ── Виджеты верхней панели (всегда видны) ──────────────────────
        self.cb_eeg = QComboBox()
        self.cb_eeg.setMinimumWidth(280)
        self.btn_refresh_eeg = QPushButton("Обновить")
        self.btn_refresh_eeg.setToolTip("Найти потоки LSL типа EEG/Signal в сети")
        self.btn_test_eeg = QPushButton("Подключить")
        self.btn_test_eeg.setToolTip(
            "Проверить поток LSL (индикатор: зелёный — данные идут). Повторный клик — отключить."
        )
        self.lbl_eeg_led = QLabel()
        self.lbl_eeg_led.setToolTip("Индикатор потока ЭЭГ")
        self.lbl_eeg_led.setFixedSize(20, 20)
        self.lbl_eeg_status = QLabel("Поток ЭЭГ не выбран. Нажмите «Обновить».")
        self.lbl_eeg_status.setWordWrap(True)
        self.lbl_eeg_status.setStyleSheet("color: #555; font-size: 11px;")
        self._set_eeg_indicator("off")

        self.ed_last_name = QLineEdit()
        self.ed_last_name.setPlaceholderText("Фамилия")
        self.ed_last_name.setMinimumWidth(140)
        self.ed_first_name = QLineEdit()
        self.ed_first_name.setPlaceholderText("Имя")
        self.ed_first_name.setMinimumWidth(120)

        self.btn_settings_toggle = QPushButton("⚙  Развернуть настройки")
        self.btn_settings_toggle.setStyleSheet(
            "text-align:left; padding:4px 8px; border:1px solid #bbb; border-radius:3px;"
        )

        # ── Виджеты настроек (скрыты по умолчанию) ─────────────────────
        self.ed_output = QLineEdit(str((_ROOT / "experiment_runs").resolve()))

        self.cb_com = QComboBox()
        self.cb_com.setEditable(True)
        self.cb_com.setMinimumWidth(180)
        self.cb_com.setCurrentText(_DEFAULT_COM_PORT)
        self.btn_refresh_com = QPushButton("Обновить")
        self.btn_refresh_com.setToolTip("Повторно найти COM-порты (как в окне migalka.py)")

        self._freq_combos: list[QComboBox] = []
        self._lamp_row_widgets: list[QWidget] = []
        migalka_box = QGroupBox("Мигалка (ССВП) — частоты ламп")
        migalka_outer = QVBoxLayout(migalka_box)
        self._lamp_rows_host = QWidget()
        self._lamp_rows_layout = QVBoxLayout(self._lamp_rows_host)
        self._lamp_rows_layout.setContentsMargins(0, 0, 0, 0)
        migalka_outer.addWidget(self._lamp_rows_host)
        lamp_btn_row = QHBoxLayout()
        self.btn_add_lamp = QPushButton("+ лампа")
        self.btn_add_lamp.setToolTip(f"Добавить лампу (не более {_MAX_LAMPS}, как в анализаторе SSVEP)")
        self.btn_remove_lamp = QPushButton("− лампа")
        self.btn_remove_lamp.setToolTip("Убрать последнюю лампу (не меньше 4 по умолчанию)")
        lamp_btn_row.addWidget(self.btn_add_lamp)
        lamp_btn_row.addWidget(self.btn_remove_lamp)
        lamp_btn_row.addStretch(1)
        migalka_outer.addLayout(lamp_btn_row)
        for f0 in DEFAULT_LAMP_FREQS_HZ:
            self._add_lamp_freq_row(float(f0))

        self._p300_ch_checkboxes: list[QCheckBox] = []
        self._ssvep_ch_checkboxes: list[QCheckBox] = []
        self._eeg_channel_count = 0

        ssvep_box = QGroupBox("ССВП — каналы ЭЭГ для MSI")
        ssvep_outer = QVBoxLayout(ssvep_box)
        ssvep_top = QHBoxLayout()
        self.cb_ssvep_profile = QComboBox()
        self.cb_ssvep_profile.addItem("Протокол (9, 10, 21)", "protocol_ssvep")
        self.cb_ssvep_profile.addItem("Все каналы", "all")
        self.cb_ssvep_profile.addItem("Общий профиль (канал 4)", "general")
        self.cb_ssvep_profile.addItem("Последние 7 сессий (каналы 5, 6)", "recent")
        self.cb_ssvep_profile.addItem("Вручную", "custom")
        ssvep_top.addWidget(QLabel("Профиль:"))
        ssvep_top.addWidget(self.cb_ssvep_profile, stretch=1)
        self.btn_ssvep_ch_all = QPushButton("Все")
        self.btn_ssvep_ch_none = QPushButton("Снять")
        ssvep_top.addWidget(self.btn_ssvep_ch_all)
        ssvep_top.addWidget(self.btn_ssvep_ch_none)
        ssvep_outer.addLayout(ssvep_top)
        self._ssvep_ch_host = QWidget()
        self._ssvep_ch_layout = QGridLayout(self._ssvep_ch_host)
        self._lbl_ssvep_ch_hint = QLabel("Сначала подключитесь к потоку ЭЭГ.")
        self._lbl_ssvep_ch_hint.setWordWrap(True)
        self._lbl_ssvep_ch_hint.setStyleSheet("color: #666;")
        ssvep_outer.addWidget(self._lbl_ssvep_ch_hint)
        ssvep_outer.addWidget(self._ssvep_ch_host)

        msi_form = QFormLayout()
        self.spin_ssvep_msi_fs = QSpinBox()
        self.spin_ssvep_msi_fs.setRange(1, 20000)
        self.spin_ssvep_msi_fs.setValue(5000)
        self.spin_ssvep_msi_fs.setToolTip(
            "Частота дискретизации для MSI (квантования). При подключении LSL "
            "в протоколе используется nominal_srate потока."
        )
        self.spin_ssvep_msi_samples = QSpinBox()
        self.spin_ssvep_msi_samples.setRange(8, 200000)
        self.spin_ssvep_msi_samples.setValue(15000)
        self.spin_ssvep_msi_samples.setToolTip(
            "Число отсчётов в окне MSIExec (как в оригинальном MSI: 15000 при 5000 Гц)."
        )
        msi_form.addRow("MSI: дискретизация, Гц:", self.spin_ssvep_msi_fs)
        msi_form.addRow("MSI: число отсчётов:", self.spin_ssvep_msi_samples)
        ssvep_outer.addLayout(msi_form)

        p300_box = QGroupBox("P300 — каналы ЭЭГ (область интереса)")
        p300_outer = QVBoxLayout(p300_box)
        p300_top = QHBoxLayout()
        self.cb_p300_profile = QComboBox()
        self.cb_p300_profile.addItem("Протокол (5–8, 19–20)", "protocol_p300")
        self.cb_p300_profile.addItem("Общий профиль (канал 4)", "general")
        self.cb_p300_profile.addItem("Последние 7 сессий (каналы 5, 6)", "recent")
        self.cb_p300_profile.addItem("Все каналы", "all")
        self.cb_p300_profile.addItem("Вручную", "custom")
        p300_top.addWidget(QLabel("Профиль:"))
        p300_top.addWidget(self.cb_p300_profile, stretch=1)
        self.btn_p300_ch_all = QPushButton("Все")
        self.btn_p300_ch_none = QPushButton("Снять")
        p300_top.addWidget(self.btn_p300_ch_all)
        p300_top.addWidget(self.btn_p300_ch_none)
        p300_outer.addLayout(p300_top)
        self._p300_ch_host = QWidget()
        self._p300_ch_layout = QGridLayout(self._p300_ch_host)
        self._lbl_p300_ch_hint = QLabel("Сначала подключитесь к потоку ЭЭГ.")
        self._lbl_p300_ch_hint.setWordWrap(True)
        self._lbl_p300_ch_hint.setStyleSheet("color: #666;")
        p300_outer.addWidget(self._lbl_p300_ch_hint)
        p300_outer.addWidget(self._p300_ch_host)
        self._p300_ch_host.hide()
        self._ssvep_ch_host.hide()

        proto_box = QGroupBox("Протокол (ручной запуск блоков)")
        proto_form = QFormLayout(proto_box)

        self.spin_calib_trials = QSpinBox()
        self.spin_calib_trials.setRange(0, 50)
        self.spin_calib_trials.setValue(5)
        self.spin_calib_trials.setToolTip(
            "Прогонов P300 на одной плитке для эталона (0 = отключить калибровку). "
            "5 прогонов × 12 sequences ≈ порог шаблона 12."
        )

        self.spin_calib_target = QSpinBox()
        self.spin_calib_target.setRange(0, 8)
        self.spin_calib_target.setValue(4)

        self.spin_template_epochs = QSpinBox()
        self.spin_template_epochs.setRange(4, 50)
        self.spin_template_epochs.setValue(12)

        self.spin_p300_main = QSpinBox()
        self.spin_p300_main.setRange(1, 200)
        self.spin_p300_main.setValue(15)
        self.spin_p300_main.setToolTip(
            "В основном блоке: каждый прогон P300 — AUC и сравнение с шаблоном (один trial)."
        )

        self.spin_ssvep = QSpinBox()
        self.spin_ssvep.setRange(1, 200)
        self.spin_ssvep.setValue(4)
        self.spin_ssvep.setToolTip(
            "ССВП: число экспериментов на режим (continuous и burst).\n"
            "Каждый эксперимент = одна лампа, 4×8 с непрерывной стимуляции (~32 с).\n"
            "Порядок ламп задаётся вручную ниже (можно повторять)."
        )

        self.ssvep_manual_box = QGroupBox("ССВП — ручной порядок ламп")
        ssvep_manual_layout = QVBoxLayout(self.ssvep_manual_box)

        self.chk_ssvep_auto_shuffle = QCheckBox(
            "Авто-перемешивание при старте режима SSVEP"
        )
        self.chk_ssvep_auto_shuffle.setChecked(True)
        self.chk_ssvep_auto_shuffle.setToolTip(
            "Новая перестановка 4 ламп без повторов при старте continuous и при старте burst "
            "(независимо). Не при Prepare — только в runtime."
        )
        ssvep_manual_layout.addWidget(self.chk_ssvep_auto_shuffle)

        self.ssvep_cont_box = QGroupBox("Continuous")
        self.ssvep_cont_form = QFormLayout(self.ssvep_cont_box)
        self._ssvep_cont_lamp_combos: list[QComboBox] = []

        cont_btn_row = QHBoxLayout()
        self.btn_shuffle_ssvep_cont = QPushButton("Перемешать лампы")
        self.btn_shuffle_ssvep_cont.setToolTip(
            "Случайная перестановка без повторов (если экспериментов ≤ числа ламп)."
        )
        cont_btn_row.addStretch(1)
        cont_btn_row.addWidget(self.btn_shuffle_ssvep_cont)

        self.ssvep_burst_box = QGroupBox("Burst")
        self.ssvep_burst_form = QFormLayout(self.ssvep_burst_box)
        self._ssvep_burst_lamp_combos: list[QComboBox] = []

        burst_btn_row = QHBoxLayout()
        self.btn_shuffle_ssvep_burst = QPushButton("Перемешать лампы")
        self.btn_shuffle_ssvep_burst.setToolTip(
            "Случайная перестановка без повторов (если экспериментов ≤ числа ламп)."
        )
        burst_btn_row.addStretch(1)
        burst_btn_row.addWidget(self.btn_shuffle_ssvep_burst)

        ssvep_manual_layout.addWidget(self.ssvep_cont_box)
        ssvep_manual_layout.addLayout(cont_btn_row)
        ssvep_manual_layout.addWidget(self.ssvep_burst_box)
        ssvep_manual_layout.addLayout(burst_btn_row)

        self.spin_pause = QDoubleSpinBox()
        self.spin_pause.setRange(0.0, 60.0)
        self.spin_pause.setDecimals(1)
        self.spin_pause.setSingleStep(0.5)
        self.spin_pause.setValue(2.0)

        self.spin_ssvep_block_sec = QDoubleSpinBox()
        self.spin_ssvep_block_sec.setRange(1.0, 120.0)
        self.spin_ssvep_block_sec.setDecimals(1)
        self.spin_ssvep_block_sec.setValue(8.0)
        self.spin_ssvep_block_sec.setToolTip(
            "Длительность одного суб-блока (с). Один эксперимент ССВП = 4 суб-блока подряд "
            "(см. «экспериментов на режим»), итого ~4× это значение (например 4×7.5≈30 с).\n"
            "Continuous: wall-clock суб-блока.\n"
            "Burst: ON-бюджет на суб-блок; паузы сверху (~+3 с к wall)."
        )

        proto_form.addRow("Калибровка: число прогонов P300:", self.spin_calib_trials)
        proto_form.addRow("Калибровка: целевая плитка (0…8):", self.spin_calib_target)
        proto_form.addRow("Мин. эпох на шаблон:", self.spin_template_epochs)
        proto_form.addRow("Основной блок: прогонов P300:", self.spin_p300_main)
        proto_form.addRow("ССВП: экспериментов на режим (cont+burst):", self.spin_ssvep)
        proto_form.addRow("Пауза между trial/блоками (с):", self.spin_pause)
        proto_form.addRow("ССВП: суб-блок (с) [×4 ≈ эксперимент]:", self.spin_ssvep_block_sec)

        stim_box = QGroupBox("Стимулятор P300 (PsychoPy, авто-режим)")
        stim_form = QFormLayout(stim_box)

        self.spin_inter_trial = QDoubleSpinBox()
        self.spin_inter_trial.setRange(0.0, 30.0)
        self.spin_inter_trial.setDecimals(1)
        self.spin_inter_trial.setSingleStep(0.5)
        self.spin_inter_trial.setValue(1.0)

        self.spin_sequences = QSpinBox()
        self.spin_sequences.setRange(1, 50)
        self.spin_sequences.setValue(12)

        self.chk_run_stimulus = QCheckBox(
            "Запускать экранную стимуляцию (PsychoPy, без кнопки START)"
        )
        self.chk_run_stimulus.setChecked(True)
        self.chk_run_stimulus.setToolTip(
            "PsychoPy: P300 с серым оверлеем между trial (как 3a0a997), по stim_control.json.\n"
            "ССВП — отдельное окно SsvepCueOverlay, PsychoPy закрывается."
        )

        stim_form.addRow("Пауза между trial P300 (с):", self.spin_inter_trial)
        stim_form.addRow("Последовательностей в прогоне:", self.spin_sequences)
        stim_form.addRow("", self.chk_run_stimulus)

        # ── Панель настроек (прокручиваемая, скрыта) ───────────────────
        settings_content = QWidget()
        settings_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        settings_v = QVBoxLayout(settings_content)
        settings_v.setContentsMargins(4, 4, 4, 4)

        settings_form = QFormLayout()
        settings_form.addRow("Папка результатов:", self.ed_output)
        settings_v.addLayout(settings_form)
        settings_v.addWidget(proto_box)
        settings_v.addWidget(self.ssvep_manual_box)
        settings_v.addWidget(stim_box)
        settings_v.addWidget(migalka_box)
        settings_v.addWidget(ssvep_box)
        settings_v.addWidget(p300_box)
        lbl_hint = QLabel(
            "COM и частоты — мигалка; каналы — ROI для P300/ССВП. Консоль: [protocol]."
        )
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #555; font-size: 11px;")
        settings_v.addWidget(lbl_hint)
        settings_v.addStretch(1)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setFrameShape(QScrollArea.NoFrame)
        settings_scroll.setWidget(settings_content)

        self._settings_panel = QWidget()
        self._settings_panel.setVisible(False)
        sp_layout = QVBoxLayout(self._settings_panel)
        sp_layout.setContentsMargins(0, 0, 0, 0)
        sp_layout.addWidget(settings_scroll)

        # ── Главный layout ──────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        compact_top = QWidget()
        compact_v = QVBoxLayout(compact_top)
        compact_v.setContentsMargins(4, 4, 4, 4)
        compact_v.setSpacing(4)

        subj_row_h = QHBoxLayout()
        subj_row_h.addWidget(QLabel("Испытуемый:"))
        subj_row_h.addWidget(QLabel("Фамилия"))
        subj_row_h.addWidget(self.ed_last_name, stretch=1)
        subj_row_h.addWidget(QLabel("Имя"))
        subj_row_h.addWidget(self.ed_first_name, stretch=1)
        compact_v.addLayout(subj_row_h)

        eeg_row_h = QHBoxLayout()
        eeg_row_h.addWidget(QLabel("ЭЭГ (LSL):"))
        eeg_row_h.addWidget(self.cb_eeg, stretch=1)
        eeg_row_h.addWidget(self.btn_refresh_eeg)
        eeg_row_h.addWidget(self.btn_test_eeg)
        eeg_row_h.addWidget(self.lbl_eeg_led)
        eeg_row_h.addStretch(1)
        compact_v.addLayout(eeg_row_h)

        com_row_h = QHBoxLayout()
        com_row_h.addWidget(QLabel("COM (мигалка):"))
        com_row_h.addWidget(self.cb_com, stretch=1)
        com_row_h.addWidget(self.btn_refresh_com)
        compact_v.addLayout(com_row_h)

        compact_v.addWidget(self.lbl_eeg_status)
        compact_v.addWidget(self.btn_settings_toggle)

        root.addWidget(compact_top)
        root.addWidget(self._settings_panel, stretch=1)

        self.btn_prepare = QPushButton("Подготовить сессию")
        self.btn_prepare.setToolTip(
            "Подключить LSL/COM, создать папку логов. После этого — кнопки блоков."
        )

        exp_row = QHBoxLayout()
        self.btn_run_p300 = QPushButton("P300 (калибр. + main)")
        self.btn_run_p300.setEnabled(False)
        self.btn_run_p300.setToolTip(
            "Калибровка (N прогонов) и основной блок P300 подряд, без ССВП."
        )
        self.btn_run_ssvep_burst = QPushButton("ССВП пакетный")
        self.btn_run_ssvep_burst.setEnabled(False)
        self.btn_run_ssvep_burst.setToolTip(
            "Пакетный SSVEP: общий ON-бюджет (поле «Длительность блока») на все лампы, "
            "fair scheduler на прошивке. Ожидаемый wall ≈ ON + 3 с."
        )
        self.btn_run_ssvep_cont = QPushButton("ССВП непрерывный")
        self.btn_run_ssvep_cont.setEnabled(False)
        self.btn_run_ssvep_cont.setToolTip(
            "Блоки SSVEP continuous (число — в настройках «ССВП: блоков на режим»)."
        )
        exp_row.addWidget(self.btn_run_p300)
        exp_row.addWidget(self.btn_run_ssvep_burst)
        exp_row.addWidget(self.btn_run_ssvep_cont)

        btn_row = QHBoxLayout()
        self.btn_stop = QPushButton("Завершить сессию")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_prepare)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_stop)
        root.addLayout(exp_row)
        root.addLayout(btn_row)

        self.lbl_status = QLabel(
            "Введите фамилию и имя, подключите ЭЭГ, затем «Подготовить сессию». "
            "Далее запустите блок P300 или ССВП."
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("padding: 4px 0;")
        root.addWidget(self.lbl_status)

        # ── Таймеры ─────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)

        self._eeg_indicator_timer = QTimer(self)
        self._eeg_indicator_timer.setInterval(200)
        self._eeg_indicator_timer.timeout.connect(self._on_eeg_indicator_tick)

        self._stim_marker_wait_timer = QTimer(self)
        self._stim_marker_wait_timer.setInterval(_STIM_MARKER_POLL_MS)
        self._stim_marker_wait_timer.timeout.connect(self._on_stim_marker_wait_tick)

        # ── Сигналы ─────────────────────────────────────────────────────
        self.btn_prepare.clicked.connect(self._on_prepare_session)
        self.btn_run_p300.clicked.connect(self._on_run_p300)
        self.btn_run_ssvep_burst.clicked.connect(lambda: self._on_run_ssvep("burst"))
        self.btn_run_ssvep_cont.clicked.connect(lambda: self._on_run_ssvep("continuous"))
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_refresh_com.clicked.connect(self._refresh_com_ports)
        self.btn_refresh_eeg.clicked.connect(self._refresh_eeg_streams)
        self.btn_test_eeg.clicked.connect(self._on_test_eeg_connect)
        self.btn_settings_toggle.clicked.connect(self._toggle_settings)
        self.cb_eeg.currentIndexChanged.connect(self._rebuild_channel_checkboxes_from_selection)
        self.cb_p300_profile.currentIndexChanged.connect(self._on_p300_profile_changed)
        self.cb_ssvep_profile.currentIndexChanged.connect(self._on_ssvep_profile_changed)
        self.btn_p300_ch_all.clicked.connect(lambda: self._set_channel_checks(self._p300_ch_checkboxes, True))
        self.btn_p300_ch_none.clicked.connect(lambda: self._set_channel_checks(self._p300_ch_checkboxes, False))
        self.btn_ssvep_ch_all.clicked.connect(lambda: self._set_channel_checks(self._ssvep_ch_checkboxes, True))
        self.btn_ssvep_ch_none.clicked.connect(lambda: self._set_channel_checks(self._ssvep_ch_checkboxes, False))
        self.btn_add_lamp.clicked.connect(self._on_add_lamp)
        self.btn_remove_lamp.clicked.connect(self._on_remove_lamp)
        self.spin_ssvep.valueChanged.connect(self._rebuild_ssvep_lamp_rows)
        self.btn_shuffle_ssvep_cont.clicked.connect(self._on_shuffle_ssvep_cont)
        self.btn_shuffle_ssvep_burst.clicked.connect(self._on_shuffle_ssvep_burst)

        self._refresh_com_ports()
        self._refresh_eeg_streams()
        self._update_lamp_buttons()
        self._rebuild_ssvep_lamp_rows()

    def _ssvep_active_lamp_count(self) -> int:
        n = len(self._ssvep_freqs_hz())
        return max(1, min(6, n if n > 0 else 4))

    @staticmethod
    def _clear_form_layout(form: QFormLayout) -> None:
        while form.rowCount() > 0:
            form.removeRow(0)

    def _read_lamp_combo_sequence(self, combos: list[QComboBox]) -> tuple[int, ...]:
        out: list[int] = []
        for cb in combos:
            data = cb.currentData()
            if data is not None:
                out.append(int(data))
            else:
                out.append(int(cb.currentIndex()))
        return tuple(out)

    def _set_lamp_combo_sequence(self, combos: list[QComboBox], lamps_0idx: tuple[int, ...]) -> None:
        for cb, lamp in zip(combos, lamps_0idx):
            idx = cb.findData(int(lamp))
            if idx >= 0:
                cb.setCurrentIndex(idx)

    def _rebuild_ssvep_lamp_rows(self) -> None:
        n_exp = int(self.spin_ssvep.value())
        n_lamps = self._ssvep_active_lamp_count()
        saved_cont = self._read_lamp_combo_sequence(self._ssvep_cont_lamp_combos)
        saved_burst = self._read_lamp_combo_sequence(self._ssvep_burst_lamp_combos)
        if not saved_cont:
            saved_cont = default_ssvep_experiment_lamps(n_exp, n_lamps)
        if not saved_burst:
            saved_burst = default_ssvep_experiment_lamps(n_exp, n_lamps)

        self._clear_form_layout(self.ssvep_cont_form)
        self._ssvep_cont_lamp_combos = []
        for i in range(n_exp):
            cb = QComboBox()
            for l in range(n_lamps):
                cb.addItem(f"L{l + 1}", l)
            lamp = saved_cont[i] if i < len(saved_cont) else (i % n_lamps)
            idx = cb.findData(int(lamp) % n_lamps)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            self.ssvep_cont_form.addRow(f"Эксперимент {i + 1}:", cb)
            self._ssvep_cont_lamp_combos.append(cb)

        self._clear_form_layout(self.ssvep_burst_form)
        self._ssvep_burst_lamp_combos = []
        for i in range(n_exp):
            cb = QComboBox()
            for l in range(n_lamps):
                cb.addItem(f"L{l + 1}", l)
            lamp = saved_burst[i] if i < len(saved_burst) else (i % n_lamps)
            idx = cb.findData(int(lamp) % n_lamps)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            self.ssvep_burst_form.addRow(f"Эксперимент {i + 1}:", cb)
            self._ssvep_burst_lamp_combos.append(cb)

    def _ssvep_sequence_rng(self, seed: int, *, burst: bool) -> random.Random:
        subseed = int(seed) if not burst else (int(seed) ^ 0xA5A5A5A5)
        return random.Random(subseed & 0x7FFFFFFF)

    def _finalize_ssvep_cue_sequences(self, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Уникальная лампа на experiment (cue); фиксируется на сессию по seed."""
        n_exp = int(self.spin_ssvep.value())
        n_lamps = self._ssvep_active_lamp_count()
        cont_raw = self._read_lamp_combo_sequence(self._ssvep_cont_lamp_combos)
        burst_raw = self._read_lamp_combo_sequence(self._ssvep_burst_lamp_combos)
        cont = ensure_ssvep_experiment_lamp_permutation(
            cont_raw,
            n_experiments=n_exp,
            n_active_lamps=n_lamps,
            rng=self._ssvep_sequence_rng(seed, burst=False),
        )
        burst = ensure_ssvep_experiment_lamp_permutation(
            burst_raw,
            n_experiments=n_exp,
            n_active_lamps=n_lamps,
            rng=self._ssvep_sequence_rng(seed, burst=True),
        )
        self._set_lamp_combo_sequence(self._ssvep_cont_lamp_combos, cont)
        self._set_lamp_combo_sequence(self._ssvep_burst_lamp_combos, burst)
        return cont, burst

    def _apply_ssvep_auto_shuffle(self) -> int:
        """Новые перестановки ламп (cont/burst независимо), без повторов если n_exp ≤ n_lamps."""
        self._rebuild_ssvep_lamp_rows()
        seed = int(random.randrange(1, 1 << 31))
        self._ssvep_lamp_shuffle_seed = seed
        cont, burst = self._finalize_ssvep_cue_sequences(seed)
        plog_info(
            f"SSVEP auto-shuffle seed={seed} "
            f"continuous={[x + 1 for x in cont]} burst={[x + 1 for x in burst]}"
        )
        return seed

    def _on_shuffle_ssvep_cont(self) -> None:
        n_exp = int(self.spin_ssvep.value())
        n_lamps = self._ssvep_active_lamp_count()
        seed = int(random.randrange(1, 1 << 31))
        seq = shuffle_ssvep_experiment_lamps(
            n_exp, n_lamps, self._ssvep_sequence_rng(seed, burst=False)
        )
        self._set_lamp_combo_sequence(self._ssvep_cont_lamp_combos, seq)
        self._ssvep_lamp_shuffle_seed = seed
        plog_info(f"SSVEP shuffle continuous: {[x + 1 for x in seq]}")

    def _on_shuffle_ssvep_burst(self) -> None:
        n_exp = int(self.spin_ssvep.value())
        n_lamps = self._ssvep_active_lamp_count()
        seed = int(random.randrange(1, 1 << 31))
        seq = shuffle_ssvep_experiment_lamps(
            n_exp, n_lamps, self._ssvep_sequence_rng(seed, burst=True)
        )
        self._set_lamp_combo_sequence(self._ssvep_burst_lamp_combos, seq)
        self._ssvep_lamp_shuffle_seed = seed
        plog_info(f"SSVEP shuffle burst: {[x + 1 for x in seq]}")

    def _toggle_settings(self) -> None:
        visible = not self._settings_panel.isVisible()
        self._settings_panel.setVisible(visible)
        if visible:
            self.btn_settings_toggle.setText("▲  Свернуть настройки")
            self.setMinimumHeight(500)
            if self.height() < 600:
                self.resize(self.width(), 660)
        else:
            self.btn_settings_toggle.setText("⚙  Развернуть настройки")
            self.setMinimumHeight(180)
            self.adjustSize()

    def _set_eeg_indicator(self, state: str) -> None:
        """off | stale | live"""
        if state == "live":
            self.lbl_eeg_led.setStyleSheet(_EEG_LED_STYLE_GREEN)
            self.lbl_eeg_led.setToolTip("Поток ЭЭГ: данные идут")
        elif state == "stale":
            self.lbl_eeg_led.setStyleSheet(_EEG_LED_STYLE_RED)
            self.lbl_eeg_led.setToolTip("Поток ЭЭГ: подключено, данных нет")
        else:
            self.lbl_eeg_led.setStyleSheet(_EEG_LED_STYLE_OFF)
            self.lbl_eeg_led.setToolTip("Поток ЭЭГ: не подключено")

    def _eeg_indicator_state_from_runner(self) -> str | None:
        return eeg_indicator_state(self._runner_view())

    def _on_eeg_indicator_tick(self) -> None:
        runner_state = self._eeg_indicator_state_from_runner()
        if runner_state is not None:
            self._set_eeg_indicator(runner_state)
            return
        if self._eeg_test_inlet is None:
            self._set_eeg_indicator("off")
            return
        try:
            try:
                _chunk, ts = self._eeg_test_inlet.pull_chunk(timeout=0.0, max_samples=64)
            except TypeError:
                _chunk, ts = self._eeg_test_inlet.pull_chunk(timeout=0.0)
            if ts:
                self._eeg_test_last_live_at = time.time()
        except Exception:
            pass
        if self._eeg_test_last_live_at > 0.0 and (
            time.time() - self._eeg_test_last_live_at
        ) < _EEG_LIVE_TIMEOUT_S:
            self._set_eeg_indicator("live")
        else:
            self._set_eeg_indicator("stale")

    def _close_eeg_test_inlet(self) -> None:
        if hasattr(self, "_eeg_indicator_timer"):
            self._eeg_indicator_timer.stop()
        if self._eeg_test_inlet is None:
            self._set_eeg_indicator(self._eeg_indicator_state_from_runner() or "off")
            return
        try:
            self._eeg_test_inlet.close_stream()
        except Exception:
            pass
        self._eeg_test_inlet = None
        self._eeg_test_last_live_at = 0.0
        self._set_eeg_indicator(self._eeg_indicator_state_from_runner() or "off")

    def _refresh_eeg_streams(self) -> None:
        self._close_eeg_test_inlet()
        current_key: tuple[str, str] = ("", "")
        if self.cb_eeg.currentIndex() >= 0:
            data = self.cb_eeg.currentData()
            if isinstance(data, tuple) and len(data) == 2:
                current_key = (str(data[0]), str(data[1]))
        self.cb_eeg.blockSignals(True)
        self.cb_eeg.clear()
        try:
            streams = discover_eeg_streams(timeout=1.5)
        except Exception as e:
            self.cb_eeg.addItem(f"(ошибка поиска: {e})")
            self.cb_eeg.blockSignals(False)
            self.lbl_eeg_status.setText(f"Ошибка поиска потоков LSL: {e}")
            return
        if not streams:
            self.cb_eeg.addItem("(потоки ЭЭГ не найдены)")
            self.cb_eeg.blockSignals(False)
            self.lbl_eeg_status.setText("Потоки не найдены. Запустите усилитель/запись с LSL и нажмите «Обновить».")
            return
        for info in streams:
            self.cb_eeg.addItem(stream_display_label(info), userData=_eeg_stream_key(info))
        self.cb_eeg.blockSignals(False)
        if current_key[0]:
            for i in range(self.cb_eeg.count()):
                data = self.cb_eeg.itemData(i)
                if isinstance(data, tuple) and tuple(data) == current_key:
                    self.cb_eeg.setCurrentIndex(i)
                    break
        self._rebuild_channel_checkboxes_from_selection()

    def _set_channel_checks(self, checkboxes: list[QCheckBox], checked: bool) -> None:
        for cb in checkboxes:
            cb.setChecked(bool(checked))

    def _profile_channels_0idx(self, profile_key: str, n_channels: int) -> list[int]:
        if profile_key == "protocol_p300":
            return [c for c in _PROTOCOL_P300_ROI_0IDX if 0 <= c < n_channels]
        if profile_key == "protocol_ssvep":
            return [c for c in _PROTOCOL_SSVEP_ROI_0IDX if 0 <= c < n_channels]
        if profile_key == "all":
            return list(range(n_channels))
        if profile_key == "general":
            prof = get_analysis_profile(ANALYSIS_PROFILE_GENERAL)
            return [c for c in prof.roi_channels_0idx if 0 <= c < n_channels]
        if profile_key == "recent":
            prof = get_analysis_profile(ANALYSIS_PROFILE_RECENT)
            return [c for c in prof.roi_channels_0idx if 0 <= c < n_channels]
        return []

    def _apply_channel_profile(self, checkboxes: list[QCheckBox], profile_key: str) -> None:
        if not checkboxes or profile_key == "custom":
            return
        n = len(checkboxes)
        if profile_key == "all":
            self._set_channel_checks(checkboxes, True)
            return
        want = set(self._profile_channels_0idx(profile_key, n))
        for i, cb in enumerate(checkboxes):
            cb.setChecked(i in want)

    def _build_channel_checkbox_grid(
        self,
        layout: QGridLayout,
        storage: list[QCheckBox],
        *,
        count: int,
        labels: list[str],
    ) -> None:
        storage.clear()
        _clear_grid_layout(layout)
        for i in range(count):
            text = labels[i] if i < len(labels) else f"Канал {i + 1}"
            if not text.startswith("Канал"):
                text = f"{i + 1}: {text}"
            cb = QCheckBox(text)
            storage.append(cb)
            row, col = divmod(i, _CHANNEL_COLUMNS)
            layout.addWidget(cb, row, col)

    def _rebuild_channel_checkboxes_from_selection(self) -> None:
        info = self._resolve_eeg_stream_info(timeout=0.8)
        if info is None:
            self._eeg_channel_count = 0
            self._p300_ch_checkboxes.clear()
            self._ssvep_ch_checkboxes.clear()
            _clear_grid_layout(self._p300_ch_layout)
            _clear_grid_layout(self._ssvep_ch_layout)
            self._p300_ch_host.hide()
            self._ssvep_ch_host.hide()
            self._lbl_p300_ch_hint.show()
            self._lbl_ssvep_ch_hint.show()
            self._on_eeg_selection_changed()
            return
        try:
            n = max(1, int(info.channel_count()))
        except Exception:
            n = 1
        self._eeg_channel_count = n
        labels = stream_channel_labels(info, n)
        self._build_channel_checkbox_grid(self._p300_ch_layout, self._p300_ch_checkboxes, count=n, labels=labels)
        self._build_channel_checkbox_grid(self._ssvep_ch_layout, self._ssvep_ch_checkboxes, count=n, labels=labels)
        self._lbl_p300_ch_hint.hide()
        self._lbl_ssvep_ch_hint.hide()
        self._p300_ch_host.show()
        self._ssvep_ch_host.show()
        self._apply_channel_profile(self._p300_ch_checkboxes, str(self.cb_p300_profile.currentData()))
        self._apply_channel_profile(self._ssvep_ch_checkboxes, str(self.cb_ssvep_profile.currentData()))
        self._on_eeg_selection_changed()

    def _selected_eeg_stream_key(self) -> tuple[str, str] | None:
        idx = self.cb_eeg.currentIndex()
        if idx < 0:
            return None
        data = self.cb_eeg.itemData(idx)
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        name, sid = str(data[0]), str(data[1])
        if not name or name.startswith("("):
            return None
        return (name, sid)

    def _resolve_eeg_stream_info(self, *, timeout: float = 1.5):
        key = self._selected_eeg_stream_key()
        if key is None:
            return None
        streams = discover_eeg_streams(timeout=timeout)
        return select_eeg_stream(streams, name=key[0], session_id=key[1])

    def _on_eeg_selection_changed(self) -> None:
        key = self._selected_eeg_stream_key()
        if key is None:
            self.lbl_eeg_status.setText("Выберите поток из списка.")
            return
        self.lbl_eeg_status.setText(
            f"Выбран: {key[0]!r}. Нажмите «Подключить» для проверки потока."
        )

    def _on_test_eeg_connect(self) -> None:
        # Повторный клик = отключить
        if self._eeg_test_inlet is not None:
            self._close_eeg_test_inlet()
            self.btn_test_eeg.setText("Подключить")
            self.lbl_eeg_status.setText("Поток отключён.")
            return
        key = self._selected_eeg_stream_key()
        if key is None:
            QMessageBox.warning(self, "ЭЭГ", "Сначала выберите поток в списке (кнопка «Обновить»).")
            return
        info = self._resolve_eeg_stream_info()
        if info is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                f"Поток «{key[0]}» не найден в сети LSL.\nЗапустите запись ЭЭГ и нажмите «Обновить».",
            )
            return
        try:
            inlet = stream_inlet_with_buffer(info, buffer_seconds=5)
            try:
                chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=32)
            except TypeError:
                chunk, ts = inlet.pull_chunk(timeout=1.0)
            n_samp = len(ts) if ts else 0
            name = info.name() or "?"
            ch = int(info.channel_count())
            fs = float(info.nominal_srate() or 0.0)
            self.lbl_eeg_status.setText(
                f"Подключено: {name}, {ch} кан., Fs={fs:g} Гц."
            )
            plog_info(f"ЭЭГ подключено: {stream_display_label(info)}, samples={n_samp}")
            self._eeg_test_inlet = inlet
            self._eeg_test_last_live_at = time.time() if n_samp > 0 else 0.0
            self._eeg_indicator_timer.start()
            self._on_eeg_indicator_tick()
            self.btn_test_eeg.setText("Отключить")
            self._rebuild_channel_checkboxes_from_selection()
        except Exception as e:
            self.lbl_eeg_status.setText(f"Ошибка подключения: {e}")
            QMessageBox.warning(self, "ЭЭГ", f"Не удалось подключиться к потоку:\n{e}")

    def _add_lamp_freq_row(self, initial_hz: float) -> None:
        if len(self._freq_combos) >= _MAX_LAMPS:
            return
        n = len(self._freq_combos) + 1
        combo = _make_lamp_freq_combo(initial_hz)
        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.addWidget(QLabel(f"Лампа {n}:"))
        row.addWidget(combo, stretch=1)
        self._freq_combos.append(combo)
        self._lamp_row_widgets.append(row_w)
        self._lamp_rows_layout.addWidget(row_w)

    def _on_add_lamp(self) -> None:
        if len(self._freq_combos) >= _MAX_LAMPS:
            return
        extras = (12.0, 15.0, 20.0, 8.57, 7.5)
        k = len(self._freq_combos)
        hz = float(extras[(k - len(DEFAULT_LAMP_FREQS_HZ)) % len(extras)]) if k >= len(DEFAULT_LAMP_FREQS_HZ) else 0.0
        self._add_lamp_freq_row(hz)
        self._update_lamp_buttons()

    def _on_remove_lamp(self) -> None:
        if len(self._freq_combos) <= len(DEFAULT_LAMP_FREQS_HZ):
            return
        row_w = self._lamp_row_widgets.pop()
        self._freq_combos.pop()
        self._lamp_rows_layout.removeWidget(row_w)
        row_w.deleteLater()
        self._update_lamp_buttons()

    def _update_lamp_buttons(self) -> None:
        n = len(self._freq_combos)
        self.btn_add_lamp.setEnabled(n < _MAX_LAMPS)
        self.btn_remove_lamp.setEnabled(n > len(DEFAULT_LAMP_FREQS_HZ))
        self.btn_add_lamp.setText(f"+ лампа ({n}/{_MAX_LAMPS})")

    def _refresh_com_ports(self) -> None:
        ports = _list_serial_ports()
        current = self.cb_com.currentText().strip()
        self.cb_com.blockSignals(True)
        self.cb_com.clear()
        if not ports:
            self.cb_com.addItem("(порты не найдены — введите вручную)")
        else:
            for p in ports:
                self.cb_com.addItem(p)
        self.cb_com.blockSignals(False)
        if current and self.cb_com.findText(current) >= 0:
            self.cb_com.setCurrentText(current)
        elif self.cb_com.findText(_DEFAULT_COM_PORT) >= 0:
            self.cb_com.setCurrentText(_DEFAULT_COM_PORT)
        elif ports:
            self.cb_com.setCurrentIndex(0)
        else:
            self.cb_com.setCurrentText(_DEFAULT_COM_PORT)

    def _on_p300_profile_changed(self, _index: int) -> None:
        self._apply_channel_profile(self._p300_ch_checkboxes, str(self.cb_p300_profile.currentData()))

    def _on_ssvep_profile_changed(self, _index: int) -> None:
        self._apply_channel_profile(self._ssvep_ch_checkboxes, str(self.cb_ssvep_profile.currentData()))

    def _ssvep_freqs_hz(self) -> tuple[float, ...]:
        return tuple(_combo_freq_hz(c) for c in self._freq_combos)

    def _cleanup_before_start(self) -> None:
        """Перед новым прогоном: закрыть старый runner/COM/LSL и убить зомби run_app."""
        self._cancel_stim_marker_wait()
        self._close_eeg_test_inlet()
        self._dismiss_ssvep_overlay()
        if self._stimulus_proc is not None:
            try:
                if self._stimulus_proc.poll() is None:
                    self._stimulus_proc.terminate()
                    try:
                        self._stimulus_proc.wait(timeout=3)
                    except Exception:
                        pass
            except Exception:
                pass
            self._stimulus_proc = None
        if self._runner is not None:
            try:
                view = self._runner.display_view()
                if view.state not in ("idle", "stopped"):
                    self._runner.stop(reason="restart")
            except Exception:
                pass
            self._runner = None
        self._session_dir = None
        self._timer.stop()
        self._update_experiment_buttons()

    def _runner_view(self) -> ProtocolDisplayView | None:
        if self._runner is None:
            return None
        return self._runner.display_view()

    def _update_experiment_buttons(self) -> None:
        r = self._runner
        view = self._runner_view()
        waiting_stim = bool(self._stim_marker_wait_active)
        idle = view is None or view.state in ("idle", "stopped")
        ready = session_ready(view)
        running = block_running(view)
        self.btn_prepare.setEnabled(idle and not waiting_stim)
        for btn in (self.btn_run_p300, self.btn_run_ssvep_burst, self.btn_run_ssvep_cont):
            btn.setEnabled(ready and not running and not waiting_stim)
        self.btn_stop.setEnabled(r is not None and not idle)

    def _spawn_stimulus(self, session_dir: Path) -> bool:
        """Запустить run_app.py с stim_control (без ожидания LSL)."""
        run_script = _ROOT / "run_app.py"
        if not run_script.exists():
            return False
        stim_args = [
            sys.executable,
            str(run_script),
            "--auto-random-protocol",
            "--no-analyzer",
            "--inter-trial-s",
            str(float(self.spin_inter_trial.value())),
            "--sequences",
            str(int(self.spin_sequences.value())),
            "--auto-plan-trials",
            "0",
            "--auto-plan-target-tile-id",
            str(int(self.spin_calib_target.value())),
            "--stim-control-dir",
            str(session_dir),
        ]
        self._stimulus_proc = subprocess.Popen(stim_args, cwd=str(_ROOT))
        return True

    def _stimulus_needed_now(self) -> bool:
        """Нужен живой PsychoPy (калибровка или P300 в main, не во время ССВП)."""
        view = self._runner_view()
        if view is None or not self.chk_run_stimulus.isChecked():
            return False
        if view.state in ("ssvep_continuous", "ssvep_burst", "finalize", "stopped", "idle", "preflight"):
            return False
        if view.ssvep_cue_visible or view.ssvep_blackout_visible:
            return False
        return view.state == "main"

    def _ensure_stimulus_for_p300(self) -> None:
        """После ССВП стимулятор убивается — перед P300 в очереди перезапускаем."""
        if not self._stimulus_needed_now():
            return
        if self._session_dir is None:
            return
        if self._stimulus_proc_usable():
            return
        self._stop_stimulus_proc()
        now = time.time()
        if now - float(self._last_stim_restart_attempt) < 8.0:
            return
        self._last_stim_restart_attempt = now
        if self._spawn_stimulus(self._session_dir):
            plog_info("перезапуск PsychoPy для P300 (после этапа ССВП)")

    def _cancel_stim_marker_wait(self) -> None:
        """Stop async BCI_StimMarkers wait (operator stop/close/restart)."""
        if self._stim_marker_wait_timer.isActive():
            self._stim_marker_wait_timer.stop()
        self._stim_marker_wait_active = False
        self._stim_marker_wait_deadline = 0.0
        self._stim_marker_wait_pending_p300 = False

    def _begin_stim_marker_wait(self, *, pending_p300: bool) -> None:
        """Non-blocking wait for PsychoPy marker stream (QTimer poll, no processEvents/sleep)."""
        self._cancel_stim_marker_wait()
        self._stim_marker_wait_active = True
        self._stim_marker_wait_pending_p300 = bool(pending_p300)
        self._stim_marker_wait_deadline = time.time() + float(_STIM_MARKER_WAIT_SEC)
        self.lbl_status.setText(
            f"Ожидание {BCI_STIM_MARKER_STREAM_NAME}… ({_STIM_MARKER_WAIT_SEC:.0f} с)"
        )
        self._update_experiment_buttons()
        self._stim_marker_wait_timer.start()

    def _fail_stim_marker_wait(self, *, user_message: str) -> None:
        self._cancel_stim_marker_wait()
        try:
            if self._stimulus_proc is not None and self._stimulus_proc.poll() is None:
                self._stimulus_proc.terminate()
        except Exception:
            pass
        self._stimulus_proc = None
        self.lbl_status.setText(user_message)
        self._update_experiment_buttons()
        QMessageBox.warning(self, "Стимулятор", user_message)

    def _on_stim_marker_wait_tick(self) -> None:
        if not self._stim_marker_wait_active:
            self._stim_marker_wait_timer.stop()
            return
        if self._runner is None and self._stim_marker_wait_pending_p300:
            self._cancel_stim_marker_wait()
            self._update_experiment_buttons()
            return
        info_mk, _ = probe_stimulus_marker_stream(timeout=0.7)
        if info_mk is not None:
            try:
                plog_info(f"стимулятор LSL OK: {info_mk.name()!r}")
            except Exception:
                plog_info("стимулятор LSL OK")
            pending_p300 = self._stim_marker_wait_pending_p300
            self._cancel_stim_marker_wait()
            self._update_experiment_buttons()
            if pending_p300:
                self._start_p300_block_after_stimulus_ready()
            return
        remain = max(0.0, self._stim_marker_wait_deadline - time.time())
        if remain <= 0.0:
            self._fail_stim_marker_wait(
                user_message=(
                    f"Поток «{BCI_STIM_MARKER_STREAM_NAME}» не появился за "
                    f"{_STIM_MARKER_WAIT_SEC:.0f} с.\n"
                    "Дождитесь окна PsychoPy и повторите."
                ),
            )
            return
        self.lbl_status.setText(
            f"Ожидание {BCI_STIM_MARKER_STREAM_NAME}… ({remain:.0f} с)"
        )

    def _build_runner_config(self) -> ProtocolConfig | None:
        last_name = self.ed_last_name.text().strip()
        first_name = self.ed_first_name.text().strip()
        subject = sanitize_subject_slug(last_name, first_name)
        out_root = self.ed_output.text().strip()
        com = self.cb_com.currentText().strip()
        if not last_name or not first_name:
            QMessageBox.warning(self, "Испытуемый", "Введите фамилию и имя испытуемого.")
            return None
        if not out_root:
            QMessageBox.warning(self, "Результаты", "Укажите папку для сохранения результатов.")
            return None
        if not com or com.startswith("("):
            QMessageBox.warning(
                self,
                "COM-порт",
                "Выберите COM-порт мигалки из списка или введите вручную (COM3, /dev/tty.usbmodem…).\n"
                "Нажмите «Обновить», если устройство подключено позже.",
            )
            return None

        eeg_key = self._selected_eeg_stream_key()
        if eeg_key is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                "Выберите поток ЭЭГ (LSL): «Обновить» → выбор в списке → «Подключить».",
            )
            return None
        eeg_name, eeg_sid = eeg_key
        if self._resolve_eeg_stream_info() is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                f"Поток «{eeg_name}» сейчас не найден в LSL. Нажмите «Обновить» и проверьте запись ЭЭГ.",
            )
            return None

        if not self._p300_ch_checkboxes:
            QMessageBox.warning(self, "P300", "Выберите поток ЭЭГ и дождитесь списка каналов.")
            return None
        if not self._ssvep_ch_checkboxes:
            QMessageBox.warning(self, "ССВП", "Выберите поток ЭЭГ и дождитесь списка каналов.")
            return None
        roi = _selected_channels_0idx(self._p300_ch_checkboxes)
        ssvep_roi = _selected_channels_0idx(self._ssvep_ch_checkboxes)
        if not any(cb.isChecked() for cb in self._p300_ch_checkboxes):
            QMessageBox.warning(self, "P300", "Отметьте хотя бы один канал ЭЭГ (или «Все»).")
            return None
        if not any(cb.isChecked() for cb in self._ssvep_ch_checkboxes):
            QMessageBox.warning(self, "ССВП", "Отметьте хотя бы один канал ЭЭГ (или «Все»).")
            return None
        freqs = self._ssvep_freqs_hz()
        if not any(float(f) > 0 for f in freqs):
            QMessageBox.warning(self, "Мигалка", "Задайте хотя бы одну лампу с частотой > 0.")
            return None

        session_dir = UnifiedExperimentLogger.allocate_session_dir(
            output_root=Path(out_root),
            subject_id=subject,
        )
        self._session_dir = session_dir

        return ProtocolConfig(
            output_root=Path(out_root),
            subject_id=subject,
            subject_last_name=last_name,
            subject_first_name=first_name,
            session_dir=session_dir,
            com_port=com,
            eeg_stream_name=eeg_name,
            eeg_stream_session_id=eeg_sid,
            p300_calib_trials=int(self.spin_calib_trials.value()),
            calib_target_tile_id=int(self.spin_calib_target.value()),
            template_warmup_target_epochs=int(self.spin_template_epochs.value()),
            p300_main_trials=int(self.spin_p300_main.value()),
            ssvep_experiments_per_mode=int(self.spin_ssvep.value()),
            ssvep_group_size=4,
            ssvep_blocks_per_mode=0,
            ssvep_continuous_lamps_0idx=self._read_lamp_combo_sequence(self._ssvep_cont_lamp_combos),
            ssvep_burst_lamps_0idx=self._read_lamp_combo_sequence(self._ssvep_burst_lamp_combos),
            ssvep_auto_shuffle_on_prepare=bool(self.chk_ssvep_auto_shuffle.isChecked()),
            ssvep_lamp_shuffle_seed=int(self._ssvep_lamp_shuffle_seed),
            pause_between_experiments_s=float(self.spin_pause.value()),
            shuffle_seed=-1,
            ssvep_block_sec=float(self.spin_ssvep_block_sec.value()),
            ssvep_fs_hz=float(self.spin_ssvep_msi_fs.value()),
            ssvep_msi_samples=int(self.spin_ssvep_msi_samples.value()),
            ssvep_window_sec=float(self.spin_ssvep_msi_samples.value())
            / max(1.0, float(self.spin_ssvep_msi_fs.value())),
            ssvep_freqs_hz=freqs,
            roi_channels_0idx=roi if roi else _PROTOCOL_P300_ROI_0IDX,
            ssvep_roi_channels_0idx=ssvep_roi if ssvep_roi else _PROTOCOL_SSVEP_ROI_0IDX,
        )

    def _on_runner_ssvep_sequence_committed(self, mode: str, lamps: tuple[int, ...]) -> None:
        """Runtime source of truth: обновить combos после shuffle при старте режима."""
        if str(mode) == "continuous":
            self._set_lamp_combo_sequence(self._ssvep_cont_lamp_combos, lamps)
        else:
            self._set_lamp_combo_sequence(self._ssvep_burst_lamp_combos, lamps)

    def _on_prepare_session(self) -> None:
        self._cleanup_before_start()
        self._rebuild_ssvep_lamp_rows()
        cfg = self._build_runner_config()
        if cfg is None:
            return
        self._runner = ProtocolRunner(cfg)
        self._runner.set_ssvep_display_clear_callback(self._restore_operator_window)
        self._runner.set_ssvep_cue_sequence_callback(self._on_runner_ssvep_sequence_committed)
        plog_info(
            f"GUI prepare: subject={cfg.subject_id!r} EEG={cfg.eeg_stream_name!r} "
            f"calib={cfg.p300_calib_trials} main_P300={cfg.p300_main_trials} "
            f"SSVEP_exp={cfg.ssvep_experiments_per_mode}×{cfg.ssvep_group_size} COM={cfg.com_port!r} "
            f"(SSVEP cue shuffle at mode start, not prepare)"
        )
        self._runner.start()
        self._timer.start()
        self._eeg_indicator_timer.start()
        self._update_experiment_buttons()
        self.lbl_status.setText("Подготовка сессии: поиск LSL и COM…")

    def _stimulus_proc_usable(self) -> bool:
        """Живой процесс и stim_control не в state=done (после прошлого блока)."""
        if self._stimulus_proc is None or self._stimulus_proc.poll() is not None:
            return False
        if self._session_dir is None:
            return True
        try:
            if stim_adapter.is_stim_session_done(self._session_dir):
                return False
        except Exception:
            pass
        return True

    def _ensure_stimulus_for_p300_block(self) -> str:
        """Prepare PsychoPy for P300. Returns ready | pending | failed."""
        if not self.chk_run_stimulus.isChecked():
            return "ready"
        if self._session_dir is None:
            return "failed"
        if self._stimulus_proc_usable():
            return "ready"
        self._stop_stimulus_proc()
        if not self._spawn_stimulus(self._session_dir):
            QMessageBox.warning(self, "Стимулятор", "Не найден run_app.py.")
            return "failed"
        plog_info("запуск стимулятора для блока P300")
        self._begin_stim_marker_wait(pending_p300=True)
        return "pending"

    def _start_p300_block_after_stimulus_ready(self) -> None:
        if self._runner is None or not session_ready(self._runner_view()):
            QMessageBox.warning(self, "P300", "Сначала нажмите «Подготовить сессию».")
            return
        if not self._runner.start_p300_block():
            QMessageBox.warning(self, "P300", "Не удалось запустить блок P300.")
            return
        self._update_experiment_buttons()
        plog_info("оператор: старт блока P300")

    def _on_run_p300(self) -> None:
        if self._runner is None or not session_ready(self._runner_view()):
            QMessageBox.warning(self, "P300", "Сначала нажмите «Подготовить сессию».")
            return
        stim = self._ensure_stimulus_for_p300_block()
        if stim == "pending":
            return
        if stim == "failed":
            return
        self._start_p300_block_after_stimulus_ready()

    def _on_run_ssvep(self, mode: str) -> None:
        if self._runner is None or not session_ready(self._runner_view()):
            QMessageBox.warning(self, "ССВП", "Сначала нажмите «Подготовить сессию».")
            return
        if not self._runner.start_ssvep_block(mode=str(mode)):
            QMessageBox.warning(self, "ССВП", f"Не удалось запустить ССВП ({mode}).")
            return
        self._update_experiment_buttons()
        plog_info(f"оператор: старт ССВП mode={mode}")

    def _dismiss_ssvep_overlay(self) -> None:
        if self._ssvep_cue_overlay is not None:
            ov = self._ssvep_cue_overlay
            self._ssvep_cue_overlay = None
            ov.setVisible(False)
            ov.hide()
            ov.close()
            ov.deleteLater()
        QApplication.processEvents()

    def _stop_stimulus_proc(self) -> None:
        """Закрыть PsychoPy, чтобы fullscreen не перекрывал окно оператора."""
        if self._stimulus_proc is None:
            return
        proc = self._stimulus_proc
        self._stimulus_proc = None
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
        except Exception:
            pass
        QApplication.processEvents()

    def _raise_operator_window(self) -> None:
        """Снять оверлей ССВП и вывести окно оператора (стимулятор не трогаем)."""
        if self._runner is not None:
            self._runner.dismiss_operator_ssvep_overlay()
        self._dismiss_ssvep_overlay()
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def _restore_operator_window(self) -> None:
        """После блока / остановки — оверлей снять и закрыть PsychoPy."""
        self._raise_operator_window()
        self._stop_stimulus_proc()

    def _sync_ssvep_cue_overlay(self) -> None:
        view = self._runner_view()
        if view is None:
            self._dismiss_ssvep_overlay()
            return
        if view.state in ("stopped",):
            self._restore_operator_window()
            return
        if view.ssvep_blackout_visible:
            if self._ssvep_cue_overlay is None:
                self._ssvep_cue_overlay = SsvepCueOverlay()
            self._ssvep_cue_overlay.show_blackout()
            return
        if not view.ssvep_cue_visible:
            if self._ssvep_cue_overlay is not None and not view.ssvep_blackout_visible:
                self._dismiss_ssvep_overlay()
            return
        if self._ssvep_cue_overlay is None:
            self._ssvep_cue_overlay = SsvepCueOverlay()
        self._ssvep_cue_overlay.show_cue(
            experiment_index=int(view.ssvep_cue_exp_index),
            experiment_total=int(view.ssvep_cue_exp_total),
            lamp_1based=int(view.ssvep_cue_lamp_1based),
            freq_hz=float(view.ssvep_cue_freq_hz),
            mode_label=str(view.ssvep_cue_mode_label),
            group_index=int(view.ssvep_cue_group_index),
        )
        self._ssvep_cue_overlay.raise_()
        self._ssvep_cue_overlay.activateWindow()

    def _on_stop(self) -> None:
        self._cancel_stim_marker_wait()
        self._close_eeg_test_inlet()
        self._dismiss_ssvep_overlay()
        if self._stimulus_proc is not None:
            try:
                if self._stimulus_proc.poll() is None:
                    self._stimulus_proc.terminate()
            except Exception:
                pass
            self._stimulus_proc = None
        if self._runner is None:
            return
        self._runner.stop(reason="user_stop")
        self._runner = None
        self._session_dir = None
        self._timer.stop()
        self._update_experiment_buttons()
        self.lbl_status.setText("Сессия завершена. Логи сохранены.")

    def _on_tick(self) -> None:
        if self._runner is None:
            return
        prev_state = str(self._last_runner_state)
        prev_status = getattr(self, "_last_status_printed", "")
        self._runner.tick()
        self._ensure_stimulus_for_p300()
        self._sync_ssvep_cue_overlay()
        QApplication.processEvents()
        # processEvents() may run the Stop handler and clear self._runner under us.
        if self._runner is None:
            return
        view = self._runner.display_view()
        st = view.status_text
        self.lbl_status.setText(st)
        if st != prev_status:
            plog_info(f"status [{view.state}]: {st}")
            self._last_status_printed = st
        cur_state = str(view.state)
        block_just_finished = (
            cur_state == "ready"
            and prev_state in ("main", "ssvep_continuous", "ssvep_burst")
        )
        if block_just_finished:
            self._restore_operator_window()
            plog_info("блок завершён: оверлей и стимулятор закрыты")
        # Закрываем PsychoPy до ССВП: иначе fullscreen перекрывает оверлей и мигалку не видно
        elif self._stimulus_proc is not None and self._stimulus_proc.poll() is None:
            stop_stim = view.state in ("ssvep_continuous", "ssvep_burst", "finalize", "stopped")
            if view.ssvep_cue_visible or view.ssvep_blackout_visible:
                stop_stim = True
            if stop_stim:
                self._stop_stimulus_proc()
                if prev_status != st and (
                    view.state in ("ssvep_continuous", "ssvep_burst") or view.ssvep_cue_visible
                ):
                    plog_info("стимулятор плиток остановлен — этап ССВП (оверлей / мигалка)")
        self._last_runner_state = cur_state
        self._update_experiment_buttons()
        self._on_eeg_indicator_tick()
        if view.state in ("stopped",):
            self._timer.stop()
            self._restore_operator_window()
            self._runner = None
            self._last_runner_state = ""
            self._update_experiment_buttons()
        elif (
            session_ready(view)
            and not block_running(view)
            and not view.pause_active
            and not view.ssvep_cue_visible
            and not view.ssvep_blackout_visible
        ):
            # Только поднять GUI; не снимать SSVEP-оверлей и не мешать паузе перед блоком.
            self._raise_operator_window()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Гарантированный session_stop при закрытии окна без кнопки «Стоп»."""
        if self._runner is not None:
            try:
                self._on_stop()
            except Exception:
                pass
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ProtocolRunnerWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

