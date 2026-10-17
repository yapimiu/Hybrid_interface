"""Полноэкранные подсказки испытуемому (P300 / ССВП / пауза / калибровка)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from experiment_protocol.subject_display import format_subject_experiment_title

__all__ = ["ProtocolInstructionOverlay", "format_subject_experiment_title"]


class ProtocolInstructionOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet("background-color: #1a1a2e;")
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 48, 48, 48)
        root.addStretch(2)

        self.lbl_title = QLabel("")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("color: #eaeaea;")
        self.lbl_title.setFont(QFont("", 40, QFont.Weight.Bold))
        root.addWidget(self.lbl_title)

        self.lbl_kind = QLabel("")
        self.lbl_kind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_kind.setStyleSheet("color: #a0a0b8;")
        self.lbl_kind.setFont(QFont("", 24))
        root.addWidget(self.lbl_kind)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setStyleSheet("color: #ffffff;")
        self.lbl_hint.setFont(QFont("", 30))
        root.addWidget(self.lbl_hint)

        self.lbl_big = QLabel("")
        self.lbl_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_big.setStyleSheet("color: #7ee787;")
        self.lbl_big.setFont(QFont("", 88, QFont.Weight.Bold))
        root.addWidget(self.lbl_big)

        self.lbl_sub = QLabel("")
        self.lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sub.setStyleSheet("color: #8b949e;")
        self.lbl_sub.setFont(QFont("", 20))
        root.addWidget(self.lbl_sub)

        root.addStretch(3)
        self.hide()
        self._key: tuple | None = None

    def _show_fullscreen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.show()
        self.raise_()

    def hide_overlay(self) -> None:
        self._key = None
        self.hide()
        self.close()

    def show_message(
        self,
        *,
        title: str,
        kind_label: str = "",
        hint: str = "",
        big_text: str = "",
        sub_text: str = "",
    ) -> None:
        key = (title, kind_label, hint, big_text, sub_text)
        if key != self._key:
            self.lbl_title.setText(title)
            self.lbl_kind.setText(kind_label)
            self.lbl_kind.setVisible(bool(kind_label))
            self.lbl_hint.setText(hint)
            self.lbl_hint.setVisible(bool(hint))
            self.lbl_big.setText(big_text)
            self.lbl_big.setVisible(bool(big_text))
            self.lbl_sub.setText(sub_text)
            self.lbl_sub.setVisible(bool(sub_text))
            self._key = key
        self._show_fullscreen()

    def show_p300_cue(
        self,
        *,
        experiment_index: int,
        experiment_total: int,
        target_tile_id: int,
        phase_label: str = "P300",
    ) -> None:
        phase = str(phase_label or "").strip().lower()
        title = format_subject_experiment_title(
            index=int(experiment_index),
            total=int(experiment_total),
            calib="калибр" in phase,
        )
        self.show_message(
            title=title,
            kind_label="",
            hint="Смотрите на указанную плитку",
            big_text=str(int(target_tile_id)),
            sub_text="Не двигайтесь",
        )

    def show_ssvep_cue(
        self,
        *,
        experiment_index: int,
        experiment_total: int,
        lamp_1based: int,
        freq_hz: float,
        mode_label: str,
    ) -> None:
        self.show_message(
            title=format_subject_experiment_title(
                index=int(experiment_index),
                total=int(experiment_total),
            ),
            kind_label="",
            hint="Смотрите в указанном направлении",
            big_text="",
            sub_text="Не двигайтесь",
        )

    def show_pause(self, *, message: str, seconds_left: float | None = None) -> None:
        self.show_message(
            title="Ожидание",
            kind_label="",
            hint=str(message) if message else "Подготовьтесь",
            big_text="",
            sub_text="Не двигайтесь",
        )

    def show_blackout(self) -> None:
        self.setStyleSheet("background-color: #000000;")
        for w in (
            self.lbl_title,
            self.lbl_kind,
            self.lbl_hint,
            self.lbl_big,
            self.lbl_sub,
        ):
            w.hide()
        self._show_fullscreen()

    def show_cue_widgets(self) -> None:
        self.setStyleSheet("background-color: #1a1a2e;")
        for w in (
            self.lbl_title,
            self.lbl_kind,
            self.lbl_hint,
            self.lbl_big,
            self.lbl_sub,
        ):
            w.show()
