"""Полноэкранная подсказка перед блоком ССВП (номер эксперимента и лампа-диод)."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from experiment_protocol.subject_display import format_subject_experiment_title

# Номер лампы (1…4) → направление взгляда испытуемого
_LAMP_ARROW: dict[int, str] = {1: "←", 2: "→", 3: "↑", 4: "↓"}
_LAMP_DIRECTION_RU: dict[int, str] = {
    1: "влево",
    2: "вправо",
    3: "вверх",
    4: "вниз",
}


def lamp_direction_arrow(lamp_1based: int) -> str:
    return _LAMP_ARROW.get(int(lamp_1based), "•")


def lamp_direction_label_ru(lamp_1based: int) -> str:
    return _LAMP_DIRECTION_RU.get(int(lamp_1based), "")


class SsvepCueOverlay(QWidget):
    """Серый fullscreen, как оверлей паузы в PsychoPy-стимуляторе."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet("background-color: #202020;")
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.addStretch(2)

        self.lbl_title = QLabel("")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("color: white;")
        self.lbl_title.setFont(QFont("", 42, QFont.Bold))
        root.addWidget(self.lbl_title)

        self.lbl_mode = QLabel("")
        self.lbl_mode.setAlignment(Qt.AlignCenter)
        self.lbl_mode.setStyleSheet("color: #cccccc;")
        self.lbl_mode.setFont(QFont("", 22))
        root.addWidget(self.lbl_mode)

        self.lbl_sub = QLabel("Смотрите в указанном направлении")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("color: white;")
        self.lbl_sub.setFont(QFont("", 28))
        root.addWidget(self.lbl_sub)

        self.lbl_arrow = QLabel("")
        self.lbl_arrow.setAlignment(Qt.AlignCenter)
        self.lbl_arrow.setStyleSheet("color: #ffdd44;")
        self.lbl_arrow.setFont(QFont("", 140, QFont.Bold))
        root.addWidget(self.lbl_arrow)

        self.lbl_direction = QLabel("")
        self.lbl_direction.setAlignment(Qt.AlignCenter)
        self.lbl_direction.setStyleSheet("color: #cccccc;")
        self.lbl_direction.setFont(QFont("", 26))
        root.addWidget(self.lbl_direction)

        self.lbl_lamp = QLabel("")
        self.lbl_lamp.setAlignment(Qt.AlignCenter)
        self.lbl_lamp.setStyleSheet("color: #ffdd44;")
        self.lbl_lamp.setFont(QFont("", 96, QFont.Bold))
        root.addWidget(self.lbl_lamp)

        self.lbl_hz = QLabel("")
        self.lbl_hz.setAlignment(Qt.AlignCenter)
        self.lbl_hz.setStyleSheet("color: #ffdd44;")
        self.lbl_hz.setFont(QFont("", 32))
        root.addWidget(self.lbl_hz)

        root.addStretch(3)
        self.hide()
        self._shown_key: tuple[int, int, int, float, str, int] | None = None
        self._cue_widgets = (
            self.lbl_title,
            self.lbl_mode,
            self.lbl_sub,
            self.lbl_arrow,
            self.lbl_direction,
            self.lbl_lamp,
            self.lbl_hz,
        )

    def _show_on_primary_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.setWindowState(Qt.WindowNoState)
        self.show()
        self.raise_()

    def show_blackout(self) -> None:
        """Чёрный экран на ноутбуке, пока испытуемый смотрит на мигалку."""
        self._shown_key = None
        self.setStyleSheet("background-color: #000000;")
        for w in self._cue_widgets:
            w.hide()
        self._show_on_primary_screen()

    def show_cue(
        self,
        *,
        experiment_index: int,
        experiment_total: int,
        lamp_1based: int,
        freq_hz: float | None = None,
        mode_label: str = "",
        group_index: int = -1,
    ) -> None:
        self.setStyleSheet("background-color: #202020;")
        for w in self._cue_widgets:
            w.show()
        key = (
            int(experiment_index),
            int(experiment_total),
            int(lamp_1based),
            float(freq_hz or 0.0),
            str(mode_label),
            int(group_index),
        )
        lamp_n = int(lamp_1based)
        self.lbl_title.setText(
            format_subject_experiment_title(
                index=int(experiment_index),
                total=int(experiment_total),
            )
        )
        self.lbl_mode.setText("")
        self.lbl_mode.hide()
        self.lbl_arrow.setText(lamp_direction_arrow(lamp_n))
        dir_ru = lamp_direction_label_ru(lamp_n)
        self.lbl_direction.setText(f"Смотрите {dir_ru}" if dir_ru else "")
        self.lbl_lamp.setText("")
        self.lbl_lamp.hide()
        self.lbl_hz.setText("")
        self.lbl_hz.hide()
        self._shown_key = key
        self._show_on_primary_screen()

    def hide_overlay(self) -> None:
        self._shown_key = None
        self.hide()
        self.close()
