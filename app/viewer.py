"""Приём и отображение LSL сигнала ЭЭГ.

Последний принятый блок доступен как SignalViewerWindow.signal_array:
numpy.ndarray формы (число отсчётов, число каналов). Значения остаются сырыми;
очистка NaN/Inf применяется только при отрисовке.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLayout, QScrollArea, QVBoxLayout, QWidget,
    QMainWindow, QPlainTextEdit, QSizePolicy, QPushButton, QFileDialog, QMessageBox,
)
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_byprop

from app.recording import CsvRecording


WINDOW_SEC = 0.2
UPDATE_INTERVAL_MS = 50
STATS_INTERVAL_MS = 500
LSL_STREAM_TYPES = ("EEG", "Signal")

pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)


def get_channel_names(info, count: int) -> list[str]:
    """Читать имена каналов из LSL, подставляя номера при отсутствии метаданных."""
    names = []
    try:
        channel = info.desc().child("channels").child("channel")
        for _ in range(count):
            if channel.empty():
                break
            names.append(channel.child_value("label") or channel.child_value("name") or "")
            channel = channel.next_sibling()
    except Exception:
        pass
    return [names[i] if i < len(names) and names[i] else f"Ch {i + 1}" for i in range(count)]


class LSLReceiver(QThread):
    """Ищет LSL поток и читает его без блокировки интерфейса."""

    connected = pyqtSignal(object)
    disconnected = pyqtSignal()
    chunk_received = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._inlet: Optional[StreamInlet] = None

    def stop(self):
        self._running = False
        self.wait(2500)

    def run(self):
        while self._running:
            info = None
            for stream_type in LSL_STREAM_TYPES:
                if not self._running:
                    return
                try:
                    streams = resolve_byprop("type", stream_type, timeout=0.5)
                    if streams:
                        info = streams[0]
                        break
                except Exception:
                    pass
            if info is None:
                continue

            try:
                self._inlet = StreamInlet(info, max_buflen=10)
                self._inlet.open_stream(timeout=1.0)
                self.connected.emit(info)
                while self._running:
                    chunk, timestamps = self._inlet.pull_chunk(timeout=0.2, max_samples=4096)
                    if chunk:
                        array = np.asarray(chunk, dtype=np.float64)
                        if array.ndim == 2 and array.shape[1] == info.channel_count():
                            self.chunk_received.emit(array, np.asarray(timestamps, dtype=np.float64))
            except Exception as exc:
                if self._running:
                    print(f"LSL: соединение прервано: {exc}", file=sys.stderr)
                    time.sleep(0.2)
            finally:
                if self._inlet is not None:
                    self._inlet.close_stream()
                    self._inlet = None
                if self._running:
                    self.disconnected.emit()


class InterfaceScrollArea(QScrollArea):
    """Прокрутка окна колесом мыши, в том числе поверх графиков."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel and self.verticalScrollBar().maximum() > 0:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            if delta:
                scrollbar = self.verticalScrollBar()
                scrollbar.setValue(scrollbar.value() - delta)
                event.accept()
                return True
        return super().eventFilter(watched, event)


class ChannelWidget(QFrame):
    """График канала в прежнем тёмном оформлении."""

    def __init__(self, channel_id: int, channel_name: str, sampling_rate: float):
        super().__init__()
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate if sampling_rate > 0 else 256.0
        self.buffer_size = max(1, int(WINDOW_SEC * self.sampling_rate))
        self.y_data = np.zeros(self.buffer_size, dtype=np.float32)
        self.x_data = np.linspace(-WINDOW_SEC, 0, self.buffer_size, dtype=np.float32)
        self.filled = 0
        self.current_y_min = -10.0
        self.current_y_max = 10.0
        self.is_active = True

        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(180)
        self.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        header = QLabel(f"[{channel_id + 1}] {channel_name}")
        header.setFont(QFont("Arial", 8, QFont.Bold))
        header.setStyleSheet("color: white; background-color: transparent; border: none; padding: 2px;")
        layout.addWidget(header)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setDownsampling(mode=None)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.getViewBox().setLimits(yMin=-1000000, yMax=1000000)
        self.plot_widget.setLabel("left", "мкВ", color="white", fontsize=6)
        self.plot_widget.setBackground("#000000")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setXRange(-WINDOW_SEC, 0, padding=0)
        color = pg.intColor(channel_id, hues=15, values=1, maxValue=255, minValue=150)
        self.signal_line = self.plot_widget.plot(
            self.x_data, self.y_data, pen=pg.mkPen(color, width=1.0), autoDownsample=False
        )
        layout.addWidget(self.plot_widget)
        self.stats_label = QLabel("Ожидание...")
        self.stats_label.setFont(QFont("Courier", 7))
        self.stats_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.stats_label.setStyleSheet("color: #a8ff9e; background-color: transparent; border: none;")
        layout.addWidget(self.stats_label)

    def push_chunk(self, values: np.ndarray):
        if not self.is_active or len(values) == 0:
            return
        clean = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), -250000.0, 250000.0)
        n = len(clean)
        self.filled = min(self.buffer_size, self.filled + n)
        if n >= self.buffer_size:
            self.y_data[:] = clean[-self.buffer_size:]
        else:
            self.y_data[:-n] = self.y_data[n:]
            self.y_data[-n:] = clean

    def update_plot(self, auto_scale: bool):
        if not self.is_active or not self.filled or not self.isVisible():
            return
        self.signal_line.setData(self.x_data, self.y_data, skipFiniteCheck=True)
        if auto_scale:
            data = self.y_data[-self.filled:]
            low, high = float(np.min(data)), float(np.max(data))
            span = high - low
            if span < 20.0:
                center = (high + low) / 2
                low, high = center - 10.0, center + 10.0
            else:
                low, high = low - span * 0.15, high + span * 0.15
            if abs(low - self.current_y_min) > 2.0 or abs(high - self.current_y_max) > 2.0:
                self.set_y_range(low, high)

    def set_y_range(self, low: float, high: float):
        if low >= high:
            return
        self.current_y_min, self.current_y_max = low, high
        self.plot_widget.setYRange(low, high, padding=0, update=False)

    def set_time_window(self, seconds: float):
        new_size = max(1, int(seconds * self.sampling_rate))
        old = self.y_data
        self.y_data = np.zeros(new_size, dtype=np.float32)
        size = min(new_size, len(old))
        self.y_data[-size:] = old[-size:]
        self.x_data = np.linspace(-seconds, 0, new_size, dtype=np.float32)
        self.buffer_size = new_size
        self.filled = min(self.filled, new_size)
        self.plot_widget.setXRange(-seconds, 0, padding=0)

    def update_stats(self):
        if not self.is_active or self.filled < 10 or not self.isVisible():
            return
        data = self.y_data[-self.filled:]
        mean, std = float(np.mean(data)), float(np.std(data))
        rms = float(np.sqrt(np.mean(data ** 2)))
        p2p = float(np.max(data) - np.min(data))
        if p2p > 200000:
            self.stats_label.setStyleSheet("color: #ff4d4d; font-weight: bold; border: none;")
            self.stats_label.setText("ОШИБКА КОНТАКТА")
        else:
            self.stats_label.setStyleSheet("color: #a8ff9e; border: none;")
            self.stats_label.setText(f"Mean:{mean:6.1f} | STD:{std:6.1f} | RMS:{rms:6.1f} | P2P:{p2p:6.1f}")


class SignalViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signal_array = np.empty((0, 0), dtype=np.float64)
        self.timestamps_array = np.empty(0, dtype=np.float64)
        self.recording = CsvRecording()
        self._connected = False
        self._array_dirty = False
        self._pending_chunks: list[np.ndarray] = []
        self.channel_widgets: list[ChannelWidget] = []
        self.sample_count = 0
        self.stream_name = ""
        self.sampling_rate = 0.0
        self._setup_ui()

        self.receiver = LSLReceiver(self)
        self.receiver.connected.connect(self._on_connected)
        self.receiver.disconnected.connect(self._on_disconnected)
        self.receiver.chunk_received.connect(self._on_chunk)
        self.receiver.start()

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(UPDATE_INTERVAL_MS)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(STATS_INTERVAL_MS)
        self.array_timer = QTimer(self)
        self.array_timer.timeout.connect(self._update_array_view)
        self.array_timer.start(250)

    def _setup_ui(self):
        self.setWindowTitle("LSL — отображение сигнала")
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a0a; color: white; }
            QLabel { color: white; }
            QCheckBox { color: white; spacing: 5px; font-size: 13px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QPushButton { background-color: #333; color: white; border-radius: 3px; padding: 8px; }
            QPushButton:hover { background-color: #444; }
            QPushButton:disabled { color: #777; }
        """)
        self.content_scroll = InterfaceScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content_scroll.setStyleSheet("QScrollArea { background-color: #0a0a0a; border: none; }")
        self.setCentralWidget(self.content_scroll)
        central = QWidget()
        self.content_scroll.setWidget(central)
        main = QHBoxLayout(central)
        main.setSizeConstraint(QLayout.SetMinimumSize)
        main.setContentsMargins(5, 5, 5, 5)

        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        controls = QVBoxLayout(sidebar)
        controls.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Поиск LSL потока EEG/Signal...")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)

        self.record_button = QPushButton("Начать запись в CSV")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_recording)
        controls.addWidget(self.record_button)
        self.record_status = QLabel("Подключите LSL поток для записи")
        self.record_status.setWordWrap(True)
        self.record_status.setStyleSheet("color: #aaa;")
        controls.addWidget(self.record_status)

        self.cb_autoscale = QCheckBox("Автомасштаб Y")
        self.cb_autoscale.setChecked(True)
        self.cb_autoscale.setStyleSheet("QCheckBox { color: #5bc0be; font-weight: bold; margin-bottom: 10px; }")
        controls.addWidget(self.cb_autoscale)

        scale_y = QGroupBox("Масштаб Y (мкВ)")
        scale_y.setStyleSheet("QGroupBox { color: white; margin-top: 6px; }")
        y_layout = QVBoxLayout(scale_y)
        for title, attribute, value in (("Мин:", "spin_y_min", -100), ("Макс:", "spin_y_max", 100)):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            spin = QDoubleSpinBox()
            spin.setRange(-500000, 500000)
            spin.setValue(value)
            spin.setDecimals(1)
            spin.setSingleStep(10)
            spin.valueChanged.connect(self._apply_manual_scale)
            setattr(self, attribute, spin)
            row.addWidget(spin)
            y_layout.addLayout(row)
        controls.addWidget(scale_y)

        scale_x = QGroupBox("Масштаб X (с)")
        scale_x.setStyleSheet("QGroupBox { color: white; margin-top: 6px; }")
        x_layout = QHBoxLayout(scale_x)
        x_layout.addWidget(QLabel("Окно:"))
        self.spin_x_window = QDoubleSpinBox()
        self.spin_x_window.setRange(0.05, 60.0)
        self.spin_x_window.setValue(WINDOW_SEC)
        self.spin_x_window.setDecimals(2)
        self.spin_x_window.setSingleStep(0.1)
        self.spin_x_window.setSuffix(" с")
        self.spin_x_window.valueChanged.connect(self._apply_x_window)
        x_layout.addWidget(self.spin_x_window)
        controls.addWidget(scale_x)

        controls.addStretch()

        right = QWidget()
        self.right_layout = QVBoxLayout(right)
        self.placeholder = QLabel("Ожидание LSL потока EEG или Signal")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.right_layout.addWidget(self.placeholder, stretch=1)
        array_group = QGroupBox("Последний принятый массив")
        array_group.setStyleSheet("QGroupBox { color: #5bc0be; }")
        array_layout = QVBoxLayout(array_group)
        self.array_shape_label = QLabel("Ожидание данных. Строки — отсчёты, столбцы — каналы.")
        array_layout.addWidget(self.array_shape_label)
        self.array_view = QPlainTextEdit()
        self.array_view.setReadOnly(True)
        self.array_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.array_view.setFont(QFont("Menlo", 10))
        self.array_view.setStyleSheet("background-color: #111; color: #a8ff9e; border: 1px solid #333;")
        self.array_view.setPlaceholderText("Здесь появится массив принятых значений LSL.")
        self.array_view.setMinimumHeight(140)
        self.array_view.setMaximumHeight(240)
        array_layout.addWidget(self.array_view)
        self.right_layout.addWidget(array_group)
        main.addWidget(sidebar, alignment=Qt.AlignTop)
        main.addWidget(right, stretch=1)
        self.resize(1600, 1000)

    def _clear_channels(self):
        for widget in self.channel_widgets:
            self.grid_layout.removeWidget(widget)
            widget.deleteLater()
        self.channel_widgets.clear()

    def _on_connected(self, info):
        if self.recording.active:
            self._stop_recording("Запись завершена при смене потока")
        self._connected = True
        self.record_button.setEnabled(True)
        if self.recording.path is None:
            self.record_status.setText("Запись не начата")
        name = info.name() or "EEG"
        count = info.channel_count()
        rate = info.nominal_srate()
        self.stream_name = name
        self.sampling_rate = rate
        self.setWindowTitle(f"LSL — {name}")
        self.status_label.setText(f"Поток: {name}\nЧастота: {rate} Гц\nСэмплов: 0")
        self.signal_array = np.empty((0, count), dtype=np.float64)
        self.timestamps_array = np.empty(0, dtype=np.float64)
        self._array_dirty = True
        self._update_array_view()
        self.sample_count = 0
        self._pending_chunks.clear()
        if hasattr(self, "grid_layout"):
            self._clear_channels()
        else:
            self.placeholder.setParent(None)
            grid_widget = QWidget()
            self.grid_layout = QGridLayout(grid_widget)
            self.grid_layout.setSpacing(4)
            self.grid_layout.setContentsMargins(0, 0, 0, 0)
            self.right_layout.insertWidget(0, grid_widget, stretch=1)
        for index, channel_name in enumerate(get_channel_names(info, count)):
            widget = ChannelWidget(index, channel_name, rate)
            widget.set_time_window(self.spin_x_window.value())
            widget.plot_widget.viewport().installEventFilter(self.content_scroll)
            self.channel_widgets.append(widget)
        self._rebuild_grid()

    def _on_disconnected(self):
        self._connected = False
        if self.recording.active:
            self._stop_recording("Поток отключён. Запись завершена")
        self.record_button.setEnabled(False)
        self.status_label.setText("Связь потеряна. Поиск LSL потока...")

    def _on_chunk(self, array: np.ndarray, timestamps: np.ndarray):
        if not self.channel_widgets or array.shape[1] != len(self.channel_widgets):
            return
        self.signal_array = array
        self.timestamps_array = timestamps
        if self.recording.active:
            try:
                self.recording.append(array, timestamps)
            except (OSError, ValueError) as exc:
                self._recording_error(exc)
            else:
                self.record_status.setText(f"Идёт запись: {self.recording.sample_count} отсчётов")
        self._array_dirty = True
        self.sample_count += len(array)
        self._pending_chunks.append(array)
        # Графики показывают только последнее окно; длинная очередь не нужна.
        if len(self._pending_chunks) > 100:
            self._pending_chunks = self._pending_chunks[-100:]
        self.status_label.setText(
            f"Поток: {self.stream_name}\nЧастота: {self.sampling_rate} Гц\nСэмплов: {self.sample_count}"
        )

    def _toggle_recording(self):
        if self.recording.active:
            self._stop_recording()
            return
        default_name = datetime.now().strftime("lsl_%Y-%m-%d_%H-%M-%S.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить запись LSL", default_name, "CSV (*.csv)"
        )
        if not path or not self._connected:
            return
        if not Path(path).suffix:
            path += ".csv"
        try:
            self.recording.start(path, [widget.channel_name for widget in self.channel_widgets])
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось открыть файл:\n{exc}")
            return
        self.record_button.setText("Остановить и сохранить")
        self.record_status.setText("Идёт запись: 0 отсчётов")
        self.record_status.setToolTip(str(self.recording.path))

    def _stop_recording(self, message: str = "Запись сохранена"):
        try:
            self.recording.stop()
        except OSError as exc:
            self._recording_error(exc)
            return
        self.record_button.setText("Начать запись в CSV")
        self.record_status.setText(
            f"{message}\n{self.recording.sample_count} отсчётов\n{self.recording.path.name}"
        )

    def _recording_error(self, error: Exception):
        try:
            self.recording.stop()
        except OSError:
            pass
        self.record_button.setText("Начать запись в CSV")
        self.record_status.setText("Запись прервана: ошибка сохранения")
        QMessageBox.critical(
            self, "Ошибка сохранения",
            f"Запись остановлена. Часть данных могла не сохраниться.\n{error}",
        )

    def _update_array_view(self):
        if not self._array_dirty:
            return
        self._array_dirty = False
        rows, columns = self.signal_array.shape
        self.array_shape_label.setText(
            f"Размер: {rows} × {columns}. Строки — отсчёты, столбцы — каналы."
        )
        self.array_view.setPlainText(np.array2string(
            self.signal_array, threshold=self.signal_array.size,
            max_line_width=160, floatmode="unique",
        ))

    def _update_plots(self):
        chunks, self._pending_chunks = self._pending_chunks, []
        for array in chunks:
            for index, widget in enumerate(self.channel_widgets):
                widget.push_chunk(array[:, index])
        auto_scale = self.cb_autoscale.isChecked()
        for widget in self.channel_widgets:
            widget.update_plot(auto_scale)
        if auto_scale:
            active = next((w for w in self.channel_widgets if w.is_active), None)
            if active:
                for spin, value in ((self.spin_y_min, active.current_y_min),
                                    (self.spin_y_max, active.current_y_max)):
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)

    def _update_stats(self):
        for widget in self.channel_widgets:
            widget.update_stats()

    def _apply_manual_scale(self):
        low, high = self.spin_y_min.value(), self.spin_y_max.value()
        if low < high:
            self.cb_autoscale.setChecked(False)
            for widget in self.channel_widgets:
                widget.set_y_range(low, high)

    def _apply_x_window(self):
        for widget in self.channel_widgets:
            widget.set_time_window(self.spin_x_window.value())

    def _rebuild_grid(self):
        if not hasattr(self, "grid_layout"):
            return
        count = len(self.channel_widgets)
        columns = 1 if count == 1 else 2 if count <= 4 else 3
        for position, widget in enumerate(self.channel_widgets):
            self.grid_layout.addWidget(widget, position // columns, position % columns)
            widget.show()

    def closeEvent(self, event):
        if self.recording.active:
            self._stop_recording()
        self.receiver.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SignalViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
