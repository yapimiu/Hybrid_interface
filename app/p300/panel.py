"""Дополнительная панель P300 внутри существующего окна просмотра EEG."""

from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QVBoxLayout)
import pyqtgraph as pg

from .config import P300Config
from .lsl import MarkerReceiver, stream_metadata
from .worker import P300Worker


class P300Panel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("P300 — калибровка, ERP и онлайн-оценки", parent)
        self.setStyleSheet("QGroupBox { color: #5bc0be; }")
        self.info = None
        self.names = []
        self.config_overrides = {}
        self.recording = False
        self.channel_names = []
        self.last_snapshot = None
        self._erp_data = {}
        self._times = None
        self._curves = {}
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.units = QComboBox()
        self.units.addItems(["uV", "mV", "V"])
        self.channels = QLineEdit()
        self.channels.setPlaceholderText("Номера EEG-каналов через запятую: 1,2,3")
        self.channels.setToolTip("Выберите только EEG; исключите EOG, триггеры и служебные каналы")
        self.apply_button = QPushButton("Применить")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_config)
        self.settings_button = QPushButton("Настройки JSON")
        self.settings_button.clicked.connect(self.load_config)
        grid.addWidget(QLabel("Единицы входа:"), 0, 0)
        grid.addWidget(self.units, 0, 1)
        grid.addWidget(QLabel("EEG-каналы:"), 0, 2)
        grid.addWidget(self.channels, 0, 3)
        grid.addWidget(self.apply_button, 0, 4)
        grid.addWidget(self.settings_button, 0, 5)
        self.marker_name = QLineEdit()
        self.marker_name.setPlaceholderText("Имя LSL-маркеров (пусто: единственный Markers/Marker)")
        self.marker_button = QPushButton("Выбрать поток маркеров")
        grid.addWidget(self.marker_name, 1, 0, 1, 4)
        grid.addWidget(self.marker_button, 1, 4, 1, 2)
        layout.addLayout(grid)
        self.state_label = QLabel("Подключите EEG, проверьте единицы и каналы, нажмите «Применить»")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.diagnostic_label = QLabel("Требуется внешний визуальный стимулятор с JSON-маркерами.")
        self.diagnostic_label.setWordWrap(True)
        layout.addWidget(self.diagnostic_label)
        row = QHBoxLayout()
        self.session_button = QPushButton("Записать эксперимент")
        self.calibration_button = QPushButton("Новая калибровка")
        self.train_button = QPushButton("Обучить обе модели")
        self.algorithm = QComboBox()
        self.algorithm.addItem("Shrinkage LDA", "shrinkage_lda")
        self.algorithm.addItem("xDAWN + LDA", "xdawn_lda")
        for widget in (self.session_button, self.calibration_button, self.train_button, self.algorithm):
            row.addWidget(widget)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.load_model_button = QPushButton("Загрузить модель NPZ")
        self.load_epochs_button = QPushButton("Загрузить калибровку NPZ")
        self.erp_channel = QComboBox()
        row.addWidget(self.load_model_button)
        row.addWidget(self.load_epochs_button)
        row.addWidget(QLabel("ERP-канал:"))
        row.addWidget(self.erp_channel)
        layout.addLayout(row)
        self.result_label = QLabel("Оценки объектов появятся после калибровки и получения стимулов.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        self.report_label = QLabel("Независимая проверка на реальных данных не проведена.")
        self.report_label.setWordWrap(True)
        layout.addWidget(self.report_label)
        self.erp_plot = pg.PlotWidget()
        self.erp_plot.setMinimumHeight(230)
        self.erp_plot.setLabel("bottom", "Время от начала стимула", units="s")
        self.erp_plot.setLabel("left", "ERP", units="µV")
        self.erp_plot.showGrid(x=True, y=True, alpha=0.2)
        self.erp_plot.addLegend()
        layout.addWidget(self.erp_plot)
        self.erp_channel.currentIndexChanged.connect(self.draw_erp)

        self.worker = P300Worker(self)
        self.marker_receiver = MarkerReceiver(self)
        self.worker.updated.connect(self.update_snapshot)
        self.worker.diagnostic.connect(self.diagnostic_label.setText)
        self.worker.trained.connect(self.show_report)
        self.worker.busy.connect(self.set_busy)
        self.marker_receiver.marker_received.connect(self.worker.submit_marker, Qt.DirectConnection)
        self.marker_receiver.state_changed.connect(self.worker.marker_state, Qt.DirectConnection)
        self.marker_receiver.diagnostic.connect(
            lambda message: self.worker.submit("diagnostic", message), Qt.DirectConnection)
        self.marker_button.clicked.connect(lambda: self.marker_receiver.select_stream(self.marker_name.text()))
        self.session_button.clicked.connect(self.toggle_experiment)
        self.calibration_button.clicked.connect(lambda: self.worker.submit("start_calibration"))
        self.train_button.clicked.connect(lambda: self.worker.submit("train", self.algorithm.currentData()))
        self.algorithm.currentIndexChanged.connect(
            lambda: self.worker.submit("algorithm", self.algorithm.currentData()))
        self.load_model_button.clicked.connect(lambda: self.load_artifact("load_model", "Модель P300"))
        self.load_epochs_button.clicked.connect(lambda: self.load_artifact("load_calibration", "Калибровка P300"))
        self._enable_actions(False)
        self.worker.start()
        self.marker_receiver.start()

    def _enable_actions(self, enabled):
        for widget in (self.session_button, self.calibration_button, self.train_button,
                       self.load_model_button, self.load_epochs_button, self.algorithm):
            widget.setEnabled(enabled)

    def set_stream(self, info, names):
        self.info, self.names = info, list(names)
        self.channels.setText(",".join(str(i + 1) for i in range(len(names))))
        self.apply_button.setEnabled(True)
        self._enable_actions(False)
        self.worker.submit("disconnect")
        self.state_label.setText("Проверьте единицы и EEG-каналы, затем нажмите «Применить»")
        self.clear_results()

    def disconnected(self):
        self.info = None
        self.apply_button.setEnabled(False)
        self._enable_actions(False)
        self.worker.submit("disconnect")
        self.state_label.setText("Нет EEG-потока")
        self.clear_results()

    def clear_results(self):
        self.recording = False
        self.session_button.setText("Записать эксперимент")
        self.result_label.setText("Нет текущего результата")
        self.report_label.setText("Независимая проверка на реальных данных не проведена.")
        self._erp_data, self._times = {}, None
        self.draw_erp()

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Параметры P300", "", "JSON (*.json)")
        if not path:
            return
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Ожидается JSON-объект параметров P300Config")
            self.config_overrides = value
            self.diagnostic_label.setText("Настройки прочитаны; нажмите «Применить». Единицы и каналы задаются в панели.")
        except (OSError, ValueError) as exc:
            self.diagnostic_label.setText(f"Ошибка настроек: {exc}")

    def apply_config(self):
        if self.info is None:
            return
        try:
            indices = [int(token.strip()) - 1 for token in self.channels.text().split(",")]
            if (not indices or len(set(indices)) != len(indices)
                    or any(i < 0 or i >= len(self.names) for i in indices)):
                raise ValueError("Номера EEG-каналов должны быть уникальными и существовать в потоке")
            c = P300Config.from_dict({**self.config_overrides, "sfreq": float(self.info.nominal_srate()),
                                      "channel_names": [self.names[i] for i in indices],
                                      "input_unit": self.units.currentText()})
            self.channel_names = list(c.channel_names)
            self.erp_channel.clear()
            self.erp_channel.addItems(self.channel_names)
            self.clear_results()
            self.worker.submit("configure", c, indices, stream_metadata(self.info))
            self._enable_actions(True)
            self.diagnostic_label.setText(
                f"Эпоха {c.tmin}…{c.tmax} с; фильтр {c.l_freq}–{c.h_freq} Гц; "
                f"задержка до результата ≈{c.tmax + c.padding_s:.1f} с. Выбранные единицы: {c.input_unit}.")
        except (ValueError, TypeError) as exc:
            self.diagnostic_label.setText(f"Неверная конфигурация P300: {exc}")

    def toggle_experiment(self):
        if self.recording:
            self.worker.submit("stop_experiment")
            return
        folder = QFileDialog.getExistingDirectory(self, "Каталог для нового эксперимента")
        if folder:
            name = datetime.now().strftime("p300_%Y-%m-%d_%H-%M-%S_") + uuid4().hex[:8]
            self.worker.submit("start_experiment", str(Path(folder) / name))

    def load_artifact(self, action, title):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Числовой NPZ (*.npz)")
        if path:
            self.worker.submit(action, path)

    def update_snapshot(self, state):
        if state is None:
            self.clear_results()
            return
        if self.info is None:
            return
        self.last_snapshot = state
        self.state_label.setText(
            f'{state["status"]}\nПринято: {state["accepted"]}; отклонено: {state["rejected"]}; '
            f'ожидают EEG: {state["pending"]}. {state["diagnostic"]}')
        self.recording = bool(state["recording"])
        self.session_button.setText("Завершить эксперимент" if self.recording else "Записать эксперимент")
        if self.recording:
            self.session_button.setToolTip(state["recording"])
        self.algorithm.blockSignals(True)
        self.algorithm.setCurrentIndex(self.algorithm.findData(state["algorithm"]))
        self.algorithm.blockSignals(False)
        result = state["result"]
        if result:
            values = "; ".join(f'{obj}: {score:.3f} (n={result["counts"][obj]})'
                               for obj, score in result["mean_scores"].items())
            candidate = result["candidate"]
            self.result_label.setText(
                f'Блок {result["block_id"]}, trial {result["trial_id"]}: {values}\n'
                + (f"Предварительный выбор: {candidate}" if candidate else
                   "Накопление/неоднозначность: выбора пока нет"))
        else:
            self.result_label.setText("Нет текущих оценок объектов")
        self._erp_data, self._times = state["erp"], state["times"]
        if state["report"]:
            self.show_report(state["report"])
        else:
            self.report_label.setText("Независимая проверка на реальных данных не проведена.")
        self.draw_erp()

    def set_busy(self, busy):
        self.train_button.setEnabled(not busy and self.info is not None)
        self.calibration_button.setEnabled(not busy and self.info is not None)
        self.load_epochs_button.setEnabled(not busy and self.info is not None)

    def draw_erp(self):
        self.erp_plot.clear()
        self._curves.clear()
        channel = self.erp_channel.currentIndex()
        if self._times is None or channel < 0:
            return
        for index, (obj, value) in enumerate(self._erp_data.items()):
            if channel < value["data"].shape[0]:
                self._curves[obj] = self.erp_plot.plot(
                    self._times, value["data"][channel] * 1e6,
                    pen=pg.mkPen(pg.intColor(index), width=2), name=f'{obj} (n={value["count"]})')

    def show_report(self, report):
        summaries = []
        for name, folds in report["algorithms"].items():
            accuracy = sum(f["balanced_accuracy"] for f in folds) / len(folds)
            auc = sum(f["roc_auc"] for f in folds) / len(folds)
            summaries.append(f"{name}: balanced accuracy={accuracy:.3f}, AUC={auc:.3f}")
        self.report_label.setText(
            "Внутренняя CV по блокам: " + "; ".join(summaries)
            + ". Независимая проверка на реальных данных не проведена.")

    def begin_shutdown(self):
        self.marker_receiver.requestInterruption()
        self.worker.shutdown()

    def stopped(self):
        return not self.worker.isRunning() and not self.marker_receiver.isRunning()
