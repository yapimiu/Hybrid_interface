"""Research operator page for one complete P300 session."""
from pathlib import Path

import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView)

from .experiment import ExperimentController, ExperimentPlan
from .panel import output_root
from .erp_sums import ObjectSumsView


class ExperimentPanel(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('''
            QWidget { background: #0a0a0a; color: #e8e8e8; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background: #1e1e1e; padding: 6px; border: 1px solid #555; }
            QPushButton { background: #283b43; padding: 10px; border: 1px solid #45636f; border-radius: 4px; }
            QPushButton:hover { background: #35515d; }
            QPushButton:disabled { color: #777; background: #1b252a; }
            QGroupBox { border: 1px solid #444; margin-top: 12px; padding: 12px; }
            QTableWidget, QHeaderView::section { background: #151b1e; color: #e8e8e8; }
        ''')
        self.viewer = viewer
        self.controller = ExperimentController(viewer.p300_panel, viewer.component_controls, self)
        layout = QVBoxLayout(self)
        title = QLabel('Эксперимент P300')
        title.setStyleSheet('font-size: 24px; font-weight: bold; color: #5bc0be;')
        layout.addWidget(title)
        description = QLabel('Программа запускает исходный стимулятор, подключает маркеры, регистрирует серию проб '
            'и сохраняет результаты. В каждой пробе испытуемый смотрит на указанную целевую плитку. '
            'Порядок вспышек и расчёт ERP/P300 выполняет исходное ПО Пронина.')
        description.setWordWrap(True)
        layout.addWidget(description)
        self.readiness = QLabel()
        self.readiness.setWordWrap(True)
        layout.addWidget(self.readiness)
        settings = QGroupBox('Параметры сессии')
        form = QFormLayout(settings)
        self.subject = QLineEdit()
        self.subject.setPlaceholderText('Например, S01')
        self.trials = QSpinBox()
        self.trials.setRange(1, 200)
        self.trials.setValue(5)
        self.repetitions = QSpinBox()
        self.repetitions.setRange(5, 40)
        self.repetitions.setValue(12)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(0)
        self.seed.setToolTip('Seed очереди целевых плиток исходного протокола. Порядок вспышек генерирует стимулятор.')
        self.pause = QDoubleSpinBox()
        self.pause.setRange(0, 60)
        self.pause.setValue(1)
        self.folder = QLineEdit(str(output_root() / 'experiments'))
        browse = QPushButton('Выбрать папку…')
        browse.clicked.connect(self.browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder)
        folder_row.addWidget(browse)
        form.addRow('Испытуемый:', self.subject)
        form.addRow('Проб в сессии:', self.trials)
        form.addRow('Вспышек каждой плитки за пробу:', self.repetitions)
        form.addRow('Seed плана целей:', self.seed)
        form.addRow('Пауза между пробами, с:', self.pause)
        form.addRow('Сохранение экспериментов:', folder_row)
        layout.addWidget(settings)
        self.settings = settings
        row = QHBoxLayout()
        self.start_button = QPushButton('Начать единый эксперимент')
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton('Остановить и закрыть архив')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.controller.stop())
        advanced = QPushButton('Настроить ERP / каналы')
        advanced.clicked.connect(lambda: viewer.tabs.setCurrentWidget(viewer.p300_panel))
        hybrid = QPushButton('Готовый протокол P300 / SSVEP')
        hybrid.clicked.connect(self.open_hybrid)
        for button in (self.start_button, self.stop_button, advanced, hybrid):
            row.addWidget(button)
        layout.addLayout(row)
        self.hybrid_button = hybrid
        self.status = QLabel('Готовность: требуется EEG с поступающими отсчётами и Python окружения Пронина.')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('padding: 12px; background: #18333c; color: white; font-size: 15px;')
        layout.addWidget(self.status)
        self.live = QLabel('Объект внимания: —')
        self.live.setWordWrap(True)
        self.live.setStyleSheet('font-size: 20px; color: #5bc0be; padding: 8px;')
        layout.addWidget(self.live)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Проба', 'Предъявленная цель', 'Предполагаемый объект', 'Чистых эпох минимум', 'Результат'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)
        self.sums_view = ObjectSumsView(viewer.p300_panel, 'Сумма эпох по объектам: текущая / последняя проба')
        layout.addWidget(self.sums_view)
        self.controller.state_changed.connect(self.show_state)
        self.controller.trial_finished.connect(self.show_result)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Папка экспериментов', self.folder.text())
        if path:
            self.folder.setText(path)

    def start(self):
        try:
            if not self.folder.text().strip():
                raise ValueError('Выберите папку для эксперимента')
            plan = ExperimentPlan(self.subject.text().strip(), self.trials.value(),
                                  self.repetitions.value(), self.seed.value(), self.pause.value())
            self.controller.start(plan, Path(self.folder.text()))
            self.table.setRowCount(0)
        except Exception as exc:
            self.status.setText(str(exc))

    def show_state(self, state, text):
        active = self.controller.active
        writing = not self.controller.stopped()
        self.settings.setEnabled(not active and not writing)
        self.start_button.setEnabled(not active and not writing)
        self.hybrid_button.setEnabled(not active and not writing)
        self.stop_button.setEnabled(active)
        self.status.setText(text)

    def show_result(self, result):
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [result['trial'], result['target'], result['winner'] if result['winner'] is not None else '—',
                  min(result['clean_counts'].values()),
                  'Выбор получен' if result['status'] == 'decision' else 'Недостаточно пригодных данных']
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def refresh(self):
        window = self.viewer
        self.readiness.setText(f'EEG: {window.stream_name or "не подключён"}; отсчётов: {window.sample_count}. '
            f'P300: {"маркеры подключены" if window.p300_panel._inlet_markers is not None else "ожидает маркеры"}. '
            'Исходный стимулятор будет запущен автоматически при начале сессии.')
        snap = window.p300_panel.analysis_snapshot()
        counts = snap['clean_counts']
        minimum = min((counts.get(f'стимул_{i}', 0) for i in range(9)), default=0)
        values = snap['decision'].get('winner_debug', {}).get('final_metric_values', [])
        distinguishable = len(values) == 9 and np.isfinite(values).all() and np.ptp(values) > 1e-9
        winner = snap['winner'] if self.controller.active and minimum >= 5 and distinguishable else None
        if self.controller.state == 'trial' or self.controller.state == 'draining':
            self.sums_view.refresh()
        elif self.controller.results and self.sums_view.result is not self.controller.last_sums:
            self.sums_view.display(self.controller.last_sums, f'Проба {self.controller.results[-1]["trial"]}')
        if not self.controller.active and self.controller.results:
            last = self.controller.results[-1]
            winner = last['winner']
            minimum = min(last['clean_counts'].values())
        self.live.setText(f'Объект внимания (предварительно / последняя проба): {winner if winner is not None else "—"}; '
                          f'эпох каждой плитки минимум: {minimum}/5')

    def open_hybrid(self):
        self.viewer.tabs.setCurrentWidget(self.viewer.component_controls)
        self.viewer.component_controls.launch('protocol')
