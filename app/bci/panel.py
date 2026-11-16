"""Embed Pronin's original analyzer; override transport and output locations only."""
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import QStandardPaths, Qt, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QLabel, QLineEdit, QComboBox, QMessageBox, QPushButton, QWidget, QVBoxLayout

import p300_analysis.qt_window as upstream
from p300_analysis.exam_session_detail_logger import ExamSessionDetailLogger
from p300_analysis.session_recorder import SessionRecorder
from p300_analysis.run_export import export_run_continuous_csv, export_run_data, export_run_data_all_stims
from .bridge import ConnectMarkers, DiscoverMarkers, SharedEEGInlet, ObservedMarkerInlet
from .marker_streams import marker_stream_label
from .erp_sums import ObjectSumsView


class ObservedSessionRecorder(SessionRecorder):
    on_winner = None

    def log_winner(self, payload):
        if self.on_winner:
            self.on_winner(payload)
        super().log_winner(payload)


def output_root():
    override = os.environ.get('HYBRID_BCI_DATA_DIR')
    root = Path(override) if override else Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)) / 'Hybrid Interface'
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_output_locations(root):
    # The original standalone GUI writes relative to __file__. Frozen executables
    # unpack into a temporary directory, so route only its logging destinations.
    upstream.SessionRecorder = partial(ObservedSessionRecorder, output_path=root / 'data/all_runs.ndjson')
    upstream.ExamSessionDetailLogger = SimpleNamespace(open_new=partial(
        ExamSessionDetailLogger.open_new, output_dir=root / 'data/examination_logs'))
    def debug(payload):
        try:
            with (root / 'p300_debug.ndjson').open('a', encoding='utf-8') as file:
                file.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except OSError:
            pass
    upstream.debug_ndjson = debug


class P300Panel(upstream.P300AnalyzerWindow):
    shutdown_progress = pyqtSignal()
    connection_changed = pyqtSignal(bool, str)
    marker_observed = pyqtSignal(str, float)
    marker_selection_required = pyqtSignal(str)
    analysis_updated = pyqtSignal(object)

    def __init__(self, parent=None):
        self.data_root = output_root()
        configure_output_locations(self.data_root)
        self.stream_info = None
        self.stream_names = []
        self.shared_inlet = None
        self.connector = None
        self.marker_discovery = None
        self._selected_marker_info = None
        self.closing = False
        self.connection_revision = 0
        self.last_decision = None
        self.managed_session = False
        super().__init__()
        self._session_recorder.on_winner = self._observe_decision
        self.setParent(parent)
        self.setWindowFlags(Qt.Widget)
        self.combo_streams.setEnabled(False)
        self.btn_connect.setEnabled(False)
        sidebar = self.btn_connect.parentWidget().layout()
        self.marker_name = QLineEdit()
        self.marker_name.setPlaceholderText('Автоматически / точное имя')
        self.marker_name.setToolTip('Пустое имя: найти единственный поток событий. '
                                    'Если потоков несколько, выберите нужный в списке.')
        self.marker_name.textChanged.connect(self._clear_marker_selection)
        self.marker_streams = QComboBox()
        self.marker_streams.addItem('Автоматически', None)
        self.marker_streams.view().setMinimumWidth(650)
        self.marker_streams.activated.connect(self._select_marker_stream)
        self.btn_find_markers = QPushButton('Найти потоки событий')
        self.btn_find_markers.clicked.connect(self._find_marker_streams)
        self.input_unit = QComboBox()
        self.input_unit.addItems(['uV', 'mV', 'V'])
        self.btn_connect.setText('Подключить маркеры P300')
        sidebar.insertWidget(2, QLabel('Имя потока маркеров:'))
        sidebar.insertWidget(3, self.marker_name)
        sidebar.insertWidget(4, self.marker_streams)
        sidebar.insertWidget(5, self.btn_find_markers)
        sidebar.insertWidget(6, QLabel('Единицы входной EEG (для P300):'))
        sidebar.insertWidget(7, self.input_unit)
        original_page = self.takeCentralWidget()
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        self.workflow_status = QLabel()
        self.workflow_status.setWordWrap(True)
        self.workflow_status.setStyleSheet('padding: 8px; background: #18333c; color: #e8f6ff;')
        self.summary_status = QLabel('ERP появятся после начала анализа и вспышек внешнего стимулятора.')
        self.summary_status.setWordWrap(True)
        outer.addWidget(self.workflow_status)
        outer.addWidget(self.summary_status)
        outer.addWidget(original_page, stretch=1)
        self.sums_view = ObjectSumsView(self)
        outer.addWidget(self.sums_view)
        self.setCentralWidget(wrapper)
        self.btn_refresh_streams.setToolTip('Показать поток EEG, подключённый в Hybrid Interface')
        self._set_status('EEG берётся из вкладки «EEG». Запустите внешний стимулятор, затем подключите P300.')

    def set_stream(self, info, names):
        self.disconnected()
        self.stream_info, self.stream_names = info, list(names)
        self._on_refresh_streams_clicked()
        self._build_channel_checkboxes(info.channel_count(), labels=self.stream_names)
        self.btn_connect.setEnabled(True)

    def _clear_marker_selection(self, *_):
        self._selected_marker_info = None
        self.marker_streams.setCurrentIndex(0)

    def _select_marker_stream(self, index):
        info = self.marker_streams.itemData(index)
        self.marker_name.setText(info.name() if info is not None else '')
        self._selected_marker_info = info
        self.marker_streams.setCurrentIndex(index)

    def _show_marker_streams(self, streams):
        if self.closing:
            return
        selected = self._selected_marker_info
        self.marker_streams.clear()
        self.marker_streams.addItem('Автоматически', None)
        for info in streams:
            self.marker_streams.addItem(marker_stream_label(info), info)
            self.marker_streams.setItemData(self.marker_streams.count() - 1,
                                           marker_stream_label(info), Qt.ToolTipRole)
        self._selected_marker_info = None
        for index, info in enumerate(streams, 1):
            if selected is not None and info.uid() == selected.uid():
                self._select_marker_stream(index)
                break
        else:
            if len(streams) == 1 and not self.marker_name.text().strip():
                self._select_marker_stream(1)

    def _find_marker_streams(self):
        if (self.closing or self.managed_session or self._inlet_markers is not None
                or (self.connector and self.connector.isRunning())
                or (self.marker_discovery and self.marker_discovery.isRunning())):
            return
        self.btn_find_markers.setEnabled(False)
        self._set_status('Поиск потоков событий LSL…')
        self.marker_discovery = DiscoverMarkers(self)
        self.marker_discovery.found.connect(self._marker_streams_found)
        self.marker_discovery.failed.connect(self._marker_search_failed)
        self.marker_discovery.finished.connect(self._marker_search_finished)
        self.marker_discovery.start()

    def _marker_streams_found(self, streams):
        if self.closing or self.managed_session or self._inlet_markers is not None:
            return
        self._show_marker_streams(streams)
        if not streams:
            self._set_status('Потоки событий не найдены. Включите передачу маркеров по LSL. '
                             'Если события находятся внутри EEG, нужен адаптер их извлечения.')
        elif len(streams) == 1:
            self._set_status('Найден поток событий: ' + marker_stream_label(streams[0]))
        else:
            self._set_status('Найдено несколько потоков событий. Выберите нужный в списке.')

    def _marker_search_failed(self, message):
        if not self.closing and not self.managed_session:
            self._set_status('Ошибка поиска потоков событий: ' + message)

    def _marker_search_finished(self):
        self._update_marker_controls()
        self.shutdown_progress.emit()

    def _update_marker_controls(self):
        busy = (self.closing or self.managed_session or self._inlet_markers is not None
                or bool(self.connector and self.connector.isRunning()))
        self.marker_streams.setEnabled(not busy)
        self.btn_find_markers.setEnabled(not busy and not (
            self.marker_discovery and self.marker_discovery.isRunning()))

    def _set_status(self, text):
        super()._set_status(text)
        if hasattr(self, 'workflow_status'):
            self.workflow_status.setText(text)

    def _observe_decision(self, payload):
        self.last_decision = payload

    def _begin_recording_session(self):
        self.last_decision = None
        super()._begin_recording_session()
        self.set_experiment_owned(self.managed_session)

    def set_experiment_owned(self, owned):
        self.managed_session = owned
        names = ('btn_connect', 'btn_disconnect', 'btn_start_analysis', 'btn_stop_analysis',
                 'btn_reset_analysis', 'spin_baseline', 'spin_x', 'spin_y', 'spin_artifact_thresh',
                 'chk_car', 'chk_epochs_after_trial', 'combo_winner_mode', 'marker_name', 'input_unit')
        for name in names:
            widget = getattr(self, name, None)
            if widget:
                widget.setEnabled(not owned)
        for widget in list(self.channel_checkboxes) + list(self._analysis_profile_buttons.values()):
            widget.setEnabled(not owned)
        if not owned:
            connected = self._inlet_markers is not None
            self.btn_connect.setEnabled(not connected and self.stream_info is not None)
            self.btn_disconnect.setEnabled(connected)
            self.btn_start_analysis.setEnabled(connected and not self._recording_epochs)
            self.btn_stop_analysis.setEnabled(connected and self._recording_epochs)
            self.btn_reset_analysis.setEnabled(connected)
        self._update_marker_controls()

    def analysis_snapshot(self):
        decision = self.last_decision or {}
        debug = decision.get('winner_debug', {})
        counts = {key: len(values) for key, values in self.epochs_data.items()}
        rejected = debug.get('artifact_rejected', {})
        clean = {key: max(0, count - int(rejected.get(key, 0))) for key, count in counts.items()}
        return {'recording': self._recording_epochs, 'connected': self._inlet_markers is not None,
                'counts': counts, 'clean_counts': clean, 'pending': len(self.pending_markers),
                'decision': decision, 'winner': decision.get('winner_digit'),
                'time_ms': self._epoch_geom.time_ms_template}

    def submit_eeg(self, samples, timestamps):
        # Called directly from LSLReceiver: only a locked queue, no Qt/DSP here.
        inlet = self.shared_inlet
        if inlet is not None:
            inlet.push(samples, timestamps)

    def _on_refresh_streams_clicked(self):
        self.combo_streams.clear()
        if self.stream_info is not None:
            self.combo_streams.addItem(f'{self.stream_info.name()} ({self.stream_info.channel_count()} ch)', self.stream_info)
            self._set_status('Выбран тот же EEG-поток, что и в основном просмотрщике.')
            self._markers_presence_label.setText('Маркеры ещё не подключены. Запустите стимулятор и подключите маркеры P300.')
        else:
            self._set_status('Ожидание EEG/Signal в основном просмотрщике.')

    def _on_connect_clicked(self):
        if (self.closing or self._timer.isActive() or self.stream_info is None
                or (self.connector and self.connector.isRunning())
                or (self.marker_discovery and self.marker_discovery.isRunning())):
            return
        self.connection_revision += 1
        revision = self.connection_revision
        self.btn_connect.setEnabled(False)
        self.marker_name.setEnabled(False)
        self.input_unit.setEnabled(False)
        self.marker_streams.setEnabled(False)
        self.btn_find_markers.setEnabled(False)
        self._set_status('Подключение к потоку маркеров…')
        self.connector = ConnectMarkers(self.marker_name.text(), self, stream_info=self._selected_marker_info)
        self.connector.found.connect(lambda streams: self._show_marker_streams(streams)
                                     if revision == self.connection_revision else None)
        self.connector.selection_required.connect(lambda message: self.marker_selection_required.emit(message)
                                                 if revision == self.connection_revision else None)
        self.connector.connected.connect(lambda inlet: self._attach_markers(inlet, revision))
        self.connector.failed.connect(lambda message: self._connect_failed(message, revision))
        self.connector.finished.connect(self.shutdown_progress)
        self.connector.finished.connect(self._update_marker_controls)
        self.connector.start()

    def _connect_failed(self, message, revision):
        if revision == self.connection_revision and not self.closing:
            self._set_status(message)
            self.btn_connect.setEnabled(self.stream_info is not None)
            self.marker_name.setEnabled(True)
            self.input_unit.setEnabled(True)
            self._markers_presence_label.setText(message)
            self.connection_changed.emit(False, message)
            self.set_experiment_owned(self.managed_session)

    def _attach_markers(self, inlet, revision):
        if self.closing or self.stream_info is None or revision != self.connection_revision:
            inlet.close_stream()
            return
        self.shared_inlet = SharedEEGInlet(self.stream_info, {'uV': 1, 'mV': 1e3, 'V': 1e6}[self.input_unit.currentText()])
        self._inlet_eeg = self.shared_inlet
        self._inlet_markers = ObservedMarkerInlet(inlet, self.marker_observed.emit)
        if len(self.channel_checkboxes) != self.stream_info.channel_count():
            self._build_channel_checkboxes(self.stream_info.channel_count(), labels=self.stream_names)
        self._begin_connection_session()
        self.set_experiment_owned(self.managed_session)
        self._markers_presence_label.setText('Маркеры: ' + inlet.info().name())
        self.connection_changed.emit(True, inlet.info().name())

    def _update_loop(self):
        try:
            super()._update_loop()
            self.sums_view.refresh()
            snap = self.analysis_snapshot()
            count = sum(snap['counts'].values())
            winner = snap['winner']
            self.summary_status.setText(f'Эпох: {count}; ожидают EEG: {snap["pending"]}; '
                                       f'предварительный объект: {winner if winner is not None else "—"}')
            self.analysis_updated.emit(snap)
        except Exception as exc:
            self._on_disconnect_clicked()
            self._set_status('P300 остановлен: ' + str(exc))

    def _on_stop_analysis_clicked(self):
        super()._on_stop_analysis_clicked()
        self.sums_view.refresh()  # эпохи, добранные при остановке

    def _on_disconnect_clicked(self):
        self.connection_revision += 1
        if self.connector and self.connector.isRunning():
            self.connector.requestInterruption()
        super()._on_disconnect_clicked()
        self.shared_inlet = None
        self.last_decision = None
        self.marker_name.setEnabled(True)
        self.input_unit.setEnabled(True)
        self.btn_connect.setEnabled(self.stream_info is not None and not self.closing)
        # The upstream exporter remains available for the completed recording.
        self.btn_save_exam.setEnabled(bool(self._last_run_export_data))
        self.connection_changed.emit(False, 'P300 отключён')
        self._update_marker_controls()

    def disconnected(self):
        self._on_disconnect_clicked()
        self.stream_info = None
        self.combo_streams.clear()
        self.btn_connect.setEnabled(False)

    def _on_save_exam_clicked(self):
        if self._recording_epochs or not self._last_run_export_data:
            QMessageBox.information(self, 'P300', 'Сначала остановите анализ с записанными данными.')
            return
        opts = self._show_export_options_dialog()
        if opts is None:
            return
        fmt = opts['format']
        if opts['mode'] == 'continuous' and fmt == 'txt':
            fmt = 'csv'
        default = self.data_root / ('p300_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.' + fmt)
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить обследование P300', str(default), f'{fmt.upper()} (*.{fmt})')
        if not path:
            return
        try:
            args = dict(run_data=self._last_run_export_data, output_path=Path(path),
                        file_format=fmt, selected_channels=opts['selected_channels'])
            if opts['mode'] == 'continuous':
                saved = export_run_continuous_csv(**args)
                self._save_continuous_meta_sidecar(saved, export_options=opts)
                files = [saved]
            elif opts['mode'] == 'all_stims':
                files = export_run_data_all_stims(**args)
            else:
                files = export_run_data(**args, stim_index=opts['stim_index'])
            self._set_status('Сохранено: ' + ', '.join(map(str, files)))
        except Exception as exc:
            QMessageBox.critical(self, 'Ошибка сохранения P300', str(exc))

    def begin_shutdown(self):
        if self.closing:
            return
        self.closing = True
        if self.marker_discovery and self.marker_discovery.isRunning():
            self.marker_discovery.requestInterruption()
        self._on_disconnect_clicked()
        if self._exam_detail_log is not None:
            self._exam_detail_log.close()
        for name in ('_monitor_eeg_win', '_monitor_markers_win', '_monitor_selected_win',
                     '_epoch_summary_win', '_ch_health_win'):
            widget = getattr(self, name, None)
            if widget:
                widget.close()

    def stopped(self):
        return not ((self.connector and self.connector.isRunning())
                    or (self.marker_discovery and self.marker_discovery.isRunning()))
