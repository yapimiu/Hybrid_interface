"""One operator workflow around the unchanged upstream P300 stimulus and analyzer."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from pylsl import local_clock
from pylsl.lib import lib as liblsl
from .session import SessionWriter
from .controls import UPSTREAM_COMMIT
from .upstream import protocol_helper
from .erp_sums import decision_window, empty_sums, live_sums, object_numbers


@dataclass(frozen=True)
class ExperimentPlan:
    subject: str
    trials: int = 5
    repetitions: int = 12
    seed: int = 0
    pause_s: float = 1.0

    def __post_init__(self):
        if not self.subject.strip():
            raise ValueError('Введите идентификатор испытуемого')
        if not 1 <= self.trials <= 200 or not 5 <= self.repetitions <= 40:
            raise ValueError('Число проб: 1..200; повторений каждой плитки: 5..40')
        if not 0 <= self.pause_s <= 60:
            raise ValueError('Пауза должна быть от 0 до 60 секунд')

    def targets(self):
        queue, _ = protocol_helper('experiment_queue').build_p300_block_queue(
            p300_calib_trials=0, calib_target_tile_id=4,
            p300_main_trials=self.trials, shuffle_seed=self.seed)
        return [item.target_tile_id for item in queue]


def stream_xml(info):
    """Full LSL stream description; NeuroSpectr writes its ECG label in cp1251, not UTF-8."""
    try:
        return info.as_xml()
    except UnicodeDecodeError:
        return liblsl.lsl_get_xml(info.obj).decode('utf-8', errors='replace')


def write_stim_control(directory, payload):
    """Use the original protocol writer and wire format unchanged."""
    protocol_helper('stim_control').write_control(Path(directory), payload)


class ExperimentController(QObject):
    state_changed = pyqtSignal(str, str)
    trial_finished = pyqtSignal(object)
    shutdown_progress = pyqtSignal()

    def __init__(self, panel, components, parent=None):
        super().__init__(parent)
        self.panel, self.components = panel, components
        self.state = 'idle'
        self.message = ''
        self.writer = None
        self.plan = None
        self.directory = None
        self.targets = []
        self.index = 0
        self.results = []
        self.last_sums = empty_sums()
        self.deadline = 0
        self.last_eeg_at = None
        self.trial_started_at = None
        self.end_received_at = None
        self.last_flash_at = None
        self.stimulus_launched = False
        self.completed = False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.panel.connection_changed.connect(self._connection_changed)
        self.panel.marker_observed.connect(self._marker)
        self.components.component_finished.connect(self._component_finished)

    @property
    def active(self):
        return self.state not in ('idle', 'completed', 'failed', 'stopped')

    def _state(self, state, text):
        self.state = state
        self.message = text
        self.state_changed.emit(state, text)
        if self.writer and self.writer.isRunning():
            self.writer.submit('state', {'state': state, 'text': text, 'local_timestamp': local_clock()})

    def start(self, plan, parent_directory):
        if self.active or (self.writer and self.writer.isRunning()):
            raise ValueError('Сначала завершите текущую сессию')
        info = self.panel.stream_info
        if info is None:
            raise ValueError('Сначала дождитесь EEG-потока во вкладке EEG')
        if self.last_eeg_at is None or time.monotonic() - self.last_eeg_at > 2.5:
            raise ValueError('EEG-поток найден, но свежие отсчёты не поступают')
        if self.panel._recording_epochs or self.components.any_running():
            raise ValueError('Остановите отдельный анализ/стимулятор/протокол перед единой сессией')
        if not any(cb.isChecked() for cb in self.panel.channel_checkboxes):
            raise ValueError('Выберите EEG-каналы ROI на вкладке P300')
        self.components.validate_python()
        self.plan = plan
        self.targets = plan.targets()
        self.index = 0
        self.results = []
        self.completed = False
        self.stimulus_launched = False
        self.directory = Path(parent_directory) / (datetime.now().strftime('session_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6])
        metadata = {'schema': 'hybrid-bci-session/v1', 'subject': plan.subject,
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                    'upstream_commit': UPSTREAM_COMMIT, 'plan': plan.__dict__, 'targets': self.targets,
                    'channels': self.panel.stream_names, 'eeg': {'name': info.name(), 'source_id': info.source_id(),
                    'nominal_srate': info.nominal_srate(), 'xml': stream_xml(info)},
                    'input_units': self.panel.input_unit.currentText(),
                    'roi_0idx': [i for i, cb in enumerate(self.panel.channel_checkboxes) if cb.isChecked()],
                    'analysis': {'mode': 'auc', 'baseline_ms': self.panel.spin_baseline.value(),
                    'window_ms': [self.panel.spin_x.value(), self.panel.spin_y.value()],
                    'car': self.panel.chk_car.isChecked(), 'artifact_uv': self.panel.spin_artifact_thresh.value()}}
        self.writer = SessionWriter(self.directory, metadata, self)
        self.writer.ready.connect(self._recording_ready)
        self.writer.failed.connect(self.fail)
        self.writer.finished.connect(self._writer_finished)
        self.panel.combo_winner_mode.setCurrentIndex(self.panel.combo_winner_mode.findData('auc'))
        self.panel.marker_name.setText('BCI_StimMarkers')
        self.panel.set_experiment_owned(True)
        self.components.managed_session = True
        self._state('preparing', 'Подготовка архива и исходного стимулятора…')
        self.writer.start()
        self.timer.start()

    def eeg(self, samples, timestamps):
        """Direct receiver callback: only timestamp bookkeeping and a bounded queue."""
        if len(timestamps):
            self.last_eeg_at = time.monotonic()
        writer = self.writer
        if self.active and writer is not None:
            writer.eeg(samples, timestamps)

    def _recording_ready(self):
        if self.state != 'preparing':
            return
        try:
            write_stim_control(self.directory, {'state': 'paused', 'message': 'Подготовка EEG и маркеров'})
            self.panel._on_disconnect_clicked()
            self.panel.set_experiment_owned(True)
            extra = ['--auto-random-protocol', '--stim-control-dir', str(self.directory),
                     '--sequences', str(self.plan.repetitions)]
            if self.components.launch('stimulus', extra_args=extra) is None:
                raise RuntimeError(self.components.status.text())
            self.stimulus_launched = True
            self.deadline = time.monotonic() + 35
            self._state('connecting', 'Ожидание исходного стимулятора и потока маркеров P300…')
        except Exception as exc:
            self.fail(str(exc))

    def _connection_changed(self, connected, message):
        if connected and self.state == 'connecting':
            self._next_trial()
        elif not connected and self.state in ('trial', 'draining', 'pause'):
            self.fail('Потеряно подключение P300: ' + message)

    def _next_trial(self):
        try:
            if self.index >= len(self.targets):
                self.stop('Все пробы завершены', protocol_complete=True)
                return
            self.panel._on_start_analysis_clicked()
            if not self.panel._recording_epochs:
                raise RuntimeError('Не удалось начать регистрацию ERP')
            self.trial_started_at = None
            self.end_received_at = None
            self.last_flash_at = None
            self.deadline = time.monotonic() + 45
            self._state('trial', f'Проба {self.index + 1}/{len(self.targets)}: ожидание cue и случайных вспышек')
            write_stim_control(self.directory, {'state': 'trial', 'target_tile_id': self.targets[self.index],
                'experiment_index': self.index + 1, 'experiment_total': len(self.targets), 'label': 'P300'})
        except Exception as exc:
            self.fail(str(exc))

    def _marker(self, value, timestamp):
        if not self.active or self.writer is None:
            return
        self.writer.submit('marker', {'value': value, 'source_timestamp': timestamp,
                                     'received_local_timestamp': local_clock(), 'trial': self.index + 1})
        parts = value.split('|')
        if len(parts) < 2 or self.state not in ('trial', 'draining'):
            return
        now = time.monotonic()
        if parts[1] == 'trial_start':
            if self.trial_started_at is not None:
                self.fail('Вторая trial_start до завершения пробы; проверьте единственность стимулятора')
                return
            from p300_analysis.marker_parsing import parse_trial_target_tile_id
            if parse_trial_target_tile_id(value) != self.targets[self.index]:
                self.fail('Метка целевого объекта не совпала с запрошенной пробой')
                return
            self.trial_started_at = now
            self.deadline = now + max(60, self.plan.repetitions * 20)
        elif parts[1] == 'on':
            self.last_flash_at = now
        elif parts[1] == 'trial_end' and self.trial_started_at is not None:
            self.end_received_at = now
            self.deadline = now + 5
            self._state('draining', 'Получен trial_end; ожидание полного ответа последней вспышки…')

    def tick(self):
        if not self.active:
            return
        now = time.monotonic()
        try:
            if self.state != 'preparing' and (self.last_eeg_at is None or now - self.last_eeg_at > 3):
                self.fail('EEG не передаёт свежие отсчёты более 3 секунд')
            elif self.state == 'connecting':
                if now > self.deadline:
                    self.fail('Не удалось подключить маркеры за 35 секунд. Проверьте Python и журнал запуска')
                elif self.panel._inlet_markers is not None:
                    self._next_trial()
                elif self.panel.connector is None or not self.panel.connector.isRunning():
                    self.panel._on_connect_clicked()
            elif self.state == 'trial' and now > self.deadline:
                self.fail('Не получен ожидаемый trial_start/trial_end; запись остановлена')
            elif self.state == 'draining':
                ready_at = max(self.end_received_at, self.last_flash_at or self.end_received_at) + 1.0
                if now >= ready_at and not self.panel.pending_markers:
                    self._finish_trial()
                elif now > self.deadline:
                    self.fail('Не удалось получить полный EEG-ответ последней вспышки')
            elif self.state == 'pause' and now >= self.deadline:
                self._next_trial()
        except Exception as exc:
            self.fail(str(exc))

    def _finish_trial(self):
        self.panel._on_stop_analysis_clicked()
        snap = self.panel.analysis_snapshot()
        counts = {str(obj): int(snap['clean_counts'].get(f'стимул_{obj}', 0)) for obj in range(9)}
        decision = snap['decision']
        values = decision.get('winner_debug', {}).get('final_metric_values', [])
        import numpy as np
        valid = (min(counts.values()) >= 5 and snap['winner'] is not None and len(values) == 9
                 and np.isfinite(values).all() and np.ptp(values) > 1e-9)
        result = {'trial': self.index + 1, 'target': self.targets[self.index],
                  'winner': snap['winner'] if valid else None, 'status': 'decision' if valid else 'insufficient_evidence',
                  'clean_counts': counts, 'mode': decision.get('mode'),
                  'scores': values, 'margin': decision.get('winner_debug', {}).get('margin'),
                  'note': 'Upstream ERP metric, not a calibrated probability or proof of P300 presence.'}
        sums = self._trial_sums()
        window = decision_window(self.panel)
        numbers, number_max = object_numbers(sums, window)
        result['flashes_per_object'] = {str(o): int(c) for o, c in zip(sums.objects, sums.counts)}
        result['sum_numbers'] = {'window_ms': list(window), 'max_object': number_max,
                                 'values_uv': {str(o): float(v) for o, v in zip(sums.objects, numbers)},
                                 'note': 'Mean baseline-corrected amplitude of the summed ERP; descriptive, not calibrated.'}
        self.writer.submit('trial', (self.index + 1, self.panel._last_run_export_data or {}, result, (sums, window)))
        self.last_sums = sums
        self.results.append(result)
        self.trial_finished.emit(result)
        self.index += 1
        write_stim_control(self.directory, {'state': 'paused', 'message': 'Между пробами'})
        self.deadline = time.monotonic() + self.plan.pause_s
        self._state('pause', f'Проба сохранена. Выполнено {self.index}/{len(self.targets)}')
        self.panel.set_experiment_owned(True)

    def _trial_sums(self):
        try:
            return live_sums(self.panel)
        except ValueError:
            return empty_sums()

    def _component_finished(self, kind, code):
        if kind == 'stimulus' and self.active and self.stimulus_launched:
            self.fail(f'Стимулятор завершился до окончания сессии (код {code})')

    def fail(self, message):
        if self.state == 'failed':
            return
        self.stop(message, failed=True)

    def stop(self, reason='Остановлено оператором', protocol_complete=False, failed=False):
        if not self.active and self.state != 'failed':
            return
        self.timer.stop()
        self.completed = protocol_complete
        self._state('failed' if failed else ('completed' if protocol_complete else 'stopped'), reason)
        errors = []
        try:
            if self.directory and self.directory.is_dir():
                try:
                    write_stim_control(self.directory, {'state': 'done'})
                except Exception as exc:
                    errors.append(str(exc))
            if self.panel._recording_epochs:
                try:
                    self.panel._on_stop_analysis_clicked()
                    run = self.panel._last_run_export_data or {}
                    partial = {'trial': self.index + 1, 'target': self.targets[self.index],
                               'winner': None, 'status': 'interrupted', 'reason': reason}
                    if self.writer and not self.writer.closing.is_set():
                        self.writer.submit('trial', (self.index + 1, run, partial,
                                                     (self._trial_sums(), decision_window(self.panel))))
                except Exception as exc:
                    errors.append(str(exc))
            try:
                self.panel._on_disconnect_clicked()
            except Exception as exc:
                errors.append(str(exc))
        finally:
            if self.stimulus_launched:
                self.components.stop_processes()
            if errors:
                reason += '; ошибка завершения: ' + '; '.join(errors)
                self._state('failed', reason)
                protocol_complete = False
            if self.writer:
                self.writer.finish(reason, protocol_complete=protocol_complete)

    def _writer_finished(self):
        self.panel.set_experiment_owned(False)
        self.components.managed_session = False
        self.state_changed.emit(self.state, f'{self.message}. Архив закрыт: {self.directory}')
        self.shutdown_progress.emit()

    def begin_shutdown(self):
        if self.active:
            self.stop('Главное окно закрыто')

    def stopped(self):
        return not (self.writer and self.writer.isRunning())
