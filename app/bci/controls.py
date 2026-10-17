"""Launch original upstream programs and its synthetic validation battery."""
import json
import os
from pathlib import Path
import shutil
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, QSettings, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QPlainTextEdit

from .panel import output_root
from .process_tree import ProcessTree

UPSTREAM_COMMIT = '4e5dc7c0165da40c1ec2eacd633032c2415e31ec'


def bundled_source():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
    return root / 'vendor' / 'hybrid_bci'


def prepare_runtime(root):
    destination = Path(root) / 'upstream_runtime' / UPSTREAM_COMMIT
    # Keep mutable upstream settings/logs outside bundled resources. All source
    # is copied verbatim, including both programs' separate app namespaces.
    if not (destination / '.ready').exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundled_source(), destination, dirs_exist_ok=True)
        (destination / '.ready').write_text(UPSTREAM_COMMIT, encoding='ascii')
    return destination


class ValidationJob(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            from validation.p300_synthetic import run_p300_synthetic_battery
            self.completed.emit(run_p300_synthetic_battery(seed=0))
        except Exception as exc:
            self.failed.emit(str(exc))


class ProcessControls(QWidget):
    shutdown_progress = pyqtSignal()
    component_finished = pyqtSignal(str, int)
    managed_session = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings('HybridInterface', 'ProninIntegration')
        self.processes = {}
        self.process_trees = {}
        self.validation_job = None
        self.closing = False
        layout = QVBoxLayout(self)
        description = QLabel('Исходные программы Сергея Пронина запускаются отдельными процессами. '
            'Для P300 используйте встроенную вкладку и «Стимулятор». '
            '«Гибридный протокол» открывает исходный операторский интерфейс P300/SSVEP со своими требованиями к оборудованию. '
            'Алгоритмы и последовательность протокола не изменены.')
        description.setWordWrap(True)
        layout.addWidget(description)
        row = QHBoxLayout()
        row.addWidget(QLabel('Python окружения Пронина:'))
        candidates = list(Path(sys.executable).resolve().parents)[:4] if getattr(sys, 'frozen', False) else [Path(__file__).resolve().parents[2]]
        local_python = next((root / '.venv-pronin' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
                             for root in candidates if (root / '.venv-pronin' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')).is_file()), None)
        default_python = str(local_python) if local_python else ('' if getattr(sys, 'frozen', False) else sys.executable)
        self.python = QLineEdit(str(self.settings.value('python', default_python)))
        row.addWidget(self.python)
        choose = QPushButton('Выбрать…')
        choose.clicked.connect(self.choose_python)
        row.addWidget(choose)
        layout.addLayout(row)
        hint = QLabel('В выбранном окружении нужны зависимости исходной программы. Для стимулятора: PsychoPy, numpy, pylsl. '
                      'Для гибридного протокола дополнительно PyQt5, pyqtgraph, scipy, pyserial и openpyxl. '
                      'Если всё уже работает в отдельном проекте, выберите его Python.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.stimulus_button = QPushButton('Запустить стимулятор Пронина')
        self.stimulus_button.clicked.connect(lambda: self.launch('stimulus'))
        row.addWidget(self.stimulus_button)
        self.protocol_button = QPushButton('Гибридный протокол Пронина')
        self.protocol_button.clicked.connect(lambda: self.launch('protocol'))
        row.addWidget(self.protocol_button)
        stop = QPushButton('Остановить запущенные компоненты')
        stop.clicked.connect(self.stop_processes)
        row.addWidget(stop)
        layout.addLayout(row)
        self.validate_button = QPushButton('Проверка P300 без оборудования (симулятор Пронина)')
        self.validate_button.clicked.connect(self.validate)
        layout.addWidget(self.validate_button)
        self.status = QLabel('Компоненты не запущены. Проверка использует искусственные данные и не измеряет качество на человеке.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(1500)
        layout.addWidget(self.log)

    def choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Python исходного окружения Пронина')
        if path:
            self.python.setText(path)
            self.settings.setValue('python', path)

    def any_running(self):
        return any(p.state() != QProcess.NotRunning for p in self.processes.values())

    def validate_python(self):
        python = self.python.text().strip()
        if not Path(python).is_file() or (getattr(sys, 'frozen', False) and Path(python).resolve() == Path(sys.executable).resolve()):
            raise ValueError('Выберите Python окружения Пронина во вкладке «Запуск и проверка»')
        return python

    def launch(self, kind, extra_args=None):
        if self.managed_session and extra_args is None:
            self.status.setText('Компонентами сейчас управляет единый эксперимент')
            return
        existing = self.processes.get(kind)
        if existing and existing.state() != QProcess.NotRunning:
            self.status.setText('Этот компонент уже запущен')
            return
        other = self.processes.get('protocol' if kind == 'stimulus' else 'stimulus')
        if other and other.state() != QProcess.NotRunning:
            self.status.setText('Сначала остановите текущий компонент: протокол сам управляет стимулятором, два потока маркеров недопустимы.')
            return
        try:
            python = self.validate_python()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        try:
            root = prepare_runtime(output_root())
        except Exception as exc:
            self.status.setText('Не удалось подготовить исходные программы: ' + str(exc))
            return
        script = 'run_app.py' if kind == 'stimulus' else 'scripts/protocol_runner_gui.py'
        args = [str(Path(__file__).with_name('launch_upstream.py')), str(root / script)] + (['--no-analyzer'] if kind == 'stimulus' else [])
        if extra_args:
            args.extend(extra_args)
        process = QProcess(self)
        process.setProgram(python)
        process.setArguments(args)
        process.setWorkingDirectory(str(root))
        env = QProcessEnvironment.systemEnvironment()
        # Do not leak the bundled host's Qt/plugin paths into another Python/Qt.
        for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM'):
            env.remove(key)
        env.insert('PYTHONUNBUFFERED', '1')
        env.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')))
        process.errorOccurred.connect(lambda _error: self._process_error(kind, process))
        process.finished.connect(lambda code, _status: self.status.setText(f'{kind}: процесс завершён, код {code}. Подробности в журнале.'))
        process.started.connect(lambda: self._own_process(kind, process))
        process.finished.connect(lambda: self._release_process(kind))
        process.finished.connect(lambda code, _status: self.component_finished.emit(kind, code))
        process.finished.connect(self.shutdown_progress)
        self.processes[kind] = process
        self.settings.setValue('python', python)
        self.log.appendPlainText('Запуск: ' + python + ' ' + ' '.join(args))
        self.status.setText('Запускается ' + kind)
        process.start()
        return process

    def _process_error(self, kind, process):
        self.status.setText('Ошибка запуска: ' + process.errorString())
        if process.error() == QProcess.FailedToStart:
            self.component_finished.emit(kind, -1)

    def _own_process(self, kind, process):
        try:
            self.process_trees[kind] = ProcessTree(process.processId())
        except Exception as exc:
            process.kill()
            self.log.appendPlainText('Не удалось установить управление дочерними процессами: ' + str(exc))

    def _release_process(self, kind):
        tree = self.process_trees.pop(kind, None)
        if tree:
            tree.close()

    def stop_processes(self):
        for process in self.processes.values():
            if process.state() != QProcess.NotRunning:
                process.terminate()
                QTimer.singleShot(3000, lambda p=process: p.kill() if p.state() != QProcess.NotRunning else None)

    def validate(self):
        if self.validation_job and self.validation_job.isRunning():
            return
        self.validate_button.setEnabled(False)
        self.status.setText('Выполняется исходная синтетическая проверка P300…')
        self.validation_job = ValidationJob(self)
        self.validation_job.completed.connect(self._validation_done)
        self.validation_job.failed.connect(lambda text: self.status.setText('Проверка не завершена: ' + text))
        self.validation_job.finished.connect(lambda: self.validate_button.setEnabled(not self.closing))
        self.validation_job.finished.connect(self.shutdown_progress)
        self.validation_job.start()

    def _validation_done(self, report):
        text = json.dumps(report, ensure_ascii=False, indent=2)
        try:
            path = output_root() / 'p300_synthetic_validation.json'
            path.write_text(text, encoding='utf-8')
            self.status.setText(f'Синтетическая проверка завершена. Отчёт: {path}. Это не проверка на реальной ЭЭГ.')
        except OSError as exc:
            self.status.setText('Проверка завершена, но отчёт не сохранён: ' + str(exc))
        self.log.appendPlainText(text)

    def begin_shutdown(self):
        self.closing = True
        self.stop_processes()

    def stopped(self):
        return all(p.state() == QProcess.NotRunning for p in self.processes.values()) and not (
            self.validation_job and self.validation_job.isRunning())
