"""Поточечная сумма эпох по объектам: требуемый массив ответа P300.

Каждая вспышка объекта даёт эпоху — массив (отсчёты × каналы ROI), начинающийся
от момента вспышки (с учётом baseline до неё). Эпохи одного объекта
складываются поточечно: первый отсчёт первой вспышки с первым отсчётом второй
и т. д. Для каждого объекта получается одна кривая на каждый канал.

Два варианта вывода:
1. Единый массив — все суммы в одной таблице: строки — отсчёты от вспышки,
   столбцы — пары «объект × канал».
2. Одно число на объект — средняя амплитуда усреднённой кривой (сумма / число
   вспышек) в окне решения после коррекции baseline, по всем каналам ROI.
   Объект с наибольшим числом — предполагаемая цель. Это простая описательная
   мера, а не откалиброванный классификатор.

Эпохи берутся у исходного анализатора Пронина без изменения: полосовой фильтр
и CAR (если включён) уже применены; отбраковка артефактов к сумме не применяется.
"""
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout

from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit


@dataclass(frozen=True)
class ObjectSums:
    objects: np.ndarray   # (n_obj,) номера плиток
    counts: np.ndarray    # (n_obj,) число просуммированных вспышек
    sums: np.ndarray      # (n_obj, отсчёты, каналы) поточечная сумма эпох
    time_ms: np.ndarray   # (отсчёты,) время относительно вспышки
    channels: list

    @property
    def means(self):
        return self.sums / np.maximum(self.counts, 1)[:, None, None]


def empty_sums(channels=()):
    return ObjectSums(np.empty(0, int), np.empty(0, int), np.empty((0, 0, len(channels))),
                      np.empty(0), list(channels))


def object_sums(epochs_data, time_ms, channels):
    """Сложить эпохи каждого объекта; ValueError при несовместимых формах."""
    keys = sorted((k for k, v in epochs_data.items() if len(v)), key=stim_key_sort_key)
    if not keys:
        return empty_sums(channels)
    epochs = {k: [np.asarray(e, dtype=np.float64).reshape(len(e), -1) for e in epochs_data[k]] for k in keys}
    shapes = {e.shape for values in epochs.values() for e in values}
    if len(shapes) != 1:
        raise ValueError(f'Эпохи имеют разные размеры {sorted(shapes)}; '
                         'вероятно, каналы ROI менялись во время записи')
    sums = [np.stack(epochs[k]).sum(axis=0) for k in keys]
    counts = [len(epochs[k]) for k in keys]
    sums = np.stack(sums)
    names = list(channels)
    if len(names) != sums.shape[2]:
        names = [f'Ch {i + 1}' for i in range(sums.shape[2])]
    time_ms = np.asarray(time_ms if time_ms is not None else [], dtype=np.float64)
    if time_ms.shape != (sums.shape[1],):
        time_ms = np.arange(sums.shape[1], dtype=np.float64)
    return ObjectSums(np.array([stim_key_to_tile_digit(k) for k in keys]), np.array(counts),
                      sums, time_ms, names)


def unified_array(result):
    """Вариант 1: (отсчёты, объекты·каналы) и имена столбцов «объект_канал»."""
    n_obj, n_samples, n_ch = result.sums.shape
    array = result.sums.transpose(1, 0, 2).reshape(n_samples, n_obj * n_ch)
    columns = [f'{obj}_{channel}' for obj in result.objects for channel in result.channels]
    return array, columns


def object_numbers(result, window_ms):
    """Вариант 2: одно число на объект (мкВ) и номер объекта с максимумом (или None)."""
    if not len(result.objects):
        return np.empty(0), None
    means = result.means
    before = result.time_ms < 0
    if before.any():
        means = means - means[:, before, :].mean(axis=1, keepdims=True)
    low, high = sorted(window_ms)
    inside = (result.time_ms >= low) & (result.time_ms <= high)
    if not inside.any():
        return np.full(len(result.objects), np.nan), None
    numbers = means[:, inside, :].mean(axis=(1, 2))
    winner = int(result.objects[int(np.argmax(numbers))]) if len(numbers) > 1 and np.ptp(numbers) > 0 else None
    return numbers, winner


def format_unified(result):
    if not len(result.objects):
        return ''
    array, columns = unified_array(result)
    lines = ['\t'.join(['t, мс'] + columns)]
    for t, row in zip(result.time_ms, array):
        lines.append('\t'.join([f'{t:.1f}'] + [f'{v:.4f}' for v in row]))
    return '\n'.join(lines)


def format_numbers(result, window_ms):
    numbers, winner = object_numbers(result, window_ms)
    if not len(numbers):
        return ''
    lines = ['Объект\tВспышек\tЧисло, мкВ']
    for obj, count, number in zip(result.objects, result.counts, numbers):
        lines.append(f'{obj}\t{count}\t{number:.4f}' + ('\t← максимум' if obj == winner else ''))
    return '\n'.join(lines)


def save_sums(result, folder, window_ms):
    """Сохранить оба варианта: NPZ (без pickle) и CSV для Excel."""
    folder = Path(folder)
    array, columns = unified_array(result)
    numbers, winner = object_numbers(result, window_ms)
    np.savez_compressed(folder / 'erp_sums.npz', objects=result.objects, counts=result.counts,
                        sums=result.sums, means=result.means, time_ms=result.time_ms,
                        channels=np.array(result.channels, dtype=str), unified=array,
                        unified_columns=np.array(columns, dtype=str), numbers=numbers,
                        window_ms=np.asarray(window_ms, dtype=np.float64))
    with (folder / 'erp_unified.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_ms'] + columns)
        for t, row in zip(result.time_ms, array):
            writer.writerow([repr(float(t))] + [repr(float(v)) for v in row])
    with (folder / 'erp_numbers.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['object', 'flashes', 'number_uv', 'is_max'])
        for obj, count, number in zip(result.objects, result.counts, numbers):
            writer.writerow([int(obj), int(count), repr(float(number)), int(obj == winner)])
    return numbers, winner


def live_sums(panel):
    """Сумма по текущим эпохам анализатора и именам выбранных каналов ROI."""
    roi = [i for i, cb in enumerate(panel.channel_checkboxes) if cb.isChecked()]
    names = [panel.stream_names[i] if i < len(panel.stream_names) else f'Ch {i + 1}' for i in roi]
    return object_sums(panel.epochs_data, panel._epoch_geom.time_ms_template, names)


def decision_window(panel):
    """Окно решения анализатора Пронина (мс), то же, что в его настройках."""
    return float(panel.spin_x.value()), float(panel.spin_y.value())


VARIANTS = ('Вариант 1: единый массив (отсчёты × объект_канал)', 'Вариант 2: одно число на объект')


class ObjectSumsView(QGroupBox):
    """Вывод суммы эпох в двух вариантах (значения можно выделить и скопировать)."""

    def __init__(self, panel=None, title='Сумма эпох по объектам (массив P300)', parent=None):
        super().__init__(title, parent)
        self.panel = panel
        self.result = empty_sums()
        self.note = ''
        self._signature = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('Вывод:'))
        self.variant = QComboBox()
        self.variant.addItems(VARIANTS)
        self.variant.currentIndexChanged.connect(self._render)
        row.addWidget(self.variant, stretch=1)
        layout.addLayout(row)
        self.label = QLabel('Ожидание эпох: нужны подключённые маркеры, начатый анализ и вспышки стимулятора.')
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setFont(QFont('Menlo', 10))
        self.text.setStyleSheet('background-color: #111; color: #a8ff9e; border: 1px solid #333;')
        self.text.setPlaceholderText('Здесь появится сумма эпох каждого объекта.')
        self.text.setMinimumHeight(160)
        self.text.setMaximumHeight(320)
        layout.addWidget(self.text)

    def window_ms(self):
        return decision_window(self.panel)

    def numbers(self):
        """Вариант 2 для текущего результата: (числа по объектам, объект с максимумом)."""
        return object_numbers(self.result, self.window_ms())

    def refresh(self):
        """Пересчитать по анализатору, только если появились новые эпохи или сменилось окно."""
        epochs = self.panel.epochs_data
        signature = (id(epochs), tuple((k, len(v)) for k, v in sorted(epochs.items())), self.window_ms())
        if signature != self._signature:
            self._signature = signature
            try:
                self.display(live_sums(self.panel), live=True)
            except ValueError as exc:
                self.display(empty_sums(), str(exc), live=True)
        return self.result

    def display(self, result, note='', live=False):
        if not live:
            self._signature = None  # следующий refresh() снова читает анализатор
        self.result, self.note = result, note
        self._render()

    def _render(self):
        result, note = self.result, self.note
        if not len(result.objects):
            self.label.setText(note or 'Эпох пока нет: нужны подключённые маркеры, начатый анализ и вспышки стимулятора.')
            self.text.clear()
            return
        counts = ', '.join(f'{o}: {c}' for o, c in zip(result.objects, result.counts))
        prefix = (note + '. ' if note else '') + f'Вспышек по объектам — {counts}. '
        bar = self.text.verticalScrollBar()
        position = bar.value()
        if self.variant.currentIndex() == 0:
            array, _columns = unified_array(result)
            self.label.setText(prefix + f'Единый массив {array.shape[0]} × {array.shape[1]}: строки — отсчёты от '
                               'вспышки, столбцы — объект_канал, значения — поточечная сумма эпох, мкВ.')
            self.text.setPlainText(format_unified(result))
        else:
            low, high = self.window_ms()
            _numbers, winner = self.numbers()
            self.label.setText(prefix + f'Число = средняя амплитуда (сумма / вспышки) в окне {low:.0f}–{high:.0f} мс '
                               f'после baseline, по каналам ROI. Максимум: объект {winner if winner is not None else "—"}. '
                               'Описательная мера, не откалиброванный классификатор.')
            self.text.setPlainText(format_numbers(result, (low, high)))
        bar.setValue(position)
