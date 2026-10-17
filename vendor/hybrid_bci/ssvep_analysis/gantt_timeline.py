"""
Диаграмма Ганта: время (OX) × события (OY).

Каждая строка — один тип события (лампа, гейт MSI). Серая дорожка = «нет события»;
цветной прямоугольник = интервал, когда событие активно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

Interval = Tuple[int, float, float]  # lamp_index, t_start, t_end
TimeInterval = Tuple[float, float]  # t_start, t_end
MsiEvent = Tuple[float, int]  # lsl_time, winner (1-based)


@dataclass(frozen=True)
class GanttExtraRow:
    """Дополнительная строка событий (например «гейт MSI»)."""

    label: str
    intervals: Sequence[TimeInterval]
    color: str = "#4caf50"


class StimGanttChart:
    """PyQtGraph: полосы ON ламп, опционально гейт, окно MSI, метки решений."""

    TRACK_COLOR = "#1c1c1c"
    TRACK_BORDER = "#3a3a3a"

    def __init__(self, plot: pg.PlotWidget, *, span_sec: float = 30.0) -> None:
        self.span_sec = span_sec
        self.row_height = 0.72
        self._plot = plot
        self._tracks: List[QtWidgets.QGraphicsRectItem] = []
        self._bars: List[QtWidgets.QGraphicsRectItem] = []
        self._msi_window: Optional[pg.LinearRegionItem] = None
        self._now_line: Optional[pg.InfiniteLine] = None
        self._msi_scatter: Optional[pg.ScatterPlotItem] = None

    @classmethod
    def create_plot(cls, span_sec: float = 30.0) -> Tuple["StimGanttChart", pg.PlotWidget]:
        plot = pg.PlotWidget(title="События по времени (■ = активно)")
        plot.showGrid(x=True, y=False, alpha=0.25)
        plot.setLabel("bottom", "Время (LSL)", units="с")
        plot.setLabel("left", "Событие")
        chart = cls(plot, span_sec=span_sec)
        chart._msi_scatter = pg.ScatterPlotItem(
            size=11,
            pen=pg.mkPen("#fff", width=1),
            brush=pg.mkBrush(255, 200, 0, 220),
            symbol="d",
        )
        plot.addItem(chart._msi_scatter)
        return chart, plot

    def clear(self) -> None:
        for item in self._tracks + self._bars:
            self._plot.removeItem(item)
        self._tracks.clear()
        self._bars.clear()
        if self._msi_window is not None:
            self._plot.removeItem(self._msi_window)
            self._msi_window = None
        if self._now_line is not None:
            self._plot.removeItem(self._now_line)
            self._now_line = None
        if self._msi_scatter is not None:
            self._msi_scatter.setData([], [])

    def _row_y_center(self, row: int) -> float:
        return float(row) + 0.5

    def _add_track(self, row: int, t_min: float, t_max: float) -> None:
        y0 = float(row) + (1.0 - self.row_height) * 0.5
        rect = QtWidgets.QGraphicsRectItem(
            QtCore.QRectF(t_min, y0, t_max - t_min, self.row_height)
        )
        brush = QtGui.QBrush(QtGui.QColor(self.TRACK_COLOR))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        rect.setBrush(brush)
        rect.setPen(QtGui.QPen(QtGui.QColor(self.TRACK_BORDER), 1))
        rect.setZValue(-20)
        self._plot.addItem(rect)
        self._tracks.append(rect)

    def _add_bar(
        self,
        row: int,
        t0: float,
        t1: float,
        *,
        color: QtGui.QColor,
        z: int = 0,
    ) -> None:
        if t1 <= t0:
            return
        y0 = float(row) + (1.0 - self.row_height) * 0.5
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(t0, y0, t1 - t0, self.row_height))
        brush = QtGui.QBrush(color)
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        rect.setBrush(brush)
        rect.setPen(QtGui.QPen(QtGui.QColor("#888"), 1))
        rect.setZValue(z)
        self._plot.addItem(rect)
        self._bars.append(rect)

    def update(
        self,
        *,
        t_now: float,
        n_lamps: int,
        lamp_labels: Optional[Sequence[str]] = None,
        intervals: Sequence[Interval],
        extra_rows: Optional[Sequence[GanttExtraRow]] = None,
        msi_window_sec: float,
        gate_allowed: bool,
        msi_events: Sequence[MsiEvent],
    ) -> None:
        extra_rows = list(extra_rows or [])
        n_extra = len(extra_rows)
        n_rows = max(n_lamps, 0) + n_extra
        if n_rows < 1:
            self.clear()
            return

        t_min = float(t_now) - self.span_sec
        t_max = float(t_now)

        for item in self._tracks + self._bars:
            self._plot.removeItem(item)
        self._tracks.clear()
        self._bars.clear()

        # Дорожки (фон «события нет») и полосы ON по лампам
        for row in range(n_lamps):
            self._add_track(row, t_min, t_max)

        for lamp, t0, t1 in intervals:
            if lamp < 0 or lamp >= n_lamps:
                continue
            t0c = max(t0, t_min)
            t1c = min(t1, t_max)
            c = pg.intColor(lamp, hues=max(n_lamps, 2))
            self._add_bar(lamp, t0c, t1c, color=QtGui.QColor(c), z=1)

        # Дополнительные строки (гейт MSI и т.п.)
        for i, erow in enumerate(extra_rows):
            row = n_lamps + i
            self._add_track(row, t_min, t_max)
            qc = QtGui.QColor(erow.color)
            for t0, t1 in erow.intervals:
                self._add_bar(row, max(t0, t_min), min(t1, t_max), color=qc, z=2)

        # Окно MSI (полупрозрачная полоса по всей высоте)
        w0 = max(t_min, t_now - float(msi_window_sec))
        if self._msi_window is None:
            self._msi_window = pg.LinearRegionItem(
                values=(w0, t_now),
                brush=pg.mkBrush(0, 180, 255, 40 if gate_allowed else 15),
                movable=False,
            )
            self._msi_window.setZValue(-10)
            self._plot.addItem(self._msi_window)
        else:
            self._msi_window.setRegion((w0, t_now))
            self._msi_window.setBrush(
                pg.mkBrush(0, 180, 255, 55 if gate_allowed else 18)
            )

        if self._now_line is None:
            self._now_line = pg.InfiniteLine(
                pos=t_now,
                angle=90,
                pen=pg.mkPen("#e8e8e8", width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            self._now_line.setZValue(10)
            self._plot.addItem(self._now_line)
        else:
            self._now_line.setPos(t_now)

        if self._msi_scatter is not None:
            xs: List[float] = []
            ys: List[float] = []
            for te, winner in msi_events:
                if te < t_min or te > t_max:
                    continue
                idx0 = int(winner) - 1
                if 0 <= idx0 < n_lamps:
                    xs.append(float(te))
                    ys.append(self._row_y_center(idx0))
            self._msi_scatter.setData(xs, ys)

        self._plot.setXRange(t_min, t_max, padding=0.02)
        self._plot.setYRange(-0.15, float(n_rows) + 0.15, padding=0)

        ticks: List[Tuple[float, str]] = []
        for i in range(n_lamps):
            if lamp_labels and i < len(lamp_labels):
                label = lamp_labels[i]
            else:
                label = f"L{i + 1}"
            ticks.append((self._row_y_center(i), label))
        for i, erow in enumerate(extra_rows):
            ticks.append((self._row_y_center(n_lamps + i), erow.label))
        ax = self._plot.getAxis("left")
        ax.setTicks([ticks])
