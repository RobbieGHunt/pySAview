"""Full colormap preview with shared-range draggable arrow handles."""
from matplotlib import colormaps
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPolygonF
from PyQt6.QtWidgets import QWidget, QSizePolicy


class ColormapRange(QWidget):
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.low, self.high = 0.0, 1.0
        self.colormap = "Blues"
        self._drag = None
        self.setFixedHeight(34)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._tooltip()

    def _tooltip(self):
        self.setToolTip(f"Drag arrows to choose shades: {self.low:.0%}–{self.high:.0%}. Shared by u and d.")

    def setRange(self, low, high):
        self.low = max(0.0, min(float(low), 1.0))
        self.high = max(self.low, min(float(high), 1.0))
        self._tooltip()
        self.update()

    def setColormap(self, name):
        self.colormap = name
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        left, span = 7, max(1, self.width() - 15)
        cmap = colormaps[self.colormap]
        for i in range(span + 1):
            rgba = cmap(i / span)
            painter.setPen(QColor.fromRgbF(*rgba))
            painter.drawLine(left + i, 1, left + i, 15)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for position in (self.low, self.high):
            x = left + span * position
            painter.setPen(QColor("white"))
            painter.setBrush(QColor("#172554"))
            painter.drawPolygon(QPolygonF([QPointF(x, 14), QPointF(x-6, 27), QPointF(x+6, 27)]))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            position = self._position(event)
            self._drag = "low" if abs(position-self.low) <= abs(position-self.high) else "high"
            self._move(event)

    def _position(self, event):
        return max(0.0, min(1.0, (event.position().x()-7) / max(1, self.width()-15)))

    def _move(self, event):
        position = self._position(event)
        if self._drag == "low":
            self.setRange(min(position, self.high), self.high)
        elif self._drag == "high":
            self.setRange(self.low, max(position, self.low))
        self.rangeChanged.emit(self.low, self.high)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._move(event)

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._move(event)
            self._drag = None
