"""Painted native monitors and the dark studio palette."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

GREEN = "#69bca4"
ORANGE = "#ed713e"

STYLE = """
QWidget { background: #202c35; color: #e3ebef; font: 10pt 'Segoe UI'; }
QMainWindow, QTabWidget::pane { background: #1b252d; }
QMenuBar, QMenu, QStatusBar { background: #18222a; }
QMenuBar::item { padding: 6px 12px; background: transparent; color: #a3b5c1; }
QMenu::item { padding: 7px 24px; }
QMenu::item:selected { background: #354954; }
QToolBar { background: #26343e; padding: 8px; spacing: 8px; border: 0; }
QToolButton { background: #32434e; border: 1px solid #40525c; border-radius: 4px; padding: 7px 12px; }
QToolButton:hover, QPushButton:hover { background: #425762; }
QDockWidget { font-weight: 600; color: #a9bbc6; }
QDockWidget::title { background: #26343e; padding: 10px; }
QDockWidget::close-button, QDockWidget::float-button { padding: 3px; }
QMainWindow::separator { background: #152129; width: 4px; height: 4px; }
QMainWindow::separator:hover { background: #69bca4; }
QTabBar::tab { background: #1b252d; color: #9aadb9; padding: 10px 22px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { background: #2d3e49; color: #fff; border-bottom: 2px solid #ed713e; }
QTabWidget::pane { border: 0; }
QLabel { background: transparent; }
QLabel[muted="true"] { color: #9aadb9; }
QLabel[heading="true"] { font-size: 15pt; font-weight: 600; }
QLabel[accent="true"] { color: #69bca4; }
QGroupBox { border: 1px solid #354751; border-radius: 5px; margin-top: 17px; padding: 16px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #b4c3cd; padding: 0 5px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background: #1b2831; border: 1px solid #3a4d59; border-radius: 4px; padding: 7px; selection-background-color: #437b70; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #69bca4; }
QLineEdit:disabled, QComboBox:disabled { color: #718591; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView { background: #293a45; selection-background-color: #405964; }
QPushButton { background: #334650; border: 1px solid #435761; border-radius: 4px; padding: 9px 14px; font-weight: 600; }
QPushButton[primary="true"], QToolButton[primary="true"] { background: #e96b39; color: white; border: 1px solid #f17e4f; }
QPushButton[primary="true"]:hover, QToolButton[primary="true"]:hover { background: #f27d4c; }
QPushButton:disabled, QToolButton:disabled { background: #293842; color: #71848f; border-color: #334550; }
QCheckBox { spacing: 8px; padding: 4px 0; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #657e88; border-radius: 3px; background: #24353f; }
QCheckBox::indicator:checked { background: #69bca4; border-color: #69bca4; }
QTreeWidget, QTableWidget { border: 0; background: #202c35; alternate-background-color: #24333d; outline: 0; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected, QTableWidget::item:selected { background: #3a505c; }
QHeaderView::section { background: #293b46; color: #a5b8c4; border: 0; padding: 8px; }
QProgressBar { background: #15212a; border: 0; border-radius: 3px; height: 8px; text-align: center; }
QProgressBar::chunk { background: #ed713e; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #40535d; }
QSlider::sub-page:horizontal { background: #ed713e; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #ed713e; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #1b2831; width: 9px; }
QScrollBar::handle:vertical { background: #40535d; min-height: 28px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: #354b58; color: #fff; border: 1px solid #607e8e; padding: 5px; }
"""


class CurvePlot(QWidget):
    def __init__(self, title, metric, parent=None):
        super().__init__(parent)
        self.title, self.series_key, self.history = title, metric, {}
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_history(self, history):
        self.history = history
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#18242c"))
        painter.setPen(QColor("#354751"))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)
        painter.setPen(QColor("#e3ebef"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(16, 27, self.title)
        painter.setFont(QFont("Segoe UI", 8))
        for x, color, text in [(self.width() - 154, GREEN, "Train"), (self.width() - 87, ORANGE, "Validation")]:
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(x, 23, x + 13, 23)
            painter.setPen(QColor("#9aadb9"))
            painter.drawText(x + 18, 27, text)
        area = QRectF(48, 48, self.width() - 66, self.height() - 80)
        series = [self.history.get(self.series_key, []), self.history.get("val_" + self.series_key, [])]
        values = [v for data in series for v in data if math.isfinite(v)]
        maximum = 1.0 if self.series_key == "accuracy" else max(values, default=1) * 1.08 or 1
        for i in range(5):
            y = area.bottom() - area.height() * i / 4
            painter.setPen(QPen(QColor("#2d404c"), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor("#829ba9"))
            painter.drawText(QRectF(3, y - 8, 38, 18), Qt.AlignmentFlag.AlignRight,
                             f"{maximum*i/4:.0%}" if self.series_key == "accuracy" else f"{maximum*i/4:.2f}")
        epochs = max(map(len, series), default=0)
        if not values:
            painter.setPen(QColor("#829ba9"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Learning curves appear after the first epoch")
        for data, color in zip(series, (GREEN, ORANGE)):
            if not data:
                continue
            points = [QPointF(area.left() + i * area.width() / max(1, epochs - 1),
                              area.bottom() - v / maximum * area.height()) for i, v in enumerate(data)]
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(color), 2.3))
            painter.drawPath(path)
            painter.setBrush(QColor(color))
            painter.drawEllipse(points[-1], 3, 3)
        painter.setPen(QColor("#829ba9"))
        painter.drawText(48, self.height() - 11, "Epoch 1")
        painter.drawText(self.width() - 80, self.height() - 11, str(epochs) if epochs else "—")


CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
               (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(0,17),(17,18),(18,19),(19,20)]


class MediaCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame, self.landmarks = None, None
        self.mirrored = False
        self.message = "Open an image, video, or landmark sample\nfrom the project explorer"
        self.setMinimumSize(240, 230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, frame, info=None):
        self.frame = frame
        self.landmarks = (info or {}).get("landmarks")
        self.update()

    def set_mirrored(self, enabled):
        self.mirrored = bool(enabled)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#111c23"))
        if self.frame is None:
            painter.setPen(QPen(QColor("#1d303b"), 1))
            for x in range(0, self.width(), 48):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), 48):
                painter.drawLine(0, y, self.width(), y)
            painter.setPen(QColor("#8fa7b6"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        size = self.frame.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        area = QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                      size.width(), size.height())
        if self.mirrored:
            painter.translate(self.width(), 0)
            painter.scale(-1, 1)
        painter.drawImage(area, self.frame)
        if self.landmarks:
            for hand in range(2):
                if not self.landmarks[126 + hand]:
                    continue
                raw = self.landmarks[hand * 63: (hand + 1) * 63]
                points = [QPointF(area.x() + raw[i] * area.width(), area.y() + raw[i+1] * area.height())
                          for i in range(0, 63, 3)]
                painter.setPen(QPen(QColor(GREEN), 2))
                for a, b in CONNECTIONS:
                    painter.drawLine(points[a], points[b])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#ffae7c"))
                for point in points:
                    painter.drawEllipse(point, 3, 3)
