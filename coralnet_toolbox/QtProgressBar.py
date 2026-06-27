import warnings

from PyQt5.QtCore import Qt, QSize, QRectF, QTimer
from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QEventLoop
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CircularProgressRing(QWidget):
    """Circular progress indicator used by ProgressBar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(app_theme.scale_int(132), app_theme.scale_int(132))

        self._value = 0
        self._maximum = 100
        self._busy = False
        self._busy_angle = 0

        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(80)
        self._busy_timer.timeout.connect(self._step_busy)

    def sizeHint(self):
        return QSize(app_theme.scale_int(140), app_theme.scale_int(140))

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_progress(self, value, maximum):
        self._value = max(0, int(value))
        self._maximum = max(0, int(maximum))
        self._busy = self._maximum == 0

        if self._busy:
            if not self._busy_timer.isActive():
                self._busy_timer.start()
        else:
            self._busy_timer.stop()

        self.update()

    def set_busy(self, busy):
        self._busy = bool(busy)
        if self._busy:
            if not self._busy_timer.isActive():
                self._busy_timer.start()
        else:
            self._busy_timer.stop()
        self.update()

    def _step_busy(self):
        self._busy_angle = (self._busy_angle + 14) % 360
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        width = self.width()
        height = self.height()
        side = min(width, height)

        outer_margin = app_theme.scale_int(8)
        max_width = app_theme.scale_int(10)
        diameter = max(app_theme.scale_int(58), side - (outer_margin * 2))

        left = (width - diameter) / 2.0
        top = (height - diameter) / 2.0

        circle_rect = QRectF(left, top, diameter, diameter)
        arc_rect = circle_rect.adjusted(
            max_width / 2.0,
            max_width / 2.0,
            -max_width / 2.0,
            -max_width / 2.0,
        )

        inner_rect = arc_rect.adjusted(
            app_theme.scale_int(5),
            app_theme.scale_int(5),
            -app_theme.scale_int(5),
            -app_theme.scale_int(5),
        )

        fill_color = QColor(app_theme.BACKGROUND_COLOR)
        fill_color.setAlpha(235)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill_color)
        painter.drawEllipse(inner_rect)

        track_color = QColor(app_theme.SURFACE_BORDER_COLOR)
        track_color.setAlpha(200)
        track_pen = QPen(track_color)
        track_pen.setWidth(app_theme.scale_int(3))
        track_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(arc_rect, 90 * 16, 360 * 16)

        if self._busy:
            glow_color = QColor(app_theme.ACCENT_HOVER_COLOR)
            glow_color.setAlpha(70)
            glow_pen = QPen(glow_color)
            glow_pen.setWidth(max_width + app_theme.scale_int(3))
            glow_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(glow_pen)
            painter.drawArc(arc_rect, self._busy_angle * 16, 250 * 16)

            busy_pen = QPen(app_theme.ACCENT_COLOR)
            busy_pen.setWidth(max_width)
            busy_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(busy_pen)
            painter.drawArc(arc_rect, self._busy_angle * 16, 250 * 16)

            center_text = "…"
        else:
            ratio = 0.0 if self._maximum <= 0 else min(1.0, self._value / float(self._maximum))
            progress_width = app_theme.scale_int(4 + (6 * ratio))

            glow_color = QColor(app_theme.ACCENT_HOVER_COLOR)
            glow_color.setAlpha(70)
            glow_pen = QPen(glow_color)
            glow_pen.setWidth(progress_width + app_theme.scale_int(3))
            glow_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(glow_pen)
            painter.drawArc(arc_rect, 90 * 16, int(-ratio * 360 * 16))

            progress_pen = QPen(app_theme.ACCENT_COLOR)
            progress_pen.setWidth(progress_width)
            progress_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(progress_pen)
            painter.drawArc(arc_rect, 90 * 16, int(-ratio * 360 * 16))

            center_text = f"{int(round(ratio * 100))}%"

        text_font = QFont(app_theme.APP_FONT_FAMILY)
        text_font.setBold(True)
        text_font.setPixelSize(app_theme.scale_int(17))
        painter.setFont(text_font)
        painter.setPen(QColor(app_theme.TEXT_PRIMARY_COLOR))
        painter.drawText(inner_rect.toRect(), Qt.AlignCenter, center_text)

        painter.end()


class ProgressBar(QDialog):
    """A dialog that displays a circular progress indicator with cancel support."""

    progress_updated = pyqtSignal(int)

    def __init__(self, parent=None, title="Progress"):
        super().__init__(parent)

        self._busy_text = "Processing..."

        self.setObjectName("ProgressDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.resize(app_theme.scale_int(260), app_theme.scale_int(260))
        self.setMinimumSize(app_theme.scale_int(220), app_theme.scale_int(220))

        try:
            if parent is not None:
                icon = parent.windowIcon()
                if icon is not None and not icon.isNull():
                    self.setWindowIcon(icon)
                else:
                    self.setWindowIcon(get_window_icon("coralnet.svg"))
            else:
                self.setWindowIcon(get_window_icon("coralnet.svg"))
        except Exception:
            try:
                self.setWindowIcon(get_window_icon("coralnet.svg"))
            except Exception:
                pass

        self.value = 0
        self.max_value = 100
        self.canceled = False

        self.progress_bar = QProgressBar(self)
        self.progress_bar.hide()
        self.progress_bar.setRange(0, self.max_value)
        self.progress_bar.valueChanged.connect(lambda _value: self._sync_indicator())

        self.ring = CircularProgressRing(self)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setObjectName("ProgressCancelButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setIcon(get_icon("remove.svg"))
        self.cancel_button.setIconSize(app_theme.scale_size(12))
        self.cancel_button.setToolTip("Stop the current operation")
        self.cancel_button.clicked.connect(self.cancel)

        self.message_label = QLabel(title, self)
        self.message_label.setObjectName("ProgressMessage")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.message_label.setMaximumWidth(app_theme.scale_int(160))

        self.panel = QFrame(self)
        self.panel.setObjectName("ProgressPanel")
        self.panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(
            app_theme.scale_int(14),
            app_theme.scale_int(14),
            app_theme.scale_int(14),
            app_theme.scale_int(12),
        )
        panel_layout.setSpacing(app_theme.scale_int(10))
        panel_layout.addWidget(self.ring, 0, Qt.AlignCenter)
        panel_layout.addWidget(self.message_label, 0, Qt.AlignCenter)
        panel_layout.addWidget(self.cancel_button, 0, Qt.AlignCenter)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(
            app_theme.scale_int(12),
            app_theme.scale_int(12),
            app_theme.scale_int(12),
            app_theme.scale_int(12),
        )
        self.layout.addStretch(1)
        self.layout.addWidget(self.panel, 0, Qt.AlignCenter)
        self.layout.addStretch(1)

        self.setStyleSheet(
            app_theme.scale_qss(
                f"""
QDialog#ProgressDialog {{
    background-color: transparent;
}}
QFrame#ProgressPanel {{
    background-color: rgba(19, 22, 31, 228);
    border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};
    border-radius: 18px;
}}
QLabel#ProgressMessage {{
    color: {app_theme.TEXT_PRIMARY_COLOR.name()};
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-size: 11px;
    font-weight: 600;
}}
QPushButton#ProgressCancelButton {{
    background-color: {app_theme.SURFACE_COLOR.name()};
    color: {app_theme.TEXT_PRIMARY_COLOR.name()};
    border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};
    border-radius: 10px;
    padding: 5px 10px;
    min-width: 82px;
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-weight: 600;
}}
QPushButton#ProgressCancelButton:hover {{
    background-color: {app_theme.SURFACE_ELEVATED_COLOR.name()};
    border-color: {app_theme.ACCENT_COLOR.name()};
}}
QPushButton#ProgressCancelButton:pressed {{
    background-color: {app_theme.ACCENT_SOFT_COLOR.name()};
}}
QPushButton#ProgressCancelButton:disabled {{
    color: {app_theme.TEXT_MUTED_COLOR.name()};
    background-color: {app_theme.BACKGROUND_ALT_COLOR.name()};
}}
"""
            )
        )

        self.progress_updated.connect(self.update_progress)

        self._sync_indicator()

    def setWindowTitle(self, title):
        """Keep the in-panel title text in sync with the dialog title."""
        super().setWindowTitle(title)
        if hasattr(self, "message_label"):
            self.message_label.setText(title)

    def _sync_indicator(self):
        self.value = int(self.progress_bar.value())
        self.max_value = int(self.progress_bar.maximum())
        self.ring.set_progress(self.value, self.max_value)

    def set_title(self, title):
        """Update the dialog title and reset progress."""
        self.setWindowTitle(title)
        self.value = 0
        self.canceled = False
        self.progress_bar.setValue(0)
        QApplication.processEvents()

    def set_busy_mode(self, busy_text="Processing..."):
        """Switch to an indeterminate busy state."""
        self._busy_text = busy_text
        self.set_title(busy_text)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.ring.set_busy(True)
        QApplication.processEvents()

    def start_progress(self, max_value):
        """Initialize the dialog with a maximum progress value."""
        self.value = 0
        self.max_value = max_value
        self.canceled = False
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setValue(0)
        self.ring.set_busy(False)
        self.ring.set_progress(0, max_value)

    def set_value(self, value):
        """Set the progress to a specific value."""
        if 0 <= value <= self.max_value:
            self.value = value
            self.progress_bar.setValue(self.value)
            if self.value >= self.max_value:
                self.stop_progress()
        elif value > self.max_value:
            pass

    def update_progress(self, new_title=None):
        """Increment the progress by one step."""
        if new_title is not None:
            self.setWindowTitle(new_title)

        if self.canceled:
            return

        if self.progress_bar.maximum() == 0:
            QApplication.processEvents()
            return

        self.value += 1

        update_interval = max(1, self.max_value // 100)
        is_last_step = self.value >= self.max_value
        is_update_step = self.value % update_interval == 0

        if is_update_step or is_last_step:
            self.progress_bar.setValue(self.value)
            QApplication.processEvents()

    def update_progress_percentage(self, percentage):
        """Update the progress based on a percentage value."""
        if not self.canceled:
            if not 0 <= percentage <= 100:
                return

            new_value = int((percentage / 100.0) * self.max_value)
            self.set_value(new_value)
            QApplication.processEvents()

    def finish_progress(self, duration_ms=500):
        """Animate the progress indicator to completion."""
        if self.value >= self.max_value:
            self.stop_progress()
            return

        self.animation = QPropertyAnimation(self.progress_bar, b"value")
        self.animation.setDuration(duration_ms)
        self.animation.setStartValue(self.value)
        self.animation.setEndValue(self.max_value)
        self.animation.start()

        loop = QEventLoop()
        self.animation.finished.connect(loop.quit)
        loop.exec_()

        self.value = self.max_value
        self.progress_bar.setValue(self.value)

    def stop_progress(self):
        """Mark the progress as complete."""
        self.value = self.max_value
        self.progress_bar.setValue(self.value)
        self.ring.set_busy(False)
        self.ring.set_progress(self.value, self.max_value)

    def cancel(self):
        """Handle user cancellation."""
        self.canceled = True
        self.stop_progress()
        QApplication.restoreOverrideCursor()

    def wasCanceled(self):
        """Return whether the progress was canceled."""
        return self.canceled
