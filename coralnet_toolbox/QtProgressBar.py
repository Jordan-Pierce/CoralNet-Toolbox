import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QEventLoop
from PyQt5.QtWidgets import (
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QPushButton,
    QApplication,
    QLabel,
    QFrame,
    QSizePolicy,
)

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ProgressBar(QDialog):
    """
    A dialog that displays a progress bar with a cancel button.
    Provides methods for updating progress and handling user cancellation.
    """
    progress_updated = pyqtSignal(int)

    def __init__(self, parent=None, title="Progress"):
        """
        Initialize the progress bar dialog.
        
        Args:
            parent: The parent widget (default: None)
            title: The window title (default: "Progress")
            text_label: Optional text to display above the progress bar
        """
        super().__init__(parent)

        self._busy_text = "Processing..."
        self._status_text = "Ready"

        # Setup the window properties
        self.setObjectName("ProgressDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(app_theme.scale_int(420))
        self.resize(app_theme.scale_int(460), app_theme.scale_int(180))
        
        # This tells Qt to delete the widget when it receives a close event.
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # Default window icon: use parent's icon if available otherwise use coralnet logo
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
            # Be robust in case parent doesn't expose windowIcon()
            try:
                self.setWindowIcon(get_window_icon("coralnet.svg"))
            except Exception:
                pass

        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(app_theme.scale_int(12), app_theme.scale_int(12), app_theme.scale_int(12), app_theme.scale_int(12))
        self.layout.setSpacing(0)

        self.card = QFrame(self)
        self.card.setObjectName("ProgressCard")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(app_theme.scale_int(14), app_theme.scale_int(12), app_theme.scale_int(14), app_theme.scale_int(12))
        self.card_layout.setSpacing(app_theme.scale_int(10))

        self.accent_bar = QFrame(self.card)
        self.accent_bar.setObjectName("ProgressAccent")
        self.accent_bar.setFixedHeight(app_theme.scale_int(4))
        self.card_layout.addWidget(self.accent_bar)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(app_theme.scale_int(10))

        self.icon_label = QLabel(self.card)
        self.icon_label.setFixedSize(app_theme.scale_int(32), app_theme.scale_int(32))
        self.icon_label.setAlignment(Qt.AlignCenter)
        try:
            icon_pixmap = self.windowIcon().pixmap(app_theme.scale_int(28), app_theme.scale_int(28))
            self.icon_label.setPixmap(icon_pixmap)
        except Exception:
            pass

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(app_theme.scale_int(2))

        self.title_label = QLabel(title, self.card)
        self.title_label.setObjectName("ProgressTitle")
        self.title_label.setWordWrap(False)

        self.status_label = QLabel("Ready", self.card)
        self.status_label.setObjectName("ProgressStatus")
        self.status_label.setWordWrap(False)

        title_column.addWidget(self.title_label)
        title_column.addWidget(self.status_label)

        header_row.addWidget(self.icon_label)
        header_row.addLayout(title_column)
        header_row.addStretch(1)
        self.card_layout.addLayout(header_row)

        # Create progress bar widget
        self.progress_bar = QProgressBar(self.card)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setObjectName("ProgressBar")
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.card_layout.addWidget(self.progress_bar)

        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(app_theme.scale_int(8))

        self.detail_label = QLabel("Ready", self.card)
        self.detail_label.setObjectName("ProgressDetail")
        self.detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Create cancel button
        self.cancel_button = QPushButton("Cancel", self.card)
        self.cancel_button.setObjectName("ProgressCancelButton")
        self.cancel_button.setIcon(get_icon("remove.svg"))
        self.cancel_button.setIconSize(app_theme.scale_size(14))
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)

        footer_row.addWidget(self.detail_label)
        footer_row.addStretch(1)
        footer_row.addWidget(self.cancel_button)
        self.card_layout.addLayout(footer_row)

        self.layout.addWidget(self.card)

        self.setStyleSheet(
            app_theme.scale_qss(
                f"""
QDialog#ProgressDialog {{
    background-color: {app_theme.BACKGROUND_COLOR.name()};
}}
QFrame#ProgressCard {{
    background-color: {app_theme.SURFACE_ELEVATED_COLOR.name()};
    border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};
    border-radius: 14px;
}}
QFrame#ProgressAccent {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {app_theme.ACCENT_COLOR.name()},
        stop:0.55 {app_theme.ACCENT_HOVER_COLOR.name()},
        stop:1 {app_theme.ACCENT_SOFT_COLOR.name()});
    border-radius: 2px;
}}
QLabel#ProgressTitle {{
    color: {app_theme.TEXT_PRIMARY_COLOR.name()};
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-size: 12px;
    font-weight: 700;
}}
QLabel#ProgressStatus {{
    color: {app_theme.TEXT_SECONDARY_COLOR.name()};
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-size: 10px;
}}
QLabel#ProgressDetail {{
    color: {app_theme.TEXT_SECONDARY_COLOR.name()};
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-size: 10px;
}}
QProgressBar {{
    background-color: {app_theme.BACKGROUND_COLOR.name()};
    border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};
    border-radius: 10px;
    color: {app_theme.TEXT_PRIMARY_COLOR.name()};
    text-align: center;
    min-height: {app_theme.scale_int(24)}px;
    padding: {app_theme.scale_int(2)}px;
    font-family: "{app_theme.APP_FONT_FAMILY}";
    font-weight: 700;
}}
QProgressBar::chunk {{
    border-radius: 9px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {app_theme.ACCENT_COLOR.name()},
        stop:0.55 {app_theme.ACCENT_HOVER_COLOR.name()},
        stop:1 {app_theme.ACCENT_COLOR.name()});
}}
QPushButton#ProgressCancelButton {{
    background-color: {app_theme.SURFACE_COLOR.name()};
    color: {app_theme.TEXT_PRIMARY_COLOR.name()};
    border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};
    border-radius: {app_theme.scale_int(8)}px;
    padding: {app_theme.scale_int(6)}px {app_theme.scale_int(12)}px;
    min-width: {app_theme.scale_int(90)}px;
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

        # Initialize state variables
        self.value = 0
        self.max_value = 100
        self.canceled = False
        self._update_status_text()

        # Connect signal
        self.progress_updated.connect(self.update_progress)

    def setWindowTitle(self, title):
        """Keep the dialog title label in sync with the actual window title."""
        super().setWindowTitle(title)
        if hasattr(self, "title_label"):
            self.title_label.setText(title)
        
    def set_title(self, title):
        """
        Update the window title of the progress bar and reset the progress.
        
        Args:
            title: The new window title
        """
        self.setWindowTitle(title)
        # Reset progress when title changes
        self.value = 0
        self.canceled = False
        self.progress_bar.setValue(0)
        self._update_status_text()
        QApplication.processEvents()
        
    def set_busy_mode(self, busy_text="Processing..."):
        """
        Sets the progress bar to an indeterminate "busy" state for tasks
        of unknown duration.
        """
        self._busy_text = busy_text
        # Update the title to reflect the current task
        self.set_title(busy_text)
        
        # Setting the min and max to 0 enables the busy/indeterminate mode
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self._update_status_text(busy_text)
        QApplication.processEvents()

    def start_progress(self, max_value):
        """
        Initialize the progress bar with a maximum value and reset progress.
        
        Args:
            max_value: The maximum value for the progress bar
        """
        self.value = 0
        self.max_value = max_value
        self.canceled = False
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self._update_status_text()

    def set_value(self, value):
        """
        Set the progress bar to a specific value.
        
        Args:
            value: The new value for the progress bar
        """
        if 0 <= value <= self.max_value:
            self.value = value
            self.progress_bar.setValue(self.value)
            self._update_status_text()
            if self.value >= self.max_value:
                self.stop_progress()
        elif value > self.max_value:
            pass  # Silently ignore values exceeding the maximum
        
    def update_progress(self, new_title=None):
        """
        Increment the progress by one step.
        Updates the UI intermittently to improve performance and checks if progress is complete.
        """
        if new_title is not None:
            self.setWindowTitle(new_title)
            
        if self.canceled:
            return

        self.value += 1

        # --- Performance Improvement ---
        # To avoid excessive UI repaints that slow down the process, we only update
        # the visual progress bar periodically. This aims for about 100 updates
        # over the entire range, ensuring a smooth look without bogging down the main task.
        # 'max(1, ...)' ensures we always have an interval of at least 1.
        update_interval = max(1, self.max_value // 100)

        # We update the bar visually only under two conditions:
        # 1. It's the very last step, to ensure it always finishes at 100%.
        # 2. The current value is a multiple of our calculated interval.
        is_last_step = self.value >= self.max_value
        is_update_step = self.value % update_interval == 0

        if is_update_step or is_last_step:
            self.progress_bar.setValue(self.value)
            self._update_status_text()
            
            # This is crucial. It processes pending events, allowing the GUI to
            # redraw with the new progress value and to respond to user input,
            # like clicking the 'Cancel' button.
            QApplication.processEvents()
            
    def update_progress_percentage(self, percentage):
        """
        Update the progress bar based on a percentage (0-100).
        
        Args:
            percentage: The percentage of completion (0-100)
        """
        if not self.canceled:
            if not 0 <= percentage <= 100:
                return

            new_value = int((percentage / 100.0) * self.max_value)
            self.set_value(new_value)
            QApplication.processEvents()
            
    def finish_progress(self, duration_ms=500):
        """
        Animate the progress bar to its maximum value using a non-blocking animation.
        This creates a smooth visual effect of completion without freezing the UI.
        
        Args:
            duration_ms: The duration in milliseconds for the animation (default: 500)
        """
        # If the progress is already complete, just set the final value and exit.
        if self.value >= self.max_value:
            self.stop_progress()
            return

        # --- Non-Blocking Animation using QPropertyAnimation ---
        # QPropertyAnimation is the standard Qt way to animate widget properties.
        # It runs on the main event loop, so it does not freeze the application
        # like the previous time.sleep() implementation. The property name "value"
        # is passed as a bytes object (b"value").
        self.animation = QPropertyAnimation(self.progress_bar, b"value")
        self.animation.setDuration(duration_ms)
        self.animation.setStartValue(self.value)
        self.animation.setEndValue(self.max_value)
        self.animation.start()

        # We run a local event loop that waits for the animation's 'finished'
        # signal. This ensures that the animation completes visually before
        # this method returns control to the calling code, which is often
        # the desired behavior for a "finishing" step.
        loop = QEventLoop()
        self.animation.finished.connect(loop.quit)
        loop.exec_()

        # Finally, update our internal state variable to match the final progress.
        self.value = self.max_value
        self._update_status_text()

    def stop_progress(self):
        """
        Set the progress bar to its maximum value, marking it as complete.
        """
        self.value = self.max_value
        self.progress_bar.setValue(self.value)
        self._update_status_text()

    def _update_status_text(self, busy_text=None):
        """Update the descriptive text shown beneath the progress bar."""
        if not hasattr(self, "status_label") or not hasattr(self, "detail_label"):
            return

        if self.progress_bar.maximum() == 0:
            text = busy_text or self._busy_text
            self.status_label.setText(text)
            self.detail_label.setText(text)
            return

        maximum = max(1, self.max_value)
        percentage = int((self.value / maximum) * 100)
        text = f"{self.value} / {self.max_value} ({percentage}%)"
        self.status_label.setText(text)
        self.detail_label.setText(text)

    def cancel(self):
        """
        Handle user cancellation.
        Sets the canceled flag to True, stops the progress, and restores the cursor.
        """
        self.canceled = True
        self.stop_progress()
        # Make cursor not busy
        QApplication.restoreOverrideCursor()

    def wasCanceled(self):
        """
        Check if the progress operation was canceled by the user.
        
        Returns:
            bool: True if the operation was canceled, False otherwise
        """
        return self.canceled
