import warnings
import time

from datetime import datetime

from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QSizePolicy

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TimerWorker(QThread):
    """Worker thread for accurate timing."""
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = False
        self.reset_flag = False
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            if self.running:
                time.sleep(1)
                self.update_signal.emit()
            else:
                time.sleep(0.1)  # Check frequently for start signal

    def start_timer(self):
        self.running = True

    def stop_timer(self):
        self.running = False

    def reset_timer(self):
        self.running = False
        self.reset_flag = True

    def stop(self):
        self.stop_flag = True


class TimerWidget(QWidget):
    """A timer widget with start, stop, and reset buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_elapsed = 0  # Time in seconds

        # Worker thread for user interactive timing
        self.worker = TimerWorker()
        self.worker.update_signal.connect(self.update_time)
        self.worker.start()  # Start the thread

        # Background worker for total duration tracking (always running)
        self.background_worker = TimerWorker()
        self.background_worker.update_signal.connect(self.update_background_time)
        self.background_worker.start()
        self.background_worker.start_timer()  # Always running
        self.background_total_duration = 0

        # Logging attributes
        self.start_time = datetime.now()
        self.events = []
        self.total_duration = 0  # Total seconds the timer has been running (from background)
        self.last_start_time = None

        self.setup_ui()

    def closeEvent(self, event):
        """Stop the worker threads when closing."""
        self.worker.stop()
        self.background_worker.stop()
        self.worker.update_signal.disconnect()
        self.background_worker.update_signal.disconnect()
        self.worker.wait()
        self.background_worker.wait()
        super().closeEvent(event)

    def stop_threads(self):
        """Stop the worker threads."""
        self.worker.stop()
        self.background_worker.stop()
        self.worker.update_signal.disconnect()
        self.background_worker.update_signal.disconnect()
        self.worker.wait()
        self.background_worker.wait()

    def setup_ui(self):
        """Set up the user interface."""
        # Compact layout and spacing to minimize height
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Time display (more readable/default styling)
        self.time_label = QLabel("00:00:00")
        self.time_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.time_label.setStyleSheet(
            "color: #333; padding: 6px; background-color: #F0F0F0; border-radius: 4px;"
        )
        self.time_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        layout.addWidget(self.time_label)

        # Buttons layout (small buttons)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(2)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        # Let the button size itself; allow horizontal shrinking but fixed height
        self.start_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)
        self.stop_button.setEnabled(False)
        self.stop_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_timer)
        self.reset_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.reset_button)

        layout.addLayout(buttons_layout)

        # Make the widget request a reasonable minimum height but allow growth
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def start_timer(self):
        """Start the timer."""
        self.worker.start_timer()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.last_start_time = datetime.now()
        self.events.append({'action': 'start', 'timestamp': self.last_start_time.isoformat()})

    def stop_timer(self):
        """Stop the timer."""
        self.worker.stop_timer()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.last_start_time:
            duration = (datetime.now() - self.last_start_time).total_seconds()
            self.total_duration += duration
            self.last_start_time = None
        self.events.append({'action': 'stop', 'timestamp': datetime.now().isoformat()})

    def reset_timer(self):
        """Reset the timer."""
        was_running = self.worker.running
        self.worker.reset_timer()
        self.time_elapsed = 0
        self.update_display()
        self.total_duration = 0
        self.last_start_time = None
        self.events.append({'action': 'reset', 'timestamp': datetime.now().isoformat()})
        
        if was_running:
            # If it was running, restart it after reset
            self.worker.start_timer()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.last_start_time = datetime.now()
            self.events.append({'action': 'start', 'timestamp': self.last_start_time.isoformat()})
        else:
            # If it was stopped, keep buttons as is (start enabled, stop disabled)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_time(self):
        """Update the elapsed time."""
        self.time_elapsed += 1
        self.update_display()

    def update_background_time(self):
        """Update the background total duration."""
        self.background_total_duration += 1
        self.total_duration = self.background_total_duration

    def update_display(self):
        """Update the time display."""
        hours = self.time_elapsed // 3600
        minutes = (self.time_elapsed % 3600) // 60
        seconds = self.time_elapsed % 60
        self.time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def to_dict(self):
        """Serialize the timer data to a dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'events': self.events,
            'total_duration': self.total_duration,
            'background_total_duration': self.background_total_duration
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize the timer data from a dictionary."""
        instance = cls()
        instance.start_time = datetime.fromisoformat(data['start_time'])
        instance.events = data['events']
        instance.total_duration = data['total_duration']
        instance.background_total_duration = data.get('background_total_duration', data['total_duration'])
        # Note: time_elapsed is reset to 0, as it's the current session display
        return instance


class TimerWindow(QWidget):
    """A simple container widget containing the timer widget.

    Note: this used to be a checkable QGroupBox; hiding/collapsing is
    removed per UI changes — the timer is always visible in its dock.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("A timer to track work sessions with start, stop, and reset functionality.\n"
                        "Tracks and saves total duration across sessions and logs all timer actions.")
        self.timer_widget = TimerWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.timer_widget)
        # Allow the container to be at least the preferred height but not smaller.
        # Use the widget's sizeHint after activating the layout so the minimal
        # height is based on the actual widget metrics rather than hardcoded pixels.
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        layout.activate()
        self.timer_widget.adjustSize()
        self.adjustSize()
        self.setMinimumHeight(self.timer_widget.sizeHint().height())

    def to_dict(self):
        """Serialize the timer widget data to a dictionary."""
        return self.timer_widget.to_dict()

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        # Replace timer widget with deserialized one
        layout = instance.layout()
        # Remove old widget
        if layout and layout.count() > 0:
            old_widget = layout.itemAt(0).widget()
            if old_widget and hasattr(old_widget, 'stop_threads'):
                old_widget.stop_threads()
            if old_widget:
                layout.removeWidget(old_widget)
                old_widget.setParent(None)

        instance.timer_widget = TimerWidget.from_dict(data)
        layout.addWidget(instance.timer_widget)
        return instance