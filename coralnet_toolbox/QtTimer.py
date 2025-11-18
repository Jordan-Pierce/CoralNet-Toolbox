import warnings
import time

from datetime import datetime

from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox

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
        layout = QVBoxLayout(self)

        # Time display
        self.time_label = QLabel("00:00:00")
        self.time_label.setFont(QFont("Arial", 8, QFont.Bold))
        self.time_label.setStyleSheet("color: #333; padding: 10px; background-color: #F0F0F0; border-radius: 5px;")
        layout.addWidget(self.time_label)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_timer)
        buttons_layout.addWidget(self.reset_button)

        layout.addLayout(buttons_layout)

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


class TimerGroupBox(QGroupBox):
    """A group box containing the timer widget."""

    def __init__(self, parent=None):
        super().__init__("Timer", parent)
        self.setToolTip("A timer to track work sessions with start, stop, and reset functionality.\n "
                        "Tracks and saves total duration across sessions and logs all timer actions.")
        self.timer_widget = TimerWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.timer_widget)

        # Make the group box checkable for collapsing
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self.on_toggled)

    def on_toggled(self, checked):
        """Handle the toggle event to show/hide the timer widget."""
        self.timer_widget.setVisible(checked)

    def to_dict(self):
        """Serialize the timer group box data to a dictionary."""
        data = self.timer_widget.to_dict()
        data['checked'] = self.isChecked()
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize the timer group box data from a dictionary."""
        # Create a new instance normally to ensure proper Qt initialization
        instance = cls()
        
        # Stop and replace the timer widget with the deserialized one
        if hasattr(instance.timer_widget, 'stop_threads'):
            instance.timer_widget.stop_threads()
        
        # Remove the old timer widget from layout
        layout = instance.layout()
        if layout and layout.count() > 0:
            old_widget = layout.itemAt(0).widget()
            if old_widget:
                layout.removeWidget(old_widget)
                old_widget.setParent(None)
        
        # Create new timer widget with deserialized data
        instance.timer_widget = TimerWidget.from_dict(data)
        layout.addWidget(instance.timer_widget)
        
        # Set the checked state
        instance.setChecked(data.get('checked', True))
        
        return instance