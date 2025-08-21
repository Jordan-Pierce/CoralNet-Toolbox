import warnings

from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QEventLoop
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QDialog, QPushButton, QApplication

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
        """
        super().__init__(parent)

        # Setup the window properties
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(350, 100)

        # Create progress bar widget
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        # Create cancel button
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)

        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.cancel_button)

        # Initialize state variables
        self.value = 0
        self.max_value = 100
        self.canceled = False

        # Connect signal
        self.progress_updated.connect(self.update_progress)
        
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
        QApplication.processEvents()
        
    def set_busy_mode(self, busy_text="Processing..."):
        """
        Sets the progress bar to an indeterminate "busy" state for tasks
        of unknown duration.
        """
        # Update the title to reflect the current task
        self.set_title(busy_text)
        
        # Setting the min and max to 0 enables the busy/indeterminate mode
        self.progress_bar.setRange(0, 0)
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

    def set_value(self, value):
        """
        Set the progress bar to a specific value.
        
        Args:
            value: The new value for the progress bar
        """
        if 0 <= value <= self.max_value:
            self.value = value
            self.progress_bar.setValue(self.value)
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

    def stop_progress(self):
        """
        Set the progress bar to its maximum value, marking it as complete.
        """
        self.value = self.max_value
        self.progress_bar.setValue(self.value)

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
