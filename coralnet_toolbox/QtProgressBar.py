import warnings

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QDialog, QPushButton, QApplication

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ProgressBar(QDialog):
    progress_updated = pyqtSignal(int)
    
    def __init__(self, parent=None, title="Progress"):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.cancel_button)

        self.value = 0
        self.max_value = 100
        self.canceled = False
        
        self.progress_updated.connect(self.update_progress)

    def update_progress(self):
        if not self.canceled:
            self.value += 1
            self.progress_bar.setValue(self.value)
            if self.value >= self.max_value:
                self.stop_progress()
            QApplication.processEvents()

    def start_progress(self, max_value):
        self.value = 0
        self.max_value = max_value
        self.canceled = False
        self.progress_bar.setRange(0, max_value)

    def set_value(self, value):
        if 0 <= value <= self.max_value:
            self.value = value
            self.progress_bar.setValue(self.value)
            if self.value >= self.max_value:
                self.stop_progress()
        elif value > self.max_value:
            pass

    def stop_progress(self):
        self.value = self.max_value
        self.progress_bar.setValue(self.value)

    def cancel(self):
        self.canceled = True
        self.stop_progress()
        # Make cursor not busy
        QApplication.restoreOverrideCursor()

    def wasCanceled(self):
        return self.canceled