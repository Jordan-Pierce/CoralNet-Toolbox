from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QDialog, QPushButton

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ProgressBar(QDialog):
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

    def update_progress(self):
        if not self.canceled:
            self.value += 1
            self.progress_bar.setValue(self.value)
            if self.value >= self.max_value:
                self.stop_progress()

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
        else:
            raise ValueError(f"Value must be between 0 and {self.max_value}")

    def stop_progress(self):
        self.value = self.max_value
        self.progress_bar.setValue(self.value)

    def cancel(self):
        self.canceled = True
        self.stop_progress()

    def wasCanceled(self):
        return self.canceled