import os
import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QLineEdit, QPushButton, QFileDialog, QApplication,
                             QMessageBox, QLabel, QButtonGroup, QRadioButton)

from coralnet_toolbox.Icons import get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


CACHE_BASE = ".cache"
SCREENSHOTS_SUBDIR = "screenshots"


def get_screenshot_dir():
    """Return the default screenshot directory path (not created here)."""
    return Path(CACHE_BASE) / SCREENSHOTS_SUBDIR


def get_default_filename():
    """Return a timestamped default file name for a capture."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"


# ----------------------------------------------------------------------------------------------------------------------
# Main Dialog Class
# ----------------------------------------------------------------------------------------------------------------------


class CaptureView(QDialog):
    def __init__(self, main_window):
        """Initialize the capture view dialog."""
        super().__init__(main_window)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("camera.svg"))
        self.setWindowTitle("Capture View")
        self.resize(600, 500)

        # Main layout for the dialog
        self.layout = QVBoxLayout(self)

        # Set up the UI sections
        self.setup_info_layout(parent_layout=self.layout)
        self.setup_source_layout(parent_layout=self.layout)
        self.setup_destination_layout(parent_layout=self.layout)
        self.setup_output_layout(parent_layout=self.layout)
        # Add a stretch to push the buttons to the bottom of the dialog
        self.layout.addStretch(1)
        self.setup_buttons_layout(parent_layout=self.layout)

        # Set initial state
        self.update_ui_for_destination()

    def showEvent(self, event):
        """Handle show event, refreshing the defaults each time the dialog opens."""
        super().showEvent(event)
        self.refresh_defaults()
        self.update_ui_for_source_availability()
        self.update_ui_for_destination()

    def setup_info_layout(self, parent_layout=None):
        """Set up the information layout section."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_text = (
            "Captures exactly what is currently on screen, at the current zoom, pan, and transparency.<br><br>"
            "<b>Application Window:</b> the entire toolbox window, including all docked panels.<br>"
            "<b>Annotation View:</b> only the annotation canvas, without the frame or scroll bars."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_source_layout(self, parent_layout=None):
        """Set up the capture source layout."""
        groupbox = QGroupBox("Source")
        layout = QVBoxLayout()

        self.application_radio = QRadioButton("Application Window")
        self.application_radio.setToolTip("Capture the entire toolbox window, including all docked panels.")
        self.annotation_radio = QRadioButton("Annotation View")
        self.annotation_radio.setToolTip("Capture only the annotation canvas, exactly as it appears on screen.")

        self.source_group = QButtonGroup(self)
        self.source_group.addButton(self.application_radio)
        self.source_group.addButton(self.annotation_radio)

        layout.addWidget(self.application_radio)
        layout.addWidget(self.annotation_radio)

        self.application_radio.setChecked(True)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_destination_layout(self, parent_layout=None):
        """Set up the capture destination layout."""
        groupbox = QGroupBox("Destination")
        layout = QVBoxLayout()

        self.clipboard_radio = QRadioButton("Copy to Clipboard")
        self.clipboard_radio.setToolTip("Copy the capture to the system clipboard for pasting into another "
                                        "application.")
        self.disk_radio = QRadioButton("Save to Disk")
        self.disk_radio.setToolTip("Write the capture to an image file in the output directory below.")

        self.destination_group = QButtonGroup(self)
        self.destination_group.addButton(self.clipboard_radio)
        self.destination_group.addButton(self.disk_radio)
        self.destination_group.buttonClicked.connect(self.update_ui_for_destination)

        layout.addWidget(self.clipboard_radio)
        layout.addWidget(self.disk_radio)

        self.clipboard_radio.setChecked(True)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_output_layout(self, parent_layout=None):
        """Set up the output directory and file name layout."""
        self.output_groupbox = QGroupBox("Output")
        layout = QFormLayout()

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.output_dir_button.setToolTip("Browse for a directory.")
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setToolTip("Name of the image file to write.\n"
                                         "Defaults to a timestamp; '.png' is added if no extension is given.")
        layout.addRow("File Name:", self.output_name_edit)

        self.output_groupbox.setLayout(layout)
        parent_layout.addWidget(self.output_groupbox)

    def setup_buttons_layout(self, parent_layout=None):
        """Set up the buttons layout."""
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.run_capture_process)
        self.capture_button.setToolTip("Capture the selected view to the selected destination.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setToolTip("Close this dialog without capturing.")
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def refresh_defaults(self):
        """Refresh the default output directory and timestamped file name."""
        screenshot_dir = get_screenshot_dir()
        if not self.output_dir_edit.text():
            self.output_dir_edit.setText(str(screenshot_dir))

        # Show the resolved absolute path so the relative default is not a mystery
        resolved = os.path.abspath(self.output_dir_edit.text())
        self.output_dir_edit.setToolTip(f"Directory where the capture will be saved.\nResolves to: {resolved}")

        # Every open gets a fresh timestamp
        self.output_name_edit.setText(get_default_filename())

    def update_ui_for_source_availability(self):
        """Disable the annotation view source when no image is loaded."""
        has_image = bool(getattr(self.annotation_window, 'current_image_path', None))
        self.annotation_radio.setEnabled(has_image)

        if has_image:
            self.annotation_radio.setToolTip("Capture only the annotation canvas, exactly as it appears on screen.")
        else:
            self.annotation_radio.setToolTip("No image is currently loaded in the Annotation Window.")
            if self.annotation_radio.isChecked():
                self.application_radio.setChecked(True)

    def update_ui_for_destination(self):
        """Enable the output group box only when saving to disk."""
        self.output_groupbox.setEnabled(self.disk_radio.isChecked())

    def browse_output_dir(self):
        """Browse for the output directory."""
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Select Output Directory",
                                                     self.output_dir_edit.text())
        if directory:
            self.output_dir_edit.setText(directory)
            resolved = os.path.abspath(directory)
            self.output_dir_edit.setToolTip(f"Directory where the capture will be saved.\nResolves to: {resolved}")

    def get_output_path(self):
        """Resolve the full output path, or None if the inputs are invalid."""
        directory = self.output_dir_edit.text().strip()
        if not directory:
            QMessageBox.warning(self,
                                "Missing Input",
                                "Please select an output directory.")
            return None

        filename = self.output_name_edit.text().strip()
        if not filename:
            QMessageBox.warning(self,
                                "Missing Input",
                                "Please enter a file name.")
            return None

        # Add a default extension if the user did not provide one
        if not os.path.splitext(filename)[1]:
            filename += ".png"

        return os.path.abspath(os.path.join(directory, filename))

    def grab_pixmap(self):
        """Grab the pixmap for the currently selected source."""
        if self.annotation_radio.isChecked():
            return self.annotation_window.viewport().grab()
        return self.main_window.grab()

    def run_capture_process(self):
        """Run the capture process for the selected source and destination."""
        to_disk = self.disk_radio.isChecked()

        # Resolve and confirm the output path BEFORE hiding, so no prompt is orphaned
        output_path = None
        if to_disk:
            output_path = self.get_output_path()
            if not output_path:
                return

            if os.path.exists(output_path):
                if QMessageBox.warning(self,
                                       "File Exists",
                                       f"{output_path} already exists.\nOverwrite it?",
                                       QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                    return

        source_name = "Annotation View" if self.annotation_radio.isChecked() else "Application Window"

        # Hide the dialog so it does not appear in its own capture
        self.hide()
        QApplication.processEvents()

        # Set the cursor only after the repaint, so the busy cursor is not captured
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pixmap = self.grab_pixmap()

            if to_disk:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if not pixmap.save(output_path):
                    raise IOError(f"Could not write the image to {output_path}")
                message = f"Captured {source_name} — Saved to {output_path}"
            else:
                QApplication.clipboard().setPixmap(pixmap)
                message = f"Captured {source_name} — Copied to Clipboard"

            self.main_window.status_bar.showMessage(message, 5000)
            self.accept()

        except Exception as e:
            # Bring the dialog back so the user is not left with a vanished window
            self.show()
            QMessageBox.critical(self, "Error", f"An error occurred during capture: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        """Handle the close event."""
        super().closeEvent(event)
