import os
import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter, QPalette
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QLineEdit, QPushButton, QFileDialog, QApplication,
                             QMessageBox, QLabel, QButtonGroup, QRadioButton, QComboBox)

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


def get_unique_output_path(directory):
    """Return an absolute timestamped path in `directory` that does not exist yet."""
    stem, ext = os.path.splitext(get_default_filename())
    path = os.path.abspath(os.path.join(directory, stem + ext))
    # Two captures in the same second would otherwise overwrite each other
    counter = 1
    while os.path.exists(path):
        path = os.path.abspath(os.path.join(directory, f"{stem}_{counter}{ext}"))
        counter += 1
    return path


def clear_transient_overlays(annotation_window):
    """Clear the active tool's hover overlays so they are not baked into a capture.

    Only the transient crosshair and cursor annotation are removed; the tool stays
    active and the current selection is left untouched.
    """
    tool = annotation_window.tools.get(annotation_window.selected_tool)
    if not tool:
        return

    tool.clear_crosshair()
    tool.clear_cursor_annotation()


def capture_high_res_pixmap(widget, scale=2.0):
    """Render a widget at `scale` times its on-screen pixel density, like a higher-DPR QWidget.grab()."""
    dpr = widget.devicePixelRatioF() * scale
    pixmap = QPixmap(widget.size() * dpr)
    # Scaling comes only from the DPR; a painter transform on top double-scales and crops
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    widget.render(pixmap)
    return pixmap


def get_overlay_windows(main_window, exclude=()):
    """Return the app's other visible windows over `main_window` as (window, rect), bottom to top.

    Dialogs, floating dock panels and menus are separate top-level windows, so render() on the
    main window never paints them. Rects are in main window coordinates. Tooltips and windows
    that do not overlap the main window (e.g. on another monitor) are skipped.
    """
    origin = main_window.mapToGlobal(QPoint(0, 0))
    main_rect = QRect(QPoint(0, 0), main_window.size())

    overlays = []
    for window in QApplication.topLevelWidgets():
        if (window is main_window or window in exclude
                or not window.isVisible() or window.isMinimized()
                or window.windowType() == Qt.ToolTip
                or window.testAttribute(Qt.WA_DontShowOnScreen)):
            continue
        rect = QRect(window.mapToGlobal(QPoint(0, 0)) - origin, window.size())
        if rect.intersects(main_rect):
            overlays.append((window, rect))

    # Top-level order is not stacking order; put whatever currently has focus on top
    def stacking_key(item):
        window = item[0]
        return (window is QApplication.activePopupWidget(),
                window is QApplication.activeModalWidget(),
                window is QApplication.activeWindow())

    return sorted(overlays, key=stacking_key)


def get_capture_bounds(main_window, overlays):
    """Return the rect covering the main window and every overlay window, including its border."""
    bounds = QRect(QPoint(0, 0), main_window.size())
    for _, rect in overlays:
        bounds = bounds.united(rect.adjusted(-1, -1, 1, 1))
    return bounds


def capture_application_pixmap(main_window, scale=2.0, exclude=()):
    """Render the main window plus any open dialogs or floating panels over it, growing to fit them."""
    base = capture_high_res_pixmap(main_window, scale)
    overlays = get_overlay_windows(main_window, exclude)
    if not overlays:
        return base

    bounds = get_capture_bounds(main_window, overlays)
    dpr = base.devicePixelRatio()
    canvas = QPixmap(bounds.size() * dpr)
    canvas.setDevicePixelRatio(dpr)
    canvas.fill(Qt.transparent)

    # render() skips the frame the OS draws, so outline each window to separate it from what is behind
    border = main_window.palette().color(QPalette.Dark)
    painter = QPainter(canvas)
    try:
        painter.translate(-bounds.topLeft())
        painter.drawPixmap(QPoint(0, 0), base)
        for window, rect in overlays:
            painter.fillRect(rect.adjusted(-1, -1, 1, 1), border)
            painter.drawPixmap(rect.topLeft(), capture_high_res_pixmap(window, scale))
    finally:
        painter.end()
    return canvas


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
        self.resize(600, 620)

        # Main layout for the dialog
        self.layout = QVBoxLayout(self)

        # Set up the UI sections
        self.setup_info_layout(parent_layout=self.layout)
        self.setup_source_layout(parent_layout=self.layout)
        self.setup_scale_layout(parent_layout=self.layout)
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
        self.update_output_size_label()

    def setup_info_layout(self, parent_layout=None):
        """Set up the information layout section."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_text = (
            "Captures exactly what is currently on screen, at the current zoom, pan, and transparency.<br><br>"
            "<b>Application Window:</b> the entire toolbox window, including all docked panels, "
            "plus any open dialogs or floating panels over it.<br>"
            "<b>Annotation View:</b> only the annotation canvas, without the frame or scroll bars.<br><br>"
            "<b>Ctrl+F1</b> captures instantly with the settings below, even while this dialog is closed. "
            "When saving to disk, each capture gets a new timestamped file name."
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
        self.application_radio.setToolTip("Capture the entire toolbox window, including all docked panels,\n"
                                          "plus any open dialogs or floating panels over it.")
        self.annotation_radio = QRadioButton("Annotation View")
        self.annotation_radio.setToolTip("Capture only the annotation canvas, exactly as it appears on screen.")

        self.source_group = QButtonGroup(self)
        self.source_group.addButton(self.application_radio)
        self.source_group.addButton(self.annotation_radio)
        self.source_group.buttonClicked.connect(self.update_output_size_label)

        layout.addWidget(self.application_radio)
        layout.addWidget(self.annotation_radio)

        self.application_radio.setChecked(True)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_scale_layout(self, parent_layout=None):
        """Set up the resolution scale layout."""
        groupbox = QGroupBox("Resolution")
        layout = QFormLayout()

        self.scale_combo = QComboBox()
        for scale in (1, 2, 3, 4):
            self.scale_combo.addItem("1x (Screen)" if scale == 1 else f"{scale}x", float(scale))
        self.scale_combo.setCurrentIndex(0)
        self.scale_combo.setToolTip("Multiple of the on-screen resolution to render at.\n"
                                    "Text and outlines are redrawn sharper; icons and thumbnails are only enlarged.\n"
                                    "3x and 4x are for large prints or cropping into a small region.")
        self.scale_combo.currentIndexChanged.connect(self.update_output_size_label)
        layout.addRow("Scale:", self.scale_combo)

        self.output_size_label = QLabel()
        layout.addRow("Output Size:", self.output_size_label)

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
                                         "Defaults to a timestamp; '.png' is added if no extension is given.\n"
                                         "Ctrl+F1 ignores this and always writes a new timestamped file.")
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

    def has_image(self):
        """Return True if an image is loaded in the annotation window."""
        return bool(getattr(self.annotation_window, 'current_image_path', None))

    def update_ui_for_source_availability(self):
        """Disable the annotation view source when no image is loaded."""
        has_image = self.has_image()
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

    def get_scale(self):
        """Return the selected resolution scale."""
        return self.scale_combo.currentData()

    def is_annotation_source(self):
        """Return True if the annotation view is selected and has an image to capture.

        Ctrl+F1 can fire after the image is closed, while the radio still says Annotation View.
        """
        return self.annotation_radio.isChecked() and self.has_image()

    def get_source_widget(self):
        """Return the widget for the currently selected source."""
        if self.is_annotation_source():
            return self.annotation_window.viewport()
        return self.main_window

    def update_output_size_label(self):
        """Show the pixel size the capture will have at the selected scale."""
        widget = self.get_source_widget()
        size = widget.size()
        if widget is self.main_window:
            overlays = get_overlay_windows(self.main_window, exclude=(self,))
            size = get_capture_bounds(self.main_window, overlays).size()
        size = size * (widget.devicePixelRatioF() * self.get_scale())
        megapixels = size.width() * size.height() / 1e6
        self.output_size_label.setText(f"{size.width()} × {size.height()} px ({megapixels:.1f} MP)")

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

    def clear_transient_overlays(self):
        """Clear the active tool's hover overlays so they are not baked into the capture."""
        clear_transient_overlays(self.annotation_window)

    def grab_pixmap(self):
        """Grab the pixmap for the currently selected source at the selected scale."""
        if self.is_annotation_source():
            return capture_high_res_pixmap(self.annotation_window.viewport(), self.get_scale())
        # Never include this dialog in its own capture, even when Ctrl+F1 fires while it is open
        return capture_application_pixmap(self.main_window, self.get_scale(), exclude=(self,))

    def capture_to_destination(self, output_path=None):
        """Capture the selected source at the selected scale, to `output_path` or the clipboard.

        Returns the status message; raises on failure.
        """
        source_name = "Annotation View" if self.is_annotation_source() else "Application Window"

        self.clear_transient_overlays()
        QApplication.processEvents()

        # Set the cursor only after the repaint, so the busy cursor is not captured
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pixmap = self.grab_pixmap()
            source_name += f" ({pixmap.width()}x{pixmap.height()})"

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if not pixmap.save(output_path):
                    raise IOError(f"Could not write the image to {output_path}")
                return f"Captured {source_name} — Saved to {output_path}"

            QApplication.clipboard().setPixmap(pixmap)
            return f"Captured {source_name} — Copied to Clipboard"
        finally:
            QApplication.restoreOverrideCursor()

    def quick_capture(self):
        """Capture with the dialog's current settings without opening it (Ctrl+F1).

        Saving to disk always writes a new timestamped file into the output directory;
        the File Name field is only used by the Capture button.
        """
        output_path = None
        if self.disk_radio.isChecked():
            directory = self.output_dir_edit.text().strip() or str(get_screenshot_dir())
            output_path = get_unique_output_path(directory)

        return self.capture_to_destination(output_path)

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

        # Hide the dialog so it does not appear in its own capture
        self.hide()
        try:
            message = self.capture_to_destination(output_path)
            self.main_window.status_bar.showMessage(message, 5000)
            self.accept()

        except Exception as e:
            # Bring the dialog back so the user is not left with a vanished window
            self.show()
            QMessageBox.critical(self, "Error", f"An error occurred during capture: {e}")

    def closeEvent(self, event):
        """Handle the close event."""
        super().closeEvent(event)
