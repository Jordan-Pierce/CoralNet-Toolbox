import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout, QLabel,
                             QDialog, QDialogButtonBox, QGroupBox, QButtonGroup)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """Dialog for performing batch inference on images using AutoDistill."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.auto_distill_deploy_model_dialog
        self.loaded_models = self.deploy_model_dialog.loaded_model

        self.setWindowTitle("Batch Inference")
        self.resize(300, 200)  # Width, height

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the options layout
        self.setup_options_layout()
        # Setup buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Perform batch inferencing on the selected images.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """
        Set up the user interface.
        """
        group_box = QGroupBox("Image Options")
        layout = QVBoxLayout()

        # Create button group for image selection
        self.image_options_group = QButtonGroup(self)

        # Create image selection options
        self.apply_filtered = QCheckBox("▼ Apply to filtered images")
        self.apply_prev = QCheckBox("↑ Apply to previous images")
        self.apply_next = QCheckBox("↓ Apply to next images")
        self.apply_all = QCheckBox("↕ Apply to all images")
        # Add options to button group
        self.image_options_group.addButton(self.apply_filtered)
        self.image_options_group.addButton(self.apply_prev)
        self.image_options_group.addButton(self.apply_next)
        self.image_options_group.addButton(self.apply_all)
        # Make selections exclusive
        self.image_options_group.setExclusive(True)
        # Default selection
        self.apply_all.setChecked(True)

        # Add widgets to layout
        layout.addWidget(self.apply_filtered)
        layout.addWidget(self.apply_prev)
        layout.addWidget(self.apply_next)
        layout.addWidget(self.apply_all)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Setup the buttons layout.
        """
        okay_button = QDialogButtonBox.Ok
        cancel_button = QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(okay_button | cancel_button)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        # Add buttons to layout
        self.layout.addWidget(button_box)

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.

        :return: List of selected image paths
        """
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []

        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                return [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[current_index:]
            else:
                return [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            return self.image_window.raster_manager.image_paths
        else:
            # Only apply to the current image
            return [current_image_path]

    def apply(self):
        """
        Apply batch inference.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.deploy_model_dialog.predict(self.get_selected_image_paths())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            self.accept()
