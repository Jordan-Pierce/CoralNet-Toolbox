import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup)

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """
    Base class for performing batch inference on images instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("wizard.png"))
        self.setWindowTitle("Batch Inference")
        self.resize(400, 100)

        self.deploy_model_dialog = None
        self.loaded_model = None

        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the image options layout
        self.setup_options_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def showEvent(self, event):
        """
        Set up the layout when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.deploy_generator_dialog = self.main_window.sam_deploy_generator_dialog
        self.loaded_model = self.deploy_generator_dialog.loaded_model

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
        Set up the layout with image options.
        """
        # Create a group box for image options
        group_box = QGroupBox("Image Options")
        layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        image_options_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        # Add the checkboxes to the button group
        image_options_group.addButton(self.apply_filtered_checkbox)
        image_options_group.addButton(self.apply_prev_checkbox)
        image_options_group.addButton(self.apply_next_checkbox)
        image_options_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        image_options_group.setExclusive(True)

        # Set the default checkbox
        self.apply_all_checkbox.setChecked(True)

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box for the buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

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
        Apply the selected batch inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the selected image paths
            self.image_paths = self.get_selected_image_paths()
            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()

        self.accept()

    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        """
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_model is not None:
            self.deploy_generator_dialog.predict(image_paths=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()
