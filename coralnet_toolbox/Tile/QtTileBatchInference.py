import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup, QComboBox)

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from PyQt5.QtWidgets import QFormLayout


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileBatchInference(QDialog):
    """
    Base class for performing batch inference on workareas within images in batch.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("wizard.png"))
        self.setWindowTitle("Tile Batch Inference")
        self.resize(400, 100)
        
        # Initialize dialogs for various tasks
        self.detect_dialog = main_window.detect_deploy_model_dialog
        self.segment_dialog = main_window.segment_deploy_model_dialog
        self.sam_dialog = main_window.sam_deploy_generator_dialog
        self.autodistill_dialog = main_window.auto_distill_deploy_model_dialog
        
        # Create a dictionary of the different model dialogs and their loaded models
        self.model_dialogs = {}

        self.loaded_model = None

        self.image_paths = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the models layout
        self.setup_models_layout()
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
        self.update_model_availability()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Perform batch inferencing on the selected images with work areas.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_models_layout(self):
        """
        Set up the layout with model selection options using a form layout.
        """
        # Create a group box for model selection
        self.models_group_box = QGroupBox("Select Model")
        form_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.update_loaded_model)
        form_layout.addRow("Model:", self.model_combo)

        self.models_group_box.setLayout(form_layout)
        self.layout.addWidget(self.models_group_box)

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
        
    def check_model_availability(self):
        """
        Check if a model is loaded and available for batch inference.

        :return: True if a model is loaded, False otherwise
        """
        # First update the model availability
        self.update_model_availability()
        
        if self.loaded_model is None:
            return False
        return True
        
    def update_model_availability(self):
        """Check each of the dialogs to see if they have their models loaded, and store the dialog if so."""
        self.model_dialogs = {}
        if self.detect_dialog and getattr(self.detect_dialog, "loaded_model", None):
            self.model_dialogs["Detect"] = self.detect_dialog
        if self.segment_dialog and getattr(self.segment_dialog, "loaded_model", None):
            self.model_dialogs["Segment"] = self.segment_dialog
        if self.sam_dialog and getattr(self.sam_dialog, "loaded_model", None):
            self.model_dialogs["SAM Generator"] = self.sam_dialog
        if self.autodistill_dialog and getattr(self.autodistill_dialog, "loaded_model", None):
            self.model_dialogs["Autodistill"] = self.autodistill_dialog
            
        # Update the model combo box with the available models
        self.update_model_combo()
        
    def update_model_combo(self):
        """
        Update the model combo box with only loaded models.
        """        
        self.model_combo.clear()
        self.model_keys = []
        
        for key, model in self.model_dialogs.items():
            if model is not None:
                self.model_combo.addItem(key)
                self.model_keys.append(key)
                
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
            
        # Update the loaded model based on the current selection
        self.update_loaded_model()
        
    def update_loaded_model(self):
        """
        Update the loaded_model attribute based on the selected model in the combo box.
        """
        idx = self.model_combo.currentIndex()
        if idx >= 0 and idx < len(self.model_keys):
            key = self.model_keys[idx]
            self.loaded_model = self.model_dialogs.get(key, None)
        else:
            self.loaded_model = None

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.

        :return: List of selected image paths
        """
        selected_images = []
        
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return selected_images

        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            selected_images = self.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_images = self.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                selected_images = [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_images = self.image_window.table_model.filtered_paths[current_index:]
            else:
                selected_images = [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            selected_images = self.image_window.raster_manager.image_paths
        else:
            # Only apply to the current image
            selected_images = [current_image_path]
            
        # Check if the select images have work areas
        selected_images_w_work_areas = []
        for image_path in selected_images:
            if self.image_window.raster_manager.get_raster(image_path).has_work_areas():
                # Only add images that have work areas
                selected_images_w_work_areas.append(image_path)
        
        return selected_images_w_work_areas

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
        progress_bar = ProgressBar(self, title="Tile Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_model is not None:
            self.loaded_model.predict(image_paths=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()
