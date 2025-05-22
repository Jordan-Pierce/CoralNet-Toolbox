import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup,
                             QFormLayout, QComboBox)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """
    Perform See Anything (YOLOE) on multiple images using a reference image and label.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("eye.png"))
        self.setWindowTitle("Batch Inference")
        self.resize(600, 100)

        self.deploy_model_dialog = None
        self.loaded_model = None

        # Reference image and label
        self.source_image_path = None
        self.source_label = None
        # Target images
        self.target_images = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the source layout
        self.setup_source_layout()
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
        self.deploy_model_dialog = self.main_window.see_anything_deploy_predictor_dialog
        self.loaded_model = self.deploy_model_dialog.loaded_model

        # Update the source images (now assuming sources are valid)
        self.update_source_images()

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

    def setup_source_layout(self):
        """
        Set up the layout with source image and label selection.
        Contains dropdown combo boxes for selecting the source image and label.
        """
        group_box = QGroupBox("Source Selection")
        layout = QFormLayout()

        # Create the source image combo box
        self.source_image_combo_box = QComboBox()
        self.source_image_combo_box.currentIndexChanged.connect(self.update_source_labels)
        layout.addRow("Source Image:", self.source_image_combo_box)

        # Create the source label combo box
        self.source_label_combo_box = QComboBox()
        layout.addRow("Source Label:", self.source_label_combo_box)

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

    def has_valid_sources(self):
        """
        Check if there are any valid source images with polygon or rectangle annotations.

        :return: True if valid sources exist, False otherwise
        """
        # Check if there are any images
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.information(None,
                                    "No Images",
                                    "No images available for batch inference.")
            return False

        # Check for images with valid annotations
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    return True

        QMessageBox.information(None,
                                "No Valid Annotations",
                                "No images have polygon or rectangle annotations for batch inference.")
        return False

    def check_valid_sources(self):
        """
        Check if there are any valid source images with polygon or rectangle annotations.

        :return: True if valid sources exist, False otherwise
        """
        # Check if there are any images
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.information(self,
                                    "No Images",
                                    "No images available for batch inference.")
            return False

        # Check for images with valid annotations
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    return True

        QMessageBox.information(self,
                                "No Valid Annotations",
                                "No images have polygon or rectangle annotations for batch inference.")
        return False

    def update_source_images(self):
        """
        Updates the source image combo box with images that have at least one label
        with a valid polygon or rectangle annotation.

        :return: True if valid source images were found, False otherwise
        """
        self.source_image_combo_box.clear()
        valid_images_found = False

        # Get all image paths from the raster_manager
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            valid_annotation_found = False
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    valid_annotation_found = True
                    break

            if valid_annotation_found:
                # Get the basename (filename)
                basename = os.path.basename(image_path)
                # Add item to combo box with full path as data
                self.source_image_combo_box.addItem(basename, image_path)
                valid_images_found = True

        if not valid_images_found:
            QMessageBox.information(self,
                                    "No Source Images",
                                    "No images available for batch inference.")
            # Close the dialog since batch inference can't proceed
            QApplication.processEvents()  # Process pending events
            self.reject()
            return False

        # Update the combo box to have the selected image first
        if self.annotation_window.current_image_path in self.image_window.raster_manager.image_paths:
            self.source_image_combo_box.setCurrentText(os.path.basename(self.annotation_window.current_image_path))

        # Update the source labels given changes in the source images
        return self.update_source_labels()

    def update_source_labels(self):
        """
        Updates the source label combo box with labels that have at least one
        polygon or rectangle annotation from the current image.

        :return: True if valid source labels were found, False otherwise
        """
        self.source_label_combo_box.clear()

        source_image_path = self.source_image_combo_box.currentData()
        if not source_image_path:
            return False

        # Get annotations for this image
        annotations = self.annotation_window.get_image_annotations(source_image_path)

        # Create a dict of labels with valid annotations
        valid_labels = {}
        for annotation in annotations:
            if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                valid_labels[annotation.label.short_label_code] = annotation.label

        # Add valid labels to combo box
        for label_code, label_obj in valid_labels.items():
            self.source_label_combo_box.addItem(label_code, label_obj)

        if not valid_labels:
            QMessageBox.information(self,
                                    "No Valid Labels",
                                    "No labels with polygon or rectangle annotations available for batch inference.")
            # Close the dialog since batch inference can't proceed
            QApplication.processEvents()  # Process pending events
            self.reject()
            return False

        return True

    def get_source_annotations(self):
        """Return a list of polygon and rectangle annotations for the
        source image belonging to the selected label."""
        source_image_path = self.source_image_combo_box.currentData()
        source_label = self.source_label_combo_box.currentData()

        # Get annotations for this image
        annotations = self.annotation_window.get_image_annotations(source_image_path)

        # Filter annotations by label
        source_annotations = []
        for annotation in annotations:
            if annotation.label.short_label_code == source_label.short_label_code:
                source_annotations.append(annotation.cropped_bbox)

        return np.array(source_annotations)

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.
        Excludes the source image path if present.
    
        :return: List of selected image paths
        """
        # Get the source image path to exclude
        source_image_path = self.source_image_combo_box.currentData()
        
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []
    
        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            selected_paths = self.image_window.table_model.filtered_paths.copy()
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_paths = self.image_window.table_model.filtered_paths[:current_index + 1].copy()
            else:
                selected_paths = [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_paths = self.image_window.table_model.filtered_paths[current_index:].copy()
            else:
                selected_paths = [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            selected_paths = self.image_window.raster_manager.image_paths.copy()
        else:
            # Only apply to the current image
            selected_paths = [current_image_path]
    
        # Remove the source image path if it's in the selected paths
        if source_image_path and source_image_path in selected_paths:
            selected_paths.remove(source_image_path)
    
        return selected_paths

    def apply(self):
        """
        Apply the selected batch inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the source image path and label
            self.source_image_path = self.source_image_combo_box.currentData()
            self.source_label = self.source_label_combo_box.currentData()
            # Get the source annotations
            self.source_annotations = self.get_source_annotations()
            # Get the selected image paths
            self.target_images = self.get_selected_image_paths()
            # Perform batch inference
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
        progress_bar.start_progress(len(self.target_images))

        if self.loaded_model is not None:
            self.deploy_model_dialog.predict_from_annotations(refer_image=self.source_image_path,
                                                              refer_label=self.source_label,
                                                              refer_annotations=self.source_annotations,
                                                              target_images=self.target_images)
        progress_bar.stop_progress()
        progress_bar.close()
