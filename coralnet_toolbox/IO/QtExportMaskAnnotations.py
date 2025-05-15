import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import ujson as json

import cv2
import numpy as np
import rasterio
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QProgressDialog,
                             QGroupBox, QListWidget, QAbstractItemView, QListWidgetItem,
                             QButtonGroup, QScrollArea, QWidget)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportMaskAnnotations(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Export Segmentation Masks")
        self.resize(500, 250)

        self.selected_labels = []
        self.annotation_types = []
        self.class_mapping = {}

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the information layout
        self.setup_info_layout()
        # Setup the output directory and file format layout
        self.setup_output_layout()
        # Setup image selection layout
        self.setup_image_selection_layout()
        # Setup the annotation layout
        self.setup_annotation_layout()
        # Setup the label layout
        self.setup_label_layout()
        # Setup the mask format layout
        self.setup_mask_format_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def showEvent(self, event):
        """Handle the show event"""
        super().showEvent(event)
        # Update the labels in the label selection list
        self.update_label_selection_list()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Export Annotations to Segmentation Masks")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """Setup the output directory and file format layout."""
        groupbox = QGroupBox("Output Directory and File Format")
        layout = QFormLayout()

        # Output directory selection
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        # Output folder name
        self.output_name_edit = QLineEdit("")
        layout.addRow("Folder Name:", self.output_name_edit)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_image_selection_layout(self):
        """Setup the image selection layout."""
        group_box = QGroupBox("Apply To")
        layout = QVBoxLayout()

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        self.apply_group = QButtonGroup(self)
        self.apply_group.addButton(self.apply_filtered_checkbox)
        self.apply_group.addButton(self.apply_prev_checkbox)
        self.apply_group.addButton(self.apply_next_checkbox)
        self.apply_group.addButton(self.apply_all_checkbox)
        self.apply_group.setExclusive(True)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_annotation_layout(self):
        """Setup the annotation types layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()

        # Annotation types checkboxes
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)

        # Include negative samples
        self.include_negative_samples_checkbox = QCheckBox("Include negative samples")
        self.include_negative_samples_checkbox.setChecked(True)

        layout.addWidget(self.patch_checkbox)
        layout.addWidget(self.rectangle_checkbox)
        layout.addWidget(self.polygon_checkbox)
        layout.addWidget(self.include_negative_samples_checkbox)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_label_layout(self):
        """Setup the label selection layout."""
        groupbox = QGroupBox("Labels to Include")
        layout = QVBoxLayout()

        # Label selection
        self.label_selection_label = QLabel("Select Labels:")
        layout.addWidget(self.label_selection_label)

        # Create a scroll area for the labels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a widget to hold the checkboxes
        self.label_container = QWidget()
        self.label_container.setMinimumHeight(200)  # Set a minimum height for the container
        self.label_layout = QVBoxLayout(self.label_container)
        self.label_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)  # Respect widget sizes

        scroll_area.setWidget(self.label_container)
        layout.addWidget(scroll_area)

        # Store the checkbox references
        self.label_checkboxes = []

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_mask_format_layout(self):
        """Setup the mask format layout."""
        groupbox = QGroupBox("Mask Format")
        layout = QFormLayout()

        # File format combo box
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems([".png", ".bmp", ".tif"])
        self.file_format_combo.setEditable(True)
        layout.addRow("File Format:", self.file_format_combo)

        # Add option to preserve georeferencing
        self.preserve_georef_checkbox = QCheckBox("Preserve georeferencing (if available)")
        self.preserve_georef_checkbox.setChecked(True)
        layout.addRow("", self.preserve_georef_checkbox)

        # Connect the checkbox to update based on file format
        self.file_format_combo.currentTextChanged.connect(self.update_georef_availability)

        # Add a note about georeferencing
        self.georef_note = QLabel("Note: Georeferencing can only be preserved with TIF format")
        self.georef_note.setStyleSheet("color: #666; font-style: italic;")
        layout.addRow("", self.georef_note)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

        # Initial update based on default format
        self.update_georef_availability()

    def setup_buttons_layout(self):
        """Setup the buttons layout."""
        button_layout = QHBoxLayout()

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_masks)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)

    def browse_output_dir(self):
        """Open a file dialog to select the output directory."""
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "", options=options
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def update_label_selection_list(self):
        """Update the label selection list with labels from the label window."""
        # Clear existing checkboxes
        for checkbox in self.label_checkboxes:
            self.label_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.label_checkboxes = []

        # Create a checkbox for each label
        for label in self.label_window.labels:
            checkbox = QCheckBox(label.short_label_code)
            checkbox.setChecked(True)  # Default to checked
            checkbox.setProperty("label", label)  # Store the label object
            self.label_checkboxes.append(checkbox)
            self.label_layout.addWidget(checkbox)

    def update_georef_availability(self):
        """Update georeferencing checkbox availability based on file format"""
        current_format = self.file_format_combo.currentText().lower()
        is_tif = '.tif' in current_format

        self.preserve_georef_checkbox.setEnabled(is_tif)
        if not is_tif:
            self.preserve_georef_checkbox.setChecked(False)
            self.georef_note.setStyleSheet("color: red; font-style: italic;")
        else:
            self.georef_note.setStyleSheet("color: #666; font-style: italic;")

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

    def export_class_mapping(self, output_path):
        """Export the class mapping to a JSON file."""
        mapping_file = os.path.join(output_path, "class_mapping.json")

        with open(mapping_file, 'w') as f:
            json.dump(self.class_mapping, f, indent=4)

        if not os.path.exists(mapping_file):
            print(f"Warning: Failed to save class mapping to {mapping_file}")

    def export_masks(self):
        """Export segmentation masks based on selected annotations and labels."""
        # Validate inputs
        if not self.output_dir_edit.text():
            QMessageBox.warning(self,
                                "Missing Output Directory",
                                "Please select an output directory.")
            return

        # Check if at least one annotation type is selected
        if not any([self.patch_checkbox.isChecked(),
                    self.rectangle_checkbox.isChecked(),
                    self.polygon_checkbox.isChecked()]):
            QMessageBox.warning(self,
                                "No Annotation Type Selected",
                                "Please select at least one annotation type.")
            return

        # Check for checked items
        self.selected_labels = []
        for checkbox in self.label_checkboxes:
            if checkbox.isChecked():
                self.selected_labels.append(checkbox.property("label"))

        # Check if at least one label is selected
        if not self.selected_labels:
            QMessageBox.warning(self,
                                "No Labels Selected",
                                "Please select at least one label.")
            return

        output_dir = self.output_dir_edit.text()
        folder_name = self.output_name_edit.text().strip()
        file_format = self.file_format_combo.currentText()

        # Ensure file_format starts with a dot
        if not file_format.startswith('.'):
            file_format = '.' + file_format

        # Create output directory
        output_path = os.path.join(output_dir, folder_name)
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self,
                                 "Error Creating Directory",
                                 f"Failed to create output directory: {str(e)}")
            return

        # Get the list of images to process
        images = self.get_selected_image_paths()
        if not images:
            QMessageBox.warning(self,
                                "No Images",
                                "No images found in the project.")
            return

        # Collect annotation types to include
        self.annotation_types = []
        if self.patch_checkbox.isChecked():
            self.annotation_types.append(PatchAnnotation)
        if self.rectangle_checkbox.isChecked():
            self.annotation_types.append(RectangleAnnotation)
        if self.polygon_checkbox.isChecked():
            self.annotation_types.append(PolygonAnnotation)
            self.annotation_types.append(MultiPolygonAnnotation)

        # Create class mapping
        self.class_mapping = {}

        for i, label in enumerate(self.selected_labels):
            # Leave 0 for background
            self.class_mapping[label.short_label_code] = {
                "label": label.to_dict(),
                "index": i + 1
            }

        # Make the cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting Segmentation Masks")
        progress_bar.show()
        progress_bar.start_progress(len(images))

        try:
            for image_path in images:
                # Create mask for this image
                self.create_mask_for_image(image_path, output_path, file_format)
                progress_bar.update_progress()

            # Write the class mapping to a JSON file
            self.export_class_mapping(output_path)

            QMessageBox.information(self,
                                    "Export Complete",
                                    "Segmentation masks have been successfully exported")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error Exporting Masks",
                                 f"An error occurred: {str(e)}")

        finally:
            # Make cursor normal again
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def get_annotations_for_image(self, image_path):
        """Get annotations for a specific image."""
        # Get the selected labels' short label codes
        selected_labels = [label.short_label_code for label in self.selected_labels]

        # Get all annotations for this image
        annotations = []

        for annotation in self.annotation_window.get_image_annotations(image_path):
            # Check that the annotation is of the correct type
            if not isinstance(annotation, tuple(self.annotation_types)):
                continue

            # Check that the annotation's label is in the selected labels
            if annotation.label.short_label_code not in selected_labels:
                continue

            # Add the annotation to the list based on its type, if selected
            if self.patch_checkbox.isChecked() and isinstance(annotation, PatchAnnotation):
                annotations.append(annotation)

            elif self.rectangle_checkbox.isChecked() and isinstance(annotation, RectangleAnnotation):
                annotations.append(annotation)

            elif self.polygon_checkbox.isChecked() and isinstance(annotation, PolygonAnnotation):
                annotations.append(annotation)
                
            elif self.polygon_checkbox.isChecked() and isinstance(annotation, MultiPolygonAnnotation):
                for polygon in annotation.polygons:
                    annotations.append(polygon)

        return annotations

    def get_image_metadata(self, image_path, file_format):
        """Get the dimensions of the image, and check for georeferencing."""
        # Check if image has georeferencing that needs to be preserved
        transform = None
        crs = None
        has_georef = False
        height = None
        width = None

        # Get the raster from the raster manager
        raster = self.image_window.raster_manager.get_raster(image_path)

        # Only check for georeferencing if using TIF format and checkbox is checked
        can_preserve_georef = self.preserve_georef_checkbox.isChecked() and file_format.lower() == '.tif'

        if raster and raster.rasterio_src:
            # Get dimensions from the raster
            width = raster.width
            height = raster.height

            # Check for georeferencing if needed
            if can_preserve_georef and hasattr(raster.rasterio_src, 'transform'):
                transform = raster.rasterio_src.transform
                if transform and not transform.is_identity:
                    crs = raster.rasterio_src.crs
                    has_georef = True
        else:
            # Fallback to direct file access if raster is not available
            try:
                if can_preserve_georef:
                    with rasterio.open(image_path) as src:
                        if src.transform and not src.transform.is_identity:
                            transform = src.transform
                            crs = src.crs
                            has_georef = True
                        width, height = src.width, src.height
                else:
                    # Use PIL for non-georeferenced images
                    image = Image.open(image_path)
                    width, height = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")

        return height, width, has_georef, transform, crs

    def draw_annotations_on_mask(self, mask, annotations):
        """Draw annotations on the mask."""
        # Draw each annotation on the mask
        for annotation in annotations:
            # Get the label index for the annotation
            short_label_code = annotation.label.short_label_code
            label_index = self.class_mapping[short_label_code]["index"]

            # Draw the patch annotation
            if isinstance(annotation, PatchAnnotation):
                # Draw a filled rectangle
                cv2.rectangle(mask,
                              (int(annotation.center_xy.x() - annotation.annotation_size / 2),
                               int(annotation.center_xy.y() - annotation.annotation_size / 2)),
                              (int(annotation.center_xy.x() + annotation.annotation_size / 2),
                               int(annotation.center_xy.y() + annotation.annotation_size / 2)),
                              label_index, -1)  # -1 means filled

            # Draw the rectangle annotation
            elif isinstance(annotation, RectangleAnnotation):
                # Draw a filled rectangle
                cv2.rectangle(mask,
                              (int(annotation.top_left.x()), int(annotation.top_left.y())),
                              (int(annotation.bottom_right.x()), int(annotation.bottom_right.y())),
                              label_index, -1)  # -1 means filled

            # Draw the polygon annotation
            elif isinstance(annotation, PolygonAnnotation):
                # Draw a filled polygon
                points = np.array([[p.x(), p.y()] for p in annotation.points]).astype(np.int32)
                cv2.fillPoly(mask, [points], label_index)
                
            # Draw the multipolygon annotation
            elif isinstance(annotation, MultiPolygonAnnotation):
                for polygon in annotation.polygons:
                    points = np.array([[p.x(), p.y()] for p in polygon.points]).astype(np.int32)
                    cv2.fillPoly(mask, [points], label_index)

        return mask

    def create_mask_for_image(self, image_path, output_path, file_format):
        """Create a segmentation mask for a single image"""
        # Get the annotations for this image
        annotations = self.get_annotations_for_image(image_path)

        if not annotations and not self.include_negative_samples_checkbox.isChecked():
            return  # Skip images with no annotations if the user doesn't want negative samples

        # Get the image dimensions, georeferencing
        height, width, has_georef, transform, crs = self.get_image_metadata(image_path, file_format)

        if not height or not width:
            print(f"Could not get dimensions for image: {image_path}")
            return

        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw annotations on the mask
        mask = self.draw_annotations_on_mask(mask, annotations)

        # Save the mask
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        mask_filename = f"{name_without_ext}{file_format}"
        mask_path = os.path.join(output_path, mask_filename)

        # If we have georeferencing and we need to preserve it, use rasterio to save
        if has_georef and file_format.lower() == '.tif':
            with rasterio.open(
                mask_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=mask.dtype,
                crs=crs,
                transform=transform,
                compress='lzw',
            ) as dst:
                dst.write(mask, 1)
        else:
            # Use OpenCV as before for non-georeferenced images
            cv2.imwrite(mask_path, mask)

        if not os.path.exists(mask_path):
            print(f"Warning: Failed to save mask to {mask_path}")

    def closeEvent(self, event):
        """Handle the close event."""
        # Clean up any resources if needed
        super().closeEvent(event)
