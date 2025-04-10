import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageDraw

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QProgressDialog,
                             QGroupBox, QListWidget, QAbstractItemView, QListWidgetItem,
                             QButtonGroup, QScrollArea, QWidget)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportGeoJSONAnnotations(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Export Annotations as GeoJSON")
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
        info_label = QLabel("Export Annotations to GeoJSON")
        
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
        layout.addRow("File Name:", self.output_name_edit)    
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)    
        
    def setup_image_selection_layout(self):
        """Setup the image selection layout."""
        group_box = QGroupBox("Apply To")
        layout = QVBoxLayout()

        self.apply_filtered_checkbox = QCheckBox("ᗊ Apply to filtered images")
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
        """Setup the annotation types, and label selection layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()
        
        # Annotation types checkboxes
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)
        
        layout.addWidget(self.patch_checkbox)
        layout.addWidget(self.rectangle_checkbox)
        layout.addWidget(self.polygon_checkbox)
        
        # Label selection
        self.label_selection_label = QLabel("Select Labels:")
        layout.addWidget(self.label_selection_label)
        
        # Create a scroll area for the labels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create a widget to hold the checkboxes
        self.label_container = QWidget()
        self.label_layout = QVBoxLayout(self.label_container)
        
        scroll_area.setWidget(self.label_container)
        layout.addWidget(scroll_area)
        
        # Store the checkbox references
        self.label_checkboxes = []
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)
        
    def setup_buttons_layout(self):
        """Setup the buttons layout."""
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_geojson)
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
            
    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.
        
        :return: List of selected image paths
        """
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.filtered_image_paths
        elif self.apply_prev_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_next_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[current_image_index:]
        else:
            return self.image_window.image_paths 
        
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
                
        return annotations
    
    def convert_annotation_to_polygon(self, annotation):
        """Convert any annotation type to a polygon."""
        if isinstance(annotation, PatchAnnotation):
            size = annotation.annotation_size / 2
            x = annotation.center_xy.x()
            y = annotation.center_xy.y()
            return [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]
        elif isinstance(annotation, RectangleAnnotation):
            return [(annotation.top_left.x(), annotation.top_left.y()),
                    (annotation.bottom_right.x(), annotation.top_left.y()),
                    (annotation.bottom_right.x(), annotation.bottom_right.y()),
                    (annotation.top_left.x(), annotation.bottom_right.y())]
        elif isinstance(annotation, PolygonAnnotation):
            return [(p.x(), p.y()) for p in annotation.points]
        return []

    def create_geojson_feature(self, annotation, image_width, image_height):
        """Create a GeoJSON feature from an annotation."""
        polygon_coords = self.convert_annotation_to_polygon(annotation)
        
        # Normalize the coordinates
        normalized_coords = []
        for x, y in polygon_coords:
            normalized_coords.append([x / image_width, y / image_height])
        
        # Ensure the polygon has at least four positions
        if len(normalized_coords) < 4:
            # If less than 4 points, skip this annotation
            print(f"Skipping annotation with less than 4 points.")
            return None
        
        # Ensure the polygon is closed (first and last positions are the same)
        if normalized_coords[0] != normalized_coords[-1]:
            normalized_coords.append(normalized_coords[0])
        
        # Create the GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [normalized_coords]
            },
            "properties": {
                "label": annotation.label.short_label_code,
                "label_properties": annotation.label.to_dict()
            }
        }
        return feature

    def export_geojson(self):
        """Export annotations as GeoJSON."""
        # Validate inputs
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, 
                                "Missing Output Directory", 
                                "Please select an output directory.")
            return

        if not self.output_name_edit.text():
            QMessageBox.warning(self, 
                                "Missing File Name", 
                                "Please enter a file name.")
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
        file_name = self.output_name_edit.text().strip()
        
        # Ensure file_name ends with .geojson
        if not file_name.endswith('.geojson'):
            file_name += '.geojson'

        # Create output directory
        output_path = os.path.join(output_dir, file_name)

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
            
        # Create the GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # CRS information
        crs = None

        # Make the cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting GeoJSON")
        progress_bar.show()
        progress_bar.start_progress(len(images))

        try:
            for image_path in images:
                # Check if the image is a .tif or .tiff
                if not image_path.lower().endswith(('.tif', '.tiff')):
                    print(f"Skipping non-TIFF image: {image_path}")
                    continue
                
                # Get the annotations for this image
                annotations = self.get_annotations_for_image(image_path)
                
                # Get the image dimensions
                try:
                    image = Image.open(image_path)
                    image_width, image_height = image.size
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
                
                # Create GeoJSON features for each annotation
                for annotation in annotations:
                    feature = self.create_geojson_feature(annotation, image_width, image_height)
                    if feature is not None:
                        geojson_data["features"].append(feature)
                
                # Attempt to read CRS from the first valid TIFF image
                if crs is None:
                    try:
                        with rasterio.open(image_path) as src:
                            if src.crs:
                                crs = src.crs.to_string()
                    except Exception as e:
                        print(f"Error reading CRS from {image_path}: {e}")

                progress_bar.update_progress()

            # Add CRS to GeoJSON data if available
            if crs:
                geojson_data["crs"] = {
                    "type": "name",
                    "properties": {
                        "name": crs
                    }
                }

            # Write the GeoJSON data to a file
            with open(output_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)

            QMessageBox.information(self, 
                                    "Export Complete", 
                                    "Annotations have been successfully exported as GeoJSON")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, 
                                 "Error Exporting GeoJSON", 
                                 f"An error occurred: {str(e)}")
            
        finally:
            # Make cursor normal again
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
        
    def closeEvent(self, event):
        """Handle the close event."""
        # Clean up any resources if needed
        super().closeEvent(event)
