import warnings
import os
import ujson as json
from rasterio.transform import Affine
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
    QApplication, QMessageBox, QLabel, QScrollArea, QWidget,
    QButtonGroup
)
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        # Setup the label selection layout
        # Setup label selection
        self.setup_label_layout()
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
        groupbox = QGroupBox("Output File")
        layout = QFormLayout()

        # Output file selection
        output_file_layout = QHBoxLayout()
        self.output_file_edit = QLineEdit()
        self.output_file_button = QPushButton("Browse...")
        self.output_file_button.clicked.connect(self.browse_output_file)
        output_file_layout.addWidget(self.output_file_edit)
        output_file_layout.addWidget(self.output_file_button)
        layout.addRow("Output File:", output_file_layout)

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
        """Setup the annotation types, and label selection layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()
    
        # Patch Annotation type checkboxes
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        layout.addWidget(self.patch_checkbox)
    
        # Setup patch representation selection
        self.setup_patch_representation_layout(layout)
    
        # Annotation types checkboxes
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        layout.addWidget(self.rectangle_checkbox)
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)
        layout.addWidget(self.polygon_checkbox)
        self.multipolygon_checkbox = QCheckBox("MultiPolygon Annotations")
        self.multipolygon_checkbox.setChecked(True)
        layout.addWidget(self.multipolygon_checkbox)
    
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_patch_representation_layout(self, layout):
        """Setup the patch representation selection layout."""
        # Patch representation selection
        self.patch_representation_label = QLabel("Patch Representation:")
        layout.addWidget(self.patch_representation_label)

        self.patch_representation_combo = QComboBox()
        self.patch_representation_combo.addItem("Polygon")
        self.patch_representation_combo.addItem("Point")
        layout.addWidget(self.patch_representation_combo)

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

    def browse_output_file(self):
        """Open a file dialog to select the output file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "GeoJSON Files (*.geojson)", options=options
        )
        if file_path:
            self.output_file_edit.setText(file_path)

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

    def get_annotations_for_image(self, image_path):
        """Get annotations for a specific image."""
        # Get the selected labels' short label codes
        selected_labels = [label.short_label_code for label in self.selected_labels]
    
        # Get all annotations for this image
        annotations = []
    
        for annotation in self.annotation_window.get_image_annotations(image_path):
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
                
            elif self.multipolygon_checkbox.isChecked() and isinstance(annotation, MultiPolygonAnnotation):
                annotations.append(annotation)
    
        return annotations

    def convert_annotation_to_polygon(self, annotation):
        """Convert any annotation type to a polygon."""

        # Convert Patch Annotation to a polygon
        if isinstance(annotation, PatchAnnotation):
            size = annotation.annotation_size / 2
            x = annotation.center_xy.x()
            y = annotation.center_xy.y()
            return [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]

        # Convert Rectangle Annotation to a polygon
        elif isinstance(annotation, RectangleAnnotation):
            return [(annotation.top_left.x(), annotation.top_left.y()),
                    (annotation.bottom_right.x(), annotation.top_left.y()),
                    (annotation.bottom_right.x(), annotation.bottom_right.y()),
                    (annotation.top_left.x(), annotation.bottom_right.y())]

        # Convert Polygon Annotation to a polygon
        elif isinstance(annotation, PolygonAnnotation):
            return [(p.x(), p.y()) for p in annotation.points]

        return []

    def transform_coordinates(self, coords, transform):
        """
        Transform coordinates from pixel space to geographic space with proper validation.

        Args:
            coords: List of (x, y) coordinate tuples in pixel space
            transform: Affine transformation matrix from rasterio

        Returns:
            List of [x, y] coordinate pairs in geographic space

        Raises:
            ValueError: If transform is invalid or coordinates cannot be transformed
        """
        # Validate transform
        if transform is None:
            raise ValueError("No coordinate transformation available")

        if not isinstance(transform, Affine):
            raise ValueError(f"Invalid transform type: {type(transform)}, expected Affine")

        # Check if transform is valid (not identity or close to identity)
        identity = Affine.identity()
        is_close_to_identity = all(abs(a - b) < 1e-10 for a, b in zip(transform, identity))

        if is_close_to_identity:
            raise ValueError("Transform appears to be identity matrix - no geographic projection available")

        transformed_coords = []
        for x, y in coords:
            try:
                # Validate input coordinates
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    raise ValueError(f"Invalid coordinate values: ({x}, {y})")

                # Apply transformation
                geo_x, geo_y = transform * (x, y)

                # Check for unreasonable values that might indicate transformation problems
                # These thresholds depend on your coordinate system, adjust as needed
                if abs(geo_x) > 1e10 or abs(geo_y) > 1e10:
                    raise ValueError(f"Transformed coordinates out of reasonable range: [{geo_x}, {geo_y}]")

                transformed_coords.append([geo_x, geo_y])
            except Exception as e:
                raise ValueError(f"Failed to transform coordinates ({x}, {y}): {str(e)}")

        return transformed_coords

    def create_geojson_feature(self, annotation, image_path, transform=None):
        """
        Create a GeoJSON feature from an annotation with proper coordinate transformation error handling.

        Args:
            annotation: The annotation object to convert
            image_path: Path to the image containing the annotation
            transform: Optional affine transformation matrix

        Returns:
            GeoJSON feature dictionary or None if conversion fails
        """
        # Handle Patch Annotations as Points
        if isinstance(annotation, PatchAnnotation) and self.patch_representation_combo.currentText() == "Point":
            x = annotation.center_xy.x()
            y = annotation.center_xy.y()

            try:
                # Transform the coordinates
                [[geo_x, geo_y]] = self.transform_coordinates([(x, y)], transform)
            except ValueError:
                # Skip this annotation
                return None

            # Create the GeoJSON feature for Point
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [geo_x, geo_y]
                },
                "properties": {
                    "label": annotation.label.short_label_code,
                    "label_properties": annotation.label.to_dict(),
                    "source_image": os.path.basename(image_path),
                    "coordinate_system": "geographic"
                }
            }
            return feature

        # Handle MultiPolygon Annotations
        elif isinstance(annotation, MultiPolygonAnnotation):
            multipolygon_coords = []

            for polygon in annotation.polygons:
                polygon_coords = self.convert_annotation_to_polygon(polygon)

                # Ensure the polygon has at least 4 points
                if len(polygon_coords) < 4:
                    print(f"Skipping polygon with less than 4 points in {image_path}")
                    continue

                try:
                    # Transform the coordinates
                    polygon_coords = self.transform_coordinates(polygon_coords, transform)
                except ValueError:
                    # Skip this polygon
                    continue

                # Ensure the polygon is closed (first and last positions are the same)
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])

                multipolygon_coords.append([polygon_coords])

            if not multipolygon_coords:
                return None

            # Create the GeoJSON feature for MultiPolygon
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": multipolygon_coords
                },
                "properties": {
                    "label": annotation.label.short_label_code,
                    "label_properties": annotation.label.to_dict(),
                    "source_image": os.path.basename(image_path),
                    "coordinate_system": "geographic"
                }
            }
            return feature
        
        else:
            # Handle all other annotations as Polygons
            polygon_coords = self.convert_annotation_to_polygon(annotation)

            # Ensure the polygon has at least 4 points
            if len(polygon_coords) < 4:
                print(f"Skipping annotation with less than 4 points in {image_path}")
                return None

            try:
                # Transform the coordinates
                polygon_coords = self.transform_coordinates(polygon_coords, transform)
            except ValueError:
                # Skip this annotation
                return None

            # Ensure the polygon is closed (first and last positions are the same)
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])

            # Create the GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                },
                "properties": {
                    "label": annotation.label.short_label_code,
                    "label_properties": annotation.label.to_dict(),
                    "source_image": os.path.basename(image_path),
                    "coordinate_system": "geographic"
                }
            }
            return feature

    def validate_images(self, images):
        """
        Validate that all selected images are TIFF files and have valid CRS and transform information.

        Args:
            images: List of image paths to validate

        Returns:
            True if all images are valid, False otherwise. Also displays warning messages.
        """
        for image_path in images:
            # Check if the image is a .tif or .tiff
            if not image_path.lower().endswith(('.tif', '.tiff')):
                QMessageBox.warning(self,
                                    "Invalid Image Format",
                                    "Non-TIF images included. Select only TIF images.")
                return False

            # Get the raster from the raster manager
            raster = self.image_window.raster_manager.get_raster(image_path)
            if not raster or not raster.rasterio_src:
                QMessageBox.warning(self,
                                    "Invalid Image",
                                    f"Could not open {os.path.basename(image_path)}.")
                return False

            try:
                # Get the image transform from the rasterio source
                transform = raster.rasterio_src.transform
                crs = raster.rasterio_src.crs

                # Check if CRS exists
                if not crs:
                    QMessageBox.warning(self,
                                        "Missing CRS Information",
                                        f"No coordinate reference system found for {os.path.basename(image_path)}.")
                    return False

                # Check the transform
                if not isinstance(transform, Affine):
                    QMessageBox.warning(self,
                                        "Invalid Transform",
                                        f"Invalid transform for {os.path.basename(image_path)}.")
                    return False

            except Exception as e:
                QMessageBox.warning(self,
                                    "Missing CRS Information",
                                    f"Could not get CRS information for {os.path.basename(image_path)}: {str(e)}")
                return False

        return True

    def export_geojson(self):
        """Export annotations as GeoJSON."""
        # Validate inputs
        if not self.output_file_edit.text():
            QMessageBox.warning(self,
                                "Missing Output File",
                                "Please select an output file.")
            return

        # Check if at least one annotation type is selected
        if not any([self.patch_checkbox.isChecked(),
                    self.rectangle_checkbox.isChecked(),
                    self.polygon_checkbox.isChecked(),
                    self.multipolygon_checkbox.isChecked()]):
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

        output_path = self.output_file_edit.text()

        # Get the list of images to process
        images = self.get_selected_image_paths()
        if not images:
            QMessageBox.warning(self,
                                "No Images",
                                "No images found in the project.")
            return

        # Validate images before proceeding
        if not self.validate_images(images):
            return

        # Collect annotation types to include
        self.annotation_types = []
        if self.patch_checkbox.isChecked():
            self.annotation_types.append(PatchAnnotation)
        if self.rectangle_checkbox.isChecked():
            self.annotation_types.append(RectangleAnnotation)
        if self.polygon_checkbox.isChecked():
            self.annotation_types.append(PolygonAnnotation)
        if self.multipolygon_checkbox.isChecked():
            self.annotation_types.append(MultiPolygonAnnotation)
            
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
            # Iterate through the images
            for image_path in images:
                # Check if the image is a .tif or .tiff
                if not image_path.lower().endswith(('.tif', '.tiff')):
                    print(f"Warning: Non-TIFF image {os.path.basename(image_path)} included; skipping.")
                    continue

                # Get the annotations for this image
                annotations = self.get_annotations_for_image(image_path)

                # Get the raster from the raster manager
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster or not raster.rasterio_src:
                    print(f"Error: Could not get raster for {os.path.basename(image_path)}; skipping.")
                    continue

                try:
                    # Get the image transform from the rasterio source
                    transform = raster.rasterio_src.transform
                    crs = raster.rasterio_src.crs.to_string()

                    # Check the transform
                    if not isinstance(transform, Affine):
                        print(f"Error: Invalid transform for {os.path.basename(image_path)}; skipping.")
                        continue

                except Exception as e:
                    print(f"Error: Could not get CRS for {os.path.basename(image_path)}; skipping: {str(e)}")
                    continue

                # Create GeoJSON features for each annotation
                for annotation in annotations:
                    feature = self.create_geojson_feature(annotation, image_path, transform)
                    if feature is not None:
                        geojson_data["features"].append(feature)

                progress_bar.update_progress()

            # Add CRS to GeoJSON data if available
            if crs:
                geojson_data["crs"] = {
                    "type": "name",
                    "properties": {
                        "name": crs
                    }
                }

            else:
                QMessageBox.critical(self,
                                     "Missing CRS",
                                     "No CRS information available for the images.")
                return

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
