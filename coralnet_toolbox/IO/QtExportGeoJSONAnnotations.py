import warnings

import os
import ujson as json
from datetime import datetime

from rasterio.warp import transform as warp_transform
from rasterio.crs import CRS

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
    QApplication, QMessageBox, QLabel, QWidget, QGridLayout,
    QListWidget, QListWidgetItem, QTabWidget
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

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Export Annotations to GeoJSON")
        self.resize(400, 500)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # 1. Info Section
        self.setup_info_layout()
        
        # 2. Export Mode (Single vs Individual) & Output Path
        self.setup_output_configuration_layout()
        
        # 3. Annotations Configuration
        self.setup_annotation_layout()
        
        # 4. Advanced Options (New: WGS84, Styles, Metadata)
        self.setup_advanced_options_layout()

        # 5. Label Selection
        self.setup_label_layout()
        
        # 6. Action Buttons
        self.setup_buttons_layout()

        # Initialize UI state
        # self.update_output_mode_ui()  # Not needed for tabs

    def showEvent(self, event):
        """Handle the show event to refresh label list."""
        super().showEvent(event)
        self.update_label_selection_list()

    # ----------------------------------------------------------------------
    # UI Setup Methods
    # ----------------------------------------------------------------------

    def setup_info_layout(self):
        """Simple information header."""
        info_label = QLabel(
            "<b>Export Annotations to GeoJSON</b><br>"
            "Export annotations for all images. Choose between a single merged file or individual files."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("margin-bottom: 5px;")
        self.layout.addWidget(info_label)

    def setup_output_configuration_layout(self):
        """Setup the export mode and dynamic output file/folder selection."""
        groupbox = QGroupBox("Export Configuration")
        layout = QVBoxLayout()

        # Tab Widget for modes
        self.tab_widget = QTabWidget()

        # Single File Tab
        single_tab = QWidget()
        single_layout = QFormLayout(single_tab)
        self.output_file_edit = QLineEdit()
        self.browse_output_file_button = QPushButton("Browse...")
        self.browse_output_file_button.clicked.connect(self.browse_output_file)
        file_field_layout = QHBoxLayout()
        file_field_layout.addWidget(self.output_file_edit)
        file_field_layout.addWidget(self.browse_output_file_button)
        single_layout.addRow("Output File:", file_field_layout)
        self.tab_widget.addTab(single_tab, "Single File")

        # Multiple Files Tab
        multi_tab = QWidget()
        multi_layout = QFormLayout(multi_tab)
        self.output_dir_edit = QLineEdit()
        self.browse_output_dir_button = QPushButton("Browse...")
        self.browse_output_dir_button.clicked.connect(self.browse_output_dir)
        dir_field_layout = QHBoxLayout()
        dir_field_layout.addWidget(self.output_dir_edit)
        dir_field_layout.addWidget(self.browse_output_dir_button)
        multi_layout.addRow("Output Directory:", dir_field_layout)
        self.tab_widget.addTab(multi_tab, "Multiple Files")

        layout.addWidget(self.tab_widget)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_annotation_layout(self):
        """Setup annotation types in a grid layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QGridLayout()
        layout.setColumnStretch(1, 1) 
        layout.setColumnStretch(2, 1)

        # Patch Annotations (Row 0)
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        layout.addWidget(self.patch_checkbox, 0, 0)

        # Patch Representation Combo (Row 0, Col 1)
        patch_rep_layout = QHBoxLayout()
        patch_rep_layout.addWidget(QLabel("Representation:"))
        self.patch_representation_combo = QComboBox()
        self.patch_representation_combo.addItems(["Polygon", "Point"])
        patch_rep_layout.addWidget(self.patch_representation_combo)
        patch_rep_layout.addStretch()
        layout.addLayout(patch_rep_layout, 0, 1)

        # Geometry Types (Row 1)
        self.rectangle_checkbox = QCheckBox("Rectangle")
        self.rectangle_checkbox.setChecked(True)
        self.polygon_checkbox = QCheckBox("Polygon")
        self.polygon_checkbox.setChecked(True)
        self.multipolygon_checkbox = QCheckBox("MultiPolygon")
        self.multipolygon_checkbox.setChecked(True)

        layout.addWidget(self.rectangle_checkbox, 1, 0)
        layout.addWidget(self.polygon_checkbox, 1, 1)
        layout.addWidget(self.multipolygon_checkbox, 1, 2)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_advanced_options_layout(self):
        """Setup advanced processing options."""
        groupbox = QGroupBox("Advanced Options")
        layout = QVBoxLayout()

        # WGS84 Reprojection
        self.wgs84_checkbox = QCheckBox("Reproject to WGS84 (Lat/Lon)")
        self.wgs84_checkbox.setChecked(True)
        self.wgs84_checkbox.setToolTip(
            "If checked, coordinates will be transformed to EPSG:4326 (Latitude/Longitude).\n"
            "This is required for compatibility with web maps (Leaflet, Mapbox, etc.)."
        )
        layout.addWidget(self.wgs84_checkbox)

        # Styling
        self.style_checkbox = QCheckBox("Include Styling Properties (Color)")
        self.style_checkbox.setChecked(True)
        self.style_checkbox.setToolTip(
            "If checked, 'marker-color', 'stroke', and 'fill' properties will be added \n"
            "based on the Label's color assignment."
        )
        layout.addWidget(self.style_checkbox)

        # Metadata
        self.metadata_checkbox = QCheckBox("Include Extra Metadata (Area, Date)")
        self.metadata_checkbox.setChecked(True)
        self.metadata_checkbox.setToolTip(
            "If checked, calculates pixel area and adds timestamps/annotator info."
        )
        layout.addWidget(self.metadata_checkbox)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_label_layout(self):
        """Setup label selection using a QListWidget."""
        groupbox = QGroupBox("Filter by Label")
        layout = QVBoxLayout()

        # Tools layout (Select All/None)
        tools_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Select None")
        btn_all.clicked.connect(lambda: self.toggle_labels(True))
        btn_none.clicked.connect(lambda: self.toggle_labels(False))
        tools_layout.addWidget(btn_all)
        tools_layout.addWidget(btn_none)
        tools_layout.addStretch()
        layout.addLayout(tools_layout)

        # List Widget
        self.label_list_widget = QListWidget()
        layout.addWidget(self.label_list_widget)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_buttons_layout(self):
        """Standard Export/Cancel buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch() 

        self.export_button = QPushButton("Export GeoJSON")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_geojson)
        self.export_button.setMinimumWidth(120)
        self.export_button.setMinimumHeight(30)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.export_button)

        self.layout.addLayout(button_layout)

    # ----------------------------------------------------------------------
    # UI Interaction Methods
    # ----------------------------------------------------------------------

    def update_output_mode_ui(self):
        """Not used with tabs."""
        pass

    def browse_output_dir(self):
        """Directory chooser for output."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def browse_output_file(self):
        """File chooser for output."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "GeoJSON Files (*.geojson);;All Files (*)"
        )
        if file_path:
            self.output_file_edit.setText(file_path)

    def update_label_selection_list(self):
        """Populate the QListWidget with labels."""
        self.label_list_widget.clear()
        for label in self.label_window.labels:
            item = QListWidgetItem(f"{label.short_label_code} - {label.long_label_code}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, label) 
            self.label_list_widget.addItem(item)

    def toggle_labels(self, select_all):
        """Helper to check/uncheck all items."""
        for i in range(self.label_list_widget.count()):
            item = self.label_list_widget.item(i)
            item.setCheckState(Qt.Checked if select_all else Qt.Unchecked)

    # ----------------------------------------------------------------------
    # Logic Helper Methods
    # ----------------------------------------------------------------------

    def get_selected_labels(self):
        """Retrieve list of label objects from checked list items."""
        selected = []
        for i in range(self.label_list_widget.count()):
            item = self.label_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def get_label_color_hex(self, label):
        """Safely extract hex color string from label object."""
        try:
            if hasattr(label, 'color'):
                c = label.color
                if isinstance(c, QColor):
                    return c.name()
                elif isinstance(c, str):
                    return c
        except Exception:
            pass
        return "#555555"  # Fallback gray

    def get_polygon_area_pixels(self, coords):
        """Calculate polygon area using shoelace formula."""
        n = len(coords)
        if n < 3: 
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0

    def get_annotations_for_image(self, image_path, selected_label_codes):
        """Get filtered annotations for a specific image."""
        annotations = []
        raw_annotations = self.annotation_window.get_image_annotations(image_path)
        
        if not raw_annotations:
            return []

        for annotation in raw_annotations:
            if annotation.label.short_label_code not in selected_label_codes:
                continue

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
        """Convert any annotation type to a pixel-coordinate polygon."""
        if isinstance(annotation, PatchAnnotation):
            size = annotation.annotation_size / 2
            x, y = annotation.center_xy.x(), annotation.center_xy.y()
            return [(x - size, y - size), (x + size, y - size), 
                    (x + size, y + size), (x - size, y + size)]

        elif isinstance(annotation, RectangleAnnotation):
            tl, br = annotation.top_left, annotation.bottom_right
            return [(tl.x(), tl.y()), (br.x(), tl.y()), 
                    (br.x(), br.y()), (tl.x(), br.y())]

        elif isinstance(annotation, PolygonAnnotation):
            return [(p.x(), p.y()) for p in annotation.points]

        return []

    def transform_coordinates(self, coords, src_transform, src_crs):
        """
        Transform pixel coordinates to Geographic coordinates.
        1. Apply Affine transform (Pixels -> Image CRS)
        2. If WGS84 requested, Apply Warp (Image CRS -> EPSG:4326)
        """
        # 1. Pixels to Source CRS
        projected_coords = []
        for x, y in coords:
            wx, wy = src_transform * (x, y)
            projected_coords.append((wx, wy))

        # 2. Source CRS to WGS84 (if requested)
        if self.wgs84_checkbox.isChecked():
            # Check if source CRS is valid
            if not src_crs:
                raise ValueError("Cannot reproject: Source image has no CRS.")
            
            # Unzip for batch processing
            xs, ys = zip(*projected_coords)
            
            try:
                # Perform the warp
                wgs84_crs = CRS.from_epsg(4326)
                txs, tys = warp_transform(src_crs, wgs84_crs, xs, ys)
                return list(zip(txs, tys))
            except Exception as e:
                raise ValueError(f"Reprojection failed: {str(e)}")
        
        return projected_coords

    def create_geojson_feature(self, annotation, image_path, transform, crs):
        """Create a single GeoJSON Feature dictionary with styles and metadata."""
        
        # Determine geometry type
        is_point = isinstance(annotation, PatchAnnotation) and self.patch_representation_combo.currentText() == "Point"
        
        # Prepare coordinates (Pixels)
        pixel_coords_lists = []  # List of lists of coords (to handle Multipolygon)
        
        if isinstance(annotation, MultiPolygonAnnotation):
            for poly in annotation.polygons:
                pts = self.convert_annotation_to_polygon(poly)
                if len(pts) >= 3: 
                    pixel_coords_lists.append(pts)
        elif is_point:
            # For points, we just treat center as single coord
            pixel_coords_lists.append([(annotation.center_xy.x(), annotation.center_xy.y())])
        else:
            # Polygon/Rect/Patch-as-Poly
            pts = self.convert_annotation_to_polygon(annotation)
            if len(pts) >= 3: 
                pixel_coords_lists.append(pts)

        if not pixel_coords_lists:
            return None

        # Transform to Geographic Coordinates
        try:
            geo_coords_lists = []
            for plist in pixel_coords_lists:
                geo_pts = self.transform_coordinates(plist, transform, crs)
                
                if not is_point:
                    # Close the loop for polygons
                    if geo_pts[0] != geo_pts[-1]: 
                        geo_pts.append(geo_pts[0])
                
                geo_coords_lists.append(geo_pts)
        except ValueError:
            return None

        # Build Geometry Object
        if is_point:
            # Point
            geometry = {
                "type": "Point",
                "coordinates": geo_coords_lists[0][0]  # First list, first point (x,y)
            }
        elif isinstance(annotation, MultiPolygonAnnotation):
            # MultiPolygon
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [[g] for g in geo_coords_lists]  # GeoJSON MultiPoly needs extra nesting
            }
        else:
            # Polygon
            geometry = {
                "type": "Polygon",
                "coordinates": [geo_coords_lists[0]]
            }

        # Build Properties
        props = {
            "short_label_code": annotation.label.short_label_code,
            "long_label_code": annotation.label.long_label_code,
            "source_image": os.path.basename(image_path)
        }

        # Add Metadata (Area, Date)
        if self.metadata_checkbox.isChecked():
            # Calculate pixel area (sum of all parts)
            total_px_area = sum(self.get_polygon_area_pixels(plist) for plist in pixel_coords_lists)
            props["area_pixels"] = round(total_px_area, 2)
            props["exported_at"] = datetime.now().isoformat()
            # If annotator info exists in your annotation object, add it here:
            # props["annotator"] = getattr(annotation, "annotator", "Unknown")

        # Add Styling (SimpleStyle)
        if self.style_checkbox.isChecked():
            color_hex = self.get_label_color_hex(annotation.label)
            props["marker-color"] = color_hex
            props["stroke"] = color_hex
            props["fill"] = color_hex
            props["fill-opacity"] = 0.5
            props["stroke-width"] = 2

        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": props
        }

    # ----------------------------------------------------------------------
    # Main Export Execution
    # ----------------------------------------------------------------------

    def validate_inputs(self):
        """Check if output paths and selections are valid."""
        mode = "single" if self.tab_widget.currentIndex() == 0 else "individual"
        
        # Output Path Validation
        if mode == "single":
            if not self.output_file_edit.text().strip():
                QMessageBox.warning(self, "Output Error", "Please select an output file.")
                return False
        else:
            if not self.output_dir_edit.text().strip():
                QMessageBox.warning(self, "Output Error", "Please select an output directory.")
                return False

        # Annotation/Label Validation
        if not any([self.patch_checkbox.isChecked(), self.rectangle_checkbox.isChecked(),
                    self.polygon_checkbox.isChecked(), self.multipolygon_checkbox.isChecked()]):
            QMessageBox.warning(self, "Selection Error", "Select at least one annotation type.")
            return False

        if not self.get_selected_labels():
            QMessageBox.warning(self, "Selection Error", "Select at least one label.")
            return False

        return True

    def export_geojson(self):
        """Main export execution method."""
        if not self.validate_inputs():
            return

        # Prepare configuration
        mode = "single" if self.tab_widget.currentIndex() == 0 else "individual"
        selected_label_objects = self.get_selected_labels()
        selected_label_codes = [label.short_label_code for label in selected_label_objects]
        
        # Determine output paths
        if mode == 'single':
            final_output_path = self.output_file_edit.text().strip()
        else:
            final_output_dir = self.output_dir_edit.text().strip()
            if not os.path.exists(final_output_dir):
                try:
                    os.makedirs(final_output_dir)
                except OSError as e:
                    QMessageBox.critical(self, "Error", f"Could not create directory: {e}")
                    return

        # Get all images in project
        all_images = self.image_window.raster_manager.image_paths
        
        # Start Progress
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting GeoJSON")
        progress_bar.show()
        progress_bar.start_progress(len(all_images))

        # Data holder for Single Mode
        combined_features = []
        
        # Determine final CRS name for GeoJSON header
        final_crs_name = "urn:ogc:def:crs:OGC:1.3:CRS84" if self.wgs84_checkbox.isChecked() else None

        try:
            for image_path in all_images:
                # 1. Check for annotations first
                annotations = self.get_annotations_for_image(image_path, selected_label_codes)
                if not annotations:
                    progress_bar.update_progress()
                    continue

                # 2. Load Raster Data
                if not image_path.lower().endswith(('.tif', '.tiff')):
                    continue
                    
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster or not raster.rasterio_src:
                    continue

                try:
                    transform = raster.rasterio_src.transform
                    src_crs = raster.rasterio_src.crs
                    
                    # Store original CRS if we aren't reprojecting and haven't set one yet
                    if not self.wgs84_checkbox.isChecked() and final_crs_name is None:
                        if src_crs: 
                            final_crs_name = src_crs.to_string()

                except Exception:
                    print(f"Skipping {os.path.basename(image_path)}: Invalid CRS/Transform")
                    continue

                # 3. Create Features
                image_features = []
                for ann in annotations:
                    feat = self.create_geojson_feature(ann, image_path, transform, src_crs)
                    if feat: 
                        image_features.append(feat)

                if not image_features:
                    progress_bar.update_progress()
                    continue

                # 4. Write Data (Depending on Mode)
                if mode == 'single':
                    combined_features.extend(image_features)
                else:
                    # Individual Mode: Write immediately
                    filename = os.path.splitext(os.path.basename(image_path))[0] + ".geojson"
                    out_file = os.path.join(final_output_dir, filename)
                    
                    feature_collection = {
                        "type": "FeatureCollection",
                        "features": image_features
                    }
                    
                    # Add CRS block if not WGS84
                    if final_crs_name and "CRS84" not in final_crs_name:
                        feature_collection["crs"] = {
                            "type": "name", "properties": {"name": final_crs_name}
                        }

                    with open(out_file, 'w') as f:
                        json.dump(feature_collection, f, indent=2)

                progress_bar.update_progress()

            # 5. Final Write for Single Mode
            if mode == 'single':
                feature_collection = {
                    "type": "FeatureCollection",
                    "features": combined_features
                }
                
                # Add CRS block if not WGS84
                if final_crs_name and "CRS84" not in final_crs_name:
                    feature_collection["crs"] = {
                        "type": "name", "properties": {"name": final_crs_name}
                    }
                
                with open(final_output_path, 'w') as f:
                    json.dump(feature_collection, f, indent=2)

            QMessageBox.information(self, "Success", "Export completed successfully.")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during export: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()