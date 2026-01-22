import warnings

import os
import math

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLineEdit, QPushButton, QFileDialog, QApplication, QMessageBox,
    QLabel, QWidget, QListWidget, QListWidgetItem, QTabWidget
)

from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Define which metrics are applicable to each annotation type
# True = metric is calculated, False = None is returned
METRICS_BY_ANNOTATION_TYPE = {
    'PatchAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': False,
        'bbox_min_y': False,
        'bbox_max_x': False,
        'bbox_max_y': False,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': False,
        'bbox_height': False,
        'equivalent_diameter': False,
        # Shape
        'major_axis': False,
        'minor_axis': False,
        'hull_area': False,
        'hull_perimeter': False,
        'aspect_ratio': False,
        'orientation': False,
        'roundness': False,
        'circularity': False,
        'compactness': False,
        'solidity': False,
        'convexity': False,
        'elongation': False,
        'rectangularity': False,
        'eccentricity': False,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'RectangleAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Rectangle now has full morphology support via get_morphology()
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'PolygonAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Full morphology support
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'MultiPolygonAnnotation': {
        # Note: MultiPolygonAnnotation constituents (PolygonAnnotation) will be exported
        # individually with parent_annotation_id set. These metrics apply to each constituent.
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Full morphology support for constituent polygons
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
}

# Organize metrics into categories for display
METRIC_CATEGORIES = {
    'Location': ['centroid_x', 'centroid_y', 'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y'],
    'Size': ['area', 'perimeter', 'bbox_width', 'bbox_height', 'equivalent_diameter'],
    'Shape': ['major_axis', 'minor_axis', 'hull_area', 'hull_perimeter', 'aspect_ratio',
              'orientation', 'roundness', 'circularity', 'compactness', 'solidity',
              'convexity', 'elongation', 'rectangularity', 'eccentricity'],
    '3D Metrics': ['volume', 'surface_area', 'min_z', 'max_z'],
}

# All spatial metrics in order
ALL_METRICS = (
    METRIC_CATEGORIES['Location'] +
    METRIC_CATEGORIES['Size'] +
    METRIC_CATEGORIES['Shape'] +
    METRIC_CATEGORIES['3D Metrics']
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportSpatialMetrics(QDialog):
    """Dialog for exporting spatial metrics of annotations to CSV."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Export Spatial Metrics (CSV)")
        self.resize(500, 600)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # 1. Info Section
        self.setup_info_layout()

        # 2. File Path Selection
        self.setup_file_path_layout()

        # 3. Column Selection (Tabbed)
        self.setup_column_selection_layout()

        # 4. Action Buttons
        self.setup_buttons_layout()

    def showEvent(self, event):
        """Handle the show event to refresh lists."""
        super().showEvent(event)
        self.update_images_list()
        self.update_labels_list()

    # ----------------------------------------------------------------------
    # UI Setup Methods
    # ----------------------------------------------------------------------

    def setup_info_layout(self):
        """Simple information header."""
        info_label = QLabel(
            "<b>Export Spatial Metrics to CSV</b><br>"
            "Export spatial metrics (area, perimeter, morphology, etc.) for annotations. "
            "Filter by images, labels, annotation types, and select which metrics to include."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("margin-bottom: 5px;")
        self.layout.addWidget(info_label)

    def setup_file_path_layout(self):
        """Setup file path selection."""
        groupbox = QGroupBox("Output File")
        layout = QFormLayout()

        self.file_path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file_path)

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)

        layout.addRow("File Path:", file_layout)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_column_selection_layout(self):
        """Setup tabbed column selection with Images, Labels, Annotation Types, and Spatial Metrics."""
        groupbox = QGroupBox("Column Selection")
        layout = QVBoxLayout()

        # Tab Widget
        self.tab_widget = QTabWidget()

        # --- Tab 1: Images ---
        images_tab = QWidget()
        images_layout = QVBoxLayout(images_tab)

        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.ExtendedSelection)
        images_layout.addWidget(self.images_list)

        images_buttons = QHBoxLayout()
        self.images_select_all_btn = QPushButton("Select All")
        self.images_select_all_btn.clicked.connect(self.select_all_images)
        self.images_deselect_all_btn = QPushButton("Deselect All")
        self.images_deselect_all_btn.clicked.connect(self.deselect_all_images)
        images_buttons.addWidget(self.images_select_all_btn)
        images_buttons.addWidget(self.images_deselect_all_btn)
        images_layout.addLayout(images_buttons)

        self.tab_widget.addTab(images_tab, "Images")

        # --- Tab 2: Labels ---
        labels_tab = QWidget()
        labels_layout = QVBoxLayout(labels_tab)

        self.labels_list = QListWidget()
        self.labels_list.setSelectionMode(QListWidget.ExtendedSelection)
        labels_layout.addWidget(self.labels_list)

        labels_buttons = QHBoxLayout()
        self.labels_select_all_btn = QPushButton("Select All")
        self.labels_select_all_btn.clicked.connect(self.select_all_labels)
        self.labels_deselect_all_btn = QPushButton("Deselect All")
        self.labels_deselect_all_btn.clicked.connect(self.deselect_all_labels)
        labels_buttons.addWidget(self.labels_select_all_btn)
        labels_buttons.addWidget(self.labels_deselect_all_btn)
        labels_layout.addLayout(labels_buttons)

        self.tab_widget.addTab(labels_tab, "Labels")

        # --- Tab 3: Annotation Types ---
        types_tab = QWidget()
        types_layout = QVBoxLayout(types_tab)

        self.types_list = QListWidget()
        self.types_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.types_list.addItems([
            "PatchAnnotation",
            "RectangleAnnotation",
            "PolygonAnnotation",
            "MultiPolygonAnnotation"
        ])
        self.types_list.selectAll()
        types_layout.addWidget(self.types_list)

        types_buttons = QHBoxLayout()
        self.types_select_all_btn = QPushButton("Select All")
        self.types_select_all_btn.clicked.connect(self.select_all_types)
        self.types_deselect_all_btn = QPushButton("Deselect All")
        self.types_deselect_all_btn.clicked.connect(self.deselect_all_types)
        types_buttons.addWidget(self.types_select_all_btn)
        types_buttons.addWidget(self.types_deselect_all_btn)
        types_layout.addLayout(types_buttons)

        self.tab_widget.addTab(types_tab, "Annotation Types")

        # --- Tab 4: Spatial Metrics ---
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_list = QListWidget()
        self.metrics_list.setSelectionMode(QListWidget.ExtendedSelection)

        # Add metrics with category headers
        for category, metrics in METRIC_CATEGORIES.items():
            # Add category separator
            category_item = QListWidgetItem(f"── {category} ──")
            category_item.setFlags(Qt.NoItemFlags)  # Non-selectable header
            self.metrics_list.addItem(category_item)
            # Add metrics
            for metric in metrics:
                self.metrics_list.addItem(metric)

        self.metrics_list.selectAll()
        metrics_layout.addWidget(self.metrics_list)

        metrics_buttons = QHBoxLayout()
        self.metrics_select_all_btn = QPushButton("Select All")
        self.metrics_select_all_btn.clicked.connect(self.select_all_metrics)
        self.metrics_deselect_all_btn = QPushButton("Deselect All")
        self.metrics_deselect_all_btn.clicked.connect(self.deselect_all_metrics)
        metrics_buttons.addWidget(self.metrics_select_all_btn)
        metrics_buttons.addWidget(self.metrics_deselect_all_btn)
        metrics_layout.addLayout(metrics_buttons)

        self.tab_widget.addTab(metrics_tab, "Spatial Metrics")

        layout.addWidget(self.tab_widget)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_buttons_layout(self):
        """Setup export and cancel buttons."""
        button_layout = QHBoxLayout()

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_spatial_metrics)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)

    # ----------------------------------------------------------------------
    # List Population Methods
    # ----------------------------------------------------------------------

    def update_images_list(self):
        """Populate the images list from the image window."""
        self.images_list.clear()
        if hasattr(self.image_window, 'raster_manager'):
            for image_path in self.image_window.raster_manager.image_paths:
                image_name = os.path.basename(image_path)
                item = QListWidgetItem(image_name)
                item.setData(Qt.UserRole, image_path)  # Store full path
                self.images_list.addItem(item)
        self.images_list.selectAll()

    def update_labels_list(self):
        """Populate the labels list from the label window."""
        self.labels_list.clear()
        if hasattr(self.label_window, 'labels'):
            for label in self.label_window.labels:
                item = QListWidgetItem(f"{label.short_label_code} - {label.long_label_code}")
                item.setData(Qt.UserRole, label.short_label_code)  # Store short code
                self.labels_list.addItem(item)
        self.labels_list.selectAll()

    # ----------------------------------------------------------------------
    # Selection Methods
    # ----------------------------------------------------------------------

    def select_all_images(self):
        self.images_list.selectAll()

    def deselect_all_images(self):
        self.images_list.clearSelection()

    def select_all_labels(self):
        self.labels_list.selectAll()

    def deselect_all_labels(self):
        self.labels_list.clearSelection()

    def select_all_types(self):
        self.types_list.selectAll()

    def deselect_all_types(self):
        self.types_list.clearSelection()

    def select_all_metrics(self):
        """Select all selectable metrics (excluding category headers)."""
        for i in range(self.metrics_list.count()):
            item = self.metrics_list.item(i)
            if item.flags() & Qt.ItemIsSelectable:
                item.setSelected(True)

    def deselect_all_metrics(self):
        self.metrics_list.clearSelection()

    # ----------------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------------

    def browse_file_path(self):
        """Open file dialog to select output CSV path."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spatial Metrics CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.file_path_edit.setText(file_path)

    def get_selected_images(self):
        """Get list of selected image paths."""
        selected = self.images_list.selectedItems()
        if not selected:
            # If none selected, return all
            return [self.images_list.item(i).data(Qt.UserRole)
                    for i in range(self.images_list.count())]
        return [item.data(Qt.UserRole) for item in selected]

    def get_selected_labels(self):
        """Get list of selected label short codes."""
        selected = self.labels_list.selectedItems()
        if not selected:
            # If none selected, return all
            return [self.labels_list.item(i).data(Qt.UserRole)
                    for i in range(self.labels_list.count())]
        return [item.data(Qt.UserRole) for item in selected]

    def get_selected_types(self):
        """Get list of selected annotation types."""
        selected = self.types_list.selectedItems()
        if not selected:
            # If none selected, return all
            return [self.types_list.item(i).text()
                    for i in range(self.types_list.count())]
        return [item.text() for item in selected]

    def get_selected_metrics(self):
        """Get list of selected metric names."""
        selected = self.metrics_list.selectedItems()
        if not selected:
            # If none selected, return all selectable metrics
            return ALL_METRICS
        return [item.text() for item in selected if item.text() in ALL_METRICS]

    # ----------------------------------------------------------------------
    # Metric Calculation Methods
    # ----------------------------------------------------------------------

    def calculate_metrics_for_annotation(self, annotation, selected_metrics, z_channel=None, z_unit=None):
        """
        Calculate spatial metrics for a single annotation.

        Args:
            annotation: The annotation object
            selected_metrics: List of metric names to calculate
            z_channel: The z-channel data for 3D metrics (optional)
            z_unit: The unit of the z-channel data (optional)

        Returns:
            dict: Dictionary with metric values (pixel and meters columns)
        """
        annotation_type = type(annotation).__name__
        type_metrics = METRICS_BY_ANNOTATION_TYPE.get(annotation_type, {})

        result = {}

        # Get scale info if available
        scale_x = annotation.scale_x
        scale_y = annotation.scale_y
        scale_units = annotation.scale_units
        has_scale = scale_x is not None and scale_y is not None and scale_units is not None

        # Calculate conversion factor to meters if scale is available
        to_meters_factor = 1.0
        if has_scale and scale_units:
            try:
                to_meters_factor = convert_scale_units(1.0, scale_units, 'metre')
            except Exception:
                to_meters_factor = 1.0

        # Get basic geometry info
        center_xy = annotation.center_xy
        top_left = annotation.get_bounding_box_top_left()
        bottom_right = annotation.get_bounding_box_bottom_right()

        # Calculate bbox dimensions
        bbox_width = abs(bottom_right.x() - top_left.x()) if top_left and bottom_right else None
        bbox_height = abs(bottom_right.y() - top_left.y()) if top_left and bottom_right else None

        # Get morphology data - always try to get it now that RectangleAnnotation supports it
        morph_data = annotation.get_morphology()

        # Process each selected metric
        for metric in selected_metrics:
            if metric not in type_metrics:
                continue

            is_applicable = type_metrics.get(metric, False)
            pixel_value = None
            meter_value = None

            if is_applicable:
                try:
                    # Location metrics
                    if metric == 'centroid_x':
                        pixel_value = float(center_xy.x()) if center_xy else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'centroid_y':
                        pixel_value = float(center_xy.y()) if center_xy else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_y * to_meters_factor
                    elif metric == 'bbox_min_x':
                        pixel_value = float(top_left.x()) if top_left else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'bbox_min_y':
                        pixel_value = float(top_left.y()) if top_left else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_y * to_meters_factor
                    elif metric == 'bbox_max_x':
                        pixel_value = float(bottom_right.x()) if bottom_right else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'bbox_max_y':
                        pixel_value = float(bottom_right.y()) if bottom_right else None
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_y * to_meters_factor

                    # Size metrics
                    elif metric == 'area':
                        pixel_value = annotation.get_area()
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * scale_y * (to_meters_factor ** 2)
                    elif metric == 'perimeter':
                        pixel_value = annotation.get_perimeter()
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'bbox_width':
                        pixel_value = bbox_width
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'bbox_height':
                        pixel_value = bbox_height
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_y * to_meters_factor
                    elif metric == 'equivalent_diameter':
                        area = annotation.get_area()
                        if area is not None and area > 0:
                            pixel_value = math.sqrt(4 * area / math.pi)
                            if has_scale:
                                area_meters = area * scale_x * scale_y * (to_meters_factor ** 2)
                                meter_value = math.sqrt(4 * area_meters / math.pi)

                    # Shape metrics from morphology - all pulled from get_morphology() now
                    elif metric in ['major_axis', 'minor_axis', 'hull_perimeter']:
                        # Linear measurements
                        if morph_data:
                            pixel_value = morph_data.get(metric)
                            if pixel_value is not None and has_scale:
                                meter_value = pixel_value * scale_x * to_meters_factor
                    elif metric == 'hull_area':
                        # Area measurement
                        if morph_data:
                            pixel_value = morph_data.get(metric)
                            if pixel_value is not None and has_scale:
                                meter_value = pixel_value * scale_x * scale_y * (to_meters_factor ** 2)
                    elif metric in ['aspect_ratio', 'orientation', 'roundness', 'circularity', 
                                    'compactness', 'solidity', 'convexity', 'elongation', 
                                    'rectangularity', 'eccentricity']:
                        # Unitless ratios - pull directly from morphology
                        if morph_data:
                            pixel_value = morph_data.get(metric)
                        meter_value = pixel_value  # Unitless

                    # 3D Metrics
                    elif metric == 'volume':
                        if z_channel is not None and has_scale:
                            # Convert scales to meters/pixel as required by get_scaled_volume
                            scale_x_meters = scale_x * to_meters_factor
                            scale_y_meters = scale_y * to_meters_factor
                            volume = annotation.get_scaled_volume(z_channel, scale_x_meters, scale_y_meters, z_unit)
                            if volume is not None:
                                # Volume is already in the correct units (cubic meters)
                                meter_value = volume
                                # Calculate pixel-based volume: sum of z-values in pixels
                                try:
                                    z_slice, mask = annotation._get_raster_slice_and_mask(z_channel)
                                    if z_slice.size > 0 and mask.size > 0 and np.any(mask):
                                        pixel_value = float(np.sum(z_slice[mask]))
                                except Exception:
                                    pixel_value = None
                    elif metric == 'surface_area':
                        if z_channel is not None and has_scale:
                            # Convert scales to meters/pixel as required by get_scaled_surface_area
                            scale_x_meters = scale_x * to_meters_factor
                            scale_y_meters = scale_y * to_meters_factor
                            surf_area = annotation.get_scaled_surface_area(z_channel, 
                                                                           scale_x_meters, 
                                                                           scale_y_meters,
                                                                           z_unit)
                            if surf_area is not None:
                                # Surface area is already in the correct units (square meters)
                                meter_value = surf_area
                                # Calculate pixel-based surface area: sum of 3D surface elements
                                try:
                                    z_slice, mask = annotation._get_raster_slice_and_mask(z_channel)
                                    if z_slice.size > 0 and mask.size > 0 and np.any(mask):
                                        # Calculate gradients in pixel space
                                        dz_dy, dz_dx = np.gradient(z_slice)
                                        # Surface area multiplier for each pixel
                                        multiplier = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
                                        # Sum surface elements inside mask (each pixel has area = 1 in pixel space)
                                        pixel_value = float(np.sum(multiplier[mask]))
                                except Exception:
                                    pixel_value = None
                    elif metric == 'min_z':
                        if z_channel is not None:
                            z_data = annotation.get_min_z(z_channel, scale_x, z_unit)
                            if z_data:
                                pixel_value = z_data.get('pixels')
                                meter_value = z_data.get('meters')
                    elif metric == 'max_z':
                        if z_channel is not None:
                            z_data = annotation.get_max_z(z_channel, scale_x, z_unit)
                            if z_data:
                                pixel_value = z_data.get('pixels')
                                meter_value = z_data.get('meters')

                except Exception as e:
                    # Log the error but continue with None values
                    print(f"Error calculating {metric} for annotation {annotation.id}: {e}")

            # Round values
            if pixel_value is not None and isinstance(pixel_value, float):
                pixel_value = round(pixel_value, 4)
            if meter_value is not None and isinstance(meter_value, float):
                meter_value = round(meter_value, 4)

            # Add to result
            result[f"{metric} (pixels)"] = pixel_value
            result[f"{metric} (meters)"] = meter_value

        return result

    # ----------------------------------------------------------------------
    # Export Method
    # ----------------------------------------------------------------------

    def export_spatial_metrics(self):
        """Export spatial metrics to CSV file."""
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "No File Selected", "Please select an output file path.")
            return

        # Get selections
        selected_images = self.get_selected_images()
        selected_labels = self.get_selected_labels()
        selected_types = self.get_selected_types()
        selected_metrics = self.get_selected_metrics()

        if not selected_images:
            QMessageBox.warning(self, "No Images", "Please select at least one image.")
            return
        if not selected_labels:
            QMessageBox.warning(self, "No Labels", "Please select at least one label.")
            return
        if not selected_types:
            QMessageBox.warning(self, "No Types", "Please select at least one annotation type.")
            return
        if not selected_metrics:
            QMessageBox.warning(self, "No Metrics", "Please select at least one spatial metric.")
            return

        # Set cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Collect annotations
            all_annotations = list(self.annotation_window.annotations_dict.values())

            # Filter annotations
            filtered_annotations = []
            for ann in all_annotations:
                ann_type = type(ann).__name__
                if ann_type not in selected_types:
                    continue
                if ann.image_path not in selected_images:
                    continue
                if ann.label.short_label_code not in selected_labels:
                    continue
                filtered_annotations.append(ann)

            if not filtered_annotations:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "No Annotations",
                                    "No annotations match the selected filters.")
                return

            # Start progress bar
            progress_bar = ProgressBar(self, "Exporting Spatial Metrics")
            progress_bar.show()
            progress_bar.start_progress(len(filtered_annotations))

            # Cache z-channel data per image to avoid repeated lookups
            z_channel_cache = {}

            # Build data rows
            rows = []
            for annotation in filtered_annotations:
                if progress_bar.wasCanceled():
                    break

                # Get z-channel data for this annotation's image if needed
                z_channel = None
                z_unit = None
                if any(metric in selected_metrics for metric in ['volume', 'surface_area', 'min_z', 'max_z']):
                    if annotation.image_path not in z_channel_cache:
                        # Try to get z-channel from raster_manager
                        raster = self.image_window.raster_manager.get_raster(annotation.image_path)
                        if raster:
                            z_channel_cache[annotation.image_path] = {
                                'z_channel': raster.z_channel_lazy,
                                'z_unit': raster.z_unit
                            }
                        else:
                            z_channel_cache[annotation.image_path] = {'z_channel': None, 'z_unit': None}
                    
                    cached_data = z_channel_cache.get(annotation.image_path, {})
                    z_channel = cached_data.get('z_channel')
                    z_unit = cached_data.get('z_unit')

                # Handle MultiPolygonAnnotation by exporting constituent polygons
                if isinstance(annotation, MultiPolygonAnnotation):
                    # Export each constituent polygon as a separate row
                    for poly in annotation.polygons:
                        # Base columns (always included)
                        row = {
                            'annotation_id': poly.id,
                            'parent_annotation_id': annotation.id,  # Reference to parent MultiPolygon
                            'image_path': poly.image_path,
                            'image_name': os.path.basename(poly.image_path),
                            'annotation_type': 'Polygon',  # Constituent is always a Polygon
                            'label_short': poly.label.short_label_code,
                            'label_long': poly.label.long_label_code,
                            'color_rgb': str(poly.label.color.getRgb()[:3]),
                        }

                        # Calculate and add spatial metrics for the constituent polygon
                        metrics = self.calculate_metrics_for_annotation(poly, selected_metrics, z_channel, z_unit)
                        row.update(metrics)

                        rows.append(row)
                else:
                    # Regular annotation (not MultiPolygon)
                    row = {
                        'annotation_id': annotation.id,
                        'parent_annotation_id': None,  # No parent for regular annotations
                        'image_path': annotation.image_path,
                        'image_name': os.path.basename(annotation.image_path),
                        'annotation_type': type(annotation).__name__.replace('Annotation', ''),
                        'label_short': annotation.label.short_label_code,
                        'label_long': annotation.label.long_label_code,
                        'color_rgb': str(annotation.label.color.getRgb()[:3]),
                    }

                    # Calculate and add spatial metrics
                    metrics = self.calculate_metrics_for_annotation(annotation, selected_metrics, z_channel, z_unit)
                    row.update(metrics)

                    rows.append(row)
                
                progress_bar.update_progress()

            # Create DataFrame and export
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(file_path, index=False)

            progress_bar.stop_progress()
            progress_bar.close()

            QApplication.restoreOverrideCursor()
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(rows)} annotations to:\n{os.path.basename(file_path)}"
            )
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")
