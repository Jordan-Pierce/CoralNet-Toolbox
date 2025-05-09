import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QLabel, QDialog, QHBoxLayout,
                             QPushButton, QComboBox, QSpinBox, QButtonGroup, QCheckBox,
                             QFormLayout, QGroupBox, QGraphicsRectItem, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtWorkArea import WorkArea
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchGraphic(QGraphicsRectItem):
    def __init__(self, x, y, size, color, parent=None):
        # Use size for both width and height; color is applied separately
        super().__init__(x, y, size, size, parent)

        # Default styling
        self.default_pen = QPen(Qt.white, 2, Qt.DashLine)
        self.hover_pen = QPen(Qt.yellow, 3, Qt.DashLine)
        self.default_brush = QBrush(QColor(color.red(), color.blue(), color.green(), 50))

        # Initial appearance
        self.setPen(self.default_pen)
        self.setBrush(self.default_brush)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        """Highlight patch on hover"""
        self.setPen(self.hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Restore default appearance on hover exit"""
        self.setPen(self.default_pen)
        super().hoverLeaveEvent(event)


class PatchSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        
        self.graphics_utility = self.annotation_window.graphics_utility

        self.setWindowTitle("Sample Annotations")
        self.setWindowIcon(get_icon("coral.png"))

        self.layout = QVBoxLayout(self)

        # Setup the sampling configuration layout
        self.setup_sampling_config_layout()
        # Setup the annotation configuration layout
        self.setup_annotation_config_layout()
        # Setup the apply options layout
        self.setup_apply_options_layout()
        # Setup the bottom button controls
        self.setup_buttons_layout()

        self.sampled_annotations = []

        # Initialize graphics list
        self.annotation_graphics = []
        # Add margin work area attribute
        self.margin_work_area = None

    def setup_sampling_config_layout(self):
        """Set up the sampling method and count configuration."""
        group_box = QGroupBox("Sampling Configuration")
        layout = QFormLayout()

        # Sampling Method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "Stratified Random", "Uniform"])
        self.method_combo.currentIndexChanged.connect(self.preview_annotations)
        layout.addRow("Sampling Method:", self.method_combo)

        # Number of Annotations
        self.num_annotations_spinbox = QSpinBox()
        self.num_annotations_spinbox.setMinimum(1)
        self.num_annotations_spinbox.setMaximum(10000)
        self.num_annotations_spinbox.setValue(10)
        self.num_annotations_spinbox.valueChanged.connect(self.preview_annotations)
        layout.addRow("Number of Annotations:", self.num_annotations_spinbox)

        # Annotation Size
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(32)
        self.annotation_size_spinbox.setMaximum(10000)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.preview_annotations)
        layout.addRow("Annotation Size:", self.annotation_size_spinbox)

        # Sample Label
        self.label_combo = QComboBox()
        for label in self.label_window.labels:
            self.label_combo.addItem(label.short_label_code, label.id)
        self.label_combo.setCurrentIndex(0)
        self.label_combo.currentIndexChanged.connect(self.preview_annotations)
        layout.addRow("Select Label:", self.label_combo)

        # Exclude Regions
        self.exclude_regions_combo = QComboBox()
        self.exclude_regions_combo.addItems(["False", "True"])
        self.exclude_regions_combo.setCurrentIndex(0)
        self.exclude_regions_combo.currentIndexChanged.connect(self.preview_annotations)
        layout.addRow("Exclude Regions:", self.exclude_regions_combo)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_annotation_config_layout(self):
        """Set up the annotation size and margin configuration."""
        # Margin Offsets
        self.margin_input = MarginInput()
        self.margin_input.value_type.currentIndexChanged.connect(self.preview_annotations)
        self.margin_input.type_combo.currentIndexChanged.connect(self.preview_annotations)
        for spin in self.margin_input.margin_spins:
            spin.valueChanged.connect(self.preview_annotations)
        for double in self.margin_input.margin_doubles:
            double.valueChanged.connect(self.preview_annotations)

        # Add margin label and input directly to main layout
        self.layout.addWidget(self.margin_input)

    def setup_apply_options_layout(self):
        """Set up the application scope options."""
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

    def setup_buttons_layout(self):
        """Set up the bottom button controls."""
        button_layout = QHBoxLayout()

        # Preview Button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_annotations)
        button_layout.addWidget(self.preview_button)

        # Accept Button
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept_annotations)
        button_layout.addWidget(self.accept_button)

        # Cancel Button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        self.layout.addLayout(button_layout)

    def showEvent(self, event):
        """Handle dialog show event."""
        self.update_checkboxes()
        self.update_label_combo()
        self.update_annotation_graphics()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.update_checkboxes()
        self.clear_annotation_graphics()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.update_checkboxes()
        self.clear_annotation_graphics()
        super().reject()
        
    def update_checkboxes(self):
        """Clear the checkboxes states."""
        # Temporarily disable exclusivity to allow unchecking all checkboxes
        self.apply_group.setExclusive(False)
        self.apply_filtered_checkbox.setChecked(False)
        self.apply_prev_checkbox.setChecked(False)
        self.apply_next_checkbox.setChecked(False)
        self.apply_all_checkbox.setChecked(False)
        # Restore exclusivity
        self.apply_group.setExclusive(True)

    def update_label_combo(self):
        """Update the label combo box with the current labels."""
        self.label_combo.clear()
        for label in self.label_window.labels:
            self.label_combo.addItem(label.short_label_code, label.id)
        self.label_combo.setCurrentIndex(0)

    def sample_annotations(self, method, num_annotations, annotation_size, 
                           margins, image_width, image_height, exclude_regions=False, exclude_polygons=None):
        """Sample annotations using the specified method, optionally excluding regions."""
        if not margins:
            return []

        left, top, right, bottom = margins
        annotations = []

        def rect_overlaps_any_polygon(x, y, size, polygons):
            """Check if the rectangle (x, y, size, size) overlaps any polygon in polygons."""
            rect = QRectF(x, y, size, size)
            rect_poly = QPolygonF([
                rect.topLeft(),
                rect.topRight(),
                rect.bottomRight(),
                rect.bottomLeft(),
            ])
            for poly in polygons:
                if poly.intersects(rect_poly):
                    return True
            return False

        # Prepare polygons for exclusion if needed
        polygons = []
        if exclude_regions and exclude_polygons:
            polygons = exclude_polygons

        if method == "Random":
            min_spacing = annotation_size // 2
            x_min = left
            x_max = image_width - annotation_size - right
            y_min = top
            y_max = image_height - annotation_size - bottom

            num_candidates = max(num_annotations * 10, 1000)
            x_candidates = np.random.randint(x_min, x_max + 1, num_candidates)
            y_candidates = np.random.randint(y_min, y_max + 1, num_candidates)
            candidates = np.column_stack((x_candidates, y_candidates))

            selected = []
            remaining_indices = np.arange(num_candidates)

            while len(selected) < num_annotations and remaining_indices.size > 0:
                idx = np.random.choice(remaining_indices)
                current = candidates[idx]
                x, y = current
                # Exclude if overlaps any polygon
                if polygons and rect_overlaps_any_polygon(x, y, annotation_size, polygons):
                    # Remove this candidate and continue
                    remaining_indices = remaining_indices[remaining_indices != idx]
                    continue
                selected.append(current)

                dx = np.abs(candidates[remaining_indices, 0] - x)
                dy = np.abs(candidates[remaining_indices, 1] - y)
                overlap_mask = ~((dx < min_spacing) & (dy < min_spacing))
                remaining_indices = remaining_indices[overlap_mask]

            annotations = [(x, y, annotation_size) for x, y in selected]

            # If still short, fill remaining positions without spacing checks
            if len(annotations) < num_annotations:
                needed = num_annotations - len(annotations)
                tries = 0
                while needed > 0 and tries < 10 * needed:
                    x = np.random.randint(x_min, x_max + 1)
                    y = np.random.randint(y_min, y_max + 1)
                    if polygons and rect_overlaps_any_polygon(x, y, annotation_size, polygons):
                        tries += 1
                        continue
                    annotations.append((x, y, annotation_size))
                    needed -= 1
                    tries += 1

        elif method in ["Uniform", "Stratified Random"]:
            grid_size = int(num_annotations ** 0.5)
            usable_width = image_width - left - right - annotation_size
            usable_height = image_height - top - bottom - annotation_size

            x_step = usable_width / max(1, grid_size - 1)
            y_step = usable_height / max(1, grid_size - 1)

            for i in range(grid_size):
                for j in range(grid_size):
                    if len(annotations) >= num_annotations:
                        break

                    if method == "Uniform":
                        x = left + int(i * x_step)
                        y = top + int(j * y_step)
                    else:  # Stratified Random
                        x = int(left + i * x_step + random.uniform(0, x_step))
                        y = int(top + j * y_step + random.uniform(0, y_step))

                    x = max(left, min(x, image_width - annotation_size - right))
                    y = max(top, min(y, image_height - annotation_size - bottom))

                    # Exclude if overlaps any polygon
                    if polygons and rect_overlaps_any_polygon(x, y, annotation_size, polygons):
                        continue

                    annotations.append((x, y, annotation_size))

        return annotations[:num_annotations]

    def update_annotation_graphics(self):
        """Create and display annotation preview graphics, including margin visualization."""
        self.clear_annotation_graphics()

        # Remove previous margin work area and its graphics if present
        if self.margin_work_area is not None:
            self.margin_work_area.remove_from_scene()
            self.margin_work_area = None
    
        # Get current parameters
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        sample_label = self.label_window.get_label_by_short_code(self.label_combo.currentText())
        exclude_regions = self.exclude_regions_combo.currentText() == "True"
    
        if not sample_label:
            return
    
        try:
            # Validate margins before sampling
            margins = self.margin_input.get_margins(self.annotation_window.pixmap_image.width(),
                                                    self.annotation_window.pixmap_image.height())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Margins", str(e))
            return
    
        # Create a work area to represent the valid annotation area (inside margins)
        image_width = self.annotation_window.pixmap_image.width()
        image_height = self.annotation_window.pixmap_image.height()
        left, top, right, bottom = margins
        
        # Calculate inner rectangle (area inside margins)
        inner_x = left
        inner_y = top
        inner_width = image_width - left - right
        inner_height = image_height - top - bottom
        
        # Create a work area for the margin visualization and store as attribute
        self.margin_work_area = WorkArea(inner_x, 
                                         inner_y, 
                                         inner_width, 
                                         inner_height, 
                                         self.annotation_window.current_image_path)
                
        # Create graphics using the WorkArea's own method
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        margin_graphics = self.margin_work_area.create_graphics(self.annotation_window.scene, 
                                                                thickness, 
                                                                include_shadow=True)
        
        # Don't show remove button for margin visualization
        self.annotation_graphics.append(margin_graphics)
    
        # Prepare polygons to exclude if needed
        polygons = []
        if exclude_regions:
            # Get all annotation polygons for the current image
            image_annotations = self.annotation_window.get_image_annotations()
            polygons = [a.get_polygon() for a in image_annotations]

        # Sample new annotations
        self.sampled_annotations = self.sample_annotations(
            method,
            num_annotations,
            annotation_size,
            margins,
            image_width,
            image_height,
            exclude_regions=exclude_regions,
            exclude_polygons=polygons
        )
    
        # Create graphics for each annotation
        for annotation in self.sampled_annotations:
            x, y, size = annotation
            graphic = PatchGraphic(x, y, size, sample_label.color)
            self.annotation_window.scene.addItem(graphic)
            self.annotation_graphics.append(graphic)

    def preview_annotations(self):
        """Preview sampled annotations."""
        self.update_annotation_graphics()

    def accept_annotations(self):
        """Accept the sampled annotations and add them to the current image."""
        self.add_sampled_annotations(self.method_combo.currentText(),
                                     self.num_annotations_spinbox.value(),
                                     self.annotation_size_spinbox.value())

    def add_sampled_annotations(self, method, num_annotations, annotation_size):
        """Add the sampled annotations to the current image."""
        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Clear the graphics
        self.clear_annotation_graphics()

        self.apply_to_filtered = self.apply_filtered_checkbox.isChecked()
        self.apply_to_prev = self.apply_prev_checkbox.isChecked()
        self.apply_to_next = self.apply_next_checkbox.isChecked()
        self.apply_to_all = self.apply_all_checkbox.isChecked()

        # Gets the label from LabelWindow
        sample_label = self.label_window.get_label_by_short_code(self.label_combo.currentText())
        if not sample_label:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", "Selected label not found")
            return

        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", "No image is currently selected")
            return

        # Prepare exclude regions flag
        exclude_regions = self.exclude_regions_combo.currentText() == "True"

        # Determine which images to apply annotations to
        if self.apply_to_filtered:
            image_paths = self.image_window.table_model.filtered_paths
        elif self.apply_to_prev:
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                image_paths = self.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                image_paths = [current_image_path]
        elif self.apply_to_next:
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                image_paths = self.image_window.table_model.filtered_paths[current_index:]
            else:
                image_paths = [current_image_path]
        elif self.apply_to_all:
            image_paths = self.image_window.raster_manager.image_paths
        else:
            # Only apply to the current image
            image_paths = [current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths) * num_annotations)

        try:
            for image_path in image_paths:
                sampled_annotations = []

                # Get the raster from the manager
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster:
                    print(f"Warning: Could not get raster for {image_path}")
                    continue

                # Get image dimensions from the raster
                width = raster.width
                height = raster.height

                # Validate margins for each image
                margins = self.margin_input.get_margins(width, height)
                
                # Prepare polygons to exclude if needed
                polygons = []
                if exclude_regions:
                    # Get all annotation polygons for this image
                    image_annotations = self.annotation_window.get_image_annotations(image_path)
                    polygons = [a.get_polygon() for a in image_annotations]

                # Sample the annotations given params
                annotations_coords = self.sample_annotations(method,
                                                             num_annotations,
                                                             annotation_size,
                                                             margins,
                                                             width,
                                                             height,
                                                             exclude_regions=exclude_regions,
                                                             exclude_polygons=polygons)

                for x, y, size in annotations_coords:
                    # Create the annotation with center point
                    new_annotation = PatchAnnotation(QPointF(x + size // 2, y + size // 2),
                                                     size,
                                                     sample_label.short_label_code,
                                                     sample_label.long_label_code,
                                                     sample_label.color,
                                                     image_path,
                                                     sample_label.id,
                                                     transparency=self.annotation_window.transparency)

                    # Add annotation to the annotation window
                    self.annotation_window.add_annotation_to_dict(new_annotation)
                    sampled_annotations.append(new_annotation)
                    progress_bar.update_progress()

                # Update the raster's annotation info
                self.image_window.update_image_annotations(image_path)
                
            # Load the annotations for current image
            self.annotation_window.load_annotations(image_path=image_path, annotations=sampled_annotations)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", f"Error adding sampled annotations: {str(e)}")
            raise e
        finally:
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()

        # Close the dialog
        self.accept()

    def clear_annotation_graphics(self):
        """Remove all annotation preview graphics, including margin visualizations."""
        for graphic in self.annotation_graphics:
            self.annotation_window.scene.removeItem(graphic)
        self.annotation_graphics = []
        # Remove margin work area and its shadow if present
        if self.margin_work_area is not None:
            self.margin_work_area.remove_from_scene()
            self.margin_work_area = None
        self.annotation_window.viewport().update()
