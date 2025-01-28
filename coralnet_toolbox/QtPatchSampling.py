import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QLabel, QDialog, QHBoxLayout, 
                             QPushButton, QComboBox, QSpinBox, QButtonGroup, QCheckBox,
                             QFormLayout, QGroupBox, QGraphicsRectItem, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.QtCommon import MarginInput


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
        

class PatchGraphic(QGraphicsRectItem):
    def __init__(self, x, y, size, parent=None):
        super().__init__(x, y, size, size, parent)
        
        # Default styling
        self.default_pen = QPen(Qt.white, 2, Qt.DashLine)
        self.hover_pen = QPen(Qt.yellow, 3, Qt.DashLine)
        self.default_brush = QBrush(QColor(255, 255, 255, 50))
        
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

        self.setWindowTitle("Sample Annotations")
        self.setWindowIcon(get_icon("coral.png"))
        
        self.layout = QVBoxLayout(self)
        
        # Create sections of the dialog
        self.setup_sampling_config_layout()
        self.setup_annotation_config_layout()
        self.setup_apply_options_layout()
        self.setup_buttons_layout()
        
        self.sampled_annotations = []
        
        # Initialize graphics list
        self.annotation_graphics = []

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

        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.apply_all_checkbox = QCheckBox("Apply to all images")

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
        self.update_annotation_graphics()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.clear_annotation_graphics()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.clear_annotation_graphics()
        super().reject()
                
    def validate_margins(self, raw_margins):
        """
        Validate and convert margins to pixel values in the order (Left, Top, Right, Bottom).
        Handles both single values and tuples, adjusting for percentage conversion based on image dimensions.
        """
        # Check if we're dealing with percentages or pixels
        is_percentage = self.margin_input.value_type.currentIndex() == 1
        image_width = self.annotation_window.image_pixmap.width()
        image_height = self.annotation_window.image_pixmap.height()

        margin_pixels = [0, 0, 0, 0]  # [Left, Top, Right, Bottom]

        try:
            # Single value input
            if isinstance(raw_margins, (int, float)):
                if is_percentage:
                    if not (0.0 <= raw_margins <= 1.0):
                        raise ValueError("Percentage must be between 0 and 1")
                    # Apply percentage to all margins using correct dimensions
                    margin_pixels = [
                        raw_margins * image_width,    # Left
                        raw_margins * image_height,   # Top
                        raw_margins * image_width,    # Right
                        raw_margins * image_height    # Bottom
                    ]
                else:
                    margin_pixels = [raw_margins] * 4

            # Multiple values input (original order: Top, Right, Bottom, Left)
            elif isinstance(raw_margins, tuple) and len(raw_margins) == 4:
                # Reorder to (Left, Top, Right, Bottom)
                ordered_margins = (
                    raw_margins[3],  # Left
                    raw_margins[0],  # Top
                    raw_margins[1],  # Right
                    raw_margins[2]   # Bottom
                )

                if is_percentage:
                    if not all(0.0 <= m <= 1.0 for m in ordered_margins):
                        raise ValueError("All percentages must be between 0 and 1")
                    # Convert each margin using appropriate dimension
                    margin_pixels = [
                        ordered_margins[0] * image_width,   # Left
                        ordered_margins[1] * image_height,  # Top
                        ordered_margins[2] * image_width,   # Right
                        ordered_margins[3] * image_height   # Bottom
                    ]
                else:
                    margin_pixels = list(ordered_margins)

            else:
                raise ValueError("Invalid margin format")

            # Convert to integers and validate
            margin_pixels = [int(m) for m in margin_pixels]
            if (margin_pixels[0] + margin_pixels[2]) >= image_width:
                raise ValueError("Horizontal margins exceed image width")
            if (margin_pixels[1] + margin_pixels[3]) >= image_height:
                raise ValueError("Vertical margins exceed image height")

            return tuple(margin_pixels)

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Margins", str(e))
            return None

    def sample_annotations(self, method, num_annotations, annotation_size, margins, image_width, image_height):
        """Sample annotations using the specified method."""
        if not margins:
            return []
        
        left, top, right, bottom = margins
        annotations = []
        
        if method == "Random":
            min_spacing = annotation_size // 2
            x_min = left
            x_max = image_width - annotation_size - right
            y_min = top
            y_max = image_height - annotation_size - bottom

            # Generate a large pool of candidate positions
            num_candidates = max(num_annotations * 10, 1000)  # Adjust based on expected density
            x_candidates = np.random.randint(x_min, x_max + 1, num_candidates)
            y_candidates = np.random.randint(y_min, y_max + 1, num_candidates)
            candidates = np.column_stack((x_candidates, y_candidates))

            selected = []
            remaining_indices = np.arange(num_candidates)  # Track which candidates are still viable

            while len(selected) < num_annotations and remaining_indices.size > 0:
                # Pick a random candidate from the remaining pool
                idx = np.random.choice(remaining_indices)
                current = candidates[idx]
                selected.append(current)

                # Remove candidates too close to the selected one
                dx = np.abs(candidates[remaining_indices, 0] - current[0])
                dy = np.abs(candidates[remaining_indices, 1] - current[1])
                overlap_mask = ~((dx < min_spacing) & (dy < min_spacing))
                remaining_indices = remaining_indices[overlap_mask]

            # Convert to list of tuples with annotation size
            annotations = [(x, y, annotation_size) for x, y in selected]

            # If still short, fill remaining positions without spacing checks
            if len(annotations) < num_annotations:
                needed = num_annotations - len(annotations)
                x_rest = np.random.randint(x_min, x_max + 1, needed)
                y_rest = np.random.randint(y_min, y_max + 1, needed)
                annotations += [(x, y, annotation_size) for x, y in zip(x_rest, y_rest)]
                        
        elif method in ["Uniform", "Stratified Random"]:
            # Calculate grid size based on number of annotations
            grid_size = int(num_annotations ** 0.5)  # Square root for grid dimensions
            
            # Calculate available space and steps between annotations
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
                    
                    # Ensure we don't exceed image boundaries and respect margins
                    x = max(left, min(x, image_width - annotation_size - right))
                    y = max(top, min(y, image_height - annotation_size - bottom))
                    
                    annotations.append((x, y, annotation_size))
        
        return annotations[:num_annotations]

    def update_annotation_graphics(self):
        """Create and display annotation preview graphics."""
        self.clear_annotation_graphics()
        
        # Get current parameters  
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        raw_margins = self.margin_input.get_value()
        
        # Validate margins before sampling
        try:
            margins = self.validate_margins(raw_margins)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Margins", str(e))
            return
        
        # Sample new annotations
        self.sampled_annotations = self.sample_annotations(
            method,
            num_annotations,
            annotation_size, 
            margins,
            self.annotation_window.image_pixmap.width(),
            self.annotation_window.image_pixmap.height()
        )

        # Create graphics for each annotation
        for annotation in self.sampled_annotations:
            x, y, size = annotation
            
            # Create simple patch graphic
            graphic = PatchGraphic(x, y, size)
            
            self.annotation_window.scene.addItem(graphic)
            self.annotation_graphics.append(graphic)

    def preview_annotations(self):
        """Preview sampled annotations."""
        self.update_annotation_graphics()

    def draw_annotation_previews(self, margins):
        """Draw annotation previews on the current image."""
        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        self.annotation_window.unselect_annotations()
        for annotation in self.sampled_annotations:
            x, y, size = annotation
            new_annotation = PatchAnnotation(QPointF(x + size // 2, y + size // 2),
                                             size,
                                             self.label_window.active_label.short_label_code,
                                             self.label_window.active_label.long_label_code,
                                             self.label_window.active_label.color,
                                             self.annotation_window.current_image_path,
                                             self.label_window.active_label.id,
                                             transparency=self.annotation_window.transparency)
            
            new_annotation.create_graphics_item(self.annotation_window.scene)

    def accept_annotations(self):
        """Accept the sampled annotations and add them to the current image."""
        margins = self.margin_input.get_value()

        self.add_sampled_annotations(self.method_combo.currentText(),
                                     self.num_annotations_spinbox.value(),
                                     self.annotation_size_spinbox.value(),
                                     margins)

    def add_sampled_annotations(self, method, num_annotations, annotation_size, raw_margins):
        """Add the sampled annotations to the current image."""
        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.apply_to_filtered = self.apply_filtered_checkbox.isChecked()
        self.apply_to_prev = self.apply_prev_checkbox.isChecked()
        self.apply_to_next = self.apply_next_checkbox.isChecked()
        self.apply_to_all = self.apply_all_checkbox.isChecked()

        # Sets the LabelWindow and AnnotationWindow to Review
        review_label = self.label_window.get_label_by_id("-1")

        # Current image path showing
        current_image_path = self.annotation_window.current_image_path

        if self.apply_to_filtered:
            image_paths = self.image_window.filtered_image_paths
        elif self.apply_to_prev:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_to_next:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[current_image_index:]
        elif self.apply_to_all:
            image_paths = self.image_window.image_paths
        else:
            # Only apply to the current image
            image_paths = [current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths) * num_annotations)

        for image_path in image_paths:
            sampled_annotations = []

            # Load the rasterio representation
            rasterio_image = self.image_window.rasterio_open(image_path)
            height, width = rasterio_image.shape[0:2]

            # Validate margins for each image
            try:
                margins = self.validate_margins(raw_margins)
            except ValueError as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Invalid Margins", f"For image {image_path}: {str(e)}")
                return

            # Sample the annotation, given params
            annotations = self.sample_annotations(method,
                                                  num_annotations,
                                                  annotation_size,
                                                  margins,
                                                  width,
                                                  height)

            for annotation in annotations:
                x, y, size = annotation
                new_annotation = PatchAnnotation(QPointF(x + size // 2, y + size // 2),
                                                 size,
                                                 review_label.short_label_code,
                                                 review_label.long_label_code,
                                                 review_label.color,
                                                 image_path,
                                                 review_label.id,
                                                 transparency=self.annotation_window.transparency)

                # Add annotation to the dict
                self.annotation_window.annotations_dict[new_annotation.id] = new_annotation
                sampled_annotations.append(new_annotation)
                progress_bar.update_progress()

            # Update the image window's image dict
            self.image_window.update_image_annotations(image_path)

        # Load the annotations for current image
        self.annotation_window.load_these_annotations(image_path=image_path, annotations=sampled_annotations)

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()
        
        # Close the dialog
        self.clear_annotation_graphics()
        self.accept()
        
    def clear_annotation_graphics(self):
        """Remove all annotation preview graphics."""
        for graphic in self.annotation_graphics:
            self.annotation_window.scene.removeItem(graphic)
        self.annotation_graphics = []
        self.annotation_window.viewport().update()
