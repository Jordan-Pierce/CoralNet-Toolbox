import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random

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
                
    def validate_margins(self, margins, image_width, image_height):
        """
        Validate the margins parameter to ensure it is a valid type and value.

        Args:
            margins: Single int/float or tuple of 4 margins (top, right, bottom, left)
            image_width: Width of image in pixels
            image_height: Height of image in pixels
            
        Returns:
            tuple: Validated margins as (left, top, right, bottom) in pixels
            
        Raises:
            ValueError: If margins are invalid
        """
        margin_pixels = [0, 0, 0, 0]  # [top, right, bottom, left]
        
        if isinstance(margins, (int, float)):
            if isinstance(margins, float):
                if not 0.0 <= margins <= 1.0:
                    raise ValueError("Margin percentage must be between 0 and 1")
                # Use minimum dimension for percentage calculation
                margin_val = int(margins * min(image_width, image_height))
                margin_pixels = [margin_val] * 4
            else:
                if margins < 0:
                    raise ValueError("Margin pixels must be non-negative") 
                margin_pixels = [margins] * 4
                
        elif isinstance(margins, tuple) and len(margins) == 4:
            for i, margin in enumerate(margins):
                if isinstance(margin, float):
                    if not 0.0 <= margin <= 1.0:
                        raise ValueError(f"Margin percentage at index {i} must be between 0 and 1")
                    # Use height for top/bottom margins, width for left/right margins
                    dim = image_width if i in (1, 3) else image_height
                    margin_pixels[i] = int(margin * dim)
                elif isinstance(margin, int):
                    if margin < 0:
                        raise ValueError(f"Margin pixels at index {i} must be non-negative")
                    margin_pixels[i] = margin
                else:
                    raise ValueError(f"Invalid margin type at index {i}")
        else:
            raise ValueError("Margins must be a single number or tuple of 4 numbers")
            
        # Convert from TRBL to LTRB order
        margin_pixels = [margin_pixels[3], margin_pixels[0], margin_pixels[1], margin_pixels[2]]

        # Validate margin sum doesn't exceed image dimensions 
        if margin_pixels[0] + margin_pixels[2] >= image_width:  # Left + Right
            raise ValueError("Horizontal margins exceed image width")
        if margin_pixels[1] + margin_pixels[3] >= image_height:  # Top + Bottom  
            raise ValueError("Vertical margins exceed image height")

        return tuple(margin_pixels)  # Returns (left, top, right, bottom)

    def sample_annotations(self, method, num_annotations, annotation_size, margins, image_width, image_height):
        """Sample annotations using specified method."""
        # Validate margins first
        margins = self.validate_margins(margins, image_width, image_height)
        
        # Extract the validated margins
        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        annotations = []

        if method == "Random":
            for _ in range(num_annotations):
                x = random.randint(margin_x_min, image_width - annotation_size - margin_x_max)
                y = random.randint(margin_y_min, image_height - annotation_size - margin_y_max)
                annotations.append((x, y, annotation_size))

        if method == "Uniform":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(i * x_step + annotation_size / 2)
                    y = margin_y_min + int(j * y_step + annotation_size / 2)
                    annotations.append((x, y, annotation_size))

        if method == "Stratified Random":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(
                        i * x_step + random.uniform(annotation_size / 2, x_step - annotation_size / 2))
                    y = margin_y_min + int(
                        j * y_step + random.uniform(annotation_size / 2, y_step - annotation_size / 2))
                    annotations.append((x, y, annotation_size))

        return annotations

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
            margins = self.validate_margins(
                raw_margins,
                self.annotation_window.image_pixmap.width(),
                self.annotation_window.image_pixmap.height()
            )
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
                margins = self.validate_margins(raw_margins, width, height)
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
