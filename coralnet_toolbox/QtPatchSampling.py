import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QDialog, QHBoxLayout,
                             QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QGroupBox, QGraphicsRectItem, QMessageBox, QLabel)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtWorkArea import WorkArea
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchGraphic(QGraphicsRectItem):
    def __init__(self, x, y, size, color, parent=None):
        super().__init__(x, y, size, size, parent)
        self.base_color = color

        # --- Animation Properties ---
        self.animation_manager = None
        self.is_animating = False

        self._pulse_alpha = 128
        self._pulse_direction = 1

        self.default_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 50))
        self.setPen(self._create_pen())
        self.setBrush(self.default_brush)
        
    def set_animation_manager(self, manager):
        """Set the animation manager for this graphic."""
        self.animation_manager = manager
        
    def is_graphics_item_valid(self):
        """Check if the graphics item is still valid (not deleted)."""
        try:
            return self.scene() is not None
        except RuntimeError:
            return False

    def _create_pen(self):
        """Create a pulsing dotted pen with brighter color."""
        # Use a lighter version of the base color for better visibility
        pen_color = QColor(self.base_color)
        pen_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
        pen = QPen(pen_color, 4)  # Increased width
        pen.setCosmetic(True)
        pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
        return pen
    
    def tick_animation(self):
        """Update the pulse alpha for a heartbeat-like effect."""
        if self._pulse_direction == 1:
            self._pulse_alpha += 30
        else:
            self._pulse_alpha -= 10

        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50
            self._pulse_direction = 1

        self.setPen(self._create_pen())
    
    def _update_pulse_alpha(self):
        """Update the pulse alpha for a heartbeat-like effect: quick rise, slow fall."""
        if self._pulse_direction == 1:
            # Quick increase (systole-like)
            self._pulse_alpha += 30
        else:
            # Slow decrease (diastole-like)
            self._pulse_alpha -= 10  # <-- Corrected from += to -=

        # Check direction before clamping to ensure smooth transition
        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255  # Clamp to max
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50   # Clamp to min
            self._pulse_direction = 1
        
        self.setPen(self._create_pen())
        
    def animate(self):
        """Start animating the graphic."""
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)

    def deanimate(self):
        """Stop animating the graphic."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
        self._pulse_alpha = 128
        self.setPen(self._create_pen())
            
    def __del__(self):
        """Clean up when the graphic is deleted."""
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()
        if hasattr(self, 'animation_timer') and self.animation_timer: # Keep for old instances
            self.animation_timer.stop()


class PatchSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        
        self.animation_manager = self.annotation_window.animation_manager

        self.setWindowTitle("Sample Annotations")
        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.layout = QVBoxLayout(self)

        # Setup the info/instructions layout
        self.setup_info_layout()
        # Setup the sampling configuration layout
        self.setup_sampling_config_layout()
        # Setup the propagation and exclusion layout
        self.setup_propagation_exclusion_layout()
        # Setup the annotation configuration layout
        self.setup_annotation_config_layout()
        # Setup the bottom button controls
        self.setup_buttons_layout()

        self.sampled_annotations = []

        # Initialize graphics list
        self.annotation_graphics = []
        # Add margin work area attribute
        self.margin_work_area = None
        
        # Add status label for highlighted images count
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.status_label)
        
        # Connect to table model signals to update highlighted count when rows are highlighted
        self.image_window.table_model.rowsChanged.connect(self.update_status_label)
        
        # Connect to image selection changes to update preview when user switches images
        self.image_window.imageSelected.connect(self.on_image_changed)
        
    def setup_info_layout(self):
        """
        Set up the info layout with explanatory text.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Specify your sampling parameters below and highlight rows within the ImageWindow to sample."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_sampling_config_layout(self):
        """Set up the core sampling method and count configuration."""
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

    def setup_propagation_exclusion_layout(self):
        """Set up the propagation and exclusion options configuration."""
        group_box = QGroupBox("Propagation & Exclusion")
        layout = QFormLayout()

        # Sample Label
        self.label_combo = QComboBox()
        for label in self.label_window.labels:
            self.label_combo.addItem(label.short_label_code, label.id)
        self.label_combo.setCurrentIndex(0)
        self.label_combo.currentIndexChanged.connect(self.preview_annotations)
        layout.addRow("Sample As:", self.label_combo)

        # Propagate Labels
        self.propagate_labels_combo = QComboBox()
        self.propagate_labels_combo.addItems(["False", "True"])
        self.propagate_labels_combo.setCurrentIndex(0)
        self.propagate_labels_combo.currentIndexChanged.connect(self.preview_annotations)
        self.propagate_labels_combo.currentIndexChanged.connect(self.on_propagate_labels_changed)
        layout.addRow("Propagate Labels:", self.propagate_labels_combo)

        # Exclude Regions
        self.exclude_regions_combo = QComboBox()
        self.exclude_regions_combo.addItems(["False", "True"])
        self.exclude_regions_combo.setCurrentIndex(0)
        self.exclude_regions_combo.currentIndexChanged.connect(self.preview_annotations)
        self.exclude_regions_combo.currentIndexChanged.connect(self.on_exclude_regions_changed)
        layout.addRow("Avoid Annotations:", self.exclude_regions_combo)

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
        super().showEvent(event)
        
        # Automatically highlight the current image if one is loaded
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            # Check if current image is already highlighted
            highlighted_paths = self.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                # Highlight only the current image
                self.image_window.table_model.set_highlighted_paths([current_image_path])
        
        self.update_label_combo()
        self.update_status_label()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.clear_annotation_graphics()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.clear_annotation_graphics()
        super().reject()
        
    def update_label_combo(self):
        """Update the label combo box with the current labels."""
        self.label_combo.clear()
        for label in self.label_window.labels:
            self.label_combo.addItem(label.short_label_code, label.id)
        self.label_combo.setCurrentIndex(0)
    
    def update_status_label(self):
        """Update the status label to show the number of images highlighted."""
        highlighted_paths = self.image_window.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        if count == 0:
            self.status_label.setText("No images highlighted")
        elif count == 1:
            self.status_label.setText("1 image highlighted")
        else:
            self.status_label.setText(f"{count} images highlighted")
    
    def on_image_changed(self, image_path):
        """Handle when the user changes the selected image in the ImageWindow.
        
        When annotation_window.set_image() is called, it clears the scene, which removes
        all graphics. We need to clear our stale references and then regenerate the preview
        if one was being shown.
        """
        # Store whether we had preview graphics before the image changed
        had_preview = len(self.annotation_graphics) > 0
        
        # Clear stale graphics references since annotation window just cleared its scene
        self.annotation_graphics = []
        self.margin_work_area = None
        
        # Only regenerate preview if we had one shown and new image is valid
        if had_preview and image_path and self.annotation_window.active_image:
            self.preview_annotations()
        
    def on_propagate_labels_changed(self, idx):
        """Handle changes to the propagate labels combo box."""
        propagate = self.propagate_labels_combo.currentText() == "True"
        if propagate:
            # turn off avoid‐regions
            self.exclude_regions_combo.setCurrentIndex(0)
            self.exclude_regions_combo.setDisabled(True)
        else:
            self.exclude_regions_combo.setDisabled(False)
        self.preview_annotations()

    def on_exclude_regions_changed(self, idx):
        """Handle changes to the exclude regions combo box."""
        exclude = self.exclude_regions_combo.currentText() == "True"
        if exclude:
            # turn off propagate
            self.propagate_labels_combo.setCurrentIndex(0)
            self.propagate_labels_combo.setDisabled(True)
        else:
            self.propagate_labels_combo.setDisabled(False)
        self.preview_annotations()

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
        propagate = self.propagate_labels_combo.currentText() == "True"
        exclude_regions = False if propagate else (self.exclude_regions_combo.currentText() == "True")
    
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
        
        # Animate the margin work area
        self.margin_work_area.set_animation_manager(self.animation_manager)
        
        # Create graphics using the WorkArea's own method
        margin_graphics = self.margin_work_area.create_graphics(self.annotation_window.scene, include_shadow=True)
        
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
    
        # Create graphics for each annotation, using propagated label color if needed
        image_annotations = self.annotation_window.get_image_annotations()
        for x, y, size in self.sampled_annotations:
            if propagate:
                center = QPointF(x + size / 2, y + size / 2)
                # find annotation whose polygon contains the center
                found = next(
                    (
                        a for a in image_annotations
                        if a.get_polygon().containsPoint(center, Qt.OddEvenFill) and
                        (isinstance(a, PolygonAnnotation) or isinstance(a, RectangleAnnotation))
                    ),
                    None
                )
                color = found.label.color if found else sample_label.color
            else:
                color = sample_label.color
                
            graphic = PatchGraphic(x, y, size, color)
            graphic.set_animation_manager(self.animation_manager)
            graphic.animate()
            
            self.annotation_window.scene.addItem(graphic)
            self.annotation_graphics.append(graphic)
        self.annotation_window.viewport().update()

    def preview_annotations(self):
        """Preview sampled annotations."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.update_annotation_graphics()
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", f"Error previewing annotations: {str(e)}")
            return
        
        finally:
            # Restore cursor to default
            QApplication.restoreOverrideCursor()

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

        # Gets the label from LabelWindow
        sample_label = self.label_window.get_label_by_short_code(self.label_combo.currentText())
        if not sample_label:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", "Selected label not found")
            return

        # Get highlighted image paths
        image_paths = self.image_window.table_model.get_highlighted_paths()
        if not image_paths:
            QApplication.restoreOverrideCursor()
            msg = "Please highlight at least one image row to apply annotations to highlighted images."
            QMessageBox.warning(self, "No Selection", msg)
            return

        # Prepare flags
        propagate = self.propagate_labels_combo.currentText() == "True"
        exclude_regions = False if propagate else (self.exclude_regions_combo.currentText() == "True")

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths) * num_annotations)

        try:
            sampled_annotations = []  # Initialize ONCE outside the loop

            for image_path in image_paths:

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
                    # Determine label based on propagation
                    used_label = sample_label  # Default to the selected sample label
                    if propagate:
                        center = QPointF(x + size // 2, y + size // 2)
                        
                        # First, check the MaskAnnotation for label propagation 
                        # (since masks and vectors don't overlap, this is safe)
                        mask_annotation = self.annotation_window.current_mask_annotation
                        if mask_annotation and image_path == mask_annotation.image_path:
                            class_id = mask_annotation.get_class_at_point(center)
                            if class_id > 0:  # Valid class ID (not background)
                                mask_label = mask_annotation.class_id_to_label_map.get(class_id)
                                if mask_label:
                                    used_label = mask_label
                        # Note: No need to check vectors here if mask provided a label, as they don't overlap
                        
                        # If no mask label (or no mask), check vector annotations
                        if used_label == sample_label:  # Only check vectors if mask didn't provide a label
                            existing = self.annotation_window.get_image_annotations(image_path)
                            found = next(
                                (
                                    a for a in existing
                                    if a.get_polygon().containsPoint(center, Qt.OddEvenFill)
                                ),
                                None
                            )
                            if found:
                                used_label = found.label
        
                    # Create the annotation with the determined label
                    new_annotation = PatchAnnotation(
                        QPointF(x + size // 2, y + size // 2),
                        size,
                        used_label.short_label_code,
                        used_label.long_label_code,
                        used_label.color,
                        image_path,
                        used_label.id,
                        transparency=self.main_window.get_transparency_value()
                    )
                    sampled_annotations.append(new_annotation)  # Appends to the SHARED list
                    progress_bar.update_progress()
                
                # Update the raster's annotation info for each processed image
                self.image_window.update_image_annotations(image_path)

            # Add sampled annotations to the annotation window
            for sampled_annotation in sampled_annotations:
                self.annotation_window.add_annotation(sampled_annotation)

            # Refresh the view of the annotation window if the current image has new sampled annotations
            if image_paths and self.annotation_window.current_image_path in image_paths:
                self.annotation_window.load_annotations(self.annotation_window.current_image_path)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", f"Error adding sampled annotations: {str(e)}")
            raise e
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

        self.accept()

    def clear_annotation_graphics(self):
        """Remove all annotation preview graphics, including margin visualizations."""
        for graphic in self.annotation_graphics:
            if hasattr(graphic, 'deanimate'):
                graphic.deanimate()
            self.annotation_window.scene.removeItem(graphic)
        self.annotation_graphics = []

        if self.margin_work_area is not None:
            self.margin_work_area.remove_from_scene()
            self.margin_work_area = None
        self.annotation_window.viewport().update()
