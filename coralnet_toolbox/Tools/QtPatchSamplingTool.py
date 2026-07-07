import warnings


import random
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygonF, QMouseEvent, QFont, QPainter
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QDialog, QHBoxLayout,
                             QPushButton, QComboBox, QSpinBox, QMessageBox, QLabel,
                             QFormLayout, QGroupBox, QGraphicsRectItem)

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.WorkArea import WorkArea
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon, get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchGraphic(QGraphicsRectItem):
    def __init__(self, x, y, size, color, label_text, parent=None):
        super().__init__(x, y, size, size, parent)
        self.base_color = color
        self.label_text = label_text

        self.default_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 50))
        self.setBrush(self.default_brush)
        self._update_pen()
        

    def is_graphics_item_valid(self):
        try:
            return self.scene() is not None
        except RuntimeError:
            return False

    def _update_pen(self):
        """Create a marching ants dashed pen."""
        pen = QPen(self.base_color, 2)
        pen.setCosmetic(True)
        # PyQt uses setDashPattern; provide floats for compatibility
        pen.setDashPattern([4.0, 4.0])
        self.setPen(pen)
    



    def paint(self, painter, option, widget=None):
        """Draw the marching ants rectangle and the floating nametag."""
        # 1. Draw the base rectangle using QGraphicsRectItem's native logic
        super().paint(painter, option, widget)
        
        # 2. Draw the floating nametag
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 6, QFont.Bold)
        painter.setFont(font)
        
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(self.label_text)
        text_height = fm.height()
        
        pad_x, pad_y = 4, 2
        
        # Position at top-left, slightly inside the patch
        r = self.rect()
        bg_rect = QRectF(r.left() + 2, r.top() + 2, text_width + pad_x * 2, text_height + pad_y * 2)
        
        # Opaque background using the label's color
        bg_color = QColor(self.base_color)
        bg_color.setAlpha(255)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(bg_rect, 4, 4)
        
        # Draw the text significantly darker than the label color for high contrast
        text_color = bg_color.darker(150) 
        painter.setPen(text_color)
        painter.drawText(bg_rect, Qt.AlignCenter, self.label_text)
    
    def itemChange(self, change, value):
        return super().itemChange(change, value)
            


class PatchSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.tool = tool
        
        self.annotation_window = tool.annotation_window
        self.main_window = tool.annotation_window.main_window
        self.label_window = tool.annotation_window.main_window.label_window
        self.image_window = tool.annotation_window.main_window.image_window

        # Multi-Annotate (MVAT) integration
        self.mvat_manager = getattr(self.main_window, 'mvat_manager', None)
        
        self.setWindowTitle("Sample Annotations")
        self.setWindowIcon(get_window_icon("coralnet.svg"))
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

        # Refresh the status hint whenever Multi-Annotate is toggled (guarded:
        # context_matrix may not exist in all configurations).
        context_matrix = getattr(self.main_window, 'context_matrix', None)
        if context_matrix is not None and hasattr(context_matrix, 'multiAnnotateToggled'):
            try:
                context_matrix.multiAnnotateToggled.connect(lambda _enabled: self.update_status_label())
            except Exception:
                pass

    def _multi_annotate_on(self) -> bool:
        """Return True when MVAT Multi-Annotate mode is currently enabled."""
        return bool(getattr(self.mvat_manager, 'multi_annotate_enabled', False))

    def _visible_camera_paths(self) -> list:
        """Return the visible context-camera paths, or [] when MVAT is unavailable."""
        if self.mvat_manager is None:
            return []
        try:
            return list(self.mvat_manager._get_visible_context_camera_paths())
        except Exception:
            return []

    def setup_info_layout(self):
        """
        Set up the info layout with explanatory text.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Specify your sampling parameters below and highlight rows within the ImageWindow to sample.\n"
            "Draw a rectangle to select the area for sampling."
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
        self.method_combo.setToolTip("Random: Unbiased sampling.\nStratified Random: Equal distribution across categories.\nUniform: Grid-based even spacing.")
        layout.addRow("Sampling Method:", self.method_combo)

        # Number of Annotations
        self.num_annotations_spinbox = QSpinBox()
        self.num_annotations_spinbox.setMinimum(1)
        self.num_annotations_spinbox.setMaximum(10000)
        self.num_annotations_spinbox.setValue(10)
        self.num_annotations_spinbox.valueChanged.connect(self.preview_annotations)
        self.num_annotations_spinbox.setToolTip("Number of annotations to generate in the selected region(s).")
        layout.addRow("Number of Annotations:", self.num_annotations_spinbox)

        # Annotation Size
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(32)
        self.annotation_size_spinbox.setMaximum(10000)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.preview_annotations)
        self.annotation_size_spinbox.setToolTip("Patch size in pixels (width/height). Larger patches capture more context.")
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
        self.label_combo.setToolTip("Label to assign to all sampled patches.")
        layout.addRow("Sample As:", self.label_combo)

        # Propagate Labels
        self.propagate_labels_combo = QComboBox()
        self.propagate_labels_combo.addItems(["False", "True"])
        self.propagate_labels_combo.setCurrentIndex(0)
        self.propagate_labels_combo.currentIndexChanged.connect(self.preview_annotations)
        self.propagate_labels_combo.currentIndexChanged.connect(self.on_propagate_labels_changed)
        self.propagate_labels_combo.setToolTip("Copy the label to all highlighted rows.")
        layout.addRow("Propagate Labels:", self.propagate_labels_combo)

        # Exclude Regions
        self.exclude_regions_combo = QComboBox()
        self.exclude_regions_combo.addItems(["False", "True"])
        self.exclude_regions_combo.setCurrentIndex(0)
        self.exclude_regions_combo.currentIndexChanged.connect(self.preview_annotations)
        self.exclude_regions_combo.currentIndexChanged.connect(self.on_exclude_regions_changed)
        self.exclude_regions_combo.setToolTip("Avoid sampling over existing annotations.\nPrevents overlap with already-labeled regions.")
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
        self.preview_button.setToolTip("Show a preview of annotations using current parameters.")
        button_layout.addWidget(self.preview_button)

        # Accept Button
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept_annotations)
        self.accept_button.setToolTip("Create and apply all sampled annotations.")
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
        self.cleanup()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.cleanup()
        super().reject()
        
    def cleanup(self):
        """Clean up temporary graphics, reset UI to defaults, and deactivate tool."""
        # Reset spinboxes and combos to default values
        self.method_combo.setCurrentIndex(0)  # Random
        self.num_annotations_spinbox.setValue(10)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.label_combo.setCurrentIndex(0)
        self.propagate_labels_combo.setCurrentIndex(0)  # False
        self.exclude_regions_combo.setCurrentIndex(0)  # False
        
        # Temporarily disconnect margin input signals to prevent triggering preview during reset
        for spin in self.margin_input.margin_spins:
            try:
                spin.valueChanged.disconnect(self.preview_annotations)
            except TypeError:
                pass  # Already disconnected
        for double in self.margin_input.margin_doubles:
            try:
                double.valueChanged.disconnect(self.preview_annotations)
            except TypeError:
                pass  # Already disconnected
        
        # Reset margin inputs to 0
        for spin in self.margin_input.margin_spins:
            spin.setValue(0)
        for double in self.margin_input.margin_doubles:
            double.setValue(0.0)
            
        # Clear temporary graphics
        self.clear_graphics()
        
        # Deactivate the tool
        self.tool.deactivate()
        
        # Untoggle all tools in the main window
        self.main_window.untoggle_all_tools()
        
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
            base_text = "No images highlighted"
        elif count == 1:
            base_text = "1 image highlighted"
        else:
            base_text = f"{count} images highlighted"

        # When Multi-Annotate is ON, hint that sampled patches will be propagated
        # to the visible context cameras.
        if self._multi_annotate_on():
            base_text += " | Multi-Annotate ON"
        else:
            base_text += " | Multi-Annotate OFF"

        self.status_label.setText(base_text)
        
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
        self.clear_graphics()
    
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
        
        
        # Create graphics using the WorkArea's own method
        margin_graphics = self.margin_work_area.create_graphics(self.annotation_window.scene, 
                                                                include_shadow=True, 
                                                                image_rect=self.annotation_window.get_image_rect())
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
    
        # Create graphics for each annotation, using propagated label if needed
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
                used_label = found.label if found else sample_label
            else:
                used_label = sample_label
                
            # --- Pass the color AND the short label code ---
            graphic = PatchGraphic(x, y, size, used_label.color, used_label.short_label_code)
            # -----------------------------------------------
            
            
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
        self.cleanup()
        self.accept()

    def _prompt_multi_annotate_mode(self):
        """Ask the user how to sample when Multi-Annotate is ON and multiple images
        are highlighted.

        Returns:
            'A' — sample on the first highlighted image only, propagate to cameras.
            'B' — sample on every highlighted image, propagate each to cameras.
            None — the user cancelled.
        """
        n_cameras = len(self._visible_camera_paths())
        n_images = len(self.image_window.table_model.get_highlighted_paths())
        num = self.num_annotations_spinbox.value()

        box = QMessageBox(self)
        box.setWindowIcon(get_window_icon("coralnet.svg"))
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Multi-Annotate Sampling")
        box.setText("Multi-Annotate is enabled — how should patches be sampled?")
        box.setInformativeText(
            f"You have <b>{n_images} images highlighted</b> and "
            f"<b>{n_cameras} visible context camera(s)</b>.<br><br>"

            f"<b>Sample the first image only</b><br>"
            f"Sample {num} patch(es) on just the first highlighted image, then "
            f"project each patch onto all {n_cameras} visible camera(s) as a linked "
            f"group. Use this when your highlighted images are different views of "
            f"the same scene and you only need to place points once.<br><br>"

            f"<b>Sample all images &amp; propagate</b><br>"
            f"Sample {num} patch(es) on <i>every one</i> of the {n_images} highlighted "
            f"images, and <i>also</i> project each image's patches onto all "
            f"{n_cameras} visible camera(s). This creates many more annotations "
            f"(roughly {n_images} × {num}, plus their projected copies).<br><br>"

            f"<b>Cancel</b><br>"
            f"Do nothing. The tool stays open so you can turn Multi-Annotate off "
            f"if you did not mean to propagate."
        )

        first_button = box.addButton("Sample first image only", QMessageBox.AcceptRole)
        first_button.setToolTip(
            "Sample on the first highlighted image, then copy each patch to every "
            "visible context camera as a linked group."
        )
        each_button = box.addButton("Sample all images && propagate", QMessageBox.AcceptRole)
        each_button.setToolTip(
            "Sample on every highlighted image, then copy each image's patches to "
            "all visible context cameras as linked groups."
        )
        cancel_button = box.addButton("Cancel", QMessageBox.RejectRole)
        cancel_button.setToolTip("Do not sample. The tool stays active so you can "
                                 "toggle Multi-Annotate off if needed.")

        box.exec_()
        clicked = box.clickedButton()
        if clicked is first_button:
            return 'A'
        if clicked is each_button:
            return 'B'
        return None

    def add_sampled_annotations(self, method, num_annotations, annotation_size):
        """Add the sampled annotations to the current image."""
        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Clear the graphics
        self.clear_graphics()

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

        # ── Multi-Annotate resolution ─────────────────────────────────────────
        # When Multi-Annotate is ON and there are visible context cameras, sampled
        # patches are propagated to those cameras as linked shared-id groups. When
        # more than one image is highlighted, ask the user which images to sample.
        mvat_active = self._multi_annotate_on() and bool(self._visible_camera_paths())
        source_paths = image_paths  # default: sample on every highlighted image
        if mvat_active and len(image_paths) > 1:
            mode = self._prompt_multi_annotate_mode()
            if mode is None:
                # Cancelled — leave the tool active so the user can toggle MVAT off.
                QApplication.restoreOverrideCursor()
                return
            if mode == 'A':
                source_paths = [image_paths[0]]  # first image only
            # mode == 'B' → sample on all highlighted images (source_paths unchanged)

        # Prepare flags
        propagate = self.propagate_labels_combo.currentText() == "True"
        exclude_regions = False if propagate else (self.exclude_regions_combo.currentText() == "True")

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(source_paths) * num_annotations)

        try:
            sampled_annotations = []  # Initialize ONCE outside the loop

            # Hoist per-run constant: transparency is the same for every patch.
            transparency = self.main_window.get_transparency_value()

            for image_path in source_paths:

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

                # Precompute the existing-annotation polygons ONCE per image. The
                # same list serves both the exclusion test and the label-propagation
                # lookup below (exclusion and propagation are mutually exclusive in
                # the UI). Building QPolygonF here — instead of inside the per-patch
                # loop — turns O(patches × annotations) polygon rebuilds into O(annotations).
                existing_polys = []  # list[(annotation, QPolygonF)]
                if exclude_regions or propagate:
                    image_annotations = self.annotation_window.get_image_annotations(image_path)
                    existing_polys = [(a, a.get_polygon()) for a in image_annotations]

                polygons = [poly for _a, poly in existing_polys] if exclude_regions else []

                # Resolve the mask annotation for this image once (used by label
                # propagation). mask_active guards the per-patch class lookup.
                mask_annotation = self.annotation_window.current_mask_annotation
                mask_active = bool(propagate and mask_annotation and
                                   image_path == mask_annotation.image_path)

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
                        if mask_active:
                            class_id = mask_annotation.get_class_at_point(center)
                            if class_id > 0:  # Valid class ID (not background)
                                mask_label = mask_annotation.class_id_to_label_map.get(class_id)
                                if mask_label:
                                    used_label = mask_label

                        # If no mask label (or no mask), check vector annotations
                        # against the precomputed polygons (no per-patch rebuild).
                        if used_label == sample_label:  # Only check vectors if mask didn't provide a label
                            for a, poly in existing_polys:
                                if poly.containsPoint(center, Qt.OddEvenFill):
                                    used_label = a.label
                                    break

                    # Create the annotation with the determined label
                    new_annotation = PatchAnnotation(
                        QPointF(x + size // 2, y + size // 2),
                        size,
                        used_label,
                        image_path,
                        transparency=transparency,
                        show_confidence=False
                    )
                    sampled_annotations.append(new_annotation)  # Appends to the SHARED list
                    progress_bar.update_progress()

                # Update the raster's annotation info for each processed image
                self.image_window.update_image_annotations(image_path)
                # TODO Check if we can move this outside the loop, and do it per image, instead of per annotation

            # Multi-Annotate: project each source patch into the visible context
            # cameras and stamp shared-id groups. Siblings are added together with
            # the source patches in a single bulk insert (one undo entry, one UI
            # refresh). This must happen BEFORE add_annotations so the shared_id is
            # already stamped on the source patches when they are inserted.
            sibling_annotations = []
            if mvat_active and sampled_annotations:
                try:
                    engine = self.annotation_window._shared_group_propagation_engine()
                    if engine is not None and hasattr(engine, 'build_sampled_patch_siblings'):
                        # Re-purpose the progress bar for the (slower) propagation
                        # phase so the user sees that work is still happening.
                        progress_bar.set_title("Propagating to visible cameras...")
                        progress_bar.start_progress(len(sampled_annotations))
                        sibling_annotations = engine.build_sampled_patch_siblings(
                            sampled_annotations, progress_bar=progress_bar
                        )
                except Exception as e:
                    print(f"Warning: Multi-Annotate propagation failed: {e}")
                    sibling_annotations = []

            # Final insert phase can also be slow for large batches — signal it.
            all_count = len(sampled_annotations) + len(sibling_annotations)
            if all_count:
                progress_bar.set_title(f"Adding {all_count} annotations...")
                progress_bar.set_busy_mode()

            # Add all sampled annotations (source + propagated siblings) in one BULK operation
            all_annotations = sampled_annotations + sibling_annotations
            if all_annotations:
                self.annotation_window.add_annotations(all_annotations, record_action=True)

                # Update annotation info for any sibling images touched by propagation
                sibling_paths = {a.image_path for a in sibling_annotations}
                for sib_path in sibling_paths:
                    self.image_window.update_image_annotations(sib_path)

                # --- PHANTOM ARCHITECTURE UPDATE ---
                # We NO LONGER force the creation of Qt graphics items here.
                # Instead, render them as sleeping phantoms using the fast readonly pass.
                affected_paths = set(source_paths) | sibling_paths
                if self.annotation_window.current_image_path in affected_paths:
                    self.annotation_window.refresh_phantom_annotations()
                    self.annotation_window.viewport().update()
                # --------------------------------------------------------------------------

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Error", f"Error adding sampled annotations: {str(e)}")
            raise e
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

    def update_margins_from_rectangle(self, rect):
        """Update margin inputs based on a drawn rectangle.
        
        Args:
            rect (QRectF): Rectangle drawn by the user in scene coordinates
        """
        if not self.annotation_window.pixmap_image:
            return
            
        # Get image dimensions
        image_width = self.annotation_window.pixmap_image.width()
        image_height = self.annotation_window.pixmap_image.height()
        
        # Calculate margins from rectangle bounds
        left = int(rect.left())
        top = int(rect.top())
        right = int(image_width - rect.right())
        bottom = int(image_height - rect.bottom())
        
        # Clamp to valid ranges
        left = max(0, min(left, image_width))
        top = max(0, min(top, image_height))
        right = max(0, min(right, image_width))
        bottom = max(0, min(bottom, image_height))
        
        # Switch to Multiple Values mode if not already
        if self.margin_input.type_combo.currentIndex() != 1:
            self.margin_input.type_combo.setCurrentIndex(1)
            
        # Switch to Pixels mode if not already
        if self.margin_input.value_type.currentIndex() != 0:
            self.margin_input.value_type.setCurrentIndex(0)
        
        # Update margin values (order: Top, Right, Bottom, Left)
        self.margin_input.margin_spins[0].setValue(top)     # Top
        self.margin_input.margin_spins[1].setValue(right)   # Right
        self.margin_input.margin_spins[2].setValue(bottom)  # Bottom
        self.margin_input.margin_spins[3].setValue(left)    # Left
        
        # Automatically trigger preview
        self.preview_annotations()

    def clear_graphics(self):
        """Remove all annotation preview graphics, including margin visualizations."""
        for graphic in self.annotation_graphics:
            # Remove from scene if it belongs to one
            if graphic.scene():
                graphic.scene().removeItem(graphic)
        self.annotation_graphics.clear()
        self.annotation_graphics = []

        if self.margin_work_area is not None:
            self.margin_work_area.remove_from_scene()
            self.margin_work_area = None
        self.annotation_window.viewport().update()
        
        self.sampled_annotations = []


class PatchSamplingTool(Tool):
    """
    Tool for sampling patch annotations with interactive rectangle drawing to define margins.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.show_crosshair = True
        
        # Create the dialog (owned by the tool)
        self.dialog = PatchSamplingDialog(self, annotation_window)
        
        # Drawing state
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.rectangle_graphic = None

    def activate(self):
        """Activate the patch sampling tool"""
        super().activate()
        
        # Show the dialog
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def deactivate(self):
        """Deactivate the patch sampling tool"""
        if not self.active:
            return
            
        # Stop any current drawing
        self.stop_current_drawing()
        
        super().deactivate()
        
        # Hide the dialog
        self.dialog.hide()
        
        # Clear all graphics
        self.dialog.clear_graphics()

    def stop_current_drawing(self):
        """Stop current rectangle drawing operation"""
        if self.is_drawing:
            self.is_drawing = False
            self.start_point = None
            self.end_point = None
            
            # Remove rectangle graphic if exists
            if self.rectangle_graphic:
                if self.rectangle_graphic.scene():
                    self.annotation_window.scene.removeItem(self.rectangle_graphic)
                self.rectangle_graphic = None

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press - start drawing rectangle"""
        # Return early if tool is not active or dialog is not visible
        if not self.active or not self.dialog.isVisible():
            return
            
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Check if cursor is in the image bounds
            if not self.annotation_window.cursorInWindow(event.pos()):
                return
                
            if not self.is_drawing:
                # Start drawing
                self.is_drawing = True
                self.start_point = scene_pos
                self.end_point = scene_pos
                
                # Create rectangle graphic
                self._create_rectangle_graphic()
            else:
                # Finish drawing
                self.end_point = scene_pos
                self._finalize_rectangle()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move - update rectangle preview and crosshair"""
        # Return early if tool is not active or dialog is not visible
        if not self.active or not self.dialog.isVisible():
            return
            
        # Call parent to handle crosshair
        super().mouseMoveEvent(event)
        
        if self.is_drawing:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.end_point = scene_pos
            self._update_rectangle_graphic()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        pass

    def _create_rectangle_graphic(self):
        """Create the rectangle graphic for visual feedback"""
        if self.rectangle_graphic:
            if self.rectangle_graphic.scene():
                self.annotation_window.scene.removeItem(self.rectangle_graphic)
                
        # Create a semi-transparent rectangle
        rect = QRectF(self.start_point, self.end_point).normalized()
        self.rectangle_graphic = QGraphicsRectItem(rect)
        
        # Style the rectangle
        pen = QPen(QColor(0, 168, 230), 2, Qt.DashLine)
        pen.setCosmetic(True)
        brush = QBrush(QColor(0, 168, 230, 30))
        
        self.rectangle_graphic.setPen(pen)
        self.rectangle_graphic.setBrush(brush)
        self.rectangle_graphic.setZValue(1000)
        
        self.annotation_window.scene.addItem(self.rectangle_graphic)

    def _update_rectangle_graphic(self):
        """Update the rectangle graphic as user drags"""
        if self.rectangle_graphic and self.start_point and self.end_point:
            rect = QRectF(self.start_point, self.end_point).normalized()
            self.rectangle_graphic.setRect(rect)

    def _finalize_rectangle(self):
        """Finalize the drawn rectangle and update dialog margins"""
        if not self.start_point or not self.end_point:
            self.stop_current_drawing()
            return
            
        # Get the normalized rectangle
        rect = QRectF(self.start_point, self.end_point).normalized()
        
        # Remove the drawing graphic
        if self.rectangle_graphic:
            if self.rectangle_graphic.scene():
                self.annotation_window.scene.removeItem(self.rectangle_graphic)
            self.rectangle_graphic = None
        
        # Reset drawing state
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        # Update dialog margins from the drawn rectangle
        self.dialog.update_margins_from_rectangle(rect)
