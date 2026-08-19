import warnings
import os
import ujson as json

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageColor

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QTableWidgetItem,
                             QButtonGroup, QWidget, QTableWidget, QHeaderView,
                             QAbstractItemView, QSpinBox, QRadioButton, QColorDialog,
                             QScrollArea, QSizePolicy)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon
from coralnet_toolbox.utilities import rasterio_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Classes
# ----------------------------------------------------------------------------------------------------------------------


class ColorSwatchWidget(QWidget):
    """A simple widget to display a color swatch with a border."""
    def __init__(self, color, parent=None):
        """Initialize the color swatch widget."""
        super().__init__(parent)
        self.color = color
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        """Paint the color swatch with border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set the brush for the fill color
        painter.setBrush(self.color)
        
        # Set the pen for the black border
        pen = QPen(QColor("black"))
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Draw the rectangle, adjusted inward so the border is fully visible
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def setColor(self, color):
        """Update the swatch's color and repaint."""
        self.color = color
        self.update()  # Triggers a repaint


class ClickableColorSwatchWidget(ColorSwatchWidget):
    """A ColorSwatchWidget that emits a clicked signal."""
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        """Handle mouse press to emit clicked signal."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ----------------------------------------------------------------------------------------------------------------------
# Main Dialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ExportMaskAnnotations(QDialog):
    def __init__(self, main_window):
        """Initialize the export mask annotations dialog."""
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("mask.svg"))
        self.setWindowTitle("Export Annotations to Masks")
        self.resize(850, 650)

        self.mask_mode = 'semantic'  # 'semantic', 'sfm', 'rgb', or 'overlay'
        self.rgb_background_color = QColor(0, 0, 0)

        # Main layout for the dialog
        self.main_layout = QVBoxLayout(self)

        # Top section
        top_section = QVBoxLayout()
        self.setup_info_layout(parent_layout=top_section)
        self.setup_output_layout(parent_layout=top_section)
        self.setup_mask_format_layout(parent_layout=top_section)
        self.main_layout.addLayout(top_section)

        # Middle section
        columns_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self.setup_annotation_layout(parent_layout=left_col)
        self.setup_label_layout(parent_layout=right_col)

        columns_layout.addLayout(left_col, 1)
        columns_layout.addLayout(right_col, 2)
        self.main_layout.addLayout(columns_layout)

        # Bottom buttons
        self.setup_buttons_layout(parent_layout=self.main_layout)

        # Set initial mode and update UI
        self.semantic_radio.setChecked(True)
        self.update_ui_for_mode()

    def showEvent(self, event):
        """Handle show event and update UI."""
        super().showEvent(event)
        self.update_ui_for_mode()

    def setup_info_layout(self, parent_layout=None):
        """Set up the information layout section."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_text = (
            "This tool exports annotations to image masks for four primary use cases:<br><br>"
            "<b>1. Semantic Segmentation (Integer IDs):</b> Creates masks where each class is represented by a "
            "unique integer. <br><i>Training Tip:</i> Set the background value to <b>255 (Ignore)</b> if your images "
            "are sparsely labeled (so the model isn't penalized for unlabeled objects). Set it to <b>0 (Background)</b> "
            "if your images are exhaustively labeled (to actively teach the model negative space).<br><br>"
            "<b>2. Structure from Motion (SfM) (Binary Mask):</b> Creates masks where a foreground value "
            "(e.g., 255) represents objects to keep, and a background value (e.g., 0) represents areas to "
            "ignore. This is used by software like Metashape to improve 3D model reconstruction.<br><br>"
            "<b>3. Visualization (RGB Colors):</b> Creates a human-readable color mask using the colors "
            "assigned to each label. Ideal for reports, presentations, and qualitative analysis.<br><br>"
            "<b>4. Overlay (Image + Color Mask):</b> Blends the color mask over the original image, using the "
            "current transparency value from the Annotation Window. This is the bulk equivalent of a screenshot. "
            "Unannotated areas keep their original pixels, and images with no annotations export as an unmodified "
            "copy of the source."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop)
        info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # This text is long enough that laying it out inline squashes everything below it, and
        # its wrapped height is not known until the dialog has a real width. A scroll area gives
        # it a fixed footprint, so the dialog opens at a stable size on any monitor or DPI.
        scroll_area = QScrollArea()
        scroll_area.setWidget(info_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumHeight(120)
        scroll_area.setMaximumHeight(160)

        layout.addWidget(scroll_area)
        group_box.setLayout(layout)
        group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        parent_layout.addWidget(group_box)

    def setup_output_layout(self, parent_layout=None):
        """Set up the output directory and format layout."""
        groupbox = QGroupBox("Output Directory and File Format")
        layout = QFormLayout()

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.output_dir_edit.setToolTip("Parent directory where the mask folder will be created.\nA new subdirectory with the specified folder name will be created here.")
        self.output_dir_button.setToolTip("Browse for a directory.")
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        self.output_name_edit = QLineEdit("masks")
        self.output_name_edit.setToolTip("Name of the subdirectory to create for the exported masks.\nMask files will be saved in: Output Directory / Folder Name /")
        layout.addRow("Folder Name:", self.output_name_edit)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_mask_format_layout(self, parent_layout=None):
        """Set up the mask format and options layout."""
        groupbox = QGroupBox("Export Mode and Format")
        main_layout = QVBoxLayout()

        # Mode Selection
        mode_layout = QGridLayout()
        self.semantic_radio = QRadioButton("Semantic Segmentation (Integer IDs)")
        self.semantic_radio.setToolTip("Each class gets a unique integer value (0-255).\nUse for machine learning training where each pixel needs a class label.")
        self.sfm_radio = QRadioButton("Structure from Motion (Binary Mask)")
        self.sfm_radio.setToolTip("Binary masks (foreground/background only).\nUse with Metashape or other SfM software for 3D reconstruction.")
        self.rgb_radio = QRadioButton("Visualization (RGB Colors)")
        self.rgb_radio.setToolTip("Colors based on label assignments.\nPerfect for visual inspection, reports, and presentations.")
        self.overlay_radio = QRadioButton("Overlay (Image + Color Mask)")
        self.overlay_radio.setToolTip("Blends the color mask over the original image using the current\n"
                                      "transparency value from the Annotation Window.\n"
                                      "Unannotated areas keep their original pixels.")

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.semantic_radio)
        self.mode_group.addButton(self.sfm_radio)
        self.mode_group.addButton(self.rgb_radio)
        self.mode_group.addButton(self.overlay_radio)
        self.mode_group.buttonClicked.connect(self.update_ui_for_mode)

        mode_layout.addWidget(self.semantic_radio, 0, 0)
        mode_layout.addWidget(self.sfm_radio, 0, 1)
        mode_layout.addWidget(self.rgb_radio, 1, 0)
        mode_layout.addWidget(self.overlay_radio, 1, 1)
        main_layout.addLayout(mode_layout)

        # Format and Options
        options_layout = QFormLayout()
        
        # General file format
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItem(".png", ".png")
        self.file_format_combo.addItem(".bmp", ".bmp")
        self.file_format_combo.addItem(".tif", ".tif")
        self.file_format_combo.addItem("RLE (.txt)", ".txt")
        self.file_format_combo.currentTextChanged.connect(self.update_georef_availability)
        self.file_format_combo.setToolTip("Output image format.\nPNG: Lossless, good compression. TIF: Supports georeferencing. RLE: Text-based encoding.")
        options_layout.addRow("File Format:", self.file_format_combo)

        # Georeferencing
        self.preserve_georef_checkbox = QCheckBox("Preserve georeferencing (if available)")
        self.preserve_georef_checkbox.setChecked(True)
        self.preserve_georef_checkbox.setToolTip("Save geographic coordinate information in GeoTIFF format.\nEnable if your source images have georeferencing and you need it in the masks.\nOnly available with TIF format.")
        options_layout.addRow(self.preserve_georef_checkbox)

        # Outlines, available to the color modes only (kept in the layout so the dialog does not resize)
        self.draw_outlines_checkbox = QCheckBox("Draw annotation outlines")
        self.draw_outlines_checkbox.setChecked(True)
        options_layout.addRow(self.draw_outlines_checkbox)

        main_layout.addLayout(options_layout)
        groupbox.setLayout(main_layout)
        parent_layout.addWidget(groupbox)
        self.update_georef_availability()

    def setup_annotation_layout(self, parent_layout=None):
        """Set up the annotations to include layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()
        
        self.mask_checkbox = QCheckBox("Mask Annotations (Base Layer)")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.setToolTip("Include manually painted mask annotations.\nThese form the base layer for all exports.")
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        self.patch_checkbox.setToolTip("Include point-based patch annotations.\nEach patch is converted to a circular region in the mask.")
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        self.rectangle_checkbox.setToolTip("Include bounding box rectangle annotations.\nEach rectangle is filled with the corresponding label value.")
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)
        self.polygon_checkbox.setToolTip("Include polygon annotations (manually drawn areas).\nPolygons provide precise, irregularly-shaped regions.")
        self.include_negative_samples_checkbox = QCheckBox("Include negative samples")
        self.include_negative_samples_checkbox.setChecked(True)
        self.include_negative_samples_checkbox.setToolTip("Export masks for images with NO annotations.\nUseful for training models to recognize negative examples.")

        layout.addWidget(self.mask_checkbox)
        layout.addWidget(self.patch_checkbox)
        layout.addWidget(self.rectangle_checkbox)
        layout.addWidget(self.polygon_checkbox)
        layout.addWidget(self.include_negative_samples_checkbox)
        
        info_text = QLabel(
            "Select which types of annotations to include in the export. <b>Mask Annotations</b> serve as the base "
            "layer, while vector annotations (patches, rectangles, polygons) are drawn on top. <b>'Include negative "
            "samples'</b> exports masks for images without annotations."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #666;")
        layout.addWidget(info_text)
        
        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_label_layout(self, parent_layout=None):
        """Set up the labels to include and rasterization order layout."""
        groupbox = QGroupBox("Labels to Include / Rasterization Order")
        layout = QVBoxLayout()
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(3)
        self.label_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.label_table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.label_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.label_table)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.move_up_button = QPushButton("▲ Move Up")
        self.move_down_button = QPushButton("▼ Move Down")
        self.move_up_button.clicked.connect(self.move_row_up)
        self.move_up_button.setToolTip("Move the selected label toward the top of this table.\nLabels nearer the top are painted first, so they end up underneath.")
        self.move_down_button.clicked.connect(self.move_row_down)
        self.move_down_button.setToolTip("Move the selected label toward the bottom of this table.\nLabels nearer the bottom are painted last, so they end up on top.")
        self.label_table.itemSelectionChanged.connect(self.update_move_buttons)
        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)
        
        order_note = QLabel(
            "<b>Rasterization Order:</b> Labels are painted in the order they appear in this table, "
            "starting at the top row and working down. A label nearer the <b>bottom of the table</b> is "
            "therefore painted over one nearer the top. Where two annotations overlap, the one whose "
            "label sits further down the table is the one you will see. Use Move Up / Move Down to "
            "reorder the rows; this is about row position, not the assigned mask value."
        )
        order_note.setStyleSheet("color: #666;")
        order_note.setWordWrap(True)
        layout.addWidget(order_note)
        
        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_buttons_layout(self, parent_layout=None):
        """Set up the buttons layout."""
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.run_export_process)
        self.export_button.setToolTip("Export annotations as masks to the specified output directory.\nCreates a new folder with the mask files in your selected format.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setToolTip("Close this dialog without exporting.")
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def _is_color_mode(self):
        """Return True for the modes that rasterize RGB colors rather than integer values."""
        return self.mask_mode in ('rgb', 'overlay')

    def get_transparency_value(self):
        """Get the current transparency (0-255) from the Annotation Window slider."""
        value = self.main_window.get_transparency_value()
        if value is None:
            value = 128
        return max(0, min(255, int(value)))

    def update_ui_for_mode(self):
        """Update the UI dynamically based on the selected export mode."""
        if self.semantic_radio.isChecked():
            self.mask_mode = 'semantic'
        elif self.sfm_radio.isChecked():
            self.mask_mode = 'sfm'
        elif self.rgb_radio.isChecked():
            self.mask_mode = 'rgb'
        elif self.overlay_radio.isChecked():
            self.mask_mode = 'overlay'

        # Outlines only make sense once labels are drawn in their own colors
        is_color = self._is_color_mode()
        if not is_color:
            self.draw_outlines_checkbox.setChecked(False)
        self.draw_outlines_checkbox.setEnabled(is_color)
        if self.mask_mode == 'overlay':
            transparency = self.get_transparency_value()
            percent = round(transparency / 255.0 * 100)
            self.draw_outlines_checkbox.setToolTip(
                "Draw a fully opaque border around each vector annotation, matching how they\n"
                "appear in the Annotation Window. Borders are drawn after the blend, so they\n"
                "stay crisp instead of fading with the fill.\n\n"
                f"Blend transparency: {transparency} ({percent}% opacity), taken from the\n"
                "transparency slider in the Annotation Window. Close this dialog to change it."
            )
        elif is_color:
            self.draw_outlines_checkbox.setToolTip(
                "Draw a border around each vector annotation, in its own label color.\n"
                "Borders are drawn after every fill, so the boundary of an annotation stays\n"
                "visible even where a later label is painted over it."
            )
        else:
            self.draw_outlines_checkbox.setToolTip(
                "Only available for the Visualization and Overlay modes, which draw labels\n"
                "in their own colors."
            )

        self.populate_label_table()
    
    def populate_label_table(self):
        """Populate the label table based on the current mode."""
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(0)

        # Set table headers based on mode
        headers = ["Include", "Label Name"]
        if self._is_color_mode():
            headers.append("Color Preview")
        else:
            headers.append("Mask Value")
        self.label_table.setHorizontalHeaderLabels(headers)

        # --- BACKGROUND ROW (ROW 0) ---
        self.label_table.insertRow(0)
        checkbox_widget = self.create_centered_checkbox(checked=True)
        self.label_table.setCellWidget(0, 0, checkbox_widget)
        
        label_item = QTableWidgetItem("background")
        label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
        label_item.setData(Qt.UserRole, "background")
        self.label_table.setItem(0, 1, label_item)

        if self.mask_mode == 'overlay':
            # The background is the source image itself, so there is no color to pick
            source_label = QLabel("Source image")
            source_label.setEnabled(False)
            source_label.setToolTip("In Overlay mode, unannotated pixels keep their original image values.")
            container_widget = QWidget()
            layout = QHBoxLayout(container_widget)
            layout.addWidget(source_label)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.label_table.setCellWidget(0, 2, container_widget)
        elif self.mask_mode == 'rgb':
            swatch = ClickableColorSwatchWidget(self.rgb_background_color)
            swatch.clicked.connect(self.pick_background_color)
            # Create a container widget to center the swatch
            container_widget = QWidget()
            layout = QHBoxLayout(container_widget)
            layout.addWidget(swatch)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.label_table.setCellWidget(0, 2, container_widget)
        else:
            spinbox = QSpinBox()
            spinbox.setRange(0, 255)
            spinbox.setValue(0)
            self.label_table.setCellWidget(0, 2, spinbox)

        # --- LABEL ROWS ---
        for i, label in enumerate(self.label_window.labels):
            row = i + 1
            self.label_table.insertRow(row)

            # Column 0: Include Checkbox
            checkbox_widget = self.create_centered_checkbox(checked=True)
            self.label_table.setCellWidget(row, 0, checkbox_widget)
            
            # Column 1: Label Name
            label_item = QTableWidgetItem(label.short_label_code)
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            label_item.setData(Qt.UserRole, label.short_label_code)
            self.label_table.setItem(row, 1, label_item)

            # Column 2: Mode-dependent widget
            if self.mask_mode == 'semantic':
                spinbox = QSpinBox()
                spinbox.setRange(0, 255)
                spinbox.setValue(i + 1)
                self.label_table.setCellWidget(row, 2, spinbox)
            elif self.mask_mode == 'sfm':
                spinbox = QSpinBox()
                spinbox.setRange(0, 255)
                spinbox.setValue(255)  # Default foreground value
                self.label_table.setCellWidget(row, 2, spinbox)
            elif self._is_color_mode():
                try:
                    q_color = QColor(label.color)
                except Exception:
                    q_color = QColor("#FFFFFF")  # Default to white on error
                
                swatch = ColorSwatchWidget(q_color)
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(swatch)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.label_table.setCellWidget(row, 2, cell_widget)
        
        # Select the first label row rather than the pinned background row, so the
        # reorder buttons are usable straight away
        if self.label_table.rowCount() > 1:
            self.label_table.selectRow(1)
        elif self.label_table.rowCount() > 0:
            self.label_table.selectRow(0)
        self.label_table.blockSignals(False)
        self.update_move_buttons()
        
    def pick_background_color(self):
        """Pick the background color using a color dialog."""
        color = QColorDialog.getColor(self.rgb_background_color, self, "Select Background Color")
        if color.isValid():
            self.rgb_background_color = color
            swatch_container = self.label_table.cellWidget(0, 2)
            if swatch_container:
                # Find the swatch inside the container
                swatch = swatch_container.findChild(ClickableColorSwatchWidget)
                if swatch:
                    swatch.setColor(color)

    def create_centered_checkbox(self, checked=True):
        """Create a centered checkbox widget."""
        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget
        
    def browse_output_dir(self):
        """Browse for the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def get_selected_image_paths(self):
        """Get all image paths."""
        return self.image_window.raster_manager.image_paths

    def validate_inputs(self):
        """Validate the user inputs before export."""
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, 
                                "Missing Input", 
                                "Please select an output directory.")
            return False
        if not any([self.patch_checkbox.isChecked(), 
                    self.rectangle_checkbox.isChecked(), 
                    self.polygon_checkbox.isChecked(),
                    self.mask_checkbox.isChecked()]):
            QMessageBox.warning(self, 
                                "Missing Input", 
                                "Please select at least one annotation type.")
            return False
        return True

    def run_export_process(self):
        """Run the export process for mask annotations."""
        if not self.validate_inputs():
            return

        self.labels_to_render = []
        self.label_code_to_export_value = {}
        self.background_value = 0 

        self.file_format = self.file_format_combo.currentData() or self.file_format_combo.currentText()
        if not self.file_format.startswith('.'):
            self.file_format = '.' + self.file_format

        if self.file_format == '.txt' and self._is_color_mode():
            QMessageBox.warning(
                self,
                "Unsupported Export Format",
                "RLE export is only supported for semantic and SfM mask modes."
            )
            return

        # Overlay blends against the source image, so capture the slider value and outline preference
        self.overlay_transparency = self.get_transparency_value()
        self.draw_outlines = self.draw_outlines_checkbox.isChecked()

        # --- Collect data from UI based on mode ---
        if self.mask_mode in ['semantic', 'sfm']:
            used_mask_values = {}
            for i in range(self.label_table.rowCount()):
                if not self.label_table.cellWidget(i, 0).findChild(QCheckBox).isChecked():
                    continue
                
                label_code = self.label_table.item(i, 1).data(Qt.UserRole)
                mask_value = self.label_table.cellWidget(i, 2).value()

                if mask_value not in used_mask_values:
                    used_mask_values[mask_value] = []
                used_mask_values[mask_value].append(label_code)

                if label_code == "background":
                    self.background_value = mask_value
                else:
                    self.label_code_to_export_value[label_code] = mask_value
                    label = next((l for l in self.label_window.labels if l.short_label_code == label_code), None)
                    if label:
                        self.labels_to_render.append((label, mask_value))
            
            # Check for duplicate values
            duplicate_values = {v: l for v, l in used_mask_values.items() if len(l) > 1}
            if duplicate_values:
                msg = "Warning: The following mask values are used by multiple labels:\n" + \
                      "\n".join([f"Value {v}: {', '.join(l)}" for v, l in duplicate_values.items()]) + \
                      "\nThis may cause unexpected behavior. Continue?"
                if QMessageBox.warning(self, 
                                       "Duplicate Values", 
                                       msg, 
                                       QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                    return

        elif self._is_color_mode():
            if self.mask_mode == 'overlay':
                # The accumulator is RGBA; alpha 0 means "not painted" so the source pixel shows through
                self.background_value = (0, 0, 0, 0)
            else:
                self.background_value = self.rgb_background_color.getRgb()[:3]  # (R, G, B) tuple

            for i in range(1, self.label_table.rowCount()):  # Skip background
                if self.label_table.cellWidget(i, 0).findChild(QCheckBox).isChecked():
                    label_code = self.label_table.item(i, 1).data(Qt.UserRole)
                    label = next((l for l in self.label_window.labels if l.short_label_code == label_code), None)
                    if label:
                        try:
                            # Check if label.color is already a QColor object
                            if isinstance(label.color, QColor):
                                color_tuple = label.color.getRgb()[:3]  # Extract (R,G,B) and ignore alpha
                            else:
                                # Otherwise, assume it's a string (hex code) and convert it
                                color_tuple = ImageColor.getrgb(label.color)

                            # Overlay carries a 4th coverage channel, fully opaque wherever it is drawn
                            if self.mask_mode == 'overlay':
                                color_tuple = tuple(color_tuple[:3]) + (255,)

                            self.labels_to_render.append((label, color_tuple))
                            self.label_code_to_export_value[label_code] = color_tuple
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid color format for label "
                                  f"'{label.short_label_code}': {label.color}. Error: {e}. Skipping.")

        # --- Check if any labels are selected to be drawn ---
        if not self.labels_to_render and not self.mask_checkbox.isChecked():
            QMessageBox.warning(self, "No Data Selected", "Please select at least one label or include Mask Annotations.")
            return

        # --- Setup paths and progress bar ---
        output_dir = self.output_dir_edit.text()
        folder_name = self.output_name_edit.text().strip()

        output_path = os.path.join(output_dir, folder_name)
        os.makedirs(output_path, exist_ok=True)
        
        images = self.get_selected_image_paths()
        if not images:
            QMessageBox.warning(self, "No Images", "No images found for processing.")
            return
            
        # Store selected annotation types
        self.include_mask_annotations = self.mask_checkbox.isChecked()
        self.annotation_types = []
        if self.patch_checkbox.isChecked(): 
            self.annotation_types.append(PatchAnnotation)
        if self.rectangle_checkbox.isChecked(): 
            self.annotation_types.append(RectangleAnnotation)
        if self.polygon_checkbox.isChecked():
            self.annotation_types.append(PolygonAnnotation)
            self.annotation_types.append(MultiPolygonAnnotation)

        # --- Run Export Loop ---
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting Masks")
        progress_bar.show()
        progress_bar.start_progress(len(images))

        # Images that could not be read are collected here and reported once at the end
        self.skipped_images = []
        self.exported_count = 0

        try:
            for image_path in images:
                self.create_mask_for_image(image_path, output_path)
                progress_bar.update_progress()

            self.export_metadata(output_path)

            message = "Masks exported successfully."
            if self.skipped_images:
                names = [os.path.basename(path) for path in self.skipped_images[:10]]
                preview = "\n".join(names)
                if len(self.skipped_images) > 10:
                    preview += "\n...and %d more." % (len(self.skipped_images) - 10)
                message += ("\n\n%d image(s) were skipped because their pixels could not be read:"
                            "\n%s" % (len(self.skipped_images), preview))

            # Status bar keeps a short record of the export after the dialog is dismissed
            mode_names = {
                'semantic': "Semantic",
                'sfm': "SfM",
                'rgb': "Visualization",
                'overlay': "Overlay"
            }
            status = (f"Exported {self.exported_count} {mode_names.get(self.mask_mode, self.mask_mode)} "
                      f"mask(s) to {output_path}")
            if self.skipped_images:
                status += f" — {len(self.skipped_images)} skipped"
            self.main_window.status_bar.showMessage(status, 5000)

            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during export: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.close()

    def create_mask_for_image(self, image_path, output_path):
        """Create a mask for the given image."""
        height, width, has_georef, transform, crs = self.get_image_metadata(image_path, self.file_format)
        if not height or not width:
            print(f"Skipping {image_path}: could not determine dimensions.")
            self.skipped_images.append(image_path)
            return

        mask = None
        has_mask_data = False
        
        # --- 1. Render MaskAnnotation (Base Layer) ---
        if self.include_mask_annotations:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and raster.mask_annotation:
                mask = self._render_mask_annotation(raster.mask_annotation, height, width)
                has_mask_data = True
        
        # --- 2. Create blank mask if no MaskAnnotation was rendered ---
        if mask is None:
            mask = np.full(self._mask_shape(height, width), self.background_value, dtype=np.uint8)

        # --- 3. Render Vector Annotations (Top Layer) ---
        # Overlay keeps the drawn annotations so their outlines can be stroked after blending
        has_vector_annotations = False
        drawn_annotations = []
        if self.annotation_types:  # Only run if vector types were selected
            for label, value in self.labels_to_render:
                annotations = self.get_annotations_for_image(image_path, label)
                if annotations:
                    has_vector_annotations = True
                    self.draw_annotations_on_mask(mask, annotations, value)
                    drawn_annotations.append((annotations, value))

        # --- 4. Check for negative samples ---
        if (not has_mask_data and not has_vector_annotations and
                not self.include_negative_samples_checkbox.isChecked()):
            return

        # --- 4b. Overlay: blend the color mask over the source image ---
        if self.mask_mode == 'overlay':
            base_image = self._load_base_image(image_path, height, width)
            if base_image is None:
                print(f"Skipping {image_path}: could not read image pixels for the overlay.")
                self.skipped_images.append(image_path)
                return

            self._composite_overlay(base_image, mask, self.overlay_transparency)
            mask = base_image

        # --- 4c. Outlines, drawn last so boundaries survive both the blend and any overpainting ---
        if self._is_color_mode() and self.draw_outlines:
            for annotations, value in drawn_annotations:
                self.draw_annotation_outlines(mask, annotations, tuple(value[:3]))

        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}{self.file_format}"
        mask_path = os.path.join(output_path, filename)

        if self.file_format.lower() == '.txt':
            self._save_rle_mask(mask, mask_path)
            self.exported_count += 1
            return

        # --- 5. Save the final mask ---

        # Check if we need to preserve georeferencing
        use_georef = has_georef and self.preserve_georef_checkbox.isChecked() and self.file_format.lower() == '.tif'
        
        if use_georef:
            # Save with georeferencing using rasterio
            if self._is_color_mode():
                # For RGB, we need to convert to the expected channel order for rasterio
                # rasterio expects (bands, height, width) with R,G,B channel order
                mask_transposed = np.transpose(mask, (2, 0, 1))
                with rasterio.open(
                    mask_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=3,
                    dtype=mask.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(mask_transposed)
            else:
                # For single-channel masks
                with rasterio.open(
                    mask_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=mask.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(mask, 1)
        else:
            # Use cv2 for non-georeferenced output
            if self._is_color_mode():
                # OpenCV expects BGR, so convert from RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            
            # Save using the appropriate format
            cv2.imwrite(mask_path, mask)

        self.exported_count += 1

    def _encode_rle_runs(self, mask: np.ndarray):
        """Encode a 2D mask as value/count runs in row-major order."""
        flat_mask = np.asarray(mask).ravel(order='C')
        if flat_mask.size == 0:
            return []

        change_points = np.flatnonzero(flat_mask[1:] != flat_mask[:-1]) + 1
        run_starts = np.concatenate(([0], change_points))
        run_ends = np.concatenate((change_points, [flat_mask.size]))

        values = flat_mask[run_starts]
        counts = run_ends - run_starts
        return list(zip(values.tolist(), counts.tolist()))

    def _save_rle_mask(self, mask: np.ndarray, output_path: str):
        """Save a single-channel mask as a plain-text value/count RLE file."""
        if mask.ndim != 2:
            raise ValueError("RLE export only supports single-channel masks.")

        height, width = mask.shape
        rle_runs = self._encode_rle_runs(mask)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(f"{height} {width}\n")
            for value, count in rle_runs:
                file.write(f"{int(value)} {int(count)}\n")

    def _render_mask_annotation(self, mask_annotation, height, width):
        """
        Converts a MaskAnnotation's internal data to a new mask using
        the export values (integer or RGB) from the UI.
        """
        try:
            # 1. Initialize the output mask
            output_mask = np.full(self._mask_shape(height, width), self.background_value, dtype=np.uint8)
            
            # 2. Get the source mask data and label mapping
            mask_data = mask_annotation.mask_data
            id_to_label_map = mask_annotation.class_id_to_label_map
            
            # 3. Get the lock bit (if it exists)
            lock_bit = getattr(mask_annotation, 'LOCK_BIT', 128) 

            # 4. Iterate over the mask's internal classes and map them
            for class_id, label_obj in id_to_label_map.items():
                label_code = label_obj.short_label_code
                
                # Find the export value (e.g., 5 or (255,0,0)) for this label
                export_value = self.label_code_to_export_value.get(label_code)
                
                # Only render if this label is included in the export
                if export_value is not None:
                    # Find all pixels (locked and unlocked) for this class ID
                    pixel_mask = (mask_data == class_id) | (mask_data == class_id + lock_bit)
                    output_mask[pixel_mask] = export_value
            
            return output_mask
        
        except Exception as e:
            print(f"Error rendering mask annotation: {e}")
            return None  # Fallback to blank mask

    def export_metadata(self, output_path):
        """Export metadata files based on the mode."""
        if self.mask_mode == 'semantic':
            class_mapping = {}
            if self.label_table.cellWidget(0, 0).findChild(QCheckBox).isChecked():
                background_label = "background"
                background_index = self.label_table.cellWidget(0, 2).value()
                class_mapping[background_label] = {
                    "label": background_label,
                    "index": background_index
                }
            
            for label, value in self.labels_to_render:
                class_mapping[label.short_label_code] = {"label": label.to_dict(), "index": value}
            
            with open(os.path.join(output_path, "class_mapping.json"), 'w') as f:
                json.dump(class_mapping, f, indent=4)
        
        elif self._is_color_mode():
            color_legend = {}
            if self.mask_mode == 'overlay':
                # Record the blend settings so an export can be reproduced later
                color_legend["_overlay"] = {
                    "background": "source image",
                    "transparency": self.overlay_transparency,
                    "outlines": self.draw_outlines
                }
            elif self.label_table.cellWidget(0, 0).findChild(QCheckBox).isChecked():
                color_legend["background"] = self.background_value

            for label, color in self.labels_to_render:
                color_legend[label.short_label_code] = tuple(color[:3])

            with open(os.path.join(output_path, "color_legend.json"), 'w') as f:
                json.dump(color_legend, f, indent=4)

        # No metadata file needed for SfM mode

    def get_annotations_for_image(self, image_path, label):
        """Get annotations for the image and label."""
        annotations = []
        # self.annotation_types now only contains VECTOR types
        if not self.annotation_types:
            return [] 
            
        for ann in self.annotation_window.get_image_annotations(image_path):
            if ann.label.short_label_code == label.short_label_code and isinstance(ann, tuple(self.annotation_types)):
                if isinstance(ann, MultiPolygonAnnotation):
                    annotations.extend(ann.polygons)
                else:
                    annotations.append(ann)
        return annotations

    def draw_annotations_on_mask(self, mask, annotations, value):
        """Draw annotations on the mask."""
        for ann in annotations:
            if isinstance(ann, (PatchAnnotation, RectangleAnnotation)):
                p1 = (int(ann.get_bounding_box_top_left().x()), int(ann.get_bounding_box_top_left().y()))
                p2 = (int(ann.get_bounding_box_bottom_right().x()), int(ann.get_bounding_box_bottom_right().y()))
                cv2.rectangle(mask, p1, p2, value, -1)
            elif isinstance(ann, PolygonAnnotation):
                points = np.array([[p.x(), p.y()] for p in ann.points], dtype=np.int32)
                cv2.fillPoly(mask, [points], value)

    def draw_annotation_outlines(self, image, annotations, color, thickness=2):
        """
        Stroke a fully opaque border around each annotation.

        Used by Overlay mode after blending, so the outlines stay crisp instead
        of fading with the translucent fill (matching the Annotation Window).
        """
        for ann in annotations:
            if isinstance(ann, (PatchAnnotation, RectangleAnnotation)):
                p1 = (int(ann.get_bounding_box_top_left().x()), int(ann.get_bounding_box_top_left().y()))
                p2 = (int(ann.get_bounding_box_bottom_right().x()), int(ann.get_bounding_box_bottom_right().y()))
                cv2.rectangle(image, p1, p2, color, thickness)
            elif isinstance(ann, PolygonAnnotation):
                points = np.array([[p.x(), p.y()] for p in ann.points], dtype=np.int32)
                cv2.polylines(image, [points], True, color, thickness)

    def _mask_shape(self, height, width):
        """Return the accumulator shape for the current mode."""
        if self.mask_mode == 'overlay':
            # The 4th channel is coverage: alpha 0 means "not painted"
            return (height, width, 4)
        if self.mask_mode == 'rgb':
            return (height, width, 3)
        return (height, width)  # semantic or sfm

    def _load_base_image(self, image_path, height, width):
        """
        Load the source image pixels as an (H, W, 3) uint8 RGB array for Overlay mode.

        Uses rasterio directly rather than Raster.get_qimage(), which caches the
        full-resolution QImage for the lifetime of the raster and would pin every
        image in memory across a bulk export.
        """
        image = None

        try:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and raster.rasterio_src:
                image = rasterio_to_numpy(raster.rasterio_src)
        except Exception as e:
            print(f"Error reading pixels for {image_path} via rasterio: {e}")
            image = None

        # rasterio_to_numpy returns a 100x100 placeholder on failure, so validate the shape
        if image is None or image.shape[:2] != (height, width):
            try:
                fallback = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if fallback is not None and fallback.shape[:2] == (height, width):
                    image = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                else:
                    image = None
            except Exception as e:
                print(f"Error reading pixels for {image_path} via OpenCV: {e}")
                image = None

        if image is None:
            return None

        # Guarantee a contiguous, writable 3-channel uint8 array for in-place blending
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return np.ascontiguousarray(image)

    def _composite_overlay(self, base_rgb, rgba, transparency, block_rows=2048):
        """
        Blend the RGBA color mask into base_rgb in place, wherever coverage alpha is set.

        Only pixels with a non-zero coverage alpha are touched, so unannotated areas
        keep their original values. Processed in row blocks to bound the size of the
        float temporary on very large images.
        """
        alpha = max(0, min(255, int(transparency))) / 255.0
        if alpha <= 0.0:
            return  # Fully transparent: the source image is unchanged

        height = base_rgb.shape[0]
        for start in range(0, height, block_rows):
            stop = min(start + block_rows, height)
            base_block = base_rgb[start:stop]
            rgba_block = rgba[start:stop]

            covered = rgba_block[:, :, 3] > 0
            if not covered.any():
                continue

            blended = (base_block[covered] * (1.0 - alpha) + rgba_block[covered, :3] * alpha)
            base_block[covered] = blended.astype(np.uint8)

    def get_image_metadata(self, image_path, file_format):
        """Get image metadata including dimensions and georeferencing."""
        transform, crs, has_georef = None, None, False
        width, height = None, None
        raster = self.image_window.raster_manager.get_raster(image_path)
        can_preserve = self.preserve_georef_checkbox.isChecked() and file_format.lower() == '.tif'

        if raster and raster.rasterio_src:
            width, height = raster.width, raster.height
            if can_preserve and hasattr(raster.rasterio_src, 'transform'):
                transform = raster.rasterio_src.transform
                if transform and not transform.is_identity:
                    crs = raster.rasterio_src.crs
                    has_georef = True
        else:
            try:
                if can_preserve:
                    with rasterio.open(image_path) as src:
                        width, height = src.width, src.height
                        if src.transform and not src.transform.is_identity:
                            transform, crs, has_georef = src.transform, src.crs, True
                else:
                    with Image.open(image_path) as img:
                        width, height = img.size
            except Exception as e:
                print(f"Error reading metadata for {image_path}: {e}")
        return height, width, has_georef, transform, crs

    # --- Row Movement and UI Helpers ---
    def update_move_buttons(self):
        """Enable the reorder buttons only for label rows that can actually move."""
        row = self.label_table.currentRow()
        last_row = self.label_table.rowCount() - 1
        # Row 0 is the fixed background row: it is the initial fill, so it is always painted
        # first and never takes part in the ordering.
        self.move_up_button.setEnabled(row > 1)
        self.move_down_button.setEnabled(1 <= row < last_row)

    def move_row_up(self):
        """Move the selected row up in the table."""
        current_row = self.label_table.currentRow()
        if current_row > 1:
            self.swap_rows(current_row, current_row - 1)
            self.label_table.selectRow(current_row - 1)

    def move_row_down(self):
        """Move the selected row down in the table."""
        current_row = self.label_table.currentRow()
        if 1 <= current_row < self.label_table.rowCount() - 1:
            self.swap_rows(current_row, current_row + 1)
            self.label_table.selectRow(current_row + 1)

    def swap_rows(self, row1, row2):
        """
        Swap two label rows, carrying the whole row with them.

        Only label rows reach this point (the background row is pinned at row 0), so
        column 2 holds the same kind of widget in both rows for any given mode.
        """
        table = self.label_table

        # Extract all state from both rows
        def extract_row(r):
            checked = table.cellWidget(r, 0).findChild(QCheckBox).isChecked()
            label_code = table.item(r, 1).data(Qt.UserRole)
            label_text = table.item(r, 1).text()

            col2_value = None
            widget = table.cellWidget(r, 2)
            if widget is not None:
                if self._is_color_mode():
                    swatch = widget.findChild(ColorSwatchWidget)
                    if swatch:
                        col2_value = QColor(swatch.color)
                else:
                    col2_value = widget.value()

            return {'checked': checked, 'code': label_code, 'text': label_text, 'col2': col2_value}

        data1 = extract_row(row1)
        data2 = extract_row(row2)

        def apply_row(r, data):
            table.cellWidget(r, 0).findChild(QCheckBox).setChecked(data['checked'])

            item = table.item(r, 1)
            item.setText(data['text'])
            item.setData(Qt.UserRole, data['code'])

            widget = table.cellWidget(r, 2)
            if widget is None or data['col2'] is None:
                return

            if self._is_color_mode():
                swatch = widget.findChild(ColorSwatchWidget)
                if swatch:
                    swatch.setColor(data['col2'])
            else:
                widget.setValue(data['col2'])

        table.blockSignals(True)
        apply_row(row1, data2)
        apply_row(row2, data1)
        table.blockSignals(False)

    def update_georef_availability(self):
        """Update georeferencing availability based on file format."""
        selected_format = self.file_format_combo.currentData() or self.file_format_combo.currentText()
        is_tif = str(selected_format).lower() == '.tif'
        self.preserve_georef_checkbox.setEnabled(is_tif)
        if not is_tif:
            self.preserve_georef_checkbox.setChecked(False)

    def closeEvent(self, event):
        """Handle the close event."""
        super().closeEvent(event)
