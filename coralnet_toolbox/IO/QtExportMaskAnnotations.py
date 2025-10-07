import warnings
import os
import ujson as json

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageColor

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QTableWidgetItem,
                             QButtonGroup, QWidget, QTableWidget, QHeaderView,
                             QAbstractItemView, QSpinBox, QRadioButton, QColorDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Classes
# ----------------------------------------------------------------------------------------------------------------------


class ColorSwatchWidget(QWidget):
    """A simple widget to display a color swatch with a border."""
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
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
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ----------------------------------------------------------------------------------------------------------------------
# Main Dialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ExportMaskAnnotations(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("mask.png"))
        self.setWindowTitle("Export Annotations to Masks")
        self.resize(1000, 800)

        self.mask_mode = 'semantic'  # 'semantic', 'sfm', or 'rgb'
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
        self.setup_image_selection_layout(parent_layout=left_col)
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
        super().showEvent(event)
        self.update_ui_for_mode()

    def setup_info_layout(self, parent_layout=None):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_text = (
            "This tool exports annotations to image masks for three primary use cases:<br><br>"
            "<b>1. Semantic Segmentation (Integer IDs):</b> Creates masks where each class (e.g., coral, rock) "
            "is represented by a unique integer value (1, 2, 3...). The background is typically 0. These are used "
            "to train machine learning models.<br><br>"
            "<b>2. Structure from Motion (SfM) (Binary Mask):</b> Creates masks where a foreground value "
            "(e.g., 255) represents objects to keep, and a background value (e.g., 0) represents areas to "
            "ignore. This is used by software like Metashape to improve 3D model reconstruction.<br><br>"
            "<b>3. Visualization (RGB Colors):</b> Creates a human-readable color mask using the colors "
            "assigned to each label. Ideal for reports, presentations, and qualitative analysis."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_output_layout(self, parent_layout=None):
        groupbox = QGroupBox("Output Directory and File Format")
        layout = QFormLayout()

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        self.output_name_edit = QLineEdit("masks")
        layout.addRow("Folder Name:", self.output_name_edit)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_mask_format_layout(self, parent_layout=None):
        groupbox = QGroupBox("Export Mode and Format")
        main_layout = QVBoxLayout()

        # Mode Selection
        mode_layout = QHBoxLayout()
        self.semantic_radio = QRadioButton("Semantic Segmentation (Integer IDs)")
        self.sfm_radio = QRadioButton("Structure from Motion (Binary Mask)")
        self.rgb_radio = QRadioButton("Visualization (RGB Colors)")

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.semantic_radio)
        self.mode_group.addButton(self.sfm_radio)
        self.mode_group.addButton(self.rgb_radio)
        self.mode_group.buttonClicked.connect(self.update_ui_for_mode)

        mode_layout.addWidget(self.semantic_radio)
        mode_layout.addWidget(self.sfm_radio)
        mode_layout.addWidget(self.rgb_radio)
        main_layout.addLayout(mode_layout)

        # Format and Options
        options_layout = QFormLayout()
        
        # General file format
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems([".png", ".bmp", ".tif"])
        self.file_format_combo.currentTextChanged.connect(self.update_georef_availability)
        options_layout.addRow("File Format:", self.file_format_combo)

        # Georeferencing
        self.preserve_georef_checkbox = QCheckBox("Preserve georeferencing (if available)")
        self.preserve_georef_checkbox.setChecked(True)
        options_layout.addRow(self.preserve_georef_checkbox)
        self.georef_note = QLabel("Note: Georeferencing is only supported for TIF format.")
        self.georef_note.setStyleSheet("color: #666; font-style: italic;")
        options_layout.addRow(self.georef_note)

        main_layout.addLayout(options_layout)
        groupbox.setLayout(main_layout)
        parent_layout.addWidget(groupbox)
        self.update_georef_availability()

    def setup_image_selection_layout(self, parent_layout=None):
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
        parent_layout.addWidget(group_box)

    def setup_annotation_layout(self, parent_layout=None):
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)
        self.include_negative_samples_checkbox = QCheckBox("Include negative samples")
        self.include_negative_samples_checkbox.setChecked(True)

        layout.addWidget(self.patch_checkbox)
        layout.addWidget(self.rectangle_checkbox)
        layout.addWidget(self.polygon_checkbox)
        layout.addWidget(self.include_negative_samples_checkbox)
        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_label_layout(self, parent_layout=None):
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
        self.move_down_button.clicked.connect(self.move_row_down)
        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)
        
        order_note = QLabel(
            "<b>Layer Order is Important:</b> Labels lower in the list will be drawn on top of labels "
            "higher in the list. For overlapping annotations, only the topmost class will appear."
        )
        order_note.setStyleSheet("color: #666;")
        order_note.setWordWrap(True)
        layout.addWidget(order_note)
        
        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_buttons_layout(self, parent_layout=None):
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.run_export_process)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def update_ui_for_mode(self):
        """Update the UI dynamically based on the selected export mode."""
        if self.semantic_radio.isChecked():
            self.mask_mode = 'semantic'
        elif self.sfm_radio.isChecked():
            self.mask_mode = 'sfm'
        elif self.rgb_radio.isChecked():
            self.mask_mode = 'rgb'
        
        self.populate_label_table()
    
    def populate_label_table(self):
        """Populate the label table based on the current mode."""
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(0)

        # Set table headers based on mode
        headers = ["Include", "Label Name"]
        if self.mask_mode in ['semantic', 'sfm']:
            headers.append("Mask Value")
        elif self.mask_mode == 'rgb':
            headers.append("Color Preview")
        self.label_table.setHorizontalHeaderLabels(headers)

        # --- BACKGROUND ROW (ROW 0) ---
        self.label_table.insertRow(0)
        checkbox_widget = self.create_centered_checkbox(checked=True)
        self.label_table.setCellWidget(0, 0, checkbox_widget)
        
        label_item = QTableWidgetItem("Background")
        label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
        label_item.setData(Qt.UserRole, "Background")
        self.label_table.setItem(0, 1, label_item)

        if self.mask_mode in ['semantic', 'sfm']:
            spinbox = QSpinBox()
            spinbox.setRange(0, 255)
            spinbox.setValue(0)
            self.label_table.setCellWidget(0, 2, spinbox)
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
            elif self.mask_mode == 'rgb':
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
        
        if self.label_table.rowCount() > 0:
            self.label_table.selectRow(0)
        self.label_table.blockSignals(False)
        
    def pick_background_color(self):
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
        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget
        
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def get_selected_image_paths(self):
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []

        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[:current_index + 1]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[current_index:]
        elif self.apply_all_checkbox.isChecked():
            return self.image_window.raster_manager.image_paths
        
        return [current_image_path]

    def validate_inputs(self):
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, 
                                "Missing Input", 
                                "Please select an output directory.")
            return False
        if not any([self.patch_checkbox.isChecked(), 
                    self.rectangle_checkbox.isChecked(), 
                    self.polygon_checkbox.isChecked()]):
            QMessageBox.warning(self, 
                                "Missing Input", 
                                "Please select at least one annotation type.")
            return False
        return True

    def run_export_process(self):
        if not self.validate_inputs():
            return

        self.labels_to_render = []
        self.background_value = 0 
        
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

                if label_code == "Background":
                    self.background_value = mask_value
                else:
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

        elif self.mask_mode == 'rgb':
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
                            self.labels_to_render.append((label, color_tuple))
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid color format for label "
                                  f"'{label.short_label_code}': {label.color}. Error: {e}. Skipping.")

        # --- Check if any labels are selected to be drawn ---
        if not self.labels_to_render:
            QMessageBox.warning(self, "No Labels Selected", "Please select at least one label to include in the masks.")
            return

        # --- Setup paths and progress bar ---
        output_dir = self.output_dir_edit.text()
        folder_name = self.output_name_edit.text().strip()
        self.file_format = self.file_format_combo.currentText()
        if not self.file_format.startswith('.'):
            self.file_format = '.' + self.file_format

        output_path = os.path.join(output_dir, folder_name)
        os.makedirs(output_path, exist_ok=True)
        
        images = self.get_selected_image_paths()
        if not images:
            QMessageBox.warning(self, "No Images", "No images found for processing.")
            return
            
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

        try:
            for image_path in images:
                self.create_mask_for_image(image_path, output_path)
                progress_bar.update_progress()

            self.export_metadata(output_path)
            QMessageBox.information(self, "Export Complete", "Masks exported successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during export: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.close()

    def create_mask_for_image(self, image_path, output_path):
        height, width, has_georef, transform, crs = self.get_image_metadata(image_path, self.file_format)
        if not height or not width:
            print(f"Skipping {image_path}: could not determine dimensions.")
            return

        # Initialize mask based on mode
        if self.mask_mode == 'rgb':
            mask = np.full((height, width, 3), self.background_value, dtype=np.uint8)
        else:  # semantic or sfm
            mask = np.full((height, width), self.background_value, dtype=np.uint8)

        has_annotations = False
        
        # Draw annotations based on mode
        if self.mask_mode in ['semantic', 'sfm']:
            for label, value in self.labels_to_render:
                annotations = self.get_annotations_for_image(image_path, label)
                if annotations:
                    has_annotations = True
                    self.draw_annotations_on_mask(mask, annotations, value)
        elif self.mask_mode == 'rgb':
            for label, color in self.labels_to_render:
                annotations = self.get_annotations_for_image(image_path, label)
                if annotations:
                    has_annotations = True
                    self.draw_annotations_on_mask(mask, annotations, color)
        
        if not has_annotations and not self.include_negative_samples_checkbox.isChecked():
            return

        # Use the selected file format
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}{self.file_format}"
        mask_path = os.path.join(output_path, filename)

        # Check if we need to preserve georeferencing
        use_georef = has_georef and self.preserve_georef_checkbox.isChecked() and self.file_format.lower() == '.tif'
        
        if use_georef:
            # Save with georeferencing using rasterio
            if self.mask_mode == 'rgb':
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
            if self.mask_mode == 'rgb':
                # OpenCV expects BGR, so convert from RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            
            # Save using the appropriate format
            cv2.imwrite(mask_path, mask)

    def export_metadata(self, output_path):
        if self.mask_mode == 'semantic':
            class_mapping = {}
            if self.label_table.cellWidget(0, 0).findChild(QCheckBox).isChecked():
                background_label = "Background"
                background_index = self.label_table.cellWidget(0, 2).value()
                class_mapping[background_label] = {
                    "label": background_label,
                    "index": background_index
                }
            
            for label, value in self.labels_to_render:
                class_mapping[label.short_label_code] = {"label": label.to_dict(), "index": value}
            
            with open(os.path.join(output_path, "class_mapping.json"), 'w') as f:
                json.dump(class_mapping, f, indent=4)
        
        elif self.mask_mode == 'rgb':
            color_legend = {}
            if self.label_table.cellWidget(0, 0).findChild(QCheckBox).isChecked():
                color_legend["Background"] = self.background_value

            for label, color in self.labels_to_render:
                color_legend[label.short_label_code] = color
            
            with open(os.path.join(output_path, "color_legend.json"), 'w') as f:
                json.dump(color_legend, f, indent=4)
        
        # No metadata file needed for SfM mode

    def get_annotations_for_image(self, image_path, label):
        annotations = []
        for ann in self.annotation_window.get_image_annotations(image_path):
            if ann.label.short_label_code == label.short_label_code and isinstance(ann, tuple(self.annotation_types)):
                if isinstance(ann, MultiPolygonAnnotation):
                    annotations.extend(ann.polygons)
                else:
                    annotations.append(ann)
        return annotations

    def draw_annotations_on_mask(self, mask, annotations, value):
        for ann in annotations:
            if isinstance(ann, (PatchAnnotation, RectangleAnnotation)):
                p1 = (int(ann.top_left.x()), int(ann.top_left.y()))
                p2 = (int(ann.bottom_right.x()), int(ann.bottom_right.y()))
                cv2.rectangle(mask, p1, p2, value, -1)
            elif isinstance(ann, PolygonAnnotation):
                points = np.array([[p.x(), p.y()] for p in ann.points], dtype=np.int32)
                cv2.fillPoly(mask, [points], value)

    def get_image_metadata(self, image_path, file_format):
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
    def move_row_up(self):
        current_row = self.label_table.currentRow()
        if current_row > 0:
            self.swap_rows(current_row, current_row - 1)
            self.label_table.selectRow(current_row - 1)

    def move_row_down(self):
        current_row = self.label_table.currentRow()
        if 0 <= current_row < self.label_table.rowCount() - 1:
            self.swap_rows(current_row, current_row + 1)
            self.label_table.selectRow(current_row + 1)

    def swap_rows(self, row1, row2):
        # Because widgets are complex to swap, we just repopulate the table
        # while preserving the core data (label order). This is simpler and more robust.
        
        # Step 1: Extract the core data and checkbox state from the table
        table_data = []
        for r in range(self.label_table.rowCount()):
            is_checked = self.label_table.cellWidget(r, 0).findChild(QCheckBox).isChecked()
            label_code = self.label_table.item(r, 1).data(Qt.UserRole)
            table_data.append({'code': label_code, 'checked': is_checked})

        # Step 2: Swap the data for the two rows
        table_data[row1], table_data[row2] = table_data[row2], table_data[row1]

        # Step 3: Map old label order to new order
        new_label_order = [data['code'] for data in table_data if data['code'] != 'Background']
        
        def label_sort_key(x):
            if x.short_label_code in new_label_order:
                return new_label_order.index(x.short_label_code)
            else:
                return float('inf')
        self.label_window.labels.sort(key=label_sort_key)

        # Step 4: Repopulate and restore checkbox states
        self.populate_label_table()
        for r, data in enumerate(table_data):
            self.label_table.cellWidget(r, 0).findChild(QCheckBox).setChecked(data['checked'])

    def update_georef_availability(self):
        is_tif = '.tif' in self.file_format_combo.currentText().lower()
        self.preserve_georef_checkbox.setEnabled(is_tif)
        if not is_tif:
            self.preserve_georef_checkbox.setChecked(False)
            self.georef_note.setStyleSheet("color: red; font-style: italic;")
        else:
            self.georef_note.setStyleSheet("color: #666; font-style: italic;")

    def closeEvent(self, event):
        super().closeEvent(event)

