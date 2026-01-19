import os
import json
import warnings

import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QTableWidgetItem,
                             QWidget, QTableWidget, QHeaderView, QAbstractItemView)

from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon

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
        self.update()


# ----------------------------------------------------------------------------------------------------------------------
# Main Dialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ImportMaskAnnotations(QDialog):
    """Dialog for importing segmentation masks and mapping them to project labels."""
    
    def __init__(self, main_window):
        """Initialize the import mask annotations dialog."""
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("mask.png"))
        self.setWindowTitle("Import Mask Annotations")
        self.resize(800, 700)

        # State variables
        self.valid_mask_pairs = []  # List of (mask_path, raster) tuples
        self.detected_mode = None  # 'semantic' (1-channel) or 'rgb' (3-channel)
        self.unique_values = []  # List of unique values found in masks
        self.mapping_widgets = {}  # Maps value -> QComboBox for label selection
        self.loaded_json_mapping = None  # Optional mapping from JSON file

        # Main layout for the dialog
        self.main_layout = QVBoxLayout(self)

        # Top section
        top_section = QVBoxLayout()
        self.setup_info_layout(parent_layout=top_section)
        self.setup_input_layout(parent_layout=top_section)
        self.main_layout.addLayout(top_section)

        # Middle section - mapping table
        self.setup_mapping_table_layout(parent_layout=self.main_layout)

        # Status label for scan results
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        self.main_layout.addWidget(self.status_label)

        # Bottom buttons
        self.setup_buttons_layout(parent_layout=self.main_layout)

        # Initial UI state
        self.import_button.setEnabled(False)
        self.load_mapping_button.setEnabled(False)

    def setup_info_layout(self, parent_layout=None):
        """Set up the information layout section."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_text = (
            "This tool imports segmentation masks from PNG files and maps their "
            "values to project labels.<br><br>"
            "<b>Supported Formats:</b><br>"
            "• <b>1-Channel (Grayscale/Index):</b> Each pixel value (0, 1, 2, ...) "
            "represents a class ID. Value 0 is typically background.<br>"
            "• <b>3-Channel (RGB):</b> Each unique RGB color represents a different class. "
            "Black (0, 0, 0) is typically background.<br><br>"
            "<b>Requirements:</b><br>"
            "• Mask filenames must match project image filenames "
            "(e.g., <code>img_01.png</code> matches <code>img_01.jpg</code>)<br>"
            "• Mask dimensions must exactly match the corresponding image dimensions<br>"
            "• Only <code>.png</code> files are supported"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_input_layout(self, parent_layout=None):
        """Set up the input directory and scan layout."""
        groupbox = QGroupBox("Input Masks")
        layout = QVBoxLayout()

        # Directory/file selection
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select mask files or a directory containing masks...")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_button)
        layout.addLayout(input_layout)

        # Scan button
        scan_layout = QHBoxLayout()
        scan_layout.addStretch(1)
        self.scan_button = QPushButton("Scan Masks")
        self.scan_button.setMinimumWidth(120)
        self.scan_button.clicked.connect(self.scan_masks)
        scan_layout.addWidget(self.scan_button)
        scan_layout.addStretch(1)
        layout.addLayout(scan_layout)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_mapping_table_layout(self, parent_layout=None):
        """Set up the value-to-label mapping table."""
        groupbox = QGroupBox("Value to Label Mapping")
        layout = QVBoxLayout()

        # Placeholder label (shown before scanning)
        self.placeholder_label = QLabel("Click 'Scan Masks' to detect values from the selected mask files.")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #888; padding: 40px;")
        layout.addWidget(self.placeholder_label)

        # Mapping table (hidden until scan completes)
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(2)
        self.mapping_table.setHorizontalHeaderLabels(["Detected Value", "Map To Label"])
        self.mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mapping_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.mapping_table.hide()
        layout.addWidget(self.mapping_table)

        # Load mapping button
        mapping_button_layout = QHBoxLayout()
        mapping_button_layout.addStretch(1)
        self.load_mapping_button = QPushButton("Load Mapping from JSON...")
        self.load_mapping_button.clicked.connect(self.load_mapping_json)
        mapping_button_layout.addWidget(self.load_mapping_button)
        mapping_button_layout.addStretch(1)
        layout.addLayout(mapping_button_layout)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_buttons_layout(self, parent_layout=None):
        """Set up the bottom action buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.run_import_process)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def browse_input(self):
        """Open file dialog to select mask files or directory."""
        options = QFileDialog.Options()
        
        # Ask user: files or directory?
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Select Input Type")
        msg_box.setText("How would you like to select masks?")
        files_button = msg_box.addButton("Select Files", QMessageBox.ActionRole)
        dir_button = msg_box.addButton("Select Directory", QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Cancel)
        msg_box.exec_()

        if msg_box.clickedButton() == files_button:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Mask Files",
                "",
                "PNG Files (*.png);;All Files (*)",
                options=options
            )
            if file_paths:
                self.input_path_edit.setText(";".join(file_paths))
                
        elif msg_box.clickedButton() == dir_button:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Masks Directory",
                "",
                options=options
            )
            if dir_path:
                self.input_path_edit.setText(dir_path)

    def get_mask_files(self):
        """Get list of mask files from the input path."""
        input_text = self.input_path_edit.text().strip()
        if not input_text:
            return []

        mask_files = []
        
        # Check if it's a directory or file list
        if os.path.isdir(input_text):
            # Scan directory for PNG files
            for filename in os.listdir(input_text):
                if filename.lower().endswith('.png'):
                    mask_files.append(os.path.join(input_text, filename))
        else:
            # It's a semicolon-separated list of files
            paths = input_text.split(";")
            for path in paths:
                path = path.strip()
                if path and os.path.isfile(path) and path.lower().endswith('.png'):
                    mask_files.append(path)

        return mask_files

    def scan_masks(self):
        """Scan selected mask files and detect unique values."""
        mask_files = self.get_mask_files()
        
        if not mask_files:
            QMessageBox.warning(self, "No Files", "No PNG mask files found. Please select valid mask files.")
            return

        # Build image path mapping (basename without extension -> full path)
        image_path_map = {}
        for path in self.image_window.raster_manager.image_paths:
            basename = os.path.splitext(os.path.basename(path))[0]
            image_path_map[basename] = path

        if not image_path_map:
            QMessageBox.warning(self, "No Images", "No images loaded in the project. Please load images first.")
            return

        # Reset state
        self.valid_mask_pairs = []
        self.unique_values = []
        self.detected_mode = None
        self.mapping_widgets = {}

        # Statistics for summary
        matched_count = 0
        skipped_no_match = 0
        skipped_dimension = 0
        all_unique_values = set()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Scanning Masks")
            progress_bar.show()
            progress_bar.start_progress(len(mask_files))

            for mask_path in mask_files:
                if progress_bar.wasCanceled():
                    break

                # Match mask filename to project image
                mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
                image_path = image_path_map.get(mask_basename)

                if not image_path:
                    skipped_no_match += 1
                    progress_bar.update_progress()
                    continue

                # Get the raster for dimension checking
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster:
                    skipped_no_match += 1
                    progress_bar.update_progress()
                    continue

                # Check dimensions using PIL (lazy loading)
                try:
                    with Image.open(mask_path) as img:
                        mask_width, mask_height = img.size
                        mask_mode = img.mode

                        # Validate dimensions
                        if mask_width != raster.width or mask_height != raster.height:
                            skipped_dimension += 1
                            progress_bar.update_progress()
                            continue

                        # Detect mode from first valid mask
                        if self.detected_mode is None:
                            if mask_mode in ('L', 'P'):
                                self.detected_mode = 'semantic'
                            elif mask_mode in ('RGB', 'RGBA'):
                                self.detected_mode = 'rgb'
                            else:
                                # Try to infer from channels
                                if len(img.getbands()) == 1:
                                    self.detected_mode = 'semantic'
                                else:
                                    self.detected_mode = 'rgb'

                        # Extract unique values
                        if self.detected_mode == 'semantic':
                            mask_array = np.array(img.convert('L'))
                            unique_vals = np.unique(mask_array)
                            for val in unique_vals:
                                all_unique_values.add(int(val))
                        else:
                            mask_array = np.array(img.convert('RGB'))
                            # Use void view trick for efficient unique RGB detection
                            reshaped = mask_array.reshape(-1, 3)
                            void_dtype = np.dtype((np.void, reshaped.dtype.itemsize * 3))
                            unique_rgb = np.unique(
                                reshaped.view(void_dtype),
                                return_index=True
                            )[1]
                            for idx in unique_rgb:
                                rgb_tuple = tuple(reshaped[idx])
                                all_unique_values.add(rgb_tuple)

                        # Valid pair found
                        self.valid_mask_pairs.append((mask_path, raster))
                        matched_count += 1

                except Exception:
                    skipped_no_match += 1

                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

        finally:
            QApplication.restoreOverrideCursor()

        # Convert to sorted list
        if self.detected_mode == 'semantic':
            self.unique_values = sorted(list(all_unique_values))
        else:
            # Sort RGB by luminance for better visual ordering
            self.unique_values = sorted(list(all_unique_values), 
                                        key=lambda x: (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114))

        # Update status
        status_parts = [f"Found {matched_count} valid mask(s)"]
        if skipped_no_match > 0:
            status_parts.append(f"{skipped_no_match} skipped (no matching image)")
        if skipped_dimension > 0:
            status_parts.append(f"{skipped_dimension} skipped (dimension mismatch)")
        if self.unique_values:
            status_parts.append(f"detected {len(self.unique_values)} unique value(s)")
            mode_str = "1-channel/semantic" if self.detected_mode == 'semantic' else "3-channel/RGB"
            status_parts.append(f"mode: {mode_str}")
        
        self.status_label.setText(" | ".join(status_parts))

        # Populate the mapping table
        if self.valid_mask_pairs and self.unique_values:
            self.populate_mapping_table()
            self.import_button.setEnabled(True)
            self.load_mapping_button.setEnabled(True)
        else:
            self.placeholder_label.setText("No valid masks found. Check file names and dimensions.")
            self.placeholder_label.show()
            self.mapping_table.hide()
            self.import_button.setEnabled(False)
            self.load_mapping_button.setEnabled(False)

    def populate_mapping_table(self):
        """Populate the mapping table with detected values."""
        self.placeholder_label.hide()
        self.mapping_table.show()
        
        self.mapping_table.setRowCount(0)
        self.mapping_widgets = {}

        # Build label options for combobox
        label_options = ["Ignore / Background"]
        label_options.extend([label.short_label_code for label in self.label_window.labels])

        for value in self.unique_values:
            row = self.mapping_table.rowCount()
            self.mapping_table.insertRow(row)

            # Column 0: Detected Value (integer or color swatch)
            if self.detected_mode == 'semantic':
                value_item = QTableWidgetItem(str(value))
                value_item.setTextAlignment(Qt.AlignCenter)
                value_item.setData(Qt.UserRole, value)
                self.mapping_table.setItem(row, 0, value_item)
            else:
                # RGB mode - show color swatch
                q_color = QColor(value[0], value[1], value[2])
                swatch = ColorSwatchWidget(q_color)
                container = QWidget()
                layout = QHBoxLayout(container)
                layout.addWidget(swatch)
                
                # Also show RGB values as text
                rgb_label = QLabel(f"({value[0]}, {value[1]}, {value[2]})")
                rgb_label.setStyleSheet("color: #666; font-size: 10px;")
                layout.addWidget(rgb_label)
                
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(5, 2, 5, 2)
                self.mapping_table.setCellWidget(row, 0, container)
                
                # Store value in a hidden item for retrieval
                hidden_item = QTableWidgetItem()
                hidden_item.setData(Qt.UserRole, value)
                self.mapping_table.setItem(row, 0, hidden_item)

            # Column 1: Label combobox
            combo = QComboBox()
            combo.addItems(label_options)
            
            # Auto-detect background value
            is_background = False
            if self.detected_mode == 'semantic' and value == 0:
                is_background = True
            elif self.detected_mode == 'rgb' and value == (0, 0, 0):
                is_background = True
            
            if is_background:
                combo.setCurrentIndex(0)  # "Ignore / Background"
            
            self.mapping_table.setCellWidget(row, 1, combo)
            self.mapping_widgets[value] = combo

        # Apply any loaded JSON mapping
        if self.loaded_json_mapping:
            self.apply_json_mapping()

    def load_mapping_json(self):
        """Load a mapping from a JSON file (e.g., exported metadata.json)."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mapping JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.loaded_json_mapping = data
            self.apply_json_mapping()
            
            QMessageBox.information(self, "Mapping Loaded", 
                                    f"Loaded mapping from: {os.path.basename(file_path)}\n\n"
                                    "Mappings have been applied to matching values.")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load JSON file:\n{str(e)}")

    def apply_json_mapping(self):
        """Apply the loaded JSON mapping to the current table."""
        if not self.loaded_json_mapping or not self.mapping_widgets:
            return

        # Expected format from export: {"labels": [{"short_label_code": "...", "mask_value": N, "color": [R,G,B]}]}
        labels_data = self.loaded_json_mapping.get('labels', [])
        
        # Build lookup: value -> label_short_code
        value_to_label = {}
        for label_info in labels_data:
            short_code = label_info.get('short_label_code')
            mask_value = label_info.get('mask_value')
            color = label_info.get('color')
            
            if short_code and mask_value is not None:
                if self.detected_mode == 'semantic':
                    value_to_label[mask_value] = short_code
                elif color and len(color) >= 3:
                    value_to_label[tuple(color[:3])] = short_code

        # Also check for reverse format: {"label_short_code": value}
        for key, val in self.loaded_json_mapping.items():
            if key != 'labels' and isinstance(val, (int, list)):
                if self.detected_mode == 'semantic' and isinstance(val, int):
                    value_to_label[val] = key
                elif self.detected_mode == 'rgb' and isinstance(val, list) and len(val) >= 3:
                    value_to_label[tuple(val[:3])] = key

        # Apply to comboboxes
        for value, combo in self.mapping_widgets.items():
            if value in value_to_label:
                label_code = value_to_label[value]
                # Find index in combobox
                idx = combo.findText(label_code)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def validate_inputs(self):
        """Validate that we have valid data for import."""
        if not self.valid_mask_pairs:
            QMessageBox.warning(self, "No Valid Masks", 
                                "No valid mask files found. Please scan masks first.")
            return False

        if not self.unique_values:
            QMessageBox.warning(self, "No Values Detected", 
                                "No unique values detected in masks.")
            return False

        # Check that at least one value is mapped to a label (not all ignored)
        has_mapping = False
        for value, combo in self.mapping_widgets.items():
            if combo.currentIndex() > 0:  # Not "Ignore / Background"
                has_mapping = True
                break

        if not has_mapping:
            result = QMessageBox.question(
                self, "No Mappings",
                "All values are set to 'Ignore / Background'. This will create empty masks.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return False

        return True

    def run_import_process(self):
        """Execute the mask import process."""
        if not self.validate_inputs():
            return

        # Check for existing mask annotations (conflicts)
        conflicts = []
        for mask_path, raster in self.valid_mask_pairs:
            if raster.mask_annotation is not None:
                conflicts.append(raster.basename)

        # Handle conflicts
        overwrite_mode = True  # Default to overwrite
        if conflicts:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Existing Masks Detected")
            msg_box.setText(f"{len(conflicts)} image(s) already have mask annotations.\n\n"
                            "How would you like to proceed?")
            detail_text = "Images with existing masks:\n" + "\n".join(conflicts[:20])
            if len(conflicts) > 20:
                detail_text += "\n..."
            msg_box.setDetailedText(detail_text)
            
            overwrite_btn = msg_box.addButton("Overwrite Existing", QMessageBox.AcceptRole)
            skip_btn = msg_box.addButton("Skip Conflicts", QMessageBox.RejectRole)
            msg_box.addButton(QMessageBox.Cancel)
            
            msg_box.exec_()
            
            if msg_box.clickedButton() == overwrite_btn:
                overwrite_mode = True
            elif msg_box.clickedButton() == skip_btn:
                overwrite_mode = False
            else:
                return  # Cancelled

        # Build the value -> label mapping
        value_to_label = {}
        for value, combo in self.mapping_widgets.items():
            if combo.currentIndex() > 0:  # Not "Ignore / Background"
                label_code = combo.currentText()
                label = self.label_window.get_label_by_short_code(label_code)
                if label:
                    value_to_label[value] = label

        # Filter pairs based on conflict handling
        pairs_to_process = []
        for mask_path, raster in self.valid_mask_pairs:
            if raster.mask_annotation is not None and not overwrite_mode:
                continue  # Skip this one
            pairs_to_process.append((mask_path, raster))

        if not pairs_to_process:
            QMessageBox.information(self, "Nothing to Import", 
                                    "No masks to import after applying conflict settings.")
            return

        # Get current labels for MaskAnnotation initialization
        project_labels = list(self.label_window.labels)

        imported_count = 0
        error_count = 0

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Importing Masks")
            progress_bar.show()
            progress_bar.start_progress(len(pairs_to_process))

            for mask_path, raster in pairs_to_process:
                if progress_bar.wasCanceled():
                    break

                try:
                    # Load the mask
                    with Image.open(mask_path) as img:
                        if self.detected_mode == 'semantic':
                            source_mask = np.array(img.convert('L'))
                        else:
                            source_mask = np.array(img.convert('RGB'))

                    # Create the internal mask array
                    height, width = raster.height, raster.width
                    internal_mask = np.zeros((height, width), dtype=np.uint8)

                    # Create temporary MaskAnnotation to get class ID mapping
                    temp_mask_anno = MaskAnnotation(
                        image_path=raster.image_path,
                        mask_data=internal_mask,
                        initial_labels=project_labels,
                        rasterio_src=raster.rasterio_src
                    )

                    # Translate external values to internal class IDs
                    for ext_value, label in value_to_label.items():
                        internal_class_id = temp_mask_anno.label_id_to_class_id_map.get(label.id)
                        if internal_class_id is None:
                            continue

                        if self.detected_mode == 'semantic':
                            # Simple integer comparison
                            internal_mask[source_mask == ext_value] = internal_class_id
                        else:
                            # RGB comparison
                            r, g, b = ext_value
                            match_mask = (
                                (source_mask[:, :, 0] == r)
                                & (source_mask[:, :, 1] == g)
                                & (source_mask[:, :, 2] == b)
                            )
                            internal_mask[match_mask] = internal_class_id

                    # Update the mask data in the annotation
                    temp_mask_anno.mask_data = internal_mask
                    temp_mask_anno._initialize_canvas()

                    # If overwriting, remove old mask first
                    if raster.mask_annotation is not None:
                        raster.mask_annotation.remove_from_scene()

                    # Assign the new mask annotation to the raster
                    raster.mask_annotation = temp_mask_anno

                    # If this is the currently displayed image, refresh the display
                    if self.annotation_window.current_image_path == raster.image_path:
                        self.annotation_window.load_mask_annotation()

                    imported_count += 1

                except Exception:
                    error_count += 1

                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

        finally:
            QApplication.restoreOverrideCursor()

        # Show summary
        summary_parts = [f"Successfully imported {imported_count} mask(s)."]
        if error_count > 0:
            summary_parts.append(f"{error_count} mask(s) failed to import.")
        
        QMessageBox.information(self, "Import Complete", "\n".join(summary_parts))
        
        # Close dialog on success
        if imported_count > 0:
            self.accept()

    def closeEvent(self, event):
        """Handle dialog close event."""
        event.accept()
