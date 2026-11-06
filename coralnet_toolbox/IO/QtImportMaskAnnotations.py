import os
import ujson as json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QTableWidgetItem,
                             QWidget, QTableWidget, QHeaderView, QAbstractItemView)

from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import build_mask_annotation, unique_mask_values

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox.utilities import STATUS_SUMMARY_MS, plural, show_status


# Pillow's colour count gives up past this many colours and the scan falls back to
# numpy. A mask with that many is anti-aliased or photographic rather than a class
# mask, and would not fit the mapping table anyway.
MAX_SCAN_COLORS = 1 << 16


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

        self.setWindowIcon(get_window_icon("mask.svg"))
        self.setWindowTitle("Import Mask Annotations")
        self.resize(1100, 500)

        # State variables
        self.valid_mask_pairs = []  # List of (mask_path, raster) tuples
        self.detected_mode = None  # 'semantic' (1-channel) or 'rgb' (3-channel)
        self.unique_values = []  # List of unique values found in masks
        self.mapping_widgets = {}  # Maps value -> QComboBox for label selection
        self.pending_labels = {}  # Short code -> label dict, for mapped labels the project lacks
        self.auto_mapping_path = ""  # Class mapping found next to the masks, not chosen by hand
        self.scan_status_text = ""  # Scan summary, kept so the mapping summary can follow it

        # Main layout for the dialog: information on the left, everything else on the right
        self.main_layout = QHBoxLayout(self)

        # Left column - information
        left_column = QVBoxLayout()
        self.setup_info_layout(parent_layout=left_column)
        self.main_layout.addLayout(left_column, 2)

        # Right column - input, mapping table, status, buttons
        right_column = QVBoxLayout()
        self.setup_input_layout(parent_layout=right_column)
        self.setup_mapping_table_layout(parent_layout=right_column)

        # Status label for scan results
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        right_column.addWidget(self.status_label)

        # Bottom buttons
        self.setup_buttons_layout(parent_layout=right_column)
        self.main_layout.addLayout(right_column, 3)

        # Initial UI state
        self.import_button.setEnabled(False)

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
            "• Only <code>.png</code> files are supported<br><br>"
            "<b>Class Mapping (optional):</b> A <code>class_mapping.json</code> written by Export Masks fills in "
            "the table for you, and adds any labels it names that the project does not have yet. One saved next "
            "to the selected masks is picked up automatically."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(info_label)
        layout.addStretch(1)
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_input_layout(self, parent_layout=None):
        """Set up the input directory and scan layout."""
        groupbox = QGroupBox("Input")
        layout = QFormLayout()

        # Directory/file selection
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select PNG mask files...")
        self.input_path_edit.setToolTip("Path(s) to PNG mask files.\nMultiple files can be selected, separated by semicolons.")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_input)
        self.browse_button.setToolTip("Browse for PNG mask files to import.")
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.browse_button)
        layout.addRow("Masks:", input_layout)

        # Optional class mapping, which fills in the value to label table after a scan
        mapping_layout = QHBoxLayout()
        self.mapping_path_edit = QLineEdit()
        self.mapping_path_edit.setPlaceholderText("Optional class_mapping.json...")
        self.mapping_path_edit.setToolTip("A class_mapping.json written by Export Masks.\n"
                                          "Its values fill in the mapping table after a scan.\n"
                                          "One saved next to the selected masks is picked up automatically.")
        self.mapping_browse_button = QPushButton("Browse...")
        self.mapping_browse_button.clicked.connect(self.browse_class_mapping)
        self.mapping_browse_button.setToolTip("Browse for a class mapping JSON file.")
        mapping_layout.addWidget(self.mapping_path_edit)
        mapping_layout.addWidget(self.mapping_browse_button)
        layout.addRow("Class Mapping:", mapping_layout)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

        # Scan button (outside groupbox)
        self.scan_button = QPushButton("Scan Masks")
        self.scan_button.clicked.connect(self.scan_masks)
        self.scan_button.setToolTip("Scan the selected mask files to detect unique values and match them to project images.")
        parent_layout.addWidget(self.scan_button)

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
        self.mapping_table.setToolTip("Select which project label each detected mask value should be mapped to.")
        
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.mapping_table.hide()
        layout.addWidget(self.mapping_table)

        groupbox.setLayout(layout)
        parent_layout.addWidget(groupbox)

    def setup_buttons_layout(self, parent_layout=None):
        """Set up the bottom action buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.run_import_process)
        self.import_button.setToolTip("Import the mask annotations using the specified value-to-label mappings.")

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setToolTip("Close this dialog without importing.")

        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def browse_input(self):
        """Open file dialog to select mask files."""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Mask Files",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_paths:
            self.input_path_edit.setText(";".join(file_paths))
            self.auto_detect_class_mapping(file_paths)

    def browse_class_mapping(self):
        """Open file dialog to select a class mapping file, applying it if masks are scanned."""
        start_dir = os.path.dirname(self.mapping_path_edit.text().strip())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Class Mapping File",
            start_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        # Chosen by hand, so a later mask selection must not replace it
        self.auto_mapping_path = ""
        self.mapping_path_edit.setText(file_path)

        # Rebuild the table so selections from a previous mapping do not linger
        if self.mapping_widgets:
            self.populate_mapping_table()

    def auto_detect_class_mapping(self, mask_files):
        """
        Pick up the class mapping Export Masks saved next to the masks.

        Only fills the field when it is empty or holds an earlier automatic pick, so a
        file the user chose by hand is never replaced.
        """
        current = self.mapping_path_edit.text().strip()
        if current and current != self.auto_mapping_path:
            return

        found = ""
        if mask_files:
            mask_dir = os.path.dirname(mask_files[0])
            # color_legend.json is what Visualization exports wrote before class_mapping.json
            for name in ("class_mapping.json", "color_legend.json"):
                candidate = os.path.join(mask_dir, name).replace("\\", "/")
                if os.path.isfile(candidate):
                    found = candidate
                    break

        self.auto_mapping_path = found
        self.mapping_path_edit.setText(found)

    def get_mask_files(self):
        """Get list of mask files from the input path."""
        input_text = self.input_path_edit.text().strip()
        if not input_text:
            return []

        mask_files = []
        # Parse semicolon-separated list of files
        paths = input_text.split(";")
        for path in paths:
            path = path.strip()
            if path and os.path.isfile(path) and path.lower().endswith('.png'):
                mask_files.append(path)

        return mask_files

    def read_mask_array(self, img):
        """
        Read a mask's pixels in the form the detected mode takes.

        Index masks are read as stored: convert('L') would turn a palette PNG's
        indices, which are its class IDs, into the brightness of their palette
        colours, and clip a 16-bit mask's values to 255.

        np.asarray rather than np.array, and no convert when the mode already
        fits, each save a copy of every pixel (116 ms -> 76 ms for a 24 MP RGB
        mask). The array is read-only.
        """
        if self.detected_mode != 'semantic':
            return np.asarray(img if img.mode == 'RGB' else img.convert('RGB'))
        if img.mode in ('L', 'P', 'I') or img.mode.startswith('I;16'):
            return np.asarray(img)
        return np.asarray(img.convert('L'))

    def read_mask_values(self, mask_path):
        """
        Return the distinct values in a mask file, in the form the detected mode takes.

        Pillow counts them in C from its own buffer -- a histogram for index
        masks, a colour table for RGB ones -- without a numpy copy of the pixels:
        ~2.5x faster for index masks and ~5x for RGB. Decoding releases the GIL,
        so scan_masks reads files in parallel (counting RGB colours does not,
        which caps their gain at ~1.4x). Masks Pillow cannot count this way go
        through numpy instead.

        Safe to run on a worker thread: it reads only detected_mode, which is
        fixed before any values are read.
        """
        with Image.open(mask_path) as img:
            if self.detected_mode == 'semantic':
                if img.mode in ('L', 'P'):
                    return [value for value, count in enumerate(img.histogram()) if count]
            else:
                colors = (img if img.mode == 'RGB' else img.convert('RGB')).getcolors(MAX_SCAN_COLORS)
                if colors is not None:
                    return [color for _, color in colors]
            return unique_mask_values(self.read_mask_array(img))

    def read_candidate_values(self, candidates, progress_bar):
        """
        Read the values of each (mask_path, raster) candidate on worker threads.

        Eight index masks scan in 54 ms rather than 305 ms. Results are taken in
        file order, so the import order is the same as a serial scan's.

        Returns:
            tuple: (set of values, list of the (mask_path, raster) pairs read,
            number of masks that could not be read)
        """
        values = set()
        read_pairs = []
        unreadable = 0

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            futures = [pool.submit(self.read_mask_values, mask_path) for mask_path, _ in candidates]
            for pair, future in zip(candidates, futures, strict=True):
                if progress_bar.wasCanceled():
                    for pending in futures:
                        pending.cancel()
                    break

                try:
                    values.update(future.result())
                    read_pairs.append(pair)
                except Exception:
                    unreadable += 1

                progress_bar.update_progress()

        return values, read_pairs, unreadable

    def scan_masks(self):
        """Scan selected mask files and detect unique values."""
        mask_files = self.get_mask_files()
        
        if not mask_files:
            QMessageBox.warning(self, "No Files", "No PNG mask files found. Please select valid mask files.")
            return

        # Mask paths may have been typed rather than browsed
        self.auto_detect_class_mapping(mask_files)

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

        show_status(self.main_window, f"Scanning {plural(len(mask_files), 'mask file')}...")

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Scanning Masks")
            progress_bar.show()
            progress_bar.start_progress(len(mask_files))

            # Match and size-check every mask from its header alone, which also
            # settles the mode that reading the values depends on
            candidates = []
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

                except Exception:
                    skipped_no_match += 1
                    progress_bar.update_progress()
                    continue

                candidates.append((mask_path, raster))

            if candidates and not progress_bar.wasCanceled():
                all_unique_values, self.valid_mask_pairs, unreadable = self.read_candidate_values(candidates,
                                                                                                  progress_bar)
                matched_count = len(self.valid_mask_pairs)
                skipped_no_match += unreadable

            progress_bar.stop_progress()
            progress_bar.close()

        except Exception as e:
            show_status(self.main_window, f"Mask scan failed: {e}", STATUS_SUMMARY_MS)
            raise

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

        self.scan_status_text = " | ".join(status_parts)
        self.status_label.setText(self.scan_status_text)
        show_status(self.main_window, f"Mask scan: {self.scan_status_text}", STATUS_SUMMARY_MS)

        # Populate the mapping table
        if self.valid_mask_pairs and self.unique_values:
            self.populate_mapping_table()
            self.import_button.setEnabled(True)
        else:
            self.placeholder_label.setText("No valid masks found. Check file names and dimensions.")
            self.placeholder_label.show()
            self.mapping_table.hide()
            self.import_button.setEnabled(False)

    def populate_mapping_table(self):
        """Populate the mapping table with detected values."""
        self.placeholder_label.hide()
        self.mapping_table.show()
        
        self.mapping_table.setRowCount(0)
        self.mapping_widgets = {}
        self.pending_labels = {}

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
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addStretch(1)
                layout.addWidget(swatch)
                layout.addStretch(1)
                
                # Set RGB values as tooltip
                container.setToolTip(f"({value[0]}, {value[1]}, {value[2]})")
                
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

        self.apply_class_mapping()

    def apply_class_mapping(self):
        """Set each detected value's label from the class mapping file, if one is given."""
        self.status_label.setText(self.scan_status_text)

        mapping_path = self.mapping_path_edit.text().strip()
        if not mapping_path or not self.mapping_widgets:
            return

        try:
            value_to_label, duplicate_count = self.read_class_mapping(mapping_path)
        except Exception as e:
            QMessageBox.warning(self, "Class Mapping Not Applied",
                                f"Could not read the class mapping file:\n{mapping_path}\n\n{e}")
            return

        matched_count = 0
        for value, combo in self.mapping_widgets.items():
            # Detected values are plain ints or int tuples, as the file's are once parsed
            if value not in value_to_label:
                continue

            matched_count += 1
            label_dict = value_to_label[value]
            if label_dict is None:
                combo.setCurrentIndex(0)  # The file's background
                continue

            label_code = self.resolve_mapping_label(label_dict)
            if not label_code:
                continue

            # A label the project lacks becomes a choice in every row, so it can be picked anywhere
            if label_code in self.pending_labels:
                for other_combo in self.mapping_widgets.values():
                    if other_combo.findText(label_code) == -1:
                        other_combo.addItem(label_code)

            combo.setCurrentIndex(combo.findText(label_code))

        mapping_parts = [f"class mapping: {matched_count} of {len(self.mapping_widgets)} value(s) matched"]
        if self.pending_labels:
            mapping_parts.append(f"{len(self.pending_labels)} new label(s) to add on import")
        if duplicate_count:
            mapping_parts.append(f"{duplicate_count} value(s) shared by several labels, first kept")
        self.status_label.setText(" | ".join([self.scan_status_text] + mapping_parts))

    def read_class_mapping(self, mapping_path):
        """
        Read a class mapping file into {mask value: label dict} for the detected mask mode.

        Accepts the class_mapping.json Export Masks writes, whose entries carry an integer
        "index" (Semantic, SfM) or an RGB "color" (Visualization), and the older
        color_legend.json, which maps a label code straight to [R, G, B]. The background
        maps to None so it stays on Ignore / Background. Where several labels share a
        value (SfM masks, typically) the first one wins.

        Returns:
            tuple: (value_to_label dict, number of values claimed by more than one label)
        """
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object keyed by label code.")

        value_key = 'index' if self.detected_mode == 'semantic' else 'color'
        value_to_label = {}
        duplicates = set()

        for name, entry in data.items():
            if str(name).startswith('_'):
                continue  # Settings, such as the Overlay blend record in color_legend.json

            if isinstance(entry, dict):
                raw_value = entry.get(value_key)
                label = entry.get('label')
            else:
                raw_value = entry
                label = name

            value = self._parse_mapping_value(raw_value)
            if value is None:
                continue  # Not a value these masks can hold, e.g. a color for 1-channel masks

            if not isinstance(label, dict):
                # Only a code is known: the background, or a color_legend.json entry, whose
                # color is the label's own since Visualization draws labels in their colors
                code = str(label).strip()
                if code.lower() == 'background':
                    label = None
                elif value_key == 'color':
                    label = {'short_label_code': code, 'color': list(value)}
                else:
                    label = {'short_label_code': code}

            if value in value_to_label:
                duplicates.add(value)
                continue
            value_to_label[value] = label

        return value_to_label, len(duplicates)

    def _parse_mapping_value(self, raw_value):
        """Return a mapping entry's value in the form the detected values take, or None."""
        if self.detected_mode == 'semantic':
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return None
            return int(raw_value)

        if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 3:
            try:
                return tuple(int(channel) for channel in raw_value[:3])
            except (TypeError, ValueError):
                return None
        return None

    def _find_project_label(self, short_label_code):
        """Return the project label with this short code (case-insensitive), or None."""
        code = short_label_code.strip().lower()
        for label in self.label_window.labels:
            if label.short_label_code.strip().lower() == code:
                return label
        return None

    def resolve_mapping_label(self, label_dict):
        """
        Return the short code a mapping entry resolves to in this project.

        The label's ID is tried first, so a label renamed since the export still matches.
        A label the project does not have is queued in pending_labels and only created
        when the import runs, so a cancelled dialog leaves the project untouched.
        """
        code = str(label_dict.get('short_label_code') or '').strip()
        if not code:
            return None

        label_id = label_dict.get('id')
        if label_id:
            for label in self.label_window.labels:
                if label.id == label_id:
                    return label.short_label_code

        existing = self._find_project_label(code)
        if existing is not None:
            return existing.short_label_code

        self.pending_labels.setdefault(code, label_dict)
        return code

    def get_or_create_label(self, short_label_code):
        """
        Return the project label for a combo choice, creating it if the mapping queued it.

        Returns:
            tuple: (label or None, whether a label was created)
        """
        existing = self._find_project_label(short_label_code)
        if existing is not None:
            return existing, False

        label_dict = self.pending_labels.get(short_label_code)
        if label_dict is None:
            return None, False

        color = None
        raw_color = label_dict.get('color')
        if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
            try:
                color = QColor(*[int(channel) for channel in raw_color[:4]])
            except (TypeError, ValueError):
                color = None

        label = self.label_window.add_label_if_not_exists(
            short_label_code,
            label_dict.get('long_label_code') or short_label_code,
            color=color,
            label_id=label_dict.get('id'),
            refresh_ui=False
        )
        return label, True

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
            
            overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.AcceptRole)
            skip_btn = msg_box.addButton("Skip", QMessageBox.RejectRole)
            msg_box.addButton("Cancel", QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            if msg_box.clickedButton() == overwrite_btn:
                overwrite_mode = True
            elif msg_box.clickedButton() == skip_btn:
                overwrite_mode = False
            else:
                return  # Cancelled

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

        # Build the value -> label mapping. Labels the class mapping named but the project
        # lacks are created here, once the import is certain to run.
        value_to_label = {}
        labels_created = False
        for value, combo in self.mapping_widgets.items():
            if combo.currentIndex() > 0:  # Not "Ignore / Background"
                label, created = self.get_or_create_label(combo.currentText())
                labels_created = labels_created or created
                if label:
                    value_to_label[value] = label

        if labels_created:
            self.label_window.refresh_after_batch_add()

        # Get current labels for MaskAnnotation initialization
        project_labels = list(self.label_window.labels)

        imported_paths = []
        error_count = 0

        show_status(self.main_window, f"Importing {plural(len(pairs_to_process), 'mask')}...")

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
                        source_mask = self.read_mask_array(img)

                    # Shared with the semantic dataset importer, so the
                    # external-value to class-ID translation lives in one place.
                    temp_mask_anno = build_mask_annotation(
                        image_path=raster.image_path,
                        source_mask=source_mask,
                        value_to_label=value_to_label,
                        project_labels=project_labels,
                        shape=(raster.height, raster.width),
                        rasterio_src=raster.rasterio_src
                    )

                    # If overwriting, remove old mask first
                    if raster.mask_annotation is not None:
                        raster.mask_annotation.remove_from_scene()

                    # Assign the new mask annotation to the raster
                    raster.mask_annotation = temp_mask_anno

                    # If this is the currently displayed image, refresh the display
                    if self.annotation_window.current_image_path == raster.image_path:
                        self.annotation_window.load_mask_annotation()

                    imported_paths.append(raster.image_path)

                except Exception:
                    error_count += 1

                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

        except Exception as e:
            show_status(self.main_window, f"Mask import failed: {e}", STATUS_SUMMARY_MS)
            raise

        finally:
            QApplication.restoreOverrideCursor()

        self.refresh_image_table(imported_paths)

        # Show summary
        summary_parts = [f"Successfully imported {len(imported_paths)} mask(s)."]
        if error_count > 0:
            summary_parts.append(f"{error_count} mask(s) failed to import.")
        
        show_status(self.main_window, f"Masks: {' '.join(summary_parts)}", STATUS_SUMMARY_MS)
        QMessageBox.information(self, "Import Complete", "\n".join(summary_parts))
        
        # Close dialog on success
        self.accept()

    def refresh_image_table(self, image_paths):
        """
        Bring the Image Window's annotation counts up to date for the masks just imported.

        A raster's count and mask flag are cached, and only update_image_annotations
        recomputes them and repaints the row; assigning raster.mask_annotation does
        neither. The filters run again too, since "has mask" and the like now differ.
        """
        if not image_paths:
            return

        for image_path in image_paths:
            try:
                self.image_window.update_image_annotations(image_path, update_counts=False)
            except Exception as e:
                print(f"Warning: could not refresh annotation info for {image_path}: {e}")

        try:
            self.label_window.update_annotation_count()
        except Exception as e:
            print(f"Warning: could not refresh the label annotation count: {e}")

        try:
            self.image_window.filter_images(use_threading=False)
        except Exception as e:
            print(f"Warning: could not refresh the image filters: {e}")

    def closeEvent(self, event):
        """Handle dialog close event."""
        event.accept()
