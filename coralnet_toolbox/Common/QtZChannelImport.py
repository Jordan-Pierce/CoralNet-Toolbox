import sys
import os
import difflib

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QSize
from PyQt5.QtGui import QColor, QIcon, QDrag, QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel, QPushButton, QListWidget, 
                             QAbstractItemView, QFileDialog, QSplitter, QMessageBox,
                             QGroupBox, QFormLayout, QScrollArea, QMenu, QComboBox)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import (
    detect_z_channel_units_from_file,
    normalize_z_unit,
    get_standard_z_units
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


COLOR_MATCHED = QColor(220, 255, 220)   # Light Green
COLOR_PARTIAL = QColor(255, 255, 220)   # Light Yellow
COLOR_MISSING = QColor(255, 220, 220)   # Light Red
COLOR_CONFLICT = QColor(255, 200, 150)  # Orange


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZDropTable(QTableWidget):
    """
    A specialized TableWidget that accepts drops from the sidebar list.
    Renamed to generic 'Z' to handle Depth, Height, DEM, etc.
    """
    fileDropped = pyqtSignal(int, str)  # row, filepath
    clearMapping = pyqtSignal(list)  # list of rows
    setBulkUnits = pyqtSignal(list)  # list of rows (for bulk units operation)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setColumnCount(4)
        
        # Updated Headers to include Units column
        self.setHorizontalHeaderLabels(["Image Source", "Z Channel", "Z Units", "Status"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.setColumnWidth(3, 100)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        position = event.pos()
        row = self.rowAt(position.y())
        
        if row >= 0 and event.mimeData().hasText():
            filepaths = event.mimeData().text().split(";")
            # Emit signal for each file starting from the dropped row
            for i, filepath in enumerate(filepaths):
                target_row = row + i
                if target_row < self.rowCount():
                    self.fileDropped.emit(target_row, filepath)
            event.accept()
        else:
            event.ignore()

    def show_context_menu(self, position):
        """Show context menu for row operations."""
        selected_rows = sorted(set(index.row() for index in self.selectedIndexes()))
        
        if selected_rows:
            menu = QMenu(self)
            
            # Set units action
            if len(selected_rows) == 1:
                units_action = menu.addAction("Set Z-Channel Units")
            else:
                units_action = menu.addAction(f"Set Z-Channel Units for {len(selected_rows)} rows")
            
            menu.addSeparator()
            
            # Clear mapping action
            if len(selected_rows) == 1:
                clear_action = menu.addAction("Clear Z-Channel Mapping")
            else:
                clear_action = menu.addAction(f"Clear Z-Channel Mapping ({len(selected_rows)} rows)")
            
            action = menu.exec_(self.mapToGlobal(position))
            if action == clear_action:
                self.clearMapping.emit(selected_rows)
            elif action == units_action:
                self.setBulkUnits.emit(selected_rows)


class DraggableList(QListWidget):
    """
    A list widget that allows dragging filenames out of it.
    """
    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()
        if not selected_items:
            return
        
        # Get all selected file paths in order
        filepaths = [item.data(Qt.UserRole) for item in selected_items]
        
        # Store as semicolon-separated list to preserve order
        mime_data = QMimeData()
        mime_data.setText(";".join(filepaths))
        
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # Use the first item for visual feedback
        pixmap = QPixmap(selected_items[0].sizeHint())
        if not pixmap.isNull():
            painter = QPainter(pixmap)
            painter.drawRect(pixmap.rect())
            painter.end()
            drag.setPixmap(pixmap)

        drag.exec_(Qt.CopyAction)


class ZPairingWidget(QWidget):
    """
    Z-Channel Import Widget.
    
    Allows users to pair image files with z-channel files (depth, height, DEM, etc).
    Supports automatic matching with manual override via drag-and-drop.
    
    Input: list of image paths, list of z_files (depth/height/dem)
    Output: signal 'mapping_confirmed' carrying dict {img_path: z_path}
    """
    mapping_confirmed = pyqtSignal(dict)

    def __init__(self, image_files, z_files):
        super().__init__()
        
        self.setWindowTitle("Z-Channel Import")
        self.setWindowIcon(get_icon("z.png"))
        self.resize(1050, 600)
        
        # Set busy cursor while loading and matching
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            self.image_files = sorted(image_files)
            # Smart sort z_files to align with image order
            self.z_files = self._smart_sort_z_files(sorted(z_files), self.image_files)
            self.mapping = {}  # {image_path: {"z_path": path_or_none, "units": unit_str, "status": status_string}}
            
            # Broadened suffixes to cover DEMs, Height maps, etc.
            self.suffixes = ['_depth', '_z', '_dem', '_height', '_d', 'depth', 'z', 'dem']
            
            # Main layout
            self.main_layout = QVBoxLayout(self)
            
            # Setup UI sections
            self.setup_info_layout()
            self.setup_pairing_layout()
            self.setup_buttons_layout()
            
            # Auto-match and populate
            self.run_auto_match()
            self.populate_ui()
            
            # Update z-file list colors to show which are in use
            self.update_z_file_colors()
        finally:
            # Restore normal cursor when dialog is ready to show
            QApplication.restoreOverrideCursor()

    def _smart_sort_z_files(self, z_files, image_files):
        """
        Intelligently reorder z_files to align with image_files order.
        Matches by filename similarity and preserves unmatched files at the end.
        
        Args:
            z_files (list): List of z-channel file paths
            image_files (list): List of image file paths
            
        Returns:
            list: Reordered z_files list
        """
        if not z_files or not image_files:
            return z_files
        
        sorted_z_files = []
        remaining_z_files = list(z_files)
        
        # For each image, try to find and match a corresponding z-file
        for img_path in image_files:
            img_basename = os.path.splitext(os.path.basename(img_path))[0].lower()
            
            best_match = None
            best_match_idx = -1
            
            # Look for exact name match or best fuzzy match
            for idx, z_path in enumerate(remaining_z_files):
                z_basename = os.path.splitext(os.path.basename(z_path))[0].lower()
                
                # Exact match (same name)
                if img_basename == z_basename:
                    best_match = z_path
                    best_match_idx = idx
                    break
                
                # Check if z_basename contains or starts with img_basename
                if img_basename in z_basename or z_basename.startswith(img_basename):
                    if best_match is None:
                        best_match = z_path
                        best_match_idx = idx
            
            # Add the matched file (or None placeholder if no match found yet)
            if best_match is not None:
                sorted_z_files.append(best_match)
                remaining_z_files.pop(best_match_idx)
            else:
                # Leave a placeholder - will be filled in later or left empty
                sorted_z_files.append(None)
        
        # Append any remaining unmatched z-files at the end
        sorted_z_files.extend(remaining_z_files)
        
        # Filter out None placeholders to get actual list
        actual_z_files = [z for z in sorted_z_files if z is not None]
        
        # Return the actual files, but maintain length for display purposes
        return actual_z_files if actual_z_files else z_files

    def setup_info_layout(self):
        """Set up the information section."""
        group_box = QGroupBox("Instructions")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        
        info_text = (
            "Import Z-channel files (Depth, Height, DEM) and pair them with your images.<br><br>"
            "<b>Automatic Matching:</b> The system will attempt to match files using exact names, "
            "suffixes, or fuzzy matching.<br><br>"
            "<b>Manual Correction:</b> Drag Z-channel files from the right panel onto table rows "
            "to override automatic matches or fill missing ones. Select multiple files with Ctrl/Shift "
            "click to batch-map in order.<br><br>"
            "<b>Units:</b> Double-click on a Z Units cell to change units, or select multiple rows and "
            "right-click to set units for all selected rows at once."
        )
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        group_box.setMaximumHeight(150)
        self.main_layout.addWidget(group_box)

    def setup_pairing_layout(self):
        """Set up the main pairing table and file list."""
        # Create horizontal splitter for two group boxes
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Z-Channel Pairing Group with Scroll Area
        pairing_group = QGroupBox("Z-Channel Pairing")
        pairing_layout = QVBoxLayout()
        
        pairing_scroll = QScrollArea()
        pairing_scroll.setWidgetResizable(True)
        pairing_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pairing_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pairing_scroll.setMinimumHeight(400)
        pairing_scroll.setMinimumWidth(400)
        
        self.table = ZDropTable()
        self.table.fileDropped.connect(self.handle_manual_drop)
        self.table.cellDoubleClicked.connect(self.handle_cell_click)
        self.table.clearMapping.connect(self.handle_clear_mapping)
        self.table.setBulkUnits.connect(self.show_bulk_units_dialog)
        pairing_scroll.setWidget(self.table)
        
        pairing_layout.addWidget(pairing_scroll)
        pairing_group.setLayout(pairing_layout)
        splitter.addWidget(pairing_group)
        
        # Right: Available Z Files Group with Scroll Area
        files_group = QGroupBox("Available Z-Channel Files")
        files_layout = QVBoxLayout()
        
        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        files_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        files_scroll.setMinimumHeight(400)
        files_scroll.setMinimumWidth(200)
        
        self.list_widget = DraggableList()
        for z_file in self.z_files:
            self.list_widget.addItem(os.path.basename(z_file))
            item = self.list_widget.item(self.list_widget.count() - 1)
            item.setTextAlignment(Qt.AlignCenter)
            item.setData(Qt.UserRole, z_file)
        
        files_scroll.setWidget(self.list_widget)
        files_layout.addWidget(files_scroll)
        files_group.setLayout(files_layout)
        splitter.addWidget(files_group)
        
        splitter.setSizes([650, 300])
        self.main_layout.addWidget(splitter)

    def setup_buttons_layout(self):
        """Set up the bottom buttons section."""
        btn_layout = QHBoxLayout()
        
        # Left side buttons (Clear All operations)
        btn_clear_mapping = QPushButton("Clear All Mappings")
        btn_clear_mapping.clicked.connect(self.clear_all_mappings)
        btn_layout.addWidget(btn_clear_mapping)
        
        btn_clear_units = QPushButton("Clear All Units")
        btn_clear_units.clicked.connect(self.clear_all_units)
        btn_layout.addWidget(btn_clear_units)
        
        # Add stretch to push right-side buttons to the right
        btn_layout.addStretch()
        
        # Right side buttons (Cancel and Confirm)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        btn_layout.addWidget(btn_cancel)
        
        btn_confirm = QPushButton("Confirm Mapping")
        btn_confirm.clicked.connect(self.finalize_mapping)
        btn_layout.addWidget(btn_confirm)
        
        self.main_layout.addLayout(btn_layout)

    def run_auto_match(self):
        """
        Automatically pair images with z-channel files using heuristic matching.
        Also attempts to detect units from matched z-channel files.
        
        Matching priority:
        1. Exact name match (same filename without extension)
        2. Suffix match (image name + known z-suffix)
        3. Fuzzy match (string similarity)
        """
        for img in self.image_files:
            img_name = os.path.splitext(os.path.basename(img))[0].lower()
            best_match = None
            match_type = "Missing"
            detected_units = None
            
            # 1. Exact Name Match
            for zf in self.z_files:
                z_name = os.path.splitext(os.path.basename(zf))[0].lower()
                if img_name == z_name:
                    best_match = zf
                    match_type = "Auto (Exact)"
                    break
            
            # 2. Suffix Match
            if not best_match:
                candidates = []
                for zf in self.z_files:
                    z_name = os.path.splitext(os.path.basename(zf))[0].lower()
                    if z_name.startswith(img_name):
                        remainder = z_name[len(img_name):]
                        # Check if remainder is a known Z suffix
                        if remainder in self.suffixes or remainder.strip("_") in self.suffixes:
                            candidates.append(zf)
                
                if len(candidates) == 1:
                    best_match = candidates[0]
                    match_type = "Auto (Suffix)"
                elif len(candidates) > 1:
                    best_match = None 
                    match_type = "Conflict"

            # 3. Fuzzy Fallback
            if not best_match and match_type != "Conflict":
                matches = difflib.get_close_matches(
                    os.path.basename(img), 
                    [os.path.basename(z) for z in self.z_files], 
                    n=1, 
                    cutoff=0.8
                )
                if matches:
                    match_name = matches[0]
                    full_path = next(p for p in self.z_files if os.path.basename(p) == match_name)
                    best_match = full_path
                    match_type = "Review (Fuzzy)"
            
            # Detect units from matched z-channel file
            if best_match:
                detected_units, confidence = detect_z_channel_units_from_file(best_match)
                if detected_units:
                    detected_units = normalize_z_unit(detected_units)
                    if confidence == 'high':
                        match_type += " ✓"

            # Store the mapping with units
            self.mapping[img] = {
                "z_path": best_match,
                "units": detected_units,
                "status": match_type
            }

    def populate_ui(self):
        """Populate the pairing table with matched files."""
        self.table.setRowCount(len(self.image_files))
        
        for r, img_path in enumerate(self.image_files):
            match_data = self.mapping[img_path]
            z_path = match_data["z_path"]
            units = match_data.get("units", None)
            status = match_data["status"]

            # Column 0: Image Name
            item_img = QTableWidgetItem(os.path.basename(img_path))
            item_img.setTextAlignment(Qt.AlignCenter)
            item_img.setToolTip(img_path)
            self.table.setItem(r, 0, item_img)
            
            # Column 1: Z Channel
            display_text = os.path.basename(z_path) if z_path else "None (Drag Here)"
            item_z = QTableWidgetItem(display_text)
            item_z.setTextAlignment(Qt.AlignCenter)
            if z_path:
                item_z.setToolTip(z_path)
            self.table.setItem(r, 1, item_z)
            
            # Column 2: Z Units (with dropdown for user selection)
            units_display = units if units else "(Set)"
            item_units = QTableWidgetItem(units_display)
            item_units.setTextAlignment(Qt.AlignCenter)
            item_units.setToolTip("Click to edit units")
            self.table.setItem(r, 2, item_units)
            
            # Column 3: Status
            item_status = QTableWidgetItem(status)
            item_status.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 3, item_status)
            
            # Color code the row based on match status
            self.update_row_color(r, status)

    def update_row_color(self, row, status):
        """
        Color code table rows based on match status.
        
        Args:
            row (int): Table row index
            status (str): Match status string
        """
        color = Qt.white
        if "Auto" in status:
            color = COLOR_MATCHED
        elif "Review" in status:
            color = COLOR_PARTIAL
        elif "Missing" in status:
            color = COLOR_MISSING
        elif "Conflict" in status:
            color = COLOR_CONFLICT
            
        for c in range(3):
            self.table.item(row, c).setBackground(color)
    
    def update_z_file_colors(self):
        """
        Update the colors of z-files in the right panel to show which are in use.
        Files that are matched get a light green background.
        Files that are unmatched remain white.
        """
        # Get all currently mapped z-file paths
        mapped_z_paths = set()
        for img_path, data in self.mapping.items():
            if data["z_path"]:
                mapped_z_paths.add(data["z_path"])
        
        # Update each item in the list
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            z_path = item.data(Qt.UserRole)
            
            # Color the item based on whether it's in use
            if z_path in mapped_z_paths:
                item.setBackground(COLOR_MATCHED)  # Light green for in-use files
            else:
                item.setBackground(Qt.white)  # White for available files

    def handle_manual_drop(self, row, z_path):
        """
        Handle manual pairing when user drags a z-file to a table row.
        
        Args:
            row (int): Table row index
            z_path (str): Path to the z-channel file
        """
        img_path = self.image_files[row]
        
        # Detect units from the manually dropped file
        detected_units, confidence = detect_z_channel_units_from_file(z_path)
        if detected_units:
            detected_units = normalize_z_unit(detected_units)
        
        # Update mapping data
        self.mapping[img_path]["z_path"] = z_path
        self.mapping[img_path]["units"] = detected_units
        self.mapping[img_path]["status"] = "Manual Fix"
        
        # Update table display
        item_z = self.table.item(row, 1)
        item_z.setText(os.path.basename(z_path))
        item_z.setTextAlignment(Qt.AlignCenter)
        item_z.setToolTip(z_path)
        
        item_units = self.table.item(row, 2)
        units_display = detected_units if detected_units else "(Set)"
        item_units.setText(units_display)
        item_units.setTextAlignment(Qt.AlignCenter)
        
        item_status = self.table.item(row, 3)
        item_status.setText("Manual Fix")
        item_status.setTextAlignment(Qt.AlignCenter)
        
        # Update row color to indicate successful match
        self.update_row_color(row, "Auto")
        
        # Update z-file list colors to show which files are now in use
        self.update_z_file_colors()

    def handle_clear_mapping(self, rows):
        """
        Clear the z-channel mapping for multiple rows.
        
        Args:
            rows (list): List of table row indices
        """
        for row in rows:
            img_path = self.image_files[row]
            
            # Clear mapping data
            self.mapping[img_path]["z_path"] = None
            self.mapping[img_path]["units"] = None
            self.mapping[img_path]["status"] = "Missing"
            
            # Update table display
            item_z = self.table.item(row, 1)
            item_z.setText("None (Drag Here)")
            item_z.setTextAlignment(Qt.AlignCenter)
            item_z.setToolTip("")
            
            item_units = self.table.item(row, 2)
            item_units.setText("(Set)")
            item_units.setTextAlignment(Qt.AlignCenter)
            item_units.setToolTip("Click to edit units")
            
            item_status = self.table.item(row, 3)
            item_status.setText("Missing")
            item_status.setTextAlignment(Qt.AlignCenter)
            
            # Update row color to indicate missing match
            self.update_row_color(row, "Missing")
        
        # Update z-file list colors to show which files are now available
        self.update_z_file_colors()

    def handle_cell_click(self, row, col):
        """
        Handle double-click on z-path cell to browse for file or units cell to edit.
        
        Args:
            row (int): Table row index
            col (int): Table column index
        """
        if col == 1:  # Z Channel Column
            fname, _ = QFileDialog.getOpenFileName(self, "Select Z-Channel File")
            if fname:
                self.handle_manual_drop(row, fname)
        elif col == 2:  # Z Units Column
            self.show_units_dialog(row)

    def show_units_dialog(self, row):
        """
        Show a dialog to allow user to select z-channel units for a single row.
        Units selection is dropdown-only (no custom text entry).
        
        Args:
            row (int): Table row index
        """
        from PyQt5.QtWidgets import QInputDialog
        
        img_path = self.image_files[row]
        standard_units = get_standard_z_units()
        
        selected, ok = QInputDialog.getItem(
            self,
            "Select Z-Channel Units",
            f"Units for {os.path.basename(img_path)}:",
            standard_units,
            editable=False
        )
        
        if ok:
            # Update mapping and table
            self.mapping[img_path]["units"] = selected
            item_units = self.table.item(row, 2)
            item_units.setText(selected if selected else "(Set)")
    
    def show_bulk_units_dialog(self, rows):
        """
        Show a dialog to set z-channel units for multiple rows at once.
        
        Args:
            rows (list): List of table row indices
        """
        from PyQt5.QtWidgets import QInputDialog
        
        standard_units = get_standard_z_units()
        
        selected, ok = QInputDialog.getItem(
            self,
            "Set Z-Channel Units for Multiple Rows",
            f"Set units for {len(rows)} selected row{'s' if len(rows) > 1 else ''}:",
            standard_units,
            editable=False
        )
        
        if ok:
            # Apply to all selected rows
            for row in rows:
                img_path = self.image_files[row]
                self.mapping[img_path]["units"] = selected
                item_units = self.table.item(row, 2)
                item_units.setText(selected if selected else "(Set)")

    def clear_all_mappings(self):
        """Clear all z-channel mappings for all rows."""
        reply = QMessageBox.question(
            self,
            "Clear All Mappings",
            "Are you sure you want to clear all z-channel mappings?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear all rows at once
            all_rows = list(range(len(self.image_files)))
            self.handle_clear_mapping(all_rows)

    def clear_all_units(self):
        """Clear all z-channel units for all rows."""
        reply = QMessageBox.question(
            self,
            "Clear All Units",
            "Are you sure you want to clear all z-channel units?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear units for all rows
            for row in range(len(self.image_files)):
                img_path = self.image_files[row]
                self.mapping[img_path]["units"] = None
                
                item_units = self.table.item(row, 2)
                item_units.setText("(Set)")
                item_units.setTextAlignment(Qt.AlignCenter)

    def finalize_mapping(self):
        """
        Finalize the mapping and emit confirmation signal.
        
        Warns user if there are unmatched images.
        Emits full mapping dict with z_path and units.
        """
        final_dict = {}
        missing_count = 0
        
        # Build final mapping with only paired images, preserving units
        for img, data in self.mapping.items():
            if data["z_path"]:
                final_dict[img] = {
                    "z_path": data["z_path"],
                    "units": data.get("units", None)
                }
            else:
                missing_count += 1
        
        # Warn if there are unpaired images
        if missing_count > 0:
            reply = QMessageBox.question(
                self, 
                "Missing Z-Channel Files", 
                f"{missing_count} image(s) have no z-channel file associated.\n\n"
                f"Continue with only {len(final_dict)} paired file(s)?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Emit the final mapping and close
        self.mapping_confirmed.emit(final_dict)
        self.close()
