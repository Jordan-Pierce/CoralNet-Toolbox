import sys
import os
import difflib

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QSize
from PyQt5.QtGui import QColor, QIcon, QDrag, QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel, QPushButton, QListWidget, 
                             QAbstractItemView, QFileDialog, QSplitter, QMessageBox,
                             QGroupBox, QFormLayout, QScrollArea, QMenu)

from coralnet_toolbox.Icons import get_icon

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

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setColumnCount(3)
        
        # Updated Headers to "Z Channel"
        self.setHorizontalHeaderLabels(["Image Source", "Z Channel", "Status"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.setColumnWidth(2, 100)
        
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
            if len(selected_rows) == 1:
                clear_action = menu.addAction("Clear Z-Channel Mapping")
            else:
                clear_action = menu.addAction(f"Clear Z-Channel Mapping ({len(selected_rows)} rows)")
            
            action = menu.exec_(self.mapToGlobal(position))
            if action == clear_action:
                self.clearMapping.emit(selected_rows)


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
        self.resize(950, 600)
        
        self.image_files = sorted(image_files)
        # Smart sort z_files to align with image order
        self.z_files = self._smart_sort_z_files(sorted(z_files), self.image_files)
        self.mapping = {}  # {image_path: {"z_path": path_or_none, "status": status_string}}
        
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
            "Import z-channel files (Depth, Height, DEM) and pair them with your images.<br><br>"
            "<b>Automatic Matching:</b> The system will attempt to match files using exact names, "
            "suffixes, or fuzzy matching.<br><br>"
            "<b>Manual Correction:</b> Drag z-channel files from the right panel onto table rows "
            "to override automatic matches or fill missing ones. Select multiple files with Ctrl/Shift "
            "click to batch-map in order."
        )
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        group_box.setMaximumHeight(100)
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
            self.list_widget.item(self.list_widget.count() - 1).setData(Qt.UserRole, z_file)
        
        files_scroll.setWidget(self.list_widget)
        files_layout.addWidget(files_scroll)
        files_group.setLayout(files_layout)
        splitter.addWidget(files_group)
        
        splitter.setSizes([650, 300])
        self.main_layout.addWidget(splitter)

    def setup_buttons_layout(self):
        """Set up the bottom buttons section."""
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        # Buttons
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        
        btn_confirm = QPushButton("Confirm Mapping")
        btn_confirm.clicked.connect(self.finalize_mapping)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_confirm)
        
        self.main_layout.addLayout(btn_layout)

    def run_auto_match(self):
        """
        Automatically pair images with z-channel files using heuristic matching.
        
        Matching priority:
        1. Exact name match (same filename without extension)
        2. Suffix match (image name + known z-suffix)
        3. Fuzzy match (string similarity)
        """
        for img in self.image_files:
            img_name = os.path.splitext(os.path.basename(img))[0].lower()
            best_match = None
            match_type = "Missing"
            
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

            # Store the mapping
            self.mapping[img] = {
                "z_path": best_match,
                "status": match_type
            }

    def populate_ui(self):
        """Populate the pairing table with matched files."""
        self.table.setRowCount(len(self.image_files))
        
        for r, img_path in enumerate(self.image_files):
            match_data = self.mapping[img_path]
            z_path = match_data["z_path"]
            status = match_data["status"]

            # Column 0: Image Name
            item_img = QTableWidgetItem(os.path.basename(img_path))
            item_img.setToolTip(img_path)
            self.table.setItem(r, 0, item_img)
            
            # Column 1: Z Channel
            display_text = os.path.basename(z_path) if z_path else "None (Drag Here)"
            item_z = QTableWidgetItem(display_text)
            if z_path:
                item_z.setToolTip(z_path)
            self.table.setItem(r, 1, item_z)
            
            # Column 2: Status
            item_status = QTableWidgetItem(status)
            item_status.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 2, item_status)
            
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

    def handle_manual_drop(self, row, z_path):
        """
        Handle manual pairing when user drags a z-file to a table row.
        
        Args:
            row (int): Table row index
            z_path (str): Path to the z-channel file
        """
        img_path = self.image_files[row]
        
        # Update mapping data
        self.mapping[img_path]["z_path"] = z_path
        self.mapping[img_path]["status"] = "Manual Fix"
        
        # Update table display
        item_z = self.table.item(row, 1)
        item_z.setText(os.path.basename(z_path))
        item_z.setToolTip(z_path)
        
        item_status = self.table.item(row, 2)
        item_status.setText("Manual Fix")
        
        # Update row color to indicate successful match
        self.update_row_color(row, "Auto")

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
            self.mapping[img_path]["status"] = "Missing"
            
            # Update table display
            item_z = self.table.item(row, 1)
            item_z.setText("None (Drag Here)")
            item_z.setToolTip("")
            
            item_status = self.table.item(row, 2)
            item_status.setText("Missing")
            
            # Update row color to indicate missing match
            self.update_row_color(row, "Missing")

    def handle_cell_click(self, row, col):
        """
        Handle double-click on z-path cell to browse for file.
        
        Args:
            row (int): Table row index
            col (int): Table column index
        """
        if col == 1:  # Z Channel Column
            fname, _ = QFileDialog.getOpenFileName(self, "Select Z-Channel File")
            if fname:
                self.handle_manual_drop(row, fname)

    def finalize_mapping(self):
        """
        Finalize the mapping and emit confirmation signal.
        
        Warns user if there are unmatched images.
        """
        final_dict = {}
        missing_count = 0
        
        # Build final mapping with only paired images
        for img, data in self.mapping.items():
            if data["z_path"]:
                final_dict[img] = data["z_path"]
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