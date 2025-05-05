import warnings

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QFont

from coralnet_toolbox.Rasters import RasterManager

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RasterTableModel(QAbstractTableModel):
    """
    Custom table model for displaying a list of Raster objects.
    """
    # Column indices
    CHECKBOX_COL = 0
    FILENAME_COL = 1
    ANNOTATION_COUNT_COL = 2
    
    def __init__(self, raster_manager: RasterManager, parent=None):
        """
        Initialize the model.
        
        Args:
            raster_manager (RasterManager): Reference to the raster manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.raster_manager = raster_manager
        self.filtered_paths: List[str] = []
        self.column_headers = ["✓", "Image Name", "Annotations"]
        
        # Column widths
        self.column_widths = [50, -1, 120]  # -1 means stretch
        
        # Connect to manager signals
        self.raster_manager.rasterAdded.connect(self.on_raster_added)
        self.raster_manager.rasterRemoved.connect(self.on_raster_removed)
        self.raster_manager.rasterUpdated.connect(self.on_raster_updated)
        
    def rowCount(self, parent=None) -> int:
        """Return the number of rows in the model."""
        return len(self.filtered_paths)
        
    def columnCount(self, parent=None) -> int:
        """Return the number of columns in the model."""
        return len(self.column_headers)
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        """Return header data for the model."""
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.column_headers[section]
        return None
        
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid() or index.row() >= len(self.filtered_paths):
            return None
            
        # Get the raster for this row
        path = self.filtered_paths[index.row()]
        raster = self.raster_manager.get_raster(path)
        
        if not raster:
            return None
            
        # Make sure display name is set
        raster.set_display_name(max_length=25)
        
        if role == Qt.DisplayRole:
            if index.column() == self.FILENAME_COL:
                return raster.display_name
            elif index.column() == self.ANNOTATION_COUNT_COL:
                return str(raster.annotation_count)
                
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
            
        elif role == Qt.FontRole:
            # Bold the selected raster's text
            if raster.is_selected:
                font = QFont()
                font.setBold(True)
                return font
                
        elif role == Qt.ToolTipRole:
            if index.column() == self.FILENAME_COL:
                # Include full path and metadata in tooltip
                dimensions = raster.metadata.get('dimensions', f"{raster.width}x{raster.height}")
                return (f"Path: {path}\n"
                        f"Dimensions: {dimensions}\n"
                        f"Has Annotations: {'Yes' if raster.has_annotations else 'No'}\n"
                        f"Has Predictions: {'Yes' if raster.has_predictions else 'No'}")
                        
        elif role == Qt.CheckStateRole:
            if index.column() == self.CHECKBOX_COL:
                return Qt.Checked if raster.checkbox_state else Qt.Unchecked
                
        return None
        
    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        """Set data for the given index and role."""
        if not index.isValid() or index.row() >= len(self.filtered_paths):
            return False
            
        # Get the raster for this row
        path = self.filtered_paths[index.row()]
        raster = self.raster_manager.get_raster(path)
        
        if not raster:
            return False
            
        if role == Qt.CheckStateRole and index.column() == self.CHECKBOX_COL:
            # Update checkbox state
            raster.checkbox_state = value == Qt.Checked
            return True
            
        return False
        
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Return flags for the given index."""
        if not index.isValid():
            return Qt.NoItemFlags
            
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        
        if index.column() == self.CHECKBOX_COL:
            flags |= Qt.ItemIsUserCheckable
            
        return flags
        
    def set_filtered_paths(self, paths: List[str]):
        """
        Set the filtered paths to display in the model.
        
        Args:
            paths (List[str]): List of image paths to display
        """
        self.beginResetModel()
        self.filtered_paths = paths
        self.endResetModel()
        
    def get_path_at_row(self, row: int) -> Optional[str]:
        """
        Get the image path at the given row.
        
        Args:
            row (int): Row index
            
        Returns:
            str or None: Image path or None if invalid
        """
        if 0 <= row < len(self.filtered_paths):
            return self.filtered_paths[row]
        return None
        
    def get_row_for_path(self, path: str) -> int:
        """
        Get the row index for the given image path.
        
        Args:
            path (str): Image path
            
        Returns:
            int: Row index, or -1 if not found
        """
        try:
            return self.filtered_paths.index(path)
        except ValueError:
            return -1
            
    def set_selected_path(self, path: str):
        """
        Set the selected path and update the model.
        
        Args:
            path (str): Image path to select
        """
        # Clear any previous selection
        for p in self.filtered_paths:
            raster = self.raster_manager.get_raster(p)
            if raster and raster.is_selected:
                raster.set_selected(False)
                row = self.get_row_for_path(p)
                if row >= 0:
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, self.columnCount() - 1)
                    )
        
        # Set new selection
        raster = self.raster_manager.get_raster(path)
        if raster:
            raster.set_selected(True)
            row = self.get_row_for_path(path)
            if row >= 0:
                self.dataChanged.emit(
                    self.index(row, 0),
                    self.index(row, self.columnCount() - 1)
                )
                
    def update_raster_data(self, path: str):
        """
        Update display for a specific raster.
        
        Args:
            path (str): Image path to update
        """
        row = self.get_row_for_path(path)
        if row >= 0:
            self.dataChanged.emit(
                self.index(row, 0),
                self.index(row, self.columnCount() - 1)
            )
            
    def on_raster_added(self, path: str):
        """Handler for when a raster is added to the manager."""
        # No need to do anything here, as filtered_paths is set separately
        pass
        
    def on_raster_removed(self, path: str):
        """Handler for when a raster is removed from the manager."""
        row = self.get_row_for_path(path)
        if row >= 0:
            self.beginRemoveRows(QModelIndex(), row, row)
            self.filtered_paths.remove(path)
            self.endRemoveRows()
            
    def on_raster_updated(self, path: str):
        """Handler for when a raster is updated in the manager."""
        self.update_raster_data(path)