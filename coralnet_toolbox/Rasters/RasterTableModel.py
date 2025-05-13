import warnings

from typing import Any, Dict, List, Optional, Set

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QFont, QColor, QBrush

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
    FILENAME_COL = 0
    ANNOTATION_COUNT_COL = 1
    
    # Row colors
    HIGHLIGHTED_COLOR = QColor(173, 216, 230)  # Light blue
    SELECTED_COLOR = QColor(144, 238, 144)     # Light green
    
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
        
        # We'll remove this separate tracking mechanism to avoid inconsistency
        # self.highlighted_paths: Set[str] = set()
        
        self.column_headers = ["Image Name", "Annotations"]
        
        # Column widths
        self.column_widths = [-1, 120]  # -1 means stretch
        
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
                
        elif role == Qt.BackgroundRole:
            # Set background color based on selection/highlight state
            if raster.is_selected:
                return QBrush(self.SELECTED_COLOR)
            elif raster.is_highlighted:
                return QBrush(self.HIGHLIGHTED_COLOR)
                
        elif role == Qt.ToolTipRole:
            if index.column() == self.FILENAME_COL:
                # Include full path and metadata in tooltip
                dimensions = raster.metadata.get('dimensions', f"{raster.width}x{raster.height}")
                return (f"Path: {path}\n"
                        f"Dimensions: {dimensions}\n"
                        f"Has Annotations: {'Yes' if raster.has_annotations else 'No'}\n"
                        f"Has Predictions: {'Yes' if raster.has_predictions else 'No'}")
                
        return None
        
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Return flags for the given index."""
        if not index.isValid():
            return Qt.NoItemFlags
            
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
            
    def highlight_path(self, path: str, highlighted: bool = True):
        """
        Set the highlight state for a specific path
        
        Args:
            path (str): Image path to highlight/unhighlight
            highlighted (bool): Whether to highlight (True) or unhighlight (False)
        """
        raster = self.raster_manager.get_raster(path)
        if raster:
            # Only update if state is changing
            if raster.is_highlighted != highlighted:
                raster.set_highlighted(highlighted)
                
                # Update the view
                row = self.get_row_for_path(path)
                if row >= 0:
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, self.columnCount() - 1)
                    )
                    
    def clear_highlights(self):
        """Clear all highlighted paths"""
        # Find all highlighted rasters
        highlighted_paths = []
        for path in self.filtered_paths:
            raster = self.raster_manager.get_raster(path)
            if raster and raster.is_highlighted:
                highlighted_paths.append(path)
        
        # Unhighlight all paths
        for path in highlighted_paths:
            self.highlight_path(path, False)
            
    def set_highlighted_paths(self, paths: List[str]):
        """
        Set the highlighted state for a list of paths, clearing all others.
        
        Args:
            paths (List[str]): List of image paths to highlight
        """
        # First get all currently highlighted paths
        current_highlighted = self.get_highlighted_paths()
        
        # Unhighlight those that shouldn't be highlighted
        for path in current_highlighted:
            if path not in paths:
                self.highlight_path(path, False)
                
        # Highlight those that should be highlighted
        for path in paths:
            if path in self.filtered_paths:  # Only highlight visible paths
                self.highlight_path(path, True)

    def clear_highlights_except(self, paths_to_keep: List[str]):
        """
        Clear highlights except for the given paths.
        
        Args:
            paths_to_keep (List[str]): Paths to keep highlighted
        """
        for path in self.get_highlighted_paths():
            if path not in paths_to_keep:
                self.highlight_path(path, False)

    def get_highlighted_paths(self) -> List[str]:
        """
        Get a list of all highlighted paths
        
        Returns:
            List[str]: List of highlighted image paths
        """
        highlighted_paths = []
        for path in self.filtered_paths:
            raster = self.raster_manager.get_raster(path)
            if raster and raster.is_highlighted:
                highlighted_paths.append(path)
        return highlighted_paths
        
    def set_filtered_paths(self, paths: List[str]):
        """
        Set the filtered paths to display in the model.
        
        Args:
            paths (List[str]): List of image paths to display
        """
        self.beginResetModel()
        self.filtered_paths = [p for p in paths if p in self.raster_manager.image_paths]
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
        # Get currently selected paths first
        selected_paths = []
        for p in self.filtered_paths:
            raster = self.raster_manager.get_raster(p)
            if raster and raster.is_selected:
                selected_paths.append(p)
        
        # Clear any previous selection
        for p in selected_paths:
            raster = self.raster_manager.get_raster(p)
            if raster:
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
            # Only update if state is changing
            if not raster.is_selected:
                raster.set_selected(True)
                row = self.get_row_for_path(path)
                if row >= 0:
                    self.dataChanged.emit(
                        self.index(row, 0),
                        self.index(row, self.columnCount() - 1)
                    )
    
    def get_selected_path(self) -> Optional[str]:
        """
        Get the currently selected path.
        
        Returns:
            str or None: Selected path or None if no selection
        """
        for path in self.filtered_paths:
            raster = self.raster_manager.get_raster(path)
            if raster and raster.is_selected:
                return path
        return None
                
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
        # If we want to automatically add new rasters to filtered list:
        # if path not in self.filtered_paths:
        #     self.beginInsertRows(QModelIndex(), len(self.filtered_paths), len(self.filtered_paths))
        #     self.filtered_paths.append(path)
        #     self.endInsertRows()
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
    
    # New methods to better sync with Qt's selection model
    def sync_with_selection_model(self, selected_indexes, deselected_indexes):
        """
        Synchronize with Qt's selection model changes.
        
        Args:
            selected_indexes: Indexes that were selected
            deselected_indexes: Indexes that were deselected
        """
        # Handle deselections
        for index in deselected_indexes:
            if index.isValid():
                path = self.get_path_at_row(index.row())
                if path:
                    raster = self.raster_manager.get_raster(path)
                    if raster and raster.is_selected:
                        raster.set_selected(False)
                        self.update_raster_data(path)
        
        # Handle selections
        for index in selected_indexes:
            if index.isValid():
                path = self.get_path_at_row(index.row())
                if path:
                    raster = self.raster_manager.get_raster(path)
                    if raster and not raster.is_selected:
                        raster.set_selected(True)
                        self.update_raster_data(path)