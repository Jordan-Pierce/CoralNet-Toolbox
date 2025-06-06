import warnings

import os
import gc
from typing import Dict, List, Optional, Set

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.Rasters import Raster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RasterManager(QObject):
    """
    Manages the collection of Raster objects in the application.
    Provides methods for adding, retrieving, and removing rasters.
    """
    # Signals
    rasterAdded = pyqtSignal(str)  # Image path
    rasterRemoved = pyqtSignal(str)  # Image path
    rasterUpdated = pyqtSignal(str)  # Image path
    
    def __init__(self):
        """Initialize the RasterManager."""
        super().__init__()
        self.rasters: Dict[str, Raster] = {}
        self.image_paths: List[str] = []
        
    def add_raster(self, image_path: str) -> bool:
        """
        Add a new raster to the manager.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            bool: True if successful, False otherwise
        """
        if image_path in self.rasters:
            return True  # Already exists
            
        try:
            raster = Raster(image_path)
            if raster.rasterio_src is None:
                return False
                
            self.rasters[image_path] = raster
            self.image_paths.append(image_path)
            self.rasterAdded.emit(image_path)
            return True
            
        except Exception as e:
            print(f"Error adding raster {image_path}: {str(e)}")
            return False
    
    def get_raster(self, image_path: str) -> Optional[Raster]:
        """
        Get a raster by its image path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Raster or None: The raster object if found, None otherwise
        """
        return self.rasters.get(image_path)
    
    def remove_raster(self, image_path: str) -> bool:
        """
        Remove a raster from the manager.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if image_path not in self.rasters:
            return False
            
        try:
            # Clean up resources
            self.rasters[image_path].cleanup()
            
            # Remove from collections
            del self.rasters[image_path]
            self.image_paths.remove(image_path)
            
            # Emit signal
            self.rasterRemoved.emit(image_path)
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error removing raster {image_path}: {str(e)}")
            return False
    
    def update_annotation_info(self, image_path: str, annotations: list) -> bool:
        """
        Update annotation information for a raster.
        
        Args:
            image_path (str): Path to the image file
            annotations (list): List of annotation objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        if image_path not in self.rasters:
            return False
            
        self.rasters[image_path].update_annotation_info(annotations)
        self.rasterUpdated.emit(image_path)
        return True
    
    def get_filtered_paths(self, 
                           search_text: str = "",
                           search_label: str = "",
                           top_k: int = 1,
                           require_annotations: bool = False,
                           require_no_annotations: bool = False,
                           require_predictions: bool = False,
                           selected_paths: List[str] = None) -> List[str]:
        """
        Get a filtered list of image paths based on criteria.
        
        Args:
            search_text (str): Text to search for in filename
            search_label (str): Label code to search for
            top_k (int): Number of top predictions to consider for label search
            require_annotations (bool): If True, must have annotations
            require_no_annotations (bool): If True, must have no annotations
            require_predictions (bool): If True, must have predictions
            selected_paths (list): Only include paths from this list
            
        Returns:
            list: Filtered list of image paths
        """
        filtered_paths = []
        
        for path in self.image_paths:
            # Skip if not in selected paths
            if selected_paths is not None and path not in selected_paths:
                continue
                
            raster = self.rasters[path]
            
            # Check if raster matches filter criteria
            if raster.matches_filter(
                search_text=search_text,
                search_label=search_label,
                top_k=top_k,
                require_annotations=require_annotations,
                require_no_annotations=require_no_annotations,
                require_predictions=require_predictions
            ):
                filtered_paths.append(path)
                
        return filtered_paths
        
    def get_thumbnail(self, image_path: str, longest_edge: int = 64) -> Optional[QPixmap]:
        """
        Get a thumbnail for a raster.
        
        Args:
            image_path (str): Path to the image file
            longest_edge (int): Length of longest edge for thumbnail
            
        Returns:
            QPixmap or None: Thumbnail as a QPixmap, or None if error
        """
        if image_path not in self.rasters:
            return None
            
        return self.rasters[image_path].get_pixmap(longest_edge=longest_edge)
    
    def clear(self):
        """Clear all rasters from the manager."""
        # Create copy of paths to avoid modification during iteration
        paths = list(self.image_paths)
        
        for path in paths:
            self.remove_raster(path)
            
    def __len__(self):
        """Get the number of rasters in the manager."""
        return len(self.image_paths)