import warnings

import os
import gc
from typing import Dict, List, Optional, Set

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.Rasters.QtRaster import Raster


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
    zChannelUpdated = pyqtSignal(str)  # Image path - emitted when z-channel data changes
    
    def __init__(self):
        """Initialize the RasterManager."""
        super().__init__()
        self.rasters: Dict[str, Raster] = {}
        self.image_paths: List[str] = []
        
    def add_raster(self, image_path: str, emit_signal: bool = True) -> bool:
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
            
            # Connect raster's z-channel signal to forward as zChannelUpdated
            raster.zChannelChanged.connect(lambda: self.zChannelUpdated.emit(image_path))
            
            if emit_signal:
                self.rasterAdded.emit(image_path)
                
            return True
            
        except Exception as e:
            print(f"Error adding raster {image_path}: {str(e)}")
            return False
    
    def get_raster(self, image_path: str) -> Optional[Raster]:
        """
        Get a raster by its image path.
        Resolves virtual frame paths of the form 'video.mp4::frame_42' transparently.
        
        Args:
            image_path (str): Path to the image file, may be a virtual frame path
            
        Returns:
            Raster or None: The raster object if found, None otherwise
        """
        # Resolve virtual video frame paths to the underlying video path
        if '::frame_' in image_path:
            video_path = image_path.rsplit('::frame_', 1)[0]
            return self.rasters.get(video_path)
        return self.rasters.get(image_path)

    def add_video_raster(self, video_path: str) -> bool:
        """
        Add a VideoRaster to the manager.

        Args:
            video_path (str): Path to the video file

        Returns:
            bool: True if successful, False otherwise
        """
        if video_path in self.rasters:
            return True  # Already exists

        try:
            # Import here to avoid circular imports at module level
            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
            raster = VideoRaster(video_path)

            self.rasters[video_path] = raster
            self.image_paths.append(video_path)

            self.rasterAdded.emit(video_path)
            return True

        except Exception as e:
            print(f"Error adding video raster {video_path}: {str(e)}")
            return False
    
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
        # Accept virtual frame paths like 'video.mp4::frame_42' and normalize
        # them to the underlying video path so VideoRaster rows update properly.
        if isinstance(image_path, str) and '::frame_' in image_path:
            image_path = image_path.rsplit('::frame_', 1)[0]

        if image_path not in self.rasters:
            return False

        self.rasters[image_path].update_annotation_info(annotations)
        self.rasterUpdated.emit(image_path)
        return True
    
    def get_filtered_paths(self, 
                           search_text: str = "",
                           search_label: str = "",
                           require_annotations: bool = False,
                           require_no_annotations: bool = False,
                           require_predictions: bool = False,
                           selected_paths: List[str] = None) -> List[str]:
        """
        Get a filtered list of image paths based on criteria.
        
        Args:
            search_text (str): Text to search for in filename
            search_label (str): Label code to search for
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