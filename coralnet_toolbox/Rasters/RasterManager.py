import warnings

import os
import gc
from typing import Dict, List, Optional, Set

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, pyqtSignal, Qt

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
    scaleUpdated = pyqtSignal(str)  # Image path - emitted when scale data changes
    
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

        # If the path looks like a video, delegate to add_video_raster so we
        # create a VideoRaster instead of a plain Raster. This avoids cases
        # where videos are treated as ImageRasters (and passed as file paths
        # to image loaders expecting images).
        try:
            ext = os.path.splitext(image_path)[1].lower()
            from coralnet_toolbox.Rasters.VideoRaster import VIDEO_EXTENSIONS
            if ext in VIDEO_EXTENSIONS:
                return self.add_video_raster(image_path)
        except Exception:
            # If the import fails for any reason, fall back to default behavior
            pass

        # No automatic guessing between Raster vs OrthoRaster here.
        # `OrthoRaster` should only be created when the user explicitly
        # imports files as orthomosaics. Drag/drop/import as images should
        # create plain `Raster` objects unless the file is a video.

        try:
            raster = Raster(image_path)
            if raster.rasterio_src is None:
                return False

            self.rasters[image_path] = raster
            self.image_paths.append(image_path)

            # Connect raster's z-channel signal to forward as zChannelUpdated
            raster.zChannelChanged.connect(lambda: self.zChannelUpdated.emit(image_path))
            # Forward scale changes so listeners can re-sync cached scale values
            raster.scaleChanged.connect(lambda: self.scaleUpdated.emit(image_path))

            if emit_signal:
                self.rasterAdded.emit(image_path)

            return True

        except Exception as e:
            print(f"Error adding raster {image_path}: {str(e)}")
            return False
    
    def has_image_path(self, image_path: str) -> bool:
        """Return True if this exact path is managed.

        O(1). `image_paths` is a list kept in step with the `rasters` dict, so
        `path in manager.image_paths` is an O(n) scan of the same information --
        which turns an import into O(n^2) once it is done once per file. Unlike
        `get_raster`, this does NOT resolve virtual `video.mp4::frame_N` paths;
        it answers only whether the path itself is managed, matching the list
        membership tests it replaces.
        """
        return image_path in self.rasters

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

            raster.scaleChanged.connect(lambda: self.scaleUpdated.emit(video_path))

            self.rasterAdded.emit(video_path)
            return True

        except Exception as e:
            print(f"Error adding video raster {video_path}: {str(e)}")
            return False

    def add_ortho_raster(self, ortho_path: str, emit_signal: bool = True) -> bool:
        """
        Add an OrthoRaster to the manager.

        Args:
            ortho_path (str): Path to the orthomosaic file

        Returns:
            bool: True if successful, False otherwise
        """
        if ortho_path in self.rasters:
            return True

        try:
            # Import here to avoid circular imports
            from coralnet_toolbox.Rasters.OrthoRaster import OrthoRaster
            raster = OrthoRaster(ortho_path)

            self.rasters[ortho_path] = raster
            self.image_paths.append(ortho_path)

            raster.scaleChanged.connect(lambda: self.scaleUpdated.emit(ortho_path))

            if emit_signal:
                self.rasterAdded.emit(ortho_path)
            return True

        except Exception as e:
            print(f"Error adding ortho raster {ortho_path}: {str(e)}")
            return False
    
    def remove_raster(self, image_path: str, collect_garbage: bool = True) -> bool:
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
            self.rasters[image_path].cleanup(collect_garbage=collect_garbage)
            
            # Remove from collections
            del self.rasters[image_path]
            self.image_paths.remove(image_path)
            
            # Emit signal
            self.rasterRemoved.emit(image_path)
            
            # Force garbage collection when requested.
            if collect_garbage:
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
                           require_mask: bool = False,
                           allowed_raster_types: Optional[Set[str]] = None,
                           require_z_channel: bool = False,
                           require_checked: bool = False,
                           selected_paths: List[str] = None) -> List[str]:
        """
        Get a filtered list of image paths based on criteria.

        Args:
            search_text (str): Text to search for in filename
            search_label (str): Label code to search for
            require_annotations (bool): If True, must have annotations
            require_no_annotations (bool): If True, must have no annotations
            require_predictions (bool): If True, must have predictions
            require_mask (bool): If True, must have a mask annotation with labeled pixels
            require_checked (bool): If True, must have its checkbox ticked
            selected_paths (list): Only include paths from this list
            
        Returns:
            list: Filtered list of image paths
        """
        filtered_paths = []

        # Hashed once rather than rescanned per path: `selected_paths` arrives
        # as a list, so the membership test below is O(n) each time and the loop
        # around it is O(n^2) on a large project.
        selected_lookup = None if selected_paths is None else set(selected_paths)

        for path in self.image_paths:
            # Skip if not in selected paths
            if selected_lookup is not None and path not in selected_lookup:
                continue
                
            raster = self.rasters[path]
            
            # Check if raster matches filter criteria
            if raster.matches_filter(
                search_text=search_text,
                search_label=search_label,
                require_annotations=require_annotations,
                require_no_annotations=require_no_annotations,
                require_predictions=require_predictions,
                require_mask=require_mask,
                allowed_raster_types=allowed_raster_types,
                require_z_channel=require_z_channel,
                require_checked=require_checked,
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
        # Handle virtual video frame paths like 'video.mp4::frame_42'
        if isinstance(image_path, str) and '::frame_' in image_path:
            video_path, frame_part = image_path.rsplit('::frame_', 1)
            try:
                frame_idx = int(frame_part)
            except Exception:
                return None

            raster = self.rasters.get(video_path)
            if raster is None:
                return None

            # If raster supports a cached preview pixmap (VideoRaster), use it
            # so repeated hovers over nearby/revisited frames skip re-decoding.
            if hasattr(raster, 'get_preview_pixmap'):
                try:
                    return raster.get_preview_pixmap(frame_idx, longest_edge=longest_edge)
                except Exception:
                    return None

            # If raster supports direct frame access, use it
            if hasattr(raster, 'get_frame'):
                try:
                    qimg = raster.get_frame(frame_idx)
                    if qimg is None:
                        return None
                    pix = QPixmap.fromImage(qimg)
                    if longest_edge is not None:
                        pix = pix.scaled(longest_edge, longest_edge,
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    return pix
                except Exception:
                    return None

            # Fallback to frame-0 pixmap
            return raster.get_pixmap(longest_edge=longest_edge)

        # Regular (non-virtual) image/video path
        raster = self.rasters.get(image_path)
        if raster is None:
            return None
        return raster.get_pixmap(longest_edge=longest_edge)
    
    def clear(self):
        """Clear all rasters from the manager."""
        # Create copy of paths to avoid modification during iteration
        paths = list(self.image_paths)
        
        for path in paths:
            self.remove_raster(path, collect_garbage=False)

        if paths:
            gc.collect()
            
    def __len__(self):
        """Get the number of rasters in the manager."""
        return len(self.image_paths)