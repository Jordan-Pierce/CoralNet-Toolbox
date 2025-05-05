import warnings

import os
import gc
from typing import Optional, Dict, Any, Set, Union, List

import rasterio
import numpy as np

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject

from coralnet_toolbox.utilities import rasterio_open, rasterio_to_qimage

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Raster(QObject):
    """
    Base class for handling raster image data and associated metadata.
    Encapsulates both the rasterio representation and QImage representation
    along with UI state and annotation information.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize a Raster object from an image path.
        
        Args:
            image_path (str): Path to the image file
        """
        super().__init__()
        
        # Basic file information
        self.image_path = image_path
        self.basename = os.path.basename(image_path)
        self.extension = os.path.splitext(image_path)[1].lower()
        
        # Initialize properties
        self._rasterio_src = None
        self._q_image = None
        self._thumbnail = None  # Single thumbnail cache
        
        # UI state and table information
        self.checkbox_state = False
        self.row_in_table = -1
        self.is_selected = False
        self.is_filtered = True
        self.is_visible = True
        self.display_name = self.basename  # Can be truncated for display
        
        # Annotation state
        self.has_annotations = False
        self.has_predictions = False
        self.labels: Set = set()
        self.annotation_count = 0
        self.annotations: List = []  # Store the actual annotations
        
        # Work Area state
        self.work_areas: List = []  # Store work area information
        
        # Image dimensions and properties (populated when rasterio_src is loaded)
        self.width = 0
        self.height = 0
        self.channels = 0
        
        # Metadata
        self.metadata = {}  # Can store any additional metadata
        
        # Load rasterio source
        self.load_rasterio()
        
    def load_rasterio(self) -> bool:
        """
        Load the image using rasterio and extract basic properties.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            self._rasterio_src = rasterio_open(self.image_path)
            
            if self._rasterio_src is None:
                return False
                
            # Extract image properties
            self.width = self._rasterio_src.width
            self.height = self._rasterio_src.height
            self.channels = self._rasterio_src.count
            
            # Update metadata
            self.metadata['dimensions'] = f"{self.width}x{self.height}"
            self.metadata['bands'] = self.channels
            if hasattr(self._rasterio_src, 'crs') and self._rasterio_src.crs:
                self.metadata['crs'] = str(self._rasterio_src.crs)
            
            return True
            
        except Exception as e:
            print(f"Error loading rasterio image {self.image_path}: {str(e)}")
            return False
    
    @property
    def rasterio_src(self):
        """Get the rasterio dataset"""
        return self._rasterio_src
        
    def get_qimage(self) -> Optional[QImage]:
        """
        Get or create the full-resolution QImage representation.
        
        Returns:
            QImage or None: The image as QImage, or None if loading fails
        """
        if self._q_image is None:
            try:
                self._q_image = QImage(self.image_path)
                if self._q_image.isNull():
                    return None
            except Exception as e:
                print(f"Error loading QImage {self.image_path}: {str(e)}")
                return None
                
        return self._q_image
    
    def get_thumbnail(self, longest_edge: int = 256) -> Optional[QImage]:
        """
        Get or create a thumbnail of the specified size.
        
        Args:
            longest_edge (int): The length of the longest edge of the thumbnail
            
        Returns:
            QImage or None: The thumbnail as QImage, or None if creation fails
        """
        # Create the thumbnail if it doesn't exist or has a different size
        if self._thumbnail is None:
            if self._rasterio_src is not None:
                try:
                    self._thumbnail = rasterio_to_qimage(self._rasterio_src, longest_edge=longest_edge)
                    if self._thumbnail.isNull():
                        return None
                except Exception as e:
                    print(f"Error creating thumbnail for {self.image_path}: {str(e)}")
                    return None
            else:
                return None
                
        return self._thumbnail
    
    def get_pixmap(self, longest_edge: Optional[int] = None) -> Optional[QPixmap]:
        """
        Get a QPixmap representation, either at full resolution or as a thumbnail.
        
        Args:
            longest_edge (int, optional): If provided, creates a thumbnail of this size
            
        Returns:
            QPixmap or None: The image as QPixmap, or None if creation fails
        """
        if longest_edge is None:
            # Return full resolution pixmap
            qimage = self.get_qimage()
        else:
            # Return thumbnail pixmap
            qimage = self.get_thumbnail(longest_edge)
            
        if qimage and not qimage.isNull():
            return QPixmap.fromImage(qimage)
        return None
        
    def update_annotation_info(self, annotations: list):
        """
        Update annotation-related information for this raster.
        
        Args:
            annotations (list): List of annotation objects
        """
        self.annotations = annotations
        self.annotation_count = len(annotations)
        self.has_annotations = bool(annotations)
        
        # Check for predictions
        predictions = [a.machine_confidence for a in annotations if a.machine_confidence != {}]
        self.has_predictions = len(predictions) > 0
        
        # Update labels
        self.labels = {annotation.label for annotation in annotations if annotation.label}
        
    def matches_filter(self, 
                       search_text="", 
                       search_label="", 
                       require_annotations=False,
                       require_no_annotations=False,
                       require_predictions=False) -> bool:
        """
        Check if this raster matches the given filter criteria
        
        Args:
            search_text (str): Text to search for in filename
            search_label (str): Label code to search for
            require_annotations (bool): If True, must have annotations
            require_no_annotations (bool): If True, must have no annotations
            require_predictions (bool): If True, must have predictions
            
        Returns:
            bool: True if this raster matches all filter criteria
        """
        # Check filename search
        if search_text and search_text not in self.basename:
            return False
            
        # Check label search
        if search_label:
            label_codes = [label.short_label_code for label in self.labels 
                           if hasattr(label, 'short_label_code')]
            if not any(search_label in code for code in label_codes):
                return False
        
        # Check annotation filters
        if require_annotations and not self.has_annotations:
            return False
        if require_no_annotations and self.has_annotations:
            return False
            
        # Check prediction filter
        if require_predictions and not self.has_predictions:
            return False
            
        return True
    
    def add_work_area(self, work_area):
        """
        Add a work area to the raster.
        
        Args:
            work_area: Work area object to add
        """
        if work_area not in self.work_areas:
            self.work_areas.append(work_area)
            
    def remove_work_area(self, work_area):
        """
        Remove a work area from the raster.
        
        Args:
            work_area: Work area object to remove
        """
        if work_area in self.work_areas:
            self.work_areas.remove(work_area)
            
    def get_work_areas(self):
        """
        Get all work areas for this raster.
        
        Returns:
            list: List of work area objects
        """
        return self.work_areas
        
    def clear_work_areas(self):
        """Clear all work areas for this raster."""
        self.work_areas.clear()
        
    def has_work_areas(self):
        """
        Check if this raster has any work areas.
        
        Returns:
            bool: True if the raster has work areas, False otherwise
        """
        return len(self.work_areas) > 0
    
    def set_selected(self, is_selected: bool):
        """Mark this raster as selected in the UI"""
        self.is_selected = is_selected
    
    def set_display_name(self, max_length=25):
        """
        Set a display name (truncated if necessary) for showing in table
        
        Args:
            max_length (int): Maximum length for display name before truncation
        """
        if len(self.basename) > max_length:
            self.display_name = self.basename[:max_length - 3] + "..."
        else:
            self.display_name = self.basename
    
    def get_work_area_data(self, work_area, as_numpy=True):
        """
        Extract image data from a work area as a QImage or numpy array.
        
        Args:
            work_area: WorkArea object or QRectF
            as_numpy (bool): If True, returns numpy array, otherwise returns QImage
            
        Returns:
            numpy.ndarray or QImage: Image data from the work area
        """
        if not self._rasterio_src:
            return None
            
        # If we got a WorkArea object, use its rect
        if hasattr(work_area, 'rect'):
            rect = work_area.rect
        else:
            rect = work_area
            
        # Create a rasterio window from the rect
        window = rasterio.windows.Window(
            col_off=int(rect.x()),
            row_off=int(rect.y()),
            width=int(rect.width()),
            height=int(rect.height())
        )
        
        # Check if window is within image bounds
        if (window.col_off < 0 or window.row_off < 0 or
            window.col_off + window.width > self._rasterio_src.width or
            window.row_off + window.height > self._rasterio_src.height):
            # Clip window to image bounds
            window = window.intersection(
                rasterio.windows.Window(0, 0, self._rasterio_src.width, self._rasterio_src.height)
            )
            
        # Get the cropped image as QImage
        from coralnet_toolbox.utilities import rasterio_to_cropped_image
        q_image = rasterio_to_cropped_image(self._rasterio_src, window)
        
        if not as_numpy:
            return q_image
            
        # Convert QImage to numpy array
        width = q_image.width()
        height = q_image.height()
        
        if q_image.format() == QImage.Format_RGB888:
            # RGB image
            ptr = q_image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        elif q_image.format() == QImage.Format_Grayscale8:
            # Grayscale image
            ptr = q_image.bits()
            ptr.setsize(height * width)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
        else:
            # Unsupported format, convert to RGB888 first
            q_image = q_image.convertToFormat(QImage.Format_RGB888)
            ptr = q_image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
        return arr

    def map_coords_from_work_area(self, work_area, coords, is_normalized=True):
        """
        Map coordinates from a work area's coordinate system to the full image coordinate system.
        
        Args:
            work_area: WorkArea object or QRectF
            coords: Numpy array of coordinates in the form [[x1, y1], [x2, y2], ...] or
                    for bounding boxes in form [x1, y1, x2, y2] (can be multiple boxes as first dimension)
            is_normalized: Whether the coordinates are normalized (0-1) within the work area
            
        Returns:
            numpy.ndarray: Coordinates mapped to the full image coordinate system
        """
        # Get the work area rectangle
        if hasattr(work_area, 'rect'):
            rect = work_area.rect
        else:
            rect = work_area
            
        # Make a copy to avoid modifying the original
        coords = np.array(coords, dtype=float).copy()
        
        # Check if coords is a bounding box format [x1, y1, x2, y2]
        is_bbox = (coords.ndim == 1 and coords.size == 4) or (coords.ndim == 2 and coords.shape[1] == 4)
        
        # Convert to a common format for processing
        if is_bbox:
            if coords.ndim == 1:
                # Single bbox
                bbox = coords.reshape(1, 4)
            else:
                # Multiple bboxes
                bbox = coords
                
            # Convert to polygon format for uniform processing
            # Each bbox becomes [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            polygons = []
            for box in bbox:
                x1, y1, x2, y2 = box
                polygons.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
            coords = np.array(polygons)
        
        # Normalize if not already normalized
        if not is_normalized:
            if coords.ndim == 2:
                # Single polygon
                coords[:, 0] /= rect.width()
                coords[:, 1] /= rect.height()
            else:
                # Multiple polygons
                for polygon in coords:
                    polygon[:, 0] /= rect.width()
                    polygon[:, 1] /= rect.height()
        
        # Map normalized coordinates to the work area space
        if coords.ndim == 2:
            # Single polygon
            coords[:, 0] = coords[:, 0] * rect.width() + rect.x()
            coords[:, 1] = coords[:, 1] * rect.height() + rect.y()
        else:
            # Multiple polygons
            for polygon in coords:
                polygon[:, 0] = polygon[:, 0] * rect.width() + rect.x()
                polygon[:, 1] = polygon[:, 1] * rect.height() + rect.y()
        
        # Convert back to bbox format if original was bbox
        if is_bbox:
            result = []
            for polygon in coords:
                # Extract bbox coordinates from the polygon
                x_min = polygon[:, 0].min()
                y_min = polygon[:, 1].min()
                x_max = polygon[:, 0].max() 
                y_max = polygon[:, 1].max()
                result.append([x_min, y_min, x_max, y_max])
            
            # Return in same format as input
            if len(result) == 1 and coords.ndim == 1:
                return np.array(result[0])
            else:
                return np.array(result)
        
        # Return polygons
        if coords.shape[0] == 1 and coords.ndim > 2:
            # If there was only one polygon but in 3D array, return as 2D
            return coords[0]
        else:
            return coords
        
    def cleanup(self):
        """Release all resources associated with this raster."""
        # Close and clean up rasterio resources
        if self._rasterio_src is not None:
            try:
                self._rasterio_src.close()
            except:
                pass
            self._rasterio_src = None
            
        # Clean up QImage resources
        self._q_image = None
        self._thumbnail = None
        
        # Clear annotations and work areas
        self.annotations = []
        self.work_areas = []
        
        # Force garbage collection
        gc.collect()
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()
