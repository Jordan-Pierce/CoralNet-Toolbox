import warnings

import os
import gc
from typing import Optional, Dict, Any, Set, Union, List

import rasterio

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
        
        # Clear annotations
        self.annotations = []
        
        # Force garbage collection
        gc.collect()
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()