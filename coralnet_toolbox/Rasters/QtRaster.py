import warnings

import os
import gc
from collections import defaultdict
from typing import Optional, Set, List

import cv2
import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject

from coralnet_toolbox.Annotations import MaskAnnotation

from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.utilities import convert_scale_units
from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import rasterio_to_qimage
from coralnet_toolbox.utilities import work_area_to_numpy
from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import load_z_channel_from_file

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
        self.is_highlighted = False  # Track if the row is highlighted
        self.is_filtered = True
        self.is_visible = True
        self.display_name = self.basename  # Can be truncated for display
        
        # Annotation state
        self.has_annotations = False
        self.has_predictions = False
        self.labels: Set = set()
        self.annotation_count = 0
        self.annotations: List = []  # Store the actual annotations
        self.label_counts = {}  # Store counts of annotations per label
        
        self.label_set: Set[str] = set()  # Add sets for efficient lookups
        self.label_to_types_map = {}  # This replaces annotation_types and annotation_type_set
        
        # Add a new attribute to hold the MaskAnnotation, initialized to None.
        self.mask_annotation: Optional[MaskAnnotation] = None
        
        # Work Area state
        self.work_areas: List = []  # Store work area information
        
        # Image dimensions and properties (populated when rasterio_src is loaded)
        self.width = 0
        self.height = 0
        self.channels = 0
        self.shape = (0, 0, 0)  # Shape of the image (height, width, channels)
        
        # Scale information (if available)
        self.scale_x: Optional[float] = None
        self.scale_y: Optional[float] = None
        self.scale_units: Optional[str] = None
        
        # Depth/elevation channel information
        self.z_channel: Optional[np.ndarray] = None  # Depth/elevation channel data (float32 or uint8)
        self.z_channel_path: Optional[str] = None  # Path to z_channel file if saved separately
        self.z_unit: Optional[str] = None  # Units for z_channel data (e.g., 'meters', 'feet')
        self.z_nodata: Optional[float] = None  # Nodata value for z_channel (NULL/missing data indicator)
        
        # Camera calibration information
        self.intrinsics: Optional[np.ndarray] = None  # Camera intrinsic parameters as numpy array
        self.extrinsics: Optional[np.ndarray] = None  # Camera extrinsic parameters as numpy array
        
        # Metadata
        self.metadata = {}  # Can store any additional metadata
        
        # Load rasterio source
        self.load_rasterio()
        
    def set_selected(self, is_selected: bool):
        """Mark this raster as selected in the UI"""
        self.is_selected = is_selected
        
    def set_highlighted(self, is_highlighted: bool):
        """Mark this raster as highlighted in the UI"""
        self.is_highlighted = is_highlighted
    
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
            self.shape = (self.height, self.width, self.channels)
            
            # Update metadata
            self.metadata['dimensions'] = f"{self.width}x{self.height}"
            self.metadata['bands'] = self.channels

            crs = self._rasterio_src.crs
            if crs is not None:
                if not getattr(crs, "is_geographic", False):
                    # Case 1: Already projected. Read the scale and convert to meters.
                    transform = self._rasterio_src.transform

                    scale_x = abs(transform.a)
                    scale_y = abs(transform.e)
                    source_units = crs.linear_units.lower()

                    # Convert scale to meters per pixel
                    scale_x_meters = convert_scale_units(scale_x, source_units, 'm')
                    scale_y_meters = convert_scale_units(scale_y, source_units, 'm')

                    self.update_scale(scale_x_meters, scale_y_meters, 'm')

                elif getattr(crs, "is_geographic", False):
                    # Case 2: Geographic. Project to Web Mercator (EPSG:3857) to get meter-based scale.
                    try:
                        # Define the destination CRS (Web Mercator)
                        dst_crs = CRS.from_epsg(3857)

                        # Calculate the transform, new width, and new height for the reprojected image
                        transform, width, height = calculate_default_transform(
                            crs,                        # Source CRS
                            dst_crs,                    # Destination CRS
                            self._rasterio_src.width,   # Source width
                            self._rasterio_src.height,  # Source height
                            *self._rasterio_src.bounds  # Source bounds
                        )

                        # Now, the transform's components are in meters
                        scale_x = abs(transform.a)
                        scale_y = abs(transform.e)
                        scale_units = 'm'  # EPSG:3857 units are meters

                        self.update_scale(scale_x, scale_y, scale_units)

                        # Update metadata to show the *derived* scale
                        self.metadata['scale_x'] = f"~{self.scale_x:.4f} {self.scale_units} (from EPSG:3857)"
                        self.metadata['scale_y'] = f"~{self.scale_y:.4f} {self.scale_units} (from EPSG:3857)"
                        self.metadata['original_crs'] = str(crs)

                    except Exception as e:
                        # Fallback if reprojection calculation fails
                        print(f"Could not calculate default transform for {self.image_path}: {e}")
                        self.metadata['units'] = "degrees (reprojection failed)"
                        # scale attributes remain None
                # else: neither projected nor geographic, do nothing
            # Case 3: No CRS. self.scale_x, self.scale_y, and self.scale_units correctly remain None

            return True
            
        except Exception as e:
            print(f"Error loading rasterio image {self.image_path}: {str(e)}")
            return False
            
    def update_scale(self, scale_x: float, scale_y: float, units: str):
        """
        Update the scale information for this raster.
        
        Args:
            scale_x (float): The horizontal scale (e.g., meters per pixel)
            scale_y (float): The vertical scale (e.g., meters per pixel)
            units (str): The name of the units (e.g., 'm', 'cm')
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_units = units
        
        # Update metadata to match
        self.metadata['scale_x'] = f"{self.scale_x:.6f} {self.scale_units}"
        self.metadata['scale_y'] = f"{self.scale_y:.6f} {self.scale_units}"

    def remove_scale(self):
        """
        Remove all scale information from this raster.
        """
        self.scale_x = None
        self.scale_y = None
        self.scale_units = None
        
        # Remove from metadata if keys exist
        self.metadata.pop('scale_x', None)
        self.metadata.pop('scale_y', None)
        self.metadata.pop('original_crs', None)  # Also remove derived crs
    
    def add_intrinsics(self, intrinsics: np.ndarray):
        """
        Add or update camera intrinsic parameters.
        
        Args:
            intrinsics (np.ndarray): Camera intrinsic matrix (typically 3x3 or 3x4)
                                    Standard format for camera calibration matrix
        """
        if not isinstance(intrinsics, np.ndarray):
            raise ValueError("Intrinsics must be a numpy array")
        self.intrinsics = intrinsics.copy()
        
    def update_intrinsics(self, intrinsics: np.ndarray):
        """
        Update camera intrinsic parameters.
        
        Args:
            intrinsics (np.ndarray): New camera intrinsic matrix
        """
        self.add_intrinsics(intrinsics)  # Same validation as add
        
    def remove_intrinsics(self):
        """Remove all camera intrinsic parameters."""
        self.intrinsics = None
        
    def add_extrinsics(self, extrinsics: np.ndarray):
        """
        Add or update camera extrinsic parameters.
        
        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix (typically 4x4 transformation matrix
                                    or 3x4 [R|T] matrix combining rotation and translation)
        """
        if not isinstance(extrinsics, np.ndarray):
            raise ValueError("Extrinsics must be a numpy array")
        self.extrinsics = extrinsics.copy()
        
    def update_extrinsics(self, extrinsics: np.ndarray):
        """
        Update camera extrinsic parameters.
        
        Args:
            extrinsics (np.ndarray): New camera extrinsic matrix
        """
        self.add_extrinsics(extrinsics)  # Same validation as add
        
    def remove_extrinsics(self):
        """Remove all camera extrinsic parameters."""
        self.extrinsics = None
        
    def add_z_channel(self, z_data: np.ndarray, z_path: Optional[str] = None):
        """
        Add or update depth/elevation channel data.
        
        Note: z_unit should be set separately via load_z_channel_from_file() or manually.
        This method does not modify z_unit.
        
        Args:
            z_data (np.ndarray): 2D numpy array containing depth or elevation data (float32 or uint8)
            z_path (str, optional): Path to the z_channel file if saved separately
        """
        if not isinstance(z_data, np.ndarray):
            raise ValueError("Z channel data must be a numpy array")
        if z_data.ndim != 2:
            raise ValueError("Z channel data must be a 2D array")
        if z_data.dtype not in [np.float32, np.uint8]:
            raise ValueError("Z channel data must be float32 or uint8 dtype")
        if z_data.shape != (self.height, self.width):
            raise ValueError(f"Z channel dimensions {z_data.shape} must match image dimensions "
                             f"({self.height}, {self.width})")
        self.z_channel = z_data.copy()
        self.z_channel_path = z_path
        # Note: z_unit is NOT set here; it should be set before calling add_z_channel()
        
    def update_z_channel(self, z_data: np.ndarray, z_path: Optional[str] = None):
        """
        Update the depth/elevation channel data.
        
        Args:
            z_data (np.ndarray): 2D numpy array containing depth or elevation data (float32 or uint8)
            z_path (str, optional): Path to the z_channel file if saved separately
        """
        self.add_z_channel(z_data, z_path)  # Same validation as add
        
    def set_z_channel_path(self, z_path: str, auto_load: bool = True):
        """
        Set the path to the z_channel file and optionally auto-load it.
        
        Args:
            z_path (str): Path to the z_channel file
            auto_load (bool): Whether to automatically attempt loading the z-channel data
        """
        self.z_channel_path = z_path
        
        # Automatically attempt to load z-channel data if requested and file exists
        if auto_load and z_path and os.path.exists(z_path):
            try:
                self.load_z_channel_from_file(z_path)
            except Exception as e:
                print(f"Warning: Could not auto-load z-channel from {z_path}: {str(e)}")
        
    def remove_z_channel(self):
        """Remove the depth/elevation channel data and path."""
        self.z_channel = None
        self.z_channel_path = None
        self.z_unit = None
        self.z_nodata = None
        
    def load_z_channel_from_file(self, z_channel_path: str, z_unit: str = None):
        """
        Load z_channel data from a file path using rasterio.
        
        The z_channel data will be either:
        - float32: Actual depth/height values (e.g., meters, feet, etc.)
        - uint8: Relative depth/height values (0-255 range)
        
        Args:
            z_channel_path (str): Path to the depth/height/DEM file
            z_unit (str, optional): Unit of measurement for z-channel data
                                   If not provided, will attempt to detect from file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        from coralnet_toolbox.utilities import (
            detect_z_channel_units_from_file,
            normalize_z_unit
        )
        
        z_data, z_path, z_nodata = load_z_channel_from_file(
            z_channel_path, 
            target_width=self.width, 
            target_height=self.height
        )
        
        if z_data is not None:
            # Determine units for the z-channel
            if z_unit is None:
                # Try to detect from file
                detected_unit, _ = detect_z_channel_units_from_file(z_channel_path)
                z_unit = normalize_z_unit(detected_unit) if detected_unit else None
            else:
                # Normalize provided unit
                z_unit = normalize_z_unit(z_unit)
            
            # Store unit and nodata value before adding z_channel
            self.z_unit = z_unit
            self.z_nodata = z_nodata
            self.add_z_channel(z_data, z_path)
            return True
        else:
            print(f"Failed to load z-channel from: {z_channel_path}")
            return False
    
    @property
    def rasterio_src(self):
        """Get the rasterio dataset"""
        return self._rasterio_src
    
    @property
    def z_channel_lazy(self):
        """
        Get the z_channel with lazy loading.
        If z_channel is None but z_channel_path exists, attempt to load it.
        
        Returns:
            numpy.ndarray or None: The z-channel data, or None if not available
        """
        # If z_channel is already loaded, return it
        if self.z_channel is not None:
            return self.z_channel
            
        # If we have a path but no loaded data, try to load it
        if self.z_channel_path and os.path.exists(self.z_channel_path):
            try:
                print(f"Lazy loading z-channel from: {self.z_channel_path}")
                if self.load_z_channel_from_file(self.z_channel_path):
                    return self.z_channel
                else:
                    print(f"Failed to lazy load z-channel from: {self.z_channel_path}")
            except Exception as e:
                print(f"Error during lazy loading of z-channel: {str(e)}")
        
        # Return None if no z-channel is available
        return None
        
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
                    # Try using rasterio_to_qimage as a fallback
                    if self._rasterio_src is not None:
                        self._q_image = rasterio_to_qimage(self._rasterio_src)
                        if self._q_image is None or self._q_image.isNull():
                            return None
                    else:
                        return None
            except Exception as e:
                print(f"Error loading QImage {self.image_path}: {str(e)}")
                # Try using rasterio_to_qimage as a fallback
                try:
                    if self._rasterio_src is not None:
                        self._q_image = rasterio_to_qimage(self._rasterio_src)
                        if self._q_image is None or self._q_image.isNull():
                            return None
                    else:
                        return None
                except Exception as e2:
                    print(f"Error loading QImage with rasterio for {self.image_path}: {str(e2)}")
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
    
    def get_numpy(self) -> Optional[np.ndarray]:
        """
        Get the image data as a numpy array.
        
        Returns:
            np.ndarray or None: The image data as a numpy array, or None if loading fails
        """
        if self._rasterio_src is not None:
            try:
                # Read the image data into a numpy array
                return pixmap_to_numpy(self.get_pixmap())
            except Exception as e:
                print(f"Error reading numpy data from {self.image_path}: {str(e)}")
                return None
        return None
        
    def update_annotation_info(self, annotations: list):
        """
        Update annotation-related information for this raster.
        This now builds a more powerful cache mapping labels to their annotation types.
        
        Args:
            annotations (list): List of annotation objects
        """
        self.annotations = annotations
        self.annotation_count = len(annotations)
        self.has_annotations = bool(annotations)
        
        predictions = [a.machine_confidence for a in annotations if a.machine_confidence]
        self.has_predictions = len(predictions) > 0
        
        # Clear previous data
        self.label_counts.clear()
        self.label_set.clear()
        self.label_to_types_map.clear()  # Clear the new map

        # Use a defaultdict to simplify the aggregation logic
        temp_map = defaultdict(set)

        for annotation in annotations:
            # Process label information
            if annotation.label:
                if hasattr(annotation.label, 'short_label_code'):
                    label_name = annotation.label.short_label_code
                else:
                    label_name = str(annotation.label)
                
                # Update label counts and the set of all labels
                self.label_counts[label_name] = self.label_counts.get(label_name, 0) + 1
                self.label_set.add(label_name)

                # Process annotation type information and link it to the label
                anno_type = annotation.__class__.__name__
                temp_map[label_name].add(anno_type)

        # Convert defaultdict back to a regular dict for the final attribute
        self.label_to_types_map = dict(temp_map)
        
    @property
    def annotation_types(self) -> dict:
        """
        Computes a simple count of each annotation type on-the-fly.
        This property provides backward compatibility for features like the tooltip
        without needing to store this data permanently.
        
        Returns:
            dict: A dictionary mapping annotation type names to their counts.
                e.g., {'PolygonAnnotation': 5, 'PointAnnotation': 2}
        """
        type_counts = defaultdict(int)
        # The self.label_to_types_map structure is {'label': {'type1', 'type2'}}
        # This is not ideal for counting total types. We need the original annotations list.
        if not self.annotations:
            return {}
            
        for annotation in self.annotations:
            anno_type = annotation.__class__.__name__
            type_counts[anno_type] += 1
            
        return dict(type_counts)
    
    def matches_filter(self, 
                       search_text="", 
                       search_label="", 
                       top_k=1,
                       require_annotations=False,
                       require_no_annotations=False,
                       require_predictions=False) -> bool:
        """
        Check if this raster matches the given filter criteria
        
        Args:
            search_text (str): Text to search for in filename
            search_label (str): Label code to search for
            top_k (int): Number of top predictions to consider for label search
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
            label_match = False
            
            # Check actual annotation labels (always consider these)
            # Look for the search label in the label_set instead of self.labels
            for label_code in self.label_set:
                if search_label in label_code:
                    label_match = True
                    break
            
            # If no match in annotation labels, check machine learning predictions
            if not label_match:
                for annotation in self.annotations:
                    if hasattr(annotation, 'machine_confidence') and annotation.machine_confidence:
                        # Get the sorted machine confidence dictionary (already sorted by confidence)
                        # Take only the top-k predictions
                        top_predictions = list(annotation.machine_confidence.items())[:top_k]
                        
                        # Check each label in the top-k predictions
                        for pred_label, confidence in top_predictions:
                            if hasattr(pred_label, 'short_label_code') and search_label in pred_label.short_label_code:
                                label_match = True
                                break
                        if label_match:
                            break
            
            if not label_match:
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
    
    def get_mask_annotation(self, project_labels: list) -> MaskAnnotation:
        """
        Gets the mask annotation for this raster, creating it if it doesn't exist.
        This is the "lazy loading" mechanism.

        Args:
            project_labels (list): The current list of all Label objects from the LabelWindow.

        Returns:
            MaskAnnotation: The mask annotation object for this raster.
        """
        if self.mask_annotation is None:
            # Create the mask on its first request
            mask_data = np.zeros((self.height, self.width), dtype=np.uint8)
            self.mask_annotation = MaskAnnotation(
                image_path=self.image_path,
                mask_data=mask_data,
                initial_labels=project_labels,
                rasterio_src=self.rasterio_src
            )
        else:
            # Ensure the mask is synced with the current project labels
            self.mask_annotation.sync_label_map(project_labels)
        return self.mask_annotation
    
    def get_mask_class_statistics(self, project_labels: list) -> dict:
        """
        Gets class statistics from the mask annotation itself, which handles caching.

        Args:
            project_labels (list): The current project labels, needed by 
                                   get_mask_annotation if the mask needs
                                   to be created.

        Returns:
            dict: The class statistics for the mask.
        """
        # Ensure mask_annotation exists
        mask = self.get_mask_annotation(project_labels) 
        
        if mask:
            # The mask now manages its own cache
            return mask.get_class_statistics() 
        else:
            return {}  # Default to empty
        
    @property
    def mask_statistics(self) -> dict | None:
        """
        Returns the cached mask statistics if they exist, without
        triggering a recalculation.
        """
        if self.mask_annotation:
            return self.mask_annotation.cached_statistics
        return None
            
    def add_work_area(self, work_area):
        """
        Add a work area to the raster.
        
        Args:
            work_area: Work area object to add
        """
        if work_area not in self.work_areas:
            # Add the new work area
            self.work_areas.append(work_area)
            
    def get_work_areas(self):
        """
        Get all work areas for this raster.
        
        Returns:
            list: List of work area objects
        """
        return self.work_areas
        
    def has_work_areas(self):
        """
        Check if this raster has any work areas.
        
        Returns:
            bool: True if the raster has work areas, False otherwise
        """
        return len(self.work_areas) > 0
    
    def count_work_items(self):
        """
        Count the number of work items for this raster.
        If work areas are defined, each work area counts as a work item.
        Otherwise, the entire image counts as a single work item.
        
        Returns:
            int: Number of work items
        """
        if self.has_work_areas():
            return len(self.work_areas)
        else:
            return 1  # The entire image counts as one work item
    
    def get_work_area_data(self, work_area, as_format='RGB'):
        """
        Extract image data from a work area as a numpy array.
        
        Args:
            work_area: WorkArea object or QRectF defining the area to extract
            as_type (str): Format to return the data in, 'cv2' converts to BGR color format
                
        Returns:
            numpy.ndarray: Image data from the work area, in BGR format if as_format='cv2',
                           otherwise in RGB format
        """
        # Convert work area to numpy array
        work_area_data = work_area_to_numpy(self._rasterio_src, work_area) 
        
        if as_format == 'BGR':
            # Convert to RGB to BGR format for OpenCV
            work_area_data = cv2.cvtColor(work_area_data, cv2.COLOR_RGB2BGR)
            
        return work_area_data
    
    def get_work_areas_data(self, as_format='RGB'):
        """
        Get image data from all work areas as a list of numpy arrays.
        
         Args:
            as_type (str): Format to return the data in, 'cv2' converts to BGR color format
        
        Returns:
            list: List of numpy arrays representing image data from each work area
        """
        work_area_data = []
        
        for work_area in self.work_areas:
            data = self.get_work_area_data(work_area, as_format=as_format)
            if data is not None:
                work_area_data.append(data)
                
        if len(work_area_data) == 0:
            work_area_data = [self.image_path]  # Fallback to the full image path
            
        return work_area_data
        
    def remove_work_area(self, work_area):
        """
        Remove a work area from the raster.
        
        Args:
            work_area: Work area object to remove
        """
        if work_area in self.work_areas:
            self.work_areas.remove(work_area)
                
    def clear_work_areas(self):
        """Clear all work areas."""
        self.work_areas.clear()
        
    def delete_mask_annotation(self):
        """Removes the mask annotation and its graphics item, then resets the attribute."""
        if self.mask_annotation:
            self.mask_annotation.remove_from_scene()
            self.mask_annotation = None
    
    def to_dict(self):
        """
        Convert the raster to a dictionary for project saving.
        
        Returns:
            dict: Dictionary representation compatible with project format
        """
        # Get work areas as dictionaries
        work_areas_list = [wa.to_dict() for wa in self.get_work_areas()]
        
        raster_data = {
            'path': self.image_path,
            'state': {
                'checkbox_state': self.checkbox_state
            },
            'work_areas': work_areas_list
        }
        
        # Include scale information if available
        if self.scale_x is not None and self.scale_y is not None and self.scale_units is not None:
            raster_data['scale'] = {
                'scale_x': self.scale_x,
                'scale_y': self.scale_y,
                'scale_units': self.scale_units
            }
        
        # Include camera calibration information if available
        if self.intrinsics is not None:
            raster_data['intrinsics'] = self.intrinsics.tolist()  # Convert numpy array to list for JSON
            
        if self.extrinsics is not None:
            raster_data['extrinsics'] = self.extrinsics.tolist()  # Convert numpy array to list for JSON
        
        # Include z_channel path if available
        if self.z_channel_path is not None:
            raster_data['z_channel_path'] = self.z_channel_path
            
        # Include z_unit if available
        if self.z_unit is not None:
            raster_data['z_unit'] = self.z_unit
        
        if self.z_channel is not None:
            raster_data['has_z_channel'] = True
            # Optionally store basic info about the z_channel
            raster_data['z_channel_info'] = {
                'shape': self.z_channel.shape,
                'dtype': str(self.z_channel.dtype),
                'min': float(np.min(self.z_channel)),
                'max': float(np.max(self.z_channel)),
                'mean': float(np.mean(self.z_channel))
            }
            
        return raster_data
    
    @classmethod
    def from_dict(cls, raster_dict):
        """
        Create a Raster instance from a dictionary (from project loading).
        
        Args:
            raster_dict (dict): Dictionary containing raster data
            
        Returns:
            Raster: A new Raster instance with loaded properties
        """
        # Create the raster with the image path
        image_path = raster_dict['path']
        raster = cls(image_path)
        
        # Load state information
        state = raster_dict.get('state', {})
        raster.checkbox_state = state.get('checkbox_state', False)
        
        # Load work areas
        work_areas_list = raster_dict.get('work_areas', [])
        for work_area_data in work_areas_list:
            try:
                from coralnet_toolbox.QtWorkArea import WorkArea
                work_area = WorkArea.from_dict(work_area_data, image_path)
                raster.add_work_area(work_area)
            except Exception as e:
                print(f"Error loading work area for {image_path}: {str(e)}")
        
        # Load scale information if available
        scale_data = raster_dict.get('scale')
        if scale_data:
            try:
                raster.update_scale(
                    scale_data['scale_x'], 
                    scale_data['scale_y'], 
                    scale_data['scale_units']
                )
            except Exception as e:
                print(f"Error loading scale information for {image_path}: {str(e)}")
        
        # Load camera calibration information if available
        intrinsics_data = raster_dict.get('intrinsics')
        if intrinsics_data:
            try:
                # Convert list back to numpy array
                intrinsics_array = np.array(intrinsics_data)
                raster.add_intrinsics(intrinsics_array)
            except Exception as e:
                print(f"Error loading intrinsics for {image_path}: {str(e)}")
                
        extrinsics_data = raster_dict.get('extrinsics')
        if extrinsics_data:
            try:
                # Convert list back to numpy array
                extrinsics_array = np.array(extrinsics_data)
                raster.add_extrinsics(extrinsics_array)
            except Exception as e:
                print(f"Error loading extrinsics for {image_path}: {str(e)}")
        
        # Load z_channel path if available and attempt to load the z-channel data
        z_channel_path = raster_dict.get('z_channel_path')
        if z_channel_path:
            raster.set_z_channel_path(z_channel_path, auto_load=False)  # Defer loading
            
        # Load z_unit if available
        z_unit = raster_dict.get('z_unit')
        if z_unit:
            raster.z_unit = z_unit
        
        return raster
    
    def update_from_dict(self, raster_dict):
        """
        Update this raster instance with data from a dictionary (from project loading).
        This is useful when the raster already exists and you want to update its properties.
        
        Args:
            raster_dict (dict): Dictionary containing raster data
        """
        # Update state information
        state = raster_dict.get('state', {})
        self.checkbox_state = state.get('checkbox_state', False)
        
        # Update work areas
        work_areas_list = raster_dict.get('work_areas', [])
        for work_area_data in work_areas_list:
            try:
                work_area = WorkArea.from_dict(work_area_data, self.image_path)
                self.add_work_area(work_area)
            except Exception as e:
                print(f"Error loading work area for {self.image_path}: {str(e)}")
        
        # Update scale information if available (overriding any auto-detected scale)
        scale_data = raster_dict.get('scale')
        if scale_data:
            try:
                self.update_scale(
                    scale_data['scale_x'], 
                    scale_data['scale_y'], 
                    scale_data['scale_units']
                )
            except Exception as e:
                print(f"Error loading scale information for {self.image_path}: {str(e)}")
        
        # Update camera calibration information if available
        intrinsics_data = raster_dict.get('intrinsics')
        if intrinsics_data:
            try:
                # Convert list back to numpy array
                intrinsics_array = np.array(intrinsics_data)
                self.add_intrinsics(intrinsics_array)
            except Exception as e:
                print(f"Error loading intrinsics for {self.image_path}: {str(e)}")
                
        extrinsics_data = raster_dict.get('extrinsics')
        if extrinsics_data:
            try:
                # Convert list back to numpy array
                extrinsics_array = np.array(extrinsics_data)
                self.add_extrinsics(extrinsics_array)
            except Exception as e:
                print(f"Error loading extrinsics for {self.image_path}: {str(e)}")
        
        # Update z_channel path if available but don't load the data yet
        z_channel_path = raster_dict.get('z_channel_path')
        if z_channel_path:
            self.set_z_channel_path(z_channel_path, auto_load=False)  # Defer loading
            
        # Update z_unit if available
        z_unit = raster_dict.get('z_unit')
        if z_unit:
            self.z_unit = z_unit
        
        # Note: z_channel data is not loaded from dictionary as it's typically stored separately
        
    def cleanup(self):
        """Release all resources associated with this raster."""
        # Close and clean up rasterio resources
        if self._rasterio_src is not None:
            try:
                self._rasterio_src.close()
            except Exception:
                pass
            self._rasterio_src = None
            
        # Clean up QImage resources
        self._q_image = None
        self._thumbnail = None
        
        # Clear annotations and work areas
        self.annotations = []
        self.work_areas = []
        # Clear mask annotation if it exists
        self.delete_mask_annotation()
        
        # Clear camera calibration and z channel data
        self.intrinsics = None
        self.extrinsics = None
        self.z_channel = None
        self.z_channel_path = None
        self.z_unit = None
        
        # Force garbage collection
        gc.collect()
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()