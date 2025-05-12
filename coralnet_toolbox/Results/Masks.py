import numpy as np
import torch

from ultralytics.engine.results import BaseTensor


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks,
    including methods for converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple): Original image shape in (height, width) format.
        _xy (List[numpy.ndarray]): A list of segments in pixel coordinates.
        _xyn (List[numpy.ndarray]): A list of normalized segments.

    Methods:
        cpu(): Returns a copy of the Masks object with the mask tensor on CPU memory.
        numpy(): Returns a copy of the Masks object with the mask tensor as a numpy array.
        cuda(): Returns a copy of the Masks object with the mask tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the Masks object with the mask tensor on specified device and dtype.
        update_xy(xy, new_shape): Updates mask with new pixel coordinates and shape.
        update_xyn(xyn, new_shape): Updates mask with new normalized coordinates and shape.
    """

    def __init__(self, masks, orig_shape) -> None:
        """
        Initialize the Masks class with detection mask data and the original image shape.

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.
        """
        # Handle tensor dimensions
        if isinstance(masks, torch.Tensor) or isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = masks[None, :]
        
        # Initialize the base tensor
        super().__init__(masks, orig_shape)
        
        # Initialize coordinate attributes
        self._xy = []
        self._xyn = []
    
    @property
    def xy(self):
        """
        Returns the [x, y] pixel coordinates for each segment in the mask tensor.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel
                coordinates for a single segmentation mask.
        """
        return self._xy

    @property
    def xyn(self):
        """
        Returns normalized xy-coordinates of the segmentation masks.

        Returns:
            (List[numpy.ndarray]): A list of numpy arrays containing normalized coordinates.
        """
        return self._xyn
    
    def update_xy(self, xy, new_shape):
        """
        Updates the mask with new pixel coordinates and shape.
        
        Args:
            xy (List[numpy.ndarray]): A list of numpy arrays containing [x, y] pixel coordinates.
            new_shape (tuple): The new image shape as (height, width).
            
        Returns:
            (Masks): Self for method chaining.
        """
        # Update the original shape
        self.orig_shape = new_shape
        
        # Store the new xy coordinates directly
        self._xy = [coords.copy() if isinstance(coords, np.ndarray) else coords for coords in xy]
        
        # Clear xyn so it will be recalculated when needed
        self._xyn = None
            
        return self
    
    def update_xyn(self, xyn, new_shape):
        """
        Updates the mask with new normalized coordinates and shape.
        
        Args:
            xyn (List[numpy.ndarray]): A list of numpy arrays containing normalized [x, y] coordinates.
            new_shape (tuple): The new image shape as (height, width).
            
        Returns:
            (Masks): Self for method chaining.
        """
        # Update the original shape
        self.orig_shape = new_shape
        
        # Store the new xyn coordinates directly
        self._xyn = [coords.copy() if isinstance(coords, np.ndarray) else coords for coords in xyn]
        
        # Clear xy so it will be recalculated when needed
        self._xy = None
        
        # Calculate xy coordinates from xyn
        self._xy = []
        for coords in self._xyn:
            if len(coords) > 0:
                pixel_coords = coords.copy()
                pixel_coords[:, 0] *= new_shape[1]  # Scale x by width
                pixel_coords[:, 1] *= new_shape[0]  # Scale y by height
                self._xy.append(pixel_coords)
            else:
                self._xy.append(np.zeros((0, 2), dtype=np.float32))
        
        return self
        
    def __len__(self):
        """
        Returns the number of masks in this object.
        
        Returns:
            (int): Number of masks
        """
        if self._xy is not None:
            return len(self._xy)
        elif hasattr(self.data, 'shape'):
            return self.data.shape[0]
        return 0