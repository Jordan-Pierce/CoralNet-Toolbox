import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(ABC):
    """
    Base class for depth estimation models.
    
    All depth models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, device):
        """
        Initialize the depth model.
        
        Args:
            device (str): Device to run the model on ('cuda', 'mps', 'cpu')
        """
        self.device = device
        self.model = None
        self.imgsz = 504  # Default image size
        
    @abstractmethod
    def load_model(self, model_path, imgsz):
        """
        Load the depth estimation model.
        
        Args:
            model_path (str): Path to the model weights
            imgsz (int): Image size for inference
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement load_model()")
        
    @abstractmethod
    def deactivate(self):
        """
        Deactivate and clean up the model from memory.
        
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement deactivate()")
        
    @abstractmethod
    def predict(self, images, intrinsics=None, extrinsics=None):
        """
        Run depth prediction on a list of images, returning depth maps and camera parameters.
        
        Args:
            images (list[str | numpy.ndarray]): List of input images as file paths or numpy arrays (H, W, C)
            intrinsics (list[numpy.ndarray], optional): List of camera intrinsic matrices (one per image).
                                                       Use None for images without existing intrinsics.
            extrinsics (list[numpy.ndarray], optional): List of camera extrinsic matrices (one per image).
                                                       Use None for images without existing extrinsics.
            
        Returns:
            dict: Dictionary containing:
                - 'depth_maps' (list[numpy.ndarray]): List of depth maps as numpy arrays (H, W) with depth values
                - 'intrinsics' (list[numpy.ndarray]): List of camera intrinsic matrices
                - 'extrinsics' (list[numpy.ndarray]): List of camera extrinsic matrices
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement predict()")
