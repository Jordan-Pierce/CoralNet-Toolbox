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
        self.imgsz = 512  # Default image size
        
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
    def predict(self, image_array):
        """
        Run depth prediction on an image.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array (H, W, C)
            
        Returns:
            numpy.ndarray: Depth map as numpy array (H, W) with depth values in meters
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement predict()")
