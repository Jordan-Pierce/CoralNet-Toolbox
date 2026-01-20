import warnings

import torch
from torch.cuda import empty_cache


from coralnet_toolbox.Z.Models.QtBase import Base

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DA3(Base):
    """
    Depth Anything 3 model for depth estimation.
    
    Uses the Depth Anything 3 API to generate depth maps from images.
    """
    
    def __init__(self, device):
        """
        Initialize the DA3 model.
        
        Args:
            device (str): Device to run the model on ('cuda', 'mps', 'cpu')
        """
        super().__init__(device)
        
    def load_model(self, model_path, imgsz):
        """
        Load the Depth Anything 3 model.
        
        Args:
            model_path (str): Model identifier (e.g., "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
            imgsz (int): Image size for inference
        """
        # Import DA3 model here in case the package is not installed
        from depth_anything_3.api import DepthAnything3
        
        self.imgsz = imgsz
        
        # Load model from pretrained
        self.model = DepthAnything3.from_pretrained(model_path)
        
        # Move to device
        if self.device.startswith('cuda'):
            device = torch.device('cuda')
        elif self.device == 'mps':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
        self.model = self.model.to(device)
        self.model.eval()
        
    def deactivate(self):
        """
        Deactivate and clean up the DA3 model from memory.
        """
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clean up GPU memory if using CUDA
            if self.device.startswith('cuda'):
                empty_cache()
                
    def predict(self, images, intrinsics=None, extrinsics=None):
        """
        Run depth prediction on a list of images using DA3, returning depth maps (meters) and camera parameters.
        
        Args:
            images (list[str | numpy.ndarray]): List of input images as file paths or numpy arrays (H, W, C)
            intrinsics (list[numpy.ndarray], optional): List of camera intrinsic matrices (one per image).
                                                       If provided, DA3 will use these instead of estimating.
            extrinsics (list[numpy.ndarray], optional): List of camera extrinsic matrices (one per image).
                                                       If provided, DA3 will use these instead of estimating.
            
        Returns:
            dict: Dictionary containing:
                - 'depth_maps' (list[numpy.ndarray]): List of depth maps as numpy arrays (H, W) with depth values
                - 'intrinsics' (list[numpy.ndarray]): List of camera intrinsic matrices
                - 'extrinsics' (list[numpy.ndarray]): List of camera extrinsic matrices
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Run inference with optional camera parameters
        prediction = self.model.inference(
            image=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            process_res=self.imgsz,
            process_res_method="upper_bound_resize",
        )
        
        # Return the results
        return {
            'depth_maps': prediction.depth,
            'intrinsics': prediction.intrinsics,
            'extrinsics': prediction.extrinsics
        }
