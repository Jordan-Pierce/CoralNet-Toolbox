import warnings

import numpy as np
import cv2

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
        from depth_anything_3.api import DepthAnything3
        import torch
        
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
                import torch
                from torch.cuda import empty_cache
                empty_cache()
                
    def predict(self, image_array):
        """
        Run depth prediction on an image using DA3.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array (H, W, C)
            
        Returns:
            numpy.ndarray: Depth map as numpy array (H, W) with depth values in meters
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Store original dimensions
        original_height, original_width = image_array.shape[:2]
        
        # Run inference
        # DA3 expects a list of images
        prediction = self.model.inference(
            image=[image_array],
            process_res=self.imgsz,
            process_res_method="upper_bound_resize",
        )
        
        # Extract depth map (first image in batch)
        depth_map = prediction.depth[0]
        
        # Convert to numpy array if not already
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map)
        
        # Post-process: resize depth map to match original input dimensions if needed
        if depth_map.shape[0] != original_height or depth_map.shape[1] != original_width:
            depth_map = cv2.resize(
                depth_map,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR
            )
            
        return depth_map
