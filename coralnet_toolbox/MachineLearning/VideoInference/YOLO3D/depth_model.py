import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DepthEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_size='small', device=None):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device
        
        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")
        
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # Create pipeline
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                # Create a CPU pipeline for this frame
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]
                
                # Convert PIL Image to numpy array if needed
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            else:
                # Re-raise the error if not MPS
                raise
        
        # Normalize depth map to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 