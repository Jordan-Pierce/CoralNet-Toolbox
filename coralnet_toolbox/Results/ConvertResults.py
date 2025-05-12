import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils import ops
from ultralytics.utils.ops import scale_masks


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConvertResults:
    """
    A class to convert to Ultralytics Results.
    """

    def __init__(self):
        pass
    
    def from_sam(self, masks, scores, image, image_path):
        """
        Converts SAM results to Ultralytics Results Object.

        Args:
            masks (torch.Tensor): Predicted masks with shape (N, 1, H, W).
            scores (torch.Tensor): Confidence scores for each mask with shape (N, 1).
            image (np.ndarray): The original, unprocessed image.
            image_path (str): Path to the image file.

        Returns:
            results (Results): Ultralytics Results object.
        """
        # Ensure the original image is in the correct format
        if not isinstance(image, np.ndarray):
            image = image.cpu().numpy()

        # Ensure masks have the correct shape (N, 1, H, W)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"Expected masks to have shape (N, 1, H, W), but got {masks.shape}")

        # Scale masks to the original image size and remove extra dimensions
        scaled_masks = ops.scale_masks(masks.float(), image.shape[:2], padding=False)
        scaled_masks = scaled_masks > 0.5  # Apply threshold to masks

        # Ensure scaled_masks is 3D (N, H, W)
        if scaled_masks.ndim == 4:
            scaled_masks = scaled_masks.squeeze(1)

        # Generate bounding boxes from masks using batched_mask_to_box
        scaled_boxes = batched_mask_to_box(scaled_masks)

        # Ensure scores has shape (N,) by removing extra dimensions
        scores = scores.squeeze().cpu()
        if scores.ndim == 0:  # If only one score, make it a 1D tensor
            scores = scores.unsqueeze(0)

        # Generate class labels
        cls = torch.arange(len(masks), dtype=torch.int32).cpu()

        # Ensure all tensors are 2D before concatenating
        scaled_boxes = scaled_boxes.cpu()
        if scaled_boxes.ndim == 1:
            scaled_boxes = scaled_boxes.unsqueeze(0)
        scores = scores.view(-1, 1)  # Reshape to (N, 1)
        cls = cls.view(-1, 1)  # Reshape to (N, 1)

        # Combine bounding boxes, scores, and class labels
        scaled_boxes = torch.cat([scaled_boxes, scores, cls], dim=1)

        # Create names dictionary (placeholder for consistency)
        names = dict(enumerate(str(i) for i in range(len(masks))))

        # Create Results object
        results = Results(image, 
                          path=image_path, 
                          names=names, 
                          masks=scaled_masks, 
                          boxes=scaled_boxes)
        
        return results
    
    def from_supervision(self, detections, images, image_paths=None, names=None):
        """
        Convert Supervision Detections to Ultralytics Results format.
        Handles both single detection/image and lists of detections/images.

        Args:
            detections: Single Detections object or list of Detections objects
            images: Single image array or list of image arrays
            image_paths: Single image path or list of image paths (optional)
            names: Dictionary mapping class ids to class names (optional)

        Returns:
            generator: Yields Ultralytics Results objects
        """
        # Convert single inputs to lists
        if not isinstance(detections, list):
            detections = [detections]
        if not isinstance(images, list):
            images = [images]
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Ensure image_paths exists
        if not image_paths:
            image_paths = [None] * len(images)

        for detection, image, path in zip(detections, images, image_paths):
            # Ensure image is numpy array
            if torch.is_tensor(image):
                image = image.cpu().numpy()

            # Create default names if not provided
            if names is None:
                names = {i: str(i) for i in range(len(detection))} if len(detection) > 0 else {}

            if len(detection) == 0:
                return [Results(orig_img=image, path=path, names=names)]

            # Handle masks if present
            if hasattr(detection, 'mask') and detection.mask is not None:
                masks = torch.as_tensor(detection.mask, dtype=torch.float32)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                scaled_masks = scale_masks(masks, image.shape[:2], padding=False)
                scaled_masks = scaled_masks > 0.5
                if scaled_masks.ndim == 4:
                    scaled_masks = scaled_masks.squeeze(1)
            else:
                scaled_masks = None

            # Convert boxes and scores
            scaled_boxes = torch.as_tensor(detection.xyxy, dtype=torch.float32)
            scores = torch.as_tensor(detection.confidence, dtype=torch.float32).view(-1, 1)
            cls = torch.as_tensor(detection.class_id, dtype=torch.int32).view(-1, 1)

            # Combine boxes, scores, and class IDs
            if scaled_boxes.ndim == 1:
                scaled_boxes = scaled_boxes.unsqueeze(0)
            scaled_boxes = torch.cat([scaled_boxes, scores, cls], dim=1)

            # Create and return Results object
            return [Results(image,
                            path=path,
                            names=names,
                            boxes=scaled_boxes, 
                            masks=scaled_masks)]
    

