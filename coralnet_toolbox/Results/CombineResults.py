import numpy as np
import torch


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CombineResults:
    """
    A class to combine multiple Results objects for the same image into a single Results object.
    """
    def __init__(self):
        pass
    
    def combine_results(self, results: list):
        """
        Combine multiple Results objects for the same image into a single Results object.
        
        This function combines detections from multiple Results objects that correspond to the same image,
        merging their boxes, masks, probs, keypoints, and/or OBB (Oriented Bounding Box) objects.
        
        Args:
            results (list): List of Results objects to combine. All Results should be for the same image.
            
        Returns:
            Results: A single combined Results object containing combined detections.
            
        Note:
            - If the input Results objects contain different types of data (e.g., some have boxes, some have masks),
              the combined Results object will contain all available data.
            - For classification results (probs), only the highest confidence classification is kept.
        """
        if not results:
            return None
        
        if any(r is None for r in results):
            raise ValueError("Cannot combine None results. Please provide valid Results objects.")
        
        if len(results) == 1:
            return results[0]
            
        # Ensure all results are for the same image
        first_path = results[0].path
        if not all(r.path == first_path for r in results):
            print("Warning: Attempting to combine Results from different images. Using the first image.")
        
        # Get the first result's data for base attributes
        combined_result = results[0].new()
        
        # combine boxes if any exist
        all_boxes = [r.boxes.data for r in results if r.boxes is not None]
        if all_boxes:
            # Concatenate all boxes
            if isinstance(all_boxes[0], torch.Tensor):
                combined_boxes = torch.cat(all_boxes, dim=0)
            else:
                combined_boxes = np.concatenate(all_boxes, axis=0)
            combined_result.update(boxes=combined_boxes)
        
        # combine masks if any exist
        all_masks = [r.masks.data for r in results if r.masks is not None]
        if all_masks:
            # Concatenate all masks
            if isinstance(all_masks[0], torch.Tensor):
                combined_masks = torch.cat(all_masks, dim=0)
            else:
                combined_masks = np.concatenate(all_masks, axis=0)
            combined_result.update(masks=combined_masks)
        
        # For classification results (probs), keep the one with highest confidence
        all_probs = [(r.probs, r.probs.top1conf) for r in results if r.probs is not None]
        if all_probs:
            # Get the probs object with highest top1 confidence
            best_probs = max(all_probs, key=lambda x: x[1])[0]
            combined_result.update(probs=best_probs.data)
        
        # combine keypoints if any exist
        all_keypoints = [r.keypoints.data for r in results if r.keypoints is not None]
        if all_keypoints:
            # Concatenate all keypoints
            if isinstance(all_keypoints[0], torch.Tensor):
                combined_keypoints = torch.cat(all_keypoints, dim=0)
            else:
                combined_keypoints = np.concatenate(all_keypoints, axis=0)
            combined_result.update(keypoints=combined_keypoints)
        
        # combine OBB (Oriented Bounding Boxes) if any exist
        all_obbs = [r.obb.data for r in results if r.obb is not None]
        if all_obbs:
            # Concatenate all OBBs
            if isinstance(all_obbs[0], torch.Tensor):
                combined_obbs = torch.cat(all_obbs, dim=0)
            else:
                combined_obbs = np.concatenate(all_obbs, axis=0)
            combined_result.update(obb=combined_obbs)
        
        # combine names dictionaries from all results
        combined_names = {}
        for r in results:
            if r.names:
                combined_names.update(r.names)
        combined_result.names = combined_names
        
        return combined_result