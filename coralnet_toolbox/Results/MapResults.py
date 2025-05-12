import gc
import copy
import numpy as np
import torch

from coralnet_toolbox.Results.Masks import Masks


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MapResults:
    """
    A class to map results from one coordinate system to another.
    """

    def __init__(self):
        pass
        
    def map_results_from_work_area(self, results, raster, work_area, map_masks=True):
        """
        Maps coordinates in Results objects from work area to original image coordinates.
        
        Args:
            results: Ultralytics Results object or list of Results objects
            raster: Raster object containing the original image dimensions
            work_area: WorkArea object containing the sub-region information
            map_masks: Boolean indicating whether to map masks or not
            
        Returns:
            Results: Updated Results object with coordinates mapped to the original image
        """
        if results is None:
            return None
                
        # Handle list of results
        if isinstance(results, list):
            return [self.map_results_from_work_area(r, raster, work_area) for r in results]
        
        # Get the raster object to get original image dimensions
        if raster is None:
            return results  # Return original results if raster not found
        
        # Create a new Results object to avoid modifying the original
        mapped_results = copy.deepcopy(results)
        
        # Copy other relevant attributes
        mapped_results.names = results.names
        mapped_results.path = raster.image_path
        mapped_results.orig_shape = raster.shape
        
        # Get work area coordinates for mapping
        working_area_top_left = work_area.rect.topLeft()
        wa_x, wa_y = int(work_area.rect.x()), int(work_area.rect.y())
        wa_w, wa_h = int(work_area.rect.width()), int(work_area.rect.height())
        
        # Map each component separately
        mapped_results = self._map_boxes(results, mapped_results, working_area_top_left, wa_w, wa_h)
        
        if map_masks:
            mapped_results = self._map_masks(results, mapped_results, raster, wa_x, wa_y, wa_w, wa_h)
            
        mapped_results = self._map_probs(results, mapped_results)
        
        gc.collect()
        
        return mapped_results
        
    def _map_boxes(self, results, mapped_results, working_area_top_left, wa_w, wa_h):
        """
        Maps bounding boxes from work area to original image coordinates.
        
        Args:
            results: Original Results object
            mapped_results: New Results object to be updated
            working_area_top_left: Top left corner of the work area
            wa_w, wa_h: Width and height of the work area
            
        Returns:
            Results: Updated Results object with mapped boxes
        """
        if results.boxes is not None and len(results.boxes) > 0:
            # Get xyxyn format (normalized pixel coordinates in the cropped image)
            boxes_xyxyn = results.boxes.xyxyn.detach().cpu().clone()
            
            # Scale box to work area size
            boxes_rel = boxes_xyxyn.clone()
            boxes_rel[:, 0] *= wa_w
            boxes_rel[:, 1] *= wa_h
            boxes_rel[:, 2] *= wa_w
            boxes_rel[:, 3] *= wa_h
            
            # Offset to full image coordinates
            boxes_abs = boxes_rel.clone()
            boxes_abs[:, 0] += working_area_top_left.x()
            boxes_abs[:, 1] += working_area_top_left.y()
            boxes_abs[:, 2] += working_area_top_left.x()
            boxes_abs[:, 3] += working_area_top_left.y()
            
            # Update the confidence values if they exist
            conf = results.boxes.conf.detach().cpu().clone()
            if conf.dim() == 1:
                conf = conf.unsqueeze(1)
            
            # Get the class values
            cls = results.boxes.cls.detach().cpu().clone()
            if cls.dim() == 1:
                cls = cls.unsqueeze(1)
            
            # Create the updated boxes tensor 
            mapped_boxes = torch.cat([boxes_abs, conf, cls], dim=1)
                    
            # Update boxes using the proper method
            mapped_results.update(boxes=mapped_boxes)
            
        return mapped_results
    
    def _map_masks(self, results, mapped_results, raster, wa_x, wa_y, wa_w, wa_h):
        """
        Maps masks from work area to original image coordinates.
        
        Args:
            results: Original Results object
            mapped_results: New Results object to be updated
            raster: Raster object containing the original image dimensions
            wa_x, wa_y: Top-left coordinates of the work area
            wa_w, wa_h: Width and height of the work area
            
        Returns:
            Results: Updated Results object with mapped masks
        """
        if results.masks is not None and len(results.masks) > 0:
            orig_h, orig_w = raster.height, raster.width
            device = results.masks.data.device
            
            # If the input masks already have polygon representations, use them directly
            if hasattr(results.masks, 'xy') and results.masks.xy:
                segments_xy = []
                segments_xyn = []
                for points in results.masks.xy:
                    # Scale and offset points to map from work area to full image coordinates
                    if len(points) > 0:
                        # Convert to numpy array if it's not already
                        if isinstance(points, list):
                            points = np.array(points)
                        # Make a copy to avoid modifying the original
                        mapped_points_xy = points.copy()
                        # Add offset to map from work area to full image
                        mapped_points_xy[:, 0] += wa_x
                        mapped_points_xy[:, 1] += wa_y
                        segments_xy.append(mapped_points_xy)
                        
                        # Store the normalized coordinates as well
                        mapped_points_xyn = points.copy()
                        mapped_points_xyn[:, 0] /= orig_w
                        mapped_points_xyn[:, 1] /= orig_h
                        segments_xyn.append(mapped_points_xyn)
                    else:
                        # Empty contour case
                        segments_xy.append(np.zeros((0, 2), dtype=np.float32))
                        segments_xyn.append(np.zeros((0, 2), dtype=np.float32))
                
                # Create a new Masks object directly using our updated Masks class
                mapped_masks = Masks(results.masks.data, orig_shape=(orig_h, orig_w))
                
                # Just set the xy coordinates directly without a special update method
                mapped_masks._xy = segments_xy  # Set the coordinates directly
                mapped_masks._xyn = segments_xyn
                
                # Assign the new masks object to mapped_results
                mapped_results.masks = mapped_masks
        
        return mapped_results
    
    def _map_probs(self, results, mapped_results):
        """
        Maps classification probabilities from original result to the mapped result.
        
        Args:
            results: Original Results object
            mapped_results: New Results object to be updated
            
        Returns:
            Results: Updated Results object with mapped probabilities
        """
        # If there are probs (classification results), copy them directly
        if hasattr(results, 'probs') and results.probs is not None:
            mapped_results.update(probs=results.probs.data)
            
        return mapped_results