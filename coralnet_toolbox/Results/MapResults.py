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
    
    # Tolerance for boundary detection (in pixels)
    BOUNDARY_TOLERANCE = 1.0

    def __init__(self):
        pass
    
    def _is_box_on_boundary(self, xmin, ymin, xmax, ymax, wa_x, wa_y, wa_x_max, wa_y_max):
        """
        Check if a bounding box touches or extends beyond the work area boundary.
        
        Args:
            xmin, ymin, xmax, ymax: Box coordinates
            wa_x, wa_y: Top-left corner of work area
            wa_x_max, wa_y_max: Bottom-right corner of work area
            
        Returns:
            bool: True if box touches or crosses boundary, False otherwise
        """
        return (
            (xmin <= wa_x + self.BOUNDARY_TOLERANCE) or
            (ymin <= wa_y + self.BOUNDARY_TOLERANCE) or
            (xmax >= wa_x_max - self.BOUNDARY_TOLERANCE) or
            (ymax >= wa_y_max - self.BOUNDARY_TOLERANCE)
        )
    
    def _is_polygon_on_boundary(self, points_xy, wa_x, wa_y, wa_x_max, wa_y_max):
        """
        Check if any vertex of a polygon touches or extends beyond the work area boundary.
        
        Args:
            points_xy: Polygon vertices as numpy array of shape (N, 2) in pixel coordinates
            wa_x, wa_y: Top-left corner of work area
            wa_x_max, wa_y_max: Bottom-right corner of work area
            
        Returns:
            bool: True if any vertex touches or is outside boundary, False otherwise
        """
        if len(points_xy) == 0:
            return False
        
        for x, y in points_xy:
            x, y = float(x), float(y)
            if (
                (x <= wa_x + self.BOUNDARY_TOLERANCE) or
                (y <= wa_y + self.BOUNDARY_TOLERANCE) or
                (x >= wa_x_max - self.BOUNDARY_TOLERANCE) or
                (y >= wa_y_max - self.BOUNDARY_TOLERANCE)
            ):
                return True
        
        return False
        
    def map_results_from_work_area(self, results, raster, work_area, map_masks=True, task='segment'):
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
            mapped_results = self._map_masks(results, mapped_results, raster, wa_x, wa_y, wa_w, wa_h, task=task)
            
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
            
            # Decide which boxes touch the work-area border and drop them.
            wa_x = int(working_area_top_left.x())
            wa_y = int(working_area_top_left.y())
            wa_x_max = wa_x + int(wa_w)
            wa_y_max = wa_y + int(wa_h)
            
            # boxes_abs is CPU tensor (Nx4)
            boxes_np = boxes_abs.detach().cpu().numpy()
            touching_flags = []
            for b in boxes_np:
                xmin, ymin, xmax, ymax = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                touches = self._is_box_on_boundary(xmin, ymin, xmax, ymax, wa_x, wa_y, wa_x_max, wa_y_max)
                touching_flags.append(bool(touches))
            
            # kept_indices = indices that do NOT touch the border
            kept_indices = [i for i, t in enumerate(touching_flags) if not t]
            mapped_results._kept_indices = kept_indices
            
            # Filter mapped_boxes to only kept indices (preserve downstream alignment)
            if len(kept_indices) == 0:
                # create empty boxes tensor with correct number of columns
                mapped_results.update(boxes=torch.empty((0, mapped_boxes.shape[1])))
            else:
                idx_tensor = torch.tensor(kept_indices, dtype=torch.long)
                mapped_results.update(boxes=mapped_boxes[idx_tensor])
            
        return mapped_results
    
    def _map_masks(self, results, mapped_results, raster, wa_x, wa_y, wa_w, wa_h, task='segment'):
        """
        Maps masks from work area to original image coordinates.
        
        Args:
            results: Original Results object
            mapped_results: New Results object to be updated
            raster: Raster object containing the original image dimensions
            wa_x, wa_y: Top-left coordinates of the work area
            wa_w, wa_h: Width and height of the work area
            task: The type of task ('segment' or 'semantic')
            
        Returns:
            Results: Updated Results object with mapped masks
        """
        if results.masks is not None and len(results.masks) > 0:
            orig_h, orig_w = raster.height, raster.width
            device = results.masks.data.device
            kept_indices = getattr(mapped_results, "_kept_indices", None)
            
            # If the input masks already have polygon representations, use them directly
            if task == 'segment' and hasattr(results.masks, 'xy') and results.masks.xy:
                segments_xy = []
                segments_xyn = []
                boundary_filtered_indices = []  # Track indices after boundary filtering
                
                # Work area boundaries for boundary checking
                wa_x_max = wa_x + int(wa_w)
                wa_y_max = wa_y + int(wa_h)
                
                # results.masks.data corresponds to the same ordering as results.masks.xy
                for i, points in enumerate(results.masks.xy):
                    # If kept_indices is provided, skip indices that were dropped by boxes filtering
                    if kept_indices is not None and i not in kept_indices:
                        continue
                    
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
                        
                        # Check if polygon touches boundary
                        if self._is_polygon_on_boundary(mapped_points_xy, wa_x, wa_y, wa_x_max, wa_y_max):
                            continue  # Skip this polygon
                        
                        segments_xy.append(mapped_points_xy)
                        
                        # Store the normalized coordinates as well
                        mapped_points_xyn = mapped_points_xy.copy()
                        mapped_points_xyn[:, 0] /= orig_w
                        mapped_points_xyn[:, 1] /= orig_h
                        segments_xyn.append(mapped_points_xyn)
                        
                        boundary_filtered_indices.append(i)
                    else:
                        # Empty contour case
                        segments_xy.append(np.zeros((0, 2), dtype=np.float32))
                        segments_xyn.append(np.zeros((0, 2), dtype=np.float32))
                        boundary_filtered_indices.append(i)
                
                # Filter mask tensor data to boundary-filtered indices
                mask_data = results.masks.data
                if len(boundary_filtered_indices) == 0:
                    # empty mask set
                    filtered_mask_data = torch.zeros((0, orig_h, orig_w), device=device, dtype=mask_data.dtype)
                else:
                    idx_tensor = torch.tensor(boundary_filtered_indices, dtype=torch.long, device=mask_data.device)
                    filtered_mask_data = mask_data[idx_tensor]
                
                # Create a new Masks object using filtered data
                mapped_masks = Masks(filtered_mask_data, orig_shape=(orig_h, orig_w))
                
                # Set polygon coordinates
                mapped_masks._xy = segments_xy
                mapped_masks._xyn = segments_xyn
                
                # Assign the new masks object to mapped_results
                mapped_results.masks = mapped_masks
                
            # ELSE: This is raster data (from SemanticModel), not polygons.
            # We must create a new full-size tensor and paste the tile into it.
            else:
                # Get the original tile mask data
                tile_masks = results.masks.data  # (N, tile_h, tile_w) e.g., (N, 640, 640)
                n, tile_h, tile_w = tile_masks.shape

                # 1. Create a new, empty, full-size tensor for the mapped masks
                full_masks = torch.zeros((n, orig_h, orig_w), device=device, dtype=tile_masks.dtype)

                # 2. Get destination coordinates
                y_start_dest, x_start_dest = wa_y, wa_x
                y_end_dest, x_end_dest = wa_y + tile_h, wa_x + tile_w
                
                # 3. Clip destination coordinates to full image bounds
                y_start_dest_clip = max(0, y_start_dest)
                x_start_dest_clip = max(0, x_start_dest)
                y_end_dest_clip = min(orig_h, y_end_dest)
                x_end_dest_clip = min(orig_w, x_end_dest)

                # 4. Calculate source coordinates from the tile based on clipping
                y_start_src = y_start_dest_clip - y_start_dest
                x_start_src = x_start_dest_clip - x_start_dest
                y_end_src = y_start_src + (y_end_dest_clip - y_start_dest_clip)
                x_end_src = x_start_src + (x_end_dest_clip - x_start_dest_clip)
                
                # 5. Paste the tile mask data into the full mask tensor
                full_masks[:, y_start_dest_clip:y_end_dest_clip, x_start_dest_clip:x_end_dest_clip] = \
                    tile_masks[:, y_start_src:y_end_src, x_start_src:x_end_src]
                
                # 6. Filter full_masks by kept indices if present, then create the Masks object
                if kept_indices is not None:
                    if len(kept_indices) == 0:
                        mapped_masks = Masks(torch.zeros((0, orig_h, orig_w), 
                                                         device=device, dtype=full_masks.dtype), 
                                             orig_shape=(orig_h, orig_w))
                    else:
                        idx_tensor = torch.tensor(kept_indices, dtype=torch.long, device=full_masks.device)
                        mapped_masks = Masks(full_masks[idx_tensor], orig_shape=(orig_h, orig_w))
                else:
                    mapped_masks = Masks(full_masks, orig_shape=(orig_h, orig_w))
                
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