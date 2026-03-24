Here is the highly detailed implementation plan for **Phase 3: The "Read" Operation & Secondary Ray True Occlusion**. 

This phase completes the synchronization loop, ensuring that once the 3D model is updated, all 2D views instantly reflect the change. It also finalizes the intelligent ray-casting logic so secondary rays obey physical occlusion. 

***

# Detailed Implementation Plan: Phase 3 ("Read" Operation & True Occlusion)

## Overview
The agent will implement the mechanisms that allow the 2D UI to react to the 3D ground truth.
1. **The Read Operation:** Instantly push updates from the 3D master array to all 2D camera masks using the Inverted Index, bypassing heavy 3D-to-2D projections.
2. **Fast Mask Updating:** Add an optimized 1D array assignment method to `MaskAnnotation`.
3. **True Occlusion Rays:** Implement the "3D Distance Threshold" logic so secondary rays accurately stop at physical obstructions instead of clipping through them. 

---

### Step 3.1: Implement Highly Optimized 2D Mask Updating
**Target File:** `QtMaskAnnotation.py` (or equivalent file defining `MaskAnnotation`)

**Context:** The standard `update_mask` method relies on 2D boolean masks and bounding boxes. We need a faster method that writes directly to flat memory indices provided by the Inverted Index.

**Tasks for the Agent:**
1.  **Add `update_mask_at_indices`:**
    * **Signature:** `def update_mask_at_indices(self, flat_indices: np.ndarray, class_id: int):`
    * **Algorithm:**
        1. Ensure `flat_indices` is a valid 1D NumPy array. If empty, return early.
        2. Flatten the existing 2D mask array: `flat_mask = self.mask_data.ravel()`.
        3. Apply the new class ID directly: `flat_mask[flat_indices] = class_id`.
        4. Reshape back to 2D: `self.mask_data = flat_mask.reshape(self.mask_data.shape)`.
        5. Register the `class_id` to `self.visible_label_ids` if not already present.
        6. Trigger the visual UI update: `self.update_graphics_item()`.

---

### Step 3.2: Implement the "Read" Operation (Instant Sync)
**Target File:** `MVATManager.py`

**Context:** Immediately after Phase 2 writes `painted_element_ids` to the 3D model, this step pushes the visual update to all cameras.

**Tasks for the Agent:**
1.  **Update Propagation Logic (`_on_brush_stroke_applied` & SAM equivalent):**
    * After the `primary_target.update_labels(...)` call from Phase 2.
    * Iterate through all currently loaded cameras (both the active camera and those in the `ContextMatrix`).
    * **Step A (Query Inverted Index):** For each `camera`:
        * Call `flat_pixel_indices = camera.get_pixels_for_elements(painted_element_ids)`.
    * **Step B (Update Mask):**
        * If `flat_pixel_indices.size > 0`:
            * Retrieve the camera's `target_mask = camera._raster.get_mask_annotation(...)`.
            * Resolve the global `class_id` back to the target camera's local `label_id` mapping if necessary (or just pass the global `class_id` if mappings are unified).
            * Call `target_mask.update_mask_at_indices(flat_pixel_indices, class_id)`.
            * Refresh the ContextMatrix canvas overlay: `context_canvas.set_mask_overlay(target_mask)`.
    * *Agent Note:* Ensure this loop is strictly using NumPy operations and does not trigger redundant UI repaints until the loop finishes.

---

### Step 3.3: Implement Secondary Ray True Occlusion (3D Distance Threshold)
**Target File:** `MVATManager.py` (`MousePositionBridge._process_pending_position`)

**Context:** Secondary rays must stop at physical occluders. We use a 3D distance tolerance to account for sub-pixel aliasing.

**Tasks for the Agent:**
1.  **Update the Highlighted Cameras Loop:**
    * The primary ray provides `primary_3d_coord` and `primary_element_id` (from Phase 2).
    * For each `target_cam` in `highlighted_cameras`:
        1.  Project `primary_3d_coord` to 2D -> `(u', v')`.
        2.  Clamp to integer bounds. If out of bounds, ray is **Occluded (Red)**.
        3.  Look up the ID at that pixel: `found_id = target_cam.index_map[v', u']`.
        4.  **Condition 1 (Exact Match):** If `found_id == primary_element_id`:
            * Ray is **Visible (Cyan)**. Terminal point = `primary_3d_coord`.
        5.  **Condition 2 (Tolerance Check):** If `found_id != primary_element_id` AND `found_id > -1`:
            * Get the 3D coord of the obstruction: `found_3d_coord = primary_target.get_element_coordinate(found_id)`.
            * Calculate distance: `dist = np.linalg.norm(found_3d_coord - primary_3d_coord)`.
            * Calculate tolerance (e.g., 5% of distance to camera): `cam_dist = np.linalg.norm(primary_3d_coord - target_cam.position)` -> `tolerance = 0.05 * cam_dist`.
            * If `dist <= tolerance`: Ray is **Visible (Cyan)**. Terminal point = `primary_3d_coord`.
            * If `dist > tolerance`: Ray is **Occluded (Red)**. Terminal point = `found_3d_coord` (ray stops at the occluder).
        6.  **Condition 3 (Background Hit):** If `found_id == -1`:
            * Ray is **Occluded (Red)**. Terminal point is computed by extending a vector from the camera through `(u', v')` to a default distance.
2.  **Pass State to Rendering:**
    * Collect the rays and their computed colors (`RAY_COLOR_HIGHLIGHTED` for Cyan/Visible, `RAY_COLOR_INVALID` for Red/Occluded).
    * Send to `self.manager.viewer.show_rays(rays_with_colors)`.

***