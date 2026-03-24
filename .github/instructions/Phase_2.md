Here is the highly detailed implementation plan for **Phase 2: The "Write" Operation & Primary Ray Snapping**. 

This phase connects the UI interactions (mouse hovering and painting) directly to the 3D data layer we built in Phase 1.

***

# Detailed Implementation Plan: Phase 2 (Write Operation & Ray Snapping)

## Overview
The agent will rewrite how the `MVATManager` interprets user input. 
1. **Ray Snapping:** The `MousePositionBridge` will stop guessing 3D coordinates using floating-point depth maps, and instead look up the exact 3D coordinate of the element under the cursor using the Index Map.
2. **The Write Operation:** When a brush stroke or SAM mask is applied, the system will extract the painted 3D elements from the index map and write the new label directly to the 3D master array, bypassing local 2D mask editing.

---

### Step 2.1: Implement Primary Ray Snapping
**Target File:** `MVATManager.py` (`MousePositionBridge._process_pending_position`)

**Context:** The primary ray dictates the focal point for all crosshairs. It must snap to the exact 3D geometry.

**Tasks for the Agent:**
1.  **Retrieve the `element_id`:**
    * Given the hover pixel `(x, y)` and the `selected_camera`, access the camera's index map: `element_id = camera.index_map[y, x]`.
2.  **Look up the 3D Coordinate:**
    * If `element_id > -1`, fetch the exact 3D point from the primary scene product: `exact_3d_point = primary_target.get_element_coordinate(element_id)`.
    * *Agent Note:* Ensure `get_element_coordinate` returns a standard `np.ndarray` of shape `(3,)`.
3.  **Construct the Primary Ray:**
    * Use a direct initialization of `CameraRay` rather than `from_pixel_and_camera` if you have the exact point:
        ```python
        direction = exact_3d_point - camera.position
        direction /= np.linalg.norm(direction)
        ray = CameraRay(
            origin=camera.position,
            direction=direction,
            terminal_point=exact_3d_point,
            has_accurate_depth=True,
            pixel_coord=(x, y),
            source_camera=camera
        )
        ```
    * *Fallback:* If `element_id == -1` (or the index map is missing), fall back to the existing `CameraRay.from_pixel_and_camera` depth-unprojection logic.
4.  **Short-Circuit Invalid Rays:**
    * If the ray is totally invalid (no `element_id` and no valid depth), immediately `return` after drawing the single red primary ray, skipping the highlighted cameras loop to save CPU.

---

### Step 2.2: Implement the "Write" Operation (Brush & SAM)
**Target File:** `MVATManager.py` (`_on_brush_stroke_applied` and SAM equivalent callbacks)

**Context:** When the user paints, we must figure out what 3D elements were touched and update the 3D master array. We no longer write to the 2D mask here.

**Tasks for the Agent:**
1.  **Extract the Painted Slice:**
    * The callback receives `scene_pos` (center of brush) and `brush_mask` (2D boolean numpy array, e.g., 90x90).
    * Calculate the top-left `(start_x, start_y)` of the brush on the active camera's full image: `start_x = int(scene_pos.x() - brush_w / 2.0)`, etc.
    * Extract the corresponding slice from the active camera's index map: `index_slice = camera.index_map[start_y:end_y, start_x:end_x]`.
    * *Safety Check:* Clamp bounds to ensure the slice doesn't exceed image dimensions.
2.  **Extract Unique Element IDs:**
    * Apply the boolean brush mask to the index slice: `painted_ids = index_slice[brush_mask]`.
    * Filter out background: `painted_ids = painted_ids[painted_ids > -1]`.
    * Get unique elements: `unique_painted_ids = np.unique(painted_ids)`.
3.  **Resolve the Class ID:**
    * The callback provides a string `label_id` (UUID).
    * *Agent Note:* A global mapping must exist (e.g., in `SceneContext`) that maps `label_id -> integer class_id` (e.g., `1, 2, 3`). Retrieve this integer `class_id`.
4.  **Update the 3D Master Array:**
    * Call `primary_target.update_labels(unique_painted_ids, class_id)`.
    * *Crucial:* Do **not** call `context_canvas.set_mask_overlay()` or update the local `MaskAnnotation` directly in this function anymore. That will be handled by Phase 3 (The "Read" Operation), which will trigger immediately after this step.

---

### Step 2.3: Ensure Fast Coordinate Lookups in Scene Products
**Target File:** `Model.py`

**Context:** Supporting Step 2.1 requires the products to quickly return the 3D coordinate for any given `element_id`.

**Tasks for the Agent:**
1.  **Update `PointCloudProduct`:**
    * Implement `get_element_coordinate(self, element_id: int) -> np.ndarray`:
        * `return self.mesh.points[element_id]`
2.  **Update `MeshProduct` & `DEMProduct`:**
    * Implement `get_element_coordinate(self, element_id: int) -> np.ndarray`:
        * If GPU tensors exist (`self._cached_face_centers_pt`), return `self._cached_face_centers_pt[element_id].cpu().numpy()`.
        * Fallback: `return self.mesh.cell_centers().points[element_id]`.

***

