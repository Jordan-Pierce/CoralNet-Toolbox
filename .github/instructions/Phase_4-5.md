Here are the highly detailed implementation plans for the final pieces of the puzzle: **Phase 4 (Camera Initialization)** and **Phase 5 (3D Label Visualization & Channel Switcher UI)**. 

These instructions will guide the agent to seamlessly map the 3D labels back to 2D when a camera is loaded, and construct the VTK/PyVista logic to display those labels dynamically in the 3D viewport.

***

# Detailed Implementation Plan: Phase 4 (Camera Initialization & On-the-Fly Masks)

## Overview
Instead of relying on outdated 2D PNG masks saved to disk, cameras will now generate their `MaskAnnotation` dynamically the moment they are loaded or requested. They will project the master 3D labels into their 2D view using the `index_map`. 

---

### Step 4.1: Dynamic Mask Generation
**Target File:** `QtRaster.py` (specifically `get_mask_annotation`)

**Context:** The `Raster.get_mask_annotation` method currently initializes a blank zero-array if no mask exists. We must change this to pull data from the 3D master array.

**Tasks for the Agent:**
1.  **Modify `get_mask_annotation` Signature:**
    * Allow the method to accept a reference to the `primary_target` (the 3D scene product).
2.  **Implement the Mapping Algorithm:**
    * If `self.mask_annotation` is `None` (meaning it's being initialized):
        1. Fetch the camera's `index_map`. If `index_map` is `None`, fallback to the legacy `np.zeros` logic.
        2. Fetch the master 3D labels: `master_labels = primary_target.get_labels()`.
        3. Create a blank 2D array: `dynamic_mask = np.zeros_like(self.index_map, dtype=np.uint8)`.
        4. Create a valid mask to ignore the background: `valid_pixels = self.index_map > -1`.
        5. **The Magic Mapping:** `dynamic_mask[valid_pixels] = master_labels[self.index_map[valid_pixels]]`.
        6. Initialize the `MaskAnnotation` with this `dynamic_mask` instead of `np.zeros`.
3.  **Handle Re-initialization:**
    * If the camera was unloaded from memory but is now being viewed again, this logic guarantees it instantly displays the most up-to-date labels from the 3D scene.

***

# Detailed Implementation Plan: Phase 5 (3D Label Visualization & Channel Switcher)

## Overview
The 3D viewer must allow the user to toggle between photorealistic "RGB" colors and semantic "Labels". The agent will build a UI combobox, generate a VTK Lookup Table (LUT) from the project's label colors, and dynamically swap the scalar arrays on the PyVista actors. 

---

### Step 5.1: Build the UI Combobox
**Target File:** `QtMVATViewer.py` (`create_top_toolbar`)

**Context:** The user needs a simple dropdown in the 3D viewer's toolbar to switch rendering modes.

**Tasks for the Agent:**
1.  **Create the Widget:**
    * Inside `create_top_toolbar`, instantiate a `QComboBox`.
    * Add two items: `"RGB"` and `"Labels"`.
    * Add it to the toolbar layout.
2.  **Connect the Signal:**
    * Connect the combobox's `currentTextChanged` signal to a new method: `self.set_display_channel(channel_name)`.

---

### Step 5.2: Generate the Project Colormap (Lookup Table)
**Target File:** `QtMVATViewer.py`

**Context:** PyVista/VTK needs to know exactly what RGB color corresponds to `class_id = 1`, `class_id = 2`, etc.

**Tasks for the Agent:**
1.  **Create `_build_label_colormap(self, project_labels)`:**
    * **Input:** The list of `Label` objects from the project (accessible via `self.parent().label_window.labels` if `MVATManager` passes them down).
    * **Algorithm:**
        1. Find the maximum `class_id` currently in use.
        2. Create a NumPy array for colors: `colors = np.zeros((max_class_id + 1, 3), dtype=np.uint8)`.
        3. Set index `0` (Background) to a neutral color, e.g., `[50, 50, 50]` or `[0, 0, 0]`.
        4. Loop through the `project_labels`. For each label, convert its hex color to an RGB tuple and assign it to `colors[label.class_id]`.
        5. Convert this NumPy array into a format PyVista/VTK accepts (e.g., a `vtkLookupTable` or a Matplotlib `ListedColormap`).

---

### Step 5.3: Implement the Scalar Swapping Logic
**Target File:** `QtMVATViewer.py` (and supporting updates in `Model.py`)

**Context:** When the user changes the combobox, the actors must swap which data array they use for coloring.

**Tasks for the Agent:**
1.  **Ensure Labels are bound to the Mesh (`Model.py`)**:
    * In `PointCloudProduct.update_labels` and `MeshProduct.update_labels`:
        * Make sure `self.mesh.point_data['Labels'] = self.labels` (or `cell_data`) is executed so the array actually exists on the PyVista object.
2.  **Implement `set_display_channel(self, channel_name)` (`QtMVATViewer.py`)**:
    * Iterate over all active actors in `self._product_actors.values()`.
    * **If `channel_name == "RGB"`:**
        * Check if the mesh has an `'RGB'` array.
        * `actor.mapper.SetScalarModeToUsePointFieldData()` (or `CellFieldData` for meshes).
        * `actor.mapper.SelectColorArray('RGB')`.
        * **CRITICAL:** `actor.mapper.SetColorModeToDirectScalars()` (Treat the 3-component array as raw RGB values, do not map them).
    * **If `channel_name == "Labels"`:**
        * Check if the mesh has a `'Labels'` array.
        * `actor.mapper.SelectColorArray('Labels')`.
        * **CRITICAL:** `actor.mapper.SetColorModeToMapScalars()` (Treat the 1-component array as indices to be mapped through a LUT).
        * Apply the colormap generated in Step 5.2: `actor.mapper.SetLookupTable(custom_lut)`.
    * Call `self.plotter.render()` to instantly redraw the scene.

***

With these final two phases, the transition to a 3D Source of Truth will be complete. The agent now has a start-to-finish roadmap to implement the Inverted Index, intercept the 2D paint strokes, write them to the 3D model, instantly sync the 2D masks, and visualize the labels natively in the 3D viewer.