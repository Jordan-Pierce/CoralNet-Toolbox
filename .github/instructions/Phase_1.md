Here is the highly detailed, step-by-step implementation plan for **Phase 1: The 3D Data Layer & Inverted Index**. This is written specifically for a coding agent to follow, ensuring memory efficiency, precise data types, and robust state management.

***

# Detailed Implementation Plan: Phase 1 (3D Data Layer & Inverted Index)

## Overview
This phase establishes the foundational data structures for the 3D-Ground-Truth architecture. The agent will:
1. Embed persistent, memory-efficient label arrays directly into the 3D scene products.
2. Add $O(1)$ 3D coordinate lookups for precise ray-casting.
3. Build and cache a highly optimized "Inverted Index" (using a Compressed Sparse Row / CSR format) during the visibility pass to enable instant 3D-to-2D pixel mapping without locking the UI.

---

### Step 1.1: Equip 3D Scene Products with Label Storage & Lookups
**Target File:** `Model.py`

**Context:** The 3D products must now act as the master semantic record and provide exact 3D coordinates for incoming 2D rays.

**Tasks for the Agent:**
1.  **Modify `AbstractSceneProduct`**:
    * Define an abstract property/method to initialize a `self.labels` array.
    * Define an abstract method: `get_element_coordinate(self, element_id: int) -> np.ndarray`.
2.  **Modify `PointCloudProduct`**:
    * In `__init__`, after loading the mesh, initialize: `self.labels = np.zeros(self.mesh.n_points, dtype=np.uint16)`.
    * Implement `update_labels(self, element_ids: np.ndarray, class_id: int)`:
        * `self.labels[element_ids] = class_id`
        * **Crucial for Phase 6:** Sync to PyVista: `self.mesh.point_data['Labels'] = self.labels`
    * Implement `get_element_coordinate(self, element_id: int)`:
        * Return `self.mesh.points[element_id]` (Safe because point IDs are generated sequentially 0..N-1).
3.  **Modify `MeshProduct` & `DEMProduct`**:
    * In `__init__` (or after geometry extraction), initialize: `self.labels = np.zeros(self.mesh.n_cells, dtype=np.uint16)`.
    * Implement `update_labels`:
        * `self.labels[element_ids] = class_id`
        * Sync to PyVista: `self.mesh.cell_data['Labels'] = self.labels`
    * Implement `get_element_coordinate(self, element_id: int)`:
        * Return `self.mesh.cell_centers().points[element_id]`.

---

### Step 1.2: Generate the CSR Inverted Index during Visibility Computation
**Target File:** `VisibilityManager.py`

**Context:** Storing a standard Python dictionary mapping millions of `element_id`s to arrays of pixels will cause a massive memory bottleneck. The agent *must* use a Compressed Sparse Row (CSR) array approach.

**Tasks for the Agent:**
1.  **Create a Helper Function `_build_csr_inverted_index(index_map)`**:
    * **Input:** `index_map` (2D NumPy array of `int32` from the visibility pass).
    * **Algorithm:**
        1. Flatten the `index_map` to 1D.
        2. Create an array of `pixel_indices = np.arange(index_map.size)`.
        3. Create a valid mask to drop the background: `valid = flat_index_map > -1`.
        4. Apply mask: `valid_ids = flat_index_map[valid]`, `valid_pixels = pixel_indices[valid]`.
        5. Sort both arrays based on the `valid_ids` using `sort_idx = np.argsort(valid_ids)`.
        6. Apply sort: `sorted_ids = valid_ids[sort_idx]`, `sorted_pixels = valid_pixels[sort_idx]`.
        7. Use `np.unique(sorted_ids, return_counts=True)` to get `unique_ids` and `counts`.
        8. Calculate `row_pointers = np.zeros(len(counts) + 1, dtype=np.int32)`.
        9. `row_pointers[1:] = np.cumsum(counts)`.
    * **Output:** Return a dictionary: `{'unique_ids': unique_ids, 'pixel_indices': sorted_pixels, 'row_pointers': row_pointers}`.
2.  **Integrate into Visibility Passes**:
    * In `_compute_numpy`, `_compute_torch`, and `_compute_ortho_*`, after generating the final `index_map_2d`, call the helper function.
    * Add the returned CSR dictionary to the final output dictionary under the key `'inverted_index'`.

---

### Step 1.3: Cache the Inverted Index to Disk
**Target File:** `CacheManager.py`

**Context:** The CSR arrays must be serialized alongside the `index_map` and `visible_indices` to prevent recomputation.

**Tasks for the Agent:**
1.  **Modify `save_visibility`**:
    * Add an optional argument `inverted_index: Optional[Dict] = None`.
    * If provided, extract the three arrays and add them to the `save_dict`:
        * `save_dict['inv_unique_ids'] = inverted_index['unique_ids']`
        * `save_dict['inv_pixel_indices'] = inverted_index['pixel_indices']`
        * `save_dict['inv_row_pointers'] = inverted_index['row_pointers']`
    * Save via the existing `np.savez_compressed`.
2.  **Modify `load_visibility`**:
    * Check if `inv_unique_ids` is in the loaded `.npz` data.
    * If present, reconstruct the dictionary: `result['inverted_index'] = {'unique_ids': data['inv_unique_ids'], ...}`.

---

### Step 1.4: Runtime Retrieval and State Management
**Target Files:** `QtRaster.py`, `Camera.py`

**Context:** The `Camera` object needs an ultra-fast query method to resolve 3D `element_id`s into 2D pixel indices using the loaded CSR arrays.

**Tasks for the Agent:**
1.  **Modify `Raster` class (`QtRaster.py`)**:
    * Add `self.inverted_index = None`.
    * Update `add_index_map` (and references to it in `MVATManager.py`) to accept `inverted_index` and store it on the Raster.
2.  **Modify `Camera` class (`Camera.py`)**:
    * Implement a new method: `get_pixels_for_elements(self, element_ids: np.ndarray) -> np.ndarray`.
    * **Algorithm for the Agent:**
        1. Access `self._raster.inverted_index`. If `None`, return an empty array.
        2. Extract `unique_ids`, `pixel_indices`, `row_pointers`.
        3. Use `np.searchsorted(unique_ids, element_ids)` to find the indices of the requested elements in the `unique_ids` array.
        4. **Safety Check:** Ensure the found indices actually match the requested `element_ids` (handling IDs that aren't in the view).
        5. For each valid index `i`, extract the pixel slice: `pixel_indices[row_pointers[i] : row_pointers[i+1]]`.
        6. `np.concatenate` the slices and return a single flat 1D NumPy array of pixel indices.

***