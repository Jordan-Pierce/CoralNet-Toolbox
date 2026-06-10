# MVAT Propagation Performance Plan — Executable Steps

Goal: make label propagation (camera ↔ mesh ↔ cameras, multi-annotate strokes,
semantic-prediction propagation) fast at scale (hundreds of cameras, 16MP images,
multi-million-face meshes) while keeping the UI responsive.

This plan complements `PAINTING_PERFORMANCE_PLAN.md` (which covers the 3D *render*
path). Everything here is the propagation *compute* and 2D mask path.

## Ground rules for the executing agent

1. **One step = one commit.** Complete a step, run its Verify block, commit with the
   given message, then move to the next. Never combine steps.
2. **Anchors are exact strings** from the current files. If an anchor does not match
   exactly (whitespace included), STOP and re-read the file around the stated method —
   do not improvise a different edit.
3. After every step run, from the repo root:
   ```
   python -m py_compile <each file edited in the step>
   ```
   A step is only done when py_compile exits 0 and the step's grep checks pass.
4. Do NOT reformat, re-wrap, or "clean up" code you are not told to touch.
5. `PropagationEngine.__getattr__` delegates unknown attributes to MVATManager.
   Any new instance attribute used by PropagationEngine MUST be assigned in
   `PropagationEngine.__init__`, otherwise attribute lookups silently fall through
   to the manager and produce confusing bugs.
6. Work on the current branch (`dev_mvat`).

Files referenced (all paths relative to repo root):

- `coralnet_toolbox/MVAT/managers/PropagationEngine.py`  (PE)
- `coralnet_toolbox/MVAT/core/Cameras.py`                (CAM)
- `coralnet_toolbox/Annotations/QtMaskAnnotation.py`     (MASK)

---

## Step 1 — Cache the index-map max ID (PE)

**Why:** `_do_universal_propagation` runs `int(np.max(target_index_map))` for every
target camera on every stroke — a full O(H×W) scan of a static array.

**Edit 1.1** — In PE, add a new method directly after `_release_propagation_buffer`
(its body ends with `self._propagation_buffer_pool.setdefault(key, []).append(buffer)`):

```python
    def _get_index_map_max_id(self, raster) -> int:
        """Return int(index_map.max()), cached on the raster per index-map object."""
        index_map = raster.index_map
        if (getattr(raster, '_index_map_max_id', None) is not None
                and getattr(raster, '_index_map_max_id_src', None) == id(index_map)):
            return raster._index_map_max_id
        max_id = int(index_map.max())
        raster._index_map_max_id = max_id
        raster._index_map_max_id_src = id(index_map)
        return max_id
```

**Edit 1.2** — In PE `_do_universal_propagation`, replace the anchor:

```python
                    target_index_map = target_camera._raster.index_map
                    target_mask_data = target_mask.mask_data
                    max_idx = int(np.max(target_index_map))
```

with:

```python
                    target_index_map = target_camera._raster.index_map
                    target_mask_data = target_mask.mask_data
                    max_idx = self._get_index_map_max_id(target_camera._raster)
```

**Verify:** `python -m py_compile coralnet_toolbox/MVAT/managers/PropagationEngine.py`
and `grep -n "np.max(target_index_map)" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
returns nothing.

**Commit:** `perf(MVAT): cache index-map max id per raster in propagation worker`

---

## Step 2 — CSR inverted index: validate once + vectorized gather (CAM)

**Why:** `_query_pixels_from_csr_inverted_index` re-validates the whole CSR structure
(O(n) scans + temp allocations) on every query, and gathers pixel chunks with a
Python per-element loop.

**Edit 2.1** — In CAM, replace the anchor:

```python
    if inv_offsets_arr[0] != 0:
        return None
    if inv_offsets_arr[-1] != inv_pixels_arr.size:
        return None
    if np.any(inv_offsets_arr < 0) or np.any(inv_offsets_arr[1:] < inv_offsets_arr[:-1]):
        return None
    if inv_ids_arr.size > 1 and np.any(inv_ids_arr[1:] < inv_ids_arr[:-1]):
        return None
```

with:

```python
    # Structural validation is O(n) over the index arrays; run it once per CSR
    # build and cache the verdict on the raster keyed by array identity.
    _csr_token = (id(inv_ids), id(inv_offsets), id(inv_pixels))
    if getattr(raster, '_csr_valid_token', None) != _csr_token:
        if inv_offsets_arr[0] != 0:
            return None
        if inv_offsets_arr[-1] != inv_pixels_arr.size:
            return None
        if np.any(inv_offsets_arr < 0) or np.any(inv_offsets_arr[1:] < inv_offsets_arr[:-1]):
            return None
        if inv_ids_arr.size > 1 and np.any(inv_ids_arr[1:] < inv_ids_arr[:-1]):
            return None
        raster._csr_valid_token = _csr_token
```

**Edit 2.2** — In the same function, replace the anchor:

```python
    matched_rows = candidate_rows[row_matches]
    chunks = []
    for row in matched_rows.tolist():
        start = int(inv_offsets_arr[row])
        end = int(inv_offsets_arr[row + 1])
        if end > start:
            chunks.append(inv_pixels_arr[start:end])

    if not chunks:
        return np.empty(0, dtype=np.int64)

    pixels = np.concatenate(chunks).astype(np.int64, copy=False)
```

with:

```python
    matched_rows = candidate_rows[row_matches]
    starts = inv_offsets_arr[matched_rows]
    lengths = inv_offsets_arr[matched_rows + 1] - starts
    nonempty = lengths > 0
    starts = starts[nonempty]
    lengths = lengths[nonempty]
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)

    # Vectorized multi-range gather: output position p belonging to row i maps
    # to starts[i] + (p - exclusive_cumsum(lengths)[i]).
    total = int(lengths.sum())
    cum = np.cumsum(lengths)
    offsets_per_out = np.repeat(starts - (cum - lengths), lengths)
    gather_idx = np.arange(total, dtype=np.int64) + offsets_per_out
    pixels = inv_pixels_arr[gather_idx].astype(np.int64, copy=False)
```

**Verify:** py_compile CAM, then run this formula check (must print `True`):

```
python -c "import numpy as np; starts=np.array([5,20,100],dtype=np.int64); lengths=np.array([3,1,4],dtype=np.int64); cum=np.cumsum(lengths); idx=np.arange(int(lengths.sum()),dtype=np.int64)+np.repeat(starts-(cum-lengths),lengths); print(idx.tolist()==[5,6,7,20,100,101,102,103])"
```

**Commit:** `perf(MVAT): cache CSR validation and vectorize inverted-index gather`

---

## Step 3 — Single-pass full-mask semantic vote extraction (PE)

**Why:** `_extract_semantic_element_votes` (full-mask branch) does one full-frame
boolean pass per class, and `_extract_source_element_ids_from_full_mask` adds one
full-frame cv2.resize per class. With C classes that is ~2C full-frame passes.
One NEAREST resize of the class layer + one combined pass replaces all of it.

**Edit 3.1** — In PE `_extract_semantic_element_votes`, replace the entire `else:`
branch. Anchor (replace ALL of this):

```python
        else:
            semantic_mask = np.asarray(source_mask_annotation.mask_data)
            if semantic_mask.ndim != 2:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

            unique_real_ids = np.unique(semantic_mask % lock_bit)
            unique_real_ids = unique_real_ids[unique_real_ids > 0]
            for real_class_id in unique_real_ids:
                label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                if label is None:
                    continue

                binary_mask = (semantic_mask % lock_bit == real_class_id)
                if not np.any(binary_mask):
                    continue

                raw_element_ids = self._extract_source_element_ids_from_full_mask(
                    source_camera,
                    binary_mask,
                )
                _append_votes(real_class_id, label, raw_element_ids)
```

Replacement:

```python
        else:
            semantic_mask = np.asarray(source_mask_annotation.mask_data)
            if semantic_mask.ndim != 2:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

            # Single pass for ALL classes: bring the class layer down to the
            # index map's resolution once, then read element + class per pixel.
            index_map = raster.index_map
            sem = semantic_mask % lock_bit
            if sem.shape != index_map.shape:
                import cv2
                sem = cv2.resize(
                    np.ascontiguousarray(sem),
                    (index_map.shape[1], index_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            valid = (sem > 0) & (index_map > -1)
            if np.any(valid):
                elements = index_map[valid].astype(np.int64, copy=False)
                classes = sem[valid].astype(np.int64, copy=False)

                # Register labels; drop votes for classes with no live label.
                unique_ids = np.unique(classes)
                keep_ids = []
                for real_class_id in unique_ids:
                    label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                    if label is not None:
                        class_label_ids[int(real_class_id)] = label.id
                        keep_ids.append(int(real_class_id))

                if len(keep_ids) != len(unique_ids):
                    keep_lut = np.zeros(int(unique_ids.max()) + 1, dtype=bool)
                    if keep_ids:
                        keep_lut[np.asarray(keep_ids, dtype=np.int64)] = True
                    keep_mask = keep_lut[classes]
                    elements = elements[keep_mask]
                    classes = classes[keep_mask]

                if elements.size:
                    element_chunks.append(elements)
                    class_chunks.append(classes)
```

Notes: `raster` is already defined earlier in this method (`raster = getattr(source_camera, '_raster', None)`),
and the method already verified `raster.index_map` is not None. The region branch
(`if prediction_regions is not None:`) stays unchanged — tiles are small.

**Verify:** py_compile PE. `grep -n "_extract_source_element_ids_from_full_mask" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
must now show only the function *definition* (the helper stays; it simply has no
caller in this method anymore — do NOT delete it).

**Commit:** `perf(MVAT): single-pass semantic vote extraction for full masks`

---

## Step 4 — Weighted per-camera vote reduction for Cameras→Mesh (PE)

**Why:** `_bg_aggregate_cameras_to_mesh` pools one int64 entry **per labeled pixel
per camera** before a single giant `np.unique`. At 100+ cameras this is multi-GB.
Reducing each camera's votes to `(element, class, count)` first bounds the merge by
unique pairs, not pixels.

**Edit 4.1** — In PE, change the signature and counting of
`resolve_class_conflicts_vectorized`. Replace the anchor:

```python
def resolve_class_conflicts_vectorized(element_ids: np.ndarray, class_ids: np.ndarray):
    """Resolve per-element class conflicts using vectorized vote counts."""
    try:
        element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
    except Exception:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if element_ids.size == 0 or class_ids.size == 0 or element_ids.size != class_ids.size:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    max_classes = max(100000, int(np.max(class_ids)) + 1)
    compound_ids = (element_ids * max_classes) + class_ids

    unique_compounds, vote_counts = np.unique(compound_ids, return_counts=True)
```

with:

```python
def resolve_class_conflicts_vectorized(element_ids: np.ndarray, class_ids: np.ndarray, weights=None):
    """Resolve per-element class conflicts using vectorized vote counts.

    ``weights`` (optional) carries pre-aggregated vote counts per row, so callers
    can pool reduced (element, class, count) triples instead of per-pixel votes.
    """
    try:
        element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
    except Exception:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if element_ids.size == 0 or class_ids.size == 0 or element_ids.size != class_ids.size:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    max_classes = max(100000, int(np.max(class_ids)) + 1)
    compound_ids = (element_ids * max_classes) + class_ids

    if weights is None:
        unique_compounds, vote_counts = np.unique(compound_ids, return_counts=True)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != compound_ids.size:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        unique_compounds, inverse = np.unique(compound_ids, return_inverse=True)
        vote_counts = np.bincount(inverse, weights=weights)
```

**Edit 4.2** — Add a module-level helper right after `resolve_class_conflicts_vectorized`
(before `def _merge_update_rects`):

```python
def reduce_vote_arrays(element_ids: np.ndarray, class_ids: np.ndarray):
    """Collapse duplicate (element, class) votes into unique pairs + counts."""
    element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
    class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
    if element_ids.size == 0 or element_ids.size != class_ids.size:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    max_classes = max(100000, int(class_ids.max()) + 1)
    compound = (element_ids * max_classes) + class_ids
    uniq, counts = np.unique(compound, return_counts=True)
    return uniq // max_classes, uniq % max_classes, counts
```

**Edit 4.3** — In `_bg_aggregate_cameras_to_mesh`, replace the anchor:

```python
        all_element_ids = []
        all_class_ids = []
```

with:

```python
        all_element_ids = []
        all_class_ids = []
        all_vote_counts = []
```

**Edit 4.4** — In the same method, replace the anchor:

```python
            all_element_ids.append(element_ids[valid])
            all_class_ids.append(canonical_ids[valid])
            contributing += 1
```

with:

```python
            # Collapse this camera's per-pixel votes to (element, class, count)
            # before pooling — bounds the merged arrays by unique pairs rather
            # than by total labeled pixels across all cameras.
            cam_elements, cam_classes, cam_counts = reduce_vote_arrays(
                element_ids[valid], canonical_ids[valid]
            )
            all_element_ids.append(cam_elements)
            all_class_ids.append(cam_classes)
            all_vote_counts.append(cam_counts)
            contributing += 1
```

**Edit 4.5** — In the same method, replace the anchor:

```python
        element_ids = np.concatenate(all_element_ids).astype(np.int64, copy=False)
        class_ids = np.concatenate(all_class_ids).astype(np.int64, copy=False)

        unique_elements, winning_classes = resolve_class_conflicts_vectorized(
            element_ids, class_ids
        )
```

with:

```python
        element_ids = np.concatenate(all_element_ids).astype(np.int64, copy=False)
        class_ids = np.concatenate(all_class_ids).astype(np.int64, copy=False)
        vote_counts = np.concatenate(all_vote_counts).astype(np.int64, copy=False)

        unique_elements, winning_classes = resolve_class_conflicts_vectorized(
            element_ids, class_ids, weights=vote_counts
        )
```

**Verify:** py_compile PE, then (skip gracefully if PyQt5 import fails in this env):

```
python -c "from coralnet_toolbox.MVAT.managers.PropagationEngine import resolve_class_conflicts_vectorized, reduce_vote_arrays; import numpy as np; e=np.array([7,7,7,9]); c=np.array([1,1,2,3]); re_,rc_,rw_=reduce_vote_arrays(e,c); we,wc=resolve_class_conflicts_vectorized(re_,rc_,weights=rw_); ue,uc=resolve_class_conflicts_vectorized(e,c); print(we.tolist()==ue.tolist()==[7,9] and wc.tolist()==uc.tolist()==[1,3])"
```

Expected output: `True`.

**Commit:** `perf(MVAT): reduce per-camera votes to weighted pairs in cameras-to-mesh`

---

## Step 5 — Mesh→Cameras: LUT at native index-map resolution (PE)

**Why:** `_compute_mesh_to_camera_repaint_tasks` upscales the full int32 index map
through float32 cv2.resize per camera (~128MB of transients on a 16MP camera), then
remaps classes via a full-frame copy plus one equality scan per class.

**Edit 5.1** — In PE `_compute_mesh_to_camera_repaint_tasks`, replace the anchor:

```python
                mask_data = mask_annotation.mask_data
                im_h, im_w = index_map.shape
                mask_h, mask_w = mask_data.shape

                # Upscale index_map to full mask resolution if it was downscaled
                if im_h != mask_h or im_w != mask_w:
                    index_map_full = cv2.resize(
                        index_map.astype(np.float32),
                        (mask_w, mask_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)
                else:
                    index_map_full = index_map

                # Vectorised LUT: pixel → face_id → mesh class_id
                valid = (index_map_full >= 0) & (index_map_full < len(mesh_class_ids))
                face_ids_at_pixels = index_map_full[valid].astype(np.int64)
                pixel_mesh_classes = mesh_class_ids[face_ids_at_pixels].astype(np.int32)

                new_class_layer = np.zeros(mask_data.shape, dtype=mask_data.dtype)
                new_class_layer.flat[np.flatnonzero(valid)] = pixel_mesh_classes
```

with:

```python
                mask_data = mask_annotation.mask_data
                im_h, im_w = index_map.shape
                mask_h, mask_w = mask_data.shape

                # Vectorised LUT at the index map's NATIVE resolution
                # (pixel → face_id → mesh class_id), then upscale the small
                # class layer. Far cheaper than upscaling the int32 index map.
                valid_small = (index_map >= 0) & (index_map < len(mesh_class_ids))
                class_layer_small = np.zeros(index_map.shape, dtype=mask_data.dtype)
                class_layer_small[valid_small] = mesh_class_ids[
                    index_map[valid_small].astype(np.int64)
                ].astype(mask_data.dtype)

                if im_h != mask_h or im_w != mask_w:
                    new_class_layer = cv2.resize(
                        class_layer_small,
                        (mask_w, mask_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    valid = cv2.resize(
                        valid_small.astype(np.uint8),
                        (mask_w, mask_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    new_class_layer = class_layer_small
                    valid = valid_small
```

**Edit 5.2** — In the same method, replace the anchor:

```python
                # Apply ALL remappings atomically using a clean copy
                if canon_to_target_lut:
                    final_class_layer = new_class_layer.copy()
                    for canon_id, target_id in canon_to_target_lut.items():
                        final_class_layer[new_class_layer == canon_id] = target_id
                    mask_data[write_mask] = final_class_layer[write_mask]
                else:
                    mask_data[write_mask] = new_class_layer[write_mask]
```

with:

```python
                # Apply ALL remappings atomically with an integer LUT — one
                # gather instead of a full-frame copy + per-class equality scans.
                written_values = new_class_layer[write_mask]
                if canon_to_target_lut:
                    lut_size = max(256, int(written_values.max()) + 1)
                    remap = np.arange(lut_size, dtype=np.int64)
                    for canon_id, target_id in canon_to_target_lut.items():
                        if 0 <= canon_id < lut_size:
                            remap[canon_id] = target_id
                    mask_data[write_mask] = remap[written_values].astype(mask_data.dtype)
                else:
                    mask_data[write_mask] = written_values
```

**Verify:** py_compile PE. `grep -n "index_map_full" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
returns nothing.

**Commit:** `perf(MVAT): mesh-to-cameras projection works at index-map resolution`

---

## Step 6 — `update_mask_at_indices`: flat color writes, bbox only for the Qt rect (MASK)

**Why:** the "bbox_diff" render path recomputes colors for the whole bounding
rectangle of the touched indices. Propagated pixels are often sparse but spread
across the frame, so a few thousand pixels can trigger a near-full-frame recolor.
Writing colors flat is O(changed pixels) always; the bbox is only needed to limit
the Qt repaint region.

**Edit 6.1** — In MASK `update_mask_at_indices`, replace the anchor (the whole render
section, from `# Determine the actual rendering path` through the end of the
`elif "bbox" in render_path:` block):

```python
        # Determine the actual rendering path for the hybrid method
        render_path = method
        if render_path == "hybrid_diff":
            if pixels_updated < 250000:
                render_path = "bbox_diff"
            else:
                render_path = "flat_diff"

        # 2. Update the visual canvas
        if "flat" in render_path:
            # --- Direct 1D Canvas Repaint ---
            color_map = self._get_color_map()
            colored_flat = self.colored_mask.reshape(-1, 4)
            colored_flat[target_indices] = color_map[class_id]

            if self.graphics_item is not None and not silent:
                self.graphics_item.update()
                
        elif "bbox" in render_path:
            # --- Localized Canvas Repaint ---
            y_coords, x_coords = np.divmod(target_indices, width)
            
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            update_rect = (
                max(0, x_min - 1), 
                max(0, y_min - 1), 
                min(width, x_max + 2), 
                min(height, y_max + 2)
            )
            
            self._update_canvas_slice(update_rect)
            
            if self.graphics_item is not None and not silent:
                qt_rect = QRectF(update_rect[0], 
                                 update_rect[1], 
                                 update_rect[2] - update_rect[0], 
                                 update_rect[3] - update_rect[1])
                self.graphics_item.update(qt_rect)
```

Replacement:

```python
        # 2. Update the visual canvas — always write colors flat (O(changed
        # pixels), never a rectangular slice recompute). The bbox is computed
        # only to limit the Qt repaint region when the touch count is small.
        color_map = self._get_color_map()
        colored_flat = self.colored_mask.reshape(-1, 4)
        colored_flat[target_indices] = color_map[class_id]

        if self.graphics_item is not None and not silent:
            if pixels_updated < 250000:
                y_coords, x_coords = np.divmod(target_indices, width)
                qt_rect = QRectF(max(0, int(x_coords.min()) - 1),
                                 max(0, int(y_coords.min()) - 1),
                                 int(x_coords.max()) - int(x_coords.min()) + 3,
                                 int(y_coords.max()) - int(y_coords.min()) + 3)
                self.graphics_item.update(qt_rect)
            else:
                self.graphics_item.update()
```

Note: the `method` parameter and the earlier `flat_raw`/`bbox_raw` lock-check branch
stay exactly as they are. Only the render section is replaced.

**Verify:** py_compile MASK. `grep -n "render_path" coralnet_toolbox/Annotations/QtMaskAnnotation.py`
returns nothing.

**Commit:** `perf: flat color writes in update_mask_at_indices, bbox only for Qt rect`

---

## Step 7 — Move densification off the main thread (PE)

**Why:** every brush/fill/erase/SAM stroke and every semantic propagation runs
`_densify_source_ids` / `_densify_source_votes` (KD-tree radius queries + normal
filtering) on the **main thread** before queuing the background job. Moving it into
`_do_universal_propagation` removes that latency from the paint cursor.

**Edit 7.1** — Add a `densify` parameter to `_execute_mask_propagation`. Replace the
anchor:

```python
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Queue a propagation job onto the single unified background worker."""
```

with:

```python
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False,
                                  densify: bool = False):
        """Queue a propagation job onto the single unified background worker."""
```

**Edit 7.2** — In the same method, replace the submit anchor:

```python
                primary_target,
                payload,
                skip_3d_paint,
            )
```

with:

```python
                primary_target,
                payload,
                skip_3d_paint,
                densify,
            )
```

**Edit 7.3** — Add the parameter to `_do_universal_propagation`. Replace the anchor:

```python
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Background worker for brush, SAM, and semantic mask propagation."""
```

with:

```python
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False,
                                  densify: bool = False):
        """Background worker for brush, SAM, and semantic mask propagation."""
```

**Edit 7.4** — In `_do_universal_propagation`, replace the anchor:

```python
            winning_elements = np.array([], dtype=np.int64)
            winning_classes = np.array([], dtype=np.int64)
            if element_ids is not None and class_ids is not None:
```

with:

```python
            # Densify here (background thread) so the paint cursor never waits
            # on KD-tree gathers. _densify_source_votes is a no-op when
            # densify_enabled is off or the target has no KD-tree.
            if densify and element_ids is not None and element_ids.size:
                element_ids, class_ids = self._densify_source_votes(
                    source_camera, element_ids, class_ids
                )

            winning_elements = np.array([], dtype=np.int64)
            winning_classes = np.array([], dtype=np.int64)
            if element_ids is not None and class_ids is not None:
```

**Edit 7.5** — Remove the four main-thread densify calls and opt in to worker
densification instead:

a) In `_on_brush_stroke_applied`, delete the line
   `painted_ids = self._densify_source_ids(self.selected_camera, painted_ids)`
   and add `densify=True,` as the last argument of its `self._execute_mask_propagation(`
   call (after the `fallback_payload={...},` argument).

b) In `_on_fill_stroke_applied`, delete the line
   `painted_ids = self._densify_source_ids(self.selected_camera, painted_ids)`
   and add `densify=True,` to its `_execute_mask_propagation` call.

c) In `_on_erase_stroke_applied`, delete the line
   `painted_ids = self._densify_source_ids(self.selected_camera, painted_ids)`
   (KEEP the explanatory comment above it: move the comment so it sits above the
   `_execute_mask_propagation` call) and add `densify=True,` to the call.

d) In `_on_sam_prediction_applied`, delete the line
   `painted_ids = self._densify_source_ids(selected_camera, painted_ids)`
   and add `densify=True,` to its `_execute_mask_propagation` call.

**Edit 7.6** — In `_on_semantic_prediction_applied`, replace the anchor:

```python
        # Densify per class so low-res prediction votes fill the dense surface.
        element_ids, class_ids = self._densify_source_votes(
            source_camera, element_ids, class_ids
        )

        _status(
```

with:

```python
        # Densification now happens inside the background worker (densify=True
        # below), so the element count shown here is the pre-densify seed count.
        _status(
```

and replace its `_execute_mask_propagation` call anchor:

```python
            class_label_ids=class_label_ids,
        )
```

with:

```python
            class_label_ids=class_label_ids,
            densify=True,
        )
```

(There are several `class_label_ids=...` call sites — this exact anchor with
`class_label_ids=class_label_ids,` immediately followed by `)` on the next line is
unique to `_on_semantic_prediction_applied`. Confirm you are inside that method.)

**Important:** do NOT add `densify=True` to `_propagate_3d_face_ids_to_context_cameras`
or `_on_viewer_sam_accepted` — those element IDs are already dense and densifying
again would expand the stroke.

**Verify:** py_compile PE. Then:
- `grep -n "_densify_source_ids" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
  must show ONLY: the method definition, the call inside `_densify_source_votes`,
  and nothing inside the four stroke handlers.
- `grep -cn "densify=True" coralnet_toolbox/MVAT/managers/PropagationEngine.py` → 5.

**Commit:** `perf(MVAT): run stroke densification in the propagation worker`

---

## Step 8 — Parallel per-camera fan-out in the propagation worker (PE)

**Why:** `_do_universal_propagation` iterates target cameras serially on the single
`_unified_bg_executor` thread, while the engine's 8-worker `_propagation_executor`
sits idle. Per-camera work touches only that camera's raster and mask, so it
parallelizes safely. Numpy/cv2 release the GIL, so this is a near-linear win for
"to all cameras" flows.

**Edit 8.1** — Make the buffer pool thread-safe. In PE imports, after the line
`import traceback`, add:

```python
import threading
```

In `__init__`, after the line `self._propagation_buffer_pool = {}`, add:

```python
        self._propagation_buffer_pool_lock = threading.Lock()
```

Replace the body of `_acquire_propagation_buffer`:

```python
        key = (tuple(shape), np.dtype(dtype).str)
        pool = self._propagation_buffer_pool.get(key)
        if pool:
            return pool.pop()
        return np.empty(shape, dtype=dtype)
```

with:

```python
        key = (tuple(shape), np.dtype(dtype).str)
        with self._propagation_buffer_pool_lock:
            pool = self._propagation_buffer_pool.get(key)
            if pool:
                return pool.pop()
        return np.empty(shape, dtype=dtype)
```

Replace in `_release_propagation_buffer`:

```python
        key = (tuple(buffer.shape), np.dtype(buffer.dtype).str)
        self._propagation_buffer_pool.setdefault(key, []).append(buffer)
```

with:

```python
        key = (tuple(buffer.shape), np.dtype(buffer.dtype).str)
        with self._propagation_buffer_pool_lock:
            self._propagation_buffer_pool.setdefault(key, []).append(buffer)
```

**Edit 8.2** — In `_do_universal_propagation`, convert the camera loop into a worker
function plus a pool fan-out. This is a structural edit; follow exactly:

1. Find the line `for target_path in target_paths:` (directly after the
   `_project_bbox_for_subset` inner function ends).
2. Replace that line with:

```python
            def _process_target(target_path):
                local_tasks = []
                local_mask_time = 0.0
```

3. Indent the ENTIRE former loop body (everything from `target_camera = self._get_camera_for_path(target_path)`
   down to and including the final `repaint_tasks.append({...})` block that starts
   with `if target_rect is not None:`) by one additional level (4 spaces), so it
   becomes the body of `_process_target`.
4. Inside that body make exactly these substitutions:
   - every `continue` → `return local_tasks, local_mask_time`
   - `mask_time += perf_counter() - t_mask_start` (occurs twice) →
     `local_mask_time += perf_counter() - t_mask_start`
   - the trailing block

     ```python
                 if target_rect is not None:
                     repaint_tasks.append({
                         'type': 'repaint',
                         'path': target_path,
                         'mask': target_mask,
                         'label_ids': tuple(sorted(target_label_ids)),
                         'update_rect': target_rect,
                     })
     ```

     becomes

     ```python
                 if target_rect is not None:
                     local_tasks.append({
                         'type': 'repaint',
                         'path': target_path,
                         'mask': target_mask,
                         'label_ids': tuple(sorted(target_label_ids)),
                         'update_rect': target_rect,
                     })
                 return local_tasks, local_mask_time
     ```

   - NOTE: the two `continue` statements inside the inner
     `for source_class_id, subset_elements in class_to_elements.items():` loop and
     the `continue` inside the fallback block that skips one target
     (`else:\n                                continue` after the projections check)
     stay as `continue` ONLY if they continue an inner `for` loop. Rule: a
     `continue` that belonged to `for target_path in target_paths:` becomes
     `return local_tasks, local_mask_time`; a `continue` inside any inner loop is
     untouched. The former are exactly the early-skip guards at the top
     (`if target_camera is None:`, `if target_raster is None:`, the two
     `if target_mask is None:` outcomes) — there are 3 such `continue`s plus one
     special case: inside the fallback block, the statement
     `else:\n                                continue` exits the *target*, so it
     becomes `return local_tasks, local_mask_time` as well.
5. Immediately after the new `_process_target` function body, add the dispatcher
   (at the same indentation as `def _process_target`):

```python
            def _process_target_safe(target_path):
                try:
                    return _process_target(target_path)
                except Exception:
                    traceback.print_exc()
                    return [], 0.0

            if len(target_paths) > 1:
                futures = [
                    self._propagation_executor.submit(_process_target_safe, p)
                    for p in target_paths
                ]
                for fut in as_completed(futures):
                    local_tasks, local_mask_time = fut.result()
                    repaint_tasks.extend(local_tasks)
                    mask_time += local_mask_time
            else:
                for p in target_paths:
                    local_tasks, local_mask_time = _process_target_safe(p)
                    repaint_tasks.extend(local_tasks)
                    mask_time += local_mask_time
```

(`as_completed` is already imported at the top of the file.)

**Verify:** py_compile PE. Read back the full `_do_universal_propagation` and confirm:
no stray `continue` at `_process_target`'s top level, `_process_target` ends with
`return local_tasks, local_mask_time`, and the dispatcher runs after the function
definitions. Then `grep -n "repaint_tasks.append" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
inside `_do_universal_propagation` should only match the 3d_paint block (before the
camera fan-out).

**Commit:** `perf(MVAT): fan per-camera propagation across the worker pool`

---

## Step 9 — Prewarm CSR inverted indexes when multi-annotate turns on (PE)

**Why:** the CSR index builds lazily on the first pixel query, so the *first* stroke
after enabling multi-annotate pays an O(P log P) build per visible camera.

**Edit 9.1** — In PE `_on_multi_annotate_toggled`, inside the `if enabled:` branch,
replace the anchor:

```python
                for path in target_paths:
                    raster = self.raster_manager.get_raster(path)
                    if raster and raster.mask_annotation is None:
                        raster.get_mask_annotation(project_labels)

            except Exception:
                pass
```

with:

```python
                for path in target_paths:
                    raster = self.raster_manager.get_raster(path)
                    if raster and raster.mask_annotation is None:
                        raster.get_mask_annotation(project_labels)

                # Prewarm CSR inverted indexes off the main thread so the FIRST
                # stroke doesn't pay the per-camera build cost. ensure_inverted_index
                # is idempotent; concurrent duplicate builds produce identical
                # arrays, so a race is wasteful but harmless.
                for path in target_paths:
                    raster = self.raster_manager.get_raster(path)
                    if raster is not None and getattr(raster, 'ensure_inverted_index', None):
                        self._propagation_executor.submit(raster.ensure_inverted_index)

            except Exception:
                pass
```

**Verify:** py_compile PE.

**Commit:** `perf(MVAT): prewarm CSR inverted indexes on multi-annotate enable`

---

## Step 10 — Batch-commit propagated patch annotations (PE)

**Why:** `_on_patch_annotation_created` commits each propagated patch with an
individual `add_annotation(record_action=True)` — one undo entry and one UI update
per target camera, all on the main thread.

**Edit 10.1** — In PE `_on_patch_annotation_created`, after the line
`self._propagating_annotation = True` and before `try:`... actually the structure is
`self._propagating_annotation = True` then `try:`. Inside the `try:` block, after the
comment block ending `# More IDs dramatically reduces stride false-negatives in the target cameras.`
and before `source_raster = getattr(self.selected_camera, '_raster', None)`, add:

```python
            propagated_annotations = []
```

**Edit 10.2** — Replace the 3D-path commit anchor:

```python
                                try:
                                    self.annotation_window.add_annotation(new_annotation, record_action=True)
                                    placed = True
                                except Exception:
                                    pass
```

with:

```python
                                propagated_annotations.append(new_annotation)
                                placed = True
```

**Edit 10.3** — Replace the 2D-fallback commit anchor:

```python
                        try:
                            self.annotation_window.add_annotation(new_annotation, record_action=True)
                        except Exception:
                            pass
```

with:

```python
                        propagated_annotations.append(new_annotation)
```

**Edit 10.4** — Replace the closing anchor:

```python
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False
```

with:

```python
                except Exception:
                    pass

            # One batched commit: a single undo entry and a single UI refresh
            # pass instead of one per target camera.
            if propagated_annotations:
                try:
                    self.annotation_window.add_annotations(propagated_annotations, record_action=True)
                except Exception:
                    for ann in propagated_annotations:
                        try:
                            self.annotation_window.add_annotation(ann, record_action=True)
                        except Exception:
                            pass
        finally:
            self._propagating_annotation = False
```

**Verify:** py_compile PE. `grep -n "add_annotation(new_annotation" coralnet_toolbox/MVAT/managers/PropagationEngine.py`
returns nothing.

**Commit:** `perf(MVAT): batch-commit propagated patch annotations`

---

## Step 11 — Time-budgeted repaint queue on the main thread (PE)

**Why:** `_on_universal_repaint` applies every repaint task in one event-loop
callback. Mesh→cameras jobs produce near-full-frame canvas recolors for every
visible canvas — hundreds of ms of main-thread stall in a single tick. Draining the
queue in ~12ms slices keeps the UI at interactive framerates.

**Edit 11.1** — In PE imports, change:

```python
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPointF
```

to:

```python
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPointF, QTimer
```

**Edit 11.2** — In `__init__`, after the line
`self._propagation_buffer_pool_lock = threading.Lock()` (added in Step 8), add:

```python
        # Chunked repaint queue: tasks drain in time-budgeted slices so a huge
        # propagation never freezes the UI inside a single callback.
        self._repaint_task_queue = deque()
        self._repaint_drain_scheduled = False
        self._repaint_needs_3d_flush = False
```

**Edit 11.3** — Restructure `_on_universal_repaint`. The current method is one big
`try/except/finally`. Replace the ENTIRE method (from `def _on_universal_repaint`
down to its final line `self._end_semantic_propagation_busy()`) with the following
four methods. The per-task bodies are copied verbatim from the existing code —
only the control flow changes (`continue` → `return`, `needs_3d_flush = True` →
`self._repaint_needs_3d_flush = True`):

```python
    def _on_universal_repaint(self, repaint_tasks: list):
        """Queue UI updates from a propagation worker; drain in budgeted slices."""
        self._repaint_task_queue.extend(repaint_tasks)
        # Sentinel marks the end of one worker job — completion bookkeeping
        # (pending counter, busy cursor) runs when the sentinel drains.
        self._repaint_task_queue.append({'type': '_job_done'})
        self._schedule_repaint_drain()

    def _schedule_repaint_drain(self):
        if self._repaint_drain_scheduled:
            return
        self._repaint_drain_scheduled = True
        QTimer.singleShot(0, self._drain_repaint_queue)

    def _drain_repaint_queue(self):
        self._repaint_drain_scheduled = False
        budget_s = 0.012  # stay under one frame
        start = perf_counter()
        while self._repaint_task_queue:
            task = self._repaint_task_queue.popleft()
            try:
                if task.get('type') == '_job_done':
                    self._finish_repaint_job()
                else:
                    self._apply_repaint_task(task)
            except Exception as e:
                print(f"Error in _drain_repaint_queue: {e}")
                traceback.print_exc()
            if self._repaint_task_queue and (perf_counter() - start) > budget_s:
                self._schedule_repaint_drain()
                return

    def _finish_repaint_job(self):
        """Completion bookkeeping for one drained propagation job."""
        self._pending_unified_propagation_jobs = max(
            0,
            self._pending_unified_propagation_jobs - 1,
        )
        self._propagating_annotation = self._pending_unified_propagation_jobs > 0
        if self._repaint_needs_3d_flush:
            self._repaint_needs_3d_flush = False
            request_flush = getattr(self, 'request_lazy_flush', None)
            if callable(request_flush):
                request_flush()
        # Clear the matrix busy state (cursor + propagate button) once the
        # last queued unified-repaint job has been applied.
        if self._pending_unified_propagation_jobs <= 0:
            if self.context_matrix is not None:
                set_busy = getattr(self.context_matrix, 'set_propagation_busy', None)
                if callable(set_busy):
                    try:
                        set_busy(False)
                    except Exception:
                        pass
            self._end_semantic_propagation_busy()

    def _apply_repaint_task(self, task: dict):
        """Apply one repaint/3d_paint/status task. Main thread only."""
        task_type = task.get('type')

        if task_type == 'status_message':
            msg = task.get('message', '')
            timeout = task.get('timeout', 5000)
            if msg:
                status_bar = getattr(self.main_window, 'status_bar', None)
                if status_bar is not None:
                    try:
                        status_bar.showMessage(msg, timeout)
                    except Exception:
                        pass
            return

        if task_type == 'reload_annotation_window':
            try:
                aw = getattr(self.main_window, 'annotation_window', None)
                if aw is not None and hasattr(aw, 'load_mask_annotation'):
                    aw.load_mask_annotation()
            except Exception:
                pass
            return

        if task_type == 'update_image_table':
            paths = task.get('paths', ())
            iw = getattr(self.main_window, 'image_window', None)
            if iw is not None:
                for _p in paths:
                    try:
                        iw.update_image_annotations(_p)
                    except Exception:
                        pass
            return

        if task_type == '3d_paint':
            label_id = task.get('label_id')
            tgt = task.get('primary_target')
            is_point = False
            try:
                is_point = tgt is not None and tgt.get_element_type() == 'point'
            except Exception:
                is_point = False
            if is_point and hasattr(self, 'submit_3d_point_paint'):
                self.submit_3d_point_paint(
                    task['painted_ids'],
                    task['target_color'],
                    task['source_class_id'],
                    primary_target=tgt,
                    label_id=label_id,
                )
            else:
                self.submit_3d_face_paint(
                    task['painted_ids'],
                    task['target_color'],
                    task['source_class_id'],
                    primary_target=tgt,
                    label_id=label_id,
                )
            self._repaint_needs_3d_flush = True
            return

        if task_type != 'repaint':
            return

        target_mask = task.get('mask')
        if target_mask is None:
            return

        for label_id in task.get('label_ids', ()):
            if label_id is not None and label_id not in target_mask.visible_label_ids:
                target_mask.visible_label_ids.add(label_id)

        target_path = task.get('path')
        context_canvas = self._get_context_canvas_for_path(target_path)
        should_update_now = (
            context_canvas is not None
            and self.context_matrix is not None
            and self.context_matrix.is_canvas_on_screen(context_canvas)
        )

        if should_update_now:
            context_canvas.set_mask_overlay(target_mask)
            target_mask.update_graphics_item(update_rect=task.get('update_rect'))
            overlay_item = getattr(context_canvas, '_mask_overlay_item', None)
            if overlay_item is not None:
                try:
                    overlay_item.update()
                except Exception:
                    pass
        elif self.context_matrix is not None:
            self.context_matrix.queue_pending_repaint(
                target_path,
                target_mask,
                update_rect=task.get('update_rect'),
                label_ids=task.get('label_ids', ()),
            )
```

NOTE: the longer explanatory comments in the original repaint branch may be dropped,
but the BEHAVIOR above mirrors the original exactly. Keep all of it.

**Verify:** py_compile PE. Confirm with grep that exactly one definition each of
`_on_universal_repaint`, `_drain_repaint_queue`, `_finish_repaint_job`,
`_apply_repaint_task` exists, and that `needs_3d_flush` (the old local variable) no
longer appears: `grep -n "needs_3d_flush = " coralnet_toolbox/MVAT/managers/PropagationEngine.py`
shows only `self._repaint_needs_3d_flush` assignments.

**Commit:** `perf(MVAT): drain propagation repaints in time-budgeted slices`

---

## Step 12 — Lazy RGBA canvas allocation for MaskAnnotation (MASK + 2 call sites)

**HIGHEST RISK STEP — follow the checklist completely.**

**Why:** `MaskAnnotation.__init__` eagerly allocates the full RGBA `colored_mask` +
QImage (4 bytes/px, ~64MB for a 16MP camera) even for masks that exist only as
background propagation targets. "Mesh → All Cameras" pre-creates masks for every
camera on the main thread: 100 cameras ≈ 6–8GB RAM plus a long UI freeze. The
canvas is only needed when a mask is *displayed*; `_initialize_canvas` already
rebuilds colors from `mask_data`, so deferring it loses nothing.

The invariant after this step: **`colored_mask`/`qimage` may be `None` until first
display.** Every write path skips color updates while `None`; every display path
calls `_ensure_canvas()` first (which builds colors from the already-written
`mask_data`).

**Edit 12.1** — In MASK `__init__`, replace:

```python
        self.colored_mask = None
        self.qimage = None
        self._initialize_canvas()
```

with:

```python
        # Lazy canvas: colored_mask/qimage are allocated on first display via
        # _ensure_canvas(). Masks that exist only as background propagation
        # targets never pay the RGBA cost; _initialize_canvas builds colors
        # from mask_data, which already reflects every silent write.
        self.colored_mask = None
        self.qimage = None
```

**Edit 12.2** — Add directly after the `_initialize_canvas` method:

```python
    def _ensure_canvas(self):
        """Allocate colored_mask/qimage on first display (no-op when present)."""
        if self.colored_mask is None:
            self._initialize_canvas()
```

**Edit 12.3** — In `_update_full_canvas`, insert as the first statements of the body:

```python
        if self.colored_mask is None:
            self._initialize_canvas()
            return
```

**Edit 12.4** — In `_update_canvas_slice`, insert as the first statements of the body:

```python
        if self.colored_mask is None:
            self._initialize_canvas()
            return
```

**Edit 12.5** — In `refresh_graphics`, after the existing early-return guards
(`except RuntimeError: return`) and before `height, width = self.mask_data.shape`,
insert:

```python
        self._ensure_canvas()
```

**Edit 12.6** — In `update_graphics_item`, insert at the very top of the body (before
`height, width = self.mask_data.shape`):

```python
        if self.colored_mask is None:
            # First display request: build the full canvas from mask_data (it
            # already includes every silent write made before now).
            self._initialize_canvas()
            if self.graphics_item is not None:
                self.graphics_item.update()
            return
```

**Edit 12.7** — In `apply_flat_values_at_indices`, replace:

```python
        color_map = self._get_color_map()
        colored_flat = self.colored_mask.reshape(-1, 4)
        colored_flat[target_indices] = color_map[new_values]
```

with:

```python
        if self.colored_mask is not None:
            color_map = self._get_color_map()
            colored_flat = self.colored_mask.reshape(-1, 4)
            colored_flat[target_indices] = color_map[new_values]
```

**Edit 12.8** — In `update_mask` (new-method branch), replace:

```python
            # 2. Update visual canvas directly (bypassing _update_canvas_slice memory allocation)
            color_map = self._get_color_map()
            target_colored_slice = self.colored_mask[clipped_y_start:y_end, clipped_x_start:x_end]
            target_colored_slice[pixels_to_change] = color_map[new_class_id]
```

with:

```python
            # 2. Update visual canvas directly (bypassing _update_canvas_slice memory allocation)
            if self.colored_mask is not None:
                color_map = self._get_color_map()
                target_colored_slice = self.colored_mask[clipped_y_start:y_end, clipped_x_start:x_end]
                target_colored_slice[pixels_to_change] = color_map[new_class_id]
```

**Edit 12.9** — In `update_mask_at_indices` (as rewritten by Step 6), replace:

```python
        color_map = self._get_color_map()
        colored_flat = self.colored_mask.reshape(-1, 4)
        colored_flat[target_indices] = color_map[class_id]
```

with:

```python
        if self.colored_mask is not None:
            color_map = self._get_color_map()
            colored_flat = self.colored_mask.reshape(-1, 4)
            colored_flat[target_indices] = color_map[class_id]
```

(`graphics_item` being non-None implies the canvas exists — MaskGraphicsItem ensures
it, Edit 12.11 — so the Qt-repaint block below it needs no extra guard.)

**Edit 12.10** — In `update_mask_with_mask` (new-method branch), replace:

```python
            # 2. Update visual canvas directly
            color_map = self._get_color_map()
            target_colored_slice = self.colored_mask[y_start:y_end, x_start:x_end]
            new_class_ids = subset_slice[pixels_to_change]
            target_colored_slice[pixels_to_change] = color_map[new_class_ids]
```

with:

```python
            # 2. Update visual canvas directly
            if self.colored_mask is not None:
                color_map = self._get_color_map()
                target_colored_slice = self.colored_mask[y_start:y_end, x_start:x_end]
                new_class_ids = subset_slice[pixels_to_change]
                target_colored_slice[pixels_to_change] = color_map[new_class_ids]
```

**Edit 12.11** — `MaskGraphicsItem` (top of the same file, the class whose `paint`
checks `if self.mask_annotation.qimage:`): in its `__init__`, immediately after the
line that assigns `self.mask_annotation = ...`, add:

```python
        # Displaying a mask requires its canvas; build it now if it was lazy.
        try:
            mask_annotation._ensure_canvas()
        except Exception:
            pass
```

(Read the actual `__init__` first; the parameter may be named `mask_annotation` —
use whatever local name holds the annotation.)

**Edit 12.12** — Find the `self.qimage.save(path)` call near line 1606 of MASK
(an export helper). Insert `self._ensure_canvas()` on its own line directly before
the statement (matching its indentation).

**Edit 12.13** — External caller check (no code change expected, verify only):

- `coralnet_toolbox/QtAnnotationWindow.py` (~line 3648) calls `ma._update_full_canvas()`
  then reads `ma.colored_mask.data`. With Edit 12.3, `_update_full_canvas` on a lazy
  mask initializes the canvas, so `colored_mask` is non-None afterward. ✔ no change.
- `coralnet_toolbox/MachineLearning/BatchInference/QtBatchInference.py` (~line 1700)
  reads `tmp_mask.qimage` with a None-guard. To preserve pre-change behavior, insert
  this line directly above the statement that contains
  `qimg_copy = tmp_mask.qimage.copy() if tmp_mask.qimage is not None else None`:

```python
                                        tmp_mask._ensure_canvas()
```

  (match the surrounding indentation exactly).

**Verify:**
- py_compile MASK, `coralnet_toolbox/QtAnnotationWindow.py`,
  `coralnet_toolbox/MachineLearning/BatchInference/QtBatchInference.py`.
- `grep -n "self.colored_mask" coralnet_toolbox/Annotations/QtMaskAnnotation.py` —
  every *read/write* site must be inside a method that either guards
  `if self.colored_mask is not None:` or runs after `_ensure_canvas()` /
  `_initialize_canvas()`. Walk each hit and confirm.
- `grep -rn "\.colored_mask\|\.qimage" coralnet_toolbox --include=*.py | grep -v QtMaskAnnotation.py | grep -v rasterio_to_qimage | grep -v _bgr_to_qimage | grep -v _rgb_to_qimage | grep -v mask_qimage | grep -v get_qimage`
  — review every remaining hit and confirm it is either None-guarded or preceded by
  `_ensure_canvas()`/`_update_full_canvas()`.

**Commit:** `perf: allocate MaskAnnotation RGBA canvas lazily on first display`

---

## Step 13 — Cleanup: dead code + debug print (PE)

**Edit 13.1** — Delete the two unused methods `_dense_mesh_hit_test` and
`_propagate_to_camera` (they are contiguous: from the line
`def _dense_mesh_hit_test(self, source_camera, pixel_mask: np.ndarray, px: int, py: int, mesh_product) -> np.ndarray:`
down to the line directly above
`def _resolve_source_mask_class_context(self, source_camera, label_id: str, project_labels: list):`).
First confirm they are unreferenced:

```
grep -rn "_dense_mesh_hit_test\|_propagate_to_camera\b" coralnet_toolbox --include=*.py
```

must show only the definitions. Then delete.

**Edit 13.2** — Gate the per-stroke console print. Replace the anchor:

```python
            print(
                f"DEBUG [Sync Worker]: {len(target_paths)} Cams | Total: {(perf_counter() - t0) * 1000:.2f}ms | "
                f"Mask Gen: {mask_time * 1000:.2f}ms"
            )
```

with:

```python
            if os.environ.get('MVAT_DEBUG_TIMING'):
                print(
                    f"DEBUG [Sync Worker]: {len(target_paths)} Cams | "
                    f"Total: {(perf_counter() - t0) * 1000:.2f}ms | "
                    f"Mask Gen: {mask_time * 1000:.2f}ms"
                )
```

(`os` is already imported in PE.)

**Verify:** py_compile PE; rerun the grep from 13.1 (no hits at all now).

**Commit:** `chore(MVAT): remove dead propagation helpers, gate debug timing print`

---

## Final acceptance pass (manual, requires the GUI)

These cannot be automated by the agent; report them as a checklist for the user:

1. Enable Multi-Annotate, paint with brush/fill/erase on a perspective camera —
   strokes propagate to visible context cameras and the ortho; cursor stays fluid.
2. SAM prediction with Multi-Annotate on — mask propagates.
3. Propagation hub: run each mode once (active→context, active→all,
   active-camera→mesh, cameras→mesh, mesh→active, mesh→visible, mesh→all).
   With `MVAT_DEBUG_TIMING=1`, compare the Sync Worker totals before/after.
4. Mesh→All Cameras on a large project: watch RAM (should no longer jump ~80MB per
   never-viewed camera) and confirm the UI stays interactive while repaints stream in.
5. Undo/redo after a propagated patch annotation (single undo entry now).
6. Semantic Deploy → predict with Multi-Annotate on (single image and Batch
   Inference) — propagation still fires per image, busy cursor clears at the end.
