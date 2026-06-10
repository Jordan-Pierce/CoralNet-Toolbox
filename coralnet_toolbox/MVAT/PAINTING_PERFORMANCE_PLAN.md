# MVAT 3D Painting Performance Plan

Status (2026-06-09): **Tier 1 and Tier 2 implemented. Tier 3 is future work.**

## Problem

Painting very large meshes / point clouds is smooth during the stroke, but there is a
noticeable pause when "finalizing" (the lazy flush ~1s after mouse release), and
per-stroke cost grows with stroke length. Root causes found by profiling the pipeline
(`BrushTool3D._apply_brush` → `MVATManager.submit_3d_face_paint` → `LabelWorker` →
`_on_overlay_ready` → debounced `_execute_lazy_flush` → `flush_labels_to_gpu`):

1. **The finalize pause is a full VTK mapper rebuild, not a numpy copy.**
   `mesh.cell_data["Labels"] = labels_cache` + `Modified()` bumps the dataset MTime.
   On the next render, VTK's OpenGL mapper rebuilds **all** of its buffer objects:
   it re-runs the cell→primitive conversion and re-uploads positions, normals, the
   index buffer, *and* colors for the entire mesh. Stock VTK mappers have no
   partial-update API, so the only fixes are changing *when* this happens (Tier 2)
   or *how* label colors reach the GPU (Tier 3).

2. **Per-stroke costs grew with stroke length.**
   - `BrushTool3D._apply_brush` ran `np.setdiff1d` + `np.union1d` against the whole
     accumulated stroke on the **main thread, every mouse move** — O(S log S).
   - `LabelWorker` ran a second `union1d` per chunk, and rebuilt the overlay
     PolyData for the entire stroke on every emit.

3. **The point-cloud path scaled worst.** Every mouse move ran
   `np.flatnonzero(class_ids != 0)` — an O(N) scan of the entire cloud — then
   gathered *every* painted point and recreated the overlay actor from scratch
   (`remove_actor` + `add_mesh`). And the lazy flush rebuilt the full base-cloud
   VBOs every stroke.

---

## Tier 1 — Cheap fixes, no architecture change (IMPLEMENTED)

- **Boolean membership masks instead of `union1d`/`setdiff1d`.**
  `BrushTool3D` keeps a `(N,) bool` stroke mask plus a list of per-move chunks;
  per-move dedupe is O(brush chunk). The mask is reset at stroke end by clearing
  only the touched indices (O(S), not O(N)). `LabelWorker` mirrors this.
- **Stroke-only overlay for point clouds.** During a stroke the manager maintains
  a small "stroke" overlay actor containing only the points painted in the current
  stroke (in-place mapper input swap, throttled render). The O(N) full-painted-set
  rebuild now happens **once per stroke end**, not per move.
- **No flicker at finalize**: the stroke overlay is only removed after the
  committed visual is in place (see Tier 2).

## Tier 2 — Eliminate the routine finalize pause (IMPLEMENTED)

Stop flushing the Labels array into the base actor after every stroke. The painted
state stays visible through a **persistent "committed" overlay actor** that sits on
top of the base mesh (VTK's default depth function is `GL_LEQUAL`, so a later-added
coplanar actor cleanly wins).

- **Meshes**: on stroke end, `LabelWorker` (worker thread) builds a welded PolyData
  payload of *all* painted faces (`class_ids != 0`) and emits it via a new
  `committed_overlay_ready` signal. The main thread swaps it into
  `_committed_overlay_actor` (in-place mapper input swap), then the live stroke
  overlay is cleared. No base-mesh rebuild, no full VBO re-upload.
- **Point clouds**: on stroke end, the existing full painted-points overlay
  (`refresh_primary_point_overlay`) is rebuilt once and the stroke overlay is
  cleared. `flush_labels_to_gpu` is no longer called per stroke.
- **Deferred flush with correctness barriers.** `flush_labels_to_gpu` now runs only:
  - in `QtMVATViewer.render_scene` (scene rebuild / scalar-array switch) — gated by
    a `_labels_dirty` flag on the product so it is free when nothing changed;
  - at stroke end **only when an erase stroke touched a previously-flushed array**
    (`stroke had class_id == 0` AND `product._vtk_labels_have_paint`): without a
    flush, the stale VTK array would show the old color under faces/points that
    dropped out of the committed overlay. Paint-only strokes never hard-flush.
- **Draw order invariants** (this is what makes coplanar overlays correct):
  base actor → committed overlay actor → stroke overlay actor. The stroke actor is
  destroyed at every stroke end and recreated on the next stroke, so it is always
  added after the committed actor. `render_scene` re-adds base actors first and then
  force-recreates the committed overlay, restoring the order.

### Known limitations of Tier 2 (accepted for now)

- The committed overlay duplicates painted geometry (welded, ≈1 vertex per painted
  face). If the user paints a very large fraction of a huge mesh, the stroke-end
  overlay swap upload grows accordingly (still far cheaper than the full base-mesh
  rebuild it replaced). Tier 3 removes this entirely.
- Erase strokes after a flush (e.g. after a scene rebuild) pay the old full-rebuild
  pause once. Paint strokes never do.
- There is a pre-existing (now still present) benign race: `_execute_lazy_flush`
  reads `_labels_cache` on the main thread while `LabelWorker` may still be applying
  queued chunks. The 1s debounce makes this effectively unobservable; Tier 3 or a
  queue-drain handshake would close it.
- A possible future refinement (deliberately **not** implemented): a "compaction"
  threshold that hard-flushes once when the painted fraction exceeds ~30% while
  viewing the Labels array, to cap overlay memory. Revisit if overlay memory becomes
  a problem in practice.

## Tier 3 — Partial GPU updates via a label buffer + shader (FUTURE)

The truly scalable end-state: per-stroke cost becomes O(brush chunk) on the GPU,
independent of mesh size, with **no overlay actors and no flush at all**.

- Maintain label colors in a persistent OpenGL texture buffer / SSBO of N×RGBA owned
  by us (not VTK).
- Inject a fragment-shader replacement on the base mesh mapper
  (`vtkOpenGLPolyDataMapper::AddShaderReplacement` / pyvista shader hooks) that
  samples `labels[gl_PrimitiveID]` and blends it over the base color. Note VTK
  already routes cell-data colors through a `gl_PrimitiveID`-indexed texture buffer
  internally — its only limitation is all-or-nothing rebuild granularity.
- Upload **only dirty face ranges** with `glBufferSubData` from a render-start
  observer. The codebase already does raw-GL injection (GaussianActor's GL observer,
  `shaders/gpu_interop.py` CUDA-GL interop), so the machinery exists.
- Point clouds: identical trick keyed by `gl_VertexID`.
- Everything above the render path stays: `_labels_cache`, `class_ids`, tools,
  LabelWorker RAM semantics, propagation. Only the overlay/flush visual path is
  replaced.
- Risks: VTK-version-specific shader-injection brittleness; needs a fallback to the
  Tier 2 path when the GL context / driver doesn't cooperate.

## File map

| Concern | File |
|---|---|
| Brush stroke accumulation, dedupe | `MVAT/tools/BrushTool3D.py` |
| Worker-thread label application + overlays | `MVAT/workers/LabelWorker.py` |
| Overlay actors, lazy flush, paint submission | `MVAT/managers/MVATManager.py` |
| Label caches, dirty flags, GPU flush | `MVAT/core/Products.py` |
| Scene rebuild flush barrier | `MVAT/ui/QtMVATViewer.py` (`render_scene`) |
