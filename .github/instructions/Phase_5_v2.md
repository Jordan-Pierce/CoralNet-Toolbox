# Phase 5: The "Target-Locked" Sync Engine

## Summary

When the user zooms into a coral head in the main `AnnotationWindow`, all N context canvases automatically zoom and center on that exact same 3D point from their respective camera angles. This is the feature that transforms separate viewports into a unified multi-angle inspection system. It requires translating 2D navigation events in one camera into 3D world coordinates, then re-projecting into each context camera's pixel space — the same math used by the marker system (Phase 4), but applied to viewport control rather than marker placement.

---

## Current Architecture (What Exists After Phases 1-4)

### Navigation Events
- `BaseCanvas` emits `viewNavigated(center_x, center_y, zoom_factor)` after pan/zoom completes.
- `AnnotationWindow` (inheriting BaseCanvas) emits `viewChanged(width, height)` for status bar updates.
- `ZoomTool.wheelEvent()` calls `self.annotation_window.scale(factor, factor)`.
- `PanTool.pan(pos)` adjusts scrollbar values.

### 3D Projection Pipeline (Already Proven in Phase 4)
- `Camera.project(points_3d_world)` → `(u, v)` pixel coordinates.
- `Camera.unproject(pixel_coord)` → `(x, y, z)` world point (requires depth).
- `CameraRay.from_pixel_and_camera(pixel, camera, depth, default_depth)` → ray with 3D terminal point.

### BaseCanvas Viewport Control API (Phase 1)
- `center_on_pixel(x, y)` — centers view on image pixel.
- `set_zoom_level(factor)` — sets absolute zoom level.
- `snap_to_target(target_x, target_y, relative_zoom)` — combined zoom + center.
- `fit_to_image()` — reset to fit whole image.

### Context Matrix (Phase 2-3)
- `ContextMatrixWidget` manages N `BaseCanvas` instances.
- `self.target_lock_enabled` toggle (Phase 2 toolbar button).
- Already has `_mvat_manager` reference for accessing cameras.

---

## Target Architecture

### Signal Flow

```
AnnotationWindow user zooms/pans
  → viewNavigated(center_x, center_y, zoom_factor) signal
  → SyncController._on_main_view_navigated()
      → Gets depth at center pixel from Z-channel
      → CameraRay.from_pixel_and_camera() → 3D world point
      → For each visible context camera:
          → Camera.project(world_point) → (target_u, target_v)
      → ContextMatrixWidget.sync_to_targets(targets_dict, zoom_factor)
          → For each canvas: canvas.snap_to_target(u, v, zoom)
```

### Soft Override Behavior
- User pans/zooms an individual context canvas → it moves freely (no state flags needed).
- Next time the main AnnotationWindow navigates → `snap_to_target` is called on ALL canvases, snapping them back.
- This "snap-back" behavior is a natural consequence of the architecture — zero extra state management.

---

## Implementation Steps

### Step 1: The Sync Toggle UI

Already partially set up in Phase 2's toolbar:

```python
# In ContextMatrixWidget.create_top_toolbar():
self._sync_btn = QToolButton()
self._sync_btn.setCheckable(True)
self._sync_btn.setChecked(True)
self._sync_btn.setText("🔗")
self._sync_btn.setToolTip("Target-Lock Sync (enabled)")
self._sync_btn.toggled.connect(self._on_sync_toggled)

def _on_sync_toggled(self, checked):
    self.target_lock_enabled = checked
    self._sync_btn.setToolTip(
        "Target-Lock Sync (enabled)" if checked else "Target-Lock Sync (disabled)"
    )
    # If re-enabled, immediately sync to current main view
    if checked:
        self._request_sync_from_main_view()
```

### Step 2: Connect `viewNavigated` to Sync Controller

In `MVATManager._setup_connections()` or `AnnotationWindow` wiring:

```python
# In MVATManager._setup_connections():
# Connect AnnotationWindow's viewNavigated signal (from BaseCanvas)
if hasattr(self.annotation_window, 'viewNavigated'):
    self.annotation_window.viewNavigated.connect(self._on_main_view_navigated)
```

> **Note:** `AnnotationWindow` inherits `viewNavigated` from `BaseCanvas`. It fires on every pan completion and zoom event. The signal emits the center pixel and zoom factor.

### Step 3: The Sync Math Pipeline

Add to `MVATManager`:

```python
def _on_main_view_navigated(self, center_x: float, center_y: float, zoom_factor: float):
    """Handle navigation events from the main AnnotationWindow.

    Projects the viewport center into 3D world space, then back into
    each visible context camera to synchronize their viewports.
    """
    # Check prerequisites
    if not hasattr(self, 'context_matrix') or self.context_matrix is None:
        return
    if not self.context_matrix.target_lock_enabled:
        return
    if self.selected_camera is None:
        return

    # Step 1: Get the 3D world point at the viewport center
    world_point = self._get_world_point_at_pixel(
        self.selected_camera, center_x, center_y
    )
    if world_point is None:
        return

    # Step 2: Project into each visible context camera
    targets = {}
    capacity = self.context_matrix._get_visible_capacity()

    for i in range(capacity):
        canvas = self.context_matrix.canvas_pool[i]
        if not canvas.isVisible() or not canvas.current_image_path:
            continue

        camera = self.cameras.get(canvas.current_image_path)
        if not camera:
            continue

        pixel = camera.project(world_point)
        if np.isnan(pixel).any():
            continue

        target_u, target_v = float(pixel[0]), float(pixel[1])

        # Bounds check: only sync if the point is within the image
        if 0 <= target_u < camera.width and 0 <= target_v < camera.height:
            targets[i] = (target_u, target_v)

    # Step 3: Command the context matrix to sync
    self.context_matrix.sync_to_targets(targets, zoom_factor)


def _get_world_point_at_pixel(self, camera, px, py):
    """Get the 3D world point at a specific pixel coordinate.

    Attempts depth-based unprojection first, falls back to scene
    median depth for a rough estimate.

    Args:
        camera: Camera object for the active image.
        px, py: Pixel coordinates (float).

    Returns:
        np.ndarray [x,y,z] world point, or None if impossible.
    """
    # Clamp to image bounds
    px = max(0, min(px, camera.width - 1))
    py = max(0, min(py, camera.height - 1))

    # Try depth from Z-channel
    raster = camera._raster
    depth = None
    if raster.z_channel is not None and raster.z_data_type == 'depth':
        depth = raster.get_z_value(int(px), int(py))

    if depth is None or depth <= 0 or np.isnan(depth):
        # Fallback to scene median depth
        try:
            default_depth = self.viewer.get_scene_median_depth(camera.position)
        except Exception:
            default_depth = 10.0
    else:
        default_depth = depth

    try:
        ray = CameraRay.from_pixel_and_camera(
            pixel_xy=(px, py),
            camera=camera,
            depth=depth,
            default_depth=default_depth
        )
        return ray.terminal_point
    except Exception:
        return None
```

### Step 4: `ContextMatrixWidget.sync_to_targets()`

```python
def sync_to_targets(self, targets: dict, zoom_factor: float):
    """Synchronize visible canvases to the projected target points.

    Args:
        targets: dict mapping canvas_index -> (target_x, target_y).
        zoom_factor: The zoom level from the main AnnotationWindow.
    """
    if not self.target_lock_enabled:
        return

    for i, (target_x, target_y) in targets.items():
        if i < len(self.canvas_pool):
            canvas = self.canvas_pool[i]
            if canvas.isVisible() and canvas.active_image:
                canvas.snap_to_target(target_x, target_y, zoom_factor)
```

### Step 5: Throttling the Sync

The `viewNavigated` signal fires on every zoom step and pan movement. To prevent the context canvases from thrashing:

```python
# In ContextMatrixWidget.__init__:
self._sync_timer = QTimer()
self._sync_timer.setSingleShot(True)
self._sync_timer.timeout.connect(self._process_pending_sync)
self._pending_sync = None
self._sync_throttle_ms = 30  # ~33 fps

def request_sync(self, targets: dict, zoom_factor: float):
    """Throttled sync request."""
    self._pending_sync = (targets, zoom_factor)
    if not self._sync_timer.isActive():
        self._sync_timer.start(self._sync_throttle_ms)

def _process_pending_sync(self):
    if self._pending_sync:
        targets, zoom_factor = self._pending_sync
        self._pending_sync = None
        self.sync_to_targets(targets, zoom_factor)
```

Then in `MVATManager._on_main_view_navigated`, call `self.context_matrix.request_sync(targets, zoom_factor)` instead of `sync_to_targets` directly.

### Step 6: Conveyor Belt Synergy

When the conveyor belt shifts (Phase 3), new cameras should load already synced to the main view's target.

```python
# In ContextMatrixWidget._refresh_visible_canvases():
def _refresh_visible_canvases(self):
    """Repaint canvases based on current conveyor state."""
    # ... existing image loading code ...

    # After loading images, sync new canvases to main view
    if self.target_lock_enabled and self._mvat_manager:
        self._request_sync_from_main_view()

def _request_sync_from_main_view(self):
    """Request a sync using the main AnnotationWindow's current viewport state."""
    if not self._mvat_manager:
        return
    aw = self._mvat_manager.annotation_window
    if not aw.active_image or not aw.pixmap_image:
        return

    # Get current viewport center in scene coordinates
    viewport_center = aw.mapToScene(aw.viewport().rect().center())
    center_x = viewport_center.x()
    center_y = viewport_center.y()
    zoom_factor = aw.zoom_factor

    # Trigger the full sync pipeline
    self._mvat_manager._on_main_view_navigated(center_x, center_y, zoom_factor)
```

This ensures that when `Ctrl+Shift+Right` loads a new camera, it immediately gets synced to the main viewport's position — the user sees it appear already zoomed into the right coral head.

### Step 7: Performance & Detail on Demand (Thumbnail Proxy)

For large rasters, loading full-resolution images for N context canvases is expensive. Implement a two-tier loading strategy:

```python
# In ContextMatrixWidget.update_context_cameras():
def update_context_cameras(self, camera_paths: List[str]):
    capacity = self._get_visible_capacity()

    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if i < len(camera_paths):
            path = camera_paths[i]
            if canvas.current_image_path == path:
                continue

            raster = self._raster_manager.get_raster(path) if self._raster_manager else None
            if raster:
                # TIER 1: Load thumbnail proxy for instant display
                thumbnail = raster.get_thumbnail(longest_edge=512)
                if thumbnail and not thumbnail.isNull():
                    canvas.load_visuals(thumbnail, path, raster=raster)
                    canvas._is_thumbnail_proxy = True

                # TIER 2: Schedule full-res load if the canvas is zoomed in
                # (This will be triggered by the sync engine when zoom > threshold)
                continue
            canvas.clear_scene()
        else:
            canvas.clear_scene()
```

Then in `BaseCanvas.snap_to_target()`, check if the zoom requires full-res:

```python
def snap_to_target(self, target_x, target_y, relative_zoom):
    """Zoom and center on a specific pixel."""
    # If currently showing thumbnail and zoom is high, request full-res
    if hasattr(self, '_is_thumbnail_proxy') and self._is_thumbnail_proxy:
        if relative_zoom > 1.5:  # Threshold: zoomed in past 150%
            self._request_full_resolution()

    self.set_zoom_level(relative_zoom)
    self.center_on_pixel(target_x, target_y)
```

> **Note:** The full-res swap is deferred work that can be optimized later. For Phase 5, the thumbnail proxy provides instant feedback while the full-res image loads asynchronously.

---

## Edge Cases & Risks

- **No depth data:** If the active camera has no Z-channel and the 3D scene has no point cloud, `_get_world_point_at_pixel` uses a default depth of 10.0 meters. This produces a rough estimate that still provides useful spatial correspondence in most scenes.

- **Extreme aspect ratio differences:** A main camera looking straight down (nadir) syncing to a camera looking horizontally will project the center point to pixel coordinates far off-screen. The bounds check in `_on_main_view_navigated` skips canvases where the projected point is outside the image — this is correct (that camera can't see the area being inspected).

- **Rapid zoom oscillation:** User rapidly scrolling the zoom wheel could queue many `viewNavigated` emissions. The throttle timer (30ms) consolidates these into ~33 updates/sec, which is smooth without being wasteful.

- **Context canvas user override:** If a user manually pans canvas #2, its scrollbars change. On the next main view navigation, `snap_to_target` resets canvas #2's scrollbars. No boolean flags needed — the override is implicitly temporary.

- **Conveyor belt + sync race condition:** `_refresh_visible_canvases()` loads images and then requests sync. If the image load is slow (large file), the sync may execute before the image is fully rendered. Mitigation: `snap_to_target` checks `self.active_image` and bails if no image is loaded yet.

- **Orthographic cameras:** Orthographic cameras have different projection math. `Camera.project()` already handles this when the camera is `OrthographicCamera`. The sync engine is agnostic to camera type.

---

## Validation Criteria

After Phase 5 is complete:

1. Zoom into a coral head in the main AnnotationWindow → all context canvases instantly zoom to show the same coral head from their angles.
2. Pan in the main window → context canvases pan to follow.
3. Disable the sync toggle → context canvases stay put when main window navigates.
4. Re-enable sync → context canvases immediately snap to the current main view target.
5. Manually pan a context canvas → it stays at the user's position. Next main window navigation snaps it back.
6. Shift the conveyor belt → new cameras appear already zoomed/centered on the correct target area.
7. Context canvases with no visible target (point projects outside their image) show their unsynced view rather than jumping erratically.
