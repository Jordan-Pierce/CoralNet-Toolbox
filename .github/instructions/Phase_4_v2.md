# Phase 4: Implementing the Dual-Marker System

## Summary

Phase 4 connects the 3D spatial math to the new `BaseCanvas` viewports so that when the user moves their mouse in the main `AnnotationWindow` or sets a focal point via the 3D viewer, corresponding markers appear at the correct pixel locations in the context canvases. This builds on the existing `MousePositionBridge` (which already calculates ray projections and visibility) and the existing `Marker` class (which draws crosshair/circles).

The key insight: the 3D projection math already exists in `Camera.project()`, `CameraRay.from_pixel_and_camera()`, and `CameraRay.project_to_cameras()`. We just need to route the results to the new `BaseCanvas` instances instead of (or in addition to) the `CameraImageWidget` thumbnail painter.

---

## Current Architecture (What Exists)

### MousePositionBridge (in `MVATManager.py`)
- Throttles mouse events (25ms timer).
- On mouse move in AnnotationWindow:
  1. Gets depth from Z-channel at cursor pixel.
  2. Creates `CameraRay` from pixel + camera + depth.
  3. For each highlighted camera: creates target ray, checks occlusion.
  4. Calls `ray.project_to_cameras(self.manager.cameras)` → dict of `{image_path: (u, v, is_valid)}`.
  5. Calls `self.manager.camera_grid.update_markers(projections, accuracies, highlighted_paths, visibility_status, selected_path)`.
  6. Also calls `self.manager.viewer.show_rays(rays_with_colors)` to draw 3D rays.

### CameraGrid Markers (Thumbnail-Based)
- `CameraImageWidget.set_marker_position(x, y, accurate, color, is_occluded)` stores position.
- `CameraImageWidget._draw_marker(painter, rect)` in `paintEvent`:
  - Transforms image pixel coords → widget pixel coords (scale by thumbnail/image ratio).
  - Draws `QPainter.drawEllipse()` with color based on accuracy/occlusion.
- `CameraGrid.update_markers()` only updates **visible** widgets (virtualization-aware).

### Existing Marker Class (`MVAT/core/Marker.py`)
- `QGraphicsItemGroup` with an `QGraphicsEllipseItem` (circle) + `QGraphicsLineItem` (crosshairs).
- Used by `AnnotationWindow.set_incoming_marker(u, v, color)` for the focal point display.
- Does **not** use `ItemIgnoresTransformations` — so it scales with zoom.

### Focal Point Pipeline
- User double-clicks in AnnotationWindow → creates `CameraRay` → calls `MVATViewer.set_focal_point(world_point)`.
- `MVATViewer.focalPointChanged` signal → `MVATManager._on_focal_point_changed(point_3d)`.
- Manager projects point into active camera → calls `annotation_window.set_incoming_marker(u, v, color)`.
- **Missing:** Projection into context cameras (CameraGrid thumbnails don't get focal markers, only dynamic hover markers).

---

## Target Architecture

### Two Marker Types per BaseCanvas

| Marker | Trigger | Appearance | Behavior |
|--------|---------|------------|----------|
| **Static Focal** | Double-click in AnnotationWindow or 3D viewer focal point | Crosshair (X pattern) | Stays fixed until next focal point is set. `ItemIgnoresTransformations` = constant screen size. |
| **Dynamic Hover** | Mouse movement in AnnotationWindow | Circle (○) | Moves with mouse, clears on mouse leave. `ItemIgnoresTransformations` = constant screen size. |

Both markers use `QGraphicsItem.ItemIgnoresTransformations` so they remain a crisp, fixed pixel size on screen regardless of canvas zoom level.

---

## Implementation Steps

### Step 1: Initialize Marker Graphics in `BaseCanvas`

Extend `BaseCanvas.__init__()` (from Phase 1):

```python
def _init_markers(self):
    """Create persistent marker graphics items for the scene."""

    # --- Static Focal Marker (Crosshair) ---
    self._static_marker = QGraphicsItemGroup()

    # Crosshair line length (in screen pixels, ignores transforms)
    L = 12  # Half-length of each crosshair arm

    h_line = QGraphicsLineItem(-L, 0, L, 0, self._static_marker)
    v_line = QGraphicsLineItem(0, -L, 0, L, self._static_marker)

    pen = QPen(QColor(255, 165, 0), 2)  # Orange, 2px
    h_line.setPen(pen)
    v_line.setPen(pen)

    self._static_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    self._static_marker.setZValue(100)
    self.scene.addItem(self._static_marker)
    self._static_marker.hide()

    # --- Dynamic Hover Marker (Circle) ---
    R = 6  # Radius in screen pixels
    self._dynamic_marker = QGraphicsEllipseItem(-R, -R, 2*R, 2*R)
    self._dynamic_marker.setBrush(Qt.NoBrush)

    self._dynamic_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    self._dynamic_marker.setZValue(100)
    self.scene.addItem(self._dynamic_marker)
    self._dynamic_marker.hide()
```

**Call `_init_markers()` from:**
- `BaseCanvas.__init__()` (initial setup)
- `BaseCanvas._on_scene_cleared()` (after `clear_scene()` recreates the scene)

### Step 2: The Canvas Marker API

```python
def update_static_marker(self, x, y, color=None):
    """Move the static crosshair marker to image pixel (x, y).

    Args:
        x, y: Pixel coordinates in image space.
        color: QColor for the marker. Default: orange.
    """
    if self._static_marker is None:
        return
    # Bounds check
    if self.pixmap_image and not (0 <= x < self.pixmap_image.width() and
                                   0 <= y < self.pixmap_image.height()):
        self._static_marker.hide()
        return

    self._static_marker.setPos(x, y)
    if color:
        pen = QPen(color, 2)
        for child in self._static_marker.childItems():
            child.setPen(pen)
    self._static_marker.show()

def clear_static_marker(self):
    """Hide the static crosshair marker."""
    if self._static_marker:
        self._static_marker.hide()

def update_dynamic_marker(self, x, y, color=None, is_valid=True):
    """Move the dynamic circle marker to image pixel (x, y).

    Args:
        x, y: Pixel coordinates in image space.
        color: QColor for the marker.
        is_valid: If False, use dashed pen (occluded/estimated).
    """
    if self._dynamic_marker is None:
        return
    # Bounds check
    if self.pixmap_image and not (0 <= x < self.pixmap_image.width() and
                                   0 <= y < self.pixmap_image.height()):
        self._dynamic_marker.hide()
        return

    self._dynamic_marker.setPos(x, y)

    pen = QPen(color or QColor(0, 255, 0), 2)  # Default green
    if not is_valid:
        pen.setStyle(Qt.DashLine)
    else:
        pen.setStyle(Qt.SolidLine)
    self._dynamic_marker.setPen(pen)
    self._dynamic_marker.show()

def clear_dynamic_marker(self):
    """Hide the dynamic circle marker."""
    if self._dynamic_marker:
        self._dynamic_marker.hide()
```

### Step 3: Extend MousePositionBridge to Feed Context Matrix

Currently `_process_pending_position()` calls `self.manager.camera_grid.update_markers(...)`. We add a parallel call to update the context matrix canvases.

```python
def _process_pending_position(self):
    # ... existing ray calculation code ...

    projections = ray.project_to_cameras(self.manager.cameras)

    # Update CameraGrid thumbnails (existing)
    highlighted_paths = {cam.image_path for cam in highlighted_cameras}
    selected_path = camera.image_path
    try:
        self.manager.camera_grid.update_markers(
            projections, accuracies, highlighted_paths,
            visibility_status, selected_path
        )
    except Exception:
        self.clear_all_markers()

    # UPDATE CONTEXT MATRIX CANVASES (NEW)
    try:
        if hasattr(self.manager, 'context_matrix') and self.manager.context_matrix:
            self.manager.context_matrix.update_dynamic_markers(
                projections, accuracies, visibility_status
            )
    except Exception:
        pass
```

### Step 4: `ContextMatrixWidget.update_dynamic_markers()`

```python
def update_dynamic_markers(self, projections: dict, accuracies: dict,
                            visibility_status: dict):
    """Update dynamic hover markers on all visible canvases.

    Args:
        projections: {image_path: (u, v, is_valid)}
        accuracies: {image_path: has_accurate_depth}
        visibility_status: {image_path: is_occluded}
    """
    capacity = self._get_visible_capacity()

    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if not canvas.isVisible() or not canvas.current_image_path:
            continue

        path = canvas.current_image_path
        proj = projections.get(path)

        if not proj:
            canvas.clear_dynamic_marker()
            continue

        px, py, is_valid = proj
        if not is_valid:
            canvas.clear_dynamic_marker()
            continue

        acc = accuracies.get(path, False)
        is_occluded = visibility_status.get(path, False)

        # Color based on accuracy and occlusion
        from coralnet_toolbox.MVAT.core.constants import (
            MARKER_COLOR_HIGHLIGHTED, MARKER_COLOR_INVALID
        )
        color = MARKER_COLOR_HIGHLIGHTED if (acc and not is_occluded) else MARKER_COLOR_INVALID

        canvas.update_dynamic_marker(px, py, color=color, is_valid=(acc and not is_occluded))

def clear_all_dynamic_markers(self):
    """Clear dynamic markers from all canvases."""
    for canvas in self.canvas_pool:
        canvas.clear_dynamic_marker()
```

### Step 5: Route Static Focal Marker to Context Matrix

Extend `MVATManager._on_focal_point_changed()`:

```python
def _on_focal_point_changed(self, point_3d):
    """Respond to viewer focal-point changes."""
    self.current_focal_point = point_3d

    # Project to active camera and show in AnnotationWindow (existing)
    if self.selected_camera and self.selected_camera.image_path in self.cameras:
        pixel = self.selected_camera.project(point_3d)
        if not np.isnan(pixel).any():
            u, v = pixel[0], pixel[1]
            depth = self.selected_camera._raster.get_z_value(int(u), int(v))
            color = MARKER_COLOR_SELECTED if depth is not None and depth > 0 else MARKER_COLOR_INVALID
            self.annotation_window.set_incoming_marker(u, v, color)
        else:
            self.annotation_window.marker.hide()

    # PROJECT TO CONTEXT CANVASES (NEW)
    if hasattr(self, 'context_matrix') and self.context_matrix:
        self.context_matrix.update_static_markers_from_3d(point_3d, self.cameras)

def # In ContextMatrixWidget:
def update_static_markers_from_3d(self, point_3d, cameras: dict):
    """Project a 3D focal point into all visible canvases.

    Args:
        point_3d: numpy array [x, y, z] in world coordinates.
        cameras: dict of {image_path: Camera} for projection.
    """
    capacity = self._get_visible_capacity()

    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if not canvas.isVisible() or not canvas.current_image_path:
            continue

        camera = cameras.get(canvas.current_image_path)
        if not camera:
            canvas.clear_static_marker()
            continue

        pixel = camera.project(point_3d)
        if np.isnan(pixel).any():
            canvas.clear_static_marker()
            continue

        u, v = float(pixel[0]), float(pixel[1])

        # Check occlusion
        is_occluded = camera.is_point_occluded_depth_based(point_3d, depth_threshold=0.15)
        from coralnet_toolbox.MVAT.core.constants import MARKER_COLOR_SELECTED, MARKER_COLOR_INVALID
        color = MARKER_COLOR_INVALID if is_occluded else MARKER_COLOR_SELECTED

        canvas.update_static_marker(u, v, color=color)
```

### Step 6: Handle Mouse Leave Events

When the mouse leaves the AnnotationWindow, clear all dynamic markers.

Extend `MousePositionBridge.clear_all_markers()`:

```python
def clear_all_markers(self):
    """Clear all dynamic markers from CameraGrid and ContextMatrix."""
    try:
        self.manager.camera_grid.clear_all_markers()
    except Exception:
        pass
    try:
        if hasattr(self.manager, 'context_matrix') and self.manager.context_matrix:
            self.manager.context_matrix.clear_all_dynamic_markers()
    except Exception:
        pass
```

The `AnnotationWindow.mouseMoveEvent` already calls `self.mouseMoved.emit(x, y)` which feeds `MousePositionBridge.on_mouse_moved()`. When the mouse exits the AnnotationWindow viewport, the bridge gets coordinates outside the image bounds and calls `clear_all_markers()`.

### Step 7: Conveyor Belt Marker Persistence

When the conveyor belt shifts (Phase 3) and a new camera slides in, it should immediately show the static focal marker if one is set.

```python
# In ContextMatrixWidget._refresh_visible_canvases():
def _refresh_visible_canvases(self):
    """Repaint canvases based on current conveyor state."""
    capacity = self._get_visible_capacity()
    if not self.ordered_context_paths:
        for i in range(capacity):
            self.canvas_pool[i].clear_scene()
        return

    target_paths = self.ordered_context_paths[
        self.current_rank_offset : self.current_rank_offset + capacity
    ]
    self.update_context_cameras(target_paths)

    # Re-apply static focal markers to newly loaded canvases
    if self._mvat_manager and self._mvat_manager.current_focal_point is not None:
        self.update_static_markers_from_3d(
            self._mvat_manager.current_focal_point,
            self._mvat_manager.cameras
        )
```

---

## Edge Cases & Risks

- **Performance of `project_to_cameras`:** This iterates over ALL cameras, not just those in the context matrix. It's called on every throttled mouse move (~40 times/sec). The existing implementation already handles this for CameraGrid. For the context matrix, we only need projections for the N visible cameras. However, `project_to_cameras` is a batch operation that's already fast (matrix multiplication). We just filter the results.

- **Marker persistence across scene clears:** `BaseCanvas.clear_scene()` destroys the scene and its items. The `_on_scene_cleared()` hook re-initializes markers, but they lose their position. After `load_visuals()` is called, the static marker should be re-applied. The conveyor belt refresh (above) handles this.

- **Occluded marker at extreme angles:** If a 3D point is behind a context camera, `project()` returns `[nan, nan]`. The marker API hides the marker for NaN coordinates. ✓

- **Dynamic marker flicker:** If the mouse is near the edge of the main image where depth is invalid, the ray's `terminal_point` may oscillate. The throttle timer (25ms) already mitigates this.

- **Color consistency:** Using the same `MARKER_COLOR_*` constants from `MVAT/core/constants.py` ensures visual consistency between CameraGrid thumbnails and BaseCanvas viewports.

---

## Validation Criteria

After Phase 4 is complete:

1. Moving the mouse in the main AnnotationWindow causes green circles to appear at the correct projected pixel positions in all visible context canvases.
2. Circles turn red/dashed when the projected point is occluded in that camera's view.
3. Double-clicking in the main window (or setting focal point in 3D viewer) places orange crosshair markers in all context canvases at the correct projected locations.
4. Markers remain correctly positioned when zooming/panning individual context canvases.
5. Markers appear crisp and the same screen size regardless of canvas zoom level (`ItemIgnoresTransformations`).
6. Shifting the conveyor belt loads new cameras with correct static markers already placed.
7. Moving the mouse off the AnnotationWindow clears all dynamic circle markers.
