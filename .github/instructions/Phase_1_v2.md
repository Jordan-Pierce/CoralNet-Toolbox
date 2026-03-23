# Phase 1: Extracting the `BaseCanvas` from `AnnotationWindow`

## Summary

`AnnotationWindow` (in `QtAnnotationWindow.py`) is a ~3000-line monolith inheriting from `QGraphicsView`. It owns the `QGraphicsScene`, image display, Z-channel visualization, annotation dictionaries, undo/redo stack, tool dispatch, mask editing, and MVAT marker logic. To create reusable "mini-canvases" for the Security Camera Matrix (Phase 2), we must extract the **pure viewing** responsibilities into a lightweight `BaseCanvas` class. `AnnotationWindow` will then inherit from `BaseCanvas` instead of `QGraphicsView`, preserving its full feature set while making the viewport logic reusable.

This phase is strictly a refactor — the application must behave identically when complete.

---

## Current Architecture (What Exists)

### AnnotationWindow class hierarchy
```
QGraphicsView
  └── AnnotationWindow  (~3000 lines)
        ├── QGraphicsScene management (self.scene)
        ├── Image loading: set_image(), display_image(), clear_scene()
        ├── Z-channel visualization: _load_z_channel_visualization(), update_z_colormap(), etc.
        ├── Navigation: wheelEvent → ZoomTool, mousePressEvent → PanTool (right-click)
        ├── Tool dispatch: tools dict, mask_tools set, selected_tool routing
        ├── Annotation storage: annotations_dict, image_annotations_dict, selected_annotations
        ├── Undo/Redo: ActionStack
        ├── MVAT marker: self.marker (Marker class), set_incoming_marker()
        ├── Placeholder label ("No image loaded")
        ├── Status bar widgets: mouse_position_label, image_dimensions_label, etc.
        └── Toolbar hooks: create_top_toolbar(), create_bottom_toolbar()
```

### Navigation Tools (External Classes)
- **`PanTool`** (`QtPanTool.py`): Right-click drag → adjusts scrollbar values. References `self.annotation_window.pan_active`, `pan_start`, `horizontalScrollBar()`, `verticalScrollBar()`.
- **`ZoomTool`** (`QtZoomTool.py`): Mouse wheel → `self.annotation_window.scale(factor, factor)`. Tracks `zoom_factor`, `min_zoom_factor`, `initial_zoom`. Does anchor-under-mouse zoom.

### MVAT Marker (Existing)
- `Marker` class in `MVAT/core/Marker.py`: A `QGraphicsItemGroup` containing an ellipse and crosshair lines.
- `AnnotationWindow.set_incoming_marker(u, v, color)` adds it to the scene.
- `MousePositionBridge` projects 3D points into 2D and calls `CameraGrid.update_markers()` (paints on thumbnails using `QPainter`).

### Key Signals Already Emitted by AnnotationWindow
- `mouseMoved(int, int)` — emitted in `mouseMoveEvent` after mapping to scene coordinates.
- `viewChanged(int, int)` — emitted after zoom/pan completes.
- `imageLoaded(int, int)` — emitted after `set_image()` finishes.

---

## Target Architecture

```
QGraphicsView
  └── BaseCanvas  (NEW — ~400 lines)
        ├── QGraphicsScene creation & management
        ├── Image display: load_visuals(), clear_scene()
        ├── Z-channel visualization (full suite)
        ├── Built-in pan (middle/right-click drag) and zoom (mouse wheel)
        ├── Placeholder label
        ├── Viewport control API: center_on_pixel(), set_zoom_level(), fit_to_image()
        ├── Dual-marker slots: static marker (crosshair), dynamic marker (circle)
        ├── Read-only annotation rendering
        ├── Signals: viewNavigated, mouseHovered
        └── No tools, no annotation dict, no undo stack, no toolbar widgets

  └── AnnotationWindow(BaseCanvas)  (REFACTORED — ~2600 lines)
        ├── Inherits all BaseCanvas viewing capabilities via super()
        ├── Tool dispatch (tools dict, selected_tool routing)
        ├── Annotation storage (annotations_dict, etc.)
        ├── Undo/Redo (ActionStack)
        ├── Status bar and toolbar widgets
        ├── Event routing: left-click → tool, middle/right → super() for pan
        └── Overrides: wheelEvent checks for Ctrl+tool modifier, else super()
```

---

## Implementation Steps

### Step 1: Create `QtBaseCanvas.py` Skeleton

**File:** `coralnet_toolbox/QtBaseCanvas.py`

```python
class BaseCanvas(QGraphicsView):
    # Signals
    viewNavigated = pyqtSignal(float, float, float)  # center_x, center_y, zoom_factor
    mouseHovered = pyqtSignal(float, float)           # scene_x, scene_y

    def __init__(self, parent=None):
        super().__init__(parent)
        # Scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # set dark background (#1e1e1e)

        # Image state
        self.pixmap_image = None
        self.current_image_path = None
        self.active_image = False
        self._base_image_item = None  # QGraphicsPixmapItem for the loaded image

        # Navigation state
        self.zoom_factor = 1.0
        self._pan_active = False
        self._pan_start = None
        self._min_zoom = 0.1

        # Placeholder
        self._placeholder_label = QLabel("No image loaded", self.viewport())
        # style and show

        # Z-channel visualization attributes
        self.z_item = None
        self.z_data_raw = None
        self.z_data_normalized = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None
        self.z_nodata_mask = None
        self.dynamic_z_scaling = False

        # Marker slots (Phase 4 will populate)
        self._static_marker = None   # QGraphicsItemGroup (crosshair)
        self._dynamic_marker = None  # QGraphicsEllipseItem (circle)

        # View configuration
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
```

### Step 2: Build the Native Navigation Engine

Move pan and zoom logic **directly into `BaseCanvas`** so it is self-contained and independent of the Tool architecture.

**wheelEvent(event):**
1. Check `self.active_image`, bail if False.
2. Compute `factor` (1.1 for zoom-in, 0.9 for zoom-out).
3. Clamp against `self._min_zoom`.
4. Store scene position at cursor before zoom.
5. Call `self.scale(factor, factor)`.
6. Update `self.zoom_factor`.
7. If zoomed to min, center image.
8. Emit `viewNavigated(center_x, center_y, self.zoom_factor)`.

**mousePressEvent(event):**
1. If `event.button()` is `MiddleButton` or `RightButton`:
   - Set `self._pan_active = True`, record `self._pan_start = event.pos()`.
   - Set cursor to `Qt.ClosedHandCursor`.

**mouseMoveEvent(event):**
1. If `self._pan_active`: compute delta, adjust scrollbars, update `self._pan_start`.
2. Map cursor to scene coordinates, emit `mouseHovered(x, y)`.

**mouseReleaseEvent(event):**
1. If panning: `self._pan_active = False`, restore cursor.
2. Compute viewport center in scene coordinates.
3. Emit `viewNavigated(center_x, center_y, self.zoom_factor)`.

**resizeEvent(event):**
1. If `self.active_image`, `fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)`.
2. Keep placeholder geometry synced.

> **Key difference from current PanTool:** `BaseCanvas` uses `MiddleButton` **or** `RightButton` for panning natively. `AnnotationWindow` will later override `mousePressEvent` to route `RightButton` to the `PanTool` only when it needs to, delegating to `super()` for the actual pan math.

### Step 3: Migrate Image and Z-Channel Render Logic

Move these methods from `AnnotationWindow` to `BaseCanvas`:

| Method | Notes |
|--------|-------|
| `clear_scene()` | Move as-is. Remove the `self.focal_marker = None` line (that's AnnotationWindow-specific). Add an overridable `_on_scene_cleared()` hook so AnnotationWindow can clean up its z-signal disconnection and focal_marker. |
| `_show_placeholder()` / `_hide_placeholder()` | Move as-is. |
| `_load_z_channel_visualization(raster)` | Move as-is. |
| `update_z_colormap(colormap_name)` | Move as-is. |
| `update_z_transparency(value)` | Move as-is. The slider reference becomes a parameter: `set_z_opacity(value / 255.0)`. |
| `toggle_dynamic_z_scaling(enabled)` | Move as-is. |
| `update_dynamic_range()` | Move as-is. |
| `schedule_dynamic_range_update()` | Move as-is. Debounce timer owned by BaseCanvas. |
| `_reset_z_channel_to_full_range()` | Move as-is but remove `self.main_window` reference. Accept colormap_name as parameter instead. |
| `clear_z_channel_visualization(image_path)` | Move as-is. |
| `refresh_z_channel_visualization()` | **Stays in AnnotationWindow** — it uses `main_window.image_window.raster_manager` which BaseCanvas should not know about. |
| `viewportToScene()` | Move as-is. |
| `get_image_dimensions()` | Move as-is. |
| `get_image_rect()` | Move as-is. |
| `reset_scene_view()` | Move partially. The `tools["zoom"].reset_zoom()` call stays in AnnotationWindow. BaseCanvas gets a `fit_to_image()` that calls `fitInView(get_image_rect(), Qt.KeepAspectRatio)`. |

**Create `load_visuals(q_image, image_path, raster=None)`:**
This is the **new entry point** for displaying an image in a BaseCanvas:
1. Call `self.clear_scene()`.
2. Hide placeholder.
3. Set `self.pixmap_image = QPixmap(q_image)`.
4. Create `QGraphicsPixmapItem`, set Z-value to -10, add to scene.
5. Set `self._base_image_item` reference.
6. Set `self.current_image_path = image_path`.
7. Set `self.active_image = True`.
8. If `raster` is provided and has Z-channel, call `_load_z_channel_visualization(raster)`.
9. Call `fit_to_image()`.

### Step 4: Add the Viewport Control API

These methods enable external managers to command the camera position programmatically (used by Phase 5's Target-Lock).

```python
def center_on_pixel(self, x, y):
    """Center the view on the given image pixel coordinate."""
    self.centerOn(QPointF(x, y))

def set_zoom_level(self, factor):
    """Set the view transformation to a specific zoom level."""
    current = self.transform().m11()
    if current > 0:
        scale_change = factor / current
        self.scale(scale_change, scale_change)
        self.zoom_factor = factor

def fit_to_image(self):
    """Reset view to fit the entire image, recalculating min zoom."""
    if self.pixmap_image:
        image_rect = self.get_image_rect()
        self.fitInView(image_rect, Qt.KeepAspectRatio)
        self.zoom_factor = self.transform().m11()
        self._min_zoom = self.zoom_factor

def snap_to_target(self, target_x, target_y, relative_zoom):
    """Zoom and center on a specific pixel. Used by Target-Lock sync."""
    self.set_zoom_level(relative_zoom)
    self.center_on_pixel(target_x, target_y)
```

### Step 5: Prepare Dual-Marker Slots

Create the marker infrastructure that Phase 4 will fully implement. For now, define the API and basic items:

```python
def _init_markers(self):
    """Create marker graphics items (hidden by default)."""
    # Static Focal Marker (Crosshair)
    self._static_marker = QGraphicsItemGroup()
    # Add crosshair lines and ellipse like existing Marker class
    self._static_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    self._static_marker.setZValue(100)
    self.scene.addItem(self._static_marker)
    self._static_marker.hide()

    # Dynamic Hover Marker (Circle)
    self._dynamic_marker = QGraphicsEllipseItem(-5, -5, 10, 10)
    self._dynamic_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    self._dynamic_marker.setZValue(100)
    self.scene.addItem(self._dynamic_marker)
    self._dynamic_marker.hide()

def update_static_marker(self, x, y, color=None): ...
def clear_static_marker(self): ...
def update_dynamic_marker(self, x, y, color=None, is_valid=True): ...
def clear_dynamic_marker(self): ...
```

> **Important:** Markers must be re-added to the scene after every `clear_scene()` call. Override `_on_scene_cleared()` to call `_init_markers()`.

### Step 6: Define Read-Only Annotation Rendering

Add a method for context canvases to display annotations passively:

```python
def render_readonly_annotations(self, annotation_data_list):
    """Render annotations as non-interactive visual overlays.

    Args:
        annotation_data_list: List of dicts with keys:
            'type': 'patch'|'rectangle'|'polygon'
            'geometry': QRectF or QPolygonF
            'color': QColor
            'transparency': int (0-255)
    """
    for data in annotation_data_list:
        item = ...  # Create appropriate QGraphicsItem
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setFlag(QGraphicsItem.ItemIsMovable, False)
        item.setFlag(QGraphicsItem.ItemIsFocusable, False)
        item.setZValue(0)  # Above image, below markers
        self.scene.addItem(item)
```

### Step 7: Rewire `AnnotationWindow` to Inherit from `BaseCanvas`

This is the most delicate step. Changes required:

1. **Change class declaration:**
   ```python
   class AnnotationWindow(BaseCanvas):
   ```

2. **Update `__init__`:**
   - Call `super().__init__(parent)` instead of `QGraphicsView.__init__(self, parent)`.
   - **Remove** all scene creation code (handled by BaseCanvas).
   - **Remove** `self.pixmap_image = None`, `self.active_image = False`, etc. (inherited).
   - **Remove** placeholder label creation (inherited).
   - **Remove** Z-channel attribute initialization (inherited).
   - **Remove** `setTransformationAnchor`, `setResizeAnchor`, scroll bar policies (inherited).
   - **Keep** everything else: `self.main_window`, `self.action_stack`, `self.annotation_size`, `self.tools`, `self.annotations_dict`, `self.selected_annotations`, `self.marker`, toolbar widgets, signal connections.

3. **Rewire `wheelEvent`:**
   ```python
   def wheelEvent(self, event):
       if self.selected_tool and event.modifiers() & Qt.ControlModifier:
           # Tool-specific wheel handling (e.g., brush size)
           self.tools[self.selected_tool].wheelEvent(event)
       elif self.active_image:
           # Delegate to BaseCanvas native zoom
           super().wheelEvent(event)
       self.viewChanged.emit(*self.get_image_dimensions())
       self.schedule_dynamic_range_update()
   ```

4. **Rewire `mousePressEvent`:**
   ```python
   def mousePressEvent(self, event):
       if event.button() == Qt.RightButton and self.active_image:
           # Use BaseCanvas native pan
           super().mousePressEvent(event)
           return
       if self.selected_tool:
           self.tools[self.selected_tool].mousePressEvent(event)
       QGraphicsView.mousePressEvent(self, event)
   ```
   > Note: The existing `PanTool.mousePressEvent` checks for `RightButton`. We can either keep delegating to PanTool (simpler, preserves existing behavior exactly) or route to `super()`. **Safest approach: keep PanTool for now** to minimize behavioral changes. BaseCanvas's native pan uses `MiddleButton` only. AnnotationWindow keeps routing `RightButton` to PanTool as before.

5. **Rewire `mouseMoveEvent`:**
   - Keep existing tool dispatching.
   - After tool handling, call `super().mouseMoveEvent(event)` to emit `mouseHovered`.
   - OR keep the existing `self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))` since AnnotationWindow uses `mouseMoved` signal (not `mouseHovered`).

6. **Rewire `set_image`:**
   - The heavy raster loading, progress display, and annotation reloading stay in AnnotationWindow.
   - Replace the manual scene building with calls to inherited methods:
     ```python
     # Instead of building QGraphicsPixmapItem manually:
     self.load_visuals(q_image, image_path, raster=raster)
     ```
   - The Z-channel signal connection and colormap application still happen in AnnotationWindow after `load_visuals`.

7. **Rewire `clear_scene`:**
   - Override `_on_scene_cleared()` to handle AnnotationWindow-specific cleanup (nullify `annotation.graphics_item`, disconnect z-channel signals, reset `focal_marker`).
   - Call `super().clear_scene()` for the base cleanup.

8. **Delete redundant code from AnnotationWindow:**
   - Remove `_show_placeholder()` / `_hide_placeholder()` (inherited).
   - Remove `viewportToScene()` (inherited).
   - Remove `get_image_dimensions()` (inherited).
   - Remove `get_image_rect()` (inherited).
   - Remove `_load_z_channel_visualization()` (inherited).
   - Remove `update_z_colormap()` (inherited).
   - Remove `clear_z_channel_visualization()` (inherited).
   - Remove `update_dynamic_range()` and related methods (inherited).

---

## Edge Cases & Risks

- **Z-channel `main_window` references:** Several Z-channel methods reference `self.main_window.z_colormap_dropdown` and `self.main_window.z_transparency_widget`. BaseCanvas must NOT reference `main_window`. Solutions:
  - `_reset_z_channel_to_full_range()` and `update_dynamic_range()` accept `colormap_name` as a parameter.
  - `_load_z_channel_visualization()` accepts `current_opacity` as a parameter.
  - AnnotationWindow overrides these methods to inject the `main_window` references.

- **Scene recreation:** `clear_scene()` currently creates a new `QGraphicsScene` and calls `self.setScene(self.scene)`. This destroys all items. After migration, `BaseCanvas.clear_scene()` must also re-initialize markers via `_on_scene_cleared()`.

- **PanTool coupling:** PanTool directly accesses `self.annotation_window.pan_active` and `self.annotation_window.pan_start`. If we move pan state to BaseCanvas, PanTool needs updating. **Decision: keep PanTool using AnnotationWindow's `pan_active`/`pan_start` attributes for now.** BaseCanvas's native pan uses its own `_pan_active`/`_pan_start`. Context canvases (Phase 2) will use BaseCanvas's native pan.

- **`pyqtgraph` dependency:** Z-channel colormap uses `pg.colormap.get()`. This stays in BaseCanvas since context views also need it.

- **Multiple scene instances:** When `clear_scene()` is called, the old scene is `deleteLater()`'d and a new one is created. The markers and Z-item must be re-added to the new scene.

---

## Open Questions (Non-Blocking)

1. **Should `BaseCanvas` own the `rasterio_image` reference?** Currently AnnotationWindow uses it for cropping annotations. Decision: No — `rasterio_image` is annotation-specific, stays in AnnotationWindow.

2. **Should BaseCanvas emit `viewChanged` like AnnotationWindow?** Decision: BaseCanvas emits `viewNavigated` (center + zoom). AnnotationWindow keeps emitting `viewChanged` (image dimensions) for backward compatibility with its status bar widgets.

---

## Validation Criteria

After Phase 1 is complete:

1. The application launches and all existing functionality works identically.
2. `BaseCanvas` can be instantiated standalone: `canvas = BaseCanvas(); canvas.load_visuals(some_qimage, "path")` — it shows the image, responds to wheel zoom and middle-click pan.
3. `AnnotationWindow` still handles all tool dispatch, annotation management, and MVAT marker display.
4. No `main_window` references exist in `BaseCanvas`.
5. Unit test: instantiate 4 `BaseCanvas` objects, call `load_visuals` on each, verify all 4 display independent images and respond to zoom/pan independently.
