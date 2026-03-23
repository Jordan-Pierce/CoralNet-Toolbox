# Phase 2: Building the "Security Camera" Matrix (`ContextMatrixWidget`)

## Summary

Replace the scrollable `CameraGrid` (a virtualized thumbnail grid using `QPainter`-drawn `CameraImageWidget` instances) with a new `ContextMatrixWidget` that manages a fixed grid of interactive `BaseCanvas` viewports. The existing `CameraGrid` will remain as-is for camera selection/highlighting. The `ContextMatrixWidget` is an **additional** dock that displays the N nearest cameras as fully zoomable/pannable mini-canvases.

---

## Current Architecture (What Exists)

### CameraGrid (`MVAT/ui/QtCameraGrid.py`)
- **Purpose:** Scrollable grid of static thumbnails for camera selection/highlighting.
- **Widget type:** `CameraImageWidget` instances positioned manually on a `QWidget` content area inside a `QScrollArea`.
- **Features:** Virtualization (show/hide based on scroll position), Ctrl+Click multi-select, double-click to select active, context menu "Select Image", marker overlay via `QPainter`, Ctrl+Wheel thumbnail resizing.
- **Signals:** `selection_requested`, `toggle_requested`, `active_requested`, `clear_requested`, `camera_selected`, `camera_hovered`.
- **Layout:** Single scrollable column/grid, thumbnail size adjustable.

### CameraGrid Marker System
- `CameraImageWidget.set_marker_position(x, y, accurate, color, is_occluded)` → paints a circle on the thumbnail using `QPainter` in `paintEvent`.
- `CameraGrid.update_markers(projections, accuracies, highlighted_paths, visibility_status, selected_path)` → loops visible widgets and calls `set_marker_position`.
- `MousePositionBridge._process_pending_position()` → calculates ray projections and calls `camera_grid.update_markers()`.

### Dock System (PyQtAds)
- `DockWrapper` wraps any QWidget into a `CDockWidget`.
- Supports `add_toolbar()` (top or bottom), `add_menu()`, `set_status_bar()`.
- Layout managed by `CDockManager` in `QtMainWindow.py`.
- Currently: `self.grid_dock = DockWrapper("Grid", "GridDock", self.camera_grid, self)`.

### MVATManager Connection Points
- `self.camera_grid` referenced throughout `MVATManager`:
  - `camera_grid.set_cameras(perspective_cameras)` — populates grid.
  - `camera_grid.set_camera_order(ordered_paths)` — reorders by proximity.
  - `camera_grid.update_markers(...)` — projects mouse position.
  - `camera_grid.clear_all_markers()` — clears dynamic markers.
  - `camera_grid._sync_ui_to_model()` — refreshes selection borders.
  - Various signal connections for selection/highlight intent.

---

## Target Architecture

```
CameraGrid (EXISTING — remains for selection/highlight browsing)
  └── DockWrapper "Grid" (scrollable thumbnail grid, no changes)

ContextMatrixWidget (NEW — interactive viewport grid)
  └── DockWrapper "Context" (new dock)
        ├── Top toolbar: layout chooser, sync toggle, rank indicator
        ├── QGridLayout containing N × BaseCanvas instances
        └── Object pool of pre-created BaseCanvas objects
```

The `ContextMatrixWidget` is a **consumer** of the same data that `CameraGrid` uses, but it presents it differently: instead of many small thumbnails, it shows N large interactive viewports.

---

## Implementation Steps

### Step 1: Create `QtContextMatrix.py` Skeleton

**File:** `coralnet_toolbox/MVAT/ui/QtContextMatrix.py`

```python
class ContextMatrixWidget(QWidget):
    """
    A flexible grid of interactive BaseCanvas viewports for displaying
    the N nearest cameras to the active camera in the MVAT workspace.
    """

    # Signals
    matrixCapacityChanged = pyqtSignal(int)        # N changed (layout rebuild)
    canvasPromoted = pyqtSignal(str)                # image_path when user double-clicks to "promote"
    rankIndicatorUpdated = pyqtSignal(int, int, int) # start_rank, end_rank, total

    def __init__(self, max_canvases=9, parent=None):
        super().__init__(parent)

        # Layout
        self._grid_layout = QGridLayout(self)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(2)

        # Canvas pool
        self.canvas_pool: List[BaseCanvas] = []
        self._max_canvases = max_canvases

        # Layout state
        self.current_rows = 1
        self.current_cols = 1
        self.auto_flow_enabled = True

        # Conveyor belt state (Phase 3 will fully populate)
        self.ordered_context_paths: List[str] = []
        self.current_rank_offset = 0

        # Sync toggle (Phase 5)
        self.target_lock_enabled = True

        # References
        self._raster_manager = None
        self._mvat_manager = None

        # Initialize
        self._create_canvas_pool()
        self._rebuild_layout(1, 1)
```

### Step 2: The Canvas Object Pool

Pre-create `BaseCanvas` instances to avoid expensive construction/destruction during layout changes.

```python
def _create_canvas_pool(self):
    """Pre-create max_canvases BaseCanvas instances, all hidden."""
    for i in range(self._max_canvases):
        canvas = BaseCanvas(parent=self)
        canvas.hide()
        # Connect double-click for "Promote to Main" (Phase 2 feature)
        canvas.mouseDoubleClickEvent = lambda event, c=canvas: self._on_canvas_double_clicked(c)
        self.canvas_pool.append(canvas)
```

**Why object pool?** Instantiating `QGraphicsView` + `QGraphicsScene` is expensive (~5ms each). With a pool, switching from a 2×2 to a 3×1 grid is instant — just hide/show and reparent in the grid.

### Step 3: Implement `_rebuild_layout(rows, cols)`

```python
def _rebuild_layout(self, rows, cols):
    """Rearrange the grid with the specified rows and columns."""
    # 1. Remove all canvases from the grid (without destroying)
    for canvas in self.canvas_pool:
        self._grid_layout.removeWidget(canvas)
        canvas.hide()

    # 2. Update state
    self.current_rows = rows
    self.current_cols = cols
    N = rows * cols

    # 3. Place canvases into the grid
    for i in range(N):
        canvas = self.canvas_pool[i]
        row = i // cols
        col = i % cols
        self._grid_layout.addWidget(canvas, row, col)
        canvas.show()

    # 4. Emit capacity change so the conveyor belt can adjust
    self.matrixCapacityChanged.emit(N)

    # 5. Re-feed current data (Phase 3 will hook into this)
    self._refresh_visible_canvases()
```

### Step 4: The Layout Chooser UI

Add a `QComboBox` to the toolbar for manual layout selection.

```python
def create_top_toolbar(self) -> QToolBar:
    toolbar = QToolBar("Context Matrix Tools")
    toolbar.setMovable(False)

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(5, 2, 5, 2)

    # Layout chooser
    self._layout_combo = QComboBox()
    self._layout_combo.addItems([
        "1 View",           # 1×1
        "Side-by-Side",     # 1×2
        "Stacked",          # 2×1
        "2×2 Grid",         # 2×2
        "Horizontal Strip",  # 1×3
        "Vertical Column",   # 3×1
    ])
    self._layout_map = {
        0: (1, 1), 1: (1, 2), 2: (2, 1),
        3: (2, 2), 4: (1, 3), 5: (3, 1),
    }
    self._layout_combo.currentIndexChanged.connect(self._on_layout_chosen)
    layout.addWidget(QLabel("Layout:"))
    layout.addWidget(self._layout_combo)

    # Sync toggle (placeholder for Phase 5)
    self._sync_btn = QToolButton()
    self._sync_btn.setCheckable(True)
    self._sync_btn.setChecked(True)
    self._sync_btn.setText("🔗")
    self._sync_btn.setToolTip("Target-Lock Sync (enabled)")
    self._sync_btn.toggled.connect(self._on_sync_toggled)
    layout.addWidget(self._sync_btn)

    # Rank indicator label (Phase 3)
    self._rank_label = QLabel("—")
    self._rank_label.setStyleSheet("color: #888; font-size: 11px;")
    layout.addStretch(1)
    layout.addWidget(self._rank_label)

    toolbar.addWidget(container)
    return toolbar

def _on_layout_chosen(self, index):
    rows, cols = self._layout_map.get(index, (1, 1))
    self.auto_flow_enabled = False  # Manual override disables auto-flow
    self._rebuild_layout(rows, cols)
```

### Step 5: Auto-Flow Reactivity (Dock Placement)

When the `ContextMatrixWidget` dock is moved to different areas, auto-adjust the layout.

**Problem:** PyQtAds (`CDockWidget`) doesn't directly emit `dockLocationChanged` like Qt's native `QDockWidget`. 

**Solution:** Use the `DockWrapper`'s `visibilityChanged` or a `resizeEvent` heuristic:

```python
def resizeEvent(self, event):
    """Auto-adjust layout based on aspect ratio when auto-flow is enabled."""
    super().resizeEvent(event)
    if not self.auto_flow_enabled:
        return

    w = self.width()
    h = self.height()
    if w == 0 or h == 0:
        return

    N = self.current_rows * self.current_cols
    aspect = w / h

    if aspect > 2.0:
        # Very wide → horizontal strip
        self._rebuild_layout(1, min(N, 3))
    elif aspect < 0.5:
        # Very tall → vertical column
        self._rebuild_layout(min(N, 3), 1)
    else:
        # Roughly square → grid
        import math
        cols = max(1, round(math.sqrt(N * aspect)))
        rows = max(1, math.ceil(N / cols))
        self._rebuild_layout(rows, cols)
```

> This is more robust than hooking dock location signals because PyQtAds can place docks in tabs, splits, and floating states that don't map cleanly to Qt dock areas.

### Step 6: Wiring the Data Feed

The widget needs to receive camera image paths and display them.

```python
def set_raster_manager(self, raster_manager):
    """Wire the RasterManager for fetching image data."""
    self._raster_manager = raster_manager

def _get_visible_capacity(self) -> int:
    """Return the number of currently visible canvas slots."""
    return self.current_rows * self.current_cols

def update_context_cameras(self, camera_paths: List[str]):
    """Load images into the visible canvases from the given paths.

    Args:
        camera_paths: Ordered list of image_paths to display.
                      Length may be <= capacity; excess canvases show placeholder.
    """
    capacity = self._get_visible_capacity()

    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if i < len(camera_paths):
            path = camera_paths[i]
            if canvas.current_image_path == path:
                continue  # Already showing this image, skip reload
            raster = self._raster_manager.get_raster(path) if self._raster_manager else None
            if raster:
                q_image = raster.get_qimage()
                if q_image and not q_image.isNull():
                    canvas.load_visuals(q_image, path, raster=raster)
                    continue
            canvas.clear_scene()
        else:
            canvas.clear_scene()

def _refresh_visible_canvases(self):
    """Repaint canvases based on current conveyor state."""
    capacity = self._get_visible_capacity()
    if self.ordered_context_paths:
        target_paths = self.ordered_context_paths[
            self.current_rank_offset : self.current_rank_offset + capacity
        ]
        self.update_context_cameras(target_paths)
```

### Step 7: The "Promote to Main" Interaction

Double-clicking a context canvas swaps it into the main AnnotationWindow.

```python
def _on_canvas_double_clicked(self, canvas):
    """Promote a context canvas to the main AnnotationWindow."""
    if canvas.current_image_path:
        self.canvasPromoted.emit(canvas.current_image_path)
```

In `MVATManager`, connect:
```python
self.context_matrix.canvasPromoted.connect(self._on_camera_selected)
```

This reuses the existing `_on_camera_selected` which calls `selection_model.set_active(path)` and loads the image.

### Step 8: Mount into MainWindow

In `QtMainWindow.py`:

```python
# After existing camera_grid creation:
from coralnet_toolbox.MVAT.ui.QtContextMatrix import ContextMatrixWidget

self.context_matrix = ContextMatrixWidget(max_canvases=9, parent=None)
self.context_matrix.set_raster_manager(self.image_window.raster_manager)

# Create dock
self.context_dock = DockWrapper("Context", "ContextDock", self.context_matrix, self)
if hasattr(self.context_matrix, 'create_top_toolbar'):
    self.context_dock.add_toolbar(self.context_matrix.create_top_toolbar())

# Place it — e.g., tabbed with the Grid dock, or below it
context_area = self.dock_manager.addDockWidget(
    ads.CenterDockWidgetArea, self.context_dock, grid_area
)
```

Wire to `MVATManager`:
```python
# In MVATManager.__init__ or a new wiring method:
self.context_matrix = main_window.context_matrix
self.context_matrix.set_raster_manager(self.raster_manager)

# When active camera changes, feed the context matrix:
# (Extend _on_active_camera_changed)
```

### Step 9: Feed Context Matrix from MVATManager

Extend `MVATManager._reorder_cameras()` to also feed the Context Matrix:

```python
def _reorder_cameras(self, reference_path, hide_distant_cameras=True):
    # ... existing proximity scoring and sorting ...
    ordered_paths = [p for p, s in camera_scores]

    # Feed CameraGrid (existing)
    self.camera_grid.set_camera_order(ordered_paths)

    # Feed ContextMatrixWidget (NEW)
    if hasattr(self, 'context_matrix') and self.context_matrix:
        # Filter out the active camera
        context_paths = [p for p in ordered_paths if p != reference_path]
        self.context_matrix.set_camera_order(context_paths, reference_path)
```

Add to `ContextMatrixWidget`:
```python
def set_camera_order(self, ordered_paths: List[str], active_path: str):
    """Set the ordered list of context cameras, excluding active."""
    self.ordered_context_paths = [p for p in ordered_paths if p != active_path]
    self.current_rank_offset = 0
    self._refresh_visible_canvases()
    self._update_rank_label()
```

---

## Edge Cases & Risks

- **Memory pressure:** Loading N full-resolution QImages simultaneously. Mitigation: `BaseCanvas.load_visuals()` can use the thumbnail proxy initially and swap to full-res on zoom (Phase 5's "Performance on Demand"). For Phase 2, load full-res directly since N ≤ 9.

- **Layout rebuild during image load:** If `_rebuild_layout` is called while `update_context_cameras` is running, canvases may be hidden mid-load. Mitigation: `_rebuild_layout` calls `_refresh_visible_canvases()` at the end, which naturally corrects any stale state.

- **No cameras loaded:** If `load_cameras()` hasn't been called, `ordered_context_paths` is empty. All canvases show placeholder. This is correct behavior.

- **PyQtAds tab resizing:** When the Context dock is tabbed with another dock, `resizeEvent` auto-flow may trigger on tab switch. Mitigation: debounce `resizeEvent` with a 100ms timer.

- **DockWrapper toolbar:** The `create_top_toolbar()` hook is called by `DockWrapper.add_toolbar()` during MainWindow setup. The toolbar must be fully constructed before the widget is shown.

---

## Validation Criteria

After Phase 2 is complete:

1. A new "Context" dock appears in the application with a layout chooser dropdown.
2. Selecting "2×2 Grid" shows 4 empty BaseCanvas viewports.
3. After loading cameras and selecting one, the 4 viewports populate with the 4 nearest neighbors.
4. Each viewport responds to mouse wheel zoom and middle-click pan independently.
5. Double-clicking a context viewport loads that image into the main AnnotationWindow.
6. The layout adapts when the dock is resized (if auto-flow is enabled).
7. The existing CameraGrid continues to function identically.
