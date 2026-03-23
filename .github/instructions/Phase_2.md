Excellent. Now that we have conceptually isolated the `BaseCanvas` as a lightweight, independent viewing engine, we can build the housing for it.

Here is the detailed execution plan for **Phase 2: Building the "Security Camera" Matrix**.

The goal of this phase is to completely replace the scrolling list of thumbnails (`CameraGrid`) with a new `ContextMatrixWidget` that rigidly manages a grid of interactive canvases.

### Step 1: Create the `ContextMatrixWidget` Skeleton

Create a new file (e.g., `QtContextMatrix.py`). This widget will act as the master controller for the context views.

**Core Attributes:**

* `self.layout`: A `QGridLayout` with zero margins (`setContentsMargins(0,0,0,0)`) to maximize screen real estate.
* `self.canvas_pool`: A list storing instances of `BaseCanvas`.
* `self.current_rows` and `self.current_cols`: Integers tracking the active layout.

### Step 2: The Canvas Object Pool (Optimization)

Instantiating and destroying `QGraphicsView` objects on the fly is computationally expensive and causes UI flickering. We will use an "Object Pool" pattern.

**How it works:**

* On initialization, `ContextMatrixWidget` pre-generates a maximum number of `BaseCanvas` instances (e.g., 9, to support up to a $3\times3$ grid).
* It stores these in `self.canvas_pool`.
* By default, all of them are set to `hide()`.
* When a layout is selected, the widget simply grabs the required number of canvases from the pool, slots them into the `QGridLayout`, and calls `show()`. The rest remain hidden in memory.

### Step 3: Implement the Layout Engine (`_rebuild_layout`)

Create a method `_rebuild_layout(rows, cols)` that handles the physical arrangement of the grid.

**The Logic:**

1. Loop through all canvases in the pool and remove them from the `QGridLayout` (without deleting the objects).
2. Hide all canvases.
3. Calculate $N$ (`rows * cols`).
4. Loop from $0$ to $N-1$:
* Grab `self.canvas_pool[i]`.
* Calculate its grid position: `row = i // cols`, `col = i % cols`.
* Add it to the layout: `self.layout.addWidget(canvas, row, col)`.
* Call `canvas.show()`.


5. Emit a signal: `matrixCapacityChanged(N)` so the `MVATManager` knows how many cameras it needs to feed this widget.

### Step 4: The Auto-Flow Reactivity (Dock Placement)

We want the grid to feel "smart" by automatically adjusting to its container shape.

**The Implementation:**

* In `QtMainWindow.py`, right after you create the `DockWrapper` for the context matrix, connect to the dock's location signal:
`self.context_matrix_dock.dockLocationChanged.connect(self.context_matrix_widget.handle_dock_location_changed)`
* Inside `ContextMatrixWidget`, define `handle_dock_location_changed(area)`:
* If `area` is `Qt.LeftDockWidgetArea` or `Qt.RightDockWidgetArea`: Call `_rebuild_layout(rows=3, cols=1)` (Tall and skinny).
* If `area` is `Qt.TopDockWidgetArea` or `Qt.BottomDockWidgetArea`: Call `_rebuild_layout(rows=1, cols=3)` (Short and wide).
* If `area` is `Qt.NoDockWidgetArea` (Floating): Call `_rebuild_layout(rows=2, cols=2)` (Box).



### Step 5: The Grid Chooser UI (Manual Override)

For power users who want to override the Auto-Flow, add a layout selector to the widget's top toolbar.

* Add a `QComboBox` (or a set of icon-based `QAction`s) to `create_top_toolbar()`.
* Populate it with options: "1 View", "Side-by-Side (1x2)", "Stacked (2x1)", "Grid (2x2)", "Strip (1x3)", "Column (3x1)".
* Connect this dropdown to `_rebuild_layout` with the corresponding row/col values.
* *Note:* When the user manually selects a layout from this dropdown, set a flag `self.auto_flow_enabled = False` so the dock placement stops overriding their manual choice.

### Step 6: Wiring the Data Feed

The widget needs a method to receive the target cameras and paint them onto the active canvases.

**Define `update_context_cameras(camera_list)`:**

1. The `camera_list` will be provided by the Conveyor Belt Engine (Phase 3).
2. Iterate through the currently visible canvases in the layout.
3. For each visible canvas, grab the corresponding camera from the list (e.g., `canvas_pool[0]` gets `camera_list[0]`).
4. If a camera exists for that slot, fetch its raster data from the `RasterManager` and call `canvas.load_visuals(q_image, image_path, z_data)`.
5. If the `camera_list` is smaller than the number of visible canvases (e.g., end of the dataset), call `canvas.clear_scene()` on the empty slots so they show the placeholder.

### Step 7: Swap it into `MainWindow`

Finally, sever the old `CameraGrid` and mount the new `ContextMatrixWidget`.

1. In `QtMainWindow.py`, replace `self.camera_grid = CameraGrid(...)` with `self.context_matrix = ContextMatrixWidget(...)`.
2. Update the DockWrapper to mount `self.context_matrix`.
3. In `MVATManager`, remove the old signals that updated the green/cyan borders of the thumbnails, as those concepts no longer apply to the new security camera layout.

### Checkpoint / Validation

By the end of Phase 2, you will be able to drag the context dock to the right side of the screen and watch it instantly snap into a vertical column of 3 empty canvases. If you dock it to the bottom, it snaps into a horizontal row. If you feed it 3 image paths, it displays them as fully navigable, zoomable `BaseCanvas` viewports.

Shall we move on to detailing **Phase 3 (The Conveyor Belt Navigation Engine)** to feed data into these slots, or **Phase 4 (Implementing the Dual-Marker System)** to get the mouse projections working?