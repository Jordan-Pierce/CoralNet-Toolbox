Here is the highly detailed execution plan for **Phase 1: The Core Refactor (Extracting the `BaseCanvas`)**.

The goal of this phase is to cleanly slice the visual rendering and basic navigation out of `AnnotationWindow` so we have a lightweight, reusable viewport class (`BaseCanvas`).

### Step 1: Create the `BaseCanvas` Class Skeleton

We will create a new file (e.g., `QtBaseCanvas.py`) to house this class. It will inherit directly from `QGraphicsView`.

**Core Attributes to Move/Create:**

* `self.scene` (The `QGraphicsScene`)
* `self.pixmap_image` & `self.current_image_path` (The active raster)
* **Z-Channel Attributes:** Move the entire Z-channel visualization suite (`self.z_item`, `self.z_data_raw`, etc.) here, as context views will also need to render DEM/depth data correctly.
* **Placeholder Logic:** Move the "No image loaded" placeholder logic here.

**Core Signals to Define:**

* `viewNavigated(center_x, center_y, zoom_factor)`: Fired whenever the user pans or zooms. (This is the critical trigger for Phase 5's Target-Lock sync).
* `mouseHovered(x, y)`: Fired when the mouse moves over the canvas.

### Step 2: Build the Native Navigation Engine

Currently, `AnnotationWindow` uses external tool classes (`PanTool`, `ZoomTool`) to navigate. `BaseCanvas` needs to be entirely self-sufficient so it doesn't rely on the complex tool architecture.

**Native Event Overrides in `BaseCanvas`:**

* **`wheelEvent(event)`:** Implement native zoom-to-cursor math. Calculate the scene position before zoom, scale the view, and adjust the scrollbars so the pixel under the mouse stays under the mouse. Emit `viewNavigated`.
* **`mousePressEvent(event)`:** If `MiddleButton` or `RightButton` is pressed, record `self._pan_start_pos` and set the cursor to a grabbing hand.
* **`mouseMoveEvent(event)`:** * If panning, calculate the delta from `self._pan_start_pos` and adjust the view's scrollbars.
* Always map the cursor to scene coordinates and emit `mouseHovered(x, y)`.


* **`mouseReleaseEvent(event)`:** If panning, clear `self._pan_start_pos`, restore the cursor, and emit `viewNavigated`.

### Step 3: Migrate Image and Render Logic

Move the heavy lifting of displaying pixels from `AnnotationWindow` down to the canvas.

**Methods to Migrate:**

* `clear_scene()`: Wipes the scene, resets Z-channel, and shows the placeholder.
* `_load_z_channel_visualization()` and `update_z_colormap()`: Move these entirely. The canvas should be responsible for its own visual layers.
* **Create `load_visuals(q_image, image_path, z_channel_data=None)`:** This is the new entry point. It sets the `QPixmap`, initializes the Z-channel, and fits the view.

### Step 4: Add the Viewport Control API

To support the "Target-Locked" sync and Conveyor Belt features later, `BaseCanvas` needs an API that allows an external manager to command its camera.

**Methods to Add:**

* `center_on_pixel(x, y)`: Centers the `QGraphicsView` scrollbars exactly on the given image coordinate.
* `set_zoom_level(factor)`: Sets the transformation matrix scale to match a specific zoom tier.
* `fit_to_image()`: Resets the zoom to show the entire image (handling letterboxing naturally).

### Step 5: Implement the Dual-Marker System

`BaseCanvas` will own two specific `QGraphicsItem` objects for spatial correspondence.

**1. The Static Focal Marker (The Crosshair):**

* Create a `QGraphicsPathItem` shaped like a crosshair.
* Method: `set_static_marker(x, y, color)`. Moves the crosshair to the pixel, ensures it stays visually the same size regardless of zoom level (using `ItemIgnoresTransformations`), and makes it visible.

**2. The Dynamic Hover Marker (The Circle):**

* Create a `QGraphicsEllipseItem`.
* Method: `set_dynamic_marker(x, y, color)`. Moves the circle to the pixel coordinate.
* Method: `clear_dynamic_marker()`. Hides it when the user's mouse leaves the main window.

### Step 6: Define Read-Only Annotation Rendering

While Phase 6 handles the full decoupling of the annotation dictionary, we need to prep `BaseCanvas` to draw annotations passively.

**Method to Add:**

* `render_readonly_annotations(annotation_data_list)`: Takes a list of raw geometric data (polygons, rects, points) and draws standard `QGraphicsPolygonItem`s and `QGraphicsRectItem`s.
* *Crucial detail:* These items must have their `ItemIsSelectable` and `ItemIsMovable` flags set to `False`. They are strictly for visualization in the context views.

### Step 7: Gut and Rewire `AnnotationWindow`

Now that `BaseCanvas` is fully capable of showing an image, panning, zooming, and drawing markers, we gut `AnnotationWindow`.

1. **Inheritance:** Change `class AnnotationWindow(QGraphicsView):` to `class AnnotationWindow(BaseCanvas):`.
2. **Delete Redundant Code:** Remove `clear_scene`, `viewportToScene`, `get_image_rect`, the Z-channel visualizers, and the placeholder label logic from `AnnotationWindow`.
3. **Rewire Event Handlers:** * In `AnnotationWindow.mousePressEvent`, check the button. If it's a `LeftButton`, route it to the active drawing tool (e.g., `self.tools[self.selected_tool]`).
* If it is a `MiddleButton` or `RightButton`, simply call `super().mousePressEvent(event)` to let `BaseCanvas` handle the panning.
* Do the same for `mouseMoveEvent` and `wheelEvent` (route tool-specific modifiers to the tool, otherwise pass to `super()`).


4. **Rewire Image Loading:** Update `AnnotationWindow.set_image(image_path)`. It should still handle the `raster_manager` lookup and progress bars, but instead of manually building `QGraphicsPixmapItem`s, it simply calls `self.load_visuals(q_image, image_path, z_data)`.

### Checkpoint / Validation

At the end of Phase 1, the software should look and behave *exactly* as it does right now. However, under the hood, you will have a pristine, isolated `BaseCanvas` class that can be safely instantiated 10 times in a grid without dragging along the heavy baggage of Action Stacks, Tool states, or massive annotation dictionaries.

Shall we move on to detailing **Phase 2: Building the "Security Camera" Matrix**, or do you need any clarification on the exact routing of Qt events in Phase 1?