Here is the detailed execution plan for **Phase 4: Implementing the Dual-Marker System**.

The goal of this phase is to provide the user with absolute spatial awareness. When they move their mouse or set a focal target in the main window, they need to see exactly where that physical 3D point lands in all of their context views simultaneously.

Because `BaseCanvas` is a clean, isolated viewport, it should be entirely "dumb" regarding 3D math. It will simply receive $X, Y$ pixel coordinates from a central manager and draw shapes.

### Step 1: Initialize the Marker Graphics in `BaseCanvas`

We need to set up the graphical items that will represent the markers. They must be created once per canvas and kept in memory to ensure high performance (no creating/destroying items on mouse move).

**The Logic in `BaseCanvas.__init__`:**

1. **The Static Focal Marker (Crosshair):**
* Create a `QGraphicsPathItem`. Draw a neat crosshair path (e.g., a vertical line and a horizontal line intersecting at 0,0).
* Set the pen color (e.g., Orange/Yellow) and width.
* **Crucial Flag:** Call `self.static_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)`. This ensures that if the user zooms in 500x, the crosshair stays a crisp 2 pixels wide on their screen rather than blowing up to cover the whole image.
* Set a high Z-value (e.g., `setZValue(100)`) so it renders above all image data and annotations.
* Add it to the scene and `hide()` it by default.


2. **The Dynamic Hover Marker (Circle):**
* Create a `QGraphicsEllipseItem` (e.g., a circle centered at 0,0 with a radius of 5 pixels).
* Apply the same `ItemIgnoresTransformations` flag, high Z-value, and add to the scene hidden.



### Step 2: The Canvas API for Marker Control

Create simple setter methods in `BaseCanvas` so the central manager can move the markers.

* `update_static_marker(x, y, color=None)`: Moves the crosshair to `(x, y)` and calls `show()`. If the coordinate is outside the image rect, it hides the marker.
* `clear_static_marker()`: Hides the crosshair.
* `update_dynamic_marker(x, y, color, is_valid)`: Moves the circle to `(x, y)`. If `is_valid` is False (meaning the point is occluded by 3D geometry from this camera's angle), you can change the pen style to a dashed line or red color.
* `clear_dynamic_marker()`: Hides the circle.

### Step 3: Routing the Static Focal Marker

This marker represents a fixed 3D point of interest (usually set by double-clicking in the 3D viewer or the main image).

**The Pipeline:**

1. The user double-clicks. The `MVATViewer` records a 3D `focal_point` and emits a `focalPointChanged(x_3d, y_3d, z_3d)` signal.
2. Your `MVATManager` receives this signal.
3. The manager loops through the *currently visible* cameras in the `ContextMatrixWidget`.
4. For each context camera, it uses the camera's projection matrix ($K, R, t$) to project the 3D `focal_point` into 2D pixel coordinates $(u, v)$.
5. The manager calls `context_matrix.update_static_markers(camera_pixel_dict)`.
6. The widget delegates the specific $(u, v)$ coordinates to the matching `BaseCanvas` instances.

### Step 4: Upgrading the `MousePositionBridge` (The Dynamic Marker)

You already have a `MousePositionBridge` in your code! We just need to rewire it to talk to the new `ContextMatrixWidget` instead of the old `CameraGrid`.

**The High-Speed Math Pipeline:**

1. The user moves their mouse over the main `AnnotationWindow`. It emits `mouseMoved(main_x, main_y)`.
2. The `MousePositionBridge` throttles the signal slightly (to prevent UI lag) and calculates the 3D world point:
* It looks up the depth (Z-value) of the primary image at `(main_x, main_y)`.
* It casts a `CameraRay` using the primary camera's parameters to find the exact 3D intersection point.


3. The bridge loops through the cameras currently assigned to the `ContextMatrixWidget`.
4. For each context camera, it projects that 3D point into the context camera's 2D pixel space.
5. **Occlusion Check:** It compares the distance of the 3D point to the context camera against the context camera's Z-buffer. If the ray is blocked by a coral head, it marks the projection as "Occluded".
6. It sends a dictionary of `{canvas_index: (proj_x, proj_y, occluded_flag)}` to the `ContextMatrixWidget`.
7. The widget calls `update_dynamic_marker` on each canvas.

### Step 5: Handling Mouse Leave Events cleanly

If the user moves their mouse *off* the main `AnnotationWindow` to click a button in a toolbar, the dynamic circle markers in the context views must vanish, otherwise, they look like ghost artifacts.

* In `BaseCanvas`, override `leaveEvent(event)` to emit a `mouseLeft()` signal.
* When the main `AnnotationWindow` emits `mouseLeft()`, the `MousePositionBridge` catches it and tells the `ContextMatrixWidget` to loop through all its canvases and call `clear_dynamic_marker()`.

### Summary of the Magic

Because you extracted `BaseCanvas` first (Phase 1), it handles zooming and panning seamlessly. Because the markers ignore transformations (Step 1 of Phase 4), they always look perfect.

When the user zooms into the main image, the context views will zoom in (via the Target-Lock we'll build in Phase 5). As the user moves the mouse around a zoomed-in rock in the main view, the green circle will dance across the same rock in the three context views, mathematically locked to the 3D world, regardless of how the user has panned or scaled the individual canvases.

Shall we move to the final piece of the puzzle, **Phase 5: The "Target-Locked" Sync Engine**, to tie the navigation and marker tracking together?