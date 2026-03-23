Here is the detailed execution plan for the final piece of the architecture, **Phase 6: Annotation Visualization in Context Views**.

This phase transforms the context views from passive image displays into active, data-rich inspection tools. It covers two distinct capabilities: displaying the native 2D annotations that already belong to the context images, and projecting "ghost" shapes of what you are actively drawing in the main window.

### Step 1: The Central `AnnotationManager` (The Source of Truth)

Currently, `AnnotationWindow` holds the master dictionary of all annotations and the Undo/Redo stack. We must extract this into a dedicated `AnnotationManager` so that any canvas can access the data without routing through the UI.

**1. Data Migration:**

* Move `self.annotations_dict`, `self.image_annotations_dict`, and `self.selected_annotations` into `AnnotationManager`.
* Move the `ActionStack` (Undo/Redo logic) here as well.

**2. The Core API:**

* **Getters:** `get_image_annotations(image_path)` and `get_selected_annotations()`.
* **Setters:** `add_annotation()`, `delete_annotation()`, `update_annotation_geometry()`.
* **Signals:** Define crucial global signals: `annotationAdded`, `annotationRemoved`, `annotationModified`, and `selectionChanged`.

**3. Rewiring the Flow:**
When the user finishes drawing a polygon in the main `AnnotationWindow`, the tool no longer adds it directly to the local scene. Instead, it calls `annotation_manager.add_annotation(...)`. The manager stores it and fires the `annotationAdded` signal.

### Step 2: Native Read-Only Rendering in `BaseCanvas`

As the conveyor belt slides a new camera into the matrix, that specific image might already have annotations drawn on it from previous sessions. The `BaseCanvas` needs to display them automatically.

**1. The Fetch-and-Draw Cycle:**

* When `BaseCanvas` is assigned a new `image_path` by the conveyor belt, it calls `annotation_manager.get_image_annotations(image_path)`.
* It loops through the returned list and creates visual representations (e.g., `QGraphicsPolygonItem`, `QGraphicsRectItem`) to add to its own local `QGraphicsScene`.

**2. Enforcing "Read-Only" State:**

* To prevent the context views from intercepting mouse clicks and breaking the pan/zoom mechanics, these graphical items must be strictly visual.
* When the `BaseCanvas` creates these items, it must explicitly set `ItemIsSelectable = False`, `ItemIsMovable = False`, and `ItemIsFocusable = False`. They become part of the background scenery.

**3. Live Updating:**

* If a bulk action (like changing a label's color or hiding a label class) occurs, the `AnnotationManager` fires `annotationModified`.
* The `ContextMatrixWidget` listens for this, checks if the affected annotation belongs to any of its currently visible canvases, and tells that specific `BaseCanvas` to redraw the shape.

### Step 3: The "Ghost" Projection Engine (Cross-View Drawing)

This is the ultimate "power user" feature. If you are drawing a complex polygon around a coral head in the main window, you want to see those lines wrapping around that same coral head in the 3D context views in real-time to ensure accuracy.

**1. Capturing the Active Shape:**

* As the user clicks to add vertices in the main `AnnotationWindow`, the active drawing tool maintains a list of temporary 2D pixels.
* Whenever a new vertex is added (or as the mouse hovers to place the next vertex), emit a `drawingUpdated(list_of_2d_points)` signal.

**2. The Spatial Translation (2D -> 3D -> 2D):**

* Your central Sync Controller (or `MousePositionBridge`) intercepts this list of points.
* It looks up the Z-depth for *each* 2D point in the main image and casts rays to find their 3D world coordinates.
* It then projects those 3D coordinates into the $N$ visible context cameras, generating a new list of 2D pixels for each context view.

**3. Drawing the Ghost Shape:**

* The `ContextMatrixWidget` passes these projected coordinate lists to the respective `BaseCanvas` instances.
* `BaseCanvas` maintains a special, high-Z-value `QGraphicsPolygonItem` or `QGraphicsPathItem` just for this purpose.
* It styles this item distinctly—for example, a semi-transparent dashed white line with no fill—so the user knows it is an active, uncommitted projection.
* When the user finishes the drawing and commits the annotation, the `drawingUpdated` signal fires with an empty list, and the context canvases instantly clear their ghost shapes.

### Step 4: Cross-View Selection Highlighting

While the context views are read-only (you cannot click an annotation inside them to select it), they should still reflect what is selected globally.

**The Logic:**

* When the user clicks an annotation in the main `AnnotationWindow` (or selects one via the gallery/explorer), the `AnnotationManager` emits `selectionChanged`.
* The `ContextMatrixWidget` tells all active canvases to update their rendering.
* If a `BaseCanvas` is displaying an annotation that is in the global `selected` list, it highlights it (e.g., changes the border to the bright lime-green `SELECT_COLOR` and increases the stroke width). This allows the user to see exactly how their selected object looks from 3 additional angles instantly.

---

### Summary of the Final Architecture

With Phase 6 complete, the architecture is a perfectly decoupled MVC (Model-View-Controller) loop:

1. **The Model:** `AnnotationManager` holds all the data.
2. **The Main View/Controller:** `AnnotationWindow` allows drawing, edits, and sends commands to the Model.
3. **The Context Views:** `BaseCanvas` instances live in the `ContextMatrixWidget`. They listen to the Model, draw native data passively, and project active math from the 3D sync engine.

This concludes the architectural roadmap. Because Phase 1 (`BaseCanvas` extraction) and Phase 6 (`AnnotationManager` decoupling) are entirely about structural refactoring rather than new UI, they can actually be tackled simultaneously.