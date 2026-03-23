# Phase 6: Annotation Visualization in Context Views

## Summary

Phase 6 makes the context canvases data-rich by displaying native annotations on context images and projecting "ghost" annotations drawn in the main window into context views in real-time. This requires extracting the annotation data store from `AnnotationWindow` into a shared `AnnotationManager` so that any `BaseCanvas` can query annotations without routing through the monolithic UI class.

This is the most architecturally significant phase because it decouples the data model (annotations) from the view (AnnotationWindow), establishing a clean MVC separation that benefits the entire application beyond just the context views.

---

## Current Architecture (What Exists)

### Annotation Storage (Inside AnnotationWindow)
```python
# AnnotationWindow attributes:
self.annotations_dict = {}           # {uuid: Annotation} — ALL annotations
self.image_annotations_dict = {}     # {image_path: [Annotation]} — per-image index
self.selected_annotations = []       # Currently selected annotations
self.action_stack = ActionStack()    # Undo/Redo
```

### Annotation Lifecycle
1. **Creation:** Tool (e.g., `PolygonTool`) creates `PolygonAnnotation` → calls `AnnotationWindow.add_annotation()` → pushes `AddAnnotationAction` to `action_stack` → stores in both dicts → calls `annotation.create_graphics_item(self.scene)` → emits `annotationCreated`.

2. **Loading:** `AnnotationWindow.load_annotations()` → loops `image_annotations_dict[current_image_path]` → calls `annotation.create_graphics_item(self.scene)` for each.

3. **Display:** `Annotation.create_graphics_item(scene)` creates a `QGraphicsPolygonItem`/`QGraphicsRectItem`/etc. and adds it to the scene.

4. **Selection:** `AnnotationWindow.select_annotation(annotation)` → `annotation.select()` → changes pen to selection style → emits signals.

5. **Deletion:** `AnnotationWindow.delete_annotation()` → pushes `DeleteAnnotationAction` → removes from dicts → removes graphics item from scene.

### Annotation Graphics Items
- Each `Annotation` subclass has a `create_graphics_item(scene)` method that adds to the scene.
- Items are interactive: `ItemIsSelectable=True`, `ItemIsMovable=True` (for select tool).
- Items reference back to the `Annotation` object.

### Key Signals Already Emitted
```python
annotationCreated = pyqtSignal(str)      # annotation_id
annotationsCreated = pyqtSignal(list)    # list of annotation_ids
annotationDeleted = pyqtSignal(str)      # annotation_id
annotationsDeleted = pyqtSignal(list)    # list of annotation_ids
annotationModified = pyqtSignal(str)     # annotation_id
annotationLabelChanged = pyqtSignal(str, str)  # annotation_id, new_label
annotationSelectionChanged = pyqtSignal(object)  # list of IDs
```

### Raster-Based Annotation Index
Each `Raster` object also tracks its annotations:
```python
raster.annotations: List           # Store the actual annotations
raster.annotation_count: int
raster.label_counts: dict
```

---

## Target Architecture

### Central `AnnotationManager` (New)
```
AnnotationManager (QObject)
  ├── annotations_dict: {uuid: Annotation}
  ├── image_annotations_dict: {image_path: [Annotation]}
  ├── selected_annotations: [Annotation]
  ├── action_stack: ActionStack
  ├── Signals: annotationAdded, annotationRemoved, annotationModified, selectionChanged
  └── API: get_image_annotations(path), add_annotation(), delete_annotation(), etc.
```

### View Roles
```
AnnotationWindow (BaseCanvas)
  └── PRIMARY VIEW: Create/Edit/Delete annotations via tools
      └── Routes all mutation commands through AnnotationManager
      └── Holds interactive QGraphicsItems (selectable, movable)

BaseCanvas (Context Views)
  └── READ-ONLY VIEW: Display annotations passively
      └── Queries AnnotationManager for annotation data
      └── Holds non-interactive QGraphicsItems (no selection/movement)
      └── Listens to AnnotationManager signals for live updates
```

---

## Implementation Steps

### Step 1: Create `AnnotationManager` Class

**File:** `coralnet_toolbox/QtAnnotationManager.py`

This is **not** a massive rewrite. It's a data accessor that wraps the existing dictionaries, moving them out of `AnnotationWindow`.

```python
class AnnotationManager(QObject):
    """Central data store for all annotations across all images.

    Decouples annotation data from the AnnotationWindow UI so that
    multiple views (context canvases, explorer) can access annotations
    without routing through the main annotation window.
    """

    # Signals (mirrors AnnotationWindow signals for backward compat)
    annotationAdded = pyqtSignal(str)            # annotation_id
    annotationsAdded = pyqtSignal(list)          # [annotation_ids]
    annotationRemoved = pyqtSignal(str)          # annotation_id
    annotationsRemoved = pyqtSignal(list)        # [annotation_ids]
    annotationModified = pyqtSignal(str)         # annotation_id
    annotationLabelChanged = pyqtSignal(str, str)  # annotation_id, new_label
    selectionChanged = pyqtSignal(object)        # list of annotation IDs

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations_dict = {}           # {uuid: Annotation}
        self.image_annotations_dict = {}     # {image_path: [Annotation]}
        self.selected_annotations = []
        self.action_stack = ActionStack()

    # --- Getters ---

    def get_annotation(self, annotation_id: str):
        return self.annotations_dict.get(annotation_id)

    def get_image_annotations(self, image_path: str) -> list:
        return self.image_annotations_dict.get(image_path, [])

    def get_all_annotations(self) -> dict:
        return self.annotations_dict

    def get_selected_annotations(self) -> list:
        return self.selected_annotations

    # --- Mutators ---

    def add_annotation(self, annotation):
        self.annotations_dict[annotation.id] = annotation
        if annotation.image_path not in self.image_annotations_dict:
            self.image_annotations_dict[annotation.image_path] = []
        self.image_annotations_dict[annotation.image_path].append(annotation)
        self.annotationAdded.emit(annotation.id)

    def remove_annotation(self, annotation_id: str):
        annotation = self.annotations_dict.pop(annotation_id, None)
        if annotation:
            img_list = self.image_annotations_dict.get(annotation.image_path, [])
            if annotation in img_list:
                img_list.remove(annotation)
            self.annotationRemoved.emit(annotation_id)

    def modify_annotation(self, annotation_id: str):
        """Emit modification signal for an existing annotation."""
        if annotation_id in self.annotations_dict:
            self.annotationModified.emit(annotation_id)

    # --- Selection ---

    def set_selection(self, annotations: list):
        self.selected_annotations = list(annotations)
        self.selectionChanged.emit([a.id for a in self.selected_annotations])

    def clear_selection(self):
        self.selected_annotations.clear()
        self.selectionChanged.emit([])
```

### Step 2: Wire AnnotationManager into MainWindow

```python
# In MainWindow.__init__(), before AnnotationWindow creation:
from coralnet_toolbox.QtAnnotationManager import AnnotationManager
self.annotation_manager = AnnotationManager(self)

# In AnnotationWindow.__init__():
# Replace local dicts with references to the central manager
self.annotation_manager = self.main_window.annotation_manager
# These become properties/aliases:
@property
def annotations_dict(self):
    return self.annotation_manager.annotations_dict

@property
def image_annotations_dict(self):
    return self.annotation_manager.image_annotations_dict

@property
def selected_annotations(self):
    return self.annotation_manager.selected_annotations

@selected_annotations.setter
def selected_annotations(self, value):
    self.annotation_manager.selected_annotations = value

@property
def action_stack(self):
    return self.annotation_manager.action_stack
```

> **Key insight:** By making these properties/aliases, ALL existing code in AnnotationWindow that reads/writes `self.annotations_dict` continues to work without modification. The migration is transparent.

### Step 3: Forward AnnotationWindow Signals Through Manager

Currently, AnnotationWindow emits `annotationCreated`, `annotationDeleted`, etc. These need to also flow through the manager:

```python
# In AnnotationWindow.__init__ (after annotation_manager is set):
# Bridge existing signals to the central manager
self.annotationCreated.connect(self.annotation_manager.annotationAdded)
self.annotationDeleted.connect(self.annotation_manager.annotationRemoved)
self.annotationModified.connect(self.annotation_manager.annotationModified)
self.annotationLabelChanged.connect(self.annotation_manager.annotationLabelChanged)
self.annotationSelectionChanged.connect(self.annotation_manager.selectionChanged)
```

This means the manager's signals fire whenever the AnnotationWindow's signals fire. External listeners (context canvases, explorer) can subscribe to the manager's signals instead.

### Step 4: Native Read-Only Rendering in Context Canvases

When a `BaseCanvas` loads a new image (via `load_visuals()`), it should display that image's existing annotations as read-only overlays.

**Add to `BaseCanvas`:**

```python
def _render_annotations_readonly(self, annotations: list):
    """Render annotations as non-interactive overlays.

    Args:
        annotations: List of Annotation objects for this canvas's image.
    """
    # Clear any previously rendered read-only annotations
    self._clear_readonly_annotations()

    for annotation in annotations:
        item = self._create_readonly_graphics_item(annotation)
        if item:
            self._readonly_annotation_items.append(item)
            self.scene.addItem(item)

def _clear_readonly_annotations(self):
    """Remove all read-only annotation items from the scene."""
    if not hasattr(self, '_readonly_annotation_items'):
        self._readonly_annotation_items = []
        return
    for item in self._readonly_annotation_items:
        if item.scene() == self.scene:
            self.scene.removeItem(item)
    self._readonly_annotation_items.clear()

def _create_readonly_graphics_item(self, annotation):
    """Create a non-interactive QGraphicsItem from an Annotation object.

    Returns the item without adding it to the scene.
    """
    from coralnet_toolbox.Annotations import (
        PatchAnnotation, PolygonAnnotation, RectangleAnnotation
    )

    color = annotation.label.color
    color.setAlpha(annotation.transparency)
    pen = QPen(color, 2)

    if isinstance(annotation, PatchAnnotation):
        size = annotation.annotation_size
        rect = QRectF(
            annotation.center_xy.x() - size / 2,
            annotation.center_xy.y() - size / 2,
            size, size
        )
        item = QGraphicsRectItem(rect)
        item.setPen(pen)
        item.setBrush(Qt.NoBrush)

    elif isinstance(annotation, RectangleAnnotation):
        rect = annotation.get_bounding_rect()
        item = QGraphicsRectItem(rect)
        item.setPen(pen)
        item.setBrush(Qt.NoBrush)

    elif isinstance(annotation, PolygonAnnotation):
        polygon = annotation.get_polygon()
        item = QGraphicsPolygonItem(polygon)
        item.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(annotation.transparency // 3)
        item.setBrush(QBrush(fill))

    else:
        return None

    # CRITICAL: Make non-interactive
    item.setFlag(QGraphicsItem.ItemIsSelectable, False)
    item.setFlag(QGraphicsItem.ItemIsMovable, False)
    item.setFlag(QGraphicsItem.ItemIsFocusable, False)
    item.setZValue(0)  # Above image (-10), below markers (100)

    return item
```

### Step 5: Trigger Annotation Rendering in Context Matrix

When the conveyor belt loads a new camera image:

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

            raster = self._raster_manager.get_raster(path)
            if raster:
                q_image = raster.get_qimage()
                if q_image and not q_image.isNull():
                    canvas.load_visuals(q_image, path, raster=raster)

                    # Load read-only annotations for this image
                    if self._annotation_manager:
                        annotations = self._annotation_manager.get_image_annotations(path)
                        canvas._render_annotations_readonly(annotations)
                    continue
            canvas.clear_scene()
        else:
            canvas.clear_scene()
```

Wire the manager reference:
```python
# In MainWindow setup or MVATManager:
self.context_matrix._annotation_manager = self.annotation_manager
```

### Step 6: Live Annotation Updates in Context Views

When an annotation is created/deleted/modified in the main window, context views showing the same image should update.

```python
# In ContextMatrixWidget.__init__:
def set_annotation_manager(self, manager):
    self._annotation_manager = manager
    # Subscribe to changes
    manager.annotationAdded.connect(self._on_annotation_changed)
    manager.annotationRemoved.connect(self._on_annotation_changed)
    manager.annotationModified.connect(self._on_annotation_changed)
    manager.annotationLabelChanged.connect(
        lambda aid, _: self._on_annotation_changed(aid)
    )

def _on_annotation_changed(self, annotation_id: str):
    """Refresh read-only annotations on context canvases if affected."""
    if not self._annotation_manager:
        return

    annotation = self._annotation_manager.get_annotation(annotation_id)
    # Annotation may have been deleted
    image_path = annotation.image_path if annotation else None

    capacity = self._get_visible_capacity()
    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if not canvas.isVisible() or not canvas.current_image_path:
            continue

        # If deleted, we need to check all canvases (annotation is gone)
        # If modified, only refresh canvases showing the same image
        if image_path and canvas.current_image_path != image_path:
            continue

        # Full redraw of annotations for this canvas
        annotations = self._annotation_manager.get_image_annotations(
            canvas.current_image_path
        )
        canvas._render_annotations_readonly(annotations)
```

### Step 7: Cross-View Selection Highlighting

When an annotation is selected in the main window, context views should highlight the same annotation.

```python
# In ContextMatrixWidget:
def _on_selection_changed(self, selected_ids: list):
    """Highlight selected annotations in context views."""
    selected_set = set(selected_ids)

    capacity = self._get_visible_capacity()
    for i in range(capacity):
        canvas = self.canvas_pool[i]
        if not canvas.isVisible() or not hasattr(canvas, '_readonly_annotation_items'):
            continue

        for item_idx, item in enumerate(canvas._readonly_annotation_items):
            # Match by annotation reference stored on the item
            annotation = getattr(item, '_source_annotation', None)
            if annotation and annotation.id in selected_set:
                # Highlight: bright pen, thicker stroke
                from coralnet_toolbox.MVAT.core.constants import SELECT_COLOR
                pen = QPen(SELECT_COLOR, 3)
                item.setPen(pen)
            else:
                # Normal style: restore original pen
                if annotation:
                    color = QColor(annotation.label.color)
                    color.setAlpha(annotation.transparency)
                    item.setPen(QPen(color, 2))
```

To make this work, tag each read-only item with its source annotation in `_create_readonly_graphics_item`:
```python
item._source_annotation = annotation  # Store reference for selection matching
```

And connect:
```python
manager.selectionChanged.connect(self._on_selection_changed)
```

### Step 8: Ghost Annotation Projection (Advanced Feature)

This projects the polygon currently being drawn in the main window into context views in real-time. This is complex and should be **deferred to a follow-up iteration** after the basic annotation visualization works.

**Deferred because:**
1. Requires intercepting in-progress vertex lists from active drawing tools.
2. Each vertex needs depth lookup and 3D projection (N vertices × M cameras per frame).
3. The polygon topology (open vs closed, fill vs no-fill) varies by drawing state.
4. Performance is critical (updates on every mouse move during drawing).

**When implemented, the approach would be:**
1. Drawing tools emit `drawingUpdated(list_of_2d_points)` as vertices are added.
2. `MousePositionBridge` or a new `GhostProjectionController` receives the points.
3. For each vertex, unproject to 3D using depth, then project to each context camera.
4. Each `BaseCanvas` maintains a special `QGraphicsPathItem` for the ghost shape.
5. Style: white dashed semi-transparent lines with no fill.
6. On drawing completion, clear ghost shapes (points list becomes empty).

---

## Migration Strategy (Minimizing Risk)

The `AnnotationManager` extraction is a structural refactor with zero visible behavior change. To minimize risk:

1. **Step 1:** Create `AnnotationManager` with the property-alias approach. AnnotationWindow still owns the actual dictionaries through property delegation. All existing code works unchanged.

2. **Step 2:** Gradually move method implementations from AnnotationWindow to AnnotationManager (e.g., `add_annotation()`, `delete_annotation()`). Each migration is independently testable.

3. **Step 3:** Once stable, convert the property aliases to true delegation (AnnotationManager owns the data, AnnotationWindow queries it).

4. **Step 4:** Wire context views to AnnotationManager signals.

This can be done across multiple PRs without breaking existing functionality.

---

## Edge Cases & Risks

- **Performance of full redraw on every change:** When a single annotation's label changes, we redraw ALL annotations on affected canvases. For images with hundreds of annotations, this could be slow. Mitigation: Track annotation→item mapping and update only the changed item. This optimization can be deferred.

- **Annotation graphics item coupling:** Currently, `Annotation.create_graphics_item(scene)` stores `self.graphics_item` on the annotation. Context views cannot use this method because it would overwrite the main scene's item. The read-only renderer uses separate `QGraphics*Item` objects, never touching `annotation.graphics_item`.

- **Mask annotations:** `MaskAnnotation` is image-sized pixel data, not geometric. Displaying it in context views would require creating a `QGraphicsPixmapItem` from the mask data. This is straightforward but heavy for large images. Consider showing mask boundaries only (contour extraction) for context views.

- **Z-ordering:** Read-only annotations at Z=0, image at Z=-10, Z-channel at Z=-5, markers at Z=100. This ensures proper layering.

- **Circular signal loops:** AnnotationWindow emits `annotationCreated` → AnnotationManager emits `annotationAdded` → ContextMatrix refreshes. No loop because context views are read-only and never emit annotation mutation signals.

---

## Validation Criteria

After Phase 6 is complete:

1. Context canvases showing image X display all existing annotations for image X as colored outlines.
2. Creating a new annotation in the main window causes it to appear immediately in any context canvas showing the same image.
3. Deleting an annotation removes it from context views.
4. Changing a label color updates the annotation appearance in context views.
5. Selecting an annotation in the main window highlights it with a bright border in context views.
6. Annotations in context views are strictly non-interactive (cannot be clicked, selected, or moved).
7. The application's existing annotation workflow (create, edit, delete, undo, redo) works identically to before the refactor.

---

## Complete Architecture Summary

With all 6 phases complete:

```
                    AnnotationManager (Data Model)
                    ├── annotations_dict
                    ├── image_annotations_dict
                    ├── action_stack
                    └── Signals: added, removed, modified, selectionChanged
                         │
          ┌──────────────┼──────────────────────┐
          ▼              ▼                       ▼
  AnnotationWindow   ContextMatrixWidget     Explorer/Gallery
  (Primary View)     (Read-Only Views)       (Read-Only View)
  ├── BaseCanvas     ├── N × BaseCanvas      └── QListView
  ├── Tools          ├── Conveyor Belt
  ├── Interactive    ├── Hotkey Nav
  │   Graphics       ├── Target-Lock Sync
  └── Editable       ├── Dual Markers
                     └── Read-Only Graphics

  MVATManager (Controller)
  ├── Camera projection math
  ├── MousePositionBridge → markers
  ├── viewNavigated → sync engine
  └── Proximity scoring → conveyor belt ordering
```
