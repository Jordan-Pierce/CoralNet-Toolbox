import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsItemGroup, QGraphicsPathItem

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Tools.QtMoveSubTool import MoveSubTool
from coralnet_toolbox.Tools.QtResizeSubTool import ResizeSubTool
from coralnet_toolbox.Tools.QtSelectSubTool import SelectSubTool
from coralnet_toolbox.Tools.QtCutSubTool import CutSubTool
from coralnet_toolbox.Tools.QtSubtractSubTool import SubtractSubTool
from coralnet_toolbox.QtActions import AnnotationGeometryEditAction

from coralnet_toolbox.Annotations import (PatchAnnotation, 
                                          PolygonAnnotation, 
                                          RectangleAnnotation,
                                          MultiPolygonAnnotation)
from coralnet_toolbox.QtActions import MergeAnnotationsAction, CutAnnotationAction

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectTool(Tool):
    """
    Tool for selecting annotations and dispatching actions like moving, resizing,
    and cutting to specialized SubTools.

    This class acts as a manager. It handles basic selection and then delegates
    more complex, stateful operations (like a drag-move or a resize) to an
    'active_subtool'.
    """

    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.PointingHandCursor
        self.default_cursor = Qt.ArrowCursor

        # --- SubTool Initialization ---
        self.move_subtool = MoveSubTool(self)
        self.resize_subtool = ResizeSubTool(self)
        self.select_subtool = SelectSubTool(self)
        self.cut_subtool = CutSubTool(self)
        self.subtract_subtool = SubtractSubTool(self)
        
        # --- State for the currently active sub-tool ---
        self.active_subtool: SubTool | None = None

        # --- Hover state for the always-on resize handles ---
        self._hovered_handle = None

        # --- Hover preview: which annotation a click would land on ---
        self._hover_annotation_id = None
        self._hover_item = None
        self._hover_probe_pos = None

        self._connect_signals()

    def _connect_signals(self):
        """Keep the resize handles in step with the selection and the view.

        Handles used to exist only while Ctrl+Shift was held, so every one of
        these signals was wired to tear them down. Now that a single selected
        annotation always carries its handles, the same signals have to rebuild
        them instead -- the selection is the thing that decides whether handles
        exist at all.
        """
        self.annotation_window.annotationSelectionChanged.connect(self._on_selection_changed)
        self.annotation_window.annotationSizeChanged.connect(self._refresh_resize_handles)
        self.annotation_window.annotationDeleted.connect(self._refresh_resize_handles)
        # Handle radii are derived from zoom at paint time; this only keeps the
        # layer's cached scale (and therefore its bounding rect) honest.
        self.annotation_window.viewChanged.connect(self._on_view_changed)

    # --- SubTool Management ---

    def set_active_subtool(self, subtool: SubTool, event: QMouseEvent, **kwargs):
        """Safely activates a sub-tool."""
        if self.active_subtool:
            self.active_subtool.deactivate()
        self.active_subtool = subtool
        if self.active_subtool:
            self.active_subtool.activate(event, **kwargs)

    def deactivate_subtool(self):
        """Safely deactivates the current sub-tool."""
        if self.active_subtool:
            self.active_subtool.deactivate()
        self.active_subtool = None

    # --- Tool Activation/Deactivation ---

    def activate(self):
        super().activate()
        self.deactivate_subtool()
        self.annotation_window.viewport().setCursor(self.cursor)
        self._hovered_handle = None
        # Picking the tool back up with something already selected should show
        # its handles immediately, not wait for the next click.
        self._refresh_resize_handles()

    def deactivate(self):
        self.deactivate_subtool()
        self._hide_resize_handles()
        self._clear_annotation_hover()
        self.annotation_window.viewport().setCursor(self.default_cursor)
        self._hovered_handle = None
        super().deactivate()

    # --- Event Handlers (Dispatcher Logic) ---

    def mousePressEvent(self, event: QMouseEvent):
        # Ignore right mouse button events (used for panning)
        if event.button() == Qt.RightButton:
            return
        
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        # If a subtool is already active, delegate the event.
        if self.active_subtool:
            self.active_subtool.mousePressEvent(event)
            return

        position = self.annotation_window.mapToScene(event.pos())
        items = self.annotation_window.scene.items(position)

        # The click is about to change the selection, so the hover outline is
        # stale either way.
        self._clear_annotation_hover()

        # --- DISPATCHER LOGIC: Decide which sub-tool to activate ---
        # PRIORITY 1: Start Resizing if a handle is under the cursor.
        # The layer answers this itself, from the same decimated point list it
        # draws, so a vertex that is too dense to be drawn is also not
        # grabbable. Grab range is deliberately tight (see GRAB_RADIUS_PX):
        # handles are always visible now, and a click near a vertex must still
        # mean "move the annotation" unless it is genuinely on the handle.
        if len(self.selected_annotations) == 1:
            handle_name = self.resize_subtool.handle_at(position)
            if handle_name:
                self.set_active_subtool(
                    self.resize_subtool, event,
                    annotation=self.selected_annotations[0],
                    handle_name=handle_name
                )
                return

        # PRIORITY 2: Start Selection if Ctrl is pressed on an empty area.
        annotation_under_cursor = self._get_annotation_from_items(items, position)
        if (event.modifiers() & Qt.ControlModifier) and not annotation_under_cursor:
            self.set_active_subtool(self.select_subtool, event)
            return

        # PRIORITY 3: Default action - Select an annotation.
        # Pass the already-resolved annotation to avoid a second expensive hit-test.
        clicked_annotation = self._handle_annotation_selection(position, items, event.modifiers(),
                                                               cached_annotation=annotation_under_cursor)

        # If a selection was made and it's a left-click, start moving it.
        if clicked_annotation and event.button() == Qt.LeftButton:
            self.set_active_subtool(self.move_subtool, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.active_subtool:
            self.active_subtool.mouseMoveEvent(event)
            return

        # Idle hover: wake the handles nearest the cursor, show the resize
        # cursor when one is in grab range, and outline whichever annotation a
        # click would land on. QGraphicsView keeps mouse tracking on its
        # viewport, so this runs with no button held.
        self._update_handle_hover(event.pos())
        self._update_annotation_hover(event.pos())

    def leave(self):
        """Drop hover state when the pointer leaves the canvas."""
        self._hovered_handle = None
        self.resize_subtool.set_cursor_scene_pos(None)
        self._clear_annotation_hover()
        if self.active:
            self.annotation_window.viewport().setCursor(self.cursor)
        super().leave()

    # --- Hover preview -----------------------------------------------------

    # Re-resolving what is under the cursor means a scene query plus a
    # geometric hit test, so it is not worth doing for sub-pixel jitter.
    HOVER_PROBE_STEP_PX = 3.0

    def _update_annotation_hover(self, view_pos):
        """Outline the annotation a click would select.

        Overlapping shapes made selection a guessing game: the only way to find
        out what a click resolved to was to click and see. The hit test that
        answers that question already exists and already runs on every click --
        this just runs it on hover and draws the answer.
        """
        if not self.active or self.active_subtool:
            self._clear_annotation_hover()
            return

        if not self.annotation_window.cursorInWindow(view_pos):
            self._clear_annotation_hover()
            return

        # A handle beats the shape underneath it; the outline would only be
        # noise while the user is aiming at a vertex.
        if self._hovered_handle is not None:
            self._clear_annotation_hover()
            return

        if self._hover_probe_pos is not None:
            moved = (view_pos - self._hover_probe_pos).manhattanLength()
            if moved < self.HOVER_PROBE_STEP_PX:
                return
        self._hover_probe_pos = view_pos

        position = self.annotation_window.mapToScene(view_pos)
        items = self.annotation_window.scene.items(position)
        annotation = self._get_annotation_from_items(items, position)

        # A selected annotation already draws its own dashed outline, tag and
        # crosshair; a second outline on top of that says nothing.
        if annotation is not None and annotation.is_selected:
            annotation = None

        if annotation is None:
            self._clear_annotation_hover()
            return

        if annotation.id == self._hover_annotation_id and self._hover_item is not None:
            return

        self._clear_annotation_hover()
        self._draw_annotation_hover(annotation)

    def _draw_annotation_hover(self, annotation):
        """Add the hover outline item for ``annotation``."""
        try:
            path = annotation.get_painter_path()
        except Exception:
            return
        if path is None or path.isEmpty():
            return

        item = QGraphicsPathItem(path)
        color = QColor(annotation.label.color).lighter(150)
        pen = QPen(color, 2.5, Qt.DotLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        # Above the annotation layers so it reads against a crowded image,
        # below the resize handles (60) so it never covers one.
        item.setZValue(50)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setAcceptHoverEvents(False)

        self.annotation_window.scene.addItem(item)
        self._hover_item = item
        self._hover_annotation_id = annotation.id

    def _clear_annotation_hover(self):
        """Remove the hover outline, if there is one."""
        self._hover_annotation_id = None
        item = self._hover_item
        self._hover_item = None
        if item is None:
            return
        try:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        except RuntimeError:
            pass  # scene teardown already destroyed it

    # --- Live drag readout -------------------------------------------------

    def show_drag_size(self, annotation):
        """Report an annotation's live size while it is being resized."""
        text = self._size_text(annotation)
        if text:
            self.show_status(text, 2000)

    def show_drag_offset(self, delta):
        """Report how far a move drag has travelled."""
        self.show_status(f"Moved  {delta.x():+.0f}, {delta.y():+.0f} px", 2000)

    @staticmethod
    def _size_text(annotation):
        """W x H for an annotation, in real units when a scale is set."""
        try:
            bbox = annotation.cropped_bbox
            if not bbox:
                return ""
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            text = f"{width:.0f} x {height:.0f} px"

            scale_x = getattr(annotation, "scale_x", None)
            scale_y = getattr(annotation, "scale_y", None)
            units = getattr(annotation, "scale_units", None)
            if scale_x and scale_y and units:
                text += f"   ({width * scale_x:.3g} x {height * scale_y:.3g} {units})"
            return text
        except Exception:
            return ""

    def _update_handle_hover(self, view_pos):
        """Feed the cursor position to the handle layer and pick a cursor shape."""
        if self.resize_subtool.handle_layer is None:
            if self._hovered_handle is not None:
                self._hovered_handle = None
                self.annotation_window.viewport().setCursor(self.cursor)
            return

        if not self.annotation_window.cursorInWindow(view_pos):
            self.resize_subtool.set_cursor_scene_pos(None)
            if self._hovered_handle is not None:
                self._hovered_handle = None
                self.annotation_window.viewport().setCursor(self.cursor)
            return

        position = self.annotation_window.mapToScene(view_pos)
        self.resize_subtool.set_cursor_scene_pos(position)

        handle_name = self.resize_subtool.handle_at(position)
        if handle_name == self._hovered_handle:
            return
        self._hovered_handle = handle_name

        shape = self.resize_subtool.handle_layer.cursor_for(handle_name)
        self.annotation_window.viewport().setCursor(shape if shape is not None else self.cursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.active_subtool:
            self.active_subtool.mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        # Delegate to active sub-tool first (e.g., for canceling cut mode)
        if self.active_subtool:
            self.active_subtool.keyPressEvent(event)
            return

        # --- Hotkeys for starting tools/actions ---
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier:
            # Ctrl+Shift no longer summons the handles -- selecting an
            # annotation does that. It now means "show me everything": every
            # vertex, undecimated and unfaded, for surgery on a dense polygon.
            if self._is_ctrl_shift_down(event):
                self.resize_subtool.set_show_all(True)
            
            # --- Ctrl+X Hotkey Overload ---
            if event.key() == Qt.Key_X:
                if len(self.selected_annotations) > 1:
                    # If more than one annotation is selected, perform subtraction.
                    self.subtract_selected_annotations(event)
                elif len(self.selected_annotations) == 1:
                    # If only one is selected, start cutting mode.
                    self.set_active_subtool(self.cut_subtool, event, annotation=self.selected_annotations[0])

            # Ctrl+C: Combine selected annotations
            elif event.key() == Qt.Key_C and len(self.selected_annotations) > 1:
                self.combine_selected_annotations()

            # Ctrl+Space: Update with top machine confidence
            elif event.key() == Qt.Key_Space:
                self.update_with_top_machine_confidence()

    @staticmethod
    def _is_ctrl_shift_down(event: QKeyEvent) -> bool:
        """True when Ctrl and Shift are both held, given this key press.

        Qt reports a modifier key press with the modifier state as it was
        *before* that key applied, so pressing Shift while Ctrl is held arrives
        as Key_Shift carrying only ControlModifier. Testing modifiers() alone
        therefore misses the moment the combination completes and only notices
        on the next auto-repeat -- which is why holding Ctrl+Shift used to feel
        like it needed a beat before anything happened. Folding the event's own
        key into the test catches it on the press itself.
        """
        modifiers = event.modifiers()
        ctrl = bool(modifiers & Qt.ControlModifier) or event.key() == Qt.Key_Control
        shift = bool(modifiers & Qt.ShiftModifier) or event.key() == Qt.Key_Shift
        return ctrl and shift

    def keyReleaseEvent(self, event: QKeyEvent):
        if self.active_subtool:
            self.active_subtool.keyReleaseEvent(event)
            return

        # Drop the show-all override once either Ctrl or Shift is released.
        # The handles themselves stay: they belong to the selection now.
        if not (event.modifiers() & Qt.ShiftModifier and event.modifiers() & Qt.ControlModifier):
            self.resize_subtool.set_show_all(False)

    def wheelEvent(self, event: QMouseEvent):
        """Handle zoom using the mouse wheel or update polygon with Ctrl+Shift+wheel."""
        delta = event.angleDelta().y()
        modifiers = event.modifiers()

        if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            if len(self.selected_annotations) == 1:
                annotation = self.selected_annotations[0]
                # Capture old geometry
                try:
                    if hasattr(annotation, 'points'):
                        old_pts = [p for p in annotation.points]
                        old_holes = [list(h) for h in getattr(annotation, 'holes', [])]
                        old_geom = (old_pts, old_holes)
                    else:
                        old_geom = None
                except Exception:
                    old_geom = None

                changed = annotation.update_polygon(delta=1 if delta > 0 else -1)
                if changed is False and delta > 0:
                    # Only densifying can be refused, and only by the vertex
                    # cap; without a word here the wheel just stops working.
                    from coralnet_toolbox.Annotations.QtPolygonAnnotation import (
                        MAX_DENSIFY_VERTICES,
                    )
                    self.show_status(
                        f"Cannot add more vertices: the limit is {MAX_DENSIFY_VERTICES} "
                        f"per annotation. Scroll down to simplify first.",
                        4000,
                    )

                # Capture new geometry and push action
                try:
                    if hasattr(annotation, 'points'):
                        new_pts = [p for p in annotation.points]
                        new_holes = [list(h) for h in getattr(annotation, 'holes', [])]
                        new_geom = (new_pts, new_holes)
                    else:
                        new_geom = None
                except Exception:
                    new_geom = None

                if old_geom is not None and new_geom is not None and old_geom != new_geom:
                    try:
                        action = AnnotationGeometryEditAction(self.annotation_window, annotation.id, old_geom, new_geom)
                        self.annotation_window.action_stack.push(action)
                    except Exception:
                        pass
                    try:
                        self.annotation_window.annotationGeometryEdited.emit(annotation.id, {'old_geom': old_geom, 'new_geom': new_geom})
                    except Exception:
                        pass
                self.resize_subtool.refresh_handle_positions()
        elif modifiers & Qt.ControlModifier:
            self.annotation_window.set_annotation_size(delta=16 if delta > 0 else -16)

    # --- Helper and Action Methods ---

    def show_status(self, message, msecs=5000):
        """Put a message where the user will actually see it.

        Several failure paths in this tool used print(), which in a windowed
        application goes nowhere the user is looking -- so refusing to combine
        annotations, for instance, was indistinguishable from the hotkey not
        working.
        """
        try:
            self.annotation_window.main_window.status_bar.showMessage(message, msecs)
        except Exception:
            pass

    def _on_selection_changed(self, *args):
        """Selection changed: handles follow it."""
        self._refresh_resize_handles()

    def _on_view_changed(self, *args):
        """Zoom or pan changed: keep the layer's cached scale current."""
        self.resize_subtool.sync_view_scale()

    def _refresh_resize_handles(self, *args):
        """Show handles for a lone selected annotation, hide them otherwise.

        Handles are a property of the selection, not of a held modifier, so this
        is the single place that decides whether they exist. Resizing is only
        meaningful for one annotation at a time, so a multi-selection has none.
        """
        if not self.active:
            self._hide_resize_handles()
            return
        if len(self.selected_annotations) != 1:
            self._hide_resize_handles()
            return
        self.resize_subtool.display_resize_handles(self.selected_annotations[0])

    def _hide_resize_handles(self, *args):
        self._hovered_handle = None
        self.resize_subtool.remove_resize_handles()
            
    def _get_annotation_from_item(self, item):
        """Gets an annotation from a QGraphicsItem or its parent group."""
        annotation_id = None
        if isinstance(item, QGraphicsItemGroup):
            for child in item.childItems():
                if child.data(0):
                    annotation_id = child.data(0)
                    break
        else:
            annotation_id = item.data(0)
        
        return self.annotation_window.annotations_dict.get(annotation_id) if annotation_id else None

    def _get_annotation_from_items(self, items, position):
        """
        Finds the first valid annotation at a position.
        Checks active UI items first (selected/awake annotations), then falls back 
        to mathematical checks on Phantoms (unselected/sleeping annotations).
        
        Returns the topmost annotation at the position, respecting Z-index ordering
        and label visibility.
        """
        # Filter out the tool's own chrome. The handle layer has no shape() of
        # its own, so scene.items() reports it for any point inside its
        # bounding rect; the hover outline would resolve to the annotation it
        # is already drawn around.
        chrome = (self.resize_subtool.handle_layer, self._hover_item)
        valid_items = [item for item in items if item not in chrome]
        
        center_threshold = 10.0  # Distance threshold in pixels to consider a click "on center"
        center_candidates = []
        general_candidates = []
        
        # ========== PHASE 1: Check Awake (Selected) items via Qt's collision detection ==========
        # Gather all potential candidates from the scene
        for item in valid_items:
            annotation = self._get_annotation_from_item(item)
            if annotation and annotation.contains_point(position):
                # Calculate distance to center
                center_distance = (position - annotation.center_xy).manhattanLength()
                if center_distance <= center_threshold:
                    center_candidates.append(annotation)
                else:
                    general_candidates.append(annotation)
        
        # Return awake item if found (high priority)
        if center_candidates:
            return center_candidates[0]
        elif general_candidates:
            return general_candidates[0]
        
        # ========== PHASE 2: Check Sleeping (Phantom) items via mathematical collision ==========
        # If the Qt layer didn't find anything, check all annotations (including phantoms)
        # Iterate in reverse to respect visual Z-index (topmost items clicked first)
        px, py = position.x(), position.y()

        # The grid narrows the scan to one cell's worth of annotations. None
        # means there is nothing to index, in which case the original scan over
        # every annotation is still correct.
        hit_index = self.annotation_window.get_phantom_hit_index()

        if hit_index is not None:
            candidates = hit_index.candidates(px, py)
            indexed = True
        else:
            candidates = self.annotation_window.get_image_annotations()
            indexed = False
        annotations_dict = self.annotation_window.annotations_dict

        best_center = None
        best_general = None

        for annotation in reversed(candidates):
            # Skip selected annotations (already checked above) and invisible labels
            if annotation.is_selected or not getattr(annotation.label, 'is_visible', True):
                continue

            # Fast bounding-box pre-filter: skip annotations whose bbox
            # doesn't contain the click point (near-zero cost vs Shapely).
            bbox = annotation.cropped_bbox
            if bbox and not (bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]):
                continue

            # A guard against an index that outlived one of its entries: the
            # membership epoch is bumped from four call sites, and a fifth
            # appearing later would otherwise resurface a deleted annotation as
            # a clickable ghost. One dict lookup, and placed after the bbox
            # filter so only the handful of survivors pay it.
            if indexed and annotation.id not in annotations_dict:
                continue

            # Full geometric check (Shapely) only for bbox survivors
            if annotation.contains_point(position):
                center_distance = (position - annotation.center_xy).manhattanLength()
                if center_distance <= center_threshold:
                    best_center = annotation
                    break  # Center hit is highest priority — stop immediately
                elif best_general is None:
                    best_general = annotation
                    # Don't break: keep looking for a possible center hit
        
        if best_center:
            return best_center
        if best_general:
            return best_general
                
        return None

    def _handle_annotation_selection(self, position, items, modifiers, cached_annotation=None):
        """
        Handles the core logic of selecting and unselecting annotations.
        Returns the annotation that was clicked on, if any.
        """
        annotation = cached_annotation if cached_annotation is not None else self._get_annotation_from_items(items, position)
        locked_label = self.get_locked_label()
        multi_select = modifiers & Qt.ControlModifier

        if not annotation:
            # Clicked on an empty area without Ctrl, so unselect all
            if not multi_select:
                self.annotation_window.unselect_annotations()
            return None

        # Check if selection is locked to a specific label
        if locked_label and annotation.label.id != locked_label.id:
            # Say so. Silently dropping the click is indistinguishable from the
            # application having stopped responding to the mouse.
            self.show_status(
                f"Selection is locked to '{locked_label.short_label_code}' -- "
                f"'{annotation.label.short_label_code}' cannot be selected. "
                f"Unlock the label in the Label Window to select it.",
                4000,
            )
            return None  # Clicked annotation doesn't match locked label

        if annotation in self.selected_annotations:
            if multi_select:
                # Ctrl-click on a selected annotation: unselect it
                self.annotation_window.unselect_annotation(annotation)
                return None
            else:
                # Click on an already selected annotation (without Ctrl)
                # If more than one is selected, make this the only selection.
                if len(self.selected_annotations) > 1:
                    self.annotation_window.unselect_annotations()
                    self.annotation_window.select_annotation(annotation, multi_select=False)
                return annotation
        else:
            # Click on a not-yet-selected annotation
            if not multi_select:
                self.annotation_window.unselect_annotations()
            self.annotation_window.select_annotation(annotation, multi_select=True)
            return annotation

    def update_with_top_machine_confidence(self):
        """Update the selected annotation(s) with their top machine confidence predictions."""
        if not self.selected_annotations:
            return
        for annotation in self.selected_annotations:
            if annotation.machine_confidence:
                top_label = next(iter(annotation.machine_confidence))
                annotation.update_user_confidence(top_label)
        if len(self.selected_annotations) == 1:
            self.annotation_window.main_window.confidence_window.refresh_display()
            
    def subtract_selected_annotations(self, event):
        """
        Initiates the subtraction operation by activating the SubtractSubTool.
        """
        self.set_active_subtool(
            self.subtract_subtool, 
            event, 
            selected_annotations=self.selected_annotations.copy()
        )

    def combine_selected_annotations(self):
        """Combine multiple selected annotations of the same type."""
        # Work on a shallow copy to avoid mutations while deleting originals
        selected_annotations = self.annotation_window.selected_annotations.copy()
        
        if len(selected_annotations) <= 1:
            self.show_status("Cannot combine: select at least 2 annotations.")
            return  # Need at least 2 annotations to combine
        
        # Check if any annotations have machine confidence
        if any(not annotation.verified for annotation in selected_annotations):
            self.show_status(
                "Cannot combine: verify by selecting and pressing Ctrl+Space, "
                "clicking a label in the ConfidenceWindow, or updating the label manually.",
                5000,
            )
            return

        # Check that all selected annotations have the same label
        if not all(annotation.label.id == selected_annotations[0].label.id for annotation in selected_annotations):
            self.show_status(
                "Cannot combine annotations with different labels. Select annotations with the same label.", 5000)
            return
        
        # Identify the types of annotations being combined
        has_patches = any(isinstance(annotation, PatchAnnotation) for annotation in selected_annotations)
        has_polygons = any(isinstance(annotation, PolygonAnnotation) for annotation in selected_annotations)
        has_multi_polygons = any(isinstance(annotation, MultiPolygonAnnotation) for annotation in selected_annotations)
        has_rectangles = any(isinstance(annotation, RectangleAnnotation) for annotation in selected_annotations)
        
        # Handle cases where we can't combine different types
        if has_rectangles and (has_patches or has_polygons or has_multi_polygons):
            self.show_status(
                "Cannot combine: rectangle annotations can only be combined with other rectangles.", 5000)
            return

        # Check if all rectangle annotations (if any) are the same type
        if has_rectangles:
            first_type = type(selected_annotations[0])
            if not all(isinstance(annotation, first_type) for annotation in selected_annotations):
                self.show_status(
                    "Cannot combine: can only combine rectangles with other rectangles.", 5000)
                return
        
        # Handle different annotation type combinations
        if has_patches:
            # PatchAnnotation.combine can handle both patches and polygons
            combined_annotation = PatchAnnotation.combine(selected_annotations)
        elif has_rectangles:
            combined_annotation = RectangleAnnotation.combine(selected_annotations)
        elif has_polygons or has_multi_polygons:
            # Convert any MultiPolygonAnnotations to individual PolygonAnnotations first
            annotations_to_combine = []
            for annotation in selected_annotations:
                if isinstance(annotation, MultiPolygonAnnotation):
                    # Cut the MultiPolygonAnnotation into individual PolygonAnnotations
                    individual_polygons = annotation.cut()
                    annotations_to_combine.extend(individual_polygons)
                else:
                    annotations_to_combine.append(annotation)
            
            # Now combine all the polygons
            combined_annotation = PolygonAnnotation.combine(annotations_to_combine)
        else:
            self.show_status("Cannot combine: unsupported annotation type.")
            return  # Unsupported annotation type
        
        if not combined_annotation:
            self.show_status(
                "Failed to combine annotations -- check that the selected shapes overlap "
                "or are otherwise combinable."
            )
            return  # Failed to combine annotations
        
        # Add the new combined annotation to the scene
        # Add the new combined annotation to the scene WITHOUT recording (we'll record a single merge action)
        self.annotation_window.add_annotation_from_tool(combined_annotation, record_action=False)

        # Push a MergeAnnotationsAction and perform deletions without recording
        try:
            action = MergeAnnotationsAction(self.annotation_window, selected_annotations.copy(), combined_annotation)
            self.annotation_window.action_stack.push(action)
        except Exception:
            pass

        # Delete originals without recording separate actions
        for ann in selected_annotations:
            try:
                self.annotation_window.delete_annotation(ann.id, record_action=False)
            except Exception:
                pass

        # Select the new combined annotation
        self.annotation_window.select_annotation(combined_annotation)
        
    def cut_selected_annotation(self, cutting_points):
        """
        Performs the cut operation on the currently selected annotation using a
        provided list of points. This method is called by the CutSubTool.
        """
        if len(self.selected_annotations) != 1 or len(cutting_points) < 2:
            return  # Not enough cutting points

        annotation_to_cut = self.selected_annotations[0]
        
        # Call the appropriate cut method based on annotation type
        if isinstance(annotation_to_cut, RectangleAnnotation):
            new_annotations = RectangleAnnotation.cut(annotation_to_cut, cutting_points)
        elif isinstance(annotation_to_cut, PolygonAnnotation):
            new_annotations = PolygonAnnotation.cut(annotation_to_cut, cutting_points)
        elif isinstance(annotation_to_cut, MultiPolygonAnnotation):
            # For MultiPolygonAnnotation, we don't cut directly
            # Instead, we decompose it into individual PolygonAnnotations
            new_annotations = annotation_to_cut.cut()
        else:
            self.cancel_cutting_mode()
            return  # Unsupported annotation type
        
        # If the cut operation failed or was not applicable, do nothing.
        if not new_annotations:
            return

        # Remove the original annotation (without recording) and add newly created annotations (also without recording)
        try:
            # add new annotations first so UI updates properly
            for new_anno in new_annotations:
                self.annotation_window.add_annotation_from_tool(new_anno, record_action=False)

            self.annotation_window.delete_annotation(annotation_to_cut.id, record_action=False)

            # Push a single CutAnnotationAction to record the operation
            action = CutAnnotationAction(self.annotation_window, annotation_to_cut, new_annotations)
            try:
                self.annotation_window.action_stack.push(action)
            except Exception:
                pass

            try:
                self.annotation_window.annotationCut.emit(annotation_to_cut.id, new_annotations)
            except Exception:
                pass
        except Exception:
            # Fallback to previous behavior if something fails
            self.annotation_window.delete_selected_annotations()
            for new_anno in new_annotations:
                self.annotation_window.add_annotation_from_tool(new_anno)
            
    def cancel_cutting_mode(self):
        """Safely cancels cutting mode."""
        if self.active_subtool and isinstance(self.active_subtool, CutSubTool):
            self.deactivate_subtool()

    # --- Convenience Properties ---
    @property
    def selected_annotations(self):
        return self.annotation_window.selected_annotations

    def get_locked_label(self):
        return self.annotation_window.main_window.label_window.locked_label