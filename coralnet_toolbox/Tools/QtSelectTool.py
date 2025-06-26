import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QGraphicsItemGroup, QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Tools.QtMoveSubTool import MoveSubTool
from coralnet_toolbox.Tools.QtResizeSubTool import ResizeSubTool
from coralnet_toolbox.Tools.QtSelectSubTool import SelectSubTool
from coralnet_toolbox.Tools.QtCutSubTool import CutSubTool

from coralnet_toolbox.Annotations import (PatchAnnotation, 
                                          PolygonAnnotation, 
                                          RectangleAnnotation,
                                          MultiPolygonAnnotation)

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
        self.active_subtool: SubTool | None = None

        # --- State for transient UI (like resize handles) ---
        self.resize_handles_visible = False

        self._connect_signals()

    def _connect_signals(self):
        """Connect signals to hide resize handles when selection changes."""
        self.annotation_window.annotationSelected.connect(self._hide_resize_handles)
        self.annotation_window.annotationSizeChanged.connect(self._hide_resize_handles)
        self.annotation_window.annotationDeleted.connect(self._hide_resize_handles)

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
        self._hide_resize_handles()
        self.annotation_window.viewport().setCursor(self.cursor)

    def deactivate(self):
        self.deactivate_subtool()
        self._hide_resize_handles()
        self.annotation_window.viewport().setCursor(self.default_cursor)
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

        # --- DISPATCHER LOGIC: Decide which sub-tool to activate ---
        # PRIORITY 1: Start Resizing if a visible handle is clicked.
        if self.resize_handles_visible:
            for item in items:
                if item in self.resize_subtool.resize_handles_items:
                    handle_name = item.data(1)
                    if handle_name and len(self.selected_annotations) == 1:
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
        clicked_annotation = self._handle_annotation_selection(position, items, event.modifiers())

        # If a selection was made and it's a left-click, start moving it.
        if clicked_annotation and event.button() == Qt.LeftButton:
            self.set_active_subtool(self.move_subtool, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.active_subtool:
            self.active_subtool.mouseMoveEvent(event)

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
            # Ctrl+Shift: Show resize handles for single selected annotation
            if modifiers & Qt.ShiftModifier and len(self.selected_annotations) == 1:
                self._show_resize_handles()
            
            # Ctrl+X: Start cutting mode
            if event.key() == Qt.Key_X and len(self.selected_annotations) == 1:
                self.set_active_subtool(self.cut_subtool, event, annotation=self.selected_annotations[0])

            # Ctrl+C: Combine selected annotations
            elif event.key() == Qt.Key_C and len(self.selected_annotations) > 1:
                self.combine_selected_annotations()

            # Ctrl+Space: Update with top machine confidence
            elif event.key() == Qt.Key_Space:
                self.update_with_top_machine_confidence()

    def keyReleaseEvent(self, event: QKeyEvent):
        if self.active_subtool:
            self.active_subtool.keyReleaseEvent(event)
            return

        # Hide resize handles if either Ctrl or Shift is released
        if not (event.modifiers() & Qt.ShiftModifier and event.modifiers() & Qt.ControlModifier):
            self._hide_resize_handles()

    def wheelEvent(self, event: QMouseEvent):
        """Handle zoom using the mouse wheel or update polygon with Ctrl+Shift+wheel."""
        delta = event.angleDelta().y()
        modifiers = event.modifiers()

        if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            if len(self.selected_annotations) == 1:
                annotation = self.selected_annotations[0]
                annotation.update_polygon(delta=1 if delta > 0 else -1)
                if self.resize_handles_visible:
                    self._show_resize_handles()
        elif modifiers & Qt.ControlModifier:
            self.annotation_window.set_annotation_size(delta=16 if delta > 0 else -16)

    # --- Helper and Action Methods ---

    def _show_resize_handles(self):
        if len(self.selected_annotations) == 1:
            self.resize_handles_visible = True
            self.resize_subtool.display_resize_handles(self.selected_annotations[0])

    def _hide_resize_handles(self):
        if self.resize_handles_visible:
            self.resize_handles_visible = False
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
        """Finds the first valid annotation at a position from a list of items."""
        for item in items:
            # We don't want to select by clicking a resize handle
            if item in self.resize_subtool.resize_handles_items:
                continue
            annotation = self._get_annotation_from_item(item)
            if annotation and annotation.contains_point(position):
                return annotation
        return None

    def _handle_annotation_selection(self, position, items, modifiers):
        """
        Handles the core logic of selecting and unselecting annotations.
        Returns the annotation that was clicked on, if any.
        """
        annotation = self._get_annotation_from_items(items, position)
        locked_label = self.get_locked_label()
        multi_select = modifiers & Qt.ControlModifier

        if not annotation:
            # Clicked on an empty area without Ctrl, so unselect all
            if not multi_select:
                self.annotation_window.unselect_annotations()
            return None

        # Check if selection is locked to a specific label
        if locked_label and annotation.label.id != locked_label.id:
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

    def combine_selected_annotations(self):
        """Combine multiple selected annotations of the same type."""
        selected_annotations = self.annotation_window.selected_annotations
        
        if len(selected_annotations) <= 1:
            print("Need at least 2 annotations to combine.")
            return  # Need at least 2 annotations to combine
        
        # Check if any annotations have machine confidence
        if any(not annotation.verified for annotation in selected_annotations):
            QMessageBox.warning(
                self.annotation_window,
                "Cannot Combine",
                "Cannot combine annotations with machine confidence. Confirm predictions (Ctrl+Space) first."
            )
            return
        
        # Check that all selected annotations have the same label
        if not all(annotation.label.id == selected_annotations[0].label.id for annotation in selected_annotations):
            QMessageBox.warning(
                self.annotation_window,
                "Cannot Combine",
                "Cannot combine annotations with different labels. Select annotations with the same label."
            )
            return
        
        # Identify the types of annotations being combined
        has_patches = any(isinstance(annotation, PatchAnnotation) for annotation in selected_annotations)
        has_polygons = any(isinstance(annotation, PolygonAnnotation) for annotation in selected_annotations)
        has_multi_polygons = any(isinstance(annotation, MultiPolygonAnnotation) for annotation in selected_annotations)
        has_rectangles = any(isinstance(annotation, RectangleAnnotation) for annotation in selected_annotations)
        
        # Handle cases where we can't combine different types
        if has_rectangles and (has_patches or has_polygons or has_multi_polygons):
            QMessageBox.warning(
                self.annotation_window,
                "Cannot Combine",
                "Rectangle annotations can only be combined with other rectangles."
            )
            return
        
        # Check if all rectangle annotations (if any) are the same type
        if has_rectangles:
            first_type = type(selected_annotations[0])
            if not all(isinstance(annotation, first_type) for annotation in selected_annotations):
                QMessageBox.warning(
                    self.annotation_window,
                    "Cannot Combine",
                    "Can only combine rectangles with other rectangles."
                )
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
            print("Failed to combine annotations. Unsupported annotation types.")
            return  # Unsupported annotation type
        
        if not combined_annotation:
            print("Failed to combine annotations. Please check the selected annotations.")
            return  # Failed to combine annotations
        
        # Add the new combined annotation to the scene
        self.annotation_window.add_annotation_from_tool(combined_annotation)
        
        # Delete the original annotations
        self.annotation_window.delete_selected_annotations()
        
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

        # Remove the original annotation
        self.annotation_window.delete_selected_annotations()

        # Add the newly created annotations from the cut.
        for new_anno in new_annotations:
            self.annotation_window.add_annotation_from_tool(new_anno)

    # --- Convenience Properties ---
    @property
    def selected_annotations(self):
        return self.annotation_window.selected_annotations

    def get_locked_label(self):
        return self.annotation_window.main_window.label_window.locked_label