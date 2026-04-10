import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QBrush, QPen, QColor
from PyQt5.QtWidgets import QGraphicsPathItem, QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor  # Explicitly set, if needed

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        # Call parent deactivate to ensure crosshair and cursor preview are properly cleared
        super().deactivate()

    def mousePressEvent(self, event: QMouseEvent):

        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None
        
        # Add cursor bounds check
        if not self.annotation_window.cursorInWindow(event.pos()):
            return None

        if event.button() == Qt.LeftButton:
            self.annotation_window.unselect_annotations()
            
            # Create a new annotation at the clicked position
            annotation = self.create_annotation(self.annotation_window.mapToScene(event.pos()), finished=True)
            self.annotation_window.add_annotation_from_tool(annotation)
            
            # After adding annotation, restore cursor annotation
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Call parent implementation to handle crosshair
        super().mouseMoveEvent(event)
        
        # Then clear any existing cursor annotation
        self.clear_cursor_annotation()
        
        # Continue with tool-specific behavior for cursor annotation
        if self.annotation_window.active_image and self.annotation_window.selected_label:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.annotation_window.cursorInWindow(event.pos()):
                self.create_cursor_annotation(scene_pos)
                if self.cursor_move_callback:
                    self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
            else:
                if self.cursor_clear_callback:
                    self.cursor_clear_callback()
        else:
            if self.cursor_clear_callback:
                self.cursor_clear_callback()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.annotation_window.active_image and self.annotation_window.selected_label:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.annotation_window.cursorInWindow(event.pos()):
                self.create_cursor_annotation(scene_pos)

    def wheelEvent(self, event: QMouseEvent):
        # Handle Zoom wheel for setting annotation size
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.annotation_window.set_annotation_size(delta=16)  # Zoom in
            else:
                self.annotation_window.set_annotation_size(delta=-16)  # Zoom out

            # Update the cursor annotation with the new size
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)
            
    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        annotation = PatchAnnotation(
            scene_pos,
            self.annotation_window.annotation_size,
            self.annotation_window.selected_label,
            self.annotation_window.current_image_path,
            transparency=self.annotation_window.main_window.get_transparency_value(),
            show_confidence=False,
        )
        return annotation

    def create_cursor_preview_item(self, u: float, v: float):
        """Return a styled patch square QGraphicsItem centred at image pixel (u, v)."""
        if not self.annotation_window.selected_label:
            return None

        label = self.annotation_window.selected_label
        size = self.annotation_window.annotation_size
        transparency = self.annotation_window.main_window.get_transparency_value()
        ann = PatchAnnotation(
            QPointF(u, v),
            size,
            label,
            "",
            transparency,
            show_confidence=False,
        )
        path = ann.get_painter_path()
        item = QGraphicsPathItem(path)
        c = QColor(label.color)
        fill = QColor(c)
        fill.setAlpha(transparency)
        item.setBrush(QBrush(fill))
        pen = QPen(c, 1)
        pen.setCosmetic(True)
        item.setPen(pen)

        return item

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a patch cursor annotation at the given position."""
        if not scene_pos or not self.annotation_window.selected_label or not self.annotation_window.active_image:
            self.clear_cursor_annotation()
            return
            
        # First ensure any existing cursor annotation is removed
        self.clear_cursor_annotation()
        
        # Create a new cursor annotation with semi-transparent appearance
        self.cursor_annotation = self.create_annotation(scene_pos)
        if self.cursor_annotation:
            # Make the cursor annotation semi-transparent to distinguish it from actual annotations
            self.cursor_annotation.update_transparency(self.annotation_window.main_window.get_transparency_value())
            # Force hydrate the cursor preview so it follows the mouse smoothly
            self.cursor_annotation.create_graphics_item(self.annotation_window.scene, force_hydrate=True)
            # Show the dimension tag while drawing
            if hasattr(self.cursor_annotation, 'dimension_tag_item') and self.cursor_annotation.dimension_tag_item:
                self.cursor_annotation.dimension_tag_item.setVisible(True)
            
            # Track current size for optimization
            self._last_annotation_size = self.annotation_window.annotation_size

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        if scene_pos is None:
            self.clear_cursor_annotation()
            return
        
        # If cursor annotation exists and size hasn't changed, just update position
        if (self.cursor_annotation and 
            hasattr(self, '_last_annotation_size') and self._last_annotation_size == self.annotation_window.annotation_size):
            # Move the existing annotation to new position
            self.cursor_annotation.update_location(scene_pos)
        else:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)