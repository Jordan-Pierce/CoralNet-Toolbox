import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMessageBox

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

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.main_window.annotation_size_spinbox.setEnabled(True)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.annotation_window.main_window.annotation_size_spinbox.setEnabled(False)
        self.clear_cursor_annotation()

    def mousePressEvent(self, event: QMouseEvent):

        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
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
        # First clear any existing cursor annotation
        self.clear_cursor_annotation()
            
        if self.annotation_window.active_image and self.annotation_window.selected_label:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.annotation_window.cursorInWindow(event.pos()):
                self.create_cursor_annotation(scene_pos)

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

        annotation = PatchAnnotation(scene_pos,
                                     self.annotation_window.annotation_size,
                                     self.annotation_window.selected_label.short_label_code,
                                     self.annotation_window.selected_label.long_label_code,
                                     self.annotation_window.selected_label.color,
                                     self.annotation_window.current_image_path,
                                     self.annotation_window.selected_label.id,
                                     transparency=self.annotation_window.selected_label.transparency)
        return annotation

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
            self.cursor_annotation.transparency = min(self.cursor_annotation.transparency + 100, 200)
            self.cursor_annotation.create_graphics_item(self.annotation_window.scene)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)