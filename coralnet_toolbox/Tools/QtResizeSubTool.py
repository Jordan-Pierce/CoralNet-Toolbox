from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsEllipseItem

from coralnet_toolbox.Tools.QtSubTool import SubTool

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResizeSubTool(SubTool):
    """SubTool for resizing a single annotation using handles."""

    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.target_annotation = None
        self.resize_handle_name = None
        self.resize_handles_items = []

    def activate(self, event, **kwargs):
        """
        Activates the resizing operation.
        Expects 'annotation' and 'handle_name' in kwargs.
        """
        super().activate(event)
        self.target_annotation = kwargs.get('annotation')
        self.resize_handle_name = kwargs.get('handle_name')

        if not self.target_annotation or not self.resize_handle_name:
            # Invalid activation, immediately deactivate.
            self.parent_tool.deactivate_subtool()

    def deactivate(self):
        super().deactivate()
        self.target_annotation = None
        self.resize_handle_name = None
        # Note: The parent SelectTool is responsible for hiding handles when keys are released.

    def mouseMoveEvent(self, event):
        """Perform the resize operation."""
        if not self.is_active or not self.target_annotation:
            return

        if not self.annotation_window.is_annotation_moveable(self.target_annotation):
            self.parent_tool.deactivate_subtool()
            return
            
        current_pos = self.annotation_window.mapToScene(event.pos())
        self.target_annotation.resize(self.resize_handle_name, current_pos)
        
        # Continuously update handles as we resize
        self.display_resize_handles(self.target_annotation)

    def mouseReleaseEvent(self, event):
        """Finalize the resize, update related windows, and deactivate."""
        if self.target_annotation:
            self.target_annotation.create_cropped_image(self.annotation_window.rasterio_image)
            self.parent_tool.main_window.confidence_window.display_cropped_image(self.target_annotation)

        self.parent_tool.deactivate_subtool()

    # --- Handle Management Logic (moved from original class) ---

    def display_resize_handles(self, annotation):
        """Display resize handles for the given annotation."""
        self.remove_resize_handles()
        handles = self._get_handles(annotation)
        handle_size = self.parent_tool.graphics_utility.get_handle_size(self.annotation_window)

        for handle_name, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - handle_size // 2,
                                           point.y() - handle_size // 2,
                                           handle_size,
                                           handle_size)
            
            handle_color = QColor(annotation.label.color)
            border_color = QColor(255 - handle_color.red(), 255 - handle_color.green(), 255 - handle_color.blue())
            
            ellipse.setPen(QPen(border_color, 2))
            ellipse.setBrush(QBrush(handle_color))
            ellipse.setData(1, handle_name)  # Store handle name
            ellipse.setAcceptHoverEvents(True)
            ellipse.setAcceptedMouseButtons(Qt.LeftButton)
            
            self.annotation_window.scene.addItem(ellipse)
            self.resize_handles_items.append(ellipse)

    def remove_resize_handles(self):
        """Remove any displayed resize handles."""
        for handle in self.resize_handles_items:
            self.annotation_window.scene.removeItem(handle)
        self.resize_handles_items.clear()

    def _get_handles(self, annotation):
        """Return the handles based on the annotation type."""
        if isinstance(annotation, RectangleAnnotation):
            return self._get_rectangle_handles(annotation)
        if isinstance(annotation, PolygonAnnotation):
            return self._get_polygon_handles(annotation)
        return {}

    def _get_rectangle_handles(self, annotation):
        """Return resize handles for a rectangle annotation."""
        top_left, bottom_right = annotation.top_left, annotation.bottom_right
        return {
            "left": QPointF(top_left.x(), (top_left.y() + bottom_right.y()) / 2),
            "right": QPointF(bottom_right.x(), (top_left.y() + bottom_right.y()) / 2),
            "top": QPointF((top_left.x() + bottom_right.x()) / 2, top_left.y()),
            "bottom": QPointF((top_left.x() + bottom_right.x()) / 2, bottom_right.y()),
            "top_left": QPointF(top_left.x(), top_left.y()),
            "top_right": QPointF(bottom_right.x(), top_left.y()),
            "bottom_left": QPointF(top_left.x(), bottom_right.y()),
            "bottom_right": QPointF(bottom_right.x(), bottom_right.y()),
        }

    def _get_polygon_handles(self, annotation):
        """Return resize handles for a polygon annotation."""
        return {f"point_{i}": QPointF(p.x(), p.y()) for i, p in enumerate(annotation.points)}