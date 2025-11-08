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
        self._current_annotation = None  # Track which annotation we're showing handles for

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
        
        # Force the scene to update immediately
        self.annotation_window.scene.update()

    def mouseReleaseEvent(self, event):
        """Finalize the resize, update related windows, and deactivate."""
        if self.target_annotation:
            # Normalize the coordinates after resize is complete
            if hasattr(self.target_annotation, 'normalize_coordinates'):
                self.target_annotation.normalize_coordinates()
                
            self.target_annotation.create_cropped_image(self.annotation_window.rasterio_image)
            self.parent_tool.main_window.confidence_window.display_cropped_image(self.target_annotation)
            self.annotation_window.annotationModified.emit(self.target_annotation.id)  # Emit modified signal
            
        self.parent_tool.deactivate_subtool()

    # --- Handle Management Logic (moved from original class) ---

    def display_resize_handles(self, annotation):
        """Display resize handles for the given annotation."""
        self.remove_resize_handles()
        handles = self._get_handles(annotation)
        
        # If we're currently resizing and the handle no longer exists, deactivate
        if self.is_active and self.resize_handle_name not in handles:
            self.parent_tool.deactivate_subtool()
            return
        
        # Connect to annotation updates to automatically refresh handles
        if hasattr(annotation, 'annotationUpdated'):
            # Disconnect any existing connections first to avoid duplicates
            try:
                annotation.annotationUpdated.disconnect(self._on_annotation_updated)
            except:
                pass  # Connection didn't exist
            # Connect to the update signal
            annotation.annotationUpdated.connect(self._on_annotation_updated)
        
        # Store reference to current annotation for updates
        self._current_annotation = annotation
        
        # Calculate handle size based on current zoom to maintain constant screen size
        scale = self.annotation_window.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero
        desired_screen_size = 15  # Desired handle size in screen pixels
        handle_size = desired_screen_size / scale

        for handle_name, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - handle_size / 2,
                                           point.y() - handle_size / 2,
                                           handle_size,
                                           handle_size)
            
            handle_color = QColor(annotation.label.color)
            border_color = QColor(255 - handle_color.red(), 255 - handle_color.green(), 255 - handle_color.blue())
            
            pen = QPen(border_color, 3)
            pen.setCosmetic(True)  # Keep pen width constant
            ellipse.setPen(pen)
            ellipse.setBrush(QBrush(handle_color))
            ellipse.setData(1, handle_name)  # Store handle name
            ellipse.setAcceptHoverEvents(True)
            ellipse.setAcceptedMouseButtons(Qt.LeftButton)
            
            self.annotation_window.scene.addItem(ellipse)
            self.resize_handles_items.append(ellipse)

    def _on_annotation_updated(self, annotation):
        """Handle annotation updates by refreshing the resize handles."""
        # Only refresh if this is the annotation we're currently showing handles for
        if hasattr(self, '_current_annotation') and annotation == self._current_annotation:
            # Don't refresh during active resize to avoid interference
            if not self.is_active:
                self.display_resize_handles(annotation)

    def remove_resize_handles(self):
        """Remove any displayed resize handles."""
        # Disconnect from annotation updates
        if hasattr(self, '_current_annotation') and hasattr(self._current_annotation, 'annotationUpdated'):
            try:
                self._current_annotation.annotationUpdated.disconnect(self._on_annotation_updated)
            except:
                pass  # Connection didn't exist
        
        # Clear the reference
        if hasattr(self, '_current_annotation'):
            del self._current_annotation
        
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
        """
        Return resize handles for a polygon, including its outer boundary and all holes.
        Uses the new handle format: 'point_{poly_index}_{vertex_index}'.
        """
        handles = {}

        # 1. Create handles for the outer boundary using the 'outer' keyword.
        for i, p in enumerate(annotation.points):
            handle_name = f"point_outer_{i}"
            handles[handle_name] = QPointF(p.x(), p.y())

        # 2. Create handles for each of the inner holes using their index.
        if hasattr(annotation, 'holes'):
            for hole_index, hole in enumerate(annotation.holes):
                for vertex_index, p in enumerate(hole):
                    handle_name = f"point_{hole_index}_{vertex_index}"
                    handles[handle_name] = QPointF(p.x(), p.y())
        
        return handles