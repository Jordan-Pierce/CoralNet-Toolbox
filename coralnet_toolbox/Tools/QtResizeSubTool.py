from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter

from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.QtActions import AnnotationGeometryEditAction

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResizeHandleItem(QGraphicsEllipseItem):
    def __init__(self, x, y, size, color, handle_name, is_polygon=False):
        # We initialize the parent ellipse to act as our "invisible hitbox"
        super().__init__(x, y, size, size)
        
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        
        self.handle_name = handle_name
        self.base_color = QColor(color)
        self.is_polygon = is_polygon
        self.is_hovered = False
        self.hitbox_size = size

        # The base item is transparent so it acts purely as a hit-detector
        self.setPen(QPen(Qt.transparent))
        self.setBrush(QBrush(Qt.transparent))

    def paint(self, painter, option, widget=None):
        """Custom drawing logic based on proximity (hover state) and shape type."""
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.boundingRect().center()
        
        if self.is_hovered:
            # PROXIMITY WAKE-UP: Crisp white center, colored border
            radius = self.hitbox_size * 0.35  # Fill about 70% of the hitbox
            
            pen = QPen(self.base_color, 2, Qt.SolidLine)
            pen.setCosmetic(True) # Stays crisp regardless of zoom
            painter.setPen(pen)
            
            painter.setBrush(QBrush(Qt.white))
            painter.setOpacity(1.0)
            
        else:
            # SLEEP STATE
            if self.is_polygon:
                # Polygons: Subdued, tiny dots to completely remove visual clutter
                radius = self.hitbox_size * 0.15
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(self.base_color))
                painter.setOpacity(0.6)
            else:
                # Rectangles: Standard un-hovered handles (rectangles have few handles, so we leave them visible)
                radius = self.hitbox_size * 0.25
                pen = QPen(self.base_color, 1.5, Qt.SolidLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(QBrush(Qt.white))
                painter.setOpacity(1.0)

        # Draw our custom dynamic handle
        painter.drawEllipse(center, radius, radius)

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()  # Trigger a re-paint
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()  # Trigger a re-paint
        super().hoverLeaveEvent(event)
        

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
            return

        # Capture original geometry for undo
        try:
            if hasattr(self.target_annotation, 'points'):
                # store shallow copies of points and holes
                pts = [QPointF(p.x(), p.y()) for p in self.target_annotation.points]
                holes = []
                if hasattr(self.target_annotation, 'holes') and self.target_annotation.holes:
                    for hole in self.target_annotation.holes:
                        holes.append([QPointF(p.x(), p.y()) for p in hole])
                self._orig_geom = (pts, holes)
            else:
                # rectangles: store top_left/bottom_right
                try:
                    tl = QPointF(self.target_annotation.top_left.x(), self.target_annotation.top_left.y())
                    br = QPointF(self.target_annotation.bottom_right.x(), self.target_annotation.bottom_right.y())
                    self._orig_geom = (tl, br)
                except Exception:
                    self._orig_geom = None
        except Exception:
            self._orig_geom = None

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
            
            # Capture new geometry and emit signals
            new_geom = None
            try:
                if hasattr(self.target_annotation, 'points'):
                    pts = [QPointF(p.x(), p.y()) for p in self.target_annotation.points]
                    holes = []
                    if hasattr(self.target_annotation, 'holes') and self.target_annotation.holes:
                        for hole in self.target_annotation.holes:
                            holes.append([QPointF(p.x(), p.y()) for p in hole])
                    new_geom = (pts, holes)
                else:
                    tl = QPointF(self.target_annotation.top_left.x(), self.target_annotation.top_left.y())
                    br = QPointF(self.target_annotation.bottom_right.x(), self.target_annotation.bottom_right.y())
                    new_geom = (tl, br)
            except Exception:
                new_geom = None

            # Push undo action and emit signals
            if self._orig_geom is not None and new_geom is not None:
                try:
                    action = AnnotationGeometryEditAction(self.annotation_window, 
                                                          self.target_annotation.id, 
                                                          self._orig_geom, new_geom)
                    self.annotation_window.action_stack.push(action)
                except Exception:
                    pass
                
                # Always emit geometry edited signal (critical for viewer updates)
                self.annotation_window.annotationGeometryEdited.emit(
                    self.target_annotation.id, 
                    {'old_geom': self._orig_geom, 'new_geom': new_geom}
                )
            
            # Also emit the general modified signal for backwards compatibility
            self.annotation_window.annotationModified.emit(self.target_annotation.id)
            
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
            
        # Increase the screen size to create a generous invisible hitbox (e.g., 24 pixels)
        desired_screen_size = 24  
        handle_size = desired_screen_size / scale

        # Determine if we are styling for a polygon
        is_poly = isinstance(annotation, PolygonAnnotation)

        for handle_name, point in handles.items():
            # Create our custom proximity-based handle
            handle_item = ResizeHandleItem(
                point.x() - handle_size / 2,
                point.y() - handle_size / 2,
                handle_size,
                annotation.label.color,
                handle_name,
                is_polygon=is_poly
            )
            
            # Keep compatibility with SelectTool logic
            handle_item.setData(1, handle_name)  
            
            self.annotation_window.scene.addItem(handle_item)
            self.resize_handles_items.append(handle_item)

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