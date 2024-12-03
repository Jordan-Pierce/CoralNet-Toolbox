import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.PointingHandCursor
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.resize_start_pos = None
        self.move_start_pos = None
        self.resize_handles = []
        self.buffer = 50
        self.rectangle_selection = False
        self.selection_rectangle = None
        self.selection_start_pos = None

        # Manage selected annotations
        self.annotation_window.annotationSelected.connect(self.clear_resize_handles)
        self.annotation_window.annotationSizeChanged.connect(self.clear_resize_handles)
        self.annotation_window.annotationDeleted.connect(self.clear_resize_handles)

    def clear_resize_handles(self, annotation_id=None):
        """Clear resize handles if annotations change."""
        self.remove_resize_handles()

    def wheelEvent(self, event: QMouseEvent):
        """Handle zoom using the mouse wheel."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.annotation_window.set_annotation_size(delta=16 if delta > 0 else -16)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events to select annotations."""
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        if event.button() == Qt.LeftButton:
            position = self.annotation_window.mapToScene(event.pos())
            items = self.get_clickable_items(position)

            if event.modifiers() & Qt.ControlModifier:
                self.rectangle_selection = True
                self.selection_start_pos = position
                self.selection_rectangle = QGraphicsRectItem()
                self.selection_rectangle.setPen(QPen(Qt.black, 2, Qt.DashLine))
                self.annotation_window.scene.addItem(self.selection_rectangle)

            selected_annotation = self.select_annotation(position, items, event.modifiers())
            if selected_annotation:
                self.init_drag_or_resize(selected_annotation, position, event.modifiers())

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for resizing, moving, or drawing selection rectangle."""
        current_pos = self.annotation_window.mapToScene(event.pos())
        if self.rectangle_selection:
            self.update_selection_rectangle(current_pos)
        elif self.resizing:
            self.handle_resize(current_pos)
        elif self.moving:
            # Check if the selected annotation has a warning message
            if len(self.annotation_window.selected_annotations):
                selected_annotation = self.annotation_window.selected_annotations[0]
                if not self.annotation_window.is_annotation_moveable(selected_annotation):
                    self.moving = False
                else:
                    self.handle_move(current_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events to stop moving, resizing, or finalize selection rectangle."""
        if self.rectangle_selection:
            self.finalize_selection_rectangle()
            self.rectangle_selection = False
            self.selection_start_pos = None
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None

        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.resize_start_pos = None
        self.annotation_window.drag_start_pos = None

    def keyPressEvent(self, event):
        """Handle key press events to show resize handles."""
        if len(self.annotation_window.selected_annotations) == 1 and event.modifiers() & Qt.ShiftModifier:
            self.display_resize_handles(self.annotation_window.selected_annotations[0])

    def keyReleaseEvent(self, event):
        """Handle key release events to hide resize handles."""
        if not event.modifiers() & Qt.ShiftModifier:
            self.remove_resize_handles()

    def get_clickable_items(self, position):
        """Get items that can be clicked in the scene."""
        items = self.annotation_window.scene.items(position)
        return [item for item in items if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem))]

    def select_annotation(self, position, items, modifiers):
        """Select an annotation based on the click position."""
        center_proximity_items = []
        for item in items:
            if self.is_annotation_clickable(item, position):
                center_proximity_items.append((item, self.calculate_distance(position, self.get_item_center(item))))

        # Sort by proximity to the center
        center_proximity_items.sort(key=lambda x: x[1])

        for item, _ in center_proximity_items:
            annotation_id = item.data(0)
            selected_annotation = self.annotation_window.annotations_dict.get(annotation_id)

            if selected_annotation:
                ctrl_pressed = modifiers & Qt.ControlModifier
                if selected_annotation in self.annotation_window.selected_annotations and ctrl_pressed:
                    # Unselect the annotation if Ctrl is pressed and it is already selected
                    self.annotation_window.unselect_annotation(selected_annotation)
                    return None
                else:
                    return self.handle_selection(selected_annotation, modifiers)

        return None

    def update_selection_rectangle(self, current_pos):
        """Update the selection rectangle while dragging."""
        # Only update the rectangle if the cursor is within the window
        if self.selection_rectangle and self.annotation_window.cursorInWindow(current_pos, mapped=True):
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_rectangle.setRect(rect)

    def finalize_selection_rectangle(self):
        """Finalize the selection rectangle and select annotations within it."""
        if self.selection_rectangle:
            rect = self.selection_rectangle.rect()
            # Don't clear previous selection when using rectangle selection
            for annotation in self.annotation_window.get_image_annotations():
                if rect.contains(annotation.center_xy):
                    # Always add to selection when using rectangle
                    self.annotation_window.select_annotation(annotation, True)

    def get_item_center(self, item):
        """Return the center point of the item."""
        if isinstance(item, QGraphicsRectItem):
            rect = item.rect()
            return QPointF(rect.x() + rect.width() / 2, rect.y() + rect.height() / 2)
        elif isinstance(item, QGraphicsPolygonItem):
            return item.polygon().boundingRect().center()
        elif isinstance(item, QGraphicsEllipseItem):
            return item.rect().center()
        return QPointF(0, 0)  # Default if item type is unsupported

    def is_annotation_clickable(self, item, position):
        """Check if the clicked position is within the annotation."""
        annotation = self.annotation_window.annotations_dict.get(item.data(0))
        return annotation and annotation.contains_point(position)

    def handle_selection(self, selected_annotation, modifiers):
        """Handle annotation selection logic."""
        ctrl_pressed = modifiers & Qt.ControlModifier

        if selected_annotation in self.annotation_window.selected_annotations:
            if ctrl_pressed:
                # Toggle selection when Ctrl is pressed
                self.annotation_window.unselect_annotation(selected_annotation)
                return None
            else:
                # If Ctrl is not pressed, keep only this annotation selected
                self.annotation_window.unselect_annotations()
                self.annotation_window.select_annotation(selected_annotation)
                return selected_annotation
        else:
            # If annotation is not selected
            if not ctrl_pressed:
                # Clear selection if Ctrl is not pressed
                self.annotation_window.unselect_annotations()
            # Add to selection
            self.annotation_window.select_annotation(selected_annotation, True)
            return selected_annotation

    def init_drag_or_resize(self, selected_annotation, position, modifiers):
        """Initialize dragging or resizing based on the current state."""
        self.annotation_window.drag_start_pos = position
        self.move_start_pos = position

        if modifiers & Qt.ShiftModifier:
            self.resize_handle = self.detect_resize_handle(selected_annotation, position)
            if self.resize_handle:
                self.resizing = True
        else:
            self.moving = True

    def handle_move(self, current_pos):
        """Handle moving the selected annotation."""
        if not len(self.annotation_window.selected_annotations):
            return  
        
        selected_annotation = self.annotation_window.selected_annotations[0]
        delta = current_pos - self.move_start_pos
        new_center = selected_annotation.center_xy + delta

        if self.annotation_window.cursorInWindow(current_pos, mapped=True):
            rasterio_image = self.annotation_window.rasterio_image
            self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
            selected_annotation.create_cropped_image(rasterio_image)
            self.annotation_window.main_window.confidence_window.display_cropped_image(selected_annotation)
            self.move_start_pos = current_pos

    def handle_resize(self, current_pos):
        """Handle resizing the selected annotation."""
        if not len(self.annotation_window.selected_annotations):
            return
        
        selected_annotation = self.annotation_window.selected_annotations[0]
        if not self.annotation_window.is_annotation_moveable(selected_annotation):
            return
        
        self.resize_annotation(selected_annotation, current_pos)
        self.display_resize_handles(selected_annotation)

    def detect_resize_handle(self, annotation, current_pos):
        """Detect the closest resize handle to the current position."""
        handles = self.get_handles(annotation)

        closest_handle = (None, None)
        min_distance = float('inf')

        for handle, point in handles.items():
            distance = self.calculate_distance(current_pos, point)
            if distance < min_distance:
                min_distance = distance
                closest_handle = (handle, point)

        if closest_handle[0] and self.calculate_distance(current_pos, closest_handle[1]) <= self.buffer:
            return closest_handle[0]

        return None  # Default if no handle is found

    def get_handles(self, annotation):
        """Return the handles based on the annotation type."""
        if isinstance(annotation, RectangleAnnotation):
            return self.get_rectangle_handles(annotation)
        elif isinstance(annotation, PolygonAnnotation):
            return self.get_polygon_handles(annotation)
        return {}

    def calculate_distance(self, point1, point2):
        """Calculate the distance between two points."""
        return (point1 - point2).manhattanLength()

    def get_rectangle_handles(self, annotation):
        """Return resize handles for a rectangle annotation."""
        top_left = annotation.top_left
        bottom_right = annotation.bottom_right
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

    def get_polygon_handles(self, annotation):
        """Return resize handles for a polygon annotation."""
        return {f"point_{i}": QPointF(point.x(), point.y()) for i, point in enumerate(annotation.points)}

    def display_resize_handles(self, annotation):
        """Display resize handles for the given annotation."""
        self.remove_resize_handles()
        handles = self.get_handles(annotation)
        handle_size = 10

        for handle, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - handle_size // 2,
                                           point.y() - handle_size // 2,
                                           handle_size,
                                           handle_size)
            
            ellipse.setPen(QPen(annotation.label.color))
            ellipse.setBrush(QBrush(annotation.label.color))
            self.annotation_window.scene.addItem(ellipse)
            self.resize_handles.append(ellipse)

    def resize_annotation(self, annotation, new_pos):
        """Resize the annotation based on the resize handle."""
        if annotation and hasattr(annotation, 'resize'):
            annotation.resize(self.resize_handle, new_pos)

    def remove_resize_handles(self):
        """Remove any displayed resize handles."""
        for handle in self.resize_handles:
            self.annotation_window.scene.removeItem(handle)
        self.resize_handles.clear()