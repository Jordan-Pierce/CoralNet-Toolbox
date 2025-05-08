import warnings

from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem, QMessageBox,
                             QGraphicsLineItem, QGraphicsPathItem)

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectTool(Tool):
    """Tool for selecting, moving, resizing, and cutting annotations."""
    
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.PointingHandCursor
        
        # State flags
        self.resizing = False
        self.moving = False
        self.rectangle_selection = False
        self.cutting_mode = False
        self.drawing_cut_line = False
        
        # Resize handle tracking
        self.resize_handle = None
        self.resize_start_pos = None
        self.resize_handles = []
        self.buffer = 50
        
        # Movement tracking
        self.move_start_pos = None
        
        # Rectangle selection
        self.selection_rectangle = None
        self.selection_start_pos = None

        # Cutting tool
        self.cutting_path = None        # QGraphicsPathItem that displays the cutting path
        self.cutting_points = []        # List of points in the cutting line
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect signals for annotation changes."""
        self.annotation_window.annotationSelected.connect(self.clear_resize_handles)
        self.annotation_window.annotationSizeChanged.connect(self.clear_resize_handles)
        self.annotation_window.annotationDeleted.connect(self.clear_resize_handles)
        
    def activate(self):
        """Activate the selection tool and set appropriate cursor."""
        super().activate()
        # Reset all tool states to their defaults
        self.resizing = False
        self.moving = False
        self.rectangle_selection = False
        self.cutting_mode = False
        self.drawing_cut_line = False
        self.resize_handle = None
        self.resize_start_pos = None
        self.move_start_pos = None
        
        # Clean up any graphics items
        self.remove_resize_handles()
        
        if self.selection_rectangle:
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None
            
        if self.cutting_path:
            self.annotation_window.scene.removeItem(self.cutting_path)
            self.cutting_path = None
            self.cutting_points = []
            
        # Reset cursor
        self.annotation_window.viewport().setCursor(self.cursor)

    def deactivate(self):
        """Deactivate the selection tool and clean up."""
        self.cutting_mode = False
        self.drawing_cut_line = False
        self.resizing = False
        self.moving = False
        self.rectangle_selection = False
        
        # Clean up any graphics items
        self.remove_resize_handles()
        
        if self.selection_rectangle:
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None
        
        if self.cutting_path:
            self.annotation_window.scene.removeItem(self.cutting_path)
            self.cutting_path = None
            
        super().deactivate()
    
    def wheelEvent(self, event: QMouseEvent):
        """Handle zoom using the mouse wheel."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.annotation_window.set_annotation_size(delta=16 if delta > 0 else -16)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events to select annotations or start/end cutting line."""
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        # If in cutting mode, handle cutting line drawing
        if self.cutting_mode and event.button() == Qt.LeftButton:
            position = self.annotation_window.mapToScene(event.pos())
            
            if not self.drawing_cut_line:
                # Start drawing the cutting line
                self.start_drawing_cut_line(position)
            else:
                # Finish drawing the cutting line
                self.finish_drawing_cut_line()
            return

        if event.button() == Qt.LeftButton:
            position = self.annotation_window.mapToScene(event.pos())
            items = self.get_clickable_items(position)
            
            # Check for resize handle interactions when Ctrl+Shift are pressed
            if self.resize_handles:
                if (event.modifiers() & Qt.ShiftModifier) and (event.modifiers() & Qt.ControlModifier):
                    if self._handle_resize_press(items):
                        return

            # Handle rectangle selection with Ctrl
            if event.modifiers() & Qt.ControlModifier:
                self._init_rectangle_selection(position)

            # Select annotation based on click position
            selected_annotation = self.select_annotation(position, items, event.modifiers())
            if selected_annotation:
                self.init_drag_or_resize(selected_annotation, position, event.modifiers())
                
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for resizing, moving, drawing selection rectangle, or cutting line."""
        current_pos = self.annotation_window.mapToScene(event.pos())
        
        if self.rectangle_selection:
            self.update_selection_rectangle(current_pos)
        elif self.resizing:
            self.handle_resize(current_pos)
        elif self.moving:
            self._handle_annotation_move(current_pos)
        elif self.drawing_cut_line:
            self.update_cut_line(current_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events to stop moving, resizing, or finalize selection rectangle."""
        if self.rectangle_selection:
            self.finalize_selection_rectangle()
            self._cleanup_rectangle_selection()

        # Reset state (don't reset cutting mode here)
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.resize_start_pos = None
        self.annotation_window.drag_start_pos = None

    def keyPressEvent(self, event):
        """Handle key press events to show resize handles and process hotkeys."""
        # Handle Ctrl+Shift for resize handles
        if len(self.annotation_window.selected_annotations) == 1:
            if event.modifiers() & Qt.ShiftModifier and event.modifiers() & Qt.ControlModifier:
                self.display_resize_handles(self.annotation_window.selected_annotations[0])
            
        # Handle Ctrl+Spacebar to update annotation with top machine confidence
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Space:
            self.update_with_top_machine_confidence()
            
        # Handle Ctrl+C to combine selected annotations, enter cutting mode, or cancel cutting mode
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            if self.cutting_mode:
                # Cancel cutting mode if already in it (pressing Ctrl+C again)
                self.cancel_cutting_mode()
            elif len(self.annotation_window.selected_annotations) > 1:
                self.combine_selected_annotations()
            elif len(self.annotation_window.selected_annotations) == 1:
                self.start_cutting_mode()
                
        # Handle Backspace to cancel cutting mode
        if event.key() == Qt.Key_Backspace and self.cutting_mode:
            self.cancel_cutting_mode()
            
    def keyReleaseEvent(self, event):
        """Handle key release events to hide resize handles."""
        # Hide resize handles if either Ctrl or Shift is released
        if not (event.modifiers() & Qt.ShiftModifier and event.modifiers() & Qt.ControlModifier):
            self.remove_resize_handles()
                
    def update_with_top_machine_confidence(self):
        """Update the selected annotation(s) with their top machine confidence predictions."""
        if not self.annotation_window.selected_annotations:
            return
        
        updated_count = 0
        for annotation in self.annotation_window.selected_annotations:
            if not annotation.machine_confidence:
                continue
            
            # Get the top confidence prediction
            top_label = next(iter(annotation.machine_confidence))
            annotation.update_user_confidence(top_label)
            updated_count += 1
        
        # Update UI based on selection state
        if len(self.annotation_window.selected_annotations) == 1 and updated_count == 1:
            selected_annotation = self.annotation_window.selected_annotations[0]
            self.annotation_window.labelSelected.emit(selected_annotation.label.id)
        else:
            self.annotation_window.unselect_annotations()
    
    def _init_rectangle_selection(self, position):
        """Initialize rectangle selection mode."""
        self.rectangle_selection = True
        self.selection_start_pos = position
        self.selection_rectangle = QGraphicsRectItem()
        
        # Get the thickness for the selection rectangle
        width = self.graphics_utility.get_rectangle_graphic_thickness(self.annotation_window)
        
        # Style the selection rectangle
        pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
        pen.setWidth(width)
        self.selection_rectangle.setPen(pen)
        
        self.selection_rectangle.setRect(QRectF(position, position))
        self.annotation_window.scene.addItem(self.selection_rectangle)
    
    def _cleanup_rectangle_selection(self):
        """Clean up rectangle selection mode."""
        self.rectangle_selection = False
        self.selection_start_pos = None
        self.annotation_window.scene.removeItem(self.selection_rectangle)
        self.selection_rectangle = None
    
    def update_selection_rectangle(self, current_pos):
        """Update the selection rectangle while dragging."""
        if self.selection_rectangle and self.annotation_window.cursorInWindow(current_pos, mapped=True):
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_rectangle.setRect(rect)

    def finalize_selection_rectangle(self):
        """Finalize the selection rectangle and select annotations within it."""
        locked_label = self.get_locked_label()

        if self.selection_rectangle:
            rect = self.selection_rectangle.rect()
            # Don't clear previous selection when using rectangle selection
            for annotation in self.annotation_window.get_image_annotations():
                if rect.contains(annotation.center_xy):
                    # Check if a label is locked, and only select annotations with this label
                    if locked_label and annotation.label.id != locked_label.id:
                        continue
                    if annotation not in self.annotation_window.selected_annotations:
                        self.annotation_window.select_annotation(annotation, True)
    
    def get_locked_label(self):
        """Get the locked label if it exists."""
        return self.annotation_window.main_window.label_window.locked_label

    def get_clickable_items(self, position):
        """Get items that can be clicked in the scene."""
        items = self.annotation_window.scene.items(position)
        # Include ellipse items (resize handles) in clickable items
        clickable_items = []
        for item in items:
            if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem)):
                clickable_items.append(item)
        return clickable_items

    def select_annotation(self, position, items, modifiers):
        """Select an annotation based on the click position."""
        locked_label = self.get_locked_label()

        # Find items sorted by proximity to center
        center_proximity_items = [
            (item, self.calculate_distance(position, self.get_item_center(item)))
            for item in items if self.is_annotation_clickable(item, position)
        ]
        center_proximity_items.sort(key=lambda x: x[1])

        # Select the closest annotation
        for item, _ in center_proximity_items:
            annotation_id = item.data(0)
            selected_annotation = self.annotation_window.annotations_dict.get(annotation_id)

            if selected_annotation:
                # Check if a label is locked
                if locked_label and selected_annotation.label.id != locked_label.id:
                    continue  # Skip annotations with a different label

                ctrl_pressed = modifiers & Qt.ControlModifier
                if selected_annotation in self.annotation_window.selected_annotations and ctrl_pressed:
                    # Unselect the annotation if Ctrl is pressed and it is already selected
                    self.annotation_window.unselect_annotation(selected_annotation)
                    return None
                else:
                    return self.handle_selection(selected_annotation, modifiers)

        return None

    def is_annotation_clickable(self, item, position):
        """Check if the clicked position is within the annotation."""
        annotation = self.annotation_window.annotations_dict.get(item.data(0))
        return annotation and annotation.contains_point(position)

    def handle_selection(self, selected_annotation, modifiers):
        """Handle annotation selection logic."""
        locked_label = self.get_locked_label()
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

            # Check if a label is locked
            if locked_label and selected_annotation.label.id != locked_label.id:
                return None

            # Add to selection
            self.annotation_window.select_annotation(selected_annotation, True)
            return selected_annotation
    
    def _handle_annotation_move(self, current_pos):
        """Handle moving the selected annotation with checks."""
        if not self.annotation_window.selected_annotations:
            return

        selected_annotation = self.annotation_window.selected_annotations[0]
        if not self.annotation_window.is_annotation_moveable(selected_annotation):
            self.moving = False
        else:
            self.handle_move(current_pos)
    
    def _handle_resize_press(self, items):
        """Handle mouse press on resize handles."""
        for item in items:
            if item in self.resize_handles:
                handle_name = item.data(1)
                if handle_name and len(self.annotation_window.selected_annotations) == 1:
                    self.resize_handle = handle_name
                    self.resizing = True
                    return True
        return False

    def init_drag_or_resize(self, selected_annotation, position, modifiers):
        """Initialize dragging or resizing based on the current state."""
        self.annotation_window.drag_start_pos = position
        self.move_start_pos = position

        if (modifiers & Qt.ShiftModifier) and (modifiers & Qt.ControlModifier):
            self.resize_handle = self.detect_resize_handle(selected_annotation, position)
            if self.resize_handle:
                self.resizing = True
        else:
            self.moving = True

    def handle_move(self, current_pos):
        """Handle moving the selected annotation."""
        if not self.annotation_window.selected_annotations:
            return

        selected_annotation = self.annotation_window.selected_annotations[0]
        delta = current_pos - self.move_start_pos
        new_center = selected_annotation.center_xy + delta

        if self.annotation_window.cursorInWindow(current_pos, mapped=True):
            self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
            self.move_start_pos = current_pos

    def handle_resize(self, current_pos):
        """Handle resizing the selected annotation."""
        if not self.annotation_window.selected_annotations:
            return

        selected_annotation = self.annotation_window.selected_annotations[0]
        if not self.annotation_window.is_annotation_moveable(selected_annotation):
            return

        self.resize_annotation(selected_annotation, current_pos)
        self.display_resize_handles(selected_annotation)
    
    def clear_resize_handles(self, annotation_id=None):
        """Clear resize handles if annotations change."""
        self.remove_resize_handles()

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

        return None

    def get_handles(self, annotation):
        """Return the handles based on the annotation type."""
        if isinstance(annotation, RectangleAnnotation):
            return self.get_rectangle_handles(annotation)
        elif isinstance(annotation, PolygonAnnotation):
            return self.get_polygon_handles(annotation)
        return {}

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
        handle_size = self.graphics_utility.get_handle_size(self.annotation_window)

        for handle, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - handle_size // 2,
                                           point.y() - handle_size // 2,
                                           handle_size,
                                           handle_size)
            
            # Make handle more visible with a contrasting color and thicker border
            handle_color = QColor(annotation.label.color)
            border_color = QColor(255 - handle_color.red(), 
                                  255 - handle_color.green(), 
                                  255 - handle_color.blue())
            
            ellipse.setPen(QPen(border_color, 2))
            ellipse.setBrush(QBrush(handle_color))
            
            # Store the handle name as data in the item
            ellipse.setData(1, handle)
            
            # Make the handle shape well-defined for hit detection
            ellipse.setAcceptHoverEvents(True)
            ellipse.setAcceptedMouseButtons(Qt.LeftButton)
            
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

    def calculate_distance(self, point1, point2):
        """Calculate the distance between two points."""
        return (point1 - point2).manhattanLength()

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
        has_rectangles = any(isinstance(annotation, RectangleAnnotation) for annotation in selected_annotations)
        
        # Handle cases where we can't combine different types
        if has_rectangles and (has_patches or has_polygons):
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
        elif has_polygons:
            combined_annotation = PolygonAnnotation.combine(selected_annotations)
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

    def start_cutting_mode(self):
        """Start cutting mode for the currently selected annotation."""
        if len(self.annotation_window.selected_annotations) != 1:
            return
        
        # Check if any annotations have machine confidence
        if any(not annotation.verified for annotation in self.annotation_window.selected_annotations):
            QMessageBox.warning(
                self.annotation_window,
                "Cannot Cut",
                "Cannot cut annotations with machine confidence. Confirm predictions (Ctrl+Space) first."
            )
            return
            
        self.cutting_mode = True
        self.drawing_cut_line = False
        self.cutting_points = []
        
        # Change cursor to indicate cutting mode
        self.annotation_window.viewport().setCursor(Qt.CrossCursor)

    def start_drawing_cut_line(self, position):
        """Start drawing the cutting line from the given position."""
        self.drawing_cut_line = True
        self.cutting_points = [position]
        
        # Create path graphics item for the cutting line
        path = QPainterPath()
        path.moveTo(position)
        
        # Get line thickness for the cutting path
        line_thickness = self.graphics_utility.get_selection_thickness(self.annotation_window)
        
        # Create the cutting path item
        self.cutting_path = QGraphicsPathItem(path)
        self.cutting_path.setPen(QPen(Qt.red, line_thickness, Qt.DashLine))
        self.annotation_window.scene.addItem(self.cutting_path)

    def update_cut_line(self, position):
        """Update the cutting line as the mouse moves, adding points to track the path."""
        if not self.drawing_cut_line or not self.cutting_path:
            return
        
        # Only add a new point if it's different enough from the last point to avoid excessive points
        if not self.cutting_points or (position - self.cutting_points[-1]).manhattanLength() > 5:
            self.cutting_points.append(position)
            
            # Update the path
            path = QPainterPath()
            path.moveTo(self.cutting_points[0])
            
            # Add each point as a line segment
            for point in self.cutting_points[1:]:
                path.lineTo(point)
                
            self.cutting_path.setPath(path)

    def finish_drawing_cut_line(self):
        """Finish drawing the cutting line and perform the cut."""
        if not self.drawing_cut_line or len(self.cutting_points) < 2:
            self.cancel_cutting_mode()
            return
        
        # Perform the cut
        self.cut_selected_annotation(self.cutting_points)
        
        # Exit the cutting mode
        self.drawing_cut_line = False
        self.cancel_cutting_mode()

    def cut_selected_annotation(self, cutting_points):
        """Finalize the cutting operation using the created cutting line."""
        if not self.cutting_mode or len(cutting_points) < 2:
            self.cancel_cutting_mode()
            return  # Not enough points to cut
        
        annotation = self.annotation_window.selected_annotations[0]
        
        # Call the appropriate cut method based on annotation type
        if isinstance(annotation, RectangleAnnotation):
            new_annotations = RectangleAnnotation.cut(annotation, cutting_points)
        elif isinstance(annotation, PolygonAnnotation):
            new_annotations = PolygonAnnotation.cut(annotation, cutting_points)
        else:
            self.cancel_cutting_mode()
            return  # Unsupported annotation type
            
        if not new_annotations:
            self.cancel_cutting_mode()  # No new annotations created
            return
            
        # Remove the original annotation
        self.annotation_window.delete_selected_annotations()
        
        # Unselect the annotations
        self.annotation_window.unselect_annotations()
        
        # Add the new cut annotations
        for new_annotation in new_annotations:
            self.annotation_window.add_annotation_from_tool(new_annotation)
            
        # Clear the cutting path
        self.cancel_cutting_mode()
        
    def cancel_cutting_mode(self):
        """Cancel cutting mode and clean up."""
        self.cutting_mode = False
        self.drawing_cut_line = False
        self.cutting_points = []
        
        if self.cutting_path:
            self.annotation_window.scene.removeItem(self.cutting_path)
            self.cutting_path = None
            
        # Reset cursor
        self.annotation_window.viewport().setCursor(self.cursor)
        # Update the annotation window
        self.annotation_window.scene.update()

