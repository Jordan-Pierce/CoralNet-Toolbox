import warnings

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush, QColor
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsSimpleTextItem, QMessageBox, 
                             QGraphicsPixmapItem, QGraphicsItemGroup, QGraphicsLineItem)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.QtWorkArea import WorkArea

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class WorkAreaTool(Tool):
    """Tool for creating and managing work areas (rectangular regions) on an image."""
    
    def __init__(self, annotation_window):
        """Initialize the work area tool with the annotation window."""
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        
        # State for drawing work areas
        self.drawing = False
        self.start_pos = None
        self.current_rect = None
        self.work_areas = []  # List to store WorkArea objects
        
        # Style settings - removed the semi-transparent brush
        self.work_area_pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
        
        # Track if Ctrl key is pressed
        self.ctrl_pressed = False
        
        # Connect to the annotation window's image load signal
        self.annotation_window.imageLoaded.connect(self.on_image_loaded)
        
    def activate(self):
        """Activate the work area tool and set the appropriate cursor."""
        super().activate()
        self.annotation_window.viewport().setCursor(self.cursor)
        
        # Load existing work areas for the current image
        self.load_work_areas()
        
    def deactivate(self):
        """Deactivate the work area tool and clean up."""
        if self.drawing:
            self.cancel_drawing()
            
        # Remove all work area graphics from the scene without clearing the raster data
        self.hide_work_area_graphics()
            
        super().deactivate()
        
    def hide_work_area_graphics(self):
        """Remove all work area graphics from the scene without clearing the data in the raster."""
        # Create a copy to safely iterate
        for work_area in self.work_areas:
            if work_area.graphics_item and work_area.graphics_item.scene():
                # Remove from scene but keep the object
                self.annotation_window.scene.removeItem(work_area.graphics_item)
                work_area.graphics_item = None
        
        # Clear our internal list since we removed the graphics items
        # This doesn't remove them from the raster
        self.work_areas = []
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events to start or finish drawing a work area."""
        if not self.annotation_window.active_image or not self.annotation_window.cursorInWindow(event.pos()):
            return
            
        # If Ctrl is pressed, don't start drawing - user is likely trying to delete work areas
        if self.ctrl_pressed:
            return
            
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            if not self.drawing:
                # Start drawing a new work area
                self.start_drawing(scene_pos)
            else:
                # Finish drawing the current work area
                self.finish_drawing(scene_pos)
                
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events to update the work area while drawing."""
        if not self.drawing or not self.annotation_window.active_image:
            return
            
        scene_pos = self.annotation_window.mapToScene(event.pos())
        self.update_drawing(scene_pos)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        # For click-drag-click interactions, we don't need to handle release
        pass
        
    def keyPressEvent(self, event):
        """Handle key press events for work area tool operations."""
        modifiers = event.modifiers()
        key = event.key()
        
        # Check for modifier combinations
        is_shift_ctrl = (modifiers & Qt.ShiftModifier) and (modifiers & Qt.ControlModifier)
        
        # Clear all work areas
        if is_shift_ctrl and key == Qt.Key_Backspace:
            self.clear_work_areas()
            return
        
        # Show remove buttons and change cursor
        if is_shift_ctrl:
            self.ctrl_pressed = True
            self.update_remove_buttons_visibility(True)
            self.annotation_window.viewport().setCursor(Qt.PointingHandCursor)
            return
        
        # Create work area from current view
        if key == Qt.Key_Space and self.annotation_window.active_image:
            self.create_work_area_from_current_view()
            return
        
        # Cancel current drawing
        if key == Qt.Key_Backspace and self.drawing:
            self.cancel_drawing()
            return
    
    def keyReleaseEvent(self, event):
        """Handle key release events."""
        # Track Ctrl key for showing/hiding remove buttons and resetting cursor
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = False
            self.update_remove_buttons_visibility(False)
            # Change cursor back to cross when Ctrl is released
            self.annotation_window.viewport().setCursor(self.cursor)
    
    def start_drawing(self, pos):
        """Start drawing a work area at the given position."""
        self.drawing = True
        self.start_pos = pos
        
        # Create an initial rectangle item for visual feedback - no brush
        self.current_rect = QGraphicsRectItem(QRectF(pos.x(), pos.y(), 0, 0))
        
        # Get the width
        width = self.graphics_utility.get_rectangle_graphic_thickness(self.annotation_window)
        
        # Set the pen properties
        self.work_area_pen.setWidth(width)
        self.current_rect.setPen(self.work_area_pen)
        self.annotation_window.scene.addItem(self.current_rect)
        
    def update_drawing(self, pos):
        """Update the work area rectangle as the mouse moves."""
        if not self.current_rect:
            return
            
        # Create a normalized rectangle (handles drawing in any direction)
        rect = QRectF(self.start_pos, pos).normalized()
        
        # Constrain to image boundaries for visual feedback during drawing
        constrained_rect = self.constrain_rect_to_image_bounds(rect)
        
        self.current_rect.setRect(constrained_rect)
        
    def finish_drawing(self, pos):
        """Finish drawing the work area and add it to the list."""
        if not self.current_rect:
            self.cancel_drawing()
            return
            
        # Get the final rectangle
        rect = QRectF(self.start_pos, pos).normalized()
        
        # Ensure the rectangle stays within image boundaries
        rect = self.constrain_rect_to_image_bounds(rect)
        
        # Check if the work area is too small
        if rect.width() < 10 or rect.height() < 10:
            QMessageBox.warning(
                self.annotation_window,
                "Work Area Too Small",
                "Work area is too small. Please create a larger area."
            )
            self.cancel_drawing()
            return
            
        # Update the current rectangle with the final dimensions
        self.current_rect.setRect(rect)
        
        # Create a WorkArea object from the rect
        work_area = WorkArea.from_rect(rect, self.get_current_image_name())
        
        # Add graphics item reference to the work area
        work_area.graphics_item = self.current_rect
        
        # Add close button but initially hidden unless Ctrl is pressed
        self.add_remove_button(work_area)
        
        # Connect to remove signal
        work_area.removed.connect(self.on_work_area_removed)
        
        # Add to work areas list
        self.work_areas.append(work_area)
        
        # Store the work area in the raster
        self.save_work_area(work_area)
        
        # Reset state
        self.drawing = False
        self.start_pos = None
        self.current_rect = None
        
    def cancel_drawing(self):
        """Cancel the current work area drawing."""
        if self.current_rect:
            self.annotation_window.scene.removeItem(self.current_rect)
            self.current_rect = None
            
        self.drawing = False
        self.start_pos = None
        
    def create_work_area_from_current_view(self):
        """Create a work area based on the current viewport view."""
        viewport_rect = self.annotation_window.viewportToScene()
        
        # Ensure the rectangle stays within image boundaries
        viewport_rect = self.constrain_rect_to_image_bounds(viewport_rect)
        
        # Check if the constrained rectangle is too small
        if viewport_rect.width() < 10 or viewport_rect.height() < 10:
            QMessageBox.information(
                self.annotation_window,
                "Work Area Too Small",
                "The visible area within the image is too small to create a work area."
            )
            return
            
        # Create a WorkArea object from the viewport rect
        work_area = WorkArea.from_rect(viewport_rect, self.get_current_image_name())
        
        # Create a new rectangle for the current view - no brush
        rect_item = QGraphicsRectItem(viewport_rect)
        rect_item.setPen(self.work_area_pen)
        # No brush - transparent interior
        
        # Add to scene
        self.annotation_window.scene.addItem(rect_item)
        
        # Add graphics item reference to the work area
        work_area.graphics_item = rect_item
        
        # Add close button but initially hidden unless Ctrl is pressed
        self.add_remove_button(work_area)
        
        # Connect to remove signal
        work_area.removed.connect(self.on_work_area_removed)
        
        # Add to work areas list
        self.work_areas.append(work_area)
        
        # Store the work area in the raster
        self.save_work_area(work_area)
        
    def constrain_rect_to_image_bounds(self, rect):
        """Constrain a rectangle to stay within the image boundaries."""
        # Get image boundaries
        if self.annotation_window.pixmap_image:
            image_rect = QGraphicsPixmapItem(self.annotation_window.pixmap_image).boundingRect()
            
            # Create a copy of the input rect to avoid modifying the original
            constrained_rect = QRectF(rect)
            
            # If rect is completely outside image, return a minimal valid rect at image edge
            if (rect.right() < 0 or rect.bottom() < 0 or 
                rect.left() > image_rect.width() or rect.top() > image_rect.height()):
                # Return a small rect at the closest corner
                x = max(0, min(rect.x(), image_rect.width() - 10))
                y = max(0, min(rect.y(), image_rect.height() - 10))
                return QRectF(x, y, 10, 10)
            
            # Constrain top-left corner
            if constrained_rect.left() < 0:
                constrained_rect.setLeft(0)
            if constrained_rect.top() < 0:
                constrained_rect.setTop(0)
                
            # Constrain bottom-right corner
            if constrained_rect.right() > image_rect.width():
                constrained_rect.setRight(image_rect.width())
            if constrained_rect.bottom() > image_rect.height():
                constrained_rect.setBottom(image_rect.height())
                
            return constrained_rect
        
        return rect
        
    def add_remove_button(self, work_area):
        """Add a remove button (X) to the work area (initially hidden)."""
        # Get the rectangle dimensions
        rect = work_area.rect
        
        # Create a group item to hold the X shape
        button_size = self.graphics_utility.get_handle_size(self.annotation_window) * 2  # Size of the button
        button_group = QGraphicsItemGroup(work_area.graphics_item)
        
        # Create two diagonal lines to form an X
        line1 = QGraphicsLineItem(0, 0, button_size, button_size, button_group)
        line2 = QGraphicsLineItem(0, button_size, button_size, 0, button_group)
        
        # Set the pen properties - thicker red lines
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        red_pen = QPen(QColor(255, 0, 0), 2)
        red_pen.setWidth(thickness)
        line1.setPen(red_pen)
        line2.setPen(red_pen)
        
        # Position in the top-right corner
        button_group.setPos(rect.right() - button_size - 10, rect.top() + 10)
        
        # Store data to identify this button and its work area
        button_group.setData(0, "remove_button")
        button_group.setData(1, work_area)  # Store reference to the work area
        
        # Make the group item clickable
        button_group.setAcceptedMouseButtons(Qt.LeftButton)
        
        # Store reference to the work area and button in the graphics item
        work_area.graphics_item.setData(0, "work_area")
        work_area.graphics_item.setData(1, work_area)  # Store reference to the work area
        work_area.graphics_item.setData(2, button_group)  # Store reference to the remove button
        
        # Initially hide the remove button
        button_group.setVisible(self.ctrl_pressed)
        
        # When the remove button is clicked, remove the work area
        def on_press(event):
            work_area.remove()
            
        # Override mousePressEvent for the button item
        button_group.mousePressEvent = on_press
        
    def update_remove_buttons_visibility(self, visible):
        """Update the visibility of all remove buttons based on Ctrl key state."""
        for work_area in self.work_areas:
            if work_area.graphics_item:
                remove_button = work_area.graphics_item.data(2)
                if remove_button:
                    remove_button.setVisible(visible)
        
    def on_work_area_removed(self, work_area):
        """Handle when a work area is removed."""
        # Remove from work areas list
        if work_area in self.work_areas:
            self.work_areas.remove(work_area)
            
        # Also remove from the raster's work areas list
        raster = self.get_current_raster()
        if raster:
            raster.remove_work_area(work_area)
        
    def get_current_image_name(self):
        """Get the name of the current image being displayed."""
        if not self.annotation_window.current_image_path:
            return None
            
        return self.annotation_window.current_image_path
        
    def get_current_raster(self):
        """Get the raster object for the current image."""
        image_path = self.get_current_image_name()
        if not image_path:
            return None
            
        return self.annotation_window.main_window.image_window.raster_manager.get_raster(image_path)
        
    def on_image_loaded(self, width, height):
        """Handle when a new image is loaded."""
        # Clear existing work areas first, regardless of whether tool is active
        self.hide_work_area_graphics()
        
        # Only reload work areas if this tool is currently active
        if self.annotation_window.selected_tool == "work_area":
            self.load_work_areas()
            
    def load_work_areas(self):
        """Load existing work areas for the current image."""
        # Clear existing work areas first
        self.clear_work_areas()
        
        # Get the raster for the current image
        raster = self.get_current_raster()
        if not raster:
            return
            
        # Get work areas from the raster's work_areas list, not just metadata
        stored_work_areas = raster.get_work_areas()
        image_path = self.get_current_image_name()
        
        # Create work area objects for each stored rectangle
        for work_area in stored_work_areas:
            # If the work area already has an image path that matches the current image
            # and a valid rect, use it directly
            if work_area.image_path == image_path and work_area.is_valid():
                # Create a graphics item for display - no brush
                rect_item = QGraphicsRectItem(work_area.rect)
                rect_item.setPen(self.work_area_pen)
                # No brush - transparent interior
                
                # Add to scene
                self.annotation_window.scene.addItem(rect_item)
                
                # Add graphics item reference to the work area
                work_area.graphics_item = rect_item
                
                # Add close button (initially hidden unless Ctrl is pressed)
                self.add_remove_button(work_area)
                
                # Connect to remove signal
                work_area.removed.connect(self.on_work_area_removed)
                
                # Add to work areas list
                self.work_areas.append(work_area)
            
    def save_work_area(self, work_area):
        """Save the work area to the current raster."""
        raster = self.get_current_raster()
        if not raster:
            return
            
        # Add to raster's work_areas list
        raster.add_work_area(work_area)
            
    def remove_work_area(self, work_area):
        """Remove a work area."""
        # Just call the work area's remove method
        work_area.remove()
            
    def clear_work_areas(self):
        """Clear all work area graphics from the scene, but don't remove from the raster."""
        # Create a copy of the list to safely iterate and remove
        work_areas_copy = self.work_areas.copy()
        
        # Clear the list first to avoid on_work_area_removed removing during iteration
        self.work_areas = []
        
        # Remove each work area's graphics item from the scene, but don't affect the raster data
        for work_area in work_areas_copy:
            if work_area.graphics_item and work_area.graphics_item.scene():
                self.annotation_window.scene.removeItem(work_area.graphics_item)
                work_area.graphics_item = None
        
    def update_cursor_annotation(self, scene_pos=None):
        """Method required by tool interface but not used for work area tool."""
        pass
        
    def clear_cursor_annotation(self):
        """Method required by tool interface but not used for work area tool."""
        pass
