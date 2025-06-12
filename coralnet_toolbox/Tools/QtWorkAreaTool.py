import warnings

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QGraphicsRectItem, QMessageBox, QGraphicsPixmapItem)

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
        self.default_cursor = Qt.ArrowCursor  # Add this for clarity

        # State for drawing work areas
        self.drawing = False
        self.start_pos = None
        self.current_rect = None
        self.work_areas = []  # List to store WorkArea objects for the current image
        
        # Style settings for drawing the work area rectangle
        self.work_area_pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
        
        # Track if Ctrl key is pressed
        self.ctrl_pressed = False
        self.temporary_work_area = None  # To store the temporary work area
        
        # Connect to the annotation window's image load signal
        self.annotation_window.imageLoaded.connect(self.on_image_loaded)
        
        # Track current image path to detect image changes
        self.current_image_path = None
        
    def activate(self):
        """Activate the work area tool and set the appropriate cursor."""
        self.active = True
        self.annotation_window.viewport().setCursor(self.cursor)
        self.load_work_areas()

    def deactivate(self):
        """Deactivate the work area tool and clean up."""
        if self.drawing:
            self.cancel_drawing()
            
        # Remove all work area graphics from the scene without clearing the raster data
        self.clear_work_area_graphics()
        self.annotation_window.viewport().setCursor(self.default_cursor)
        self.active = False
        
    def clear_work_area_graphics(self):
        """Remove all work area graphics from the scene without clearing the data in the raster."""
        # Create a copy to safely iterate
        for work_area in self.work_areas:
            # Then handle the main graphics item and its children
            if work_area.graphics_item and work_area.graphics_item.scene():
                try:
                    # Remove from scene but keep the object
                    self.annotation_window.scene.removeItem(work_area.graphics_item)
                except RuntimeError as e:
                    print(f"Error removing graphics item: {e}")
                
            # Always clear references, even if scene removal fails
            work_area.graphics_item = None
            work_area.remove_button = None
        
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

        # Ctrl+Alt for temporary work area
        if (modifiers & Qt.ControlModifier) and (modifiers & Qt.AltModifier):
            if not self.temporary_work_area:
                self.temporary_work_area = self._create_temporary_work_area()
                if self.temporary_work_area:
                    self.save_work_area(self.temporary_work_area)  # Save to raster
            return

        # Ctrl+Shift to show remove buttons
        if (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier):
            self.ctrl_pressed = True
            if self.work_areas:  # Only show remove buttons if there are work areas exist
                self.update_remove_buttons_visibility(True)
                self.annotation_window.viewport().setCursor(Qt.PointingHandCursor)
            
            # Clear all work areas (Ctrl+Shift+Backspace)
            if key == Qt.Key_Backspace:
                if self.temporary_work_area:
                    self._remove_temporary_work_area()
                self.clear_work_areas()
            return

        # Ctrl+Space to create a work area from current view
        if key == Qt.Key_Space and self.annotation_window.active_image:
            self.create_work_area_from_current_view()
            return

        # Cancel current drawing (Backspace - without modifiers)
        if key == Qt.Key_Backspace and self.drawing and not (modifiers & Qt.ControlModifier):
            self.cancel_drawing()
            return

    def keyReleaseEvent(self, event):
        """Handle key release events."""
        modifiers = event.modifiers()
        
        # For Ctrl+Alt: remove temporary work area when either key is released
        if (event.key() == Qt.Key_Control or event.key() == Qt.Key_Alt) and not (
            (modifiers & Qt.ControlModifier) and (modifiers & Qt.AltModifier)
        ):
            if self.temporary_work_area is not None:
                self._remove_temporary_work_area()
        
        # For Ctrl+Shift: hide remove buttons when either key is released
        if (event.key() == Qt.Key_Control or event.key() == Qt.Key_Shift) and not (
            (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier)
        ):
            self.ctrl_pressed = False
            self.update_remove_buttons_visibility(False)
            self.annotation_window.viewport().setCursor(self.cursor)

    def _create_temporary_work_area(self):
        """Create a temporary work area from the current view."""
        viewport_rect = self.annotation_window.viewportToScene()
        constrained_viewport_rect = self.constrain_rect_to_image_bounds(viewport_rect)

        if constrained_viewport_rect.width() < 10 or constrained_viewport_rect.height() < 10:
            return None

        work_area = WorkArea.from_rect(constrained_viewport_rect, self.get_current_image_name())
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        work_area.create_graphics(self.annotation_window.scene, thickness)

        if work_area.graphics_item and work_area.graphics_item.scene():
            return work_area

        return None

    def _remove_temporary_work_area(self):
        """Remove the temporary work area from the scene and raster."""
        if self.temporary_work_area is None:
            return

        raster = self.get_current_raster()
        if raster:
            raster.remove_work_area(self.temporary_work_area)

        if self.temporary_work_area.graphics_item and self.temporary_work_area.graphics_item.scene():
            self.annotation_window.scene.removeItem(self.temporary_work_area.graphics_item)
            self.temporary_work_area.graphics_item = None

        self.temporary_work_area = None
        
    def start_drawing(self, pos):
        """Start drawing a work area at the given position."""
        self.drawing = True
        self.start_pos = pos
        
        # Create an initial rectangle item for visual feedback
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
            
        # Remove the temporary drawing rectangle
        if self.current_rect in self.annotation_window.scene.items():
            self.annotation_window.scene.removeItem(self.current_rect)
            
        # Create a WorkArea object from the rect
        work_area = WorkArea.from_rect(rect, self.get_current_image_name())
        
        # Create graphics using the WorkArea's own method
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        work_area.create_graphics(self.annotation_window.scene, thickness)
        
        # Add close button but initially hidden unless Ctrl is pressed
        button_size = self.graphics_utility.get_handle_size(self.annotation_window) * 2
        work_area.create_remove_button(button_size, thickness)
        work_area.set_remove_button_visibility(self.ctrl_pressed)
        
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
        
        # Add work area graphics to the scene in one step
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        button_size = self.graphics_utility.get_handle_size(self.annotation_window) * 2
        
        if work_area.add_to_scene(self.annotation_window.scene, thickness, button_size):
            # Set initial button visibility based on Ctrl state
            work_area.set_remove_button_visibility(self.ctrl_pressed)
            
            # Connect to remove signal
            work_area.removed.connect(self.on_work_area_removed)
            
            # Add to work areas list
            self.work_areas.append(work_area)
            
            # Store the work area in the raster
            self.save_work_area(work_area)
        else:
            print(f"Warning: Failed to add work area to scene: {work_area.to_dict()}")
            
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
        
    def update_remove_buttons_visibility(self, visible):
        """Update the visibility of all remove buttons based on Ctrl key state."""
        for work_area in self.work_areas:
            work_area.set_remove_button_visibility(visible)
        
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
        # Get current image path
        new_image_path = self.get_current_image_name()
        
        # Check if the image has actually changed
        if new_image_path != self.current_image_path:
            # Update our tracking variable
            self.current_image_path = new_image_path
            
            # Always reload work areas when the image changes, regardless of whether this tool is active
            if self.annotation_window.selected_tool == "work_area":
                self.load_work_areas()
        
    def check_and_reload_work_areas(self):
        """
        Check if we need to reload work areas - this can be called when the tool becomes
        visible or when we need to ensure graphics are displayed.
        """
        # If this is the active tool, always reload work areas
        if self.annotation_window.selected_tool == "work_area":
            self.load_work_areas()
            
    def load_work_areas(self):
        """Load existing work areas for the current image."""
        # Remove existing work area graphics from the scene first
        self.clear_work_area_graphics()
        
        # Get the raster for the current image
        raster = self.get_current_raster()
        if not raster:
            return
            
        # Get work areas from the raster's work_areas list
        stored_work_areas = raster.get_work_areas()
        image_path = self.get_current_image_name()
        
        if not stored_work_areas:
            return
            
        # Create graphics for each stored work area
        for work_area in stored_work_areas:
            # If the work area already has an image path that matches the current image
            # and a valid rect, use it directly
            if work_area.image_path == image_path and work_area.is_valid():
                # Ensure work area has no existing graphics references
                # This is important when reloading previously viewed images
                work_area.graphics_item = None
                work_area.remove_button = None
                
                # Add work area graphics to the scene
                thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
                button_size = self.graphics_utility.get_handle_size(self.annotation_window) * 2
                
                if work_area.add_to_scene(self.annotation_window.scene, thickness, button_size):
                    # Set initial button visibility based on Ctrl state
                    work_area.set_remove_button_visibility(self.ctrl_pressed)
                    
                    # Connect to remove signal
                    # Disconnect first to avoid duplicate connections
                    try:
                        work_area.removed.disconnect(self.on_work_area_removed)
                    except TypeError:
                        # Ignore if not connected
                        pass
                    work_area.removed.connect(self.on_work_area_removed)
                    
                    # Add to work areas list
                    self.work_areas.append(work_area)
                else:
                    print(f"Warning: Failed to add work area to scene: {work_area.to_dict()}")
                    # Debug information
                    print(f"  - Graphics item: {work_area.graphics_item}")
                    print(f"  - Remove button: {work_area.remove_button}")
            
    def save_work_area(self, work_area):
        """Save the work area to the current raster."""
        raster = self.get_current_raster()
        if not raster:
            return
            
        # Add to raster's work_areas list
        raster.add_work_area(work_area)
            
    def clear_work_areas(self):
        """Clear all work areas completely - removing both graphics and data from the raster."""
        # Get the current raster first
        raster = self.get_current_raster()
        
        # Create a copy of the list to safely iterate and remove
        work_areas_copy = self.work_areas.copy()
        
        # Clear the list first to avoid on_work_area_removed removing during iteration
        self.work_areas = []
        
        # Remove each work area's graphics item from the scene
        for work_area in work_areas_copy:
            # If the graphics item exists and is in the scene, it will automatically
            # remove all its children (including the remove_button) when removed
            if work_area.graphics_item and work_area.graphics_item.scene():
                self.annotation_window.scene.removeItem(work_area.graphics_item)
                work_area.graphics_item = None
                # We don't need to explicitly remove the button as it was a child item
                # Just make sure we clear our reference to it
                work_area.remove_button = None
                
        # Also clear all work areas from the raster data
        if raster:
            raster.clear_work_areas()
        
    def update_cursor_annotation(self, scene_pos=None):
        """Method required by tool interface but not used for work area tool."""
        pass
        
    def clear_cursor_annotation(self):
        """Method required by tool interface but not used for work area tool."""
        pass