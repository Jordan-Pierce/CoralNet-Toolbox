import warnings

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZoomTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.ArrowCursor
        self.min_zoom_factor = 1.0
        self.initial_zoom = 1.0

    def calculate_min_zoom(self):
        """Calculate minimum zoom factor to fit image in view"""
        scene = self.annotation_window.scene
        if not scene:
            return 1.0
            
        view_rect = self.annotation_window.viewport().rect()
        scene_rect = scene.sceneRect()
        
        # Calculate ratios to fit width and height
        width_ratio = view_rect.width() / scene_rect.width() if scene_rect.width() else 1.0
        height_ratio = view_rect.height() / scene_rect.height() if scene_rect.height() else 1.0
        
        # Store initial zoom when first calculated
        min_zoom = max(min(width_ratio, height_ratio), 0.1)
        if self.initial_zoom == 1.0:
            self.initial_zoom = min_zoom
            self.annotation_window.zoom_factor = min_zoom
            
        return min_zoom

    def reset_zoom(self):
        """Reset zoom state"""
        self.initial_zoom = 1.0
        self.min_zoom_factor = 1.0
        self.annotation_window.zoom_factor = 1.0
    
    def center_image_in_view(self):
        """Center the image in the viewport"""
        if not self.annotation_window.scene:
            return
            
        # Get the scene rect (image boundaries)
        scene_rect = self.annotation_window.scene.sceneRect()
        
        # Center the view on the scene rect
        self.annotation_window.centerOn(scene_rect.center())
        
        # Update view dimensions in status bar
        self.annotation_window.viewChanged.emit(
            *self.annotation_window.get_image_dimensions())

    def wheelEvent(self, event: QMouseEvent):
        """Handle zoom events with proper boundary constraints and centering"""
        if not self.annotation_window.active_image:
            return
            
        # Determine zoom direction
        if event.angleDelta().y() > 0:
            factor = 1.1  # Zoom in
        else:
            factor = 0.9  # Zoom out

        # Calculate new zoom level
        new_zoom = self.annotation_window.zoom_factor * factor
        min_zoom = self.calculate_min_zoom()
        
        # Prevent zooming out beyond minimum (image boundaries)
        if new_zoom < min_zoom and factor < 1:
            # Instead of returning, set to minimum zoom and force centering
            new_zoom = min_zoom
            factor = new_zoom / self.annotation_window.zoom_factor
            
            # Apply zoom and save the new factor
            self.annotation_window.scale(factor, factor)
            self.annotation_window.zoom_factor = new_zoom
            
            # Center the image in view
            self.center_image_in_view()
            return
            
        # For normal zooming, store position before zoom
        old_pos = self.annotation_window.mapToScene(event.pos())
        
        # Apply zoom and save the new factor
        self.annotation_window.scale(factor, factor)
        self.annotation_window.zoom_factor = new_zoom
        
        # Correct position after zoom for a more natural zoom effect
        new_pos = self.annotation_window.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.annotation_window.translate(delta.x(), delta.y())
        
        # When zoomed out completely, ensure perfect centering
        if abs(new_zoom - min_zoom) < 0.01:
            self.center_image_in_view()