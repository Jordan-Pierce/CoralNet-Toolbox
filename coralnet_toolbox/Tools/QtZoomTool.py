import warnings

from PyQt5.QtCore import Qt
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

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        new_zoom = self.annotation_window.zoom_factor * factor
        min_zoom = self.calculate_min_zoom()
        
        # Prevent zooming out beyond minimum
        if new_zoom < min_zoom and factor < 1:
            return

        self.annotation_window.zoom_factor = new_zoom
        self.annotation_window.scale(factor, factor)