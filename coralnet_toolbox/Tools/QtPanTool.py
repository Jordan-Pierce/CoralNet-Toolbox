import warnings

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QMouseEvent

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PanTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.ClosedHandCursor
        self.default_cursor = Qt.ArrowCursor  # Explicitly set

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.annotation_window.pan_active = True
            self.annotation_window.pan_start = event.pos()
            self.annotation_window.setCursor(self.cursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.annotation_window.pan_active:
            self.pan(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.annotation_window.pan_active = False
            self.annotation_window.setCursor(self.default_cursor)

    def pan(self, pos):
        """Pan the view with boundary constraints to keep image in view"""
        if not self.annotation_window.active_image:
            return
            
        # Calculate the delta movement
        delta = pos - self.annotation_window.pan_start
        self.annotation_window.pan_start = pos
        
        # Get current scrollbar values
        h_scroll = self.annotation_window.horizontalScrollBar()
        v_scroll = self.annotation_window.verticalScrollBar()
        
        # Calculate new positions with delta applied
        new_x = h_scroll.value() - delta.x()
        new_y = v_scroll.value() - delta.y()
        
        # Apply the new scroll positions
        h_scroll.setValue(new_x)
        v_scroll.setValue(new_y)
    
    def ensure_image_in_view(self):
        """Ensure the image stays within view boundaries and centered appropriately"""
        if not self.annotation_window.scene or not self.annotation_window.active_image:
            return
            
        # Get current visible area in scene coordinates
        visible_rect = self.annotation_window.viewportToScene()
        
        # Get scene rect (image boundaries)
        scene_rect = self.annotation_window.scene.sceneRect()
        
        # Check if the visible area is larger than the scene, which means 
        # the image is smaller than the viewport and needs centering
        if visible_rect.width() > scene_rect.width() or visible_rect.height() > scene_rect.height():
            # Center the image in the view
            self.annotation_window.centerOn(scene_rect.center())
            
        # Update the view dimensions in the status bar
        self.annotation_window.viewChanged.emit(*self.annotation_window.get_image_dimensions())