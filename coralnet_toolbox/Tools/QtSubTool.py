from abc import ABC

from PyQt5.QtGui import QMouseEvent, QKeyEvent


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SubTool(ABC):
    """
    Abstract Base Class for an operation within the SelectTool.
    Each SubTool represents a specific state or mode, like moving or resizing.
    """
    def __init__(self, parent_tool):
        self.parent_tool = parent_tool
        self.annotation_window = parent_tool.annotation_window
        self.is_active = False

    def activate(self, event: QMouseEvent, **kwargs):
        """Called when the sub-tool becomes active."""
        self.is_active = True

    def deactivate(self):
        """Called to clean up when the sub-tool is no longer active."""
        self.is_active = False

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for this specific sub-tool."""
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for this specific sub-tool."""
        pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for this specific sub-tool."""
        pass

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events for this specific sub-tool."""
        pass

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events for this specific sub-tool."""
        pass