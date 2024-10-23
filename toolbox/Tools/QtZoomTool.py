import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

from toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZoomTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.ArrowCursor

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.annotation_window.zoom_factor *= factor
        self.annotation_window.scale(factor, factor)

        if self.annotation_window.selected_tool in ["select", "patch", "rectangle", "polygon", "sam"]:
            # Update the cursor for the given tool
            self.annotation_window.setCursor(self.annotation_window.tools[self.annotation_window.selected_tool].cursor)

            if self.annotation_window.selected_tool in ['"patch']:
                # Update the cursor annotation when increasing the size
                self.annotation_window.toggle_cursor_annotation()
        else:
            # Use the default cursor
            self.annotation_window.setCursor(Qt.ArrowCursor)