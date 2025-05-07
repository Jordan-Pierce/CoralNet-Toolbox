from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsRectItem


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MarginOverlay(QGraphicsRectItem):
    def __init__(self, image_width, image_height, margins, parent=None):
        super().__init__(parent)
        left, top, right, bottom = margins
        overlays = []
        color = QColor(255, 0, 0, 60)  # semi-transparent red
        # Top margin
        if top > 0:
            overlays.append(QGraphicsRectItem(0, 0, image_width, top, self))
        # Bottom margin
        if bottom > 0:
            overlays.append(QGraphicsRectItem(0, image_height - bottom, image_width, bottom, self))
        # Left margin
        if left > 0:
            overlays.append(QGraphicsRectItem(0, top, left, image_height - top - bottom, self))
        # Right margin
        if right > 0:
            overlays.append(QGraphicsRectItem(image_width - right, top, right, image_height - top - bottom, self))
        for overlay in overlays:
            overlay.setBrush(QBrush(color))
            overlay.setPen(QPen(Qt.NoPen))
        self.overlays = overlays