import random
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem

class TileInference:
    def __init__(self, annotation_window):
        self.annotation_window = annotation_window
        self.tile_graphics = []

    def update_tile_graphics(self):
        self.clear_tile_graphics()

        if not self.annotation_window.image_pixmap:
            return

        image_width = self.annotation_window.image_pixmap.width()
        image_height = self.annotation_window.image_pixmap.height()

        tile_width = self.shape_x
        tile_height = self.shape_y
        overlap_x = self.overlap_x
        overlap_y = self.overlap_y
        margins = self.margins

        x_start = margins[3]
        y_start = margins[0]
        x_end = image_width - margins[1]
        y_end = image_height - margins[2]

        x = x_start
        while x < x_end:
            y = y_start
            while y < y_end:
                tile = QGraphicsRectItem(x, y, tile_width, tile_height)
                tile_color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
                tile.setBrush(QBrush(tile_color))
                tile.setPen(QPen(QColor(0, 0, 0), 1, Qt.DotLine))
                tile.setOpacity(0.5)
                self.annotation_window.scene.addItem(tile)
                self.tile_graphics.append(tile)
                y += tile_height - overlap_y
            x += tile_width - overlap_x

    def clear_tile_graphics(self):
        for tile in self.tile_graphics:
            self.annotation_window.scene.removeItem(tile)
        self.tile_graphics = []
