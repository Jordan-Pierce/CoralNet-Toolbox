import warnings

from importlib.resources import files

import numpy as np

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPen, QColor, QPainter, QPainterPath
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle, QStyleOptionComboBox, QComboBox, QStyleOptionViewItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_icon_path(icon_name):
    """

    :param icon_name:
    :return:
    """
    icon_dir = files('coralnet_toolbox').joinpath('Icons')
    return str(icon_dir.joinpath(icon_name))


def get_icon(icon_name):
    """

    :param icon_name:
    :return:
    """
    return QIcon(get_icon_path(icon_name))


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ColormapDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cache = {}  # Cache to store generated pixmaps

    def create_colormap_pixmap(self, name, rect):
        """Generates the gradient pixmap on demand using pyqtgraph."""
        try:
            width = rect.width() if rect.width() > 0 else 100
            height = rect.height() if rect.height() > 0 else 20
            
            cmap = pg.colormap.get(name)
            lut = cmap.getLookupTable(nPts=width)
            
            if lut.shape[1] == 3:
                alpha = np.full((width, 1), 255, dtype=np.uint8)
                lut = np.hstack((lut, alpha))
            
            img_data = np.tile(lut, (height, 1)).reshape(height, width, 4)
            img_data = np.ascontiguousarray(img_data, dtype=np.uint8)
            
            qimg = QImage(img_data.data, width, height, width * 4, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def paint(self, painter, option, index):
        painter.save()
        
        # Get text (e.g., "Viridis")
        text = index.data(Qt.DisplayRole)

        # Draw Selection Highlight
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Define drawing area with padding
        rect = option.rect.adjusted(2, 2, -2, -2)

        if text == "None":
            # Draw White Rounded Box
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.setBrush(Qt.white)
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 5, 5)
        else:
            # Draw Gradient with Rounded Corners
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), 5, 5)
            painter.setClipPath(path)
            
            cache_key = f"{text}_{rect.width()}_{rect.height()}"
            
            if cache_key not in self.cache:
                pixmap = self.create_colormap_pixmap(text, rect)
                if pixmap:
                    self.cache[cache_key] = pixmap
            
            pixmap = self.cache.get(cache_key)
            if pixmap:
                painter.drawPixmap(rect, pixmap)
                
        # NOTE: We intentionally do NOT draw text here.
        painter.restore()


class ColorComboBox(QComboBox):
    """
    A specific ComboBox that forces the 'button' area to be painted 
    by the delegate, suppressing the text display.
    """
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 1. Initialize Style Option
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        
        # 2. Force text to be empty so the default style doesn't draw it
        option.currentText = ""
        
        # 3. Draw the ComboBox frame and arrow (but no text)
        self.style().drawComplexControl(QStyle.CC_ComboBox, option, painter, self)
        
        # 4. Draw our custom content (the swatch) using the delegate
        if self.currentIndex() != -1:
            # Get the rect where the content (text/swatch) typically goes
            rect = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
            
            # Prepare a ViewItem option for the delegate
            subOpt = QStyleOptionViewItem()
            subOpt.rect = rect
            subOpt.state = option.state 
            
            # Get the model index for the current item
            index = self.model().index(self.currentIndex(), 0)
            
            # Ask the installed delegate to paint the swatch in this area
            if self.itemDelegate():
                self.itemDelegate().paint(painter, subOpt, index)
    
