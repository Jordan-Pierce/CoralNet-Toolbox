import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os 

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QComboBox, QSpinBox, QHBoxLayout,
                             QWidget, QStackedWidget, QGridLayout, QMessageBox,
                             QDialog, QListWidget, QPushButton, QFileDialog,
                             QGraphicsView)
from PyQt5.QtCore import Qt                           

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileSizeInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Tile Size", parent)
        layout = QFormLayout(self)

        # Width input in pixels
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 10000)
        self.width_spin.setValue(1024)

        # Height input in pixels
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 10000)
        self.height_spin.setValue(768)

        layout.addRow("Width:", self.width_spin)
        layout.addRow("Height:", self.height_spin)
        
    def get_value(self):
        """
        Get the current tile size values.
        
        Returns:
            tuple: (width, height)
        """
        return self.width_spin.value(), self.height_spin.value()

    def get_sizes(self, image_width=None, image_height=None, validate=True):
        """
        Get tile sizes with optional validation against image dimensions.
        
        Args:
            image_width (int, optional): Width of source image
            image_height (int, optional): Height of source image
            validate (bool): Whether to validate against image dimensions
            
        Returns:
            tuple: (width, height) if valid
            None: if invalid when validation enabled
        """
        tile_size = self.get_value()
        
        try:
            # Basic type validation always performed
            if not isinstance(tile_size, tuple) or len(tile_size) != 2:
                raise ValueError("Tile size must be a tuple of two values")
                
            width, height = tile_size
            
            # Size validation only if requested and dimensions provided
            if validate and image_width and image_height:
                if width > image_width:
                    raise ValueError(f"Tile width ({width}) cannot exceed image width ({image_width})")
                if height > image_height:
                    raise ValueError(f"Tile height ({height}) cannot exceed image height ({image_height})")
        
            return width, height
            
        except ValueError as e:
            if validate:
                QMessageBox.warning(self, "Invalid Tile Size", str(e))
                return None
            return tile_size