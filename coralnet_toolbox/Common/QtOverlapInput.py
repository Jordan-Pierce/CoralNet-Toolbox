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


class OverlapInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Overlap", parent)
        layout = QFormLayout(self)

        # Unit selection
        self.value_type = QComboBox()
        self.value_type.addItems(["Pixels", "Percentage"])
        layout.addRow("Unit:", self.value_type)

        # Width and height inputs
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 10000)
        self.width_spin.setValue(0)
        self.width_double = QDoubleSpinBox()
        self.width_double.setRange(0, 1)
        self.width_double.setValue(0)
        self.width_double.setSingleStep(0.1)
        self.width_double.setDecimals(2)
        self.width_double.hide()

        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 10000)
        self.height_spin.setValue(0)
        self.height_double = QDoubleSpinBox()
        self.height_double.setRange(0, 1)
        self.height_double.setValue(0)
        self.height_double.setSingleStep(0.1)
        self.height_double.setDecimals(2)
        self.height_double.hide()

        # Instead of adding separate rows, create container widgets
        width_container = QWidget()
        width_layout = QHBoxLayout(width_container)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(self.width_spin)
        width_layout.addWidget(self.width_double)

        height_container = QWidget()
        height_layout = QHBoxLayout(height_container)
        height_layout.setContentsMargins(0, 0, 0, 0)
        height_layout.addWidget(self.height_spin)
        height_layout.addWidget(self.height_double)

        # Add the containers as rows
        layout.addRow("Width:", width_container)
        layout.addRow("Height:", height_container)

        self.value_type.currentIndexChanged.connect(self.update_input_mode)

    def update_input_mode(self, index):
        is_percentage = index == 1
        self.width_spin.setVisible(not is_percentage)
        self.width_double.setVisible(is_percentage)
        self.height_spin.setVisible(not is_percentage)
        self.height_double.setVisible(is_percentage)
        
    def get_overlap(self, image_width=None, image_height=None, validate=True, return_pixels=False):
        """
        Get overlap values with optional validation against image dimensions.
        
        Args:
            image_width (int, optional): Width of source image
            image_height (int, optional): Height of source image  
            validate (bool): Whether to validate against image dimensions
            return_pixels (bool): If True, returns values in pixels; if False, percentages
            
        Returns:
            tuple: (width, height) in pixels or percentages if valid
            None: if invalid when validation enabled
        """
        # Cannot return pixel values if image dimensions not provided
        if not image_width and not image_height:
            return_pixels = False
            
        try:
            # Get current values
            is_percentage = self.value_type.currentIndex() == 1
            if is_percentage:
                width_pct = self.width_double.value() * 100  
                height_pct = self.height_double.value() * 100
                
                if validate and image_width and image_height:
                    width_px = int((width_pct / 100.0) * image_width)
                    height_px = int((height_pct / 100.0) * image_height)
                else:
                    width_px = width_pct  # Keep as percentage
                    height_px = height_pct
            else:
                width_px = self.width_spin.value()
                height_px = self.height_spin.value()
                
                if validate and image_width and image_height:
                    width_pct = (width_px / image_width) * 100
                    height_pct = (height_px / image_height) * 100
                else:
                    width_pct = width_px  # Keep as pixels
                    height_pct = height_px

            # Validate if requested and dimensions provided
            if validate and image_width and image_height:
                if width_px >= image_width:
                    raise ValueError(f"Overlap width ({width_px}px) must be < image width ({image_width}px)")
                if height_px >= image_height:
                    raise ValueError(f"Overlap height ({height_px}px) must be < image height ({image_height}px)")
            
            # Return in requested format
            if return_pixels:
                return width_px, height_px
            return width_pct / 100.0, height_pct / 100.0

        except (ValueError, ZeroDivisionError) as e:
            if validate:
                QMessageBox.warning(self, "Invalid Overlap", str(e))
                return None
            return self.get_value()