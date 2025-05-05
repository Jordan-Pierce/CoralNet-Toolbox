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


class MarginInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Margins", parent)
        layout = QVBoxLayout(self)

        # Input type selection
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single Value", "Multiple Values"])
        self.type_combo.setCurrentIndex(1)  # Set Multiple Values as default
        self.value_type = QComboBox()
        self.value_type.addItems(["Pixels", "Percentage"])
        self.value_type.setCurrentIndex(0)  # Set Pixels as default

        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_layout.addWidget(QLabel("Unit:"))
        type_layout.addWidget(self.value_type)
        layout.addLayout(type_layout)

        # Stacked widget for different input types
        self.stack = QStackedWidget()

        # Single value widgets
        single_widget = QWidget()
        single_layout = QHBoxLayout(single_widget)
        self.single_spin = QSpinBox()
        self.single_spin.setRange(0, 10000)
        self.single_double = QDoubleSpinBox()
        self.single_double.setRange(0, 1)
        self.single_double.setSingleStep(0.1)
        self.single_double.setDecimals(2)
        single_layout.addWidget(self.single_spin)
        single_layout.addWidget(self.single_double)
        self.single_double.hide()

        # Multiple values widgets
        multi_widget = QWidget()
        multi_layout = QGridLayout(multi_widget)
        self.margin_spins = []
        self.margin_doubles = []
        positions = [("Top", 0, 1),
                     ("Right", 1, 2),
                     ("Bottom", 2, 1),
                     ("Left", 1, 0)]

        for label, row, col in positions:
            spin = QSpinBox()
            spin.setRange(0, 10000)
            double = QDoubleSpinBox()
            double.setRange(0, 1)
            double.setSingleStep(0.1)
            double.setDecimals(2)
            double.hide()

            self.margin_spins.append(spin)
            self.margin_doubles.append(double)
            multi_layout.addWidget(QLabel(label), row, col)
            multi_layout.addWidget(spin, row + 1, col)
            multi_layout.addWidget(double, row + 1, col)

        self.stack.addWidget(single_widget)
        self.stack.addWidget(multi_widget)
        layout.addWidget(self.stack)

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.value_type.currentIndexChanged.connect(self.update_input_mode)
        
        # Initialize to Multiple Values
        self.stack.setCurrentIndex(1)
        self.update_input_mode(0)  # Initialize with Pixels mode

    def update_input_mode(self, index):
        is_percentage = index == 1
        if is_percentage:
            self.single_spin.hide()
            self.single_double.show()
            for spin, double in zip(self.margin_spins, self.margin_doubles):
                spin.hide()
                double.show()
        else:
            self.single_double.hide()
            self.single_spin.show()
            for spin, double in zip(self.margin_spins, self.margin_doubles):
                double.hide()
                spin.show()

    def get_value(self):
        is_percentage = self.value_type.currentIndex() == 1
        if self.type_combo.currentIndex() == 0:
            return self.single_double.value() if is_percentage else self.single_spin.value()
        else:
            widgets = self.margin_doubles if is_percentage else self.margin_spins
            return tuple(w.value() for w in widgets)

    def get_margins(self, image_width=None, image_height=None, validate=True):
        """
        Get margin values with optional validation.
        
        Args:
            image_width (int, optional): Width of source image
            image_height (int, optional): Height of source image
            validate (bool): Whether to validate margins against image dimensions
            
        Returns:
            tuple: (Left, Top, Right, Bottom) margins in pixels if valid
            None: if invalid when validation enabled
        """
        is_percentage = self.value_type.currentIndex() == 1
        margin_pixels = [0, 0, 0, 0]  # [Left, Top, Right, Bottom]
    
        try:
            raw_margins = self.get_value()
            
            # Single value input
            if isinstance(raw_margins, (int, float)):
                if is_percentage:
                    if validate and not (0.0 <= raw_margins <= 1.0):
                        raise ValueError("Percentage must be between 0 and 1")
                        
                    if validate and image_width is not None and image_height is not None:
                        margin_pixels = [
                            int(raw_margins * image_width),    # Left
                            int(raw_margins * image_height),   # Top  
                            int(raw_margins * image_width),    # Right
                            int(raw_margins * image_height)    # Bottom
                        ]
                        return tuple(margin_pixels)
                    else:
                        # If no image dimensions, return percentage as is
                        return (raw_margins,) * 4
                else:
                    # Return pixels as is
                    return (raw_margins,) * 4
    
            # Multiple values input (Top, Right, Bottom, Left)
            elif isinstance(raw_margins, tuple) and len(raw_margins) == 4:
                ordered_margins = (
                    raw_margins[3],  # Left
                    raw_margins[0],  # Top
                    raw_margins[1],  # Right
                    raw_margins[2]   # Bottom
                )
    
                if is_percentage:
                    if validate and not all(0.0 <= m <= 1.0 for m in ordered_margins):
                        raise ValueError("All percentages must be between 0 and 1")
                        
                    if validate and image_width is not None and image_height is not None:
                        margin_pixels = [
                            int(ordered_margins[0] * image_width),   # Left
                            int(ordered_margins[1] * image_height),  # Top
                            int(ordered_margins[2] * image_width),   # Right
                            int(ordered_margins[3] * image_height)   # Bottom
                        ]
                        return tuple(margin_pixels)
                    else:
                        # If no image dimensions, return percentages as is
                        return ordered_margins
                else:
                    # Return pixels as is
                    return ordered_margins
            else:
                raise ValueError("Invalid margin format")
            
        except ValueError as e:
            if validate:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Invalid Margins", str(e))
                return None
            return raw_margins