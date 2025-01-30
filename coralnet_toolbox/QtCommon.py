import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QComboBox, QSpinBox, QHBoxLayout,
                             QWidget, QStackedWidget, QGridLayout, QMessageBox)


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

        layout.addRow("Width:", self.width_spin)
        layout.addRow("", self.width_double)
        layout.addRow("Height:", self.height_spin)
        layout.addRow("", self.height_double)

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


class MarginInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Margins", parent)
        layout = QVBoxLayout(self)

        # Input type selection
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single Value", "Multiple Values"])
        self.value_type = QComboBox()
        self.value_type.addItems(["Pixels", "Percentage"])

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
                        
                    if validate and image_width and image_height:
                        margin_pixels = [
                            raw_margins * image_width,    # Left
                            raw_margins * image_height,   # Top  
                            raw_margins * image_width,    # Right
                            raw_margins * image_height    # Bottom
                        ]
                    else:
                        return (raw_margins,) * 4
                else:
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
                        
                    if validate and image_width and image_height:
                        margin_pixels = [
                            ordered_margins[0] * image_width,   # Left
                            ordered_margins[1] * image_height,  # Top
                            ordered_margins[2] * image_width,   # Right
                            ordered_margins[3] * image_height   # Bottom
                        ]
                    else:
                        return ordered_margins
                else:
                    return ordered_margins
            else:
                raise ValueError("Invalid margin format")
    
            # Validate if requested
            if validate and image_width and image_height:
                margin_pixels = [int(m) for m in margin_pixels]
                if (margin_pixels[0] + margin_pixels[2]) >= image_width:
                    raise ValueError("Horizontal margins exceed image width")
                if (margin_pixels[1] + margin_pixels[3]) >= image_height:
                    raise ValueError("Vertical margins exceed image height")
    
            return tuple(margin_pixels)
    
        except ValueError as e:
            if validate:
                QMessageBox.warning(self, "Invalid Margins", str(e))
                return None
            return raw_margins
        

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
