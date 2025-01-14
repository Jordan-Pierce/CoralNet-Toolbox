import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QComboBox, QSpinBox, QHBoxLayout,
                             QWidget, QStackedWidget, QGridLayout)


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
        self.width_spin.setRange(0, 9999)
        self.width_spin.setValue(0)
        self.width_double = QDoubleSpinBox()
        self.width_double.setRange(0, 1)
        self.width_double.setValue(0)
        self.width_double.setSingleStep(0.1)
        self.width_double.setDecimals(2)
        self.width_double.hide()

        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 9999)
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

    def get_value(self, tile_width, tile_height):
        is_percentage = self.value_type.currentIndex() == 1
        if is_percentage:
            return self.width_double.value(), self.height_double.value()
        else:
            width_percentage = (self.width_spin.value() / tile_width) * 100
            height_percentage = (self.height_spin.value() / tile_height) * 100
            return width_percentage, height_percentage


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
        self.single_spin.setRange(0, 9999)
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
            spin.setRange(0, 9999)
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


class TileSizeInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Tile Size", parent)
        layout = QFormLayout(self)

        # Width input in pixels
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 9999)
        self.width_spin.setValue(640)

        # Height input in pixels
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 9999)
        self.height_spin.setValue(480)

        layout.addRow("Width:", self.width_spin)
        layout.addRow("Height:", self.height_spin)

    def get_value(self):
        return (self.width_spin.value(), self.height_spin.value())
