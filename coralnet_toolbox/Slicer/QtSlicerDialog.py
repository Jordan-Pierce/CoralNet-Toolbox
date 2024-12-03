from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
                             QDoubleSpinBox, QCheckBox)

from coralnet_toolbox.Slicer.QtSlicer import Slicer


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.slice_enabled = False
        self.slice_size = (640, 640)
        self.slice_overlap = 0.2
        self.slicer_handler = None

    def setup_slicer_layout(self):
        slicer_group = QGroupBox("Image Slicing")
        slicer_layout = QVBoxLayout()
        
        # Enable slicer checkbox
        self.slicer_checkbox = QCheckBox("Enable Image Slicing")
        self.slicer_checkbox.stateChanged.connect(self.toggle_slicer)
        
        # Slice size controls
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Slice Size:"))
        self.slice_size_spin = QSpinBox()
        self.slice_size_spin.setRange(256, 1024)
        self.slice_size_spin.setValue(640)
        size_layout.addWidget(self.slice_size_spin)
        
        # Overlap controls
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap:"))
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.5)
        self.overlap_spin.setValue(0.2)
        overlap_layout.addWidget(self.overlap_spin)
        
        slicer_layout.addWidget(self.slicer_checkbox)
        slicer_layout.addLayout(size_layout)
        slicer_layout.addLayout(overlap_layout)
        slicer_group.setLayout(slicer_layout)
        self.layout.addWidget(slicer_group)

    def toggle_slicer(self, state):
        self.slice_enabled = bool(state)
        if self.slice_enabled and self.loaded_model:
            self.slicer_handler = Slicer(
                model=self.loaded_model,
                slice_size=(self.slice_size_spin.value(), self.slice_size_spin.value()),
                overlap=self.overlap_spin.value()
            )
        else:
            self.slicer_handler = None

    # Modify existing process_image method
    def process_image(self, image):
        if self.slice_enabled and self.slicer_handler:
            return self.slicer_handler.process_image(image)
        else:
            # Original processing code
            return self.loaded_model.predict(image)