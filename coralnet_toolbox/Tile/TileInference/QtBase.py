import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import shutil
import random

from patched_yolo_infer import MakeCropsDetectThem
from patched_yolo_infer import CombineDetections

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout, QLabel, QDialog,
                             QDialogButtonBox, QGroupBox, QFormLayout, QLineEdit,
                             QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QSpinBox,
                             QHBoxLayout, QSlider, QWidget, QGraphicsRectItem)

from coralnet_toolbox.Tile.QtCommon import TileSizeInput, OverlapInput, MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for performing tiled inference on images using object detection, and instance segmentation 
    datasets using YOLO-Patch-Based-Inference.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Inference")
        self.resize(300, 600)
        
        # Initialize debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_tile_graphics)

        # Tile parameters
        self.tile_params = {}
        self.tile_inference_params = {}
        
        self.shape_x = None
        self.shape_y = None
        self.overlap_x = None
        self.overlap_y = None
        self.margins = None
        
        self.match_metric = None
        self.class_agnostic_nms = None
        self.intelligent_sorter = None
        self.sorter_bins = None
        self.memory_optimize = None
        
        self.image = None
        self.tile_graphics = []

        self.layout = QVBoxLayout(self)
        
        # Info layout at top
        self.setup_info_layout()

        # Create horizontal layout for configs
        config_layout = QHBoxLayout()
        
        # Left side - Tile Config
        self.tile_config_widget = QWidget()
        tile_layout = QVBoxLayout(self.tile_config_widget)
        self.setup_tile_config_layout(tile_layout)
        config_layout.addWidget(self.tile_config_widget)
        
        # Right side - Inference Config
        self.inference_config_widget = QWidget()
        inference_layout = QVBoxLayout(self.inference_config_widget)
        self.setup_inference_config_layout(inference_layout)
        config_layout.addWidget(self.inference_config_widget)
        
        # Add the horizontal config layout to main layout
        self.layout.addLayout(config_layout)

        # Buttons at bottom
        self.setup_buttons_layout()
        
    def showEvent(self, event):
        # Get the image pixmap from the annotation window
        self.image = self.annotation_window.image_pixmap
        self.update_tile_graphics()
        
    def closeEvent(self, event):
        self.clear_tile_graphics()
        
    def debounce_update(self):
        """Debounce the update_tile_graphics call by Nms"""
        self.update_timer.start(1000)

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Tile an image into smaller non / overlapping images, performing inference on each.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_tile_config_layout(self, parent_layout):
        """
        Set up the tile config parameters layout.
        """
        group_box = QGroupBox("Tile Configuration Parameters")
        layout = QFormLayout()

        # Tile Size
        self.tile_size_input = TileSizeInput()
        
        # Connect width and height spinboxes with debounce
        self.tile_size_input.width_spin.valueChanged.connect(self.debounce_update)
        self.tile_size_input.height_spin.valueChanged.connect(self.debounce_update)
        layout.addRow(self.tile_size_input)

        # Overlap
        self.overlap_input = OverlapInput() 
        
        # Connect all spinboxes/doublespinboxes with debounce
        self.overlap_input.width_spin.valueChanged.connect(self.debounce_update)
        self.overlap_input.height_spin.valueChanged.connect(self.debounce_update)
        self.overlap_input.width_double.valueChanged.connect(self.debounce_update)
        self.overlap_input.height_double.valueChanged.connect(self.debounce_update)
        layout.addRow(self.overlap_input)

        # Margins
        self.margins_input = MarginInput()
        
        # Connect single value inputs with debounce 
        self.margins_input.single_spin.valueChanged.connect(self.debounce_update)
        self.margins_input.single_double.valueChanged.connect(self.debounce_update)
        
        # Connect all margin spinboxes with debounce
        for spin in self.margins_input.margin_spins:
            spin.valueChanged.connect(self.debounce_update)
            
        # Connect all margin doublespinboxes with debounce
        for double in self.margins_input.margin_doubles:
            double.valueChanged.connect(self.debounce_update)
            
        layout.addRow(self.margins_input)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)
        
    def setup_inference_config_layout(self, parent_layout):
        """
        Set up the inference configuration parameters layout.
        """
        group_box = QGroupBox("Inference Configuration Parameters")
        layout = QFormLayout()
        
        # Match metric
        self.match_metric_input = QComboBox()
        self.match_metric_input.addItems(["IOU", "IOS"])
        self.match_metric_input.setCurrentText("IOS")
        layout.addRow("Match Metric:", self.match_metric_input)

        # Class agnostic NMS
        self.class_agnostic_nms_input = QComboBox()
        self.class_agnostic_nms_input.addItems(["True", "False"])
        layout.addRow("Class Agnostic NMS:", self.class_agnostic_nms_input)

        # Intelligent sorter
        self.intelligent_sorter_input = QComboBox()
        self.intelligent_sorter_input.addItems(["True", "False"])
        layout.addRow("Intelligent Sorter:", self.intelligent_sorter_input)

        # Sorter bins
        self.sorter_bins_input = QSpinBox()
        self.sorter_bins_input.setRange(1, 10)
        self.sorter_bins_input.setValue(5)
        layout.addRow("Sorter Bins:", self.sorter_bins_input)

        # Memory optimization
        self.memory_input = QComboBox()
        self.memory_input.addItems(["True", "False"])
        layout.addRow("Memory Optimization:", self.memory_input)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)
        
    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box with custom buttons
        button_box = QDialogButtonBox()
        apply_button = QPushButton("Apply")
        unapply_button = QPushButton("Unapply")
        cancel_button = QPushButton("Cancel")
        
        button_box.addButton(apply_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(unapply_button, QDialogButtonBox.ActionRole)
        button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)
        
        button_box.accepted.connect(self.apply)
        unapply_button.clicked.connect(self.unapply)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)

    def validate_slice_wh(self, slice_wh):
        """
        Validate the slice_wh parameter to ensure it is a tuple of two integers.

        :param slice_wh: Slice width and height
        :return: True if valid, False otherwise
        """
        if not isinstance(slice_wh, tuple) or len(slice_wh) != 2 or not all(isinstance(i, int) for i in slice_wh):
            QMessageBox.warning(self, "Invalid Tile Size", "The tile size must be a tuple of two integers.")
            return False
        return True

    def validate_overlap_wh(self, overlap_wh):
        """
        Validate the overlap_wh parameter to ensure it is a tuple of two floats.

        :param overlap_wh: Overlap width and height
        :return: True if valid, False otherwise
        """
        correct_type = all(isinstance(i, (int, float)) for i in overlap_wh)
        if not isinstance(overlap_wh, tuple) or len(overlap_wh) != 2 or not correct_type:
            QMessageBox.warning(self, "Invalid Overlap", "The overlap must be a tuple of two floats.")
            return False
        return True

    def validate_margins(self, margins):
        """
        Validate the margins parameter to ensure it is a valid type and value.

        :param margins: Margins
        :return: True if valid, False otherwise
        """
        if isinstance(margins, (int, float)):
            if isinstance(margins, float) and not (0.0 <= margins <= 1.0):
                QMessageBox.warning(self,
                                    "Invalid Margins",
                                    "The margin percentage must be between 0 and 1.")
                return False
            return True
        elif isinstance(margins, tuple) and len(margins) == 4:
            if all(isinstance(i, (int, float)) for i in margins):
                if all(isinstance(i, float) for i in margins) and not all(0.0 <= i <= 1.0 for i in margins):
                    QMessageBox.warning(self,
                                        "Invalid Margins",
                                        "All margin percentages must be between 0 and 1.")
                    return False
                return True
        QMessageBox.warning(self,
                            "Invalid Margins",
                            "The margins must be a single integer, float, or a tuple of four integers/floats.")
        return False
    
    def update_params(self):
        """
        Update the tile inference and inference parameters.
        """
        # Tile parameters
        slice_wh = self.tile_size_input.get_value()
        overlap_wh = self.overlap_input.get_value(slice_wh[0], slice_wh[1])
        margins = self.margins_input.get_value()
        
        # Inference parameters 
        match_metric = self.match_metric_input.currentText()
        sorter_bins = self.sorter_bins_input.value()

        # Perform all validation checks
        if not self.validate_slice_wh(slice_wh):
            return
            
        if not self.validate_overlap_wh(overlap_wh):
            return
            
        if not self.validate_margins(margins):
            return

        if match_metric not in ["IOU", "IOS"]:
            QMessageBox.warning(self, 
                                "Invalid Parameter", 
                                "Match metric must be either 'IOU' or 'IOS'.")
            return

        if not (1 <= sorter_bins <= 10):
            QMessageBox.warning(self, 
                                "Invalid Parameter",
                                "Sorter bins must be between 1 and 10.")
            return

        # Extract components after validation
        self.shape_x, self.shape_y = slice_wh
        self.overlap_x, self.overlap_y = overlap_wh
        self.margins = margins
        
        self.match_metric = match_metric
        self.sorter_bins = sorter_bins
        self.class_agnostic_nms = self.class_agnostic_nms_input.currentText() == "True"
        self.intelligent_sorter = self.intelligent_sorter_input.currentText() == "True"
        self.memory_optimize = self.memory_input.currentText() == "True"

    def apply(self):
        """
        Apply the tile inference options.
        """
        try:          
            # Create tiling parameters dictionary
            self.tile_params = {
                "shape_x": self.shape_x,
                "shape_y": self.shape_y, 
                "overlap_x": self.overlap_x,
                "overlap_y": self.overlap_y,
                "margins": self.margins,
                "show_crops": False,
                "show_processing_status": True
            }

            # Create inference parameters dictionary
            self.tile_inference_params = {
                "match_metric": self.match_metric,
                "class_agnostic_nms": self.class_agnostic_nms,
                "intelligent_sorter": self.intelligent_sorter,
                "sorter_bins": self.sorter_bins,
                "memory_optimize": self.memory_optimize
            }
            
            QMessageBox.information(self, 
                                    "Success", 
                                    "Tile Inference parameters set successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, 
                                 "Error", 
                                 f"Failed to set Tile Inference parameters: {str(e)}")
        finally:
            self.clear_tile_graphics()

        self.accept()
        
    def unapply(self):
        """
        Reset tile inference configurations.
        """
        try:           
            self.tile_params = {}
            self.tile_inference_params = {}
            QMessageBox.information(self, 
                                    "Success", 
                                    "Tile inference parameters reset successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, 
                                 "Error", 
                                 f"Failed to reset Tile Inference parameters: {str(e)}")
        finally:
            self.clear_tile_graphics()
            
        self.accept()
            
    def get_tile_params(self):
        """
        Get the tile parameters.

        :return: Tile parameters
        """
        return self.tile_params
    
    def get_tile_inference_params(self):
        """
        Get the tile inference parameters.

        :return: Tile inference parameters
        """
        return self.tile_inference_params

    def get_params(self):
        """
        Get both the tile and tile inference parameters.

        :return: Tuple of tile and tile inference parameters
        """
        return self.get_tile_params(), self.get_tile_inference_params()
    
    def update_tile_graphics(self):
        """
        Uses class tile parameters to create a grid of tiles on the annotation window image.
        """
        # Clear existing tile graphics
        self.clear_tile_graphics()
        
        # Update and validate all parameters
        self.update_params()

        if not self.annotation_window.image_pixmap:
            return

        # Get image dimensions
        image_full_width = self.annotation_window.image_pixmap.width()
        image_full_height = self.annotation_window.image_pixmap.height()

        # Calculate overlap coefficients
        if isinstance(self.overlap_x, float):
            cross_coef_x = 1 - self.overlap_x  # Float between 0-1
        else:
            cross_coef_x = 1 - (self.overlap_x / self.shape_x)  # Pixel value

        if isinstance(self.overlap_y, float): 
            cross_coef_y = 1 - self.overlap_y  # Float between 0-1
        else:
            cross_coef_y = 1 - (self.overlap_y / self.shape_y)  # Pixel value

        # Handle margins
        margin_pixels = [0, 0, 0, 0]  # [top, right, bottom, left]
        
        if self.margins is not None:
            if isinstance(self.margins, (int, float)):
                if isinstance(self.margins, float):
                    margin_val = int(self.margins * min(image_full_width, image_full_height))
                    margin_pixels = [margin_val] * 4
                else:
                    margin_pixels = [self.margins] * 4
            elif isinstance(self.margins, tuple):
                for i, margin in enumerate(self.margins):
                    if isinstance(margin, float):
                        margin_pixels[i] = int(margin * (image_full_height if i % 2 == 0 else image_full_width))
                    else:
                        margin_pixels[i] = margin

        # Calculate grid boundaries
        x_start = margin_pixels[3]  
        y_start = margin_pixels[0]  
        x_end = image_full_width - margin_pixels[1]
        y_end = image_full_height - margin_pixels[2]

        # Calculate grid steps, adjusted to fit within margins
        x_steps = int((x_end - x_start - self.shape_x) / (self.shape_x * cross_coef_x)) + 1
        y_steps = int((y_end - y_start - self.shape_y) / (self.shape_y * cross_coef_y)) + 1

        # Calculate line thickness based on resolution
        line_thickness = max(10, min(20, max(image_full_width, image_full_height) // 1000))

        # Draw tiles
        for i in range(y_steps + 1):
            for j in range(x_steps + 1):
                x = x_start + int(self.shape_x * j * cross_coef_x)
                y = y_start + int(self.shape_y * i * cross_coef_y)

                # Truncate tile width and height if extending beyond boundaries
                tile_width = min(self.shape_x, x_end - x)
                tile_height = min(self.shape_y, y_end - y)

                # Determine tile color and transparency
                if tile_width == self.shape_x and tile_height == self.shape_y:
                    # Full tiles within image boundaries
                    color = QColor(0, 0, 0) if (i * (x_steps + 1) + j) % 2 == 0 else QColor(255, 255, 255)
                    opacity = 0.5
                    line_style = Qt.DotLine
                    brush = QBrush(color)
                else:
                    # Tiles outside/partially outside image boundaries
                    color = QColor(255, 0, 0)  # Red color
                    opacity = 1.0
                    line_style = Qt.SolidLine
                    brush = QBrush(Qt.NoBrush)  # No fill for boundary tiles

                # Skip if tile is completely outside image
                if tile_width > 0 and tile_height > 0:
                    tile = QGraphicsRectItem(x, y, tile_width, tile_height)
                    tile.setPen(QPen(color, line_thickness, line_style))
                    tile.setBrush(brush)
                    tile.setOpacity(opacity)
                        
                    self.annotation_window.scene.addItem(tile)
                    self.tile_graphics.append(tile)
                    
    def clear_tile_graphics(self):
        """
        Clear the tile graphics from the annotation window.
        """
        # Remove all tile graphics from the scene
        for tile_graphic in self.tile_graphics:
            tile_graphic.scene().removeItem(tile_graphic)
        # Update the viewport to remove the tiles
        self.annotation_window.viewport().update()
        self.annotation_window.fitInView(self.annotation_window.scene.sceneRect(), Qt.KeepAspectRatio)
        # Clear the tile graphics list
        self.tile_graphics = []
        