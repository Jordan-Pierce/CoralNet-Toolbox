import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import shutil

from patched_yolo_infer import MakeCropsDetectThem
from patched_yolo_infer import CombineDetections

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout, QLabel, QDialog,
                             QDialogButtonBox, QGroupBox, QFormLayout, QLineEdit,
                             QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QSpinBox,
                             QHBoxLayout, QSlider, QWidget)

from coralnet_toolbox.Tile.QtCommon import TileSizeInput, OverlapInput, MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for perfoming tiled inference on images using object detection, and instance segmentation 
    datasets using YOLO-Patch-Based-Inference.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Inference")
        self.resize(1000, 600)

        # Object Detection / Instance Segmentation
        self.annotation_type = None
        self.loaded_model = None

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
        layout.addRow(self.tile_size_input)

        # Overlap
        self.overlap_input = OverlapInput()
        layout.addRow(self.overlap_input)

        # Margins
        self.margins_input = MarginInput()
        layout.addRow(self.margins_input)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)      
        
    def setup_inference_config_layout(self, parent_layout):
        """
        Set up the inference configuration parameters layout.
        """
        group_box = QGroupBox("Inference Configuration Parameters")
        layout = QFormLayout()
        
        # Uncertainty threshold controls
        self.uncertainty_thresh = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_threshold_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        layout.addRow("", self.uncertainty_threshold_label)
        
        # IoU threshold controls
        self.iou_thresh = self.main_window.get_iou_thresh()
        self.iou_threshold_slider = QSlider(Qt.Horizontal)
        self.iou_threshold_slider.setRange(0, 100)
        self.iou_threshold_slider.setValue(int(self.iou_thresh * 100))
        self.iou_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_threshold_slider.setTickInterval(10)
        self.iou_threshold_slider.valueChanged.connect(self.update_iou_label)
        self.iou_threshold_label = QLabel(f"{self.iou_thresh:.2f}")
        layout.addRow("IoU Threshold", self.iou_threshold_slider)
        layout.addRow("", self.iou_threshold_label)
        
        # Area threshold controls
        min_val, max_val = self.main_window.get_area_thresh()
        self.area_thresh_min = int(min_val * 100)
        self.area_thresh_max = int(max_val * 100)
        self.area_threshold_slider = QRangeSlider(Qt.Horizontal)
        self.area_threshold_slider.setRange(0, 100)
        self.area_threshold_slider.setValue((self.area_thresh_min, self.area_thresh_max))
        self.area_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_slider.setTickInterval(10)
        self.area_threshold_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        layout.addRow("Area Threshold", self.area_threshold_slider)
        layout.addRow("", self.area_threshold_label)
        
        # NMS threshold
        self.nms_threshold_input = QDoubleSpinBox()
        self.nms_threshold_input.setRange(0.0, 1.0)
        self.nms_threshold_input.setSingleStep(0.01)
        self.nms_threshold_input.setValue(0.3)
        layout.addRow("NMS Threshold:", self.nms_threshold_input)

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
        
        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        layout.addRow("Use SAM for creating Polygons:", self.use_sam_dropdown)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)
        
    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box for the buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)
        
    def initialize_uncertainty_threshold(self):
        """Initialize the uncertainty threshold slider with the current value"""
        current_value = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider.setValue(int(current_value * 100))
        self.uncertainty_thresh = current_value

    def initialize_iou_threshold(self):
        """Initialize the IOU threshold slider with the current value"""
        current_value = self.main_window.get_iou_thresh()
        self.iou_threshold_slider.setValue(int(current_value * 100))
        self.iou_thresh = current_value
        
    def initialize_area_threshold(self):
        """Initialize the area threshold range slider"""
        current_min, current_max = self.main_window.get_area_thresh()
        self.area_threshold_slider.setValue((int(current_min * 100), int(current_max * 100)))
        self.area_thresh_min = current_min
        self.area_thresh_max = current_max

    def update_uncertainty_label(self, value):
        """Update uncertainty threshold and label"""
        value = value / 100.0
        self.uncertainty_thresh = value
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value 
        self.main_window.update_iou_thresh(value)
        self.iou_threshold_label.setText(f"{value:.2f}")

    def update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val, max_val = self.area_threshold_slider.value()  # Returns tuple of values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")      
        
    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_model_dialog'):
            return False
        
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        return True 

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

    def apply(self):
        """
        Apply the tile inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Tile inference
            self.tile_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to tile inference image: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()

        self.accept()

    def tile_inference(self):
        """
        Use YOLO-Patch-Based-Inference.
        """
        # Extract the tile parameters
        slice_wh = self.tile_size_input.get_value()
        overlap_wh = self.overlap_input.get_value()
        margins = self.margins_input.get_value()

        # Extract the inference parameters
        uncertainty_thresh=self.main_window.get_uncertainty_thresh()
        iou_thresh=self.main_window.get_iou_thresh()
        min_area_thresh=self.main_window.get_area_thresh_min()
        max_area_thresh=self.main_window.get_area_thresh_max()
                
        nms_threshold = self.nms_threshold_input.value()
        match_metric = self.match_metric_input.currentText()
        class_agnostic_nms = self.class_agnostic_nms_input.currentText() == "True"
        intelligent_sorter = self.intelligent_sorter_input.currentText() == "True"
        sorter_bins = self.sorter_bins_input.value()
        memory_optimize = self.memory_input.currentText() == "True"
        
        # -------------------------
        # Perform validation checks

        # Validate the slice_wh parameter
        if not self.validate_slice_wh(slice_wh):
            return
        else:
            shape_x, shape_y = slice_wh

        # Validate the overlap_wh parameter
        if not self.validate_overlap_wh(overlap_wh):
            return
        else:
            overlap_x, overlap_y = overlap_wh

        # Validate the margins parameter
        if not self.validate_margins(margins):
            return
        
        # Validation checks        
        if match_metric not in ["IOU", "IOS"]:
            QMessageBox.warning(self, 
                                "Invalid Parameter", 
                                "Match metric must be either 'IOU' or 'IOS'.")
            return False
            
        if not (1 <= sorter_bins <= 10):
            QMessageBox.warning(self, 
                                "Invalid Parameter",
                                "Sorter bins must be between 1 and 10.")
            return False

        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create and show the progress bar
        self.progress_bar = ProgressBar(self, title="Tile Inference Progress")
        self.progress_bar.show()

        def progress_callback(task, current, total):
            self.progress_bar.setWindowTitle(f"{task}")
            progress_percentage = int((current / total) * 100)
            self.progress_bar.set_value(progress_percentage)
            self.progress_bar.update_progress()
            if self.progress_bar.wasCanceled():
                raise Exception("Tile Inference was canceled by the user.")

        try:
            pass

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error",
                                 f"Failed to tile inference: {str(e)}")
        finally:
            self.progress_bar.stop_progress()
            self.progress_bar.close()
            QApplication.restoreOverrideCursor()
