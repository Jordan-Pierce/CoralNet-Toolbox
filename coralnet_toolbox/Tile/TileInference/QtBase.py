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
    Base class for performing tiled inference on images using object detection, and instance segmentation 
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
        
        self.shape_x = None
        self.shape_y = None
        self.overlap_x = None
        self.overlap_y = None
        
        self.iou_threshold = None
        self.conf_threshold = None
        self.nms_threshold = None
        self.match_metric = None
        self.class_agnostic_nms = None
        self.intelligent_sorter = None
        self.sorter_bins = None
        self.memory_optimize = None

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
        
        # IOU threshold
        self.iou_threshold_input = QDoubleSpinBox()
        self.iou_threshold_input.setRange(0.0, 1.0)
        self.iou_threshold_input.setSingleStep(0.01)
        self.iou_threshold_input.setValue(0.5)
        layout.addRow("IOU Threshold:", self.iou_threshold_input)
        
        # Confidence threshold
        self.conf_threshold_input = QDoubleSpinBox()
        self.conf_threshold_input.setRange(0.0, 1.0)
        self.conf_threshold_input.setSingleStep(0.01)
        self.conf_threshold_input.setValue(0.5)
        layout.addRow("Confidence Threshold:", self.conf_threshold_input)
        
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
        label = QLabel("<b>Use SAM for creating Polygons:</b>")
        layout.addRow(label, self.use_sam_dropdown)

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
    
    def get_current_image(self):
        """Get the current image from the annotation window."""
        pass

    def apply(self):
        """
        Apply the tile inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Tile inference
            self.setup_tile_inference()
            # Apply tile inference
            # self.apply_tile_inference()  # TODO

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to tile inference image: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()

        self.accept()

    def setup_tile_inference(self):
        """
        Use YOLO-Patch-Based-Inference.
        """
        # Tile parameters
        slice_wh = self.tile_size_input.get_value()
        overlap_wh = self.overlap_input.get_value(slice_wh[0], slice_wh[1])
        margins = self.margins_input.get_value()
        
        # Inference parameters 
        self.iou_threshold = self.iou_threshold_input.value()
        self.conf_threshold = self.conf_threshold_input.value()
        self.nms_threshold = self.nms_threshold_input.value()
        self.match_metric = self.match_metric_input.currentText()
        self.class_agnostic_nms = self.class_agnostic_nms_input.currentText() == "True"
        self.intelligent_sorter = self.intelligent_sorter_input.currentText() == "True"
        self.sorter_bins = self.sorter_bins_input.value()
        self.memory_optimize = self.memory_input.currentText() == "True"

        # Perform all validation checks
        if not self.validate_slice_wh(slice_wh):
            return
            
        if not self.validate_overlap_wh(overlap_wh):
            return
            
        if not self.validate_margins(margins):
            return

        if self.match_metric not in ["IOU", "IOS"]:
            QMessageBox.warning(self, 
                                "Invalid Parameter", 
                                "Match metric must be either 'IOU' or 'IOS'.")
            return

        if not (1 <= self.sorter_bins <= 10):
            QMessageBox.warning(self, 
                                "Invalid Parameter",
                                "Sorter bins must be between 1 and 10.")
            return

        # Extract components after validation
        self.shape_x, self.shape_y = slice_wh
        self.overlap_x, self.overlap_y = overlap_wh
        
    def apply_tile_inference(self):
        """ """
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
            # Perform tile cropping
            element_crops = MakeCropsDetectThem(
                image=img,
                model=self.deploy_model_dialog.loaded_model,
                segment=True if self.annotation_type == "instance_segmentation" else False,
                show_crops=False,
                shape_x=self.shape_x,
                shape_y=self.shape_y,
                overlap_x=self.overlap_x,
                overlap_y=self.overlap_y,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                show_processing_status=True,
                progress_callback=progress_callback,
            )

            # Perform inference on cropped images
            result = CombineDetections(element_crops, 
                                       nms_threshold=self.nms_threshold, 
                                       sorter_bins=self.sorter_bins,
                                       match_metric=self.match_metric,
                                       class_agnostic_nms=self.class_agnostic_nms,
                                       intelligent_sorter=self.intelligent_sorter,
                                       memory_optimize=self.memory_optimize)

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error",
                                 f"Failed to tile inference: {str(e)}")
        finally:
            self.progress_bar.stop_progress()
            self.progress_bar.close()
            QApplication.restoreOverrideCursor()
