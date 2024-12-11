import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import os

import numpy as np
import torch

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QGroupBox)

from x_segment_anything import SamPredictor
from x_segment_anything import sam_model_registry
from x_segment_anything import sam_model_urls

from torch.cuda import empty_cache
from ultralytics.utils import ops

from coralnet_toolbox.ResultsProcessor import ResultsProcessor
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import preprocess_image
from coralnet_toolbox.utilities import rasterio_to_numpy
from coralnet_toolbox.utilities import attempt_download_asset


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployPredictorDialog(QDialog):
    def __init__(self, main_window, parent=None):
        """Initialize the SAM Deploy Model dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("sam.png"))
        self.setWindowTitle("SAM Deploy Model") 
        self.resize(400, 325)

        # Initialize instance variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        self.model_path = None
        self.loaded_model = None
        self.image_path = None
        self.original_image = None
        self.resized_image = None

        # Create the layout
        self.layout = QVBoxLayout(self)
        
        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        # Setup the status layout
        self.setup_status_layout()
        
    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
        self.initialize_area_threshold()
        
    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a Predictor to deploy and use interactively with the SAM tool and others.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_models_layout(self):
        """
        Setup the models layout.
        """
        group_box = QGroupBox("Models")
        layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        
        # Define available models
        self.models = {
            "RepViT-SAM": "repvit.pt",
            "EdgeSAM": "edge_sam_3x.pt",
            "MobileSAM": "vit_t.pt",
            "CoralSCOP": "vit_b_coralscop.pt",
            "SAM-Base": "vit_b.pt",
            "SAM-Large": "vit_l.pt",
            "SAM-Huge": "vit_h.pt"
        }

        # Add all models to combo box
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)

        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()
        
        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        layout.addRow("Resize Image:", self.resize_image_dropdown)
        
        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 65536)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

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
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_buttons_layout(self):
        """
        Setup action buttons in a group box.
        """
        group_box = QGroupBox("Actions")
        layout = QHBoxLayout()
        
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)
        
        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)        
        
    def setup_status_layout(self):
        """
        Setup status display in a group box.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()
        
        self.status_bar = QLabel("No model loaded")
        layout.addWidget(self.status_bar)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
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
        
    def download_model_weights(self, model_path):
        """
        Download the model weights if they are not present.
        """
        model = os.path.basename(model_path).split(".")[0]
        model_url = sam_model_urls[model]

        # Download the model weights if they are not present
        attempt_download_asset(self, model_path, model_url)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not downloaded: {model_path}")

    def load_model(self):
        """
        Load the selected model.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get selected model path and download weights if needed
            self.model_path = self.models[self.model_combo.currentText()]
            self.download_model_weights(self.model_path)

            # Determine model type from filename
            if "repvit" in self.model_path.lower():
                model_type = "repvit"
            elif "edge_" in self.model_path.lower():
                model_type = "edge_sam"
            elif "_coralscop" in self.model_path.lower():
                model_type = "vit_b_coralscop"
            elif "_t" in self.model_path.lower():
                model_type = "vit_t"
            elif "_b" in self.model_path.lower():
                model_type = "vit_b"
            elif "_l" in self.model_path.lower():
                model_type = "vit_l"
            elif "_h" in self.model_path.lower():
                model_type = "vit_h"
            else:
                raise ValueError(f"Model type not recognized from filename: {self.model_path}")

            # Load model using registry
            model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.loaded_model = SamPredictor(model)

            # Move to device and set eval mode
            self.loaded_model.model.to(device=self.main_window.device)
            self.loaded_model.model.eval()
            
            self.status_bar.setText("Model loaded")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")

        progress_bar.stop_progress()
        progress_bar.close()
        QApplication.restoreOverrideCursor()
        self.accept()

    def resize_image(self, image):
        """
        Resize the image to the specified size.
        """
        imgsz = self.imgsz_spinbox.value()
        target_shape = self.get_target_shape(image, imgsz)
        return ops.scale_image(image, target_shape)

    def get_target_shape(self, image, imgsz):
        """
        Determine the target shape based on the long side.
        """
        h, w = image.shape[:2]
        if h > w:
            return imgsz, int(w * (imgsz / h))
        else:
            return int(h * (imgsz / w)), imgsz

    def set_image(self, image, image_path):
        """
        Set the image in the predictor.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Setting Image")
        progress_bar.show()

        try:
            if self.loaded_model is not None:
                
                if image is None and image_path is not None:
                    # Open the image using rasterio
                    image = self.main_window.image_window.rasterio_open(image_path)
                    image = rasterio_to_numpy(image)
                    
                # Preprocess the image
                image = preprocess_image(image)

                # Save the original image
                self.original_image = image
                self.image_path = image_path

                # Resize the image if the checkbox is checked
                if self.resize_image_dropdown.currentText() == "True":
                    image = self.resize_image(image)

                # Set the image in the predictor
                self.loaded_model.set_image(image)
                self.resized_image = image
            else:
                raise Exception("Model not loaded")

        except Exception as e:
            QMessageBox.critical(self, "Error Setting Image", f"Error setting image: {e}")

        finally:
            # Ensure cleanup happens even if an error occurs
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

    def scale_points(self, points):
        """
        Scale the points based on the original and resized image dimensions.

        Args:
            points (torch.tensor): The points to scale.
        """
        # Calculate scaling factors
        original_height, original_width = self.original_image.shape[:2]
        resized_height, resized_width = self.resized_image.shape[:2]

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        # Scale the points based on the original image dimensions
        scaled_points = points.clone().float()  # Cast to float32
        scaled_points[:, :, 0] *= scale_x
        scaled_points[:, :, 1] *= scale_y
        point_coords = scaled_points.long()  # Cast back to int64
        return point_coords

    def scale_boxes(self, boxes):
        """
        Scale the bounding boxes based on the original and resized image dimensions.
        Handles both single boxes (4,) and multiple boxes (N, 4).

        Args:
            boxes (torch.tensor): The bounding boxes to scale, shape (4,) or (N, 4)
        """
        # Convert to correct shape
        original_shape = boxes.shape
        if len(original_shape) == 1:
            boxes = boxes.unsqueeze(0)  # Add batch dimension

        # Calculate scaling factors
        original_height, original_width = self.original_image.shape[:2]
        resized_height, resized_width = self.resized_image.shape[:2]

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        # Scale the box based on the original image dimensions
        scaled_bbox = boxes.clone().float()  # Cast to float32
        scaled_bbox[:, 0] *= scale_x
        scaled_bbox[:, 1] *= scale_y
        scaled_bbox[:, 2] *= scale_x
        scaled_bbox[:, 3] *= scale_y
        bbox_coords = scaled_bbox.long()  # Cast back to int64

        # Return in original shape if it was a single box
        if len(original_shape) == 1:
            bbox_coords = bbox_coords.squeeze(0)
            
        return bbox_coords

    def transform_points(self, points, labels):
        """
        Transform the points based on the original and resized image dimensions.

        Args:
            points (np.ndarray): The points to transform.
            labels (list): The labels for each point.
        """
        input_labels = torch.tensor(labels)
        point_labels = input_labels.to(self.main_window.device).unsqueeze(0)

        # Provide prompt to SAM model in form of numpy array
        input_points = torch.as_tensor(points.astype(int), dtype=torch.int64)
        input_points = input_points.to(self.main_window.device).unsqueeze(0)

        # Scale the points
        point_coords = self.scale_points(input_points)

        point_coords = self.loaded_model.transform.apply_coords_torch(point_coords,
                                                                      self.resized_image.shape[:2])

        return point_coords, point_labels

    def transform_bboxes(self, bboxes):
        """
        Transform the bounding boxes based on the original and resized image dimensions.

        Args:
            bboxes (np.ndarray): The bounding boxes to transform.
        """
        input_bbox = torch.as_tensor(bboxes, dtype=torch.int64)
        input_bbox = input_bbox.to(self.main_window.device)
        
        # Scale the bounding boxes
        bbox_coords = self.scale_boxes(input_bbox)

        bbox_coords = self.loaded_model.transform.apply_boxes_torch(bbox_coords,
                                                                    self.resized_image.shape[:2])

        return bbox_coords

    def predict_from_prompts(self, bbox, points, labels):
        """
        Make predictions using the currently loaded model using prompts.
        
        Args:
            bbox (np.ndarray): The bounding boxes to use as prompts.
            points (np.ndarray): The points to use as prompts.
            labels (list): The labels for each point.
            
        Returns:
            results (Results): Ultralytics Results object
        """
        if not self.loaded_model:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded, cannot make predictions")
            return None

        try:
            point_labels = None
            point_coords = None
            bbox_coords = None

            if len(points) != 0:
                point_coords, point_labels = self.transform_points(points, labels)

            if len(bbox) != 0:
                bbox_coords = self.transform_bboxes(bbox)

            masks, scores, _ = self.loaded_model.predict_torch(boxes=bbox_coords,
                                                               point_coords=point_coords,
                                                               point_labels=point_labels,
                                                               num_multimask_outputs=1)
            
            # Create a results processor
            results_processor = ResultsProcessor(self.main_window, 
                                                 class_mapping=None,
                                                 uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
                                                 iou_thresh=self.main_window.get_iou_thresh(),
                                                 min_area_thresh=self.main_window.get_area_thresh_min(),
                                                 max_area_thresh=self.main_window.get_area_thresh_max())

            # Post-process the results
            results = results_processor.from_sam(masks, scores, self.original_image, self.image_path)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")
            return None

        return results

    def predict_from_results(self, results_generator, class_mapping):
        """
        Make predictions using the currently loaded model using results.
        
        Args:
            results_generator (generator): A generator that yields Ultralytics Results.
        """
        # Create a result processor
        result_processor = ResultsProcessor(self.main_window, 
                                            class_mapping=class_mapping,
                                            uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
                                            iou_thresh=self.main_window.get_iou_thresh(),
                                            min_area_thresh=self.main_window.get_area_thresh_min(),
                                            max_area_thresh=self.main_window.get_area_thresh_max())

        results_dict = {}

        for results in results_generator:
            results = result_processor.apply_filters(results)
            for result in results:
                if result:
                    # Extract the results
                    image_path, cls_id, cls_name, conf, *bbox = result_processor.extract_detection_result(result)

                    if image_path not in results_dict:
                        results_dict[image_path] = []

                    # Add the results to the dictionary
                    results_dict[image_path].append(np.array(bbox))

        # Loop through each unique image path
        for image_path in results_dict:
            try:
                # Set the image
                self.set_image(image=None, image_path=image_path)

                # Unpack the results
                bboxes = np.stack(results_dict[image_path])

                # Make predictions
                new_results = self.predict_from_prompts(bboxes, [], [])
                new_results.names = results.names
                new_results.boxes = results.boxes

                yield new_results

            except Exception as e:
                QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        self.loaded_model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None
        self.resized_image = None
        gc.collect()
        empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")