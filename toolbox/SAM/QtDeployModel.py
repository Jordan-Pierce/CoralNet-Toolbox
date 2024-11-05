import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc

import numpy as np
import sam2
import torch

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox, QSlider, QLabel, QHBoxLayout, QPushButton,
                             QTabWidget, QComboBox, QMessageBox, QApplication, QWidget)

from torch.cuda import empty_cache
from mobile_sam import SamPredictor as MobileSamPredictor
from mobile_sam import sam_model_registry as mobile_sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor as Sam2Predictor
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils import ops
from ultralytics.utils.downloads import attempt_download_asset

from toolbox.QtProgressBar import ProgressBar
from toolbox.ResultsProcessor import ResultsProcessor

from toolbox.utilities import preprocess_image
from toolbox.utilities import rasterio_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def to_ultralytics(masks, scores, original_image):
    """
    Converts SAM output to Ultralytics Results Object.

    Args:
        masks (torch.Tensor): Predicted masks with shape (N, 1, H, W).
        scores (torch.Tensor): Confidence scores for each mask with shape (N, 1).
        original_image (np.ndarray): The original, unprocessed image.

    Returns:
        (Results): Ultralytics Results object containing detection masks, bounding boxes, and other metadata.
    """
    # Ensure the original image is in the correct format
    if not isinstance(original_image, np.ndarray):
        original_image = original_image.cpu().numpy()

    # Ensure masks have the correct shape (N, 1, H, W)
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Expected masks to have shape (N, 1, H, W), but got {masks.shape}")

    # Scale masks to the original image size and remove extra dimensions
    scaled_masks = ops.scale_masks(masks.float(), original_image.shape[:2], padding=False)
    scaled_masks = scaled_masks > 0.5  # Apply threshold to masks

    # Ensure scaled_masks is 3D (N, H, W)
    if scaled_masks.ndim == 4:
        scaled_masks = scaled_masks.squeeze(1)

    # Generate bounding boxes from masks using batched_mask_to_box
    pred_bboxes = batched_mask_to_box(scaled_masks)

    # Ensure scores has shape (N,) by removing extra dimensions
    scores = scores.squeeze().cpu()
    if scores.ndim == 0:  # If only one score, make it a 1D tensor
        scores = scores.unsqueeze(0)

    # Generate class labels
    cls = torch.arange(len(masks), dtype=torch.int32).cpu()

    # Ensure all tensors are 2D before concatenating
    pred_bboxes = pred_bboxes.cpu()
    if pred_bboxes.ndim == 1:
        pred_bboxes = pred_bboxes.unsqueeze(0)
    scores = scores.view(-1, 1)  # Reshape to (N, 1)
    cls = cls.view(-1, 1)  # Reshape to (N, 1)

    # Combine bounding boxes, scores, and class labels
    pred_bboxes = torch.cat([pred_bboxes, scores, cls], dim=1)

    # Create names dictionary (placeholder for consistency)
    names = dict(enumerate(str(i) for i in range(len(masks))))

    # Create Results object
    return Results(original_image, path="", names=names, masks=scaled_masks, boxes=pred_bboxes)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        """
        Initialize the SAM Deploy Model dialog.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("SAM Deploy Model")
        self.resize(300, 200)

        self.imgsz = 1024
        self.conf = 0.25
        self.model_path = None
        self.loaded_model = None

        self.original_image = None
        self.resized_image = None

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Create and set up the tabs
        self.setup_tabs()

        # Custom parameters section
        self.form_layout = QFormLayout()

        # Add resize image dropdown (True / False)
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.form_layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Add imgsz parameter
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # Set the threshold slider for uncertainty
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        self.form_layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        self.form_layout.addRow("", self.uncertainty_threshold_label)

        # Load and Deactivate buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        button_layout.addWidget(deactivate_button)

        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addLayout(button_layout)

        # Status bar label
        self.status_bar = QLabel("No model loaded")
        self.main_layout.addWidget(self.status_bar)

    def update_uncertainty_label(self):
        """
        Update the uncertainty threshold label when the slider value changes.
        """
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def on_uncertainty_changed(self, value):
        """
        Update the uncertainty threshold slider and label when the shared data changes.
        
        Args:
            value (float): The new value of the uncertainty threshold.
        """
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def setup_tabs(self):
        """
        Set up the tabs for the different models.
        """
        self.tabs = QTabWidget()

        # Create tabs
        self.mobile_sam_tab = self.create_model_tab("MobileSAM")
        self.sam_tab = self.create_model_tab("SAM")
        self.sam2_tab = self.create_model_tab("SAM2")

        # Add tabs to the tab widget
        self.tabs.addTab(self.mobile_sam_tab, "MobileSAM")
        self.tabs.addTab(self.sam_tab, "SAM")
        self.tabs.addTab(self.sam2_tab, "SAM2")

        self.main_layout.addWidget(self.tabs)

    def create_model_tab(self, model_name):
        """
        Create a tab for the specified model.
        
        Args:
            model_name (str): The name of the model to create a tab for.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        combo_box = QComboBox()
        combo_box.setEditable(True)

        # Define items for each model
        model_items = {
            "MobileSAM": ["mobile_sam.pt"],
            "SAM": ["sam_b.pt", "sam_l.pt"],
            "SAM2": ["sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt"],
        }

        # Add items to the combo box based on the model name
        if model_name in model_items:
            combo_box.addItems(model_items[model_name])

        layout.addWidget(QLabel(f"Select or Enter {model_name} Model:"))
        layout.addWidget(combo_box)
        return tab

    def download_model_weights(self, model_path):
        """
        Download the model weights if they are not present.
        """
        attempt_download_asset(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not downloaded: {model_path}")

    def load_mobile_model(self, model_path):
        """
        Load a mobile SAM model.
        """
        model = "vit_t"
        loaded_model = mobile_sam_model_registry[model](checkpoint=model_path)
        return MobileSamPredictor(loaded_model)

    def load_sam_model(self, model_path):
        """
        Load a SAM model.
        """
        if "_b" in model_path.lower():
            model = "vit_b"
        elif "_l" in model_path.lower():
            model = "vit_l"
        else:
            raise ValueError(f"Model not recognized: {model_path}")

        loaded_model = sam_model_registry[model](checkpoint=model_path)
        return SamPredictor(loaded_model)

    def load_sam2_model(self, model_path):
        """
        Load a SAM2 model.
        """
        model_ver = model_path.split("_")[0]
        config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
        config = f"{config_dir}\\{model_ver}\\"

        if "_t" in model_path.lower():
            config += f"{model_ver}_hiera_t.yaml"
        elif "_s" in model_path.lower():
            config += f"{model_ver}_hiera_s.yaml"
        elif "_b" in model_path.lower():
            config += f"{model_ver}_hiera_b+.yaml"
        elif "_l" in model_path.lower():
            config += f"{model_ver}_hiera_l.yaml"
        else:
            raise ValueError(f"Model not recognized: {model_path}")

        loaded_model = build_sam2(config_file=config,
                                  ckpt_path=model_path,
                                  device=self.main_window.device,
                                  apply_postprocess=False)

        return Sam2Predictor(loaded_model)

    def load_model(self):
        """
        Load the model selected in the combo box.
        """
        # Make the cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get the model path from the current tab
            self.model_path = self.tabs.currentWidget().layout().itemAt(1).widget().currentText()
            self.download_model_weights(self.model_path)

            if "mobile_" in self.model_path.lower():
                self.loaded_model = self.load_mobile_model(self.model_path)
            elif "sam_" in self.model_path.lower():
                self.loaded_model = self.load_sam_model(self.model_path)
            elif "sam2" in self.model_path.lower():
                self.loaded_model = self.load_sam2_model(self.model_path)
            else:
                raise ValueError(f"Model not recognized: {self.model_path}")

            self.loaded_model.model.to(device=self.main_window.device)
            self.loaded_model.model.eval()
            self.status_bar.setText(f"Model loaded")
            QMessageBox.information(self, "Model Loaded", f"Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
        # Reset the cursor
        QApplication.restoreOverrideCursor()
        # Exit the dialog box
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

    def set_image(self, image):
        """
        Set the image in the predictor.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Setting Image")
        progress_bar.show()

        try:
            if self.loaded_model is not None:
                # Preprocess the image
                image = preprocess_image(image)

                # Save the original image
                self.original_image = image

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

        Args:
            boxes (torch.tensor): The bounding boxes to scale.
        """
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

        if not self.model_path.startswith("sam2"):
            point_coords = self.loaded_model.transform.apply_coords_torch(point_coords,
                                                                          self.resized_image.shape[:2])

        return point_coords, point_labels

    def transform_bboxes(self, bboxes):
        """
        Transform the bounding boxes based on the original and resized image dimensions.

        Args:
            bboxes (np.ndarray): The bounding boxes to transform.
        """
        try:
            input_bbox = torch.as_tensor(bboxes, dtype=torch.int64)
            input_bbox = input_bbox.to(self.main_window.device).unsqueeze(0)
            # Scale the bounding boxes
            bbox_coords = self.scale_boxes(input_bbox)
        except Exception as e:
            input_bbox = torch.as_tensor(bboxes, dtype=torch.int64)
            input_bbox = input_bbox.to(self.main_window.device)
            # Scale the bounding boxes
            bbox_coords = self.scale_boxes(input_bbox)

        if not self.model_path.startswith("sam2"):
            bbox_coords = self.loaded_model.transform.apply_boxes_torch(bbox_coords,
                                                                        self.resized_image.shape[:2])

        return bbox_coords

    def predict_from_prompts(self, bbox, points, labels):
        """
        Make predictions using the currently loaded model using prompts.
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

            if not self.model_path.startswith("sam2"):
                mask, score, _ = self.loaded_model.predict_torch(boxes=bbox_coords,
                                                                 point_coords=point_coords,
                                                                 point_labels=point_labels,
                                                                 multimask_output=False)
            else:
                mask, score, _ = self.loaded_model.predict(box=bbox_coords,
                                                           point_coords=point_coords,
                                                           point_labels=point_labels,
                                                           multimask_output=False)
                mask = torch.tensor(mask).unsqueeze(0)
                score = torch.tensor(score).unsqueeze(0)

            # Post-process the results
            results = to_ultralytics(mask, score, self.original_image)[0]

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")
            return None

        return results

    def predict_from_results(self, results_generator):
        """
        Make predictions using the currently loaded model using results.
        """
        # Create a result processor
        result_processor = ResultsProcessor(self.main_window, class_mapping=None)

        results_dict = {}

        for results in results_generator:
            for result in results:
                # Extract the results
                image_path, cls, cls_name, conf, *bbox = result_processor.extract_detection_result(result)

                if image_path not in results_dict:
                    results_dict[image_path] = []

                # Add the results to the dictionary
                results_dict[image_path].append(np.array(bbox))

        # Loop through each unique image path
        for image_path in results_dict:
            try:
                # Convert rasterio image to numpy array
                image = self.main_window.image_window.rasterio_open(image_path)
                image = rasterio_to_numpy(image)

                # Set the image
                self.set_image(image)

                # Unpack the results as lists
                bboxes, scores, names = map(np.array, zip(*results_dict[image_path]))
                bboxes = np.stack(bboxes)

                # Make predictions
                new_results = self.predict_from_prompts(bboxes, [], [])
                new_results.path = image_path
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
        self.original_image = None
        self.resized_image = None
        gc.collect()
        empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")