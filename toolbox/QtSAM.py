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


class SAMDeployModelDialog(QDialog):
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
        self.predictor = None

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
            self.model_path = self.tabs.currentWidget().layout().itemAt(1).widget().currentText()

            # Download the weights from ultralytics if they are not present
            attempt_download_asset(self.model_path)

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not downloaded: {self.model_path}")

            # Load the model
            if "mobile_" in self.model_path.lower():
                model = "vit_t"
                self.loaded_model = mobile_sam_model_registry[model](checkpoint=self.model_path)
                self.predictor = MobileSamPredictor(self.loaded_model)
            elif "sam_" in self.model_path.lower():
                if "_b" in self.model_path.lower():
                    model = "vit_b"
                elif "_l" in self.model_path.lower():
                    model = "vit_l"
                else:
                    raise ValueError(f"Model not recognized: {self.model_path}")
                self.loaded_model = sam_model_registry[model](checkpoint=self.model_path)
                self.predictor = SamPredictor(self.loaded_model)
            elif "sam2" in self.model_path.lower():
                model_ver = self.model_path.split("_")[0]
                config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
                config = f"{config_dir}\\{model_ver}\\"
                if "_t" in self.model_path.lower():
                    config += f"{model_ver}_hiera_t.yaml"
                elif "_s" in self.model_path.lower():
                    config += f"{model_ver}_hiera_s.yaml"
                elif "_b" in self.model_path.lower():
                    config += f"{model_ver}_hiera_b+.yaml"
                elif "_l" in self.model_path.lower():
                    config += f"{model_ver}_hiera_l.yaml"
                else:
                    raise ValueError(f"Model not recognized: {self.model_path}")

                self.loaded_model = build_sam2(config_file=config,
                                               ckpt_path=self.model_path,
                                               device=self.main_window.device,
                                               apply_postprocess=False)

                self.predictor = Sam2Predictor(self.loaded_model)
            else:
                raise ValueError(f"Model not recognized: {self.model_path}")

            self.predictor.model.to(device=self.main_window.device)
            self.predictor.model.eval()

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
        
        Args:
            image (np.ndarray): The image to resize.
        """
        # Get the current image size
        imgsz = self.imgsz_spinbox.value()
        # Determine the target shape based on the long side
        h, w = image.shape[:2]
        if h > w:
            target_shape = (imgsz, int(w * (imgsz / h)))
        else:
            target_shape = (int(h * (imgsz / w)), imgsz)

        # Use scale_image to resize and pad the image
        resized_image = ops.scale_image(image, target_shape)
        return resized_image

    def set_image(self, image):
        """
        Set the image in the predictor.
        
        Args:
            image (np.ndarray): The image to set.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Setting Image")
        progress_bar.show()

        try:
            if self.predictor is not None:
                # Reshape image if needed
                image = preprocess_image(image)

                # Save the original image
                self.original_image = image

                # Resize the image if the checkbox is checked
                if self.resize_image_dropdown.currentText() == "True":
                    image = self.resize_image(image)

                # Verify final dimensions
                if len(image.shape) != 3 or image.shape[2] != 3:
                    raise ValueError(f"Invalid image dimensions: {image.shape}. Expected (H, W, 3)")

                # Set the image in the predictor
                self.predictor.set_image(image)
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

    def predict(self, bbox, points, labels):
        """
        Make predictions using the currently loaded model.
        
        Args:
            bbox (list): List of bounding box coordinates in the form [xmin, ymin, xmax, ymax].
            points (np.ndarray): Array of points in the form [[x1, y1], [x2, y2], ...].
            labels (list): List of class labels for each point.
        """
        if not self.predictor:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded, cannot make predictions")
            return None

        try:
            point_labels = None
            point_coords = None
            bbox_coords = None

            # Calculate scaling factors
            original_height, original_width = self.original_image.shape[:2]
            resized_height, resized_width = self.resized_image.shape[:2]
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height

            has_points = len(points) != 0
            has_bbox = len(bbox) != 0

            if not has_points and not has_bbox:
                return None

            if has_points:
                input_labels = torch.tensor(labels)
                point_labels = input_labels.to(self.main_window.device).unsqueeze(0)

                # Provide prompt to SAM model in form of numpy array
                input_points = torch.as_tensor(points.astype(int), dtype=torch.int64)
                input_points = input_points.to(self.main_window.device).unsqueeze(0)

                # Scale the points based on the original image dimensions
                scaled_points = input_points.clone().float()  # Cast to float32
                scaled_points[:, :, 0] *= scale_x
                scaled_points[:, :, 1] *= scale_y
                point_coords = scaled_points.long()  # Cast back to int64

                if not self.model_path.startswith("sam2"):
                    # Apply the scaled points to the predictor
                    point_coords = self.predictor.transform.apply_coords_torch(point_coords,
                                                                               self.resized_image.shape[:2])

            if has_bbox:
                input_bbox = torch.as_tensor(bbox, dtype=torch.int64)
                input_bbox = input_bbox.to(self.main_window.device).unsqueeze(0)

                # Scale the box based on the original image dimensions
                scaled_bbox = input_bbox.clone().float()  # Cast to float32
                scaled_bbox[:, 0] *= scale_x
                scaled_bbox[:, 1] *= scale_y
                scaled_bbox[:, 2] *= scale_x
                scaled_bbox[:, 3] *= scale_y
                bbox_coords = scaled_bbox.long()  # Cast back to int64

                if not self.model_path.startswith("sam2"):
                    # Apply the scaled boxes to the predictor
                    bbox_coords = self.predictor.transform.apply_boxes_torch(bbox_coords,
                                                                             self.resized_image.shape[:2])

            if self.model_path.startswith("sam2"):
                mask, score, _ = self.predictor.predict(box=bbox_coords,
                                                        point_coords=point_coords,
                                                        point_labels=point_labels,
                                                        multimask_output=False)

                mask = torch.tensor(mask).unsqueeze(0)
                score = torch.tensor(score).unsqueeze(0)

            else:
                mask, score, _ = self.predictor.predict_torch(boxes=bbox_coords,
                                                              point_coords=point_coords,
                                                              point_labels=point_labels,
                                                              multimask_output=False)

            # Post-process the results
            results = to_ultralytics(mask, score, self.original_image)[0]

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")
            return None

        return results

    def boxes_to_masks(self, results_generator):
        """
        Convert bounding boxes to masks using the currently loaded model.
        
        Args:
            results_generator (generator): Generator of Results objects containing bounding boxes.
        """
        results_dict = {}
        for results in results_generator:
            path = results.path.replace("\\", "/")
            results_dict[path] = []
            for i, result in enumerate(results):
                if not len(result):
                    continue
                # Extract the results
                cls = int(result.boxes.cls.cpu().numpy()[0])
                cls_name = result.names[cls]
                conf = float(result.boxes.conf.cpu().numpy()[0])
                xmin, ymin, xmax, ymax = map(float, result.boxes.xyxy.cpu().numpy()[0])
                bbox = np.array([xmin, ymin, xmax, ymax])
                results_dict[path].append((bbox, conf, cls_name))

        for image_path in results_dict:
            # Convert rasterio image to numpy array
            image = self.main_window.image_window.rasterio_open(image_path)
            image = rasterio_to_numpy(image)

            # Set the image
            self.set_image(image)

            # Calculate scaling factors
            original_height, original_width = self.original_image.shape[:2]
            resized_height, resized_width = self.resized_image.shape[:2]
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height

            bboxes, scores, cls_names = zip(*results_dict[image_path])

            # Prepare the input
            input_bbox = torch.as_tensor(bboxes, dtype=torch.int64)
            input_bbox = input_bbox.to(self.main_window.device)

            # Scale the box based on the original image dimensions
            scaled_bbox = input_bbox.clone().float()  # Cast to float32
            scaled_bbox[:, 0] *= scale_x
            scaled_bbox[:, 1] *= scale_y
            scaled_bbox[:, 2] *= scale_x
            scaled_bbox[:, 3] *= scale_y
            bbox_coords = scaled_bbox.long()  # Cast back to int64

            if not self.model_path.startswith("sam2"):
                # Apply the scaled boxes to the predictor
                bbox_coords = self.predictor.transform.apply_boxes_torch(bbox_coords,
                                                                         self.resized_image.shape[:2])

            if self.model_path.startswith("sam2"):
                mask, score, logit = self.predictor.predict(box=bbox_coords,
                                                            point_coords=None,
                                                            point_labels=None,
                                                            multimask_output=False)

                mask = torch.tensor(mask).unsqueeze(0)
                score = torch.tensor(score).unsqueeze(0)

            else:
                mask, score, _ = self.predictor.predict_torch(boxes=bbox_coords,
                                                              point_coords=None,
                                                              point_labels=None,
                                                              multimask_output=False)

            # Post-process the results
            sam_results = to_ultralytics(mask, score, self.original_image)
            sam_results.boxes = results.boxes
            sam_results.names = results.names
            sam_results.path = image_path

            yield sam_results

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        self.loaded_model = None
        self.predictor = None
        self.model_path = None
        self.original_image = None
        self.resized_image = None
        gc.collect()
        empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")