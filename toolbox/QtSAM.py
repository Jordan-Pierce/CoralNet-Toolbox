import os
import gc
import warnings

import numpy as np
import torch
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox, QSlider, QLabel, QHBoxLayout, QPushButton,
                             QTabWidget, QComboBox, QMessageBox, QApplication, QWidget)
from PyQt5.QtCore import Qt

from mobile_sam import sam_model_registry as mobile_sam_model_registry
from mobile_sam import SamPredictor as MobileSamPredictor
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils.downloads import attempt_download_asset

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMDeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
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

        self.image = None

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Create and set up the tabs
        self.setup_tabs()

        # Custom parameters section
        self.form_layout = QFormLayout()

        # Add imgsz parameter
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 2048)
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
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def setup_tabs(self):
        self.tabs = QTabWidget()

        # Create tabs
        self.mobile_sam_tab = self.create_model_tab("MobileSAM")
        self.sam_tab = self.create_model_tab("SAM")

        # Add tabs to the tab widget
        self.tabs.addTab(self.mobile_sam_tab, "MobileSAM")
        self.tabs.addTab(self.sam_tab, "SAM")

        self.main_layout.addWidget(self.tabs)

    def create_model_tab(self, model_name):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        combo_box = QComboBox()
        combo_box.setEditable(True)

        # Define items for each model
        model_items = {
            "MobileSAM": ["mobile_sam.pt"],
            "SAM": ["sam_b.pt", "sam_l.pt"]
        }

        # Add items to the combo box based on the model name
        if model_name in model_items:
            combo_box.addItems(model_items[model_name])

        layout.addWidget(QLabel(f"Select or Enter {model_name} Model:"))
        layout.addWidget(combo_box)
        return tab

    def load_model(self):
        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            self.model_path = self.tabs.currentWidget().layout().itemAt(1).widget().currentText()

            # Download the weights if they are not present
            attempt_download_asset(self.model_path)

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not downloaded: {self.model_path}")

            # Load the model
            if "mobile" in self.model_path.lower():
                model = "vit_t"
                self.loaded_model = mobile_sam_model_registry[model](checkpoint=self.model_path)
                self.predictor = MobileSamPredictor(self.loaded_model)
            else:
                if "vit_b" in self.model_path.lower():
                    model = "vit_b"
                else:
                    model = "vit_l"
                self.loaded_model = sam_model_registry[model](checkpoint=self.model_path)
                self.predictor = SamPredictor(self.loaded_model)

            self.predictor.model.to(device=self.main_window.device)
            self.predictor.model.eval()

            QApplication.restoreOverrideCursor()
            self.status_bar.setText(f"Model loaded")
            QMessageBox.information(self, "Model Loaded", f"Model loaded successfully")

            # Exit the dialog box
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")

    def set_image(self, image):
        if self.predictor is not None:
            # Reset the image in the predictor
            self.predictor.reset_image()
            # Set the image in the predictor
            self.predictor.set_image(image)
            self.image = image
        else:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded")

    def predict(self, points, labels):
        if not self.predictor:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded")
            return None
        try:
            # Provide prompt to SAM model in form of numpy array
            input_labels = torch.tensor(labels)
            input_points = torch.as_tensor(points.astype(int), dtype=torch.int64)
            input_labels = input_labels.to(self.main_window.device).unsqueeze(0)
            input_points = input_points.to(self.main_window.device).unsqueeze(0)
            transformed_points = self.predictor.transform.apply_coords_torch(input_points, self.image.shape[:2])

            mask, score, logit = self.predictor.predict_torch(point_coords=transformed_points,
                                                              point_labels=input_labels,
                                                              multimask_output=False)

            # Post-process the results
            results = self.custom_postprocess(mask, score, logit, self.image)[0]

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")
            return None

        return results

    @staticmethod
    def custom_postprocess(mask, score, logit, original_image):
        """
        Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.

        Args:
            mask (torch.Tensor): Predicted masks with shape (1, 1, H, W).
            score (torch.Tensor): Confidence scores for each mask with shape (1, 1).
            logit (torch.Tensor): Logits for each mask with shape (1, 1, H, W).
            original_image (np.ndarray): The original, unprocessed image.

        Returns:
            (Results): Results object containing detection masks, bounding boxes, and other metadata.
        """
        # Ensure the original image is in the correct format
        if not isinstance(original_image, np.ndarray):
            original_image = original_image.cpu().numpy()

        # Ensure mask has the correct shape (1, 1, H, W)
        if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(f"Expected mask to have shape (1, 1, H, W), but got {mask.shape}")

        # Scale masks to the original image size
        scaled_masks = ops.scale_masks(mask.float(), original_image.shape[:2], padding=False)[0]
        scaled_masks = scaled_masks > 0.5  # Apply threshold to masks

        # Generate bounding boxes from masks using batched_mask_to_box
        pred_bboxes = batched_mask_to_box(scaled_masks)

        # Ensure score and cls have the correct shape
        score = score.squeeze(1)  # Remove the extra dimension
        cls = torch.arange(len(mask), dtype=torch.int32, device=mask.device)

        # Combine bounding boxes, scores, and class labels
        pred_bboxes = torch.cat([pred_bboxes, score[:, None], cls[:, None]], dim=-1)

        # Create names dictionary (placeholder for consistency)
        names = dict(enumerate(str(i) for i in range(len(mask))))

        # Create Results object
        result = Results(original_image, path="", names=names, masks=scaled_masks, boxes=pred_bboxes)

        return result

    def deactivate_model(self):
        self.loaded_model = None
        self.predictor = None
        self.model_path = None
        self.image = None
        gc.collect()
        torch.cuda.empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")


class SAMBatchInferenceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM Batch Inference")
        # Add additional initialization code here