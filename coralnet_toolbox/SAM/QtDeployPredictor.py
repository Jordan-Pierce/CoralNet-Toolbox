import warnings

import os
import gc

import numpy as np

from torch.cuda import empty_cache
from torch.cuda import is_available as cuda_is_available

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSpinBox, QVBoxLayout, QGroupBox)

from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM2Predictor, SAM3Predictor

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Common import ThresholdsWidget
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployPredictorDialog(QDialog):
    def __init__(self, main_window, parent=None):
        """Initialize the SAM Deploy Model dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("wizard.svg"))
        self.setWindowTitle("SAM Deploy Model")
        self.resize(400, 325)

        # Initialize instance variables
        self.imgsz = 640 if not cuda_is_available() else 1024  # Default to smaller size on CPU for performance
        self.model_path = None
        self.loaded_model = None
        self.image_path = None
        self.original_image = None

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the thresholds layout
        self.setup_thresholds_layout()
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
        self.thresholds_widget.initialize_thresholds()

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

        # Define available models with official Ultralytics weights
        self.models = {
            "MobileSAM": "mobile_sam.pt",
            "SAM-Base": "sam_b.pt",
            "SAM-Large": "sam_l.pt",
            "SAM-Huge": "sam_h.pt",
            "SAM 2 Tiny": "sam2_t.pt",
            "SAM 2 Small": "sam2_s.pt",
            "SAM 2 Base": "sam2_b.pt",
            "SAM 2 Large": "sam2_l.pt",
            "SAM 2.1 Tiny": "sam2.1_t.pt",
            "SAM 2.1 Small": "sam2.1_s.pt",
            "SAM 2.1 Base": "sam2.1_b.pt",
            "SAM 2.1 Large": "sam2.1_l.pt"
        }
        
        # Check for SAM 3 weights in the current directory and add to models if found
        if os.path.exists(os.path.join(os.getcwd(), "sam3.pt")):
            self.models["SAM 3"] = "sam3.pt"

        # Add all models to combo box
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)

        models = list(self.models.keys())
        self.model_combo.setCurrentIndex(models.index("SAM 2.1 Tiny"))

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
        
        # Output type dropdown (polygon or rectangle)
        self.output_type_dropdown = QComboBox()
        self.output_type_dropdown.addItems(["Polygon", "Rectangle", "Mask"])
        self.output_type_dropdown.setCurrentIndex(0)  # Default to Polygon
        layout.addRow("Output Type:", self.output_type_dropdown)
        
        # Allow holes dropdown
        self.allow_holes_dropdown = QComboBox()
        self.allow_holes_dropdown.addItems(["True", "False"])
        self.allow_holes_dropdown.setCurrentIndex(1)  # Default to False
        layout.addRow("Allow Holes:", self.allow_holes_dropdown)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(640, 2048)
        self.imgsz_spinbox.setSingleStep(24)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_thresholds_layout(self):
        """
        Setup the thresholds layout using ThresholdsWidget.
        """
        # Add ThresholdsWidget for all threshold controls
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=False,
            show_uncertainty=True,
            show_iou=False,
            show_area=False
        )
        self.layout.addWidget(self.thresholds_widget)

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

    def get_allow_holes(self):
        """Return the current setting for allowing holes."""
        return self.allow_holes_dropdown.currentText() == "True"
    
    def get_output_type(self):
        """Return the current setting for output type."""
        return self.output_type_dropdown.currentText()

    def load_model(self):
        """
        Load the selected SAM model using the appropriate Predictor class.
        
        - SAM 2 models use SAM2Predictor
        - SAM 3 models use SAM3Predictor
        - Other models use SAMPredictor
        
        Ultralytics will automatically download missing weights.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get selected model path and name
            selected_model_name = self.model_combo.currentText()
            self.model_path = self.models[selected_model_name]
            
            # Get imgsz and confidence from UI
            imgsz = self.imgsz_spinbox.value()
            conf = self.thresholds_widget.get_uncertainty_thresh()

            # Create overrides dictionary
            overrides = dict(
                task="detect" if self.get_output_type() == "Rectangle" else "segment",
                mode="predict",
                imgsz=imgsz,
                model=self.model_path,
                conf=conf,
                device=self.main_window.device,
                retina_masks=False,
                half=True if selected_model_name != "MobileSAM" else False,  # MobileSAM doesn't support half precision
                save=False, 
                show=False, 
                save_txt=False
            )
            
            # Select the appropriate predictor class based on model
            if "SAM 2" in selected_model_name:
                # SAM 2 and SAM 2.1 models use SAM2Predictor
                self.loaded_model = SAM2Predictor(overrides=overrides)
            elif "SAM 3" in selected_model_name:
                # SAM 3 models use SAM3Predictor
                self.loaded_model = SAM3Predictor(overrides=overrides)
            else:
                # SAM, MobileSAM use standard SAMPredictor
                self.loaded_model = SAMPredictor(overrides=overrides)

            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {self.model_path}")
            QMessageBox.information(self.annotation_window, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self.annotation_window, "Error Loading Model", f"Error loading model: {e}")
            self.loaded_model = None
            self.model_path = None

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def set_image(self, image, image_path):
        """
        Set the image in the SAM predictor for subsequent prompt inference.
        
        This sets the image once so we can efficiently pass multiple prompts
        without re-processing the same image.
        
        Args:
            image (np.ndarray): The original image array (H, W, C).
            image_path (str): Path to the image file.
        """
        if self.loaded_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Make cursor busy while setting the image
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        self.original_image = image
        self.image_path = image_path
        
        # Ultralytics will download the model for the user
        if not os.path.exists(self.model_path):
            # Inform the user that the model is being downloaded via main window status bar
            self.main_window.status_bar.showMessage(f"Downloading model weights for {self.model_path}...", 5000)
        else:
            # Indicate that the image is being set for the predictor
            self.main_window.status_bar.showMessage("Setting image for predictor...", 2000)
        
        try:
            # Set the image in the predictor
            self.loaded_model.set_image(image)
            
        except Exception as e:
            QMessageBox.critical(self.annotation_window, "Error Setting Image", f"Error setting image: {e}")
            self.original_image = None
            self.image_path = None
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def predict_from_prompts(self, bbox=None, points=None, labels=None):
        """
        Run SAM inference with prompts on the currently set image.
        
        The image should be set first with set_image(). This method only passes
        the prompts to the predictor for efficient inference.
        
        Args:
            bbox (list or np.ndarray): Bounding box [x1, y1, x2, y2] or list of bboxes (optional).
            points (list or np.ndarray): Point coordinates [[x, y], ...] (optional).
            labels (list or np.ndarray): Point labels [1, -1, ...] (optional).
        
        Returns:
            list: List of Ultralytics Results objects.
        """
        if self.loaded_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.original_image is None:
            raise RuntimeError("No image set. Call set_image() first.")
        
        # Update the parameters and threshold from the UI
        self.loaded_model.args.imgsz = self.imgsz_spinbox.value()
        self.loaded_model.args.conf = self.thresholds_widget.get_uncertainty_thresh()
        
        # Make cursor busy while predicting
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Running prediction...", 2000)
        
        try:
            # Call the predictor with prompts
            # Image is already set, so we just pass the prompts
            results = self.loaded_model(
                points=points,
                labels=labels,
                bboxes=bbox
            )
            
            return results
            
        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            return None
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def predict_from_results(self, results_list, image_path=None):
        """
        Apply SAM to YOLO detection results to add segmentation masks.
        Extracts bounding boxes from YOLO detections and uses them as prompts for SAM.
        """
        if self.loaded_model is None or len(results_list) == 0:
            return results_list
        
        output_results = []
        import torch
        
        for results in results_list:
            original_image = results.orig_img
            
            # Fast shape check to avoid re-running the heavy ViT encoder
            image_changed = True
            if self.original_image is not None:
                if self.original_image.shape == original_image.shape:
                    if np.array_equal(self.original_image, original_image):
                        image_changed = False
                        
            if image_changed:
                self.set_image(original_image, image_path or results.path)
            
            if results.boxes is None or len(results.boxes) == 0:
                output_results.append(results)
                continue
            
            # Keep bounding boxes on the GPU as a PyTorch tensor!
            bboxes = results.boxes.xyxy 
            
            # --- THE HARDWARE-AGNOSTIC FIX: ADAPTIVE CHUNKING ---
            # Start with a highly optimistic chunk size
            current_chunk_size = 256
            
            # We use a while loop so we can dynamically adjust the chunk size if we hit an OOM
            i = 0
            all_sam_masks = []
            
            while i < len(bboxes):
                bbox_chunk = bboxes[i:i + current_chunk_size]
                
                try:
                    # Attempt to run SAM with the current chunk
                    sam_results = self.predict_from_prompts(bbox=bbox_chunk)
                    
                    if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
                        # --- THE VRAM SWAP FIX: COMPRESS TO BOOLEAN ---
                        # Shrinks the stored output by 32x to prevent memory accumulation
                        compressed_masks = sam_results[0].masks.data.to(torch.bool)
                        all_sam_masks.append(compressed_masks)
                    
                    # If successful, move forward by the chunk size
                    i += current_chunk_size
                    
                except RuntimeError as e:
                    # Check if the error is a CUDA Out Of Memory error
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        # If we are already at chunk size 1 and it OOMs, the user's GPU is simply too small
                        if current_chunk_size <= 1:
                            from PyQt5.QtWidgets import QMessageBox
                            QMessageBox.critical(self.annotation_window, 
                                                 "GPU Memory Error", 
                                                 "Your GPU does not have enough memory to process this image.")
                            break
                        
                        # The chunk was too big. Clear the failed memory allocation...
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        # ...cut the chunk size in half, and try this exact same index (i) again!
                        current_chunk_size = current_chunk_size // 2
                        print(f"VRAM limit reached. Reducing SAM batch size to {current_chunk_size}...")
                    else:
                        # If it's a different runtime error, raise it normally
                        raise e
            
            # Stitch all the chunks back together into a single tensor
            if all_sam_masks:
                # Concatenate the boolean masks, then cast back to float to satisfy Ultralytics
                combined_masks = torch.cat(all_sam_masks, dim=0).float()
                results.update(masks=combined_masks)
            
            output_results.append(results)
        
        return output_results

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        # Clear the model
        self.loaded_model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None
        # Clear the cache
        gc.collect()
        empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update the status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self.annotation_window, "Model Deactivated", "Model deactivated")
