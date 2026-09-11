import warnings

import os
import gc

import numpy as np

import torch
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
from coralnet_toolbox.Icons import get_icon, get_window_icon

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

        self.setWindowIcon(get_window_icon("wizard.svg"))
        self.setWindowTitle("SAM Deploy Model")
        self.resize(400, 325)

        # Initialize instance variables
        self.imgsz = 640 if not cuda_is_available() else 1024  # Default to smaller size on CPU for performance
        self.model_path = None
        self.loaded_model = None
        self.image_path = None
        self.original_image = None
        # Model input size the cached features were encoded at, as (h, w).
        # Prompts must be scaled to this, not to the spinbox, which can change
        # after encoding and which ultralytics rounds per model stride.
        self.features_imgsz = None

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
            
 # Check for SAM 3 weights in the current directory and add to models if found
        if os.path.exists(os.path.join(os.getcwd(), "sam3.1_multiplex.pt")):
            self.models["SAM 3.1 Multiplex"] = "sam3.1_multiplex.pt"
        # Add all models to combo box
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)

        models = list(self.models.keys())
        self.model_combo.setCurrentIndex(models.index("SAM 2.1 Tiny"))
        self.model_combo.setToolTip("Choose a SAM variant for interactive segmentation.\nTiny/Small: Faster, less memory.\nBase/Large: Higher accuracy, more resources.")

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
        self.output_type_dropdown.setToolTip("Format for SAM output annotations.\nPolygon: Free-form shapes.\nRectangle: Bounding boxes.\nMask: Binary segmentation masks.")
        layout.addRow("Output Type:", self.output_type_dropdown)

        # Allow holes dropdown
        self.allow_holes_dropdown = QComboBox()
        self.allow_holes_dropdown.addItems(["True", "False"])
        self.allow_holes_dropdown.setCurrentIndex(1)  # Default to False
        self.allow_holes_dropdown.setToolTip("Allow segmentation masks with holes (interior regions).\nEnable for complex shapes, disable for simpler contours.")
        layout.addRow("Allow Holes:", self.allow_holes_dropdown)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        self.resize_image_dropdown.setToolTip("(Automatic) Resize image to match model input requirements.")
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(640, 2048)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.imgsz_spinbox.setToolTip("Input image size for SAM model.\nLarger sizes improve accuracy but use more GPU memory.\n"
                                      "Rounded to a multiple of 32.")
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
        load_button.setToolTip("Load the selected SAM model for interactive segmentation.")
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        deactivate_button.setToolTip("Unload the current SAM model and free GPU memory.")
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

    def _on_cpu(self):
        """True when inference runs on the CPU (device is 'cpu', 'cuda:0', 'mps' or '0,1 ')."""
        return str(self.main_window.device).strip().lower() == "cpu"

    def _snapped_imgsz(self):
        """Return the spinbox image size rounded to a multiple of 32.

        SAM 2's Hiera encoder raises in set_image on sizes that aren't (1000 and
        688, both reachable with the old 24-px step); multiples of 32 work for
        SAM, SAM 2 and SAM 3. The spinbox is updated so it shows what is used.
        """
        value = self.imgsz_spinbox.value()
        snapped = int(round(value / 32)) * 32
        snapped = max(self.imgsz_spinbox.minimum(), min(self.imgsz_spinbox.maximum(), snapped))
        if snapped != value:
            self.imgsz_spinbox.setValue(snapped)
        return snapped

    def load_model(self):
        """
        Load the selected SAM model using the appropriate Predictor class.
        
        - SAM 2 models use SAM2Predictor
        - SAM 3 models use SAM3Predictor
        - Other models use SAMPredictor
        
        Ultralytics will automatically download missing weights.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Obtaining model...", 3000)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get selected model path and name
            selected_model_name = self.model_combo.currentText()
            self.model_path = self.models[selected_model_name]
            
            # Get imgsz and confidence from UI
            imgsz = self._snapped_imgsz()
            conf = self.thresholds_widget.get_uncertainty_thresh()

            # FP16 only off the CPU: MobileSAM doesn't support it, and on the CPU
            # SAM's .half() cast is ~15x slower than FP32 (sam2.1_t encodes in
            # 17 s vs 1.1 s).
            use_fp16 = selected_model_name != "MobileSAM" and not self._on_cpu()

            # Create overrides dictionary
            overrides = dict(
                task="detect" if self.get_output_type() == "Rectangle" else "segment",
                mode="predict",
                imgsz=imgsz,
                model=self.model_path,
                conf=conf,
                device=self.main_window.device,
                retina_masks=False,
                quantize=16 if use_fp16 else 32,
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
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")
            # The dialog has done its job; leaving it up meant it reappeared
            # behind the message box and had to be dismissed a second time.
            # A failed load keeps it open instead, so the choice can be retried.
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")
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
            # Encode at the spinbox size, then record the size ultralytics
            # actually used (SAM 3's stride of 14 turns 1024 into 1036).
            self.loaded_model.args.imgsz = self._snapped_imgsz()

            # no_grad: SAM's parameters keep requires_grad=True and this
            # ultralytics version's set_image has no inference decorator, so
            # the features would otherwise hold the encoder's autograd graph
            # (~1.3 GB for sam2.1_t, vs ~0.1 GB). Not inference_mode: the model
            # is built lazily inside this call, and weights created there would
            # be inference tensors.
            with torch.no_grad():
                self.loaded_model.set_image(image)
            self.features_imgsz = tuple(self.loaded_model.imgsz)

        except Exception as e:
            QMessageBox.critical(self.annotation_window, "Error Setting Image", f"Error setting image: {e}")
            self.original_image = None
            self.image_path = None
            self.features_imgsz = None
            # Don't leave the previous image's features behind to be
            # prompted against.
            self.loaded_model.reset_image()

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def has_image(self, image):
        """Return True if the cached features were encoded from `image` (the same array object).

        The predictor is shared: batch callers (predict_from_results) and the
        See Anything tool encode their own images into it, so an interactive
        tool must check before prompting against features it assumes are its own.
        """
        return (self.loaded_model is not None
                and image is not None
                and self.original_image is image
                and getattr(self.loaded_model, 'features', None) is not None)

    def _snapshot_session(self):
        """Capture the currently encoded image so a batch call can put it back."""
        features = getattr(self.loaded_model, 'features', None)
        if self.original_image is None or features is None:
            return None
        return self.original_image, self.image_path, features, self.features_imgsz

    def _restore_session(self, session):
        """Put back an encoded image captured by _snapshot_session.

        Skipped if the batch encoded at a different size: set_image resizes the
        prompt encoder to match, so the old features would no longer line up.
        Tools then see has_image() is False and re-encode.
        """
        if session is None or self.loaded_model is None:
            return
        image, image_path, features, features_imgsz = session
        if tuple(getattr(self.loaded_model, 'imgsz', None) or ()) != features_imgsz:
            return
        self.original_image = image
        self.image_path = image_path
        self.loaded_model.features = features
        self.features_imgsz = features_imgsz

    def predict_from_prompts(self, bbox=None, points=None, labels=None):
        """
        Run SAM inference with prompts on the currently set image.
        
        Optimized version: Bypasses the high-level Ultralytics Predictor loop
        and directly queries the mask decoder using cached image features.

        Returns None instead of raising when there is no model or image: the
        SAM tool calls this from hover timers and mouse handlers, where an
        exception reaches the app's excepthook, which exits the app.
        """
        if self.loaded_model is None:
            self.main_window.status_bar.showMessage("No SAM model loaded.", 3000)
            return None

        if self.original_image is None or getattr(self.loaded_model, 'features', None) is None:
            self.main_window.status_bar.showMessage("No image set for the SAM predictor.", 3000)
            return None

        # Update the threshold from the UI. The image size is not: prompts are
        # scaled to features_imgsz, the size these features were encoded at.
        self.loaded_model.args.conf = self.thresholds_widget.get_uncertainty_thresh()

        # Make cursor busy while predicting
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Running fast prediction...", 2000)

        try:
            from ultralytics.engine.results import Results

            src_shape = self.original_image.shape[:2]

            # Run inference directly on the pre-encoded features. SAM 3's
            # interactive predictor is SAM 2-style (points, boxes, dst_shape),
            # so one call covers every model.
            with torch.inference_mode():
                pred_masks, pred_bboxes = self.loaded_model.inference_features(
                    features=self.loaded_model.features,
                    src_shape=src_shape,
                    dst_shape=self.features_imgsz,
                    bboxes=bbox,
                    points=points,
                    labels=labels,
                    multimask_output=False
                )

            # Gracefully handle empty predictions
            if pred_masks is None or len(pred_masks) == 0:
                return []
                
            # Create dummy names dictionary (Ultralytics Results objects expect this)
            names = {i: str(i) for i in range(len(pred_masks))}
            
            # Pack the raw tensors back into an Ultralytics Results object 
            # so the rest of your app doesn't break
            results = [Results(
                orig_img=self.original_image, 
                path=self.image_path, 
                names=names, 
                masks=pred_masks, 
                boxes=pred_bboxes
            )]
            
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
        
        Optimized version: Directly queries the mask decoder using cached features,
        bypassing UI overhead and redundant wrapper logic during batch processing.
        """
        if self.loaded_model is None or not results_list:
            return results_list

        # Whatever is encoded now (the SAM or See Anything tool's work area) is
        # put back afterwards; see the finally block.
        session = self._snapshot_session()

        # Set UI state once for the entire batch operation
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Refining masks with SAM...", 2000)

        try:
            output_results = []

            # Sync the threshold once (set_image syncs the image size)
            self.loaded_model.args.conf = self.thresholds_widget.get_uncertainty_thresh()

            for results in results_list:
                original_image = results.orig_img
                
                # Fast shape check to avoid re-running the heavy ViT encoder
                image_changed = True
                if self.original_image is not None:
                    if self.original_image.shape == original_image.shape:
                        if np.array_equal(self.original_image, original_image):
                            image_changed = False
                            
                if image_changed:
                    # set_image handles the heavy ViT backbone feature extraction
                    self.set_image(original_image, image_path or results.path)

                # A failed set_image has already reported itself and cleared
                # the features; pass these results through unrefined.
                if results.boxes is None or len(results.boxes) == 0 or self.features_imgsz is None:
                    output_results.append(results)
                    continue

                # Keep bounding boxes on the GPU as a PyTorch tensor!
                bboxes = results.boxes.xyxy

                # --- THE HARDWARE-AGNOSTIC FIX: ADAPTIVE CHUNKING ---
                current_chunk_size = 256
                i = 0
                all_sam_masks = []

                src_shape = original_image.shape[:2]

                while i < len(bboxes):
                    bbox_chunk = bboxes[i:i + current_chunk_size]

                    try:
                        # Run fast inference directly on the chunk. One call
                        # covers SAM, SAM 2 and SAM 3 (see predict_from_prompts).
                        with torch.inference_mode():
                            pred_masks, _ = self.loaded_model.inference_features(
                                features=self.loaded_model.features,
                                src_shape=src_shape,
                                dst_shape=self.features_imgsz,
                                bboxes=bbox_chunk,
                                multimask_output=False
                            )
                        
                        if pred_masks is not None and len(pred_masks) > 0:
                            # --- THE VRAM SWAP FIX: COMPRESS TO BOOLEAN AND MOVE TO CPU ---
                            # Move each chunk to CPU immediately to avoid accumulating
                            # GPU-resident tensors prior to the final concat.
                            compressed_masks = pred_masks.to(torch.bool).cpu()
                            all_sam_masks.append(compressed_masks)
                        
                        i += current_chunk_size
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                            if current_chunk_size <= 1:
                                QMessageBox.critical(self.annotation_window, 
                                                     "GPU Memory Error", 
                                                     "Your GPU does not have enough memory to process this image.")
                                break
                            
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                            current_chunk_size = current_chunk_size // 2
                            print(f"VRAM limit reached. Reducing SAM batch size to {current_chunk_size}...")
                        else:
                            raise e
                
                if all_sam_masks:
                    # Keep masks as bool on CPU — 4× smaller than float32 and
                    # Ultralytics' Masks.xy normalises via .astype('uint8') before
                    # cv2.findContours, so bool input is accepted downstream.
                    combined_masks = torch.cat(all_sam_masks, dim=0)
                    results.update(masks=combined_masks)
                    del all_sam_masks, combined_masks
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            empty_cache()
                        except Exception:
                            pass

                output_results.append(results)

            return output_results

        finally:
            # Release the batch images' features and arrays so RAM and VRAM
            # don't accumulate across a long batch, then put back what was
            # encoded before this call. Interactive tools share this predictor:
            # wiping their work area's features left the SAM tool prompting
            # against nothing, which raised inside a hover handler and exited
            # the app.
            try:
                self.original_image = None
                self.image_path = None
                self.features_imgsz = None
                # reset_image() zeros both .im (preprocessed tensor, GPU) and
                # .features (ViT output, GPU); covers SAM / SAM2 / SAM3.
                if hasattr(self.loaded_model, 'reset_image'):
                    try:
                        self.loaded_model.reset_image()
                    except Exception:
                        pass
                # Drop references rather than clearing dicts in place: SAM 2/3
                # features are a dict that may be the session's, restored below.
                for attr in ('features', 'im', 'interm_features', 'dataset'):
                    if hasattr(self.loaded_model, attr):
                        try:
                            setattr(self.loaded_model, attr, None)
                        except Exception:
                            pass
                # Ultralytics pops from .prompts, so it must stay a dict.
                prompts = getattr(self.loaded_model, 'prompts', None)
                if isinstance(prompts, dict):
                    prompts.clear()
            except Exception:
                pass
            self._restore_session(session)
            gc.collect()
            try:
                empty_cache()
            except Exception:
                pass
            QApplication.restoreOverrideCursor()

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        # Clear the model
        self.loaded_model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None
        self.features_imgsz = None
        # Clear the cache
        gc.collect()
        empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update the status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self.annotation_window, "Model Deactivated", "Model deactivated")
