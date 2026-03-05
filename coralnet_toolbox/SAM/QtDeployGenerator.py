import warnings

import gc
import os

import numpy as np

import torch
from torch.cuda import empty_cache
from torch.cuda import is_available as is_cuda_available

from ultralytics import SAM, FastSAM

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QSpinBox,
                             QVBoxLayout, QGroupBox)

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployGeneratorDialog(QDialog):
    """
    Dialog for deploying SAM/FastSAM models for unified segment-everything generation.
    Supports SAM, SAM2, SAM2.1, SAM3, FastSAM, and MobileSAM models.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the DeployGeneratorDialog.

        Args:
            main_window: The main application window.
            parent: The parent widget, default is None.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("wizard.svg"))
        self.setWindowTitle("SAM Generator (Ctrl + 5)")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 640 if not is_cuda_available() else 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40

        self.task = 'detect'
        self.max_detect = 300
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None
        self.model_type = None  # Either 'fastSAM' or 'sam'

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
        self.update_detect_as_combo()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("SAM generator for segment-everything inference. Select a model and tune thresholds.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup model selection dropdown in a group box.
        """
        group_box = QGroupBox("Models")
        layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models with official Ultralytics weights
        self.models = {
            "FastSAM Small": "FastSAM-s.pt",
            "FastSAM Large": "FastSAM-x.pt",
            "MobileSAM": "mobile_sam.pt",
            "SAM Base": "sam_b.pt",
            "SAM Large": "sam_l.pt",
            "SAM Huge": "sam_h.pt",
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

        # Set default to MobileSAM (fastest startup)
        models_list = list(self.models.keys())
        if "MobileSAM" in models_list:
            self.model_combo.setCurrentIndex(models_list.index("SAM 2.1 Tiny"))

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

        # Sample Label
        self.detect_as_combo = QComboBox()
        for label in self.label_window.labels:
            self.detect_as_combo.addItem(label.short_label_code, label.id)
        self.detect_as_combo.setCurrentIndex(0)
        self.detect_as_combo.currentIndexChanged.connect(self.update_class_mapping)
        layout.addRow("Detect as:", self.detect_as_combo)
        
        # Task dropdown
        self.use_task_dropdown = QComboBox()
        self.use_task_dropdown.addItems(["detect", "segment"])
        self.use_task_dropdown.currentIndexChanged.connect(self.update_task)
        layout.addRow("Task:", self.use_task_dropdown)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(640, 65536)
        self.imgsz_spinbox.setSingleStep(24)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_thresholds_layout(self):
        """
        Setup threshold control section using ThresholdsWidget.
        """
        # For SAM Generator: show all parameters including max_detections
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
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

    def update_detect_as_combo(self):
        """Update the label combo box with the current labels, preserving previous selection."""
        # Store the previously selected index
        previous_index = self.detect_as_combo.currentIndex() if hasattr(self, 'detect_as_combo') else 0

        self.detect_as_combo.clear()
        for label in self.label_window.labels:
            self.detect_as_combo.addItem(label.short_label_code, label.id)

        # Restore the previous selection if possible
        if 0 <= previous_index < self.detect_as_combo.count():
            self.detect_as_combo.setCurrentIndex(previous_index)
        else:
            self.detect_as_combo.setCurrentIndex(0)

    def update_class_mapping(self):
        """Update the class mapping based on the selected label."""
        detect_as = self.detect_as_combo.currentText()
        label = self.label_window.get_label_by_short_code(detect_as)
        self.class_mapping = {0: label}

    def update_task(self):
        """Update the task based on the dropdown selection."""
        self.task = self.use_task_dropdown.currentText()

    def load_model(self):
        """
        Load the selected SAM or FastSAM model with the current configuration.
        Dynamically instantiates SAM or FastSAM based on the model name.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        
        try:
            # Get selected model name and path
            selected_model_name = self.model_combo.currentText()
            self.model_path = self.models[selected_model_name]
            self.task = self.use_task_dropdown.currentText()
            
            # Determine which class to instantiate
            if "FastSAM" in selected_model_name:
                self.loaded_model = FastSAM(self.model_path)
                self.model_type = "fastSAM"
            else:
                self.loaded_model = SAM(self.model_path)
                self.model_type = "sam"
            
            # Warm-up run to initialize on GPU
            imgsz = self.get_imgsz()
            with torch.no_grad():
                blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                self.loaded_model(
                    blank,
                    conf=self.thresholds_widget.get_uncertainty_thresh(),
                    imgsz=imgsz,
                    device=self.main_window.device
                )
            
            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {self.model_path}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))
            self.loaded_model = None
            self.model_path = None
            self.model_type = None
        
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def get_imgsz(self):
        """Get the image size for the model."""
        self.imgsz = self.imgsz_spinbox.value()
        return self.imgsz

    def predict(self, image_paths=None, progress_bar=None):
        """
        Make predictions on the given image paths using the loaded model.
        Processes tiles in mini-batches for speed, but post-processes
        one-by-one to provide UI feedback.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
            progress_bar: Optional progress bar to use.
        """
        if not self.loaded_model:
            return
        
        if not image_paths:
            # Predict only the current image
            if self.annotation_window.current_image_path is None:
                QMessageBox.warning(self, "Warning", "No image is currently loaded for annotation.")
                return
            image_paths = [self.annotation_window.current_image_path]

        # --- Define a batch size for prediction ---
        BATCH_SIZE = 16 

        # Create a results processor (it's stateless, so creating it once is fine)
        results_processor = ResultsProcessor(
            self.main_window,
            self.class_mapping
        )

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Track if we created the progress bar ourselves
        progress_bar_created_here = progress_bar is None
        
        try:
            # --- We process one image at a time ---
            for idx, image_path in enumerate(image_paths):
                
                # --- 1. Get Raster and Work Items ---
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    print(f"Warning: Could not get raster for {image_path}. Skipping.")
                    continue
                
                # Get the list of items to process
                is_full_image = self.annotation_window.get_selected_tool() != "work_area"
                
                if is_full_image:
                    work_items_data = [raster.image_path]  # List with one string
                    work_areas = [None]  # Dummy list to make loops match
                else:
                    # Get both parallel lists: coordinate objects and data arrays
                    work_areas = raster.get_work_areas()  # List of WorkArea objects
                    work_items_data = raster.get_work_areas_data()  # List of np.ndarray

                if not work_items_data or not work_areas:
                    print(f"Warning: No work items found for {image_path}. Skipping.")
                    continue
                    
                # --- 2. Setup Progress Bar ---
                title = f"Predicting: {idx + 1}/{len(image_paths)} - {os.path.basename(image_path)}"
                if progress_bar is None:
                    progress_bar = ProgressBar(self.annotation_window)
                    progress_bar.show()
                progress_bar.set_title(title)
                progress_bar.start_progress(len(work_items_data))  # Total is still number of tiles

                # --- 3. Process Tiles and Collect Results ---
                
                # Create a list to hold all results for THIS image
                results_for_this_image = []
                is_segmentation = self.task == 'segment'
                
                try:
                    # --- Loop over the data in mini-batches ---
                    for i in range(0, len(work_items_data), BATCH_SIZE):
                        
                        # Get the mini-batch chunks
                        data_chunk = work_items_data[i: i + BATCH_SIZE]
                        area_chunk = work_areas[i: i + BATCH_SIZE]
                        
                        # --- 3a. Apply Model with OOM Adaptive Batching ---
                        # Try with current batch size; on OOM, halve and retry
                        current_batch_size = BATCH_SIZE
                        batch_results_list = None
                        temp_data_chunk = data_chunk
                        temp_area_chunk = area_chunk
                        
                        while current_batch_size > 0:
                            try:
                                batch_results_list = self._apply_model(temp_data_chunk)
                                break  # Success, exit retry loop
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    # Halve batch size and retry
                                    current_batch_size = max(1, current_batch_size // 2)
                                    gc.collect()
                                    empty_cache()
                                    print(f"OOM detected. Retrying with batch size {current_batch_size}.")
                                    
                                    # Re-slice to new size
                                    temp_data_chunk = data_chunk[:current_batch_size]
                                    temp_area_chunk = area_chunk[:current_batch_size]
                                    
                                    # If batch is now empty, skip
                                    if not temp_data_chunk:
                                        break
                                else:
                                    raise  # Re-raise non-OOM errors

                        if batch_results_list is None:
                            print(f"Warning: Failed to process batch starting at index {i}. Skipping.")
                            for _ in area_chunk:
                                progress_bar.update_progress()
                            continue

                        # Safety check
                        if len(batch_results_list) != len(temp_area_chunk):
                            print(f"Warning: Mismatch in batch results (Got {len(batch_results_list)}, "
                                  f"expected {len(temp_area_chunk)}). Skipping batch.")
                            
                            # Update progress bar for the skipped items
                            for _ in area_chunk:
                                progress_bar.update_progress()
                            continue
                            
                        # --- 3b. Post-process (Streaming w/ Highlight) ---
                        # Loop through the flat lists
                        for results_obj, work_area in zip(batch_results_list, temp_area_chunk):
                            
                            # --- Highlight at the START of post-processing ---
                            if work_area:
                                work_area.highlight()

                            if not results_obj:
                                if work_area: 
                                    work_area.unhighlight()
                                progress_bar.update_progress()
                                continue

                            # Get the single result object
                            results_obj.path = image_path
                            results_obj.names = {0: self.class_mapping[0].short_label_code}
                            
                            # Remap all detected class IDs to 0 (single-class unified generator)
                            # Modify the underlying data tensor (column 5 is class ID in Ultralytics)
                            if results_obj.boxes is not None and len(results_obj.boxes) > 0:
                                new_data = results_obj.boxes.data.clone()
                                new_data[:, 5] = 0
                                results_obj.boxes.data = new_data
                            
                            # --- 3c. Map Result (logic from _process_results) ---
                            if work_area:
                                # Highlight is already active
                                mapped_result = MapResults().map_results_from_work_area(
                                    results_obj,
                                    raster,
                                    work_area,
                                    map_masks=is_segmentation,
                                    task=self.task
                                )
                            else:
                                mapped_result = results_obj

                            # --- 3e. Append to list, DO NOT process yet ---
                            results_for_this_image.append(mapped_result)

                            # --- 3f. Update progress bar for this tile ---
                            progress_bar.update_progress()
                            
                            # --- 3g. Unhighlight at the END of post-processing ---
                            if work_area:
                                work_area.unhighlight()

                        # --- Clean up GPU memory *after* the mini-batch ---
                        gc.collect()
                        empty_cache()

                except Exception as e:
                    print(f"An error occurred during prediction on {image_path}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if progress_bar_created_here:
                        progress_bar.finish_progress()
                        progress_bar.stop_progress()
                        progress_bar.close()
                
                # --- 4. Process All Results for This Image at Once ---
                if results_for_this_image:
                    # This processing is now batched for the UI
                    if is_segmentation:
                        results_processor.process_segmentation_results(results_for_this_image)
                    else:
                        results_processor.process_detection_results(results_for_this_image)

        except Exception as e:
            print(f"A fatal error occurred during the prediction workflow: {e}")
        finally:
            # Only close the progress bar if we created it here
            if progress_bar_created_here and progress_bar is not None:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
            QApplication.restoreOverrideCursor()
            gc.collect()
            empty_cache()

    def _apply_model(self, inputs):
        """
        Apply the model to the inputs with task-aware parameters.
        Constructs kwargs dynamically based on task selection and model type.
        """
        # Base kwargs always passed
        kwargs = {
            'conf': self.thresholds_widget.get_uncertainty_thresh(),
            'imgsz': self.imgsz_spinbox.value(),
            'max_det': self.thresholds_widget.get_max_detections(),
            'device': self.main_window.device
        }
        
        # Task-specific kwargs
        if self.task == 'segment':
            kwargs['retina_masks'] = True  # High-quality, non-blocky polygons
        # For 'detect' task, omit retina_masks to maximize speed & minimize memory
        
        # Always pass iou; let Ultralytics ignore it if not applicable
        kwargs['iou'] = self.thresholds_widget.get_iou_thresh()
        
        # MobileSAM precision constraint: prevent FP16 crash by using FP32
        if "MobileSAM" in self.model_combo.currentText():
            kwargs['half'] = False  # FP32 mode for MobileSAM
        else:
            kwargs['half'] = True   # FP16 mode for others (faster)
        
        results_list = []
        import cv2
        
        for input_image in inputs:
            img = input_image
            try:
                # Read image if path string
                if isinstance(input_image, str):
                    img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"Warning: cv2 failed to read {input_image}")
                        results_list.append(None)
                        continue
                
                # Normalize channel dimensions: ensure HxWx3
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.ndim == 3 and img.shape[2] > 3:
                        img = img[:, :, :3]
                else:
                    # Unsupported input type
                    results_list.append(None)
                    continue
                
                # Call model with dynamically constructed kwargs
                with torch.no_grad():
                    results = self.loaded_model(img, **kwargs)
                    results_list.append(results[0] if results else None)
                    
            except Exception as e:
                print(f"Error running model on input: {e}")
                results_list.append(None)
        
        return results_list

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None
        # Clean up resources
        gc.collect()
        torch.cuda.empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
