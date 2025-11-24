import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import os

import numpy as np

import torch
from torch.cuda import empty_cache
from ultralytics.models.fastsam import FastSAMPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QSpinBox,
                             QVBoxLayout, QGroupBox)

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployGeneratorDialog(QDialog):
    """
    Dialog for deploying FastSAM.
    Allows users to load, configure, and deactivate models, as well as make predictions on images.
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
        self.sam_dialog = None

        self.setWindowIcon(get_icon("wizard.png"))
        self.setWindowTitle("FastSAM Generator (Ctrl + 5)")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40

        self.task = 'detect'
        self.max_detect = 300
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the SAM layout
        self.setup_sam_layout()
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
        info_label = QLabel("Choose a Generator to deploy and use.")

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

        # Define available models
        self.models = {
            "FastSAM-s": "FastSAM-s.pt",
            "FastSAM-x": "FastSAM-x.pt"
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
        self.imgsz_spinbox.setRange(1024, 65536)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_sam_layout(self):
        """Use SAM model for segmentation."""
        group_box = QGroupBox("Use SAM to Create Polygons")
        layout = QFormLayout()

        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        layout.addRow("Use SAM Polygons:", self.use_sam_dropdown)

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
        """Update the task based on the dropdown selection and handle UI/model effects."""
        self.task = self.use_task_dropdown.currentText()

        # Update UI elements based on task
        if self.task == "segment":
            # Deactivate model if one is loaded and we're switching to segment task
            if self.loaded_model:
                self.deactivate_model()

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        return True

    def update_sam_task_state(self):
        """
        Centralized method to check if SAM is loaded and update task accordingly.
        If the user has selected to use SAM, this function ensures the task is set to 'segment'.
        Crucially, it does NOT alter the task if SAM is not selected, respecting the
        user's choice from the 'Task' dropdown.
        """
        # Check if the user wants to use the SAM model
        if self.use_sam_dropdown.currentText() == "True":
            # SAM is requested. Check if it's actually available.
            sam_is_available = (
                hasattr(self, 'sam_dialog') and
                self.sam_dialog is not None and
                self.sam_dialog.loaded_model is not None
            )

            if sam_is_available:
                # If SAM is wanted and available, the task must be segmentation.
                self.task = 'segment'
            else:
                # If SAM is wanted but not available, revert the dropdown and do nothing else.
                # The 'is_sam_model_deployed' function already handles showing an error message.
                self.use_sam_dropdown.setCurrentText("False")

        # If use_sam_dropdown is "False", do nothing. Let self.task be whatever the user set.

    def load_model(self):
        """
        Load the selected model with the current configuration.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        
        try:
            # Check if SAM is active and update task state
            self.update_sam_task_state()
            
            # Get selected model path
            self.model_path = self.models[self.model_combo.currentText()]
            self.task = self.use_task_dropdown.currentText()

            # Set the parameters
            overrides = dict(model=self.model_path,
                             task=self.task,
                             mode='predict',
                             save=False,
                             retina_masks=self.task == "segment",
                             max_det=self.thresholds_widget.get_max_detections(),
                             imgsz=self.get_imgsz(),
                             conf=self.thresholds_widget.get_uncertainty_thresh(),
                             iou=self.thresholds_widget.get_iou_thresh(),
                             device=self.main_window.device)

            # Load the model
            self.loaded_model = FastSAMPredictor(overrides=overrides)
            self.loaded_model.names = {0: self.class_mapping[0].short_label_code}

            with torch.no_grad():
                # Run a blank through the model to initialize it
                self.loaded_model(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {self.model_path}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
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
                is_segmentation = self.task == 'segment' or self.use_sam_dropdown.currentText() == "True"
                
                try:
                    # --- Loop over the data in mini-batches ---
                    for i in range(0, len(work_items_data), BATCH_SIZE):
                        
                        # Get the mini-batch chunks
                        data_chunk = work_items_data[i: i + BATCH_SIZE]
                        area_chunk = work_areas[i: i + BATCH_SIZE]
                        
                        # --- 3a. Apply Model (Batched) ---
                        # Returns a flat list: [res1, res2, ...]
                        batch_results_list = self._apply_model(data_chunk)
                        
                        # --- 3b. Apply SAM (Batched) ---
                        # Takes a flat list, returns a flat list: [sam_res1, sam_res2, ...]
                        sam_results_list = self._apply_sam(batch_results_list, image_path)

                        # Safety check
                        if len(sam_results_list) != len(area_chunk):
                            print(f"Warning: Mismatch in batch results (Got {len(sam_results_list)}, "
                                  f"expected {len(area_chunk)}). Skipping batch.")
                            
                            # Update progress bar for the skipped items
                            for _ in area_chunk:
                                progress_bar.update_progress()
                            continue
                            
                        # --- 3c. Post-process (Streaming w/ Highlight) ---
                        # Loop through the flat lists
                        for results_obj, work_area in zip(sam_results_list, area_chunk):
                            
                            # --- Highlight at the START of post-processing ---
                            if work_area:
                                work_area.highlight()

                            if not results_obj:  # Handle potential empty result from SAM
                                if work_area: 
                                    work_area.unhighlight()
                                progress_bar.update_progress()
                                continue

                            # Get the single result object
                            results_obj.path = image_path
                            results_obj.names = {0: self.class_mapping[0].short_label_code}
                            
                            # --- 3d. Map Result (logic from _process_results) ---
                            if work_area:
                                # Highlight is already active
                                mapped_result = MapResults().map_results_from_work_area(
                                    results_obj,
                                    raster,
                                    work_area,
                                    is_segmentation
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
        Apply the model to the inputs.
        """
        # Update the model with user parameters
        self.loaded_model.conf = self.thresholds_widget.get_uncertainty_thresh()
        self.loaded_model.iou = self.thresholds_widget.get_iou_thresh()
        self.loaded_model.max_det = self.thresholds_widget.get_max_detections()

        results_list = []
        for input_image in inputs:
            with torch.no_grad():
                results = self.loaded_model(input_image)
                results_list.append(results[0] if results else None)

        # Returns a flat list: [res1, res2, ...]
        return results_list

    def _apply_sam(self, results_list, image_path):
        """
        Apply SAM to the results if needed.
        Accepts a flat list of Results objects [res1, res2, ...]
        Returns a flat list of SAM-processed Results objects [sam_res1, sam_res2, ...]
        """
        # Check if SAM model is deployed and loaded
        self.update_sam_task_state()
        if self.task != 'segment':
            return results_list
        
        if not self.sam_dialog or self.use_sam_dropdown.currentText() == "False":
            # If SAM is not deployed or not selected, return the results as is
            return results_list

        if self.sam_dialog.loaded_model is None:
            # If SAM is not loaded, ensure we do not use it accidentally
            self.task = 'detect'
            self.use_sam_dropdown.setCurrentText("False")
            return results_list

        updated_results = []
        for results_obj in results_list:
            # 'results_obj' is a single Results object (e.g., res1)
            if results_obj:
                # --- Pass [results_obj] to SAM, as it expects a list ---
                sam_result_list = self.sam_dialog.predict_from_results([results_obj], image_path)
                
                # --- Unpack the list returned by SAM ---
                if sam_result_list:
                    updated_results.append(sam_result_list[0])
                else:
                    updated_results.append(None)  # Keep list length consistent
            else:
                updated_results.append(None)  # Keep list length consistent

        # Returns a flat list: [sam_res1, sam_res2, ...]
        return updated_results

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
