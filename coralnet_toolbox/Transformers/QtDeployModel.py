import warnings


import os
import gc
import traceback

import torch
from torch.cuda import empty_cache
from autodistill.detection import CaptionOntology

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QVBoxLayout, QGroupBox)


from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import ConvertResults
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import rasterio_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(QDialog):
    """
    Dialog for deploying and managing Transformers models.
    Allows users to load, configure, and deactivate models, as well as make predictions on images.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the TransformersDeployModelDialog.

        Args:
            main_window: The main application window.
            parent: The parent widget, default is None.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Transformers Deploy Model (Ctrl + 7)")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        self.loaded_model = None
        self.model_name = None
        self.ontology = None
        self.class_mapping = {}
        self.ontology_pairs = []
        
        self.task = 'detect'

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the ontology layout
        self.setup_ontology_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the SAM layout
        self.setup_sam_layout()
        # Setup the thresholds layout
        self.setup_thresholds_layout()
        # Setup the button layout
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
        self.update_label_options()
        self.thresholds_widget.initialize_thresholds()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a model to deploy and use.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup model selection dropdown in a group box.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            # "OmDetTurbo-SwinT",
            "OWLViT",
            "GroundingDINO-SwinT",
            "GroundingDINO-SwinB",
        ])

        layout.addWidget(self.model_dropdown)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_ontology_layout(self):
        """
        Setup ontology mapping section in a group box with a single fixed pair.
        """
        group_box = QGroupBox("Ontology Mapping")
        layout = QVBoxLayout()

        # Create a single pair of text input and dropdown
        pair_layout = QHBoxLayout()

        self.text_input = QLineEdit()
        self.text_input.setMaxLength(100)  # Cap the width at 100 characters
        self.text_input.setPlaceholderText("Enter keyword or description")

        self.label_dropdown = QComboBox()
        self.label_dropdown.addItems([label.short_label_code for label in self.label_window.labels])

        pair_layout.addWidget(self.text_input)
        pair_layout.addWidget(self.label_dropdown)

        layout.addLayout(pair_layout)

        # Store the single pair for later reference
        self.ontology_pairs = [(self.text_input, self.label_dropdown)]

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        # Currently no parameters other than thresholds for transformers
        pass
    
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
        # For Transformers: show uncertainty, iou, and area (no max_detections)
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=False,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
        )
        self.layout.addWidget(self.thresholds_widget)

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

    def update_label_options(self):
        """
        Update the label options in the single ontology pair based on available labels.
        """
        label_options = [label.short_label_code for label in self.label_window.labels]
        previous_label = self.label_dropdown.currentText()
        self.label_dropdown.clear()
        self.label_dropdown.addItems(label_options)
        if previous_label in label_options:
            self.label_dropdown.setCurrentText(previous_label)

    def get_ontology_mapping(self):
        """
        Retrieve the ontology mapping from the single user input.

        Returns:
            Dictionary mapping text to label code.
        """
        ontology_mapping = {}
        if self.text_input.text() != "":
            ontology_mapping[self.text_input.text()] = self.label_dropdown.currentText()
        return ontology_mapping

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.
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
        # Get the ontology mapping
        ontology_mapping = self.get_ontology_mapping()

        if not ontology_mapping:
            QMessageBox.critical(self,
                                 "Error",
                                 "Please provide at least one ontology mapping.")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        
        try:
            # Update the state of the SAM and task
            self.update_sam_task_state()
            # Set the ontology
            self.ontology = CaptionOntology(ontology_mapping)
            # Set the class mapping
            self.class_mapping = {k: v for k, v in enumerate(self.ontology.classes())}

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()

            if model_name != self.model_name:
                self.load_new_model(model_name)
            else:
                # Update the model with the new ontology
                self.loaded_model.ontology = self.ontology

            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {model_name}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
        # Restore cursor
        QApplication.restoreOverrideCursor()

    def load_new_model(self, model_name):
        """
        Load a new model based on the selected model name.

        Args:
            model_name: Name of the model to load.
            uncertainty_thresh: Threshold for uncertainty.
        """
        
        # Clear the model
        self.loaded_model = None
        self.model_name = None
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        
        if "GroundingDINO" in model_name:
            from coralnet_toolbox.Transformers.Models.GroundingDINO import GroundingDINOModel

            model = model_name.split("-")[1].strip()
            self.model_name = model_name
            self.loaded_model = GroundingDINOModel(ontology=self.ontology,
                                                   model=model,
                                                   device=self.main_window.device)

        elif "OmDetTurbo" in model_name:
            from coralnet_toolbox.Transformers.Models.OmDetTurbo import OmDetTurboModel

            self.model_name = model_name
            self.loaded_model = OmDetTurboModel(ontology=self.ontology,
                                                device=self.main_window.device)

        elif "OWLViT" in model_name:
            from coralnet_toolbox.Transformers.Models.OWLViT import OWLViTModel

            self.model_name = model_name
            self.loaded_model = OWLViTModel(ontology=self.ontology,
                                            device=self.main_window.device)

    def predict(self, image_paths=None, progress_bar=None):
        """
        Make predictions on the given image paths using the loaded model.
        Processes tiles in mini-batches for speed, but post-processes
        one-by-one to provide UI feedback.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return
        
        if not image_paths:
            # Predict only the current image
            if self.annotation_window.current_image_path is None:
                QMessageBox.warning(self, "Warning", "No image is currently loaded for annotation.")
                return
            image_paths = [self.annotation_window.current_image_path]

        # --- Define a batch size for prediction ---
        # Transformers models are VRAM-heavy, so use a smaller batch size.
        BATCH_SIZE = 8

        # Create a results processor
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
                    work_items_data = [raster.get_numpy()]  # Get full numpy array
                    work_areas = [None]  # Dummy list to make loops match
                else:
                    # Get both parallel lists: coordinate objects and data arrays
                    work_areas = raster.get_work_areas()  # List of WorkArea objects
                    work_items_data = raster.get_work_areas_data()  # List of np.ndarray

                if not work_items_data or not work_areas:
                    print(f"Warning: No work items found for {image_path}. Skipping.")
                    continue
                    
                if len(work_items_data) != len(work_areas):
                    print(f"Error: Mismatch in work items. Data: {len(work_items_data)}, Areas: {len(work_areas)}")
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
                                # --- Force UI redraw ---
                                QApplication.processEvents()

                            if not results_obj:  # Handle potential empty result from SAM or Model
                                if work_area: 
                                    work_area.unhighlight()
                                progress_bar.update_progress()
                                continue

                            # Get the single result object
                            results_obj.path = image_path
                            
                            # --- 3d. Map Result ---
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
                    traceback.print_exc()
                
                # --- 4. Process All Results for This Image at Once ---
                if results_for_this_image:
                    # This function is now just a simple wrapper
                    self._process_results(results_processor, results_for_this_image, image_path)

        except Exception as e:
            print(f"A fatal error occurred during the prediction workflow: {e}")
            traceback.print_exc()
        finally:
            # --- 5. Final Cleanup ---
            if progress_bar_created_here and progress_bar is not None:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
                
            QApplication.restoreOverrideCursor()
            gc.collect()
            empty_cache()

    def _get_inputs(self, image_path):
        """Get the inputs for the model prediction."""
        raster = self.image_window.raster_manager.get_raster(image_path)
        if raster is None:
            return None  # Return None on failure
        
        if self.annotation_window.get_selected_tool() != "work_area":
            # Use the image path
            work_areas_data = [raster.get_numpy()]
        else:
            # Get the work areas
            work_areas_data = raster.get_work_areas_data()

        return work_areas_data

    def _apply_model(self, inputs):
        """
        Apply the model to the inputs (one-by-one).
        NOTE: This is NOT batched, as the QtBaseModel.predict
        function combines all results into one.
        """
        results_list = []
        
        # We must loop and predict one by one
        for input_image in inputs:
            try:
                # Run the model on the single input image
                # QtBaseModel.predict expects an image or list, so we pass [input_image]
                # It will return [combined_results] or []
                result = self.loaded_model.predict([input_image])
                
                # If no detections, result is [], so we append None
                if not result:
                    results_list.append(None)
                else:
                    # Append the single combined_results object
                    results_list.append(result[0])
                    
            except Exception as e:
                print(f"Error during single-item prediction: {e}")
                results_list.append(None) # Add None to keep list length consistent
        
        # Clean up GPU memory
        gc.collect()
        empty_cache()

        return results_list

    def _apply_sam(self, results_list, image_path):
        """
        Apply SAM to the results if needed.
        (Removes internal progress bar and cursor logic)
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
        for results_obj in results_list:  # results_list is now flat [res1, res2]
            if results_obj:
                # SAM's predict_from_results expects a list
                sam_result = self.sam_dialog.predict_from_results([results_obj], image_path)
                if sam_result:
                    updated_results.append(sam_result[0])
                else:
                    updated_results.append(None)  # Keep list length consistent
            else:
                updated_results.append(None)   # Pass through Nones

        return updated_results

    def _process_results(self, results_processor, results_list, image_path):
        """
        Process the results using the result processor.
        (Simplified: This now just prepares the list for the processor)
        """
        
        # This function no longer needs:
        # - Raster object, total, work_areas
        # - Progress bar
        # - MapResults, highlight/unhighlight (done in predict)

        updated_results = []
        
        # results_list is already a flat list of mapped results
        for results_obj in results_list:
            if results_obj:
                # Update path and names
                results_obj.path = image_path
                updated_results.append(results_obj)

        # Process the Results
        if self.use_sam_dropdown.currentText() == "True":
            results_processor.process_segmentation_results(updated_results)
        else:
            results_processor.process_detection_results(updated_results)

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        # Clear the model
        self.loaded_model = None
        self.model_name = None
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
