import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QLabel, QGroupBox, QFormLayout,
                             QComboBox, QSlider, QSpinBox)

from torch.cuda import empty_cache

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.MachineLearning.SMP import SemanticModel 

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results.MapResults import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Helper Function
# ----------------------------------------------------------------------------------------------------------------------


def _reconstruct_semantic_mask(results, model_class_names, project_class_mapping, mask_annotation_map):
    """
    Converts an Ultralytics Results object (instance format) back into a
    single semantic mask (H, W) with internal class IDs.
    """
    # If no masks or boxes, return an empty mask
    if results.masks is None or results.boxes is None:
        return np.zeros(results.orig_shape[:2], dtype=np.uint8)
        
    h, w = results.orig_shape[:2]
    semantic_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get masks and class indices
    masks = results.masks.data.cpu().numpy().astype(bool)  # (N, H, W)
    classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,)
    
    # Iterate over each detected instance (from highest conf to lowest)
    for i in range(len(classes)):
        # 1. Get the class name from the model's results (e.g., index 1)
        model_class_index = classes[i] 
        if model_class_index >= len(model_class_names):
            continue
        model_class_name = model_class_names[model_class_index]
        
        # Skip 'background' class by name
        if model_class_name.lower() == 'background':
            continue
        
        # 2. Find the corresponding Label object from the project (e.g., 'coral-a')
        label = project_class_mapping.get(model_class_name)
        if not label:
            continue  # Skip if this class isn't mapped in the project
        
        # 3. Find the internal ID for this label in the MaskAnnotation (e.g., 3)
        mask_class_id = mask_annotation_map.get(label['id'])
        if not mask_class_id:
            continue  # Skip if this label isn't in the mask's map
            
        # 4. Apply this class ID to the semantic mask
        instance_mask = masks[i]  # (H, W)
        semantic_mask[instance_mask] = mask_class_id
        
    return semantic_mask


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Semantic Segmentation Model (Ctrl + 4)")

        self.task = 'semantic'

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()

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

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_sam_layout(self):
        pass

    def load_model(self):
        """
        Load the semantic model using the SemanticModel class.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try: 
            # --- Use SemanticModel ---
            self.loaded_model = SemanticModel(self.model_path)

            try:
                # Get imgsz from the loaded model
                imgsz = self.loaded_model.imgsz if self.loaded_model.imgsz else 640
            except Exception:
                imgsz = 640

            # Warm up the model
            self.loaded_model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            
            # Get class names from the loaded model
            # We keep the original list from the model (including background)
            # to preserve the model's output index mapping.
            self.class_names = self.loaded_model.class_names
            self.class_names = [name for name in self.class_names if name.lower() != 'background']
            
            # We can still filter the mapping dictionary, as the new 
            # label methods will handle 'background' properly.
            self.class_mapping = {k: v for k, v in self.class_mapping.items() if k.lower() != 'background'}

            # These methods (now updated in QtBase.py) will
            # intelligently synchronize with the project's labels.
            if not self.class_mapping:
                self.handle_missing_class_mapping()
            else:
                self.add_labels_to_label_window()

            # Display the class names
            self.check_and_display_class_names()

            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model:
            return

        if not image_paths:
            # Predict only the current image
            if self.annotation_window.current_image_path is None:
                QMessageBox.warning(self, "Warning", "No image is currently loaded for annotation.")
                return
            image_paths = [self.annotation_window.current_image_path]

        # Get project labels for mask annotation creation
        project_labels = list(self.class_mapping.values())

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Prediction Workflow")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        try:
            for image_path in image_paths:
                # --- Get raster object as well ---
                inputs, raster = self._get_inputs(image_path)
                if inputs is None:
                    continue

                # Ensure the raster has a mask annotation
                if raster.mask_annotation is None:
                    raster.get_mask_annotation(project_labels)

                results = self._apply_model(inputs)
                # --- Pass raster to _process_results ---
                self._process_results(results, image_path, raster)

                # Update the progress bar
                progress_bar.update_progress()

        except Exception as e:
            print("An error occurred during prediction:", e)
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

        gc.collect()
        empty_cache()

    def _get_inputs(self, image_path):
        """Get the inputs for the model prediction."""
        raster = self.image_window.raster_manager.get_raster(image_path)
        if raster is None:
            return None, None
        
        if self.annotation_window.get_selected_tool() != "work_area":
            # Use the image path
            work_areas_data = [raster.image_path]
        else:
            # Get the work areas
            work_areas_data = raster.get_work_areas_data()

        # --- Return raster object ---
        return work_areas_data, raster

    def _apply_model(self, inputs):
        """Apply the SemanticModel to the inputs."""
        
        # Get prediction parameters
        confidence = self.main_window.get_uncertainty_thresh()
        
        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(inputs))
        
        # Run prediction on the batch of inputs
        # Our SemanticModel returns a list of Results objects
        results_list = self.loaded_model.predict(
            source=inputs,
            confidence_threshold=confidence,
            # Add other inference params here if needed
        )
        
        # Manually update the progress bar to 100%
        # (predict is blocking, so we can't update per-item)
        for _ in inputs:
            progress_bar.update_progress()

        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

        # Clean up GPU memory
        gc.collect()
        empty_cache()
        
        return results_list

    def _process_results(self, results_list, image_path, raster):
        """Process the results and update the mask annotation."""
        # (The 'raster' argument is new, passed from predict())
        
        # Get the raster object and number of work items
        total = raster.count_work_items()

        # Get the work areas (if any)
        work_areas = raster.get_work_areas()

        # --- Get the mask annotation layer ---
        mask_annotation = raster.mask_annotation
        if mask_annotation is None:
            print("Error: No mask annotation layer found to apply results to.")
            return

        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Processing Results")
        progress_bar.show()
        progress_bar.start_progress(total)

        updated_results = []
        map_results_util = MapResults()

        for idx, results in enumerate(results_list):
            # Each Results is a list (within the results_list, [[], ]
            if results:
                # Update path
                results[0].path = image_path
                # Check if the work area is valid, or the image path is being used
                if work_areas and self.annotation_window.get_selected_tool() == "work_area":
                    # Map results from work area to the full image
                    results_obj = map_results_util.map_results_from_work_area(results[0],
                                                                              raster,
                                                                              work_areas[idx],
                                                                              task=self.task)
                else:
                    results_obj = results[0]

                # Append the result object (not a list) to the updated results list
                updated_results.append(results_obj)

                # Update the index for the next work area
                idx += 1
                progress_bar.update_progress()

        # --- Process the Results ---
        
        # Get the mapping from Label UUID -> internal mask class ID
        mask_annotation_map = mask_annotation.label_id_to_class_id_map  # TODO if a label is deleted?
        
        # Process all results for this image.
        # If tiled, updated_results will have many items.
        # If full-image, updated_results will have one item.
        
        for idx, results in enumerate(updated_results):
            # Reconstruct the (H, W) semantic mask from the Results object
            reconstructed_mask = _reconstruct_semantic_mask(
                results,
                ['background'] + self.class_names,              # List of model class names ['Coral-A']
                self.class_mapping,                             # Project's map {'Coral-A': LabelObj}
                mask_annotation_map                             # Mask's map {LabelObj.id: 2}
            )
    
            # Update the main mask annotation with this tile's data
            # This method respects locked pixels
            mask_annotation.update_mask_with_prediction_mask(reconstructed_mask, top_left=(0, 0))
        
        # Close the progress bar
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()