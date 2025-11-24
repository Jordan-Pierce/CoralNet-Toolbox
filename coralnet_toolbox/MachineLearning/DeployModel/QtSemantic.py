import warnings

import gc
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox)

from torch.cuda import empty_cache

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.MachineLearning.SMP import SemanticModel 

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Function
# ----------------------------------------------------------------------------------------------------------------------


def _reconstruct_semantic_mask(results, model_class_names, label_window, mask_annotation_map):
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
        
        label_obj = label_window.get_label_by_short_code(model_class_name, return_review=False)
        if not label_obj:
            # This label isn't in the project (e.g., it was deleted). Skip it.
            continue
        
        mask_class_id = mask_annotation_map.get(label_obj.id)
        if not mask_class_id:
            # This should not happen if sync_label_map was called correctly,
            # but it's a good safeguard.
            continue
            
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
        self.thresholds_widget.initialize_thresholds()

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        # Currently no parameters other than thresholds for semantic segmentation
        pass
    
    def setup_sam_layout(self):
        pass

    def setup_thresholds_layout(self):
        """
        Setup threshold control section using ThresholdsWidget.
        """
        # For semantic segmentation: only show uncertainty threshold
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=False,
            show_uncertainty=True,
            show_iou=False,
            show_area=False
        )
        self.layout.addWidget(self.thresholds_widget)

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

            # Check for unmapped classes
            mapped_classes, unmapped_classes, unused_mapping_keys = self._find_unmapped_classes()

            # These methods (now updated in QtBase.py) will
            # intelligently synchronize with the project's labels.
            if not self.class_mapping:
                # No mapping file at all
                self.handle_missing_class_mapping()
            elif unmapped_classes:
                # Partial mapping - some classes are missing
                self.add_labels_to_label_window()
                self.handle_missing_class_mapping(unmapped_classes)
            else:
                # Complete mapping - all classes are mapped
                self.add_labels_to_label_window()

            # Display the class names
            self.check_and_display_class_names()

            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except RuntimeError:
            # Model load was cancelled by user
            self.loaded_model = None
            self.class_names = []
            self.auto_created_labels = set()
            QApplication.restoreOverrideCursor()
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def predict(self, image_paths=None, progress_bar=None):
        """
        Make predictions on the given image paths using the loaded model.
        Processes tiles one-by-one to conserve memory and update the UI in real-time.

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
        project_labels = self.main_window.label_window.labels
        # Get full list of model class names for _reconstruct_semantic_mask
        model_class_names = ['background'] + self.class_names 

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

                # Ensure the raster has a mask annotation
                if raster.mask_annotation is None:
                    raster.get_mask_annotation(project_labels)
                    
                mask_annotation = raster.mask_annotation
                # This call ensures the mask is synced with the live labels
                mask_annotation.sync_label_map(project_labels)
                # Get the mapping from Label UUID -> internal mask class ID (once)
                mask_annotation_map = mask_annotation.label_id_to_class_id_map
                
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
                    
                if len(work_items_data) != len(work_areas):
                    print(f"Error: Mismatch in work items. Data: {len(work_items_data)}, Areas: {len(work_areas)}")
                    continue

                # --- 2. Setup Progress Bar (New Style) ---
                title = f"Predicting: {idx + 1}/{len(image_paths)} - {os.path.basename(image_path)}"
                if progress_bar is None:
                    progress_bar = ProgressBar(self.annotation_window)
                    progress_bar.show()
                progress_bar.set_title(title)
                progress_bar.start_progress(len(work_items_data))  # Total is number of tiles
                
                # --- 3. Process One Item at a Time (Streaming) ---
                try:
                    # Loop by index to keep parallel lists in sync
                    for idx_tile in range(len(work_items_data)):
                        
                        # Get the data and the corresponding coordinate object
                        input_data = [work_items_data[idx_tile]]
                        item = work_areas[idx_tile]  # This is the WorkArea object or None
                        
                        # --- 3a. Get Input Data and Offset ---
                        if is_full_image:
                            # Full image path, offset is 0
                            offset = (0, 0)
                        else:
                            # WorkArea object
                            item.highlight()  # Highlight current tile
                            # Get (x, y) coords from the rect
                            offset = (int(item.rect.x()), int(item.rect.y()))
                            QApplication.processEvents()

                        # --- 3b. Apply Model ---
                        results_list = self._apply_model(input_data)
                        
                        if not results_list or not results_list[0]:
                            if not is_full_image: 
                                item.unhighlight()
                            progress_bar.update_progress()
                            continue
                            
                        # Get the single Results object. 
                        results_obj = results_list[0][0] 
                        results_obj.path = image_path  # Fix path

                        # --- 3c. Reconstruct Small Mask ---
                        reconstructed_mask = _reconstruct_semantic_mask(
                            results_obj,
                            model_class_names,                            # List of model class names
                            self.main_window.label_window,                # Live project labels
                            mask_annotation_map                           # Mask's map {LabelObj.id: 2}
                        )
                        
                        # --- 3d. Update Main Annotation (Streaming) ---
                        # This updates the UI *immediately* for each tile
                        mask_annotation.update_mask_with_mask(
                            reconstructed_mask, 
                            top_left=offset
                        )
                        
                        if not is_full_image: 
                            item.unhighlight()
                        progress_bar.update_progress()
                        
                        # Break if this was a full image
                        if is_full_image:
                            break

                except Exception as e:
                    print(f"An error occurred during prediction on {image_path}: {e}")
                    # Let the outer finally block handle cleanup
                
                # --- 4. Recalculate Stats ---
                # This is called *after* all tiles for an image are done
                mask_annotation.recalculate_class_statistics()

        except Exception as e:
            print(f"A fatal error occurred during the prediction workflow: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # --- 5. Final Cleanup ---
            # This block now runs ONCE at the end of the entire function
            if progress_bar_created_here and progress_bar is not None:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
                
            QApplication.restoreOverrideCursor()
            gc.collect()
            empty_cache()

    def _apply_model(self, inputs):
        """
        Apply the SemanticModel to the inputs.
        (This method no longer shows its own progress bar)
        """
        
        # Get prediction parameters
        confidence = self.thresholds_widget.get_uncertainty_thresh()
        
        # Run prediction on the batch of inputs
        # Our SemanticModel returns a list of Results objects
        results_list = self.loaded_model.predict(
            source=inputs,
            confidence_threshold=confidence,
            # Add other inference params here if needed
        )
        
        # Clean up GPU memory
        gc.collect()
        empty_cache()
        
        return results_list