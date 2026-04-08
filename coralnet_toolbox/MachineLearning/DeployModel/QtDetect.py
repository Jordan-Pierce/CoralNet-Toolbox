import warnings

import gc
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QGroupBox, QFormLayout, QComboBox)

from torch.cuda import empty_cache
from ultralytics import YOLO

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results.MapResults import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Detection Model (Ctrl + 2)")

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync thresholds.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        # Initialize thresholds in the widget
        self.thresholds_widget.initialize_thresholds()

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        # No additional parameters needed - max_detections is in thresholds widget
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
        Setup threshold control section using the reusable ThresholdsWidget.
        """
        # Create the thresholds widget with all controls enabled
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
        )
        self.layout.addWidget(self.thresholds_widget)

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
            sam_is_available = False
            try:
                sam_is_available = hasattr(self, 'sam_dialog') and self.sam_dialog is not None
                if sam_is_available:
                    sam_is_available = getattr(self.sam_dialog, 'loaded_model', None) is not None
            except Exception:
                sam_is_available = False

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
        Load the detection model.
        """
        self.task = 'detect'
                
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # TODO: Improve batch size handling for different model types
            # Set BATCH_SIZE based on model type.
            # .engine models require a fixed batch size (usually 1)
            if self.model_path.endswith('.engine'):
                self.BATCH_SIZE = 1
            else:
                self.BATCH_SIZE = 16
                
            # Ensure task is correct after loading model
            self.update_sam_task_state()  
            # Load the model (8.3.141) YOLO handles RTDETR too
            self.loaded_model = YOLO(self.model_path, task=self.task)

            try:
                imgsz = self.loaded_model.__dict__['overrides']['imgsz']
            except Exception:
                imgsz = 640

            self.loaded_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            self.class_names = list(self.loaded_model.names.values())

            # Check for unmapped classes
            mapped_classes, unmapped_classes, unused_mapping_keys = self._find_unmapped_classes()

            # Handle class mapping (complete or partial)
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
        Processes tiles in mini-batches for speed, but post-processes
        one-by-one to provide UI feedback.

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

                # Handle video virtual-frame paths specially (video.mp4::frame_N)
                work_items_data = None
                work_areas = None
                if isinstance(image_path, str) and '::frame_' in image_path:
                    # Import here to avoid circular imports
                    from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                    video_path, frame_idx = VideoRaster.parse_frame_path(image_path)

                    # RasterManager.get_raster resolves virtual paths to the underlying video raster
                    # Ensure the raster is a VideoRaster before using frame APIs
                    if hasattr(raster, 'raster_type') and raster.raster_type == 'VideoRaster':
                        if is_full_image:
                            # For full-image detect on a video frame, pass the raw BGR numpy array
                            bgr = raster.get_bgr_frame(frame_idx)
                            if bgr is None:
                                print(f"Warning: Could not read frame {frame_idx} from {video_path}. Skipping.")
                                continue
                            work_items_data = [bgr]
                            work_areas = [None]
                        else:
                            # For work-area mode, update the shim to point at this frame
                            raster.update_shim_for_frame(frame_idx)
                            work_areas = raster.get_work_areas()
                            work_items_data = raster.get_work_areas_data(as_format='BGR')
                    else:
                        # Fallback to normal handling if raster isn't a VideoRaster
                        if is_full_image:
                            work_items_data = [raster.image_path]
                            work_areas = [None]
                        else:
                            work_areas = raster.get_work_areas()
                            work_items_data = raster.get_work_areas_data(as_format='BGR')
                else:
                    # Regular (non-video) path handling
                    if is_full_image:
                        work_items_data = [raster.image_path]  # List with one string
                        work_areas = [None]  # Dummy list to make loops match
                    else:
                        # Get both parallel lists: coordinate objects and data arrays
                        work_areas = raster.get_work_areas()  # List of WorkArea objects
                        work_items_data = raster.get_work_areas_data(as_format='BGR')  # List of np.ndarray

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
                    for i in range(0, len(work_items_data), self.BATCH_SIZE):
                        
                        # Get the mini-batch chunks
                        data_chunk = work_items_data[i: i + self.BATCH_SIZE]
                        area_chunk = work_areas[i: i + self.BATCH_SIZE]
                        
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
                    # ---> FAST PATH: Render-First, Bake-Later <---

                    # 4a. Fast Render: Send paths directly to the OpenGL canvas
                    try:
                        fast_paths = []
                        for res in results_for_this_image:
                            paths = results_processor.generate_fast_render_paths(res, self.task)
                            if paths:
                                fast_paths.extend(paths)

                        if getattr(self.annotation_window, '_base_image_item', None) is not None:
                            try:
                                self.annotation_window._base_image_item.set_readonly_annotations(fast_paths)
                                # Force Qt to paint the fast paths right now so the user sees progress
                                QApplication.processEvents()
                            except Exception:
                                # Non-fatal: if fast render fails, fall back to normal baking later
                                pass
                    except Exception as e:
                        print(f"Warning: Fast render failed in Detect.predict: {e}")

                    # 4b. Cache the raw results for later hydration
                    if not hasattr(self.annotation_window, 'batch_results_cache'):
                        self.annotation_window.batch_results_cache = {}

                    # Store a flattened list of all results for this specific image path
                    self.annotation_window.batch_results_cache[image_path] = results_for_this_image

        except Exception as e:
            print(f"A fatal error occurred during the prediction workflow: {e}")
        finally:
            # --- 5. BAKE (HYDRATE) ALL IMAGES AT THE END ---
            # If this predict call wasn't triggered by an external batch dialog,
            # we need to bake the cached results right now.
            if not progress_bar_created_here:
                # We are part of a larger BatchInference process. Let QtBatchInference
                # handle the final baking so we don't freeze between every image.
                pass
            else:
                # We are running a single image (or manual selection). Bake now!
                cache = getattr(self.annotation_window, 'batch_results_cache', {})
                if cache:
                    self.annotation_window.is_streaming_inference = True  # Suppress UI loops

                    bake_pb = ProgressBar(self.annotation_window, title="Saving Annotations...")
                    bake_pb.show()
                    bake_pb.start_progress(len(cache))

                    is_segmentation = self.task == 'segment' or self.use_sam_dropdown.currentText() == "True"

                    for path, results_list in cache.items():
                        if is_segmentation:
                            results_processor.process_segmentation_results(results_list)
                        else:
                            results_processor.process_detection_results(results_list)
                        bake_pb.update_progress()
                        QApplication.processEvents()

                    bake_pb.close()
                    self.annotation_window.is_streaming_inference = False

                    # One final UI update
                    try:
                        self.main_window.label_window.update_annotation_count()
                        for path in cache.keys():
                            self.image_window.update_image_annotations(path, update_counts=False)
                    except Exception:
                        pass

                    self.annotation_window.batch_results_cache = {}

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
        results_generator = self.loaded_model(inputs,
                                              agnostic_nms=True,
                                              conf=self.thresholds_widget.get_uncertainty_thresh(),
                                              iou=self.thresholds_widget.get_iou_thresh(),
                                              max_det=self.thresholds_widget.get_max_detections(),
                                              device=self.main_window.device,
                                              retina_masks=False,
                                              half=True,
                                              stream=True)  # memory efficient inference

        # Stream generator converted directly to list for the batch
        return list(results_generator)

    def _apply_sam(self, results_list, image_path):
        """
        Apply SAM to the results if needed.
        Accepts a flat list of Results objects [res1, res2, ...]
        Returns a flat list of SAM-processed Results objects [sam_res1, sam_res2, ...]
        """
        self.update_sam_task_state()
        if self.task != 'segment':
            return results_list
        
        if not self.sam_dialog or self.use_sam_dropdown.currentText() == "False":
            return results_list

        if self.sam_dialog.loaded_model is None:
            self.task = 'detect'
            self.use_sam_dropdown.setCurrentText("False")
            return results_list

        # OPTIMIZATION: Pass the entire batch directly to SAM. 
        # The sam_dialog handles the list iteration natively.
        sam_result_list = self.sam_dialog.predict_from_results(results_list, image_path)
        
        # Ensure we always return a list of the exact same length, substituting None for failures
        updated_results = []
        for orig_res, sam_res in zip(results_list, sam_result_list):
            updated_results.append(sam_res if sam_res else None)
            
        return updated_results
