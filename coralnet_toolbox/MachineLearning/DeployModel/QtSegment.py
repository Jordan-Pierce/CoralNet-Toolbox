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


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Segmentation Model (Ctrl + 3)")

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
        else:
            self.task = 'segment'

        # If use_sam_dropdown is "False", do nothing. Let self.task be whatever the user set.

    def load_model(self):
        """
        Load the segmentation model.
        """
        self.task = 'segment'

        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
            
        try:
            # Ensure task is correct after loading model
            self.update_sam_task_state()  

            # TODO: Improve batch size handling for different model types
            # Set BATCH_SIZE based on model type.
            # .engine models require a fixed batch size (usually 1)
            if self.model_path.endswith('.engine'):
                self.BATCH_SIZE = 1
            else:
                self.BATCH_SIZE = 16

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
            self.model_state_changed.emit()

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

    def predict(self, image_paths=None):
        """Run interactive inference on one or more images.

        Manages its own progress bar and bakes results immediately afterwards.
        For batch inference over many images or video frames, use
        BatchInferenceDialog which routes Detect/Segment tasks through the
        async BatchInferenceWorker.

        Args:
            image_paths: List of image paths to process.  If None, processes
                         the currently displayed image.
        """
        if not self.loaded_model:
            return

        if not image_paths:
            if self.annotation_window.current_image_path is None:
                QMessageBox.warning(self, "Warning", "No image is currently loaded for annotation.")
                return
            image_paths = [self.annotation_window.current_image_path]

        results_processor = ResultsProcessor(self.main_window, self.class_mapping)

        use_sam = (
            getattr(self, 'use_sam_dropdown', None) is not None
            and self.use_sam_dropdown.currentText() == "True"
            and getattr(self, 'sam_dialog', None) is not None
            and getattr(self.sam_dialog, 'loaded_model', None) is not None
        )
        is_segmentation = self.task == 'segment' or use_sam

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Running Inference")
        progress_bar.show()

        cache = {}  # image_path → [Results, …]

        try:
            for idx, image_path in enumerate(image_paths):
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    print(f"Segment.predict: no raster for {image_path}, skipping.")
                    continue

                use_tiles = (
                    raster.has_work_areas()
                    and self.annotation_window.get_selected_tool() == "work_area"
                )
                work_areas = raster.get_work_areas() if use_tiles else [None]

                progress_bar.set_title(
                    f"Image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}"
                )
                progress_bar.start_progress(len(work_areas))

                results_for_image = []

                for i in range(0, len(work_areas), self.BATCH_SIZE):
                    wa_chunk = work_areas[i:i + self.BATCH_SIZE]

                    inputs = []
                    for wa in wa_chunk:
                        if wa is not None:
                            wa.highlight()
                            inputs.append(raster.get_work_area_data(wa, as_format='BGR'))
                        else:
                            # Full-image: if this is a virtual video frame path,
                            # fetch the raw BGR frame and pass the ndarray directly
                            # to the model to avoid giving a non-filesystem path.
                            if isinstance(image_path, str) and '::frame_' in image_path:
                                try:
                                    from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                                    _, frame_idx = VideoRaster.parse_frame_path(image_path)
                                    if frame_idx is not None and hasattr(raster, 'get_bgr_frame'):
                                        bgr = raster.get_bgr_frame(int(frame_idx))
                                        if bgr is not None:
                                            inputs.append(bgr)
                                            continue
                                except Exception:
                                    # Any failure falls back to using the image path
                                    pass
                            inputs.append(image_path)

                    batch_results = self._apply_model(inputs)

                    # Remap tile coordinates + collect.  With SAM + work areas,
                    # SAM runs on the tile crop before the remap so the ViT
                    # encoder only ever sees the small tile — never the full raster.
                    for wa, result in zip(wa_chunk, batch_results):
                        result.path = image_path
                        if wa is not None:
                            if (use_sam and result.boxes is not None
                                    and len(result.boxes)):
                                try:
                                    sam_out = self.sam_dialog.predict_from_results(
                                        [result], image_path)
                                    if sam_out and sam_out[0] is not None:
                                        result = sam_out[0]
                                except Exception as e:
                                    print(f"Segment.predict: per-tile SAM failed: {e}")
                            result = MapResults().map_results_from_work_area(
                                result, raster, wa,
                                map_masks=True,
                                task='segment' if (use_sam or self.task == 'segment')
                                      else self.task,
                            )
                            try:
                                result.orig_img = None
                            except Exception:
                                pass
                            wa.unhighlight()
                        results_for_image.append(result)
                        progress_bar.update_progress()

                    gc.collect()
                    empty_cache()

                # Full-image SAM pass: only when work areas were NOT used.
                # With tiles, SAM already ran per-tile above.
                if use_sam and results_for_image and not use_tiles:
                    try:
                        import cv2
                        numpy_arr = raster.get_numpy()
                        if numpy_arr is not None:
                            full_bgr = cv2.cvtColor(numpy_arr, cv2.COLOR_RGB2BGR)
                            for r in results_for_image:
                                r.orig_img = full_bgr
                            del full_bgr, numpy_arr
                        results_for_image = self.sam_dialog.predict_from_results(
                            results_for_image, image_path)
                        for r in results_for_image:
                            try:
                                r.orig_img = None
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Segment.predict: SAM pass failed for {image_path}: {e}")
                    finally:
                        gc.collect()
                        empty_cache()

                if use_sam:
                    is_segmentation = True

                cache[image_path] = results_for_image

                if image_path == self.annotation_window.current_image_path:
                    try:
                        self._fast_render_image(
                            image_path, raster, results_for_image, results_processor)
                    except Exception as e:
                        print(f"Segment.predict: fast render failed: {e}")

        except Exception as e:
            print(f"Segment.predict: fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cache:
                self.annotation_window.is_streaming_inference = True
                progress_bar.set_title("Saving Annotations...")
                progress_bar.start_progress(len(cache))

                for path, results_list in cache.items():
                    if is_segmentation:
                        results_processor.process_segmentation_results(results_list)
                    else:
                        results_processor.process_detection_results(results_list)
                    progress_bar.update_progress()
                    QApplication.processEvents()

                self.annotation_window.is_streaming_inference = False

                try:
                    self.annotation_window.refresh_phantom_annotations()
                except Exception:
                    pass
                try:
                    self.main_window.label_window.update_annotation_count()
                    for path in cache:
                        self.image_window.update_image_annotations(path, update_counts=False)
                except Exception:
                    pass

            progress_bar.close()
            QApplication.restoreOverrideCursor()
            gc.collect()
            empty_cache()

    def _fast_render_image(self, image_path, raster, results_for_image, results_processor):
        """Push a ghost-render of new predictions to the OpenGL canvas without baking."""
        from coralnet_toolbox.utilities import rasterio_to_qimage
        aw = self.annotation_window

        try:
            q_img = rasterio_to_qimage(raster.rasterio_src)
        except Exception:
            q_img = None

        if getattr(aw, '_base_image_item', None) is not None:
            if q_img is not None:
                try:
                    aw.current_image_path = image_path
                    aw._base_image_item.set_image(q_img)
                except Exception:
                    pass

        fast_paths = []
        for res in results_for_image:
            try:
                fast_paths.extend(results_processor.generate_fast_render_paths(res, self.task))
            except Exception:
                pass
        try:
            for ann in aw.get_image_annotations(image_path):
                if getattr(ann.label, 'is_visible', True) and not hasattr(ann, 'mask_data'):
                    try:
                        fast_paths.append((ann.get_painter_path(), ann.label.color, ann.transparency))
                    except Exception:
                        pass
        except Exception:
            pass

        if getattr(aw, '_base_image_item', None) is not None:
            try:
                aw._base_image_item.set_readonly_annotations(fast_paths)
                QApplication.processEvents()
            except Exception:
                pass

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
                                              retina_masks=self.task == "segment",
                                              half=True,
                                              stream=True)  # memory efficient inference

        results_list = []
        for results in results_generator:
            # --- Append the object directly, not a list ---
            results_list.append(results) 

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

        # OPTIMIZATION: Pass the entire batch directly to SAM. 
        # The sam_dialog handles the list iteration natively.
        sam_result_list = self.sam_dialog.predict_from_results(results_list, image_path)
        
        # Ensure we always return a list of the exact same length, substituting None for failures
        updated_results = []
        for orig_res, sam_res in zip(results_list, sam_result_list):
            updated_results.append(sam_res if sam_res else None)
            
        return updated_results
