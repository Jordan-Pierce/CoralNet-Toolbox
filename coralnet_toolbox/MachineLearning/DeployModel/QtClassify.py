import warnings

import gc
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from torch.cuda import empty_cache
from ultralytics import YOLO

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.Results import ResultsProcessor

from coralnet_toolbox.Common import ThresholdsWidget

from rasterio.windows import Window as _RasterioWindow

from coralnet_toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Classification Model (Ctrl + 1)")

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

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
        # Currently no parameters other than thresholds for classification
        pass
    
    def setup_sam_layout(self):
        pass

    def setup_thresholds_layout(self):
        """
        Setup threshold control section using the reusable ThresholdsWidget.
        """
        # Create the thresholds widget with only uncertainty threshold enabled
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=False,
            show_boundary=False,
            show_uncertainty=True,
            show_iou=False,
            show_area=False
        )
        self.layout.addWidget(self.thresholds_widget)

    def load_model(self):
        """
        Load the classification model.
        """
        self.task = 'classify'
        
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # TODO: Improve batch size handling for different model types
            # Set BATCH_SIZE based on model type.
            # .engine models require a fixed batch size (usually 1)
            if self.model_path.endswith('.engine'):
                self.BATCH_SIZE = 1
            else:
                self.BATCH_SIZE = 0

            # Load the model (8.3.141) YOLO handles RTDETR too
            self.loaded_model = YOLO(self.model_path, task=self.task)

            try:
                imgsz = self.loaded_model.__dict__['overrides']['imgsz']
            except Exception:
                imgsz = 256

            self.imgsz = imgsz
            # Pass the same device/quantize the predict paths use: ultralytics
            # fixes fp16 when the predictor is created and silently rebuilds
            # the whole predictor on the first call whose device= differs, so
            # the warmup must match or later quantize=True calls run in fp32.
            self.loaded_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                              device=self.main_window.device, quantize=True)
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

    def predict(self, inputs=None, progress_bar=None):
        """
        Predict the classification results for the given inputs.
        
        Args:
            inputs: List of annotations to predict on. If None, uses selected or all review annotations.
            progress_bar: Optional progress bar instance to use. If None, no progress bar is shown.
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the selected annotation
            inputs = self.annotation_window.selected_annotations.copy()
            # Unselect the annotations (regardless)
            self.annotation_window.unselect_annotations()

        if not inputs:
            # If no annotations are selected, predict all annotations in the image
            inputs = self.annotation_window.get_image_review_annotations()

        if not inputs:
            # If no annotations are available, return
            QApplication.restoreOverrideCursor()
            return

        # Create lists to store valid images and their corresponding annotations
        images_np = []
        valid_inputs = []

        # ------------------------------------------------------------------
        # Fast path: read rasterio → numpy directly, bypassing the
        # QPixmap round-trip (rasterio→QImage→QPixmap→QImage→bytes→numpy).
        # We group annotations by image_path so each rasterio source is
        # opened only once, then build one numpy array per patch.
        # ------------------------------------------------------------------
        from collections import defaultdict
        groups = defaultdict(list)
        for ann in inputs:
            groups[ann.image_path].append(ann)

        for image_path, anns in groups.items():
            try:
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if not (raster and raster.rasterio_src):
                    # Fallback: use cached QPixmap if rasterio not available
                    for ann in anns:
                        if ann.cropped_image:
                            try:
                                images_np.append(pixmap_to_numpy(ann.cropped_image))
                                valid_inputs.append(ann)
                            except Exception as e:
                                print(f"Error converting pixmap to numpy for {ann.id}: {e}")
                    continue

                src = raster.rasterio_src
                n_bands = src.count

                for ann in anns:
                    try:
                        half = ann.annotation_size / 2
                        px = int(ann.center_xy.x())
                        py = int(ann.center_xy.y())
                        col_off = max(0, px - half)
                        row_off = max(0, py - half)
                        width  = min(src.width  - col_off, ann.annotation_size)
                        height = min(src.height - row_off, ann.annotation_size)
                        window = _RasterioWindow(col_off=col_off, row_off=row_off,
                                                 width=width, height=height)

                        if n_bands >= 3:
                            arr = src.read([1, 2, 3], window=window)   # (3, H, W)
                            arr = np.transpose(arr, (1, 2, 0))          # (H, W, 3)
                        else:
                            band = src.read(1, window=window)           # (H, W)
                            arr = np.stack([band, band, band], axis=-1) # (H, W, 3)

                        # Normalise to uint8
                        if arr.dtype != np.uint8:
                            max_val = arr.max()
                            if max_val > 0:
                                arr = (arr.astype(np.float32) * (255.0 / max_val)).astype(np.uint8)
                            else:
                                arr = arr.astype(np.uint8)

                        arr = np.ascontiguousarray(arr)
                        images_np.append(arr)
                        valid_inputs.append(ann)

                        # Keep the QPixmap cache warm for the visible image so
                        # the confidence window / thumbnail still works.
                        if not ann.cropped_image:
                            ann.create_cropped_image(src)

                    except Exception as e:
                        print(f"Error reading patch for annotation {ann.id}: {e}")
                        # Fallback to QPixmap path for this one annotation
                        if ann.cropped_image:
                            try:
                                images_np.append(pixmap_to_numpy(ann.cropped_image))
                                valid_inputs.append(ann)
                            except Exception as e2:
                                print(f"Error in pixmap fallback for {ann.id}: {e2}")

            except Exception as e:
                print(f"Error processing image group {image_path}: {e}")
                # Fallback: QPixmap path for whole group
                for ann in anns:
                    if ann.cropped_image:
                        try:
                            images_np.append(pixmap_to_numpy(ann.cropped_image))
                            valid_inputs.append(ann)
                        except Exception as e2:
                            print(f"Error in pixmap fallback for {ann.id}: {e2}")

        # Only proceed if we have valid images to process
        if images_np:
            # stream=True yields results lazily one-by-one, avoiding the deepcopy
            # overhead of the old non-streaming engine path.  YOLO honours the
            # model's compiled batch size internally, so this path is safe for
            # both normal (.pt) and fixed-batch TensorRT (.engine) models.
            results = self.loaded_model(
                images_np,
                conf=self.thresholds_widget.get_uncertainty_thresh(),
                device=self.main_window.device,
                quantize=True,
                stream=True,
                verbose=False,
            )

            # Create a result processor
            results_processor = ResultsProcessor(self.main_window,
                                                 self.class_mapping)

            # Process the classification results using the valid inputs
            # Pass the progress_bar parameter to avoid creating nested progress bars
            results_processor.process_classification_results(results, valid_inputs, progress_bar=progress_bar)

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()
