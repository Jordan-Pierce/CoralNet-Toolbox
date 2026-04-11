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

    def predict(self, image_paths=None):
        """Run inference on one or more images with the loaded SAM/FastSAM model.

        Manages its own progress bar and always bakes results at the end.
        OOM-adaptive batching: on GPU out-of-memory errors the batch size is
        halved and the failing chunk is retried automatically.

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

        BATCH_SIZE = 32

        results_processor = ResultsProcessor(self.main_window, self.class_mapping)
        is_segmentation = self.task == 'segment'

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Running Inference")
        progress_bar.show()

        cache = {}  # image_path → [Results, …]

        try:
            for idx, image_path in enumerate(image_paths):
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    print(f"SAM.predict: no raster for {image_path}, skipping.")
                    continue

                use_tiles = (
                    raster.has_work_areas()
                    and self.annotation_window.get_selected_tool() == "work_area"
                )
                if use_tiles:
                    work_areas = raster.get_work_areas()
                    work_items_data = raster.get_work_areas_data()  # RGB
                else:
                    work_areas = [None]
                    # If the requested image_path is a virtual video frame, try
                    # to fetch the raw BGR frame and pass it directly to the model.
                    # Any failure falls back to using the raster.image_path.
                    if isinstance(image_path, str) and '::frame_' in image_path:
                        try:
                            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                            _, frame_idx = VideoRaster.parse_frame_path(image_path)
                            if frame_idx is not None and hasattr(raster, 'get_bgr_frame'):
                                bgr = raster.get_bgr_frame(int(frame_idx))
                                if bgr is not None:
                                    work_items_data = [bgr]
                                else:
                                    work_items_data = [raster.image_path]
                            else:
                                work_items_data = [raster.image_path]
                        except Exception:
                            work_items_data = [raster.image_path]
                    else:
                        work_items_data = [raster.image_path]

                if not work_items_data:
                    print(f"SAM.predict: no work items for {image_path}, skipping.")
                    continue

                progress_bar.set_title(
                    f"Image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}"
                )
                progress_bar.start_progress(len(work_items_data))

                results_for_image = []
                current_batch_size = BATCH_SIZE

                for i in range(0, len(work_items_data), current_batch_size):
                    data_chunk = work_items_data[i:i + current_batch_size]
                    area_chunk = work_areas[i:i + current_batch_size]

                    # Highlight all tiles in this batch before inference so the
                    # user can see which regions are queued for processing.
                    for wa in area_chunk:
                        if wa is not None:
                            wa.highlight()

                    # OOM-adaptive: halve batch and retry on out-of-memory
                    batch_results = None
                    tmp_data = data_chunk
                    tmp_areas = area_chunk
                    tmp_bs = current_batch_size

                    while tmp_bs > 0:
                        try:
                            batch_results = self._apply_model(tmp_data)
                            break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                tmp_bs = max(1, tmp_bs // 2)
                                import gc as _gc
                                _gc.collect()
                                empty_cache()
                                print(f"OOM: retrying with batch size {tmp_bs}.")
                                tmp_data  = data_chunk[:tmp_bs]
                                tmp_areas = area_chunk[:tmp_bs]
                                if not tmp_data:
                                    break
                            else:
                                raise

                    if batch_results is None:
                        for wa in area_chunk:
                            if wa is not None:
                                wa.unhighlight()
                            progress_bar.update_progress()
                        continue

                    for wa, result in zip(tmp_areas, batch_results):
                        if not result:
                            if wa is not None:
                                wa.unhighlight()
                            progress_bar.update_progress()
                            continue

                        result.path = image_path
                        result.names = {0: self.class_mapping[0].short_label_code}

                        # Collapse all class IDs to 0 (single-class generator)
                        if result.boxes is not None and len(result.boxes) > 0:
                            new_data = result.boxes.data.clone()
                            new_data[:, 5] = 0
                            result.boxes.data = new_data

                        if wa is not None:
                            result = MapResults().map_results_from_work_area(
                                result, raster, wa,
                                map_masks=is_segmentation,
                                task=self.task,
                            )
                            wa.unhighlight()

                        results_for_image.append(result)
                        progress_bar.update_progress()

                    import gc as _gc
                    _gc.collect()
                    empty_cache()

                cache[image_path] = results_for_image

                if results_for_image and image_path == self.annotation_window.current_image_path:
                    try:
                        self._fast_render_image(
                            image_path, raster, results_for_image, results_processor)
                    except Exception as e:
                        print(f"SAM.predict: fast render failed: {e}")

        except Exception as e:
            print(f"SAM.predict: fatal error: {e}")
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
            import gc as _gc
            _gc.collect()
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
                    aw.fit_to_image()
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
