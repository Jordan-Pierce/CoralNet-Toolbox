import time
import warnings

import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox,
                             QFormLayout, QComboBox, QHBoxLayout, QCheckBox, QButtonGroup,
                             QSpinBox, QPushButton)

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox.Common import ThresholdsWidget
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.MachineLearning.BatchInference.Contracts import (
    InferenceItem,
    InferenceResult,
    InferenceThresholds,
    SemanticOverlayRecord,
)
from coralnet_toolbox.MachineLearning.BatchInference.ItemBuilder import build_inference_items
from coralnet_toolbox.MachineLearning.BatchInference.Task import make_batch_inference_task
from coralnet_toolbox.MachineLearning.BatchInference.Timing import (
    BatchInferenceTiming,
    BatchTimingRecord,
)
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------------------
# Background worker for batch inference (video-focused)
# ------------------------------------------------------------------------------

class BatchInferenceWorker(QThread):
    """
    Background worker for running machine learning inference without freezing the Qt GUI.
    Optimized for streaming Video frames.
    """
    # Signals
    itemProcessed = pyqtSignal(object)      # emits InferenceResult
    progressUpdated = pyqtSignal(int, int)  # current, total
    timingSummary = pyqtSignal(object)      # emits aggregate timing dict
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, items, initial_thresholds,
                 device=None, task='detect', batch_size=16, is_semantic=False,
                 sam_enabled=False, parent=None):
        super().__init__(parent)
        self.model = model
        self.items = list(items)
        self.device = device
        self._task = task
        self._batch_size = max(1, int(batch_size))
        self._is_semantic = is_semantic
        # When SAM is enabled, skip tile→full-image remap in the worker so
        # _apply_sam_to_cache can run SAM on the tile crop (tile-space boxes
        # + tile orig_img) and then remap boxes and masks together afterwards.
        self._sam_enabled = bool(sam_enabled)
        self._is_running = True
        self._waiting_for_ui = False
        self._mutex = QMutex()
        self._thresholds = InferenceThresholds.from_mapping(initial_thresholds)
        self._timing = BatchInferenceTiming()

    def stop(self):
        """Safely signal the worker loop to exit."""
        locker = QMutexLocker(self._mutex)
        try:
            self._is_running = False
        finally:
            del locker

    def release_frame_gate(self):
        """Called by the main thread after painting a video frame."""
        self._waiting_for_ui = False

    def update_thresholds(self, conf=None, iou=None, max_det=None, boundary_tolerance=None):
        """Thread-safe update of inference thresholds."""
        locker = QMutexLocker(self._mutex)
        try:
            self._thresholds.update(
                conf=conf,
                iou=iou,
                max_det=max_det,
                boundary_tolerance=boundary_tolerance,
            )
        finally:
            del locker

    def _decode_source(self, item):
        """Convert an InferenceItem's source to a YOLO-compatible input.

        Returns a file-path string (YOLO reads it directly) or a BGR numpy
        array (for tiles and video frames decoded on this worker thread).
        """
        if isinstance(item.source, str):
            return item.source                       # full image file path
        if isinstance(item.source, int):
            # Video frame index → decode via VideoCapture on the worker thread
            bgr = item.raster.get_bgr_frame(item.source)
            return bgr if bgr is not None else np.zeros((640, 640, 3), dtype=np.uint8)
        # WorkArea object → decode the tile crop on the worker thread
        try:
            return item.raster.get_work_area_data(item.source, as_format='BGR')
        except Exception:
            return np.zeros((640, 640, 3), dtype=np.uint8)

    def _decode_video_source(self, item):
        """Decode a video frame into a model-ready tensor and preview QImage."""
        try:
            decoder = getattr(item.raster, "get_frame_for_inference", None)
            if decoder is not None:
                model_input, q_image = decoder(item.source, device=self.device)
                if model_input is not None:
                    return model_input, q_image
        except Exception:
            pass

        try:
            raw = item.raster.get_bgr_frame(item.source)
            if raw is None:
                return np.zeros((640, 640, 3), dtype=np.uint8), None

            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster

            return raw, VideoRaster._bgr_to_qimage(raw)
        except Exception:
            return np.zeros((640, 640, 3), dtype=np.uint8), None

    def _resolve_target_device(self):
        """Convert the configured device into a torch.device when possible."""
        if torch is None:
            return None
        if self.device is None:
            return None
        if isinstance(self.device, torch.device):
            return self.device
        try:
            return torch.device(self.device)
        except Exception:
            return None

    def _resolve_model_stride(self):
        """Best-effort resolution of the model stride for tensor inputs."""
        if torch is None:
            return 32
        stride = None
        try:
            stride = getattr(self.model, 'stride', None)
        except Exception:
            stride = None

        if stride is None:
            try:
                overrides = getattr(self.model, 'overrides', None)
                if isinstance(overrides, dict):
                    stride = overrides.get('stride', None)
            except Exception:
                stride = None

        if torch.is_tensor(stride):
            try:
                if stride.numel() > 0:
                    stride = int(stride.max().item())
            except Exception:
                stride = None
        elif isinstance(stride, (tuple, list)):
            try:
                stride = max(int(value) for value in stride if value is not None)
            except Exception:
                stride = None

        try:
            stride = int(stride)
        except Exception:
            stride = 32

        return max(1, stride)

    def _prepare_video_input(self, video_input):
        """Convert a video frame into a stride-safe BCHW tensor for YOLO."""
        if torch is None:
            return video_input

        if video_input is None:
            return torch.zeros((1, 3, 640, 640), dtype=torch.float32)

        if not torch.is_tensor(video_input):
            if isinstance(video_input, np.ndarray):
                video_input = torch.from_numpy(np.ascontiguousarray(video_input))
            else:
                video_input = torch.as_tensor(video_input)

        if video_input.ndim == 3:
            if video_input.shape[0] == 3:
                video_input = video_input.unsqueeze(0)
            elif video_input.shape[-1] == 3:
                video_input = video_input.permute(2, 0, 1).unsqueeze(0)
        elif video_input.ndim == 4 and video_input.shape[1] != 3 and video_input.shape[-1] == 3:
            video_input = video_input.permute(0, 3, 1, 2)

        if video_input.ndim != 4:
            return torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=self._resolve_target_device() or 'cpu')

        target_device = self._resolve_target_device()
        if target_device is not None and video_input.device != target_device:
            video_input = video_input.to(target_device)

        if not torch.is_floating_point(video_input):
            video_input = video_input.float().div(255.0)
        else:
            video_input = video_input.float()

        stride = self._resolve_model_stride()
        height = int(video_input.shape[2])
        width = int(video_input.shape[3])
        pad_height = (-height) % stride
        pad_width = (-width) % stride

        if pad_height or pad_width:
            video_input = F.pad(video_input, (0, pad_width, 0, pad_height), value=0.0)

        return video_input

    def run(self):
        """Inference loop executed on the worker thread."""
        try:
            total = len(self.items)
            processed = 0
            i = 0

            while i < total:
                # Apply the per-video-frame gate before starting any new work
                gate_wait_seconds = 0.0
                if self._waiting_for_ui:
                    gate_start = time.perf_counter()
                    while self._waiting_for_ui and self._is_running:
                        time.sleep(0.002)
                    gate_wait_seconds = time.perf_counter() - gate_start
                        

                # Check stop flag and snapshot thresholds
                locker = QMutexLocker(self._mutex)
                try:
                    if not self._is_running:
                        break
                    thresholds = self._thresholds.snapshot()
                    conf = thresholds.conf
                    iou = thresholds.iou
                    max_det = thresholds.max_det
                    boundary_tolerance = thresholds.boundary_tolerance
                finally:
                    del locker

                # Determine the batch boundary:
                # - Video items process one at a time (gated) for smooth playback.
                # - Consecutive non-video items are batched up to _batch_size for throughput.
                if self.items[i].is_video:
                    batch_end = i + 1
                else:
                    batch_end = i
                    while (batch_end < min(i + self._batch_size, total)
                           and not self.items[batch_end].is_video):
                        batch_end += 1
                    if batch_end == i:
                        batch_end = i + 1

                batch = self.items[i:batch_end]

                # Decode each source in the batch (file path, frame, or tile)
                inputs = []
                video_q_images = []
                decode_start = time.perf_counter()
                for item in batch:
                    try:
                        if item.is_video:
                            model_input, q_image = self._decode_video_source(item)
                            inputs.append(model_input)
                            video_q_images.append(q_image)
                        else:
                            inputs.append(self._decode_source(item))
                            video_q_images.append(None)
                    except Exception:
                        inputs.append(np.zeros((640, 640, 3), dtype=np.uint8))
                        video_q_images.append(None)
                decode_seconds = time.perf_counter() - decode_start

                # Run YOLO on the mini-batch
                model_source = self._prepare_video_input(inputs[0]) if len(batch) == 1 and batch[0].is_video else inputs
                inference_start = time.perf_counter()
                try:
                    results = list(self.model(
                        model_source,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        device=self.device,
                        retina_masks=self._is_semantic,
                        half=True,
                        agnostic_nms=True,
                        stream=True,
                        verbose=False,
                    ))
                except Exception as e:
                    inference_seconds = time.perf_counter() - inference_start
                    self._timing.add_record(BatchTimingRecord(
                        batch_size=len(batch),
                        is_video=any(item.is_video for item in batch),
                        decode_seconds=decode_seconds,
                        inference_seconds=inference_seconds,
                        postprocess_seconds=0.0,
                        gate_wait_seconds=gate_wait_seconds,
                    ))
                    self.error.emit(str(e))
                    processed += len(batch)
                    self.progressUpdated.emit(processed, total)
                    i = batch_end
                    continue
                inference_seconds = time.perf_counter() - inference_start

                # Post-process each result and emit to the main thread
                postprocess_start = time.perf_counter()
                for j, (item, result) in enumerate(zip(batch, results)):
                    try:
                        result.path = item.image_path

                        # Remap tile coordinates to full-image space (pure math — thread-safe).
                        # Skipped for semantic: we paint the tile-sized mask at the offset
                        # ourselves in on_item_processed, so orig_shape and masks.data must
                        # stay consistent at tile dimensions.
                        # Skipped when SAM is enabled: _apply_sam_to_cache needs
                        # tile-space boxes + tile orig_img so SAM's ViT encoder
                        # runs on the small crop instead of the full raster; it
                        # remaps boxes and masks together after SAM completes.
                        if (item.work_area is not None
                                and not self._is_semantic
                                and not self._sam_enabled):
                            from coralnet_toolbox.Results.MapResults import MapResults
                            result = MapResults().map_results_from_work_area(
                                result,
                                item.raster,
                                item.work_area,
                                map_masks=(self._task == 'segment'),
                                task=self._task,
                                boundary_tolerance=boundary_tolerance,
                            )

                        # Build a QImage from the raw BGR frame (video items only)
                        q_img = None
                        if item.is_video:
                            q_img = video_q_images[j]
                            if q_img is None:
                                raw = inputs[j]
                                if isinstance(raw, np.ndarray):
                                    try:
                                        from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                                        q_img = VideoRaster._bgr_to_qimage(raw)
                                    except Exception:
                                        pass

                        self.itemProcessed.emit(InferenceResult(
                            batch_key=item.batch_key,
                            image_path=item.image_path,
                            yolo_result=result,
                            work_area=item.work_area,
                            is_video=item.is_video,
                            q_image=q_img,
                        ))

                        # Gate after video frames to prevent queue buildup
                        if item.is_video:
                            self._waiting_for_ui = True

                    except Exception as e:
                        self.error.emit(str(e))

                    processed += 1
                    self.progressUpdated.emit(processed, total)
                postprocess_seconds = time.perf_counter() - postprocess_start
                self._timing.add_record(BatchTimingRecord(
                    batch_size=len(batch),
                    is_video=any(item.is_video for item in batch),
                    decode_seconds=decode_seconds,
                    inference_seconds=inference_seconds,
                    postprocess_seconds=postprocess_seconds,
                    gate_wait_seconds=gate_wait_seconds,
                ))

                i = batch_end

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.timingSummary.emit(self._timing.summary())
            except Exception:
                pass
            self.finished.emit()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """
    Consolidated batch inference dialog for all models.
    Supports: Classify, Detect, Segment, Semantic, SAM, SeeAnything, Z-Inference.

    Detect and Segment tasks (including video and tiled variants) are routed
    through the async BatchInferenceWorker for maximum throughput.
    Classify, Semantic, SAM, SeeAnything, and Z-Inference use their own
    synchronous predict() paths.

    Images are selected through the ImageWindow context menu (right-click).

    :param main_window: MainWindow object
    :param parent: Parent widget
    :param highlighted_images: List of image paths to process (required)
    """
    def __init__(self, main_window, parent=None, highlighted_images=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Batch Inference")
        self.resize(500, 400)
        # Keep this dialog on top so users can update highlights while it is open
        try:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        except Exception:
            # Fallback for PyQt versions without setWindowFlag
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Initialize references to various deployment dialogs
        self._refresh_model_dialog_references()

        # Dictionary to store available model dialogs
        self.model_dialogs = {}
        self.model_keys = []
        self.loaded_model = None
        self.current_selected_model = None  # Track the current selected model

        # Storage for image paths
        self.image_paths = []
        
        # Store highlighted images if provided
        self.highlighted_images = highlighted_images if highlighted_images else []

        self.layout = QVBoxLayout(self)

        # Setup layouts in order
        self.setup_info_layout()
        self.setup_options_layout()
        self.setup_task_specific_layout()
        self.setup_thresholds_layout()
        self.setup_buttons_layout()

    def _refresh_model_dialog_references(self):
        """Refresh cached dialog references from MainWindow."""
        self.classify_dialog = getattr(self.main_window, 'classify_deploy_model_dialog', None)
        self.detect_dialog = getattr(self.main_window, 'detect_deploy_model_dialog', None)
        self.segment_dialog = getattr(self.main_window, 'segment_deploy_model_dialog', None)
        self.semantic_dialog = getattr(self.main_window, 'semantic_deploy_model_dialog', None)
        self.sam_dialog = getattr(self.main_window, 'sam_deploy_generator_dialog', None)
        self.see_anything_dialog = getattr(self.main_window, 'see_anything_deploy_generator_dialog', None)
        self.z_dialog = getattr(self.main_window, 'z_deploy_model_dialog', None)

    def _update_worker_thresholds(self, *args):
        """Safely pass updated global thresholds from MainWindow to the active worker.

        This slot listens to MainWindow signals (`uncertaintyChanged`, `iouChanged`,
        `maxDetectionsChanged`, `boundaryToleranceChanged`) and forwards the latest values to the running
        BatchInferenceWorker (if present). Connecting to MainWindow avoids
        creating duplicate anonymous lambda handlers on every run.
        """
        worker = getattr(self, '_batch_worker', None)
        if worker is None:
            return
        try:
            conf = self.main_window.get_uncertainty_thresh() if hasattr(self.main_window, 'get_uncertainty_thresh') else None
            iou = self.main_window.get_iou_thresh() if hasattr(self.main_window, 'get_iou_thresh') else None
            max_det = self.main_window.get_max_detections() if hasattr(self.main_window, 'get_max_detections') else None
            boundary_tolerance = self.main_window.get_boundary_tolerance() if hasattr(self.main_window, 'get_boundary_tolerance') else None
            worker.update_thresholds(conf=conf, iou=iou, max_det=max_det, boundary_tolerance=boundary_tolerance)
        except Exception:
            pass

        # Route global threshold changes (MainWindow) into the active worker.
        # Connect once per dialog lifetime to avoid duplicate connections.
        try:
            if not getattr(self, '_thresholds_connected', False):
                self.main_window.uncertaintyChanged.connect(self._update_worker_thresholds)
                self.main_window.iouChanged.connect(self._update_worker_thresholds)
                self.main_window.maxDetectionsChanged.connect(self._update_worker_thresholds)
                if hasattr(self.main_window, 'boundaryToleranceChanged'):
                    self.main_window.boundaryToleranceChanged.connect(self._update_worker_thresholds)
                self._thresholds_connected = True
        except Exception:
            pass

    def showEvent(self, event):
        """
        Update model availability when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.update_status_label()
        self.update_model_availability()
        
        # Check if any models are available
        if not self.model_dialogs:
            QMessageBox.warning(self, 
                                "No Models Available", 
                                "Please load a model before opening batch inference.")
            self.reject()
            return
        
        # Untoggle all tools in the annotation window
        self.annotation_window.toolChanged.emit(None)
        # Ensure main window tools are restored (in case cleanup was called mid-run)
        try:
            try:
                self.main_window.set_video_playback_tools_enabled(True)
            except Exception:
                pass
        except Exception:
            pass
        
        if hasattr(self, 'thresholds_widget'):
            self.thresholds_widget.initialize_thresholds()
        # Ensure video UI reflects current highlights and, if a single
        # video is selected, default the start frame to the current
        # frame and the end frame to the last frame of the video.
        try:
            # Update video UI based on current highlights first
            self.update_highlighted_images(self.highlighted_images)

            if len(self.highlighted_images) == 1:
                vp = self.highlighted_images[0]
                try:
                    rm = getattr(self.image_window, 'raster_manager', None)
                    raster = None
                    if rm is not None:
                        raster = rm.get_raster(vp)

                    if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                        max_frame = int(getattr(raster, 'frame_count', 1))
                        # Configure spinbox ranges
                        self.video_start_spin.setMaximum(max(0, max_frame - 1))
                        self.video_end_spin.setMaximum(max(0, max_frame - 1))

                        # Default start = current frame if available, else 0
                        current_idx = getattr(self.annotation_window, '_current_frame_idx', None)
                        if current_idx is not None:
                            try:
                                ci = int(current_idx)
                            except Exception:
                                ci = 0
                            ci = min(max(0, ci), max(0, max_frame - 1))
                            self.video_start_spin.setValue(ci)
                        else:
                            self.video_start_spin.setValue(0)

                        # Default end = last frame
                        self.video_end_spin.setValue(max(0, max_frame - 1))

                        # Ensure controls are enabled for single-video case
                        try:
                            self.video_group.setEnabled(True)
                            self.video_start_spin.setEnabled(True)
                            self.video_end_spin.setEnabled(True)
                            self.video_stride_spin.setEnabled(True)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def setup_info_layout(self):
        """
        Set up the info layout with explanatory text.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Perform batch inferencing on the selected images. "
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """
        Combined Options: Model, Inference Type, and Save Annotations.
        """
        group_box = QGroupBox("Options")
        form_layout = QFormLayout()

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        form_layout.addRow("Model:", self.model_combo)

        # Inference type selection
        self.inference_type_combo = QComboBox()
        self.inference_type_combo.addItem("Standard")
        self.inference_type_combo.addItem("Tiled")
        self.inference_type_combo.currentTextChanged.connect(self.on_inference_type_changed)
        form_layout.addRow("Type:", self.inference_type_combo)

        # Save annotations (moved out of Video Options so editable for non-video runs)
        self.save_annotations_combo = QComboBox()
        self.save_annotations_combo.addItems(["True", "False"])
        self.save_annotations_combo.setCurrentText("True")
        self.save_video_annotations = True
        try:
            self.save_annotations_combo.currentTextChanged.connect(self._on_save_annotations_changed)
        except Exception:
            pass
        form_layout.addRow("Save Annotations:", self.save_annotations_combo)

        # Batch size — controls the mini-batch sent to the GPU per inference call.
        # Reduce if you see out-of-memory errors; increase for higher GPU utilisation.
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(256)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setToolTip(
            "Number of tiles/frames sent to the GPU per inference call.\n"
            "Reduce if you get out-of-memory errors."
        )
        form_layout.addRow("Batch Size:", self.batch_size_spin)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def setup_inference_layout(self):
        """
        Set up the inference type selection dropdown.
        """
        group_box = QGroupBox("Inference Type")
        form_layout = QFormLayout()

        self.inference_type_combo = QComboBox()
        self.inference_type_combo.addItem("Standard")
        self.inference_type_combo.addItem("Tiled")
        self.inference_type_combo.currentTextChanged.connect(self.on_inference_type_changed)
        form_layout.addRow("Type:", self.inference_type_combo)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def setup_thresholds_layout(self):
        """
        Set up the ThresholdWidget for configurable thresholds.
        All thresholds are shown, but specific ones are disabled based on model type.
        Can be overridden by subclasses to configure which thresholds to show.
        """
        # Show all thresholds - they will be enabled/disabled based on model selection
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_boundary=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True,
            title="Thresholds"
        )
        self.layout.addWidget(self.thresholds_widget)

    def setup_task_specific_layout(self):
        """
        Override in subclasses to add task-specific options.
        For Classify model, adds annotation selection options (review vs all).
        Visibility is controlled by on_model_changed().
        """
        # Create a group box for Classify-specific annotation options
        group_box = QGroupBox("Classify Options")
        layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        annotation_options_group = QButtonGroup(self)

        # Add the checkboxes to the button group
        self.review_checkbox = QCheckBox("Predict Review Annotation")
        self.all_checkbox = QCheckBox("Predict All Annotations")
        annotation_options_group.addButton(self.review_checkbox)
        annotation_options_group.addButton(self.all_checkbox)

        # Ensure only one checkbox can be checked at a time
        annotation_options_group.setExclusive(True)
        # Set the default checkbox
        self.review_checkbox.setChecked(True)

        # Build the annotation layout
        layout.addWidget(self.review_checkbox)
        layout.addWidget(self.all_checkbox)

        group_box.setLayout(layout)
        self.task_specific_group = group_box
        self.layout.addWidget(group_box)

        # --- Video-specific options (Start / End / Stride) ---
        video_box = QGroupBox("Video Options")
        video_layout = QFormLayout()

        # Save annotations moved to Options group so users can change it
        # even when Video Options are disabled.

        # Start frame with 'Set to Current' button
        start_h = QHBoxLayout()
        self.video_start_spin = QSpinBox()
        self.video_start_spin.setMinimum(0)
        self.video_start_spin.setMaximum(99999999)
        self.set_start_current_btn = QPushButton("Set to Current")
        start_h.addWidget(self.video_start_spin)
        start_h.addWidget(self.set_start_current_btn)

        # End frame with 'Set to Current' button
        end_h = QHBoxLayout()
        self.video_end_spin = QSpinBox()
        self.video_end_spin.setMinimum(0)
        self.video_end_spin.setMaximum(99999999)
        self.set_end_current_btn = QPushButton("Set to Current")
        end_h.addWidget(self.video_end_spin)
        end_h.addWidget(self.set_end_current_btn)

        # Stride (Every N frames)
        stride_h = QHBoxLayout()
        self.video_stride_spin = QSpinBox()
        self.video_stride_spin.setMinimum(1)
        self.video_stride_spin.setMaximum(9999)
        self.video_stride_spin.setValue(1)
        stride_h.addWidget(self.video_stride_spin)

        # Reset button
        self.reset_video_range_btn = QPushButton("Reset to Full Video")

        video_layout.addRow("Start Frame:", start_h)
        video_layout.addRow("End Frame:", end_h)
        video_layout.addRow("Every N Frames:", stride_h)
        video_layout.addRow("", self.reset_video_range_btn)

        video_box.setLayout(video_layout)
        self.video_group = video_box
        self.layout.addWidget(video_box)

        # Connect handlers
        try:
            self.set_start_current_btn.clicked.connect(self._on_set_start_to_current)
            self.set_end_current_btn.clicked.connect(self._on_set_end_to_current)
            self.reset_video_range_btn.clicked.connect(self._on_reset_video_range)
            # Keep start <= end by silently fixing the other spinbox when needed
            self.video_start_spin.valueChanged.connect(self._on_video_start_changed)
            self.video_end_spin.valueChanged.connect(self._on_video_end_changed)
        except Exception:
            pass

        # Initially disabled until videos are selected
        try:
            self.video_group.setEnabled(False)
        except Exception:
            pass

    def setup_buttons_layout(self):
        """
        Set up the dialog buttons with status label.
        """
        # Create a horizontal layout for buttons and status label
        button_layout = QHBoxLayout()
        
        # Status label on the left
        self.status_label = QLabel()
        self.update_status_label()
        button_layout.addWidget(self.status_label)

        # Stretch to push buttons to the right
        button_layout.addStretch()
        
        # Buttons on the right
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.apply)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)
        
        self.layout.addLayout(button_layout)

    def update_status_label(self):
        """
        Update the status label to show the number of images for batch inference.
        """
        num_images = len(self.highlighted_images)
        if num_images == 0:
            self.status_label.setText("No rasters selected")
        elif num_images == 1:
            self.status_label.setText("1 raster selected")
        else:
            self.status_label.setText(f"{num_images} rasters selected")

    def update_status_label_for_tiled(self):
        """
        Update the status label to show the number of images and work areas for Tiled mode.
        Counts work areas across all selected images.
        """
        num_images = len(self.highlighted_images)
        total_work_areas = 0

        # Count total work areas across all selected images
        if self.image_window and hasattr(self.image_window, 'raster_manager'):
            for image_path in self.highlighted_images:
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster and hasattr(raster, 'work_areas'):
                    total_work_areas += len(raster.work_areas)

        # Format the status text
        if num_images == 0:
            self.status_label.setText("No rasters selected")
        elif num_images == 1:
            if total_work_areas == 0:
                self.status_label.setText("1 raster, no work areas")
            elif total_work_areas == 1:
                self.status_label.setText("1 raster, 1 work area")
            else:
                self.status_label.setText(f"1 raster, {total_work_areas} work areas")
        else:
            if total_work_areas == 0:
                self.status_label.setText(f"{num_images} rasters, no work areas")
            elif total_work_areas == 1:
                self.status_label.setText(f"{num_images} rasters, 1 work area")
            else:
                self.status_label.setText(f"{num_images} rasters, {total_work_areas} work areas")

    def update_model_availability(self):
        """
        Check which models are loaded and populate the model dialog dictionary.
        """
        self._refresh_model_dialog_references()
        self.model_dialogs = {}

        if self.classify_dialog and getattr(self.classify_dialog, "loaded_model", None):
            self.model_dialogs["Classify"] = self.classify_dialog
        if self.detect_dialog and getattr(self.detect_dialog, "loaded_model", None):
            self.model_dialogs["Detect"] = self.detect_dialog
        if self.segment_dialog and getattr(self.segment_dialog, "loaded_model", None):
            self.model_dialogs["Segment"] = self.segment_dialog
        if self.semantic_dialog and getattr(self.semantic_dialog, "loaded_model", None):
            self.model_dialogs["Semantic"] = self.semantic_dialog
        if self.sam_dialog and getattr(self.sam_dialog, "loaded_model", None):
            self.model_dialogs["SAM"] = self.sam_dialog
        if self.see_anything_dialog and getattr(self.see_anything_dialog, "loaded_model", None):
            self.model_dialogs["See Anything"] = self.see_anything_dialog
        if self.z_dialog and getattr(self.z_dialog, "loaded_model", None):
            self.model_dialogs["Z-Inference"] = self.z_dialog

        self.update_model_combo()

    def update_model_combo(self):
        """
        Update the model dropdown with available models.
        Preserves the last selected model if it's still available.
        """
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_keys = []

        # Separate Z-Inference from other models
        z_inference_dialog = None
        other_models = {}
        
        for key, dialog in self.model_dialogs.items():
            if key == "Z-Inference":
                z_inference_dialog = dialog
            else:
                other_models[key] = dialog
        
        # Add sorted models (excluding Z-Inference)
        for key in sorted(other_models.keys()):
            self.model_combo.addItem(key)
            self.model_keys.append(key)
        
        # Add separator and Z-Inference at the bottom
        if z_inference_dialog:
            self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItem("Z-Inference")
            self.model_keys.append("Z-Inference")

        # Try to restore the current selected model, otherwise default to index 0
        selected_index = 0
        if self.current_selected_model and self.current_selected_model in self.model_keys:
            selected_index = self.model_keys.index(self.current_selected_model)
        
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(selected_index)

        self.model_combo.blockSignals(False)
        self.update_loaded_model()
        
        # Manually trigger the UI update for the selected model
        if self.model_combo.count() > 0:
            self.on_model_changed(selected_index)

    def update_loaded_model(self):
        """
        Update the loaded_model based on the current selection.
        """
        idx = self.model_combo.currentIndex()
        if 0 <= idx < len(self.model_keys):
            key = self.model_keys[idx]
            self.loaded_model = self.model_dialogs.get(key, None)
        else:
            self.loaded_model = None

    def on_model_changed(self, index):
        """
        Handle model selection change and update thresholds accordingly.
        
        :param index: Index of the selected model
        """
        # Update the loaded model
        self.update_loaded_model()
        
        # Update threshold visibility based on selected model
        if 0 <= index < len(self.model_keys):
            selected_model = self.model_keys[index]
            # Save the selected model for later
            self.current_selected_model = selected_model
            self.update_thresholds_for_model(selected_model)
            
            # Enable only for Classify; otherwise keep the group visible but
            # disabled so users can see options but not interact with them.
            if selected_model == "Classify":
                self.task_specific_group.setVisible(True)
                self.task_specific_group.setEnabled(True)
                # Disable inference type dropdown for Classify model
                self.inference_type_combo.setEnabled(False)
                self.inference_type_combo.setCurrentText("Standard")
            elif selected_model == "Z-Inference":
                # Keep annotation/classify options visible but disabled for Z-Inference
                self.task_specific_group.setVisible(True)
                self.task_specific_group.setEnabled(False)
                self.inference_type_combo.setEnabled(False)
                self.inference_type_combo.setCurrentText("Standard")
            else:
                # Show group but grey it out for non-Classify models
                self.task_specific_group.setVisible(True)
                self.task_specific_group.setEnabled(False)
                # Enable inference type dropdown for other models
                self.inference_type_combo.setEnabled(True)
    
    def on_inference_type_changed(self, inference_type):
        """
        Handle inference type change and update the annotation window tool accordingly.
        Also updates the status label to show work area information for Tiled mode.
        
        :param inference_type: The selected inference type ("Standard" or "Tiled")
        """
        if inference_type == "Tiled":
            # Set the annotation window tool to work_area
            self.annotation_window.toolChanged.emit("work_area")
            # Update status label to show images and work areas
            self.update_status_label_for_tiled()
        else:
            # Set the annotation window tool to None (untoggle all tools)
            self.annotation_window.toolChanged.emit(None)
            # Reset status label to show only images
            self.update_status_label()
    
    def update_thresholds_for_model(self, model_name):
        """
        Update the enabled state of thresholds based on model type.
        All thresholds remain visible, but specific ones are disabled.
        
        :param model_name: Name of the selected model
        """
        # Classify and Semantic only use uncertainty threshold
        if model_name in ("Classify", "Semantic"):
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=True,
                enable_iou=False,
                enable_area=False
            )
        # Detect, Segment, SAM, and See Anything use max_detections, uncertainty, iou, and area
        elif model_name in ("Detect", "Segment", "SAM", "See Anything"):
            self.configure_thresholds(
                enable_max_detections=True,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # Z-Inference doesn't use any thresholds
        elif model_name == "Z-Inference":
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=False,
                enable_iou=False,
                enable_area=False
            )
    
    def configure_thresholds(self, enable_max_detections, enable_uncertainty,
                             enable_iou, enable_area):
        """
        Configure which thresholds are enabled/disabled.
        All thresholds remain visible, but specific ones are disabled and greyed out.
        
        :param enable_max_detections: Whether to enable max detections
        :param enable_uncertainty: Whether to enable uncertainty threshold
        :param enable_iou: Whether to enable IoU threshold
        :param enable_area: Whether to enable area threshold
        """
        # Helper function to get all QLabel widgets from the form layout
        def get_form_labels_for_widget(widget):
            """Find and return all QLabel widgets associated with a control in the form layout."""
            layout = self.thresholds_widget.layout()
            labels = []
            if layout and isinstance(layout, QFormLayout):
                for row in range(layout.rowCount()):
                    label_item = layout.itemAt(row, QFormLayout.LabelRole)
                    widget_item = layout.itemAt(row, QFormLayout.FieldRole)
                    
                    if widget_item and widget_item.widget() == widget:
                        if label_item and label_item.widget():
                            labels.append(label_item.widget())
            return labels
        
        # Enable/disable max detections
        if hasattr(self.thresholds_widget, 'max_detections_spinbox'):
            spinbox = self.thresholds_widget.max_detections_spinbox
            spinbox.setEnabled(enable_max_detections)
            # Find and disable associated labels
            labels = get_form_labels_for_widget(spinbox)
            for label in labels:
                label.setEnabled(enable_max_detections)
        
        # Enable/disable uncertainty threshold
        if hasattr(self.thresholds_widget, 'uncertainty_threshold_slider'):
            slider = self.thresholds_widget.uncertainty_threshold_slider
            slider.setEnabled(enable_uncertainty)
            value_label = getattr(self.thresholds_widget, 'uncertainty_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_uncertainty)
            # Find and disable associated title labels
            labels = get_form_labels_for_widget(slider)
            for label in labels:
                label.setEnabled(enable_uncertainty)
        
        # Enable/disable IoU threshold
        if hasattr(self.thresholds_widget, 'iou_threshold_slider'):
            slider = self.thresholds_widget.iou_threshold_slider
            slider.setEnabled(enable_iou)
            value_label = getattr(self.thresholds_widget, 'iou_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_iou)
            # Find and disable associated title labels
            labels = get_form_labels_for_widget(slider)
            for label in labels:
                label.setEnabled(enable_iou)
        
        # Enable/disable area threshold
        if hasattr(self.thresholds_widget, 'area_threshold_min_slider'):
            min_slider = self.thresholds_widget.area_threshold_min_slider
            max_slider = self.thresholds_widget.area_threshold_max_slider
            min_slider.setEnabled(enable_area)
            max_slider.setEnabled(enable_area)
            value_label = getattr(self.thresholds_widget, 'area_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_area)
            # Find and disable associated title labels
            min_labels = get_form_labels_for_widget(min_slider)
            max_labels = get_form_labels_for_widget(max_slider)
            for label in min_labels + max_labels:
                label.setEnabled(enable_area)

    def check_model_availability(self):
        """
        Check if a model is loaded and available for batch inference.

        :return: True if a model is loaded, False otherwise
        """
        self.update_model_availability()
        return self.loaded_model is not None

    def get_selected_image_paths(self):
        """
        Get the selected image paths.
        Images are provided at initialization through the highlighted_images parameter.

        :return: List of selected image paths
        """
        # Return the highlighted images provided at initialization
        return self.highlighted_images

    def update_highlighted_images(self, image_paths):
        """Update the dialog's highlighted image list and refresh the status label.

        This method can be called while the dialog is open to reflect changes
        in which rows are highlighted in the ImageWindow.
        """
        try:
            self.highlighted_images = image_paths or []

            # Video-specific UI behavior:
            # - 0 videos selected: disable the Video group
            # - 1 video selected: enable Video group and allow Start/End edits
            # - >1 videos selected: show Video group, disable Start/End, keep Stride enabled
            try:
                # Gather any VideoRaster paths from highlighted_images
                video_paths = []
                for p in self.highlighted_images:
                    try:
                        raster_manager = getattr(self.image_window, 'raster_manager', None)
                        raster = None
                        if raster_manager is not None:
                            raster = raster_manager.get_raster(p)
                        if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                            video_paths.append(p)
                    except Exception:
                        continue

                vcount = len(video_paths)
                if not hasattr(self, 'video_group'):
                    # No video UI created; skip
                    pass
                else:
                    if vcount == 0:
                        self.video_group.setEnabled(False)
                    elif vcount == 1:
                        self.video_group.setEnabled(True)
                        # enable controls
                        self.video_start_spin.setEnabled(True)
                        self.video_end_spin.setEnabled(True)
                        self.video_stride_spin.setEnabled(True)

                        # set spinbox ranges / defaults from the single video
                        vp = video_paths[0]
                        try:
                            raster_manager = getattr(self.image_window, 'raster_manager', None)
                            raster = None
                            if raster_manager is not None:
                                raster = raster_manager.get_raster(vp)
                            if raster is not None and hasattr(raster, 'frame_count'):
                                max_frame = max(1, int(getattr(raster, 'frame_count', 1)))
                                self.video_start_spin.setMaximum(max_frame - 1)
                                self.video_end_spin.setMaximum(max_frame - 1)
                                # default to full video
                                self.video_start_spin.setValue(0)
                                self.video_end_spin.setValue(max_frame - 1)
                        except Exception:
                            pass
                    else:
                        # Multiple videos selected: allow stride, but not explicit start/end
                        self.video_group.setEnabled(True)
                        self.video_start_spin.setEnabled(False)
                        self.video_end_spin.setEnabled(False)
                        self.video_stride_spin.setEnabled(True)

            except Exception:
                # non-fatal video UI update errors
                pass

            # Update the status label depending on inference type
            if hasattr(self, 'inference_type_combo') and self.inference_type_combo.currentText() == 'Tiled':
                self.update_status_label_for_tiled()
            else:
                self.update_status_label()
        except Exception:
            # Swallow errors to avoid noisy exceptions from signal handlers
            pass

    def _on_set_start_to_current(self):
        """Set the Start spinbox to the main annotation window's current frame."""
        try:
            idx = getattr(self.annotation_window, '_current_frame_idx', None)
            if idx is None:
                return
            self.video_start_spin.setValue(int(idx))
        except Exception:
            pass

    def _on_save_annotations_changed(self, text):
        """Handler for the Save Annotations combobox.

        Stores a boolean flag on the dialog which is checked when video
        frames are processed. Defaults to True.
        """
        try:
            # Accept several truthy string forms, but primarily expect "True"/"False"
            self.save_video_annotations = True if str(text).lower() in ("true", "1", "yes") else False
        except Exception:
            # Fallback to True to avoid accidentally dropping saved results
            self.save_video_annotations = True

    def _on_set_end_to_current(self):
        """Set the End spinbox to the main annotation window's current frame."""
        try:
            idx = getattr(self.annotation_window, '_current_frame_idx', None)
            if idx is None:
                return
            self.video_end_spin.setValue(int(idx))
        except Exception:
            pass

    def _on_reset_video_range(self):
        """Reset start/end to full video (0 .. frame_count-1) for single selected video."""
        try:
            if not self.highlighted_images:
                return
            # Prefer the first highlighted video
            for p in self.highlighted_images:
                raster = None
                try:
                    raster_manager = getattr(self.image_window, 'raster_manager', None)
                    if raster_manager is not None:
                        raster = raster_manager.get_raster(p)
                except Exception:
                    continue
                if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                    max_frame = int(getattr(raster, 'frame_count', 1))
                    self.video_start_spin.setMaximum(max_frame - 1)
                    self.video_end_spin.setMaximum(max_frame - 1)
                    self.video_start_spin.setValue(0)
                    self.video_end_spin.setValue(max_frame - 1)
                    return
        except Exception:
            pass

    def _on_video_start_changed(self, value):
        """Ensure start <= end; if not, silently set end to start."""
        try:
            # If end is less than new start, bump end up to start
            end_val = self.video_end_spin.value()
            if end_val < value:
                self.video_end_spin.blockSignals(True)
                self.video_end_spin.setValue(value)
                self.video_end_spin.blockSignals(False)
        except Exception:
            pass

    def _on_video_end_changed(self, value):
        """Ensure end >= start; if not, silently set start to end."""
        try:
            start_val = self.video_start_spin.value()
            if start_val > value:
                self.video_start_spin.blockSignals(True)
                self.video_start_spin.setValue(value)
                self.video_start_spin.blockSignals(False)
        except Exception:
            pass

    def apply(self):
        """
        Apply the selected batch inference options.
        """
        if not self.check_model_availability():
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            self.image_paths = self.get_selected_image_paths()
            if not self.image_paths:
                QMessageBox.warning(self, "No Images", "No images selected for inference.")
                return

            # Get the selected model type
            idx = self.model_combo.currentIndex()
            if 0 <= idx < len(self.model_keys):
                selected_model = self.model_keys[idx]
                
                # Verify the model is available
                if selected_model not in self.model_dialogs:
                    QMessageBox.warning(self, 
                                        "Model Not Available",
                                        f"{selected_model} model is not loaded.")
                    return

            # If a video is active, ensure the player is synced to the
            # requested start frame before beginning batch inference.
            try:
                aw = getattr(self, 'annotation_window', None)
                if aw is not None and hasattr(aw, '_active_video_raster') and aw._active_video_raster is not None:
                    # Determine requested start frame
                    try:
                        start_frame = int(self.video_start_spin.value()) if hasattr(self, 'video_start_spin') else 0
                    except Exception:
                        start_frame = 0

                    # If player is playing, pause it via the AnnotationWindow handler
                    try:
                        vp = getattr(aw, '_video_player', None)
                        timer = getattr(aw, '_playback_timer', None)
                        is_playing = False
                        if vp is not None:
                            is_playing = getattr(vp, 'is_playing', False)
                        if not is_playing and timer is not None:
                            try:
                                is_playing = timer.isActive()
                            except Exception:
                                pass

                        if is_playing:
                            try:
                                # Prefer the window-level pause handler to keep state consistent
                                if hasattr(aw, '_on_video_pause'):
                                    aw._on_video_pause()
                                else:
                                    if vp is not None:
                                        vp.set_paused()
                                        try:
                                            vp.pauseClicked.emit()
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                        # Seek to the start frame so the worker and UI are in sync
                        try:
                            if hasattr(aw, '_on_video_seek'):
                                aw._on_video_seek(start_frame)
                            elif hasattr(aw, '_display_video_frame'):
                                aw._display_video_frame(start_frame)
                        except Exception:
                            pass

                    except Exception:
                        pass
                    
            except Exception:
                pass

            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete batch inference: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            # Only perform final cleanup and close the dialog if no background
            # worker was started or it is already finished. If a worker is
            # running, leave cleanup to `_on_worker_finished` so the thread can
            # complete normally without being killed immediately.
            worker = getattr(self, '_batch_worker', None)
            if worker is None or not getattr(worker, 'isRunning', lambda: False)():
                try:
                    self.cleanup()
                except Exception:
                    pass
                try:
                    self.accept()
                except Exception:
                    pass

    def _build_inference_items(self, image_paths, inference_type,
                               video_start=0, video_end=None, video_stride=1):
        """Build a flat list of InferenceItem objects from the selected image paths.

        Handles three source types:
        - VideoRaster, Standard  → one item per frame (is_video=True)
        - Raster, Standard       → one item per image (file-path source)
        - Raster, Tiled          → one item per WorkArea

        VideoRaster tiled mode falls back to Standard (frame-by-frame, no tiles)
        because tile coordinates on video frames are not yet supported.
        """
        raster_manager = getattr(self.image_window, 'raster_manager', None)
        return build_inference_items(
            image_paths,
            raster_manager,
            inference_type,
            video_start=video_start,
            video_end=video_end,
            video_stride=video_stride,
        )

    def _start_worker_task(self, task_runner, progress_bar):
        """Start an async BatchInferenceTask through BatchInferenceWorker."""
        items = task_runner.build_items()
        if not items:
            try:
                progress_bar.close()
            except Exception:
                pass
            return False

        task_runner.prepare_before_worker(items)

        progress_bar.set_title(task_runner.progress_title)
        progress_bar.start_progress(len(items))
        try:
            progress_bar.set_value(0)
            progress_bar.progress_bar.setValue(0)
            QApplication.processEvents()
        except Exception:
            pass

        self._progress_bar = progress_bar

        # Clear stale video cache if user opted out of saving.
        if hasattr(self, 'save_video_annotations') and not self.save_video_annotations:
            if hasattr(self.annotation_window, 'batch_results_cache'):
                self.annotation_window.batch_results_cache = {}

        self._batch_worker = task_runner.create_worker(BatchInferenceWorker, items)

        try:
            self.annotation_window.is_streaming_inference = True
            if hasattr(self.annotation_window, '_clear_current_frame_annotation_graphics'):
                self.annotation_window._clear_current_frame_annotation_graphics()
        except Exception:
            pass

        self._batch_worker.itemProcessed.connect(self.on_item_processed)
        self._batch_worker.progressUpdated.connect(self._on_worker_progress)
        self._batch_worker.timingSummary.connect(self._on_worker_timing_summary)
        self._batch_worker.error.connect(
            lambda msg: print(f"BatchInferenceWorker error: {msg}"))
        self._batch_worker.finished.connect(self._on_worker_finished)

        try:
            self._update_worker_thresholds()
        except Exception:
            pass

        # Remap OK → Stop Inference.
        try:
            try:
                self.button_box.accepted.disconnect()
            except Exception:
                pass
            self.button_box.accepted.connect(self._on_stop_inference_clicked)
            ok_btn = self.button_box.button(QDialogButtonBox.Ok)
            if ok_btn:
                ok_btn.setText("Stop Inference")
                ok_btn.setEnabled(True)
                ok_btn.setStyleSheet("background-color: #d9534f; color: white;")
        except Exception:
            pass

        try:
            progress_bar.cancel_button.setEnabled(True)
            progress_bar.cancel_button.clicked.connect(
                lambda checked=False: self._on_stop_inference_clicked())
        except Exception:
            pass

        self.set_ui_processing_state(True)
        try:
            if hasattr(self, 'thresholds_widget') and self.thresholds_widget is not None:
                self.thresholds_widget.setEnabled(True)
        except Exception:
            pass

        self._batch_worker.start()
        return True

    def batch_inference(self):
        """
        Perform batch inference on selected images.

        Detect, Segment, and Semantic tasks (all input types: full images,
        tiled images, and video frames) are routed through the unified async
        BatchInferenceWorker. Classify, SAM, SeeAnything, and Z-Inference use
        synchronous predict() paths.
        """
        self._refresh_model_dialog_references()

        # Determine the selected model type
        idx = self.model_combo.currentIndex()
        if idx < 0 or idx >= len(self.model_keys):
            raise ValueError("No model selected")

        selected_model = self.model_keys[idx]
        model_dialog = self.model_dialogs.get(selected_model)
        if model_dialog is None:
            raise ValueError(f"No model loaded for {selected_model}")

        self._active_model_dialog = model_dialog

        try:
            from coralnet_toolbox.Results import ResultsProcessor
            self._results_processor = ResultsProcessor(
                self.main_window,
                getattr(model_dialog, 'class_mapping', {}),
            )
        except Exception:
            self._results_processor = None

        progress_bar = ProgressBar(None, title="Batch Inference")
        progress_bar.setWindowFlags(progress_bar.windowFlags() | Qt.WindowStaysOnTopHint)
        progress_bar.show()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        task_runner = make_batch_inference_task(
            selected_model,
            self,
            model_dialog,
            self.image_paths,
        )
        self._active_batch_task = task_runner
        worker_started = False

        try:
            # ── Detect / Segment → unified async worker ───────────────────────
            if selected_model in ("Detect", "Segment"):
                worker_started = self._start_worker_task(task_runner, progress_bar)
                return  # _on_worker_finished drives everything from here

            # ── Classify ──────────────────────────────────────────────────────
            elif selected_model == "Classify":
                if not task_runner.run(progress_bar):
                    try:
                        progress_bar.close()
                    except Exception:
                        pass
                    return

            # ── Semantic → unified async worker (same path as Detect/Segment) ───
            elif selected_model == "Semantic":
                worker_started = self._start_worker_task(task_runner, progress_bar)
                return  # _on_worker_finished drives everything from here

            # ── SAM / See Anything ────────────────────────────────────────────
            elif selected_model in ("SAM", "See Anything"):
                # These dialogs manage their own progress bars; close the batch one first
                try:
                    progress_bar.finish_progress()
                    progress_bar.stop_progress()
                    progress_bar.close()
                except Exception:
                    pass
                model_dialog.predict(self.image_paths)

            # ── Z-Inference ───────────────────────────────────────────────────
            elif selected_model == "Z-Inference":
                overwrite_mode = self.z_dialog._show_overwrite_dialog(is_batch=True)
                if overwrite_mode is None:
                    QApplication.restoreOverrideCursor()
                    try:
                        progress_bar.close()
                    except Exception:
                        pass
                    return
                model_dialog.predict(
                    self.image_paths, progress_bar,
                    overwrite_mode=overwrite_mode, show_dialog=False,
                )

            else:
                raise ValueError(f"Unknown model type: {selected_model}")

            # Synchronous paths land here
            try:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
            except Exception:
                pass

            self._bake_cached_annotations()

            try:
                self.accept()
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch inference failed: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            if not worker_started:
                try:
                    progress_bar.finish_progress()
                    progress_bar.stop_progress()
                    progress_bar.close()
                except Exception:
                    pass

    # -------------------- Worker signal handlers --------------------
    def _on_worker_progress(self, current, total):
        """Update the progress bar from worker signals (main thread)."""
        try:
            pb = getattr(self, '_progress_bar', None)
            if pb is None:
                return
            if getattr(pb, 'max_value', None) != total:
                pb.start_progress(total)
            pb.set_value(current)
        except Exception:
            pass

    def _on_worker_timing_summary(self, summary):
        """Store and report aggregate worker timings for speed tuning."""
        try:
            self._last_batch_timing_summary = dict(summary or {})
        except Exception:
            self._last_batch_timing_summary = summary

        try:
            message = BatchInferenceTiming.human_summary(summary or {})
            print(message)
            status_bar = getattr(self.main_window, 'status_bar', None)
            if status_bar is not None:
                status_bar.showMessage(message, 8000)
        except Exception:
            pass

    def on_item_processed(self, inf_result):
        """Main-thread slot: receives one InferenceResult from the background worker.

        For video items:  updates the canvas bitmap, fast-renders predictions,
                          and updates the scrub-bar position.
        For image/tile items: silently accumulates results into the cache;
                              the canvas is updated per-image in _on_worker_finished.
        """
        try:
            aw = self.annotation_window

            # ── Ensure cache exists ──────────────────────────────────────────
            if not hasattr(aw, 'batch_results_cache') or aw.batch_results_cache is None:
                aw.batch_results_cache = {}
            cache = aw.batch_results_cache

            save = getattr(self, 'save_video_annotations', True)

            # ── Semantic: reconstruct mask and store per-frame overlays (do not mutate per-raster mask for videos)
            is_semantic_run = getattr(getattr(self, '_batch_worker', None), '_is_semantic', False)
            if is_semantic_run:
                if inf_result.yolo_result is not None:
                    try:
                        from coralnet_toolbox.MachineLearning.DeployModel.QtSemantic import (
                            _reconstruct_semantic_mask)
                        # Defer heavy imports to avoid circular top-level imports
                        try:
                            from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation
                        except Exception:
                            MaskAnnotation = None

                        raster_manager = getattr(self.image_window, 'raster_manager', None)
                        raster = raster_manager.get_raster(inf_result.image_path) if raster_manager else None

                        project_labels = self.main_window.label_window.labels

                        # Build a stable label_id -> class_id map (same ordering as MaskAnnotation.sync_label_map)
                        mask_ann_map = {lbl.id: (i + 1) for i, lbl in enumerate(project_labels)}

                        offset = (0, 0)
                        if inf_result.work_area is not None:
                            offset = (int(inf_result.work_area.rect.x()),
                                      int(inf_result.work_area.rect.y()))

                        reconstructed = _reconstruct_semantic_mask(
                            inf_result.yolo_result,
                            getattr(self, '_semantic_model_class_names', []),
                            self.main_window.label_window,
                            mask_ann_map,
                            include_background=getattr(self, '_semantic_include_bg', False),
                        )

                        # Video frames: create a temporary MaskAnnotation to produce a colored QImage,
                        # store it in the annotation window cache keyed by the virtual path.
                        if inf_result.is_video:
                            if save and reconstructed is not None and reconstructed.size:
                                try:
                                    if MaskAnnotation is not None:
                                        tmp_mask = MaskAnnotation(
                                            image_path=inf_result.batch_key,
                                            mask_data=reconstructed.copy(),
                                            initial_labels=project_labels,
                                            transparency=128,
                                            rasterio_src=None,
                                        )
                                        # Make a deep copy of the QImage so we can drop tmp_mask safely
                                        qimg_copy = tmp_mask.qimage.copy() if tmp_mask.qimage is not None else None
                                        opacity = tmp_mask.get_current_transparency() / 255.0 if tmp_mask.qimage is not None else 1.0
                                    else:
                                        qimg_copy = None
                                        opacity = 128 / 255.0

                                    # Cache per-frame mask overlay (virtual path key).
                                    # Keep the legacy dict shape for playback code while
                                    # constructing it through an explicit record contract.
                                    cache[inf_result.batch_key] = SemanticOverlayRecord(
                                        mask_qimage=qimg_copy,
                                        mask_arr=reconstructed.copy(),
                                        opacity=opacity,
                                    ).to_legacy_dict()

                                    # Show overlay immediately for the displayed frame
                                    try:
                                        if qimg_copy is not None:
                                            aw._base_image_item.set_mask_image(qimg_copy, opacity)
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"Semantic temp mask error: {e}")
                            else:
                                # Not saving per-frame masks: ensure overlay cleared
                                try:
                                    aw._base_image_item.set_mask_image(None)
                                except Exception:
                                    pass
                        else:
                            # Non-video: persist into the raster-level mask_annotation (existing behavior)
                            if raster is not None:
                                try:
                                    if raster.mask_annotation is None:
                                        try:
                                            self.main_window.status_bar.showMessage(
                                                f"Creating mask annotation for "
                                                f"{os.path.basename(inf_result.image_path)}\u2026", 3000
                                            )
                                        except Exception:
                                            pass
                                    mask_ann = raster.get_mask_annotation(project_labels)
                                    mask_ann.sync_label_map(project_labels)
                                    mask_ann.update_mask_with_mask(reconstructed, top_left=offset)

                                    if not hasattr(self, '_semantic_processed_images'):
                                        self._semantic_processed_images = set()
                                    self._semantic_processed_images.add(inf_result.image_path)
                                except Exception as e:
                                    print(f"Semantic persist mask error: {e}")
                    except Exception as e:
                        print(f"Semantic inline paint error: {e}")
                # Preserve original early-return for non-video semantic items
                if not inf_result.is_video:
                    # Non-video Semantic items: nothing left to do for this item
                    return

            # ── Store result (Detect / Segment only) ─────────────────────────
            if not is_semantic_run and inf_result.yolo_result is not None:
                if inf_result.is_video:
                    # Video: one result per virtual frame path
                    if save:
                        inf_result.yolo_result.path = inf_result.batch_key
                        cache[inf_result.batch_key] = inf_result.yolo_result
                else:
                    # Image / tile: accumulate into a list keyed by image_path.
                    # Stash serializable tile metadata so _apply_sam_to_cache can
                    # reconstruct the WorkArea when SAM is enabled without keeping
                    # the Qt object itself in the cache.
                    try:
                        inf_result.yolo_result._tile_work_area_data = (
                            inf_result.work_area.to_dict()
                            if inf_result.work_area is not None
                            else None
                        )
                    except Exception:
                        pass
                    if inf_result.batch_key not in cache:
                        cache[inf_result.batch_key] = []
                    cache[inf_result.batch_key].append(inf_result.yolo_result)

            # ── Video-specific canvas updates ─────────────────────────────────
            if inf_result.is_video:
                try:
                    from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                    _, frame_idx = VideoRaster.parse_frame_path(inf_result.batch_key)
                    aw.current_image_path = inf_result.batch_key
                    aw._current_frame_idx = int(frame_idx)
                except Exception:
                    pass

                # Push the decoded frame bitmap to the OpenGL canvas
                if inf_result.q_image is not None:
                    try:
                        if getattr(aw, '_base_image_item', None) is not None:
                            aw._base_image_item.set_image(inf_result.q_image)
                    except Exception:
                        pass

                # Fast annotation overlay: new predictions + existing saved annotations
                try:
                    if getattr(aw, '_base_image_item', None) is not None:
                        paths_data = []
                        if inf_result.yolo_result is not None:
                            model_type = getattr(self._active_model_dialog, 'task', 'detect')
                            rp = getattr(self, '_results_processor', None)
                            if rp is not None:
                                try:
                                    paths_data.extend(
                                        rp.generate_fast_render_paths(inf_result.yolo_result, model_type))
                                except Exception:
                                    pass
                        try:
                            for ann in aw.get_image_annotations(inf_result.batch_key):
                                if (getattr(ann.label, 'is_visible', True)
                                        and not hasattr(ann, 'mask_data')):
                                    try:
                                        paths_data.append(
                                            (ann.get_painter_path(), ann.label.color, ann.transparency))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        aw._base_image_item.set_readonly_annotations(paths_data)
                except Exception as e:
                    print(f"Warning: fast render failed: {e}")

                # Update video scrub bar
                try:
                    if hasattr(aw, '_video_player') and aw._video_player:
                        vp = aw._video_player
                        vp.slider.blockSignals(True)
                        try:
                            vp.slider.setValue(aw._current_frame_idx)
                        finally:
                            vp.slider.blockSignals(False)
                        total_frames = getattr(
                            getattr(aw, '_active_video_raster', None), 'frame_count', 0)
                        vp.lbl_frame.setText(f"{aw._current_frame_idx} / {total_frames}")
                except Exception:
                    pass

        except Exception as e:
            print(f"on_item_processed error: {e}")
        finally:
            # Release the per-video-frame gate so the worker can proceed
            try:
                if (inf_result.is_video
                        and hasattr(self, '_batch_worker')
                        and self._batch_worker is not None):
                    self._batch_worker.release_frame_gate()
            except Exception:
                pass

    def _apply_sam_to_cache(self):
        """Run SAM over every image in the cache that holds list-style results.

        For tile results (work_area != None) SAM runs on the tile crop — boxes
        are in tile-space, orig_img is the tile BGR the worker already stashed.
        After SAM, the result is remapped to full-image coordinates so boxes
        and masks land in the same reference frame.  This keeps SAM's ViT
        encoder bounded by tile size instead of the full raster size.

        For whole-image results SAM runs once on the existing orig_img — no
        remap needed.

        Video frames are skipped (they use the raw YOLO detections).
        """
        active_dialog = getattr(self, '_active_model_dialog', None)
        if active_dialog is None:
            return

        # Only applicable to Detect/Segment dialogs with SAM enabled
        use_sam = (
            hasattr(active_dialog, 'use_sam_dropdown')
            and active_dialog.use_sam_dropdown.currentText() == "True"
            and hasattr(active_dialog, 'sam_dialog')
            and active_dialog.sam_dialog is not None
            and getattr(active_dialog.sam_dialog, 'loaded_model', None) is not None
        )
        if not use_sam:
            return

        cache = getattr(self.annotation_window, 'batch_results_cache', {})
        if not cache:
            return

        sam_dialog     = active_dialog.sam_dialog
        raster_manager = getattr(self.image_window, 'raster_manager', None)

        image_paths_for_sam = [
            p for p, c in cache.items()
            if isinstance(c, list) and c
        ]
        if not image_paths_for_sam:
            return

        from coralnet_toolbox.Results.MapResults import MapResults
        from torch.cuda import empty_cache as _empty_cache
        import gc as _gc

        # Mark as segmentation up front so any downstream baking uses the
        # correct processor path even if individual images fail SAM.
        try:
            active_dialog.task = 'segment'
        except Exception:
            pass

        # Inline baking per image: converts masks → annotation objects and
        # immediately drops the cache entry so full-image mask tensors don't
        # accumulate across the batch.  Without this, peak RAM scales with
        # total_images × per_image_masks; with it, peak is one image's worth.
        results_processor = getattr(self, '_results_processor', None)

        def _bake_and_drop(image_path, results_list):
            try:
                self.annotation_window.is_streaming_inference = True
            except Exception:
                pass
            try:
                if results_processor is not None:
                    results_processor.process_segmentation_results(results_list)
            except Exception as e:
                print(f"_apply_sam_to_cache bake error for {image_path}: {e}")
            try:
                self.image_window.update_image_annotations(
                    image_path, update_counts=False)
            except Exception:
                pass
            try:
                if image_path in cache:
                    del cache[image_path]
            except Exception:
                pass
            # Explicitly drop per-result references; masks can be huge.
            try:
                for r in results_list:
                    try:
                        r.masks = None
                        r.orig_img = None
                    except Exception:
                        pass
                results_list.clear()
            except Exception:
                pass

        sam_pb = ProgressBar(None, title="Applying SAM...")
        sam_pb.setWindowFlags(sam_pb.windowFlags() | Qt.WindowStaysOnTopHint)
        sam_pb.show()
        sam_pb.start_progress(len(image_paths_for_sam))
        QApplication.processEvents()

        for path in image_paths_for_sam:
            cached = cache.get(path) or []
            if not cached:
                sam_pb.update_progress()
                continue

            raster = raster_manager.get_raster(path) if raster_manager else None

            # Determine whether any tiles are present; if all work_areas are
            # None we can batch the full-image SAM call into a single ViT pass.
            tile_mode = any(getattr(r, '_tile_work_area_data', None) is not None
                            for r in cached)

            updated = []
            try:
                if tile_mode:
                    # Per-tile: each tile has a unique orig_img (tile crop), so
                    # the ViT re-encode happens per tile regardless of call
                    # shape.  We call SAM per result to keep the remap tied to
                    # the correct work_area.
                    for result in cached:
                        wa_data = getattr(result, '_tile_work_area_data', None)
                        wa = None
                        if wa_data is not None:
                            try:
                                from coralnet_toolbox.WorkArea import WorkArea
                                wa = WorkArea.from_dict(wa_data, path)
                            except Exception:
                                wa = None
                        try:
                            if (wa is not None
                                    and result.boxes is not None
                                    and len(result.boxes)):
                                sam_out = sam_dialog.predict_from_results(
                                    [result], path)
                                if sam_out and sam_out[0] is not None:
                                    result = sam_out[0]
                            if wa is not None and raster is not None:
                                result = MapResults().map_results_from_work_area(
                                    result, raster, wa,
                                    map_masks=True,
                                    task='segment',
                                    boundary_tolerance=self.thresholds_widget.get_boundary_tolerance(),
                                )
                        except Exception as e:
                            print(f"_apply_sam_to_cache tile error for {path}: {e}")
                        # Drop tile BGR + work_area ref so they can be reclaimed
                        try:
                            result.orig_img = None
                        except Exception:
                            pass
                        try:
                            result._tile_work_area_data = None
                        except Exception:
                            pass
                        updated.append(result)
                else:
                    # Whole-image results: orig_img was set by YOLO to the full
                    # image during inference, so SAM can run in a single batched
                    # call (ViT fires once thanks to the identity cache).
                    try:
                        sam_results = sam_dialog.predict_from_results(cached, path)
                    except Exception as e:
                        print(f"_apply_sam_to_cache full-image error for {path}: {e}")
                        sam_results = [None] * len(cached)
                    for r, s in zip(cached, sam_results):
                        out = s if s is not None else r
                        try:
                            out.orig_img = None
                        except Exception:
                            pass
                        try:
                            out._tile_work_area_data = None
                        except Exception:
                            pass
                        updated.append(out)

                # Drop the pre-SAM references held in `cached` so the only
                # live refs to the updated results are in `updated`.
                try:
                    cached.clear()
                except Exception:
                    pass

                _bake_and_drop(path, updated)

            except Exception as e:
                print(f"_apply_sam_to_cache error for {path}: {e}")
                # Leave the entry in the cache so _bake_cached_annotations can
                # make a second attempt; do not lose data on a failure path.
                cache[path] = updated
            finally:
                _gc.collect()
                _empty_cache()

            sam_pb.update_progress()
            QApplication.processEvents()

        sam_pb.close()

    def _bake_cached_annotations(self):
        """Bakes all cached tensors into real Annotation objects and saves them to the project.

        Handles both image results (stored as lists) and video-frame results (stored as
        single Results objects) via the unified ``batch_results_cache`` on the
        AnnotationWindow.  Safe to call even when the cache is empty.
        """
        # If the user has chosen not to save annotations for video runs,
        # skip baking entirely and clear any cached results. This protects
        # against stale caches from being unintentionally written.
        try:
            if hasattr(self, 'save_video_annotations') and not self.save_video_annotations:
                try:
                    if hasattr(self.annotation_window, 'batch_results_cache'):
                        self.annotation_window.batch_results_cache = {}
                except Exception:
                    pass
                return
        except Exception:
            pass

        if not hasattr(self, '_results_processor') or self._results_processor is None:
            return

        # ── Semantic: results were painted inline; just finalize stats and UI ───
        is_semantic_run = isinstance(
            getattr(self, '_active_model_dialog', None),
            type(None)
        )
        try:
            from coralnet_toolbox.MachineLearning.DeployModel.QtSemantic import Semantic as _Semantic
            is_semantic_run = isinstance(getattr(self, '_active_model_dialog', None), _Semantic)
        except Exception:
            is_semantic_run = False

        if is_semantic_run:
            processed = getattr(self, '_semantic_processed_images', set())
            raster_manager = getattr(self.image_window, 'raster_manager', None)
            for image_path in processed:
                try:
                    raster = raster_manager.get_raster(image_path) if raster_manager else None
                    if raster and raster.mask_annotation:
                        raster.mask_annotation.recalculate_class_statistics()
                    self.image_window.update_image_annotations(image_path)
                except Exception as e:
                    print(f"Semantic finalize error for {image_path}: {e}")
            # Reload the mask graphics for the currently displayed image
            try:
                if getattr(self.annotation_window, 'current_image_path', None) in processed:
                    self.annotation_window.load_mask_annotation()
            except Exception:
                pass
            self._semantic_processed_images = set()
            return

        cache = getattr(self.annotation_window, 'batch_results_cache', {})

        # Fall through even when cache is empty so the final master sync still
        # fires — _apply_sam_to_cache may have baked (and dropped) every entry
        # inline to keep peak RAM bounded to one image's worth of masks.
        if cache:
            bake_pb = ProgressBar(None, title="Saving Annotations to Project...")
            bake_pb.setWindowFlags(bake_pb.windowFlags() | Qt.WindowStaysOnTopHint)
            bake_pb.show()
            bake_pb.start_progress(len(cache))
            QApplication.processEvents()  # ensure bar is painted before first heavy bake call

            # Suppress O(N²) UI updates during the loop
            self.annotation_window.is_streaming_inference = True

            is_segmentation = getattr(self._active_model_dialog, 'task', '') == 'segment'

            for path, cached_results in cache.items():
                # Skip non-Results overlay entries (e.g., dict overlays for masks)
                if isinstance(cached_results, dict):
                    # Count it as processed for the progress bar and continue
                    bake_pb.update_progress()
                    QApplication.processEvents()
                    continue

                # Normalise: images store lists, video frames store single Results objects
                results_to_process = cached_results if isinstance(cached_results, list) else [cached_results]
                try:
                    if is_segmentation:
                        self._results_processor.process_segmentation_results(results_to_process)
                    else:
                        self._results_processor.process_detection_results(results_to_process)
                except Exception as e:
                    print(f"Error hydrating cached results for {path}: {e}")

                bake_pb.update_progress()
                QApplication.processEvents()  # Keep UI responsive

            bake_pb.close()

        # Re-enable UI updates and trigger one final master sync
        self.annotation_window.is_streaming_inference = False
        try:
            self.main_window.label_window.update_annotation_count()
            for path in cache.keys():
                self.image_window.update_image_annotations(path, update_counts=False)
        except Exception:
            pass

        # Clear the unified cache
        self.annotation_window.batch_results_cache = {}

        # Repopulate the phantom layer with the freshly-baked real annotations
        # (replaces ghost paths immediately so annotations are visible without a click)
        try:
            self.annotation_window.refresh_phantom_annotations()
        except Exception:
            pass

    def _on_worker_finished(self):
        """Cleanup UI and restore button behavior when worker finishes."""

        # 0. Defensive cursor reset — clear any override cursor that was left on
        # the stack by batch_inference() or predict() paths that returned early.
        try:
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()
        except Exception:
            pass
        
        # 1. Close and clean up the video streaming progress bar
        pb = getattr(self, '_progress_bar', None)
        if pb is not None:
            try:
                pb.finish_progress()
                pb.stop_progress()
                pb.close()
            except Exception:
                pass
            self._progress_bar = None
        QApplication.processEvents()  # flush bar close before SAM / bake bars appear

        # 2. SAM post-processing pass (Detect/Segment with SAM enabled, non-video images only)
        #    Runs once per image on all collected tile results before baking so SAM's ViT
        #    encoder fires only once per source image regardless of tile count.
        self._apply_sam_to_cache()

        # 3. BAKE ALL CACHED FRAMES (images + video, unified)
        self._bake_cached_annotations()

        # 4. Turn off streaming mode in the annotation window (failsafe)
        try:
            self.annotation_window.is_streaming_inference = False
        except Exception:
            pass

        # 4. Unlock the rest of the dialog UI
        try:
            self.set_ui_processing_state(False)
        except Exception:
            pass

        # 4b. Restore main window tools that were locked for streaming inference
        try:
            try:
                self.main_window.set_video_playback_tools_enabled(True)
            except Exception:
                pass
        except Exception:
            pass

        # 5. Restore the Apply button routing and text
        try:
            try:
                self.button_box.accepted.disconnect()
            except TypeError:
                pass  # Ignore if not connected
            
            self.button_box.accepted.connect(self.apply)
            ok_btn = self.button_box.button(QDialogButtonBox.Ok)
            if ok_btn:
                ok_btn.setText("Apply")
                ok_btn.setEnabled(True)
                try:
                    ok_btn.setStyleSheet("")
                except Exception:
                    pass
        except Exception:
            pass

        # 6. Trigger a full frame redraw on the exact frame we stopped on
        try:
            if getattr(self.annotation_window, '_active_video_raster', None) is not None:
                current_frame = getattr(self.annotation_window, '_current_frame_idx', None)
                if current_frame is not None:
                    self.annotation_window._display_video_frame(current_frame)

                # For Semantic video: reload the mask graphics item so it persists
                # after the frame display clears and rebuilds the scene.
                try:
                    from coralnet_toolbox.MachineLearning.DeployModel.QtSemantic import Semantic as _Sem
                    if isinstance(getattr(self, '_active_model_dialog', None), _Sem):
                        self.annotation_window.load_mask_annotation()
                except Exception:
                    pass

                # Refresh the scrub bar so the new annotation tick marks appear all at once
                self.annotation_window._update_video_annotation_marks()
        except Exception:
            pass

        # 7. Auto-close the dialog once inference and baking are complete
        try:
            self.accept()
        except Exception:
            pass

        # 8. Safely clean up the worker thread
        if hasattr(self, '_batch_worker') and self._batch_worker is not None:
            try:
                self._batch_worker.deleteLater()
            except Exception:
                pass
            self._batch_worker = None

    def _on_stop_inference_clicked(self):
        """Called when the user clicks 'Stop Inference' (OK button remapped)."""
        try:
            if hasattr(self, '_batch_worker') and getattr(self._batch_worker, 'isRunning', lambda: False)():
                try:
                    self._batch_worker.stop()
                except Exception:
                    pass
                try:
                    ok_btn = self.button_box.button(QDialogButtonBox.Ok)
                    if ok_btn:
                        ok_btn.setText('Stopping...')
                        ok_btn.setEnabled(False)
                except Exception:
                    pass
            else:
                # Fallback to default apply behavior
                try:
                    self.apply()
                except Exception:
                    pass
        except Exception:
            pass

    def set_ui_processing_state(self, processing: bool):
        """Enable/disable UI widgets while processing is active.

        This is a minimal helper used to re-enable controls after streaming inference.
        """
        try:
            enabled = not processing
            try:
                self.model_combo.setEnabled(enabled)
            except Exception:
                pass
            try:
                self.inference_type_combo.setEnabled(enabled)
            except Exception:
                pass

            # Thresholds and task-specific controls
            # Do NOT disable thresholds_widget here; keep thresholds editable
            # during streaming inference so users can tune values live.
            try:
                if hasattr(self, 'task_specific_group') and self.task_specific_group is not None:
                    self.task_specific_group.setEnabled(enabled)
            except Exception:
                pass
            try:
                if hasattr(self, 'video_group') and self.video_group is not None:
                    self.video_group.setEnabled(enabled)
            except Exception:
                pass

            # Disable/enable the Cancel button explicitly (Ok/Stop handled elsewhere)
            try:
                cancel_btn = self.button_box.button(QDialogButtonBox.Cancel)
                if cancel_btn:
                    cancel_btn.setEnabled(enabled)
            except Exception:
                pass

            # Also disable the video player widget so playback controls can't be used
            try:
                if hasattr(self.annotation_window, '_video_player') and self.annotation_window._video_player is not None:
                    self.annotation_window._video_player.setEnabled(enabled)
            except Exception:
                pass

            # Mirror this for the main window toolbar actions related to video playback
            try:
                self.main_window.set_video_playback_tools_enabled(enabled)
            except Exception:
                pass
        except Exception:
            pass

    def cleanup(self):
        """
        Clean up resources after batch inference.
        """
        # Ensure any running worker is signaled to stop
        try:
            if hasattr(self, '_batch_worker') and getattr(self._batch_worker, 'isRunning', lambda: False)():
                try:
                    self._batch_worker.stop()
                except Exception:
                    pass
        except Exception:
            pass
        self.image_paths = []
        
        # Reset inference type to Standard
        self.inference_type_combo.blockSignals(True)
        self.inference_type_combo.setCurrentText("Standard")
        self.inference_type_combo.blockSignals(False)
        
        # Untoggle all tools in the annotation window
        self.annotation_window.toolChanged.emit(None)
