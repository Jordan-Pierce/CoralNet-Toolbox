import time
import threading
import warnings

import os
from concurrent.futures import ThreadPoolExecutor

import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox,
                             QFormLayout, QComboBox, QHBoxLayout,
                             QSpinBox, QPushButton, QTabWidget, QWidget)

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
    stageChanged = pyqtSignal(str)          # emits a short human-readable status message
    timingSummary = pyqtSignal(object)      # emits aggregate timing dict
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, items, initial_thresholds,
                 device=None, task='detect', batch_size=16, imgsz=None,
                 is_semantic=False, sam_enabled=False, live_preview=True,
                 model_call_overrides=None, collapse_classes=False,
                 result_names=None, supports_tensor_input=True,
                 parent=None):
        super().__init__(parent)
        self.model = model
        self.items = list(items)
        self.device = device
        self._task = task
        self._batch_size = max(1, int(batch_size))
        self._imgsz = imgsz
        self._is_semantic = is_semantic
        # Per-task tweaks for non-YOLO ultralytics models (SAM, YOLOE).  These
        # all still produce Results objects, so the worker pipeline is shared:
        #   - model_call_overrides: merged over the default predict() kwargs
        #     (e.g. visual_prompts=[] for YOLOE, quantize=32 for MobileSAM).
        #   - collapse_classes: collapse every detected class id to 0 (the
        #     single-class generators output one logical class).
        #   - result_names: dict assigned to result.names after inference.
        #   - supports_tensor_input: SAM/YOLOE preprocess pixels themselves, so
        #     they take BGR frames rather than the preprocessed BCHW tensor the
        #     live-preview video path builds for plain YOLO.
        self._model_call_overrides = dict(model_call_overrides or {})
        self._collapse_classes = bool(collapse_classes)
        self._result_names = dict(result_names) if result_names else None
        self._supports_tensor_input = bool(supports_tensor_input)
        # When SAM is enabled, skip tile→full-image remap in the worker so
        # _apply_sam_to_cache can run SAM on the tile crop (tile-space boxes
        # + tile orig_img) and then remap boxes and masks together afterwards.
        self._sam_enabled = bool(sam_enabled)
        # Live preview gates the worker on each painted video frame.  When
        # off, video frames batch like images and previews are throttled.
        self._live_preview = bool(live_preview)
        self._last_preview_time = 0.0
        self._is_running = True
        self._waiting_for_ui = False
        self._mutex = QMutex()
        # Serialises rasterio work-area reads from the decode pool.
        self._rasterio_lock = threading.Lock()
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

    def _decode_source_eager(self, item):
        """Pool-side decode of one NON-VIDEO item to a BGR array.

        Decoding file paths here (instead of inside YOLO's predictor) lets
        several images decode in parallel and overlap the GPU call for the
        previous batch.  rasterio handles are not thread-safe, so work-area
        reads are serialised; they still overlap GPU work.
        """
        try:
            if isinstance(item.source, str):
                img = cv2.imread(item.source)
                # Fallback: let YOLO try the path itself (e.g. exotic TIFFs).
                return img if img is not None else item.source
            with self._rasterio_lock:
                return item.raster.get_work_area_data(item.source, as_format='BGR')
        except Exception:
            return np.zeros((640, 640, 3), dtype=np.uint8)

    def _batch_bounds(self, i, total):
        """Exclusive end index of the batch starting at items[i].

        Video items run one at a time (per-frame UI gate); consecutive
        non-video items batch up to _batch_size.
        """
        if self.items[i].is_video and self._live_preview:
            return i + 1
        end = i
        is_video = self.items[i].is_video
        while (end < min(i + self._batch_size, total)
               and self.items[end].is_video == is_video):
            end += 1
        return max(end, i + 1)

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

    def _warmup_model(self):
        """Run one tiny inference so fuse/compile/cudnn-autotune cost lands
        before the progress bar starts instead of inside the first batch."""
        try:
            kwargs = dict(device=self.device, quantize=16, verbose=False)
            if self._imgsz is not None:
                kwargs["imgsz"] = self._imgsz
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            list(self.model(dummy, **kwargs))
        except Exception:
            pass

    def run(self):
        """Inference loop executed on the worker thread."""
        try:
            # NOTE: do NOT set torch.backends.cudnn.benchmark here.  Ultralytics
            # predicts with rect=True, so the input tensor shape changes with
            # every image aspect ratio and batch size; cuDNN then re-runs its
            # exhaustive autotune per new shape (~10-17 s each on an RTX 5090),
            # which made batch runs ~200x slower than single-image inference.
            self._warmup_model()
            decode_pool = ThreadPoolExecutor(
                max_workers=min(8, (os.cpu_count() or 4)))
            prefetch = None  # (start_idx, end_idx, [futures]) for the next batch
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

                batch_end = self._batch_bounds(i, total)
                batch = self.items[i:batch_end]

                # Let the UI know what's about to happen, since progressUpdated
                # only fires once this batch's decode+inference finishes — for
                # the first batch (or a slow batch) that would otherwise leave
                # the progress dialog looking stuck at 0.
                try:
                    start_name = os.path.basename(batch[0].image_path)
                except Exception:
                    start_name = ""
                self.stageChanged.emit(
                    f"Running inference {i + 1}-{batch_end} of {total}: {start_name}")

                # Decode the batch.  Non-video batches may already be in
                # flight from last iteration's prefetch; video frames decode
                # synchronously here (the VideoCapture is not thread-safe).
                inputs = []
                video_q_images = []
                decode_start = time.perf_counter()
                if batch[0].is_video and self._live_preview and self._supports_tensor_input:
                    for item in batch:
                        try:
                            model_input, q_image = self._decode_video_source(item)
                            inputs.append(model_input)
                            video_q_images.append(q_image)
                        except Exception:
                            inputs.append(np.zeros((640, 640, 3), dtype=np.uint8))
                            video_q_images.append(None)
                elif batch[0].is_video:
                    # Fast mode: sequential BGR decode on this thread (the
                    # capture is not thread-safe), preview at most ~10 Hz.
                    for item in batch:
                        bgr = None
                        try:
                            bgr = item.raster.get_bgr_frame(item.source)
                        except Exception:
                            bgr = None
                        if bgr is None:
                            bgr = np.zeros((640, 640, 3), dtype=np.uint8)
                        inputs.append(bgr)
                        now = time.monotonic()
                        if now - self._last_preview_time >= 0.1:
                            self._last_preview_time = now
                            try:
                                from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                                video_q_images.append(VideoRaster._bgr_to_qimage(bgr))
                            except Exception:
                                video_q_images.append(None)
                        else:
                            video_q_images.append(None)
                else:
                    futures = None
                    if (prefetch is not None
                            and prefetch[0] == i and prefetch[1] == batch_end):
                        futures = prefetch[2]
                    if futures is None:
                        futures = [decode_pool.submit(self._decode_source_eager, item)
                                   for item in batch]
                    for future in futures:
                        try:
                            inputs.append(future.result())
                        except Exception:
                            inputs.append(np.zeros((640, 640, 3), dtype=np.uint8))
                        video_q_images.append(None)
                prefetch = None

                # Kick off decode of the next batch so it overlaps this
                # batch's GPU call.
                if batch_end < total and not self.items[batch_end].is_video:
                    next_end = self._batch_bounds(batch_end, total)
                    prefetch = (
                        batch_end, next_end,
                        [decode_pool.submit(self._decode_source_eager, item)
                         for item in self.items[batch_end:next_end]],
                    )
                decode_seconds = time.perf_counter() - decode_start

                # Run YOLO on the mini-batch
                # The BCHW tensor path expects RGB from get_frame_for_inference;
                # fast-mode video uses BGR ndarrays, which YOLO handles itself.
                use_tensor_path = (len(batch) == 1 and batch[0].is_video
                                   and self._live_preview
                                   and self._supports_tensor_input)
                model_source = self._prepare_video_input(inputs[0]) if use_tensor_path else inputs
                inference_start = time.perf_counter()
                _model_kwargs = dict(
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    # Ultralytics defaults to batch=1, which turns a list of N
                    # images into N sequential inferences; pass the real batch
                    # size so the GPU sees one batched forward pass.
                    batch=len(batch),
                    device=self.device,
                    retina_masks=self._is_semantic,
                    quantize=16,
                    agnostic_nms=True,
                    stream=True,
                    verbose=False,
                )
                if self._imgsz is not None:
                    _model_kwargs["imgsz"] = self._imgsz
                # Per-task overrides last so they win (e.g. YOLOE visual_prompts,
                # MobileSAM quantize=32, segment-task retina_masks).
                if self._model_call_overrides:
                    _model_kwargs.update(self._model_call_overrides)
                try:
                    results = list(self.model(model_source, **_model_kwargs))
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

                        # Single-class generators (SAM, See Anything) emit one
                        # logical class; collapse every detected class id to 0
                        # and stamp the project label name.  Done before the
                        # tile remap / SAM pass so downstream stages see class 0.
                        if self._collapse_classes:
                            try:
                                if result.boxes is not None and len(result.boxes):
                                    new_data = result.boxes.data.clone()
                                    new_data[:, 5] = 0
                                    result.boxes.data = new_data
                            except Exception:
                                pass
                        if self._result_names is not None:
                            try:
                                result.names = dict(self._result_names)
                            except Exception:
                                pass

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

                        # Release the full-resolution source pixels unless the
                        # SAM post-pass needs them.  SAM skips video frames, so
                        # those can always be freed.  Semantic results are
                        # consumed immediately in on_item_processed and are
                        # left untouched here out of caution.
                        if (not self._is_semantic
                                and (not self._sam_enabled or item.is_video)):
                            try:
                                result.orig_img = None
                            except Exception:
                                pass

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
                        if item.is_video and self._live_preview:
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
                decode_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
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
    Supports: Classify, Detect, Segment, Semantic, SAM, SeeAnything, Feature.

    Detect and Segment tasks (including video and tiled variants) are routed
    through the async BatchInferenceWorker for maximum throughput.
    Classify, Semantic, SAM, and SeeAnything use their own
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
        self.resize(500, 500)
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

        # Short, lightweight intro line (no heavyweight group box).
        self.setup_info_layout()

        # Two-tab layout: "Settings" holds the model/source-dependent controls
        # (sections are shown/hidden by _update_visible_sections rather than
        # greyed out); "Thresholds" holds the threshold widget.
        self.tabs = QTabWidget()

        self.settings_tab = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_tab)
        self.thresholds_tab = QWidget()
        self.thresholds_layout = QVBoxLayout(self.thresholds_tab)

        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.thresholds_tab, "Thresholds")
        # Stretch factor 1 so the tab area absorbs extra height; the button
        # row below keeps its natural size instead of the tabs being truncated.
        self.layout.addWidget(self.tabs, 1)

        # Setup sections into their respective tabs
        self.setup_options_layout()
        self.setup_task_specific_layout()
        self.setup_thresholds_layout()
        self.settings_layout.addStretch()
        self.thresholds_layout.addStretch()

        self.setup_buttons_layout()

    def _refresh_model_dialog_references(self):
        """Refresh cached dialog references from MainWindow."""
        self.classify_dialog = getattr(self.main_window, 'classify_deploy_model_dialog', None)
        self.detect_dialog = getattr(self.main_window, 'detect_deploy_model_dialog', None)
        self.segment_dialog = getattr(self.main_window, 'segment_deploy_model_dialog', None)
        self.semantic_dialog = getattr(self.main_window, 'semantic_deploy_model_dialog', None)
        self.sam_dialog = getattr(self.main_window, 'sam_deploy_generator_dialog', None)
        self.see_anything_dialog = getattr(self.main_window, 'see_anything_deploy_generator_dialog', None)
        self.feature_dialog = getattr(self.main_window, 'feature_deploy_model_dialog', None)

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
        Set up a single-line intro label (no group box) to keep the dialog compact.
        """
        info_label = QLabel("Perform batch inferencing on the selected rasters.\nSelect images via right-click in the Image window.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        info_label.setToolTip("Run a deployed model on multiple selected images to generate automatic predictions.\nResults can be saved directly to the project.")
        self.layout.addWidget(info_label)

    def setup_options_layout(self):
        """
        Combined Options: Model, Inference Type, and Save Annotations.
        """
        group_box = QGroupBox("Options")
        form_layout = QFormLayout()

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.model_combo.setToolTip("Select which deployed model to use for inference.\nEnsure the model is loaded in the respective deployment dialog.")
        form_layout.addRow("Model:", self.model_combo)

        # Inference type selection
        self.inference_type_combo = QComboBox()
        self.inference_type_combo.addItem("Standard")
        self.inference_type_combo.addItem("Tiled")
        self.inference_type_combo.currentTextChanged.connect(self.on_inference_type_changed)
        self.inference_type_combo.setToolTip("Standard: Run inference on full images.\nTiled: Split images into tiles for better handling of large images or GPU memory constraints.")
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
        self.save_annotations_combo.setToolTip("Save generated annotations to the project.\nTrue: automatically save predictions. False: preview only.")
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

        # Keep handles on the rows that are hidden for some models so
        # _update_visible_sections can show/hide them per task.
        self._options_form = form_layout
        self.save_annotations_combo_label = form_layout.labelForField(self.save_annotations_combo)
        self.batch_size_spin_label = form_layout.labelForField(self.batch_size_spin)

        group_box.setLayout(form_layout)
        self.settings_layout.addWidget(group_box)

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
        self.thresholds_layout.addWidget(self.thresholds_widget)

    def setup_task_specific_layout(self):
        """
        Override in subclasses to add task-specific options.
        For Classify model, adds annotation selection options (review vs all).
        Visibility is controlled by on_model_changed().
        """
        # Create a group box for Classify-specific annotation options
        group_box = QGroupBox("Classify Options")
        layout = QFormLayout()

        # Mutually-exclusive True/False dropdowns (replaces the old radio checkboxes)
        self.review_combo = QComboBox()
        self.review_combo.addItems(["True", "False"])
        self.review_combo.setCurrentText("True")
        self.review_combo.setToolTip("Generate predictions for Review annotations.\nTrue: Classify model will predict labels for annotations marked as Review.")
        layout.addRow("Predict Review Annotations:", self.review_combo)

        self.all_combo = QComboBox()
        self.all_combo.addItems(["True", "False"])
        self.all_combo.setCurrentText("False")
        self.all_combo.setToolTip("Generate predictions for ALL annotations.\nTrue: Classify model will predict labels for all annotations (including already-verified).")
        layout.addRow("Predict All Annotations:", self.all_combo)

        # Keep the two options mutually exclusive, like the old radio checkboxes
        self.review_combo.currentTextChanged.connect(self._on_review_combo_changed)
        self.all_combo.currentTextChanged.connect(self._on_all_combo_changed)

        group_box.setLayout(layout)
        self.task_specific_group = group_box
        self.settings_layout.addWidget(group_box)

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

        # Live preview toggle — False trades the per-frame canvas
        # preview for batched GPU inference (much faster on long ranges).
        self.live_preview_combo = QComboBox()
        self.live_preview_combo.addItems(["True", "False"])
        self.live_preview_combo.setCurrentText("True")
        self.live_preview_combo.setToolTip("Set to False for max speed (no per-frame preview)")

        # Keyframes-only filter: restrict inference to frames the user starred
        self.video_keyframes_combo = QComboBox()
        self.video_keyframes_combo.addItems(["True", "False"])
        self.video_keyframes_combo.setCurrentText("False")
        self.video_keyframes_combo.setToolTip(
            "Set to True to restrict inference to frames marked as keyframes "
            "(within the range/stride above)"
        )

        # Reset button
        self.reset_video_range_btn = QPushButton("Reset to Full Video")

        video_layout.addRow("Start Frame:", start_h)
        video_layout.addRow("End Frame:", end_h)
        video_layout.addRow("Every N Frames:", stride_h)
        video_layout.addRow("Keyframes Only:", self.video_keyframes_combo)
        video_layout.addRow("Live Preview:", self.live_preview_combo)
        video_layout.addRow("", self.reset_video_range_btn)

        video_box.setLayout(video_layout)
        self.video_group = video_box
        self.settings_layout.addWidget(video_box)

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
        if self.feature_dialog and getattr(self.feature_dialog, "loaded_model", None):
            self.model_dialogs["Feature"] = self.feature_dialog

        self.update_model_combo()

    def update_model_combo(self):
        """
        Update the model dropdown with available models.
        Preserves the last selected model if it's still available.
        """
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_keys = []

        # Add sorted models
        for key in sorted(self.model_dialogs.keys()):
            self.model_combo.addItem(key)
            self.model_keys.append(key)

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

            # Drive section visibility and the Type combo from (task x source).
            self._update_visible_sections()

    def _update_visible_sections(self):
        """Show only the controls that apply to the current model and source.

        Sections are shown/hidden (not greyed) based on:
          - the selected model (Classify / Detect / Semantic / ...), and
          - whether any highlighted raster is a VideoRaster.

        This is the single place that decides dialog layout, so it is safe to
        call from both on_model_changed and update_highlighted_images.
        """
        model = getattr(self, 'current_selected_model', None)

        # --- Task-specific (Classify) block: only relevant for Classify ---
        is_classify = (model == "Classify")
        if hasattr(self, 'task_specific_group'):
            self.task_specific_group.setVisible(is_classify)
            self.task_specific_group.setEnabled(is_classify)

        # --- Save Annotations / Batch Size: hidden for models that ignore them ---
        # Feature manages its own output; Classify writes directly, so the
        # worker-oriented controls don't apply to them.
        wants_worker_opts = model in ("Detect", "Segment", "Semantic", "SAM", "See Anything")
        if hasattr(self, 'save_annotations_combo'):
            self.save_annotations_combo.setVisible(wants_worker_opts)
            if getattr(self, 'save_annotations_combo_label', None):
                self.save_annotations_combo_label.setVisible(wants_worker_opts)
        if hasattr(self, 'batch_size_spin'):
            self.batch_size_spin.setVisible(wants_worker_opts)
            if getattr(self, 'batch_size_spin_label', None):
                self.batch_size_spin_label.setVisible(wants_worker_opts)

        # --- Video block: only when at least one highlighted raster is video ---
        has_video = self._highlighted_has_video()
        if hasattr(self, 'video_group'):
            self.video_group.setVisible(has_video)

        # --- Type combo: tiling only applies to image rasters on tiling models.
        # VideoRaster has no "Tiled" option, so force Standard and grey it out.
        model_supports_tiling = model not in ("Classify", "Feature")
        type_enabled = model_supports_tiling and not has_video
        if hasattr(self, 'inference_type_combo'):
            if not type_enabled and self.inference_type_combo.currentText() != "Standard":
                self.inference_type_combo.setCurrentText("Standard")
            self.inference_type_combo.setEnabled(type_enabled)
    
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
        # Feature doesn't use any thresholds
        elif model_name in ("Feature",):
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

    def _highlighted_has_video(self):
        """Return True if any highlighted raster is a VideoRaster."""
        try:
            raster_manager = getattr(self.image_window, 'raster_manager', None)
            if raster_manager is None:
                return False
            for p in self.highlighted_images:
                try:
                    raster = raster_manager.get_raster(p)
                except Exception:
                    continue
                if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                    return True
        except Exception:
            pass
        return False

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
                        # Group is hidden entirely by _update_visible_sections.
                        pass
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

            # Source changed → re-evaluate which sections apply (e.g. Video).
            self._update_visible_sections()
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

    def _on_review_combo_changed(self, text):
        """Keep the Review/All annotation dropdowns mutually exclusive."""
        if text == "True" and self.all_combo.currentText() == "True":
            self.all_combo.setCurrentText("False")

    def _on_all_combo_changed(self, text):
        """Keep the Review/All annotation dropdowns mutually exclusive."""
        if text == "True" and self.review_combo.currentText() == "True":
            self.review_combo.setCurrentText("False")

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
                               video_start=0, video_end=None, video_stride=1,
                               keyframes_only=False):
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
            keyframes_only=keyframes_only,
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
            # Prepare the annotation window for streaming inference: prefer
            # the new _prepare_scene_for_streaming helper (clears scene and
            # installs a fresh base image item). Fall back to the older
            # per-annotation clear method when the helper is not available.
            try:
                if hasattr(self.annotation_window, '_prepare_scene_for_streaming'):
                    self.annotation_window._prepare_scene_for_streaming()
                elif hasattr(self.annotation_window, '_clear_current_frame_annotation_graphics'):
                    self.annotation_window._clear_current_frame_annotation_graphics()
            except Exception:
                try:
                    if hasattr(self.annotation_window, '_clear_current_frame_annotation_graphics'):
                        self.annotation_window._clear_current_frame_annotation_graphics()
                except Exception:
                    pass
        except Exception:
            pass

        self._batch_worker.itemProcessed.connect(self.on_item_processed)
        self._batch_worker.progressUpdated.connect(self._on_worker_progress)
        self._batch_worker.stageChanged.connect(self._on_worker_stage)
        self._batch_worker.timingSummary.connect(self._on_worker_timing_summary)
        self._batch_worker.error.connect(
            lambda msg: print(f"BatchInferenceWorker error: {msg}"))
        self._batch_worker.finished.connect(self._on_worker_finished)

        try:
            self._update_worker_thresholds()
        except Exception:
            pass

        try:
            progress_bar.cancel_button.setEnabled(True)
            progress_bar.cancel_button.clicked.connect(
                lambda checked=False: self._batch_worker.stop()
                    if hasattr(self, '_batch_worker') and
                    getattr(self._batch_worker, 'isRunning', lambda: False)()
                    else None)
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
        BatchInferenceWorker. Classify, SAM, SeeAnything, and Feature use
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

        # The progress bar runs modally on the main thread; hide this dialog so
        # only the progress bar (and the live canvas preview) are visible while
        # processing. Every completion path calls accept()/cleanup(), so the
        # dialog never needs to be re-shown.
        try:
            self.hide()
        except Exception:
            pass

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

        # Let the task finalize model-dialog state (class_mapping, task, imgsz,
        # VPE prompts) before the results processor / worker are built.  A
        # False return aborts cleanly (e.g. See Anything without a reference).
        try:
            setup_ok = task_runner.setup_model_dialog()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model setup failed: {str(e)}")
            setup_ok = False
        if not setup_ok:
            QApplication.restoreOverrideCursor()
            try:
                progress_bar.close()
            except Exception:
                pass
            try:
                self.show()
            except Exception:
                pass
            return

        # Rebuild the results processor now that class_mapping is finalized
        # (SAM / See Anything set theirs in setup_model_dialog above).
        try:
            from coralnet_toolbox.Results import ResultsProcessor
            self._results_processor = ResultsProcessor(
                self.main_window,
                getattr(model_dialog, 'class_mapping', {}) or {},
            )
        except Exception:
            pass

        try:
            # ── Detect / Segment / Semantic / SAM / See Anything → unified async worker ──
            if selected_model in ("Detect", "Segment", "Semantic", "SAM", "See Anything"):
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

            # ── Feature ───────────────────────────────────────────────────────
            elif selected_model == "Feature":
                if not task_runner.run(progress_bar):
                    try:
                        progress_bar.close()
                    except Exception:
                        pass
                    return

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
            # The dialog was hidden before processing; re-show it on failure so
            # the user isn't left with an orphaned hidden window.
            try:
                self.show()
            except Exception:
                pass
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

    def _on_worker_stage(self, text):
        """Update the progress dialog's message label with the current stage (main thread)."""
        try:
            pb = getattr(self, '_progress_bar', None)
            if pb is not None and hasattr(pb, 'message_label'):
                pb.message_label.setText(text)
                QApplication.processEvents()
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
                            # The live-preview decode path runs the model on a
                            # preprocessed (e.g. 1024x1024) tensor, so retina_masks /
                            # results.orig_shape yield a mask at MODEL resolution, not
                            # the native frame size. The shared per-frame edit buffer
                            # (vr.mask_annotation.mask_data) is native-resolution, so a
                            # mismatched mask_arr is silently dropped by the shape guard
                            # in _restore_video_frame_mask_data — the cached prediction
                            # never becomes editable and the first brush stroke wipes it.
                            # Resize to native frame size so cache, overlay and edit
                            # buffer all agree. (Fast/BGR mode already matches.)
                            if (reconstructed is not None and reconstructed.size
                                    and raster is not None):
                                try:
                                    target_h = int(raster.height)
                                    target_w = int(raster.width)
                                    if reconstructed.shape[:2] != (target_h, target_w):
                                        import cv2 as _cv2
                                        reconstructed = _cv2.resize(
                                            reconstructed, (target_w, target_h),
                                            interpolation=_cv2.INTER_NEAREST,
                                        )
                                except Exception:
                                    pass
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
                                        tmp_mask._ensure_canvas()
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

                # Fast annotation overlay: new predictions only.
                # Existing saved annotations are intentionally excluded here to
                # prevent them from flashing on the first frame right after
                # _prepare_scene_for_streaming cleared the scene.  They reload
                # correctly via _display_video_frame when inference pauses/stops.
                try:
                    live = (getattr(self, 'live_preview_combo', None) is None
                            or self.live_preview_combo.currentText() == "True")
                    if ((live or inf_result.q_image is not None)
                            and getattr(aw, '_base_image_item', None) is not None):
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

        images_since_cleanup = 0
        for path in image_paths_for_sam:
            try:
                sam_pb.message_label.setText(f"Applying SAM: {os.path.basename(str(path))}")
            except Exception:
                pass

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
                # Full GC + CUDA allocator flush are expensive (each forces a
                # device sync); amortise them instead of paying per image.
                images_since_cleanup += 1
                if images_since_cleanup >= 25:
                    images_since_cleanup = 0
                    _gc.collect()
                    _empty_cache()

            sam_pb.update_progress()
            QApplication.processEvents()

        _gc.collect()
        _empty_cache()
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

            # Video semantic batch inference: _semantic_processed_images is never
            # populated for video frames (results go to batch_results_cache instead).
            # Scan the cache for ::frame_ entries and update the table count for each
            # unique video path so the RasterTable reflects the annotated frame count.
            try:
                cache = getattr(self.annotation_window, 'batch_results_cache', {}) or {}
                video_paths_to_update = set()
                for key, cached in cache.items():
                    if isinstance(key, str) and '::frame_' in key and isinstance(cached, dict):
                        video_paths_to_update.add(key.rsplit('::frame_', 1)[0])
                for vpath in video_paths_to_update:
                    try:
                        self.image_window.update_image_annotations(vpath)
                    except Exception as e:
                        print(f"Semantic video count update error for {vpath}: {e}")
            except Exception:
                pass

            return

        cache = getattr(self.annotation_window, 'batch_results_cache', {})

        # Fall through even when cache is empty so the final master sync still
        # fires — _apply_sam_to_cache may have baked (and dropped) every entry
        # inline to keep peak RAM bounded to one image's worth of masks.
        all_baked_annotations = []
        if cache:
            bake_pb = ProgressBar(None, title="Saving Annotations to Project...")
            bake_pb.setWindowFlags(bake_pb.windowFlags() | Qt.WindowStaysOnTopHint)
            bake_pb.show()
            bake_pb.start_progress(len(cache))
            QApplication.processEvents()  # ensure bar is painted before first heavy bake call

            # Suppress O(N²) UI updates during the loop
            self.annotation_window.is_streaming_inference = True

            is_segmentation = getattr(self._active_model_dialog, 'task', '') == 'segment'
            build_fn = (self._results_processor.build_segmentation_annotations
                        if is_segmentation
                        else self._results_processor.build_detection_annotations)

            # Accumulate all annotations across every image before committing so
            # add_annotations only toggles the scene index once for the entire batch.
            _last_pump = time.monotonic()

            for path, cached_results in cache.items():
                try:
                    bake_pb.message_label.setText(f"Saving annotations: {os.path.basename(str(path))}")
                except Exception:
                    pass

                # Skip non-Results overlay entries (e.g., dict overlays for masks)
                if not isinstance(cached_results, dict):
                    # Normalise: images store lists, video frames store single Results objects
                    results_to_process = (cached_results
                                          if isinstance(cached_results, list)
                                          else [cached_results])
                    try:
                        all_baked_annotations.extend(build_fn(results_to_process))
                    except Exception as e:
                        print(f"Error hydrating cached results for {path}: {e}")

                bake_pb.update_progress()

                # Throttle UI pumps to at most ~10 Hz to keep the bar moving
                # without spending more time in processEvents than in annotation work.
                _now = time.monotonic()
                if _now - _last_pump >= 0.1:
                    QApplication.processEvents()
                    _last_pump = _now

            # Commit the entire batch in one shot
            if all_baked_annotations:
                self.annotation_window.add_annotations(all_baked_annotations)
                # Reload graphics for the currently visible image if it was in the batch
                try:
                    cur = self.annotation_window.current_image_path
                    if any(a.image_path == cur for a in all_baked_annotations):
                        self.annotation_window.load_annotations()
                except Exception:
                    pass

            bake_pb.close()

        # Re-enable UI updates and trigger one final master sync
        self.annotation_window.is_streaming_inference = False
        annotated_paths = {a.image_path for a in all_baked_annotations} if cache else set()
        try:
            self.main_window.label_window.update_annotation_count()
            for path in annotated_paths:
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
        try:
            if hasattr(self, '_batch_worker') and getattr(self._batch_worker, 'isRunning', lambda: False)():
                try:
                    self._batch_worker.stop()
                except Exception:
                    pass
        except Exception:
            pass

        self.image_paths = []

        self.inference_type_combo.blockSignals(True)
        self.inference_type_combo.setCurrentText('Standard')
        self.inference_type_combo.blockSignals(False)

        self.annotation_window.toolChanged.emit(None)
