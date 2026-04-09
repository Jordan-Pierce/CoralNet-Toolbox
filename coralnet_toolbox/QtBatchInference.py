import time
import warnings

import os
from itertools import groupby
from operator import attrgetter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox,
                             QFormLayout, QComboBox, QHBoxLayout, QCheckBox, QButtonGroup,
                             QSpinBox, QPushButton)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.Common import ThresholdsWidget
from coralnet_toolbox.QtProgressBar import ProgressBar
import numpy as np

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
    # Signals to communicate back to the Main Thread
    progressUpdated = pyqtSignal(int, int)  # current_index, total_items
    frameProcessed = pyqtSignal(str, object, object)  # virtual_path, q_image, yolo_results
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, expanded_paths, initial_thresholds, raster_manager, device=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.paths = list(expanded_paths)
        self.raster_manager = raster_manager
        self.device = device

        self._is_running = True
        # Drop-frame gate: worker waits until the UI has consumed the previous frame
        self._waiting_for_ui = False

        # Mutex to ensure thread-safe threshold updates mid-inference
        self._mutex = QMutex()
        self._thresholds = dict(initial_thresholds or {})

    def stop(self):
        """Safely signals the worker loop to exit."""
        locker = QMutexLocker(self._mutex)
        try:
            self._is_running = False
        finally:
            del locker

    def release_frame_gate(self):
        """Called by the Main Thread once it finishes painting a frame.

        Clears the drop-frame gate so the decode loop can proceed to the
        next frame immediately, giving the illusion of buttery-smooth playback.
        """
        self._waiting_for_ui = False

    def update_thresholds(self, conf=None, iou=None, max_det=None):
        """Thread-safe slot called by the Main Thread when the user moves a slider."""
        locker = QMutexLocker(self._mutex)
        try:
            if conf is not None:
                self._thresholds['conf'] = conf
            if iou is not None:
                self._thresholds['iou'] = iou
            if max_det is not None:
                self._thresholds['max_det'] = max_det
        finally:
            del locker

    def run(self):
        """The main loop executed on the background thread."""
        try:
            total_items = len(self.paths)

            for i, path in enumerate(self.paths):
                try:
                    if self._waiting_for_ui:
                        while self._waiting_for_ui and self._is_running:
                            time.sleep(0.002)
                except Exception:
                    pass

                # Check if the user clicked "Stop"
                locker = QMutexLocker(self._mutex)
                try:
                    if not self._is_running:
                        break
                    # Grab a snapshot of the current thresholds for this frame
                    current_conf = self._thresholds.get('conf', 0.25)
                    current_iou = self._thresholds.get('iou', 0.7)
                    current_max_det = self._thresholds.get('max_det', 300)
                finally:
                    del locker

                # Video Frame Extraction
                if isinstance(path, str) and '::frame_' in path:
                    # Import here to avoid circular dependencies
                    from coralnet_toolbox.Rasters.VideoRaster import VideoRaster

                    base_path, frame_idx = VideoRaster.parse_frame_path(path)
                    raster = self.raster_manager.get_raster(base_path)

                    if not raster:
                        # Skip missing raster
                        self.progressUpdated.emit(i + 1, total_items)
                        continue

                    # Extract the raw numpy array directly from cv2
                    try:
                        frame_bgr = raster.get_bgr_frame(frame_idx)
                    except Exception:
                        frame_bgr = None

                    if frame_bgr is None:
                        self.progressUpdated.emit(i + 1, total_items)
                        continue

                    # Run Inference
                    try:
                        # Provide common flags similar to in-dialog inference where possible
                        call_kwargs = dict(conf=current_conf,
                                           iou=current_iou,
                                           max_det=current_max_det,
                                           device=self.device,
                                           stream=False,
                                           verbose=False)

                        results = self.model(frame_bgr, **call_kwargs)

                        # YOLO may return a single Results or a list
                        try:
                            if isinstance(results, (list, tuple)) and len(results) > 0:
                                final = results[0]
                            else:
                                final = results
                        except Exception:
                            final = results

                        # Convert BGR→RGB and build QImage in the worker thread
                        # (saves the UI thread from doing the expensive pixel crunch)
                        try:
                            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
                            q_img = VideoRaster._bgr_to_qimage(frame_bgr)
                        except Exception:
                            q_img = None

                        # Emit back to the Main Thread; raise gate so worker waits
                        self._waiting_for_ui = True
                        self.frameProcessed.emit(path, q_img, final)

                    except Exception as e:
                        # Emit error but continue with next frames
                        self.error.emit(str(e))

                else:
                    # Currently only video-frame fast-path implemented
                    pass

                # Update progress bar
                self.progressUpdated.emit(i + 1, total_items)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()



# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """
    Consolidated batch inference dialog for all models.
    Supports: Classify, Detect, Segment, Semantic, SAM, SeeAnything, Z-Inference.
    
    This dialog provides:
    - Model selection dropdown for all loaded models
    - ThresholdWidget for configurable inference thresholds
    - Task-specific options through subclassing
    - Images are selected through ImageWindow context menu (right-click)

    :param main_window: MainWindow object
    :param parent: Parent widget
    :param highlighted_images: List of image paths to process (required)
    """
    def __init__(self, main_window, parent=None, highlighted_images=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coralnet.svg"))
        self.setWindowTitle("Batch Inference")
        self.resize(500, 400)
        # Keep this dialog on top so users can update highlights while it is open
        try:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        except Exception:
            # Fallback for PyQt versions without setWindowFlag
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Initialize references to various deployment dialogs
        self.classify_dialog = getattr(main_window, 'classify_deploy_model_dialog', None)
        self.detect_dialog = getattr(main_window, 'detect_deploy_model_dialog', None)
        self.segment_dialog = getattr(main_window, 'segment_deploy_model_dialog', None)
        self.semantic_dialog = getattr(main_window, 'semantic_deploy_model_dialog', None)
        self.sam_dialog = getattr(main_window, 'sam_deploy_generator_dialog', None)
        self.see_anything_dialog = getattr(main_window, 'see_anything_deploy_generator_dialog', None)
        self.z_dialog = getattr(main_window, 'z_deploy_model_dialog', None)

        # Dictionary to store available model dialogs
        self.model_dialogs = {}
        self.model_keys = []
        self.loaded_model = None
        self.current_selected_model = None  # Track the current selected model

        # Storage for image paths and annotations
        self.annotations = []
        self.prepared_patches = []
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

    def _update_worker_thresholds(self, *args):
        """Safely pass updated global thresholds from MainWindow to the active worker.

        This slot listens to MainWindow signals (`uncertaintyChanged`, `iouChanged`,
        `maxDetectionsChanged`) and forwards the latest values to the running
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
            worker.update_thresholds(conf=conf, iou=iou, max_det=max_det)
        except Exception:
            pass

        # Route global threshold changes (MainWindow) into the active worker.
        # Connect once per dialog lifetime to avoid duplicate connections.
        try:
            if not getattr(self, '_thresholds_connected', False):
                self.main_window.uncertaintyChanged.connect(self._update_worker_thresholds)
                self.main_window.iouChanged.connect(self._update_worker_thresholds)
                self.main_window.maxDetectionsChanged.connect(self._update_worker_thresholds)
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
        # Detect and Segment use max_detections, uncertainty, iou, and area
        elif model_name in ("Detect", "Segment"):
            self.configure_thresholds(
                enable_max_detections=True,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # SAM uses uncertainty only
        elif model_name == "SAM":
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # See Anything uses uncertainty only
        elif model_name == "See Anything":
            self.configure_thresholds(
                enable_max_detections=False,
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

    def preprocess_annotations(self):
        """
        Get annotations based on user selection and preprocess them (Classify only).
        Groups annotations by image and crops them using concurrent processing.
        """
        # Get the annotations based on user selection
        if self.review_checkbox.isChecked():
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Check if annotations need to be cropped
        annotations_to_crop = []
        for annotation in self.annotations:
            if hasattr(annotation, 'cropped_image') and annotation.cropped_image:
                # Annotation already has cropped image, add to prepared patches
                self.prepared_patches.append(annotation)
            else:
                # Annotation needs to be cropped
                annotations_to_crop.append(annotation)

        # Only crop annotations that need cropping
        if annotations_to_crop:
            self.bulk_preprocess_patch_annotations(annotations_to_crop)

    def bulk_preprocess_patch_annotations(self, annotations_to_crop=None):
        """
        Bulk preprocess patch annotations by cropping the images concurrently.
        Uses ThreadPoolExecutor for parallel processing.

        Args:
            annotations_to_crop: List of annotations that need to be cropped.
                                If None, uses self.annotations.
        """
        if annotations_to_crop is None:
            annotations_to_crop = self.annotations

        if not annotations_to_crop:
            return

        # Get unique image paths for annotations that need cropping
        crop_image_paths = list(set(a.image_path for a in annotations_to_crop))

        # Create progress bar for cropping
        progress_bar = ProgressBar(self.annotation_window, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(crop_image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations_to_crop, key=attrgetter('image_path')),
                                      key=attrgetter('image_path'))

        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                # Dictionary to track futures and their corresponding image paths
                futures = {}

                # Process each group of annotations by image path
                for image_path, group in grouped_annotations:
                    # Convert group iterator to list for reuse
                    image_annotations = list(group)

                    # Submit cropping task asynchronously for each image
                    # Returns a Future object representing pending execution
                    future = executor.submit(self.annotation_window.crop_annotations,
                                             image_path,
                                             image_annotations,
                                             verbose=False)

                    # Store image path for each future for error reporting
                    futures[future] = image_path

                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        # Get cropped patches from completed task
                        cropped = future.result()
                        # Add cropped patches to prepared patches list
                        self.prepared_patches.extend(cropped)
                    except Exception as exc:
                        print(f"{futures[future]} generated an exception: {exc}")
                    finally:
                        # Update progress bar after each image is processed
                        progress_bar.update_progress()

        except Exception as e:
            print(f"Error in bulk preprocessing: {e}")

        finally:
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

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
        For Classify, runs preprocessing first to crop annotations.
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

                # For Classify, preprocess annotations first
                if selected_model == "Classify":
                    self.preprocess_annotations()

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

    def batch_inference(self):
        """
        Perform batch inference on selected images based on the selected model type.
        Routes to appropriate predict method with correct parameters for each model.
        """
        # Determine the selected model type
        idx = self.model_combo.currentIndex()
        if idx < 0 or idx >= len(self.model_keys):
            raise ValueError("No model selected")

        selected_model = self.model_keys[idx]
        # Get the correct model dialog from the dictionary based on selected model
        model_dialog = self.model_dialogs.get(selected_model, None)

        if model_dialog is None:
            raise ValueError(f"No model loaded for {selected_model}")

        # Store active model and results processor for later baking
        self._active_model_dialog = model_dialog
        try:
            from coralnet_toolbox.Results import ResultsProcessor
            self._results_processor = ResultsProcessor(
                self.main_window,
                getattr(model_dialog, 'class_mapping', {})
            )
        except Exception:
            self._results_processor = None

        # Create progress bar as a top-level window (not parented, so it stays visible independently)
        progress_bar = ProgressBar(None, title="Batch Inference")
        progress_bar.setWindowFlags(progress_bar.windowFlags() | Qt.WindowStaysOnTopHint)
        progress_bar.show()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # --- Video fast-path: start background worker for video frames ---
            try:
                # Identify video vs non-video selected paths
                video_paths = []
                non_video_paths = []
                rm = getattr(self.image_window, 'raster_manager', None)
                for p in self.image_paths:
                    try:
                        raster = rm.get_raster(p) if rm is not None else None
                        if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                            video_paths.append(p)
                        else:
                            non_video_paths.append(p)
                    except Exception:
                        non_video_paths.append(p)

                # Only run worker for detection/segmentation-style models
                if video_paths and selected_model in ("Detect", "Segment", "Semantic", "SAM", "See Anything"):
                    from coralnet_toolbox.Rasters.VideoRaster import VideoRaster

                    # If there are non-video images, process them first synchronously
                    if non_video_paths:
                        try:
                            model_dialog.predict(non_video_paths, progress_bar)
                        except Exception:
                            pass

                    # Expand video frame virtual paths
                    expanded_paths = []
                    for vp in video_paths:
                        try:
                            raster = rm.get_raster(vp)
                            if raster is None:
                                continue
                            frame_count = int(getattr(raster, 'frame_count', 1))

                            if len(video_paths) == 1:
                                start = int(self.video_start_spin.value()) if hasattr(self, 'video_start_spin') else 0
                                end = int(self.video_end_spin.value()) if hasattr(self, 'video_end_spin') else (frame_count - 1)
                            else:
                                start = 0
                                end = max(0, frame_count - 1)

                            stride = int(self.video_stride_spin.value()) if hasattr(self, 'video_stride_spin') else 1
                            stride = max(1, stride)

                            for fi in range(start, end + 1, stride):
                                expanded_paths.append(VideoRaster.make_frame_path(raster.image_path, fi))
                        except Exception:
                            continue

                    if expanded_paths:
                        # Prepare progress bar for worker
                        progress_bar.set_title("Video Inference")
                        progress_bar.start_progress(len(expanded_paths))
                        # Ensure the visible bar is reset to zero before the
                        # background worker begins (prevents a residual 100%)
                        try:
                            progress_bar.set_value(0)
                            progress_bar.progress_bar.setValue(0)
                            QApplication.processEvents()
                        except Exception:
                            pass

                        # Store for handlers
                        self._progress_bar = progress_bar
                        self._active_model_dialog = model_dialog

                        # If user requested NOT to save annotations for video
                        # runs, ensure any existing unified cache is cleared so
                        # nothing gets baked later.
                        try:
                            if hasattr(self, 'save_video_annotations') and not self.save_video_annotations:
                                if hasattr(self.annotation_window, 'batch_results_cache'):
                                    self.annotation_window.batch_results_cache = {}
                        except Exception:
                            pass

                        # Initial thresholds snapshot
                        initial_thresholds = {
                            'conf': self.thresholds_widget.get_uncertainty_thresh() if hasattr(self, 'thresholds_widget') else 0.25,
                            'iou': self.thresholds_widget.get_iou_thresh() if hasattr(self, 'thresholds_widget') else 0.7,
                            'max_det': self.thresholds_widget.get_max_detections() if hasattr(self, 'thresholds_widget') else 300
                        }

                        # Create and start worker
                        self._batch_worker = BatchInferenceWorker(
                            model_dialog.loaded_model,
                            expanded_paths,
                            initial_thresholds,
                            self.image_window.raster_manager,
                            device=getattr(self.main_window, 'device', None)
                        )
                        # Enable streaming inference mode on the AnnotationWindow
                        try:
                            self.annotation_window.is_streaming_inference = True
                            
                            # --- Clear the paused frame's heavy graphics ---
                            # Strip the heavy items so they don't stay glued to the screen 
                            if hasattr(self.annotation_window, '_clear_current_frame_annotation_graphics'):
                                self.annotation_window._clear_current_frame_annotation_graphics()
                                
                        except Exception:
                            pass

                        # Wire signals: use lightweight frame handler for streaming playback
                        self._batch_worker.frameProcessed.connect(self.on_frame_processed)
                        self._batch_worker.progressUpdated.connect(self._on_worker_progress)
                        self._batch_worker.error.connect(lambda msg: print(f"BatchInferenceWorker error: {msg}"))
                        self._batch_worker.finished.connect(self._on_worker_finished)

                        # Push the initial thresholds into the worker and
                        # connect main-window threshold signals -> worker updater
                        try:
                            self._update_worker_thresholds()
                        except Exception:
                            pass

                        # Transform Apply (Ok) button into Stop Inference
                        try:
                            try:
                                self.button_box.accepted.disconnect()
                            except Exception:
                                pass
                            self.button_box.accepted.connect(self._on_stop_inference_clicked)
                            ok_btn = self.button_box.button(QDialogButtonBox.Ok)
                            if ok_btn:
                                ok_btn.setText("Stop Inference")
                        except Exception:
                            pass

                        # Allow cancel button on progress dialog and wire it
                        # to the same stop handler used by the dialog's OK
                        # button so the user can stop inference from the
                        # progress dialog as an alternative.
                        try:
                            progress_bar.cancel_button.setEnabled(True)
                            try:
                                progress_bar.cancel_button.clicked.connect(lambda checked=False: self._on_stop_inference_clicked())
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Cache ResultsProcessor for reuse across frames
                        try:
                            from coralnet_toolbox.Results import ResultsProcessor
                            self._results_processor = ResultsProcessor(
                                self.main_window,
                                getattr(model_dialog, 'class_mapping', {})
                            )
                        except Exception:
                            self._results_processor = None

                        # Disable interactive UI and video playback tools so the
                        # user cannot change settings during streaming inference.
                        try:
                            # Disable controls in this dialog and the video player
                            # but keep thresholds enabled for live tuning.
                            self.set_ui_processing_state(True)
                            try:
                                if hasattr(self, 'thresholds_widget') and self.thresholds_widget is not None:
                                    # Keep thresholds enabled so the user can adjust them
                                    self.thresholds_widget.setEnabled(True)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Make the Stop button visually prominent (red)
                        try:
                            ok_btn = self.button_box.button(QDialogButtonBox.Ok)
                            if ok_btn:
                                ok_btn.setText("Stop Inference")
                                ok_btn.setEnabled(True)
                                ok_btn.setStyleSheet("background-color: #d9534f; color: white;")
                        except Exception:
                            pass

                        # Start worker and return early — worker will drive UI updates via signals
                        self._batch_worker.start()
                        return
                    else:
                        # No video frames could be expanded (edge case).
                        # Non-video images were already processed synchronously above;
                        # bake their cached results now before falling through.
                        self._bake_cached_annotations()
            except Exception:
                # Non-fatal: fall back to synchronous processing below
                pass

            # Classify: predict on grouped annotation patches
            if selected_model == "Classify":
                if not self.prepared_patches:
                    # No annotations to process, silently return
                    progress_bar.finish_progress()
                    progress_bar.stop_progress()
                    progress_bar.close()
                    return

                # Group annotations by image path
                groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')),
                                 key=attrgetter('image_path'))

                # Count number of unique image paths
                num_paths = len(set(a.image_path for a in self.prepared_patches))

                # Make predictions on each image's annotations
                for idx_path, (path, patches) in enumerate(groups):
                    try:
                        progress_bar.set_title(f"Predicting: {idx_path + 1}/{num_paths} - {os.path.basename(path)}")
                        model_dialog.predict(inputs=list(patches), progress_bar=progress_bar)
                    except Exception as e:
                        print(f"Failed to make predictions on {path}: {e}")
                        continue

            # Detect, Segment, Semantic: predict on image paths
            elif selected_model in ("Detect", "Segment", "Semantic"):
                model_dialog.predict(self.image_paths, progress_bar)

            # SAM, See Anything: predict on image paths (using deploy_generator_dialog)
            elif selected_model in ("SAM", "See Anything"):
                model_dialog.predict(self.image_paths, progress_bar)
            
            # Z-Inference: predict on image paths with user-selected overwrite mode
            elif selected_model == "Z-Inference":
                # Show the overwrite dialog to get user choice
                overwrite_mode = self.z_dialog._show_overwrite_dialog(is_batch=True)
                if overwrite_mode is None:
                    # User cancelled
                    QApplication.restoreOverrideCursor()
                    progress_bar.finish_progress()
                    progress_bar.stop_progress()
                    progress_bar.close()
                    return
                # Predict with the selected overwrite mode
                model_dialog.predict(self.image_paths, progress_bar, overwrite_mode=overwrite_mode, show_dialog=False)

            else:
                raise ValueError(f"Unknown model type: {selected_model}")

            # Close the inference progress bar BEFORE baking so the bake bar is clearly visible
            try:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
            except Exception:
                pass

            # Bake any annotations cached by synchronous processing (images-only path)
            self._bake_cached_annotations()

            # Auto-close the dialog once everything is done
            try:
                self.accept()
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete batch inference: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
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

    def on_frame_processed(self, virtual_path, q_image, yolo_results):
        """Slot that receives data from the background worker and updates the streaming UI."""
        try:
            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
            
            # 1. Update UI State
            try:
                _, frame_idx = VideoRaster.parse_frame_path(virtual_path)
                self.annotation_window.current_image_path = virtual_path
                self.annotation_window._current_frame_idx = int(frame_idx)
            except Exception:
                pass

            # 2. CACHE THE RAW RESULTS (Do not create Annotation objects yet!)
            # Only cache/save video-frame results when the user has enabled it
            # via the `Save Annotations` control in the Video Options. This
            # allows fast visualization runs without persisting results.
            if getattr(self, 'save_video_annotations', True):
                if not hasattr(self.annotation_window, 'batch_results_cache'):
                    self.annotation_window.batch_results_cache = {}
                if yolo_results is not None:
                    yolo_results.path = virtual_path
                    self.annotation_window.batch_results_cache[virtual_path] = yolo_results
            else:
                # Skip caching/saving for this frame (visualization/debug mode)
                pass

            # 3. FAST RENDERING: Send QImage directly to the OpenGL canvas
            try:
                if q_image is not None and getattr(self.annotation_window, '_base_image_item', None) is not None:
                    self.annotation_window._base_image_item.set_image(q_image)
            except Exception:
                pass

            # 4. FAST ANNOTATIONS: Combine new predictions with existing annotations
            try:
                if getattr(self.annotation_window, '_base_image_item', None) is not None:
                    paths_data = []

                    # A. Generate fast paths for the NEW YOLO predictions
                    if yolo_results is not None:
                        model_type = getattr(self._active_model_dialog, 'task', 'detect')
                        rp = getattr(self, '_results_processor', None)
                        if rp is not None:
                            try:
                                paths_data.extend(rp.generate_fast_render_paths(yolo_results, model_type))
                            except Exception:
                                pass

                    # B. Grab EXISTING annotations so they don't disappear during playback!
                    try:
                        existing_annotations = self.annotation_window.get_image_annotations(virtual_path)
                        for a in existing_annotations:
                            # Only grab visible vector annotations (ignore heavy raster masks)
                            if getattr(a.label, 'is_visible', True) and not hasattr(a, 'mask_data'):
                                try:
                                    paths_data.append((a.get_painter_path(), a.label.color, a.transparency))
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # C. Send the combined paths directly to the OpenGL painter
                    try:
                        self.annotation_window._base_image_item.set_readonly_annotations(paths_data)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: Fast annotation render failed: {e}")

            # 5. Update the Video Player UI Slider
            try:
                if hasattr(self.annotation_window, '_video_player') and self.annotation_window._video_player:
                    vp = self.annotation_window._video_player
                    vp.slider.blockSignals(True)
                    try:
                        vp.slider.setValue(self.annotation_window._current_frame_idx)
                    finally:
                        vp.slider.blockSignals(False)

                    total_frames = getattr(self.annotation_window._active_video_raster, 'frame_count', 0)
                    vp.lbl_frame.setText(f"{self.annotation_window._current_frame_idx} / {total_frames}")
            except Exception:
                pass

        except Exception as e:
            print(f"Error in on_frame_processed: {e}")
        finally:
            try:
                if hasattr(self, '_batch_worker') and self._batch_worker is not None:
                    self._batch_worker.release_frame_gate()
            except Exception:
                pass

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

        cache = getattr(self.annotation_window, 'batch_results_cache', {})
        if not cache:
            return

        bake_pb = ProgressBar(None, title="Saving Annotations to Project...")
        bake_pb.setWindowFlags(bake_pb.windowFlags() | Qt.WindowStaysOnTopHint)
        bake_pb.show()
        bake_pb.start_progress(len(cache))

        # Suppress O(N²) UI updates during the loop
        self.annotation_window.is_streaming_inference = True

        is_segmentation = getattr(self._active_model_dialog, 'task', '') == 'segment'

        for path, cached_results in cache.items():
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

        # 2. BAKE ALL CACHED FRAMES (images + video, unified)
        self._bake_cached_annotations()

        # 3. Turn off streaming mode in the annotation window (failsafe)
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
        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []
        
        # Reset inference type to Standard
        self.inference_type_combo.blockSignals(True)
        self.inference_type_combo.setCurrentText("Standard")
        self.inference_type_combo.blockSignals(False)
        
        # Untoggle all tools in the annotation window
        self.annotation_window.toolChanged.emit(None)
