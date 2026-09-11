import warnings

import gc
import os

import numpy as np

import torch
from torch.cuda import empty_cache
from torch.cuda import is_available as is_cuda_available

from ultralytics import FastSAM
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM2Predictor, SAM3Predictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
                             QToolButton, QVBoxLayout, QGroupBox, QWidget)

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import bgr_to_qimage, decode_video_frame

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox.SAM import SharedWeights

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Segment-everything settings for SAM models, as Predictor.generate() keywords.
# FastSAM is a YOLO model and takes none of them.
#   points_stride           prompt points per side of the grid (16 -> 256 points)
#   crop_n_layers           extra passes over zoomed-in crops: layer n adds 4**n
#                           crops, each encoded at the full input size
#   conf_thres              minimum predicted mask quality (IoU)
#   stability_score_thresh  minimum mask stability under threshold changes
#   min_mask_region_area    islands and holes smaller than this (original-image
#                           pixels) are removed from each mask
GENERATE_PRESETS = {
    "Fast": dict(points_stride=16, crop_n_layers=0, conf_thres=0.88,
                 stability_score_thresh=0.95, min_mask_region_area=0),
    "Balanced": dict(points_stride=32, crop_n_layers=0, conf_thres=0.86,
                     stability_score_thresh=0.92, min_mask_region_area=100),
    # One crop layer, so each quarter of the image is encoded at the full input
    # size and small organisms cover 2x the pixels, and a denser grid, so more
    # of them get a prompt point of their own.
    "Small objects": dict(points_stride=48, crop_n_layers=1, conf_thres=0.82,
                          stability_score_thresh=0.90, min_mask_region_area=25),
}
CUSTOM_PRESET = "Custom"
DEFAULT_PRESET = "Balanced"

# Points per side is halved for each crop layer. A layer-1 crop covers about a
# quarter of the image, so the absolute point density stays the same and the
# crop layer spends its cost on resolution rather than on 4x the prompts.
CROP_DOWNSCALE_FACTOR = 2


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SegmentEverything:
    """Callable stand-in for ultralytics.SAM that runs Predictor.generate() with the dialog's settings.

    SAM()(img, crop_n_layers=1) is rejected: Model.predict checks every
    keyword against the ultralytics config, which has none of generate()'s.
    The predictor's own __call__ hands extra keywords through inference() to
    generate() when no prompts are set, so this calls the predictor instead.

    It takes the same keywords as SAM() so the batch inference worker can call
    it unchanged. conf, iou and imgsz are applied; the rest are ignored, since
    the device and precision were fixed when the model was loaded.
    """

    _ARGS = ('conf', 'iou', 'imgsz')

    def __init__(self, predictor, get_generate_kwargs):
        self.predictor = predictor
        # Read on each call. The batch worker calls from its own thread, so the
        # dialog hands over a plain dict it rebuilds on the GUI thread rather
        # than having this read the widgets.
        self.get_generate_kwargs = get_generate_kwargs

    @property
    def model(self):
        return self.predictor.model

    def __call__(self, source, stream=False, **kwargs):
        for key in self._ARGS:
            if key in kwargs:
                setattr(self.predictor.args, key, kwargs[key])
        return self.predictor(source, stream=stream, **self.get_generate_kwargs())


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

        self.setWindowIcon(get_window_icon("wizard.svg"))
        self.setWindowTitle("SAM Generator (Ctrl + 5)")
        self.resize(800, 400)

        # Initialize variables
        self.imgsz = 640 if not is_cuda_available() else 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30

        self.task = 'segment'
        self.max_detect = 300
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None
        self.model_type = None  # Either 'fastSAM' or 'sam'
        self._oom_skipped = 0  # Inputs skipped for GPU out-of-memory in the current predict()
        # Predictor.generate() keywords from the Segment Everything group,
        # rebuilt whenever a setting changes (see SegmentEverything)
        self.generate_kwargs = {}

        # Information across the top, then two columns (landscape);
        # self.layout is the layout being filled
        root = QVBoxLayout(self)
        self.layout = root
        # Setup the info layout
        self.setup_info_layout()

        columns = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left)
        columns.addLayout(right)
        root.addLayout(columns)

        self.layout = left
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the segment-everything layout
        self.setup_generate_layout()
        left.addStretch()

        self.layout = right
        # Setup the thresholds layout
        self.setup_thresholds_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        # Setup the status layout
        self.setup_status_layout()
        right.addStretch()

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
        self.model_combo.setCurrentText("MobileSAM")
        self.model_combo.setToolTip("Choose a SAM variant for segment-everything inference.\nFastSAM: Fastest, lower accuracy.\nSAM 2.1/3: Slower, higher quality segmentation.\nMobileSAM: Lightweight, good balance.")

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
        self.detect_as_combo.setToolTip("Label to assign to all segmentations produced by SAM.")
        layout.addRow("Detect as:", self.detect_as_combo)

        # Task dropdown
        self.use_task_dropdown = QComboBox()
        self.use_task_dropdown.addItems(["detect", "segment"])
        self.use_task_dropdown.setCurrentText(self.task)
        self.use_task_dropdown.currentIndexChanged.connect(self.update_task)
        self.use_task_dropdown.setToolTip("Task mode for SAM.\nDetect: Bounding boxes only.\nSegment: Full instance segmentation masks.")
        layout.addRow("Task:", self.use_task_dropdown)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        self.resize_image_dropdown.setToolTip("(Automatic) Resize image to match model input requirements.")
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        # Same cap as the Predictor dialog. SAM's cost grows at least with the
        # square of this, and every crop layer re-encodes at it; large images
        # are better covered with work areas than with one huge input.
        self.imgsz_spinbox.setRange(640, 2048)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.imgsz_spinbox.setToolTip("Input image size for SAM.\nLarger sizes improve segmentation quality but increase processing time.\n"
                                      "Rounded to a multiple of 32.")
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_generate_layout(self):
        """
        Setup the segment-everything settings (SAM models only) in a group box.
        """
        self.generate_group = QGroupBox("Segment Everything")
        layout = QVBoxLayout()
        preset_layout = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(GENERATE_PRESETS) + [CUSTOM_PRESET])
        self.preset_combo.setCurrentText(DEFAULT_PRESET)
        self.preset_combo.setToolTip(
            "Fast: sparse grid, fewest masks.\n"
            "Balanced: the standard 32x32 grid.\n"
            "Small objects: denser grid plus one crop layer, for small organisms.\n"
            "The slowest: the crop layer encodes the image 5 times instead of once.\n"
            "Editing any of its settings switches to Custom.")
        preset_layout.addRow("Preset:", self.preset_combo)
        layout.addLayout(preset_layout)

        # The individual settings start folded away: the dialog is already tall,
        # and a preset is what most runs need.
        self.generate_settings_toggle = QToolButton()
        self.generate_settings_toggle.setText("Settings")
        self.generate_settings_toggle.setCheckable(True)
        self.generate_settings_toggle.setStyleSheet("QToolButton { border: none; }")
        self.generate_settings_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.generate_settings_toggle.setArrowType(Qt.RightArrow)
        self.generate_settings_toggle.toggled.connect(self.toggle_generate_settings)
        self.generate_settings_toggle.setToolTip("Show the settings behind the preset.")
        layout.addWidget(self.generate_settings_toggle)
        # Presets only for now: the toggle (and so the settings) stays hidden
        self.generate_settings_toggle.setVisible(False)

        self.generate_settings_widget = QWidget()
        settings_layout = QFormLayout(self.generate_settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(4, 128)
        self.points_spinbox.setToolTip(
            "Prompt points per side of the grid SAM segments from (32 -> 1024 points).\n"
            "More points find more, smaller objects, and take longer.")
        settings_layout.addRow("Points per Side:", self.points_spinbox)

        self.crop_layers_spinbox = QSpinBox()
        self.crop_layers_spinbox.setRange(0, 2)
        self.crop_layers_spinbox.setToolTip(
            "Extra passes over zoomed-in crops of the image. Layer 1 adds 4 crops,\n"
            "layer 2 another 16, each encoded at the full Image Size, so small objects\n"
            "are seen at higher resolution. Each layer multiplies the run time.")
        settings_layout.addRow("Crop Layers:", self.crop_layers_spinbox)

        self.mask_conf_spinbox = QDoubleSpinBox()
        self.mask_conf_spinbox.setRange(0.0, 1.0)
        self.mask_conf_spinbox.setSingleStep(0.01)
        self.mask_conf_spinbox.setDecimals(2)
        self.mask_conf_spinbox.setToolTip(
            "Minimum mask quality SAM predicts for itself (its IoU estimate).\n"
            "The Uncertainty Threshold below is applied as well.")
        settings_layout.addRow("Mask Confidence:", self.mask_conf_spinbox)

        self.stability_spinbox = QDoubleSpinBox()
        self.stability_spinbox.setRange(0.0, 1.0)
        self.stability_spinbox.setSingleStep(0.01)
        self.stability_spinbox.setDecimals(2)
        self.stability_spinbox.setToolTip(
            "Minimum mask stability: how little the mask changes when its cutoff moves.\n"
            "Lower keeps more, fuzzier-edged masks.")
        settings_layout.addRow("Stability:", self.stability_spinbox)

        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 100000)
        self.min_area_spinbox.setSingleStep(25)
        self.min_area_spinbox.setSuffix(" px")
        self.min_area_spinbox.setToolTip(
            "Islands and holes smaller than this many image pixels are removed\n"
            "from each mask. 0 skips the cleanup, which is the fastest.")
        settings_layout.addRow("Min Region Area:", self.min_area_spinbox)

        self.generate_settings_widget.setVisible(False)
        layout.addWidget(self.generate_settings_widget)

        self.generate_group.setLayout(layout)
        self.layout.addWidget(self.generate_group)

        self.preset_combo.currentTextChanged.connect(self.apply_generate_preset)
        for spinbox in (self.points_spinbox, self.crop_layers_spinbox, self.mask_conf_spinbox,
                        self.stability_spinbox, self.min_area_spinbox):
            spinbox.valueChanged.connect(self.on_generate_setting_changed)
        self.model_combo.currentTextChanged.connect(self.update_generate_enabled)

        self.apply_generate_preset(DEFAULT_PRESET)
        self.update_generate_enabled()

    def _generate_spinboxes(self):
        """Map each Predictor.generate() keyword to the spinbox that sets it."""
        return {
            'points_stride': self.points_spinbox,
            'crop_n_layers': self.crop_layers_spinbox,
            'conf_thres': self.mask_conf_spinbox,
            'stability_score_thresh': self.stability_spinbox,
            'min_mask_region_area': self.min_area_spinbox,
        }

    def apply_generate_preset(self, name):
        """Fill the segment-everything settings from a preset (Custom leaves them as they are)."""
        preset = GENERATE_PRESETS.get(name)
        if preset is not None:
            for key, spinbox in self._generate_spinboxes().items():
                spinbox.blockSignals(True)
                spinbox.setValue(preset[key])
                spinbox.blockSignals(False)
        self.on_generate_setting_changed()

    def on_generate_setting_changed(self):
        """Rebuild generate_kwargs and show the preset the settings match, or Custom."""
        values = {key: spinbox.value() for key, spinbox in self._generate_spinboxes().items()}

        match = next((name for name, preset in GENERATE_PRESETS.items()
                      if all(abs(values[k] - v) < 1e-6 for k, v in preset.items())), CUSTOM_PRESET)
        if self.preset_combo.currentText() != match:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(match)
            self.preset_combo.blockSignals(False)

        # A new dict rather than an update in place: the batch worker reads
        # this from its own thread (see SegmentEverything)
        self.generate_kwargs = {**values, 'crop_downscale_factor': CROP_DOWNSCALE_FACTOR}

    def toggle_generate_settings(self, checked):
        """Show or fold away the settings behind the preset."""
        self.generate_settings_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.generate_settings_widget.setVisible(checked)
        self.adjustSize()

    def update_generate_enabled(self):
        """Grey the segment-everything settings out for FastSAM, which doesn't use them."""
        self.generate_group.setEnabled("FastSAM" not in self.model_combo.currentText())

    def setup_thresholds_layout(self):
        """
        Setup threshold control section using ThresholdsWidget.
        """
        # For SAM Generator: show all parameters including max_detections
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_boundary=True,
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
        load_button.setToolTip("Load the selected SAM model for segment-everything inference.")
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        deactivate_button.setToolTip("Unload the current SAM model and free GPU memory.")
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

        FastSAM loads as an ultralytics FastSAM model. SAM models load as a
        SAM predictor wrapped in SegmentEverything, which is what lets the
        segment-everything settings reach Predictor.generate(), and lets the
        predictor share its model with the Predictor dialog (SharedWeights).
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Obtaining model...", 3000)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get selected model name and path
            selected_model_name = self.model_combo.currentText()
            self.model_path = self.models[selected_model_name]
            self.task = self.use_task_dropdown.currentText()
            imgsz = self.get_imgsz()
            shared = False

            # Warm-up input. Letterboxed up to imgsz for SAM, so the encoder
            # still runs at full size; kept small so preparing it costs nothing.
            blank = np.zeros((64, 64, 3), dtype=np.uint8)

            # Determine which class to instantiate
            if "FastSAM" in selected_model_name:
                self.loaded_model = FastSAM(self.model_path)
                self.model_type = "fastSAM"
                # Warm-up at the precision the real calls use: ultralytics
                # rebuilds the predictor when quantize changes between calls.
                with torch.no_grad():
                    self.loaded_model(
                        blank,
                        conf=self.thresholds_widget.get_uncertainty_thresh(),
                        imgsz=imgsz,
                        device=self.main_window.device,
                        quantize=self.get_quantize(),
                        verbose=False
                    )
            else:
                shared = self._load_sam(imgsz)
                self.model_type = "sam"
                # Warm-up: one encoder pass and a one-point decode, instead of
                # segmenting everything in a blank image (1024 prompts)
                self.loaded_model.predictor(blank, point_grids=[np.array([[0.5, 0.5]])])

            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {self.model_path}"
                                    + (" (weights shared with SAM Predictor)" if shared else ""))
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")
            # Close on success, as the Predictor dialog does; a failed load
            # keeps it open so the choice can be retried.
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))
            self.loaded_model = None
            self.model_path = None
            self.model_type = None
        
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def _load_sam(self, imgsz):
        """
        Build the SAM predictor for self.model_path and wrap it as self.loaded_model.

        The device and precision are fixed here, as in the Predictor dialog.

        Returns:
            bool: True if the model was borrowed from the Predictor dialog.
        """
        stem = os.path.splitext(os.path.basename(self.model_path))[0]
        # The same choice ultralytics.SAM makes from the file name
        if "sam2" in stem:
            predictor_class = SAM2Predictor
        elif "sam3" in stem:
            predictor_class = SAM3Predictor
        else:
            predictor_class = SAMPredictor

        device = self.main_window.device
        quantize = self.get_quantize()
        predictor = predictor_class(overrides=dict(
            model=self.model_path,
            imgsz=imgsz,
            conf=self.thresholds_widget.get_uncertainty_thresh(),
            iou=self.thresholds_widget.get_iou_thresh(),
            device=device,
            quantize=quantize,
            save=False,
            verbose=False,
        ))
        # Builds the model from the weights (downloading them if needed)
        # unless the Predictor dialog has already built it
        module = SharedWeights.get(self.model_path, device, quantize)
        predictor.setup_model(model=module, verbose=False)
        SharedWeights.register(self.model_path, device, quantize, predictor.model)

        self.loaded_model = SegmentEverything(predictor, lambda: self.generate_kwargs)
        return module is not None

    def get_imgsz(self):
        """Get the image size for the model, rounded to a multiple of 32.

        SAM 2's Hiera encoder raises on sizes that aren't (1000 and 688, both
        reachable with the old 24-px step). The spinbox is updated to match.
        """
        value = self.imgsz_spinbox.value()
        snapped = int(round(value / 32)) * 32
        snapped = max(self.imgsz_spinbox.minimum(), min(self.imgsz_spinbox.maximum(), snapped))
        if snapped != value:
            self.imgsz_spinbox.setValue(snapped)
        self.imgsz = snapped
        return self.imgsz

    def get_quantize(self):
        """Precision for model calls: 32 for MobileSAM (crashes in FP16) and on the CPU, else 16.

        On the CPU, SAM's .half() cast is ~15x slower than FP32. A SAM
        predictor's precision is fixed when load_model builds it; FastSAM and
        the batch worker pass this on every call. It is the Predictor dialog's
        rule too, so the two dialogs can share a model (see SharedWeights).
        """
        on_cpu = str(self.main_window.device).strip().lower() == "cpu"
        is_mobile_sam = "mobile_sam" in (self.model_path or "")
        return 32 if on_cpu or is_mobile_sam else 16

    def predict(self, image_paths=None):
        """Run inference on one or more images with the loaded SAM/FastSAM model.

        Manages its own progress bar and always bakes results at the end.
        Images or tiles the GPU runs out of memory on are retried once by
        _apply_model, then skipped and reported when the run finishes.

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

        # Tiles highlighted together; the model still runs one image at a time
        BATCH_SIZE = 32
        self._oom_skipped = 0

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

                # Virtual video-frame paths (video.mp4::frame_N): decode the frame
                # first, so the raster shim that work-area crops read from is
                # pinned to this frame, and keep the array.  Holding it for the
                # fast render below means the painted pixels are the exact ones
                # the model saw — re-reading raster.rasterio_src afterwards would
                # not, because the progress bar yields to the event loop and a
                # Ctrl+hover preview can repoint the shim at another frame.
                frame_bgr = decode_video_frame(image_path, raster)

                use_tiles = (
                    raster.has_work_areas()
                    and self.annotation_window.get_selected_tool() == "work_area"
                )
                if use_tiles:
                    # A VideoRaster stores work areas for every frame in one flat
                    # list keyed by the virtual frame path; restrict to the frame
                    # being processed so we don't tile other frames' areas against
                    # the current frame's pixels.
                    all_areas = raster.get_work_areas()
                    frame_areas = [wa for wa in all_areas if wa.image_path == image_path]
                    if frame_areas and len(frame_areas) != len(all_areas):
                        work_areas = frame_areas
                        work_items_data = [
                            raster.get_work_area_data(wa) for wa in frame_areas
                        ]  # RGB
                    else:
                        work_areas = all_areas
                        work_items_data = raster.get_work_areas_data()  # RGB
                else:
                    work_areas = [None]
                    if frame_bgr is not None:
                        work_items_data = [frame_bgr]
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

                for i in range(0, len(work_items_data), BATCH_SIZE):
                    data_chunk = work_items_data[i:i + BATCH_SIZE]
                    area_chunk = work_areas[i:i + BATCH_SIZE]

                    # Highlight all tiles in this batch before inference so the
                    # user can see which regions are queued for processing.
                    for wa in area_chunk:
                        if wa is not None:
                            wa.highlight()

                    # One result per input (None for failures), so every tile
                    # in the chunk is unhighlighted and counted below.
                    batch_results = self._apply_model(data_chunk)

                    for wa, result in zip(area_chunk, batch_results):
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
                                boundary_tolerance=self.thresholds_widget.get_boundary_tolerance(),
                            )
                            # map_results_from_work_area resets result.path to the
                            # bare raster.image_path; for a virtual video frame
                            # (video.mp4::frame_N) that drops the frame suffix and
                            # the baked annotations get keyed to the wrong path and
                            # vanish on redraw. Restore the per-frame image_path.
                            result.path = image_path
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
                            image_path, raster, results_for_image, results_processor,
                            frame_bgr=frame_bgr)
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

            if self._oom_skipped:
                QMessageBox.warning(
                    self.annotation_window, "GPU Out of Memory",
                    f"{self._oom_skipped} image(s) or tile(s) were skipped because the GPU "
                    "ran out of memory.\nTry a smaller Image Size or a smaller model.")

    def _fast_render_image(self, image_path, raster, results_for_image, results_processor,
                           frame_bgr=None):
        """Push a ghost-render of new predictions to the OpenGL canvas without baking."""
        from coralnet_toolbox.utilities import rasterio_to_qimage
        aw = self.annotation_window

        # For a video frame, paint the array the model was given.  Reading the
        # shim instead lets a preview decode that landed between inference and
        # here put a different frame under this frame's detections.
        q_img = bgr_to_qimage(frame_bgr) if frame_bgr is not None else None

        if q_img is None:
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
        Apply the model to the inputs with task-aware parameters.
        Constructs kwargs dynamically based on task selection and model type.

        A SAM model (SegmentEverything) uses conf, iou and imgsz from these and
        adds the Segment Everything settings itself; the rest are for FastSAM.
        """
        # Base kwargs always passed
        kwargs = {
            'conf': self.thresholds_widget.get_uncertainty_thresh(),
            'imgsz': self.get_imgsz(),
            'max_det': self.thresholds_widget.get_max_detections(),
            'device': self.main_window.device
        }

        # Task-specific kwargs
        if self.task == 'segment':
            kwargs['retina_masks'] = True  # High-quality, non-blocky polygons
        # For 'detect' task, omit retina_masks to maximize speed & minimize memory

        # Always pass iou; let Ultralytics ignore it if not applicable
        kwargs['iou'] = self.thresholds_widget.get_iou_thresh()

        kwargs['quantize'] = self.get_quantize()

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
                results_list.append(self._run_model(img, kwargs))

            except Exception as e:
                print(f"Error running model on input: {e}")
                results_list.append(None)

        return results_list

    def _run_model(self, img, kwargs):
        """
        Run the model on one image, retrying once after a GPU out-of-memory error.

        Returns the image's Results, or None if it runs out of memory again;
        those are counted in self._oom_skipped for predict() to report.
        """
        for attempt in (1, 2):
            try:
                with torch.no_grad():
                    results = self.loaded_model(img, **kwargs)
                break
            except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
                if "out of memory" not in str(e).lower():
                    raise
                # Free cached blocks; fragmentation alone can cause this
                gc.collect()
                empty_cache()
                if attempt == 2:
                    print(f"SAM generator: GPU out of memory, skipping input: {e}")
                    self._oom_skipped += 1
                    return None

        result = results[0] if results else None

        # SAM's segment-everything ignores max_det (FastSAM, a YOLO model,
        # applies it itself), so keep the most confident ones here.
        if result is not None and self.model_type == "sam":
            result = self._keep_top_detections(result, kwargs['max_det'])
        return result

    @staticmethod
    def _keep_top_detections(result, max_det):
        """Return `result` limited to its `max_det` most confident detections."""
        if result.boxes is None or len(result.boxes) <= max_det:
            return result
        keep = torch.argsort(result.boxes.conf, descending=True)[:max_det]
        return result[keep]

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None
        self.model_type = None
        # Clean up resources (a model shared with the Predictor dialog stays
        # loaded there until that dialog lets it go too)
        gc.collect()
        torch.cuda.empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
