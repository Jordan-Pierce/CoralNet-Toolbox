import os
import gc
import hashlib
from dataclasses import dataclass, field

import numpy as np

import torch
from torch.cuda import empty_cache

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QApplication, QFileDialog,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QLineEdit,
                             QFormLayout, QComboBox, QSpinBox, QPushButton, QTabWidget, QWidget,
                             QHBoxLayout)

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import bgr_to_qimage, decode_video_frame

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_window_icon
from coralnet_toolbox.paths import resolve_weights

# Both See Anything dialogs open on the same model
from coralnet_toolbox.SeeAnything.QtDeployPredictor import DEFAULT_MODEL

from coralnet_toolbox.SeeAnything.PromptAlignment import (ensure_vp_predictor,
                                                          to_model_space)
from coralnet_toolbox.SeeAnything.QtPromptSession import (PromptButtons,
                                                          describe_sources,
                                                          fit_height,
                                                          fixed_width_text,
                                                          level_columns,
                                                          reserve_lines,
                                                          reserve_text_height,
                                                          session_warnings,
                                                          shared_prompt,
                                                          show_warnings,
                                                          warning_label)
from coralnet_toolbox.SeeAnything.PromptSession import (KIND_BOXES,
                                                        ORIGIN_ANNOTATIONS,
                                                        PromptSession,
                                                        collapse_to_one_class,
                                                        decoys_can_compete,
                                                        keep_positive_detections,
                                                        new_prototype,
                                                        stack_classes,
                                                        stem_from_path)
from coralnet_toolbox.SeeAnything.PromptCalibration import (DEFAULT_PER_TRY,
                                                            DEFAULT_TRIES,
                                                            MIN_IMAGES,
                                                            THRESHOLDS,
                                                            Calibration,
                                                            curve_for,
                                                            draw_tries,
                                                            match_image,
                                                            new_rng,
                                                            split_images)


def calibration_text(threshold, used, total, subset, held_out, found, known, false_positives):
    """What a calibration settled on and found, in one sentence."""
    images = f"{used} of {total} images" if subset else f"all {total} images"
    return (f"Confidence {threshold:.2f} from {images}. On {held_out} test images: "
            f"{found} of {known} boxes found, {false_positives} false positives.")


# The longest result `calibration_text` gives in practice (at most 40 test images),
# for the room its label keeps: calibrating then changes the text, never the height.
CALIBRATION_SAMPLE = calibration_text(0.99, 9999, 9999, True, 40, 9999, 9999, 9999)

# Lines the status keeps room for: the model, the prompt and the last thing done.
STATUS_LINES = 3


# Input-size granularity for YOLOE. Everything in the family is a stride-32
# model, so the letterbox pads to a multiple of this.
IMGSZ_STRIDE = 32


def boxes_signature(bboxes):
    """A short fingerprint of an image's annotation boxes.

    Stored on an example from annotations, so adding the same image again can
    tell whether its annotations changed and skip the embedding when they did not.
    """
    rounded = np.round(np.asarray(bboxes, dtype=np.float64).reshape(-1, 4), 1)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


@dataclass
class AnnotationAddReport:
    """What adding images from annotations did, image by image."""

    added: list = field(default_factory=list)       # Prototypes new to the prompt
    updated: list = field(default_factory=list)     # Prototypes that replaced an image's older example
    unchanged: list = field(default_factory=list)   # Image paths already in the prompt as they are
    empty: list = field(default_factory=list)       # Image paths without such annotations
    failed: list = field(default_factory=list)      # Image paths that could not be embedded

    @property
    def prototypes(self):
        """Every example added or updated."""
        return self.added + self.updated

    def message(self, label_code):
        """One sentence for the status."""
        def images(items):
            return f"{len(items)} image{'s' if len(items) != 1 else ''}"

        parts = []
        if self.added:
            parts.append(f"added {images(self.added)}")
        if self.updated:
            parts.append(f"updated {images(self.updated)} whose annotations changed")
        if self.unchanged:
            parts.append(f"{images(self.unchanged)} already in the prompt and unchanged")
        if self.empty:
            parts.append(f"{images(self.empty)} without {label_code} rectangles or polygons")
        if self.failed:
            parts.append(f"{images(self.failed)} could not be embedded")
        if not parts:
            return f"Nothing to add from {label_code}."
        text = "; ".join(parts)
        return f"{label_code}: {text[0].upper()}{text[1:]}."


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployGeneratorDialog(QDialog):
    """
    Perform See Anything (YOLOE) on multiple images from one prompt.

    The prompt is a `PromptSession` shared with the See Anything Predictor and its
    tool (QtPromptSession.SharedPrompt): examples Calibrate adds from annotations,
    examples drawn or clicked in the tool, and any loaded from a file. The tensor a
    run uses is `session.build_classes()`, the same function the tool predicts with.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.setWindowIcon(get_window_icon("eye.svg"))
        self.setWindowTitle("See Anything (YOLOE) Generator (Ctrl + 6)")
        # Landscape, to fit the two panels side by side. The height is fitted to
        # the contents when the dialog shows.
        self.resize(1000, 700)

        self.deploy_model_dialog = None

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30

        self.task = 'detect'
        self.max_detect = 300
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = {}

        # What detections are saved as. Its own control: it used to be the label
        # whose annotations were embedded, so finding Porites from branching-coral
        # examples meant saving the results as branching coral.
        self.output_label = None
        self.last_output_label_code = None
        # Choosing an annotation label sets the output label to match; one picked
        # by hand holds until the annotation label changes again.
        self.last_annotation_label_code = None

        # The prompt, shared with the See Anything Predictor and its tool.
        self.prompt = shared_prompt(main_window)
        self.prompt.changed.connect(self._on_prompt_changed)
        # Class indices at or above this are decoys (see _setup_model_with_vpes).
        self.n_positive_classes = None
        # The last thing done, shown under the model and prompt in the status.
        self._last_action = ""
        # What the last calibration settled on, and which images it was settled
        # for. The prompt has moved on from it as soon as its images differ.
        self._calibration_result = None
        self._calibration_images = None

        self.device = None  # Will be set in showEvent

        # Main vertical layout for the dialog
        self.layout = QVBoxLayout(self)

        # Setup the info layout at the top
        self.setup_info_layout()

        # Create horizontal layout for the two panels
        self.horizontal_layout = QHBoxLayout()
        self.layout.addLayout(self.horizontal_layout)

        # Create left panel
        self.left_panel = QVBoxLayout()
        self.horizontal_layout.addLayout(self.left_panel, 1)

        # Create right panel
        self.right_panel = QVBoxLayout()
        self.horizontal_layout.addLayout(self.right_panel, 1)

        # What to load and where detections go on the left
        self.setup_models_layout()
        self.setup_parameters_layout()
        self.setup_sam_layout()
        self.setup_output_layout()

        # The examples and what it finds on the right. The two columns end
        # level: the shorter one's boxes share the difference.
        self.setup_prompt_layout()
        self.setup_thresholds_layout()
        level_columns(self.left_panel, self.right_panel)

        # The actions in one row, then where things stand, across the full width
        self.setup_model_buttons_layout()
        self.setup_status_layout()

        # Setup the buttons layout at the bottom
        self.setup_buttons_layout()

        self.refresh_prompt()

    @property
    def session(self):
        """The shared prompt's session: Calibrate's examples, the tool's, and any loaded."""
        return self.prompt.session

    # --- The output label, and its old name --------------------------------------------------------------------------

    @property
    def reference_label(self):
        """Old name for `output_label`, kept for anything not yet converted."""
        return self.output_label

    @reference_label.setter
    def reference_label(self, label):
        self.output_label = label

    def showEvent(self, event):
        """
        Set up the layout when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.thresholds_widget.initialize_thresholds()

        # Update the device
        self.device = self.main_window.device
        # Populate both label dropdowns and restore the last selections.
        self.update_label_combos()
        self.refresh_prompt()
        # Room for the text planned for, then no spare height.
        QWidget.layout(self).activate()
        self.reserve_space()
        fit_height(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Text wraps differently at another width, so the room it needs changes.
        self.reserve_space()

    def reserve_space(self):
        """Keep room for the longest calibration result and the usual status lines.

        Calibrating, loading a model or adding examples then changes text, not
        the dialog's height.
        """
        if hasattr(self, 'calibration_label'):
            reserve_text_height(self.calibration_label, CALIBRATION_SAMPLE)
        if hasattr(self, 'status_bar'):
            reserve_lines(self.status_bar, STATUS_LINES)

    def on_annotation_label_changed(self, _index=None):
        """A different label to take examples from: follow it with the output label."""
        annotation_label = self.annotation_label_combo.currentData()

        if annotation_label is None:
            self.refresh_prompt()
            return

        code = annotation_label.short_label_code
        # A different annotation label sets the output label. This used to stop
        # for good once an output label was picked by hand, even the one
        # already shown, so later label changes no longer reached it. Now a
        # hand-picked output holds until the annotation label changes again,
        # and re-opening on the same label keeps it.
        if code != self.last_annotation_label_code or self.output_label is None:
            self._follow_annotation_label(annotation_label)
        # Remembered for a better experience on re-opening.
        self.last_annotation_label_code = code

        # Calibrating works on the chosen label, so what it says and whether it
        # can run change with it.
        self.refresh_prompt()

    def paths_with_label_shapes(self, label_code):
        """Images with at least one rectangle or polygon of this label, from the pre-computed cache.

        Video rasters are left out: embedding reads a static image from the path,
        which a video file is not. Extract frames first to use them.
        """
        manager = self.main_window.image_window.raster_manager
        valid_types = {"RectangleAnnotation", "PolygonAnnotation"}
        paths = []
        for path in manager.image_paths:
            raster = manager.get_raster(path)
            if not raster or getattr(raster, 'raster_type', None) == 'VideoRaster':
                continue
            if not valid_types.isdisjoint(raster.label_to_types_map.get(label_code, set())):
                paths.append(path)
        return paths

    def run_blocker(self):
        """Why the prompt cannot run yet, or None when it can.

        OK is disabled while this says anything.
        """
        if self.loaded_model is None:
            return "Load a model first."
        if self.output_label is None:
            return "Choose a label under 'Save detections as'."
        if not self.session.has_positives():
            return "Press Calibrate, or add examples with the See Anything tool."
        try:
            self.session.check_stem(self._prompt_embedding_stem())
        except ValueError:
            return "The prompt was made with another model."
        return None

    def update_ok_state(self):
        """Enable OK only for a prompt that can run, and say why otherwise.

        Returns:
            str | None: The reason it cannot run, if any.
        """
        reason = self.run_blocker()
        if getattr(self, 'ok_button', None) is not None:
            self.ok_button.setEnabled(reason is None)
            self.ok_reason_label.setText(reason or "")
        return reason

    def accept(self):
        """Close the dialog, ready to run -- only if the prompt can run."""
        reason = self.update_ok_state()
        if reason:
            QMessageBox.warning(self, "Not Ready", reason)
            return
        super().accept()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout that spans the top.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Load a model, choose a label and press Calibrate: it adds that label's verified images "
                            "and picks the confidence threshold on some it holds out. "
                            "Press it again after adding annotations.\n"
                            "The See Anything tool shares this prompt. "
                            "Detections are saved under 'Save detections as'.")
        info_label.setTextFormat(Qt.PlainText)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)  # Add to main layout so it spans both panels

    def setup_models_layout(self):
        """
        Setup the models layout with tabbed interface for model selection.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        # Create tabbed widget
        tab_widget = QTabWidget()

        # Tab 1: Select model from dropdown
        model_select_tab = QWidget()
        model_select_layout = QFormLayout(model_select_tab)

        # Model combo box
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models
        self.models = [
            'yoloe-v8s-seg.pt',
            'yoloe-v8m-seg.pt',
            'yoloe-v8l-seg.pt',
            'yoloe-11s-seg.pt',
            'yoloe-11m-seg.pt',
            'yoloe-11l-seg.pt',
            'yoloe-26n-seg.pt',
            'yoloe-26s-seg.pt',
            'yoloe-26m-seg.pt',
            'yoloe-26l-seg.pt',
            'yoloe-26x-seg.pt',
        ]

        # Add all models to combo box
        for model_name in self.models:
            self.model_combo.addItem(model_name)

        # Set the default model (shared with the Predictor dialog)
        self.model_combo.setCurrentIndex(self.models.index(DEFAULT_MODEL))
        model_select_layout.addRow("Model:", self.model_combo)

        # The "Custom VPE" file row that sat here is gone: a prompt file is loaded
        # into the prompt list with its Load button, as rows like any other.

        tab_widget.addTab(model_select_tab, "Select Model")

        # Tab 2: Use existing model (custom weights) - only model browse
        model_existing_tab = QWidget()
        model_existing_layout = QFormLayout(model_existing_tab)

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        model_existing_layout.addRow("Model File:", model_layout)

        tab_widget.addTab(model_existing_tab, "Use Existing Model")

        layout.addWidget(tab_widget)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

    def browse_model_file(self):
        """
        Open a file dialog to browse for a model file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Model File",
                                                   "",
                                                   "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model_edit.setText(file_path)

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()

        # Task dropdown
        self.use_task_dropdown = QComboBox()
        self.use_task_dropdown.addItems(["detect", "segment"])
        self.use_task_dropdown.currentIndexChanged.connect(self.update_task)
        self.use_task_dropdown.setToolTip("Task mode for See Anything.\nDetect: Bounding boxes only.\nSegment: Full instance segmentation with masks.")
        layout.addRow("Task:", self.use_task_dropdown)

        # The greyed-out "Resize Image" dropdown that used to sit here is gone:
        # it was permanently disabled and described a manual resize the code no
        # longer performs -- ultralytics letterboxes the input itself.

        # Image size control. The old 1024-65536 range in steps of 1024 offered
        # sizes no GPU can run; cost grows with the square of this.
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(IMGSZ_STRIDE)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.imgsz_spinbox.setToolTip("Input image size for the See Anything model.\n"
                                      "Larger sizes improve accuracy but increase processing time and memory usage.\n"
                                      f"Rounded to a multiple of {IMGSZ_STRIDE}.\n"
                                      "Examples embed differently at different sizes: examples from annotations\n"
                                      "are embedded again at this size when a run starts; examples from the tool cannot be.")
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # A torch.compile toggle was offered here and has been removed. It never
        # amortizes in this dialog: predict() rebuilds YOLOE from the weights
        # file on every run, so each run starts with a cold compile cache, and
        # set_classes re-parameterizes the promptable head on top of that. Work
        # areas also vary in size, so each new tile shape recompiles again.
        # Measured on an RTX 5090: one 576x1024 inference went from tens of
        # milliseconds to 59.8 seconds, with dynamo emitting dynamic-shape
        # failures throughout.

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_sam_layout(self):
        """Use SAM model for segmentation."""
        group_box = QGroupBox("Use SAM to Create Polygons")
        layout = QFormLayout()

        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        self.use_sam_dropdown.setToolTip("Refine See Anything detections with SAM for higher-quality polygons.\nRequires SAM model to be deployed first.")
        layout.addRow("Use SAM Polygons:", self.use_sam_dropdown)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left pane

    def setup_thresholds_layout(self):
        """
        Setup threshold control section using ThresholdsWidget.
        """
        # For See Anything Generator: show all parameters including max_detections
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_boundary=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
        )
        self.right_panel.addWidget(self.thresholds_widget)

    def setup_output_layout(self):
        """The label detections are saved under, separate from the examples' label."""
        group_box = QGroupBox("Output")
        layout = QFormLayout()

        self.output_label_combo = QComboBox()
        self.output_label_combo.setToolTip(
            "Every detection is saved under this label.\n"
            "It need not be the label of the annotations used as examples: find Porites\n"
            "from branching-coral examples and save them as Porites.")
        self.output_label_combo.currentIndexChanged.connect(self._on_output_label_changed)
        layout.addRow("Save detections as:", self.output_label_combo)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)

    def setup_model_buttons_layout(self):
        """
        Setup the action buttons in one row across the dialog: the model, then the prompt.
        """
        group_box = QGroupBox("Actions")
        layout = QHBoxLayout()

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        load_button.setToolTip("Load the selected See Anything model.")

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        deactivate_button.setToolTip("Unload the current model. The prompt is kept.")

        self.prompt_buttons = PromptButtons("Remove every example, here and in the See Anything tool.")
        self.prompt_buttons.clearRequested.connect(self.clear_prompt)
        self.prompt_buttons.saveRequested.connect(self.save_session_file)
        self.prompt_buttons.loadRequested.connect(self.load_session_file)

        # Equal shares of the width, however wide the dialog is.
        for button in [load_button, deactivate_button] + self.prompt_buttons.buttons:
            layout.addWidget(button, 1)

        # The image-size warnings depend on the spinbox.
        self.imgsz_spinbox.valueChanged.connect(lambda _: self.refresh_prompt())

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """
        Setup status display in a group box across the dialog.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_bar = QLabel("No model loaded")
        self.status_bar.setWordWrap(True)
        # The status changes after every action; its length must not resize the dialog.
        fixed_width_text(self.status_bar)
        layout.addWidget(self.status_bar)

        self.warning_label = warning_label()
        layout.addWidget(self.warning_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_prompt_layout(self):
        """The label to take examples from, and the button that turns them into a prompt."""
        group_box = QGroupBox("Examples from annotations")
        layout = QVBoxLayout(group_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label:"))
        self.annotation_label_combo = QComboBox()
        self.annotation_label_combo.setToolTip("Whose verified annotations are the examples.")
        self.annotation_label_combo.currentIndexChanged.connect(self.on_annotation_label_changed)
        row.addWidget(self.annotation_label_combo, 1)
        layout.addLayout(row)

        self.calibrate_button = QPushButton("Calibrate")
        self._calibrate_tooltip = (
            "Add the label's verified images to the prompt and choose the confidence threshold,\n"
            "tested on some images left out of the prompt.\n"
            "Press it again after adding more annotations.\n"
            "Only verified annotations are used. Objects nobody annotated count as false positives,\n"
            "so the threshold may come out a little high.")
        self.calibrate_button.setToolTip(self._calibrate_tooltip)
        self.calibrate_button.clicked.connect(self.calibrate_prompt)
        layout.addWidget(self.calibrate_button)

        # What the last calibration found. It keeps room for the longest result
        # (see `reserve_space`), so calibrating never makes the dialog taller.
        self.calibration_label = fixed_width_text(QLabel())
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)
        self.right_panel.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        row = QHBoxLayout()
        row.addStretch(1)

        self.ok_reason_label = QLabel()
        self.ok_reason_label.setStyleSheet("color: #b36b00;")
        row.addWidget(self.ok_reason_label)

        # Create a button box for the buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.ok_button = self.button_box.button(QDialogButtonBox.Ok)
        row.addWidget(self.button_box)

        self.layout.addLayout(row)

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            if self.use_sam_dropdown.currentText() != "False":
                self.use_sam_dropdown.blockSignals(True)
                self.use_sam_dropdown.setCurrentText("False")
                self.use_sam_dropdown.blockSignals(False)
            if hasattr(self, 'update_sam_task_state'):
                self.update_sam_task_state()
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        if hasattr(self, 'update_sam_task_state'):
            self.update_sam_task_state()

        return True

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
            sam_is_available = (
                hasattr(self, 'sam_dialog') and
                self.sam_dialog is not None and
                self.sam_dialog.loaded_model is not None
            )

            if sam_is_available:
                # If SAM is wanted and available, the task must be segmentation.
                self.task = 'segment'
            else:
                # If SAM is wanted but not available, revert the dropdown and do nothing else.
                # The 'is_sam_model_deployed' function already handles showing an error message.
                self.use_sam_dropdown.setCurrentText("False")
        else:
            self.task = self.use_task_dropdown.currentText()

        # If use_sam_dropdown is "False", do nothing. Let self.task be whatever the user set.

    def update_task(self):
        """Update the task based on the dropdown selection and handle UI/model effects."""
        self.task = self.use_task_dropdown.currentText()

        # Update UI elements based on task
        if self.task == "segment":
            # Deactivate model if one is loaded and we're switching to segment task
            if self.loaded_model:
                self.deactivate_model()

    # --- Labels ---------------------------------------------------------------------------------------------------------

    def update_label_combos(self):
        """Fill the annotation and output label dropdowns with every project label.

        The "Review" label with id "-1" is excluded: it is a placeholder, not a
        thing to find or to save detections as.
        """
        labels = [label for label in self.main_window.label_window.labels
                  if not (label.short_label_code == "Review" and str(label.id) == "-1")]
        labels.sort(key=lambda label: label.short_label_code)

        for combo, code in ((self.annotation_label_combo, self.last_annotation_label_code),
                            (self.output_label_combo, self.last_output_label_code)):
            combo.blockSignals(True)
            try:
                combo.clear()
                for label in labels:
                    combo.addItem(label.short_label_code, label)
                if code:
                    index = combo.findText(code)
                    if index != -1:
                        combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)

        self._on_output_label_changed()
        # Filtering also lets the output label follow the annotation label.
        self.on_annotation_label_changed()

    def _follow_annotation_label(self, label):
        index = self.output_label_combo.findText(label.short_label_code)
        if index != -1 and index != self.output_label_combo.currentIndex():
            self.output_label_combo.setCurrentIndex(index)

    def _on_output_label_changed(self, _index=None):
        self.output_label = self.output_label_combo.currentData()
        if self.output_label is not None:
            self.last_output_label_code = self.output_label.short_label_code
        self.update_ok_state()

    # --- Annotations as examples --------------------------------------------------------------------------------------

    def get_label_annotations(self, image_path, label_code):
        """
        Return the bboxes and polygon points of one label's verified annotations on an image.

        :param image_path: The path of the image to get annotations from.
        :param label_code: The short label code to filter annotations by.
        :return: A tuple containing a numpy array of bboxes and a list of masks.
        """
        if not image_path or not label_code:
            return np.array([]), []

        # Get all annotations for the specified image
        annotations = self.annotation_window.get_image_annotations(image_path)

        # Filter annotations by the provided label
        bboxes = []
        masks = []
        for annotation in annotations:
            if annotation.label.short_label_code != label_code or not annotation.verified:
                continue
            if isinstance(annotation, (PolygonAnnotation, RectangleAnnotation)):
                bbox = annotation.cropped_bbox
                bboxes.append(bbox)
                if isinstance(annotation, PolygonAnnotation):
                    points = np.array([[p.x(), p.y()] for p in annotation.points])
                    masks.append(points)
                else:
                    x1, y1, x2, y2 = bbox
                    masks.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))

        return np.array(bboxes), masks

    def _embed_annotations(self, image_path, label_code, bboxes, masks):
        """One example from one label's boxes on one image, at the run's image size.

        The visual-prompt predictor must already be in place (`_ensure_vp_predictor`),
        so a batch of images pays for one warm-up rather than one each.

        The image's boxes merge into one example (`cls` all 0), exactly the reference
        VPE "Generate VPEs" used to make, so a prompt built from reference images
        predicts what it always did.

        Returns:
            Prototype: The example, not yet in the session.
        """
        count = len(bboxes)
        predictor = self.loaded_model.predictor
        predictor.set_prompts({'bboxes': bboxes, 'masks': masks, 'cls': np.zeros(count)})
        vpe = predictor.get_vpe(image_path)
        origin = {"source": ORIGIN_ANNOTATIONS, "image": image_path, "label": label_code,
                  "boxes": boxes_signature(bboxes)}
        name = os.path.basename(str(image_path))
        return new_prototype(KIND_BOXES, vpe, f"{count} {label_code} on {name}",
                             imgsz=self.get_imgsz(), origin=origin)

    def examples_from_annotations(self, label_code=None):
        """The prompt's examples from annotations, optionally of one label."""
        return [p for p in self.session.positives
                if p.source == ORIGIN_ANNOTATIONS and (label_code is None or p.origin.get("label") == label_code)]

    def add_from_annotations(self, image_paths, label_code):
        """Embed one label's boxes on each image and add them to the prompt, one example per image.

        An image already in the prompt for this label is left alone when its boxes
        and the image size are unchanged -- nothing is embedded, and the report says
        so -- and replaced in place when either changed.

        Args:
            image_paths (list[str]): Images whose annotations to embed.
            label_code (str): The label to take annotations of.

        Returns:
            AnnotationAddReport: What happened to each image.

        Raises:
            RuntimeError: If no visual-prompt predictor can be built.
        """
        report = AnnotationAddReport()
        if self.loaded_model is None or not label_code or not image_paths:
            return report

        imgsz = self.get_imgsz()
        predictor_ready = False
        for image_path in image_paths:
            bboxes, masks = self.get_label_annotations(image_path, label_code)
            if len(bboxes) == 0:
                report.empty.append(image_path)
                continue

            key = (image_path, label_code)
            existing = [p for p in self.session.positives if p.group_key == key]
            signature = boxes_signature(bboxes)
            if existing and all(p.origin.get("boxes") == signature and p.imgsz == imgsz for p in existing):
                report.unchanged.append(image_path)
                continue

            # Warm up only once something needs embedding.
            if not predictor_ready:
                if not self._ensure_vp_predictor():
                    raise RuntimeError("No visual-prompt predictor is available to embed annotations with.")
                predictor_ready = True

            try:
                prototype = self._embed_annotations(image_path, label_code, bboxes, masks)
            except Exception as e:
                print(f"Warning: could not embed annotations on {image_path}: {e}")
                report.failed.append(image_path)
                continue

            if existing:
                prototype.enabled = any(p.enabled for p in existing)
                report.updated.append(prototype)
            else:
                report.added.append(prototype)
            self.session.replace_group(False, key, [prototype])

        if report.prototypes and self.session.model_stem is None:
            self.session.model_stem = self._prompt_embedding_stem()
        self.refresh_prompt()
        return report

    def reembed_off_size_examples(self):
        """Embed annotation examples again at the run's image size.

        An embedding depends on the image size it was taken at (cosine 0.926
        between 640 and 1024), so an example added before the spinbox changed is
        not the example the run would see. Annotation examples can simply be
        embedded again, since the annotations are still in the project. Tool
        examples cannot: the work-area crop they came from is gone, and the panel warns.

        Returns:
            int: How many images' examples were embedded again.
        """
        imgsz = self.get_imgsz()
        stale = [p for p in self.session.positives
                 if p.can_reembed and p.imgsz and int(p.imgsz) != imgsz]
        if not stale:
            return 0
        if not self._ensure_vp_predictor():
            print("Warning: no visual-prompt predictor available; examples keep their image size.")
            return 0

        done = 0
        for old in stale:
            image_path, label_code = old.group_key
            bboxes, masks = self.get_label_annotations(image_path, label_code)
            if len(bboxes) == 0:
                print(f"Warning: {image_path} no longer has {label_code} annotations; "
                      f"its example keeps its image size.")
                continue
            try:
                fresh = self._embed_annotations(image_path, label_code, bboxes, masks)
            except Exception as e:
                print(f"Warning: could not embed {image_path} again: {e}")
                continue
            # Same row, same uid and on/off state; only what the embedding depends on changes.
            old.embedding = fresh.embedding
            old.imgsz = fresh.imgsz
            old.label = fresh.label
            old.origin = fresh.origin
            done += 1
        return done

    def _prompt_embedding_stem(self):
        """The checkpoint identifier ultralytics binds prompt embeddings to, or None.

        Private in ultralytics, so this is best-effort: a missing or failing
        method just means the architecture guard is skipped, not that loading
        breaks.
        """
        model = self.loaded_model
        if model is None:
            return None
        try:
            return model._prompt_embedding_model()
        except Exception:
            return None

    def load_model(self):
        """
        Load the selected model.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Obtaining model...", 3000)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Load the model using reload_model method
            self.reload_model()
            # Examples belong to one checkpoint; ones made with another model are
            # meaningless to this one.
            dropped = self._adopt_model_stem()

            message = f"Model loaded ({os.path.basename(str(self.model_path))})"
            if dropped:
                message += ("\n\nThe prompt, which the See Anything Predictor shares, was cleared: its "
                            "examples were made with a different model and cannot be used with this one.")
                self.report("Prompt cleared: its examples were made with a different model.")
            else:
                self.report("Model loaded.")

            # Finish progress bar
            progress_bar.finish_progress()
            QMessageBox.information(self.annotation_window, "Model Loaded", message)

        except Exception as e:
            self.loaded_model = None
            QMessageBox.critical(self.annotation_window,
                                 "Error Loading Model",
                                 f"Error loading model: {e}")

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None
            self.refresh_prompt()

    def _adopt_model_stem(self):
        """Bind the prompt to the model just loaded, dropping examples it cannot use.

        Returns:
            bool: True if examples were dropped because they belong to another model.
        """
        stem = self._prompt_embedding_stem()
        dropped = (not self.session.is_empty()
                   and self.session.model_stem is not None
                   and stem is not None
                   and stem != self.session.model_stem)
        if dropped:
            self.session.clear()
        if stem is not None:
            self.session.model_stem = stem
        self.refresh_prompt()
        return dropped

    def reload_model(self):
        """
        Load YOLOE fresh from the weights and warm it up with a visual prompt.

        Every run starts here: `set_classes` re-parameterizes the head, so a run
        begins from a clean model rather than the last run's classes.
        """
        self.loaded_model = None

        # Get model path - either from custom file or dropdown
        if self.model_edit.text().strip():
            # Use custom model file
            self.model_path = self.model_edit.text().strip()
        else:
            # Use selected model from dropdown
            self.model_path = self.model_combo.currentText()

        # Load model using registry
        self.loaded_model = YOLOE(resolve_weights(self.model_path))

        # Create a dummy visual dictionary for standard model loading
        visual_prompts = dict(
            bboxes=np.array(
                [
                    [120, 425, 160, 445],  # Random box
                ],
            ),
            cls=np.array(
                np.zeros(1),
            ),
        )

        # Run a dummy prediction to load the model
        self.loaded_model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            imgsz=640,
            conf=0.99,
        )

    def predict(self, image_paths=None):
        """Run SeeAnything (YOLOE) inference on one or more images.

        Manages its own progress bar and always bakes results at the end.
        SAM is applied per tile-batch inline (required for YOLOE prompt context).

        Args:
            image_paths: List of image paths to process.  If None, processes
                         the currently displayed image.
        """
        if not self.loaded_model or not self.output_label:
            QMessageBox.warning(self, "Setup Error",
                                "A model must be loaded and a label chosen under 'Save detections as'.")
            return

        if not image_paths:
            if self.annotation_window.current_image_path is None:
                QMessageBox.warning(self, "Warning", "No image is currently loaded for annotation.")
                return
            image_paths = [self.annotation_window.current_image_path]

        self.class_mapping = {0: self.output_label}
        results_processor = ResultsProcessor(self.main_window, self.class_mapping)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Running Inference")
        progress_bar.show()

        # Bake per-image so full-resolution mask tensors don't accumulate
        # across the whole batch — peak RAM stays at one image's worth.
        processed_paths = []

        # Set the model up once for the whole run. This used to sit inside the
        # per-image loop, where it rebuilt YOLOE from the weights file, ran a
        # warm-up prediction and re-applied set_classes for every image: a
        # 500-image run paid for 500 model loads. Nothing in it varies per
        # image — the task comes from the dropdown and the prompt is fixed once
        # the dialog is accepted.
        self.task = self.use_task_dropdown.currentText()
        if not self._setup_model_with_vpes():
            print("SeeAnything.predict: model setup failed; nothing to run.")
            progress_bar.close()
            QApplication.restoreOverrideCursor()
            return

        try:
            for img_idx, image_path in enumerate(image_paths):
                progress_bar.set_title(
                    f"Image {img_idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}"
                )
                inferred = self._infer_image(image_path, progress_bar)
                if inferred is None:
                    continue
                raster, frame_bgr, results_for_this_image = inferred

                # --- Bake this image's results immediately ---
                if results_for_this_image:
                    # Fast render must happen before baking — baking clears the
                    # per-result tensors that generate_fast_render_paths reads.
                    if image_path == self.annotation_window.current_image_path:
                        try:
                            self._fast_render_image(
                                image_path, raster, results_for_this_image, results_processor,
                                frame_bgr=frame_bgr)
                        except Exception as e:
                            print(f"SeeAnything.predict: fast render failed: {e}")

                    try:
                        self.annotation_window.is_streaming_inference = True
                    except Exception:
                        pass
                    try:
                        self._process_results(
                            results_processor, results_for_this_image, image_path)
                        processed_paths.append(image_path)
                    except Exception as e:
                        print(f"Error baking results for {image_path}: {e}")
                    try:
                        self.image_window.update_image_annotations(
                            image_path, update_counts=False)
                    except Exception:
                        pass

                    # Explicit drop: null masks + orig_img so the huge tensors
                    # can be reclaimed before the next image runs.
                    for r in results_for_this_image:
                        try:
                            r.masks = None
                        except Exception:
                            pass
                        try:
                            r.orig_img = None
                        except Exception:
                            pass
                    results_for_this_image.clear()
                    gc.collect()
                    empty_cache()

        except Exception as e:
            print(f"A fatal error occurred during the prediction workflow: {e}")
        finally:
            if processed_paths:
                try:
                    self.annotation_window.is_streaming_inference = False
                except Exception:
                    pass
                try:
                    self.annotation_window.refresh_phantom_annotations()
                except Exception:
                    pass
                try:
                    self.main_window.label_window.update_annotation_count()
                except Exception:
                    pass

            progress_bar.close()
            QApplication.restoreOverrideCursor()
            gc.collect()
            empty_cache()

    def _infer_image(self, image_path, progress_bar, batch_size=16):
        """Run the prompt over one image and return what it found, saving nothing.

        The model must already be set up (`_setup_model_with_vpes`). Decoy
        detections are dropped and every class collapsed to the output label, so
        the results are ready to draw or bake.

        Args:
            image_path (str): The image, or a virtual video-frame path.
            progress_bar: Advanced once per work item.
            batch_size (int): Work items per model call.

        Returns:
            tuple | None: `(raster, frame_bgr, results)`, or None when the image
            has nothing to run on.
        """
        # --- Get Raster and Work Items ---
        raster = self.image_window.raster_manager.get_raster(image_path)
        if raster is None:
            print(f"SeeAnything.predict: no raster for {image_path}, skipping.")
            return None

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
                ]
            else:
                work_areas = all_areas
                work_items_data = raster.get_work_areas_data()
        else:
            work_areas = [None]
            if frame_bgr is not None:
                work_items_data = [frame_bgr]
            else:
                work_items_data = [raster.image_path]

        if not work_items_data:
            print(f"SeeAnything.predict: no work items for {image_path}, skipping.")
            return None

        progress_bar.start_progress(len(work_items_data))

        results_for_this_image = []
        is_segmentation = self.task == 'segment' or self.use_sam_dropdown.currentText() == "True"

        try:
            # --- Loop over the data in mini-batches ---
            for i in range(0, len(work_items_data), batch_size):

                # Get the mini-batch chunks
                data_chunk = work_items_data[i: i + batch_size]
                area_chunk = work_areas[i: i + batch_size]

                # Highlight all tiles in this batch before inference
                for wa in area_chunk:
                    if wa is not None:
                        wa.highlight()

                # --- Apply Model (Batched) ---
                # Returns a flat list: [res1, res2, ...]
                batch_results_list = self._apply_model(data_chunk)

                # Decoy classes are part of the prompt, never the output:
                # drop what they won before SAM refines it and before class
                # IDs are collapsed to the output label below, after
                # which a decoy's detection could not be told apart.
                if self.n_positive_classes is not None:
                    batch_results_list = [keep_positive_detections(r, self.n_positive_classes)
                                          for r in batch_results_list]

                # --- Apply SAM (Batched) ---
                # Takes a flat list, returns a flat list.  SAM sees the
                # tile crop (YOLOE set orig_img to the input ndarray)
                # with tile-space boxes, so its ViT encoder is bounded
                # by tile size instead of full-raster size.
                sam_results_list = self._apply_sam(batch_results_list, image_path)

                # Safety check
                if len(sam_results_list) != len(area_chunk):
                    print("Warning: Mismatch in batch results. Skipping batch.")
                    for wa in area_chunk:
                        if wa is not None:
                            wa.unhighlight()
                        progress_bar.update_progress()
                    continue

                # --- Post-process ---
                for results_obj, work_area in zip(sam_results_list, area_chunk):

                    if not results_obj:  # Handle potential empty result
                        if work_area:
                            work_area.unhighlight()
                        progress_bar.update_progress()
                        continue

                    results_obj.path = image_path
                    # --- Map Result ---
                    if work_area:
                        mapped_result = MapResults().map_results_from_work_area(
                            results_obj, raster, work_area, is_segmentation,
                            boundary_tolerance=self.thresholds_widget.get_boundary_tolerance()
                        )
                        # map_results_from_work_area resets result.path to
                        # the bare raster.image_path; for a virtual video
                        # frame (video.mp4::frame_N) that drops the frame
                        # suffix and the baked annotations get keyed to the
                        # wrong path and vanish on redraw. Restore it.
                        mapped_result.path = image_path
                    else:
                        mapped_result = results_obj

                    # Release tile/full BGR reference; downstream baking
                    # only needs boxes + masks.xy, not orig_img.
                    try:
                        mapped_result.orig_img = None
                    except Exception:
                        pass

                    results_for_this_image.append(mapped_result)

                    progress_bar.update_progress()

                    if work_area:
                        work_area.unhighlight()

                # --- Clean up GPU memory *after* the mini-batch ---
                gc.collect()
                empty_cache()

        except Exception as e:
            print(f"An error occurred during prediction on {image_path}: {e}")
            import traceback
            traceback.print_exc()

        # Several positive classes are an implementation detail of the prompt;
        # See Anything saves one label. Collapse every class ID to 0.
        target_label_name = self.output_label.short_label_code
        for r in results_for_this_image:
            if r is not None:
                collapse_to_one_class(r, target_label_name)

        return raster, frame_bgr, results_for_this_image

    def _annotation_paths(self, image_path):
        """Ready-to-draw paths for the image's own (non-mask, visible) annotations."""
        paths = []
        try:
            for ann in self.annotation_window.get_image_annotations(image_path):
                if getattr(ann.label, 'is_visible', True) and not hasattr(ann, 'mask_data'):
                    try:
                        paths.append((ann.get_painter_path(), ann.label.color, ann.transparency))
                    except Exception:
                        pass
        except Exception:
            pass
        return paths

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
        fast_paths.extend(self._annotation_paths(image_path))

        if getattr(aw, '_base_image_item', None) is not None:
            try:
                aw._base_image_item.set_readonly_annotations(fast_paths)
                QApplication.processEvents()
            except Exception:
                pass

    # --- Model setup and inference ------------------------------------------------------------------------------------

    def _ensure_vp_predictor(self):
        """Make sure `loaded_model.predictor` is a visual-prompt predictor at the run's size.

        The Predictor dialog needs the same guarantee before it can suggest text
        for a drawn box, so the logic -- and the ultralytics behaviour it works
        around -- lives in `PromptAlignment.ensure_vp_predictor`.

        The size, device and precision are passed so the warm-up leaves the
        predictor at the ones `_apply_model` predicts with. `get_vpe` letterboxes
        at the predictor's own image size, and `reload_model` warms up at 640, so
        without them every example was embedded at 640 while the run predicted
        at the spinbox size. Measured: the same boxes embedded at 640 and at 1024
        have cosine 0.926, and detection confidences move by up to 0.07.

        Returns:
            bool: True if a visual-prompt predictor is now in place.
        """
        return ensure_vp_predictor(self.loaded_model, **self._embedding_args())

    def _embedding_args(self):
        """Size, device and precision to extract embeddings at.

        The same ones `_apply_model` predicts with, so an example is embedded
        exactly as the run will see it.
        """
        return dict(imgsz=self.get_imgsz(),
                    device=self.main_window.device,
                    quantize=self.get_quantize())

    def _setup_model_with_vpes(self):
        """Load the model fresh and give it the prompt's classes.

        Annotation examples embedded at another image size are embedded again
        first. The classes are `session.build_classes()` -- the function the tool
        predicts with -- so a prompt runs the same tensor wherever it was built:
        enabled positives, then decoys. Batch Inference calls this too, before its
        worker starts.

        Returns True on success, False on failure.
        """
        # Ensure model is loaded fresh
        self.reload_model()
        self.reembed_off_size_examples()
        self.refresh_prompt()

        session = self.session
        if not session.has_positives():
            QMessageBox.warning(self, "No Prompt", "Add an example or a phrase to the prompt first.")
            return False
        try:
            session.check_stem(self._prompt_embedding_stem())
        except ValueError as e:
            QMessageBox.warning(self, "Different Model", str(e))
            return False

        # One ultralytics "class" per example. These are an implementation
        # detail, never shown to the user: every class ID is collapsed back to 0
        # and named after the output label, because See Anything is single-class
        # by design.
        #
        # YOLOE.predict sets overrides["agnostic_nms"] = True unconditionally,
        # so NMS runs across all of these classes at once. Two examples that fire
        # on the same object therefore yield one detection, not two -- and a decoy
        # that matches an object better than every positive takes it.
        names, stacked, n_positive = session.build_classes()
        self.n_positive_classes = n_positive if len(names) > n_positive else None
        self._apply_classes(names, stacked)
        return True

    def _apply_classes(self, names, stacked):
        """Give the loaded model these class embeddings; `stack_classes` output.

        Calibration calls this once per try on the same model, which reproduces a
        fresh model's detections exactly (checked in scripts/SeeAnything_Testbed.py).
        """
        self.loaded_model.is_fused = lambda: False
        # Ensure underlying model.names is a dict mapping index->name
        mdl = getattr(self.loaded_model, 'model', None)
        if mdl is not None:
            names_attr_inner = getattr(mdl, 'names', None)
            if isinstance(names_attr_inner, list):
                try:
                    mdl.names = {i: n for i, n in enumerate(names_attr_inner)}
                except Exception:
                    pass
        self.loaded_model.set_classes(names, to_model_space(self.loaded_model, stacked))
        # Ensure `loaded_model.names` is a dict mapping index->name
        names_attr = getattr(self.loaded_model, 'names', None)
        if isinstance(names_attr, list):
            try:
                self.loaded_model.names = {i: n for i, n in enumerate(names_attr)}
            except Exception:
                pass

    def get_imgsz(self):
        """Return the spinbox image size, rounded to a multiple of the model stride.

        The spinbox is updated so it shows what is actually used.
        """
        low, high = self.imgsz_spinbox.minimum(), self.imgsz_spinbox.maximum()

        value = self.imgsz_spinbox.value()
        snapped = int(round(value / IMGSZ_STRIDE)) * IMGSZ_STRIDE
        # Round inwards at the ends, so the result is a multiple of the stride
        # rather than the range's own bound.
        if snapped < low:
            snapped = -(-low // IMGSZ_STRIDE) * IMGSZ_STRIDE
        elif snapped > high:
            snapped = (high // IMGSZ_STRIDE) * IMGSZ_STRIDE

        if snapped != value:
            self.imgsz_spinbox.setValue(snapped)
        self.imgsz = snapped
        return snapped

    def get_quantize(self):
        """Precision for model calls: 32 on the CPU, else 16.

        FP16 on the CPU is slower than FP32, not faster. Same rule as the SAM
        dialogs, so the two stay comparable.
        """
        on_cpu = str(self.main_window.device).strip().lower() == "cpu"
        return 32 if on_cpu else 16

    def _apply_model(self, inputs):
        """
        Apply the model (which is already set up) to the inputs.
        """
        # The model is ALREADY configured by _setup_model_with_vpes.
        # We just need to run prediction on the batch.
        #
        # quantize is the change here: nothing was passed before, so every
        # generator run was FP32 even on a GPU. rect=True only states the
        # minimum-padding letterboxing predict mode already defaults to, so the
        # geometry this depends on does not rest on an ultralytics default.
        #
        # compile is deliberately NOT passed: see setup_parameters_layout for
        # why torch.compile makes this dialog dramatically slower, not faster.
        results_generator = self.loaded_model.predict(inputs,
                                                      visual_prompts=[],  # Prompts are already in the model
                                                      imgsz=self.get_imgsz(),
                                                      rect=True,
                                                      conf=self.thresholds_widget.get_uncertainty_thresh(),
                                                      iou=self.thresholds_widget.get_iou_thresh(),
                                                      max_det=self.thresholds_widget.get_max_detections(),
                                                      device=self.main_window.device,
                                                      quantize=self.get_quantize(),
                                                      retina_masks=self.task == "segment")

        results_list = []
        for results in results_generator:
            # Append the object directly, not a list
            results_list.append(results)

        # Returns a flat list: [res1, res2, ...]
        return results_list

    def _apply_sam(self, results_list, image_path):
        """
        Apply SAM to a batch of results.
        Accepts a flat list of Results objects [res1, res2, ...]
        Returns a flat list of SAM-processed Results objects [sam_res1, sam_res2, ...]
        """
        # Check if SAM model is deployed and loaded
        self.update_sam_task_state()
        if self.task != 'segment':
            return results_list

        if not self.sam_dialog or self.use_sam_dropdown.currentText() == "False":
            return results_list

        if self.sam_dialog.loaded_model is None:
            self.task = 'detect'
            self.use_sam_dropdown.setCurrentText("False")
            return results_list

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # OPTIMIZATION: Pass the entire batch directly to SAM. The sam_dialog
        # handles batch iteration natively and is more efficient than per-item calls.
        sam_result_list = self.sam_dialog.predict_from_results(results_list, image_path)

        # Ensure we always return a list of the exact same length, substituting None for failures
        updated_results = []
        for orig_res, sam_res in zip(results_list, sam_result_list):
            updated_results.append(sam_res if sam_res else None)

        # Make cursor normal
        QApplication.restoreOverrideCursor()

        return updated_results

    def _process_results(self, results_processor, results_list, image_path):
        """
        Bake a list of already-remapped results into Annotation objects.
        Class remapping (objectN → 0 + target label name) is done earlier,
        before caching, so results here are already clean single-class results.
        """
        valid_results = [r for r in results_list if r is not None]

        if self.task == 'segment' or self.use_sam_dropdown.currentText() == "True":
            results_processor.process_segmentation_results(valid_results)
        else:
            results_processor.process_detection_results(valid_results)

    # --- Calibration --------------------------------------------------------------------------------------------------

    def calibration_pool(self, label_code):
        """The prompt's examples from this label's annotations, by image, in use or left out.

        Calibrating decides which of these are in use, so examples an earlier
        calibration left out are candidates again.

        Returns:
            dict[str, list[Prototype]]: Image path to its examples, in prompt order.
        """
        pool = {}
        for prototype in self.session.positives:
            if prototype.source == ORIGIN_ANNOTATIONS and prototype.origin.get("label") == label_code:
                pool.setdefault(prototype.origin.get("image"), []).append(prototype)
        return pool

    def label_annotation_boxes(self, image_path, label_code):
        """One label's rectangle and polygon boxes on an image, split into verified and unverified.

        Returns:
            tuple[np.ndarray, np.ndarray]: (V, 4) verified and (U, 4) unverified xyxy boxes.
        """
        verified, unverified = [], []
        for annotation in self.annotation_window.get_image_annotations(image_path):
            if annotation.label.short_label_code != label_code:
                continue
            if isinstance(annotation, (PolygonAnnotation, RectangleAnnotation)):
                (verified if annotation.verified else unverified).append(annotation.cropped_bbox)
        return (np.asarray(verified, dtype=np.float64).reshape(-1, 4),
                np.asarray(unverified, dtype=np.float64).reshape(-1, 4))

    def calibration_blocker(self):
        """Why Calibrate cannot run, or None when it can.

        Calibrate is disabled while this says anything, so nobody presses it only
        to be told there is too little to test on. Stops as soon as enough images
        are found, so it stays cheap on every prompt refresh.
        """
        if self.loaded_model is None:
            return "Load a model first."
        label = self.annotation_label_combo.currentData()
        if label is None:
            return "Choose a label under Examples from annotations."
        try:
            self.session.check_stem(self._prompt_embedding_stem())
        except ValueError:
            return "The prompt was made with another model."

        code = label.short_label_code
        usable = 0
        for path in self.paths_with_label_shapes(code):
            usable += bool(len(self.label_annotation_boxes(path, code)[0]))
            if usable >= MIN_IMAGES:
                return None
        return (f"Calibrating needs at least {MIN_IMAGES} images with verified {code} annotations. "
                f"There {'is' if usable == 1 else 'are'} {usable}.")

    def update_calibrate_state(self):
        """Enable Calibrate only when it can run; its tooltip says why not.

        Returns:
            str | None: The reason it cannot run, if any.
        """
        reason = self.calibration_blocker()
        button = getattr(self, 'calibrate_button', None)
        if button is not None:
            button.setEnabled(reason is None)
            button.setToolTip(self._calibrate_tooltip if reason is None
                              else f"{reason}\n\n{self._calibrate_tooltip}")
        return reason

    def calibrate_prompt(self):
        """The Calibrate button: add the label's verified images, then choose the confidence threshold.

        Every image of the label with a verified rectangle or polygon becomes an
        example (images already in the prompt are only embedded again if their
        annotations changed). Some images are then held out, the rest are
        stacked as the prompt, and the threshold that scores best on the held out
        images is applied; the held out images are left out of the prompt so the
        prompt that runs is the one that was tested. Pressing it again after more
        annotations are added starts over from all of them.
        """
        # The button is disabled while this says anything; checked again in
        # case it was pressed before a refresh caught up.
        reason = self.update_calibrate_state()
        if reason:
            QMessageBox.information(self, "Cannot Calibrate", reason)
            return
        code = self.annotation_label_combo.currentData().short_label_code

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Calibrating")
        progress_bar.show()
        outcome, error = None, None
        try:
            outcome = self.run_calibrate(code, progress_bar)
        except Exception as e:
            error = e
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            self.refresh_prompt()

        if error is not None:
            QMessageBox.warning(self, "Could Not Calibrate", str(error))
            return
        if outcome is None:
            return  # Canceled
        calibration, choice, pool, report = outcome
        if not calibration.found_anything():
            QMessageBox.information(self, "Nothing Found",
                                    f"No set of images found any of the verified {code} annotations on the "
                                    f"{len(calibration.validation)} test images, so nothing was changed.")
            return

        self.apply_calibration(pool, calibration, choice, code)
        self.report(f"{len(report.failed)} image{'s' if len(report.failed) != 1 else ''} could not be embedded."
                    if report.failed else "")

    def run_calibrate(self, label_code, progress_bar):
        """Add the label's images, split them, score the tries and choose.

        Returns:
            tuple | None: (Calibration, Choice, pool, AnnotationAddReport), or None if canceled.

        Raises:
            ValueError: If there are too few images to hold any out.
        """
        progress_bar.set_busy_mode("Adding the images...")
        report = self.add_from_annotations(self.paths_with_label_shapes(label_code), label_code)

        pool = self.calibration_pool(label_code)
        boxes = {path: self.label_annotation_boxes(path, label_code)
                 for path in set(self.paths_with_label_shapes(label_code)) | set(pool)}
        # An image is a candidate when it has a verified box to embed; one that
        # could not be added (or no longer has any) can still be tested on.
        eligible = [path for path in pool if len(boxes[path][0])]
        outside = [path for path, (verified, _) in boxes.items() if len(verified) and path not in pool]
        rng = new_rng(eligible + outside)
        candidates, validation = split_images(eligible, outside, rng)
        counts = {path: len(boxes[path][0]) for path in candidates}
        sets = draw_tries(candidates, DEFAULT_PER_TRY, DEFAULT_TRIES, rng, box_counts=counts)

        calibration = self.run_calibration(pool, boxes, candidates, validation, sets, progress_bar)
        if calibration is None:
            return None
        return calibration, calibration.choose(), pool, report

    def run_calibration(self, pool, boxes, candidates, validation, sets, progress_bar):
        """Score every try on the validation images.

        A try is the prompt as it would run with only the try's images in use:
        the enabled examples from elsewhere stay in, and so do the decoys. Each
        try predicts once per image at the lowest threshold tried, and every
        higher threshold is read off those detections.

        Args:
            pool (dict): `calibration_pool(label_code)`; rows outside `candidates`
                are in no try.
            boxes (dict): Image path to `label_annotation_boxes`: verified boxes
                are the answers, detections on unverified ones are ignored.
            candidates (list): Pool images a try can use.
            validation (list): Images to test on; in no try.
            sets (list[tuple]): The tries, as image paths.
            progress_bar (ProgressBar): Advanced per prediction; its cancel stops the run.

        Returns:
            Calibration | None: None if canceled.
        """
        progress_bar.set_busy_mode("Preparing the examples...")
        # A run embeds off-size examples again before predicting; so must this,
        # or the tries would not be the prompts that run.
        self.reembed_off_size_examples()
        self.refresh_prompt()
        self.session.check_stem(self._prompt_embedding_stem())

        pool_uids = {p.uid for group in pool.values() for p in group}
        fixed = {p.uid for p in self.session.enabled_positives() if p.uid not in pool_uids}
        decoys = self.session.enabled_negatives()

        calibration = Calibration(candidates=list(candidates), validation=list(validation))
        progress_bar.set_title(f"Scoring {len(sets)} {'try' if len(sets) == 1 else 'tries'} "
                               f"on {len(validation)} images")
        progress_bar.start_progress(len(sets) * len(validation))
        for keys in sets:
            chosen = fixed | {p.uid for key in keys for p in pool[key]}
            # Prompt order, so a try builds the tensor the applied prompt will.
            positives = [p for p in self.session.positives if p.uid in chosen]
            negatives = decoys if decoys_can_compete(positives) else []
            names, stacked, n_positive = stack_classes([p.embedding for p in positives],
                                                       [n.embedding for n in negatives])
            self._apply_classes(names, stacked)

            matches = []
            for path in validation:
                if progress_bar.wasCanceled():
                    return None
                detected, confidence = self._calibration_detections(path, n_positive)
                verified, unverified = boxes[path]
                matches.append(match_image(detected, confidence, verified, ignore=unverified))
                progress_bar.update_progress()
            calibration.tries.append(curve_for(keys, matches))
        return calibration

    def _calibration_detections(self, image_path, n_positive):
        """What the model, set up with a try's classes, finds on one whole image; decoys dropped.

        As `_apply_model` predicts, but at the lowest threshold tried and
        without work-area tiling, SAM or the area filter.

        Returns:
            tuple[np.ndarray, np.ndarray]: (N, 4) xyxy boxes in image pixels, and (N,) confidences.
        """
        results = self.loaded_model.predict(image_path,
                                            visual_prompts=[],
                                            imgsz=self.get_imgsz(),
                                            rect=True,
                                            conf=float(THRESHOLDS[0]),
                                            iou=self.thresholds_widget.get_iou_thresh(),
                                            max_det=self.thresholds_widget.get_max_detections(),
                                            device=self.main_window.device,
                                            quantize=self.get_quantize(),
                                            retina_masks=False,
                                            verbose=False)
        result = keep_positive_detections(results[0], n_positive)
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()

    def apply_calibration(self, pool, calibration, choice, label_code):
        """Keep the chosen images' examples in use, leave out the label's others, and set the threshold.

        The label's examples that were in no try, held out for testing or without
        a verified box, are left out too, so the prompt that runs is the one that
        was tested.
        """
        chosen = set(choice.score.keys)
        for path, group in pool.items():
            for prototype in group:
                prototype.enabled = path in chosen
        self.main_window.update_uncertainty_thresh(choice.score.threshold)
        self._calibration_result = self.calibration_summary(calibration, choice)
        self._calibration_images = self.annotation_signature(label_code)
        self.refresh_prompt()

    @staticmethod
    @staticmethod
    def calibration_summary(calibration, choice):
        """What a calibration settled on and found, in one sentence."""
        score = choice.score
        return calibration_text(score.threshold, len(score.keys), len(calibration.candidates), choice.subset,
                                len(calibration.validation), score.true_positives, score.known,
                                score.false_positives)

    def annotation_signature(self, label_code):
        """Which images' examples a calibration was made for: the label and every annotation example's id."""
        return label_code, frozenset(p.uid for p in self.examples_from_annotations())

    def calibration_note(self):
        """The last calibration's result while it still fits the prompt; otherwise what to do."""
        label = self.annotation_label_combo.currentData() if hasattr(self, 'annotation_label_combo') else None
        if label is None:
            return "Choose a label, then press Calibrate."
        code = label.short_label_code
        if not self.examples_from_annotations(code):
            return "Not calibrated yet."
        if self._calibration_result and self._calibration_images == self.annotation_signature(code):
            return self._calibration_result
        return "Not calibrated for these images. Press Calibrate."

    # --- The prompt ---------------------------------------------------------------------------------------------------

    def refresh_prompt(self):
        """Redraw this dialog, and tell the Predictor the shared prompt may have changed."""
        self._refresh_view()
        self.prompt.notify(self)

    def _refresh_view(self):
        """Redraw the buttons and the status; re-check whether the prompt can run."""
        buttons = getattr(self, 'prompt_buttons', None)
        if buttons is not None:
            buttons.refresh(self.session)
        label = getattr(self, 'calibration_label', None)
        if label is not None:
            label.setText(self.calibration_note())
        self.update_ok_state()
        self.update_calibrate_state()
        self.update_status()

    def _on_prompt_changed(self, source):
        """The Predictor or its tool changed or redrew the shared prompt: redraw."""
        if source is not self:
            self._refresh_view()

    def clear_prompt(self):
        """The Clear Prompt button: forget every example."""
        self.session.clear()
        self.refresh_prompt()

    def report(self, message):
        """Show what was just done, under the model and prompt lines of the status."""
        self._last_action = message or ""
        self.update_status()

    def update_status(self):
        """The model, where the prompt's examples came from, the last thing done, and warnings.

        The calibration result is not here: it has its own label under Calibrate.
        """
        status = getattr(self, 'status_bar', None)
        if status is None:
            return
        if self.loaded_model is None:
            lines = ["No model loaded."]
        else:
            lines = [f"Model: {os.path.basename(str(self.model_path))}"]
        lines.append(f"Prompt: {describe_sources(self.session)}")
        if self._last_action:
            lines.append(self._last_action)
        status.setText("\n".join(lines))
        show_warnings(self.warning_label, session_warnings(self.session, stem=self._expected_stem(),
                                                           imgsz=self.get_imgsz()))
        # Room is kept for the usual lines; anything longer (warnings) needs more
        # height, and is never squeezed.
        if self.isVisible():
            fit_height(self, grow_only=True)

    def _expected_stem(self):
        """The checkpoint stem a prompt must match: the loaded model's, else the selected one's."""
        stem = self._prompt_embedding_stem()
        if stem:
            return stem
        path = self.model_edit.text().strip() if hasattr(self, 'model_edit') else ""
        if not path and hasattr(self, 'model_combo'):
            path = self.model_combo.currentText()
        return stem_from_path(path)

    def import_session(self, session, source="the file"):
        """Add a prompt session's examples to the prompt.

        Appended, not replacing: tool examples sit beside annotation examples, and
        an example already here (same uid, or the same embedding from a file loaded
        twice) is skipped, so sending twice is harmless. Its threshold and image
        size are applied too: the threshold is what the user settled on, and tool
        examples cannot be embedded again, so the run predicts at their size.

        Args:
            session (PromptSession): The session to add.
            source (str): Where it came from, for the status.

        Returns:
            bool: True if the session was accepted.
        """
        if session is None or not session.has_positives():
            QMessageBox.information(self, "Nothing to Add",
                                    "The prompt session has no enabled positive examples.")
            return False

        try:
            session.check_stem(self._expected_stem())
            if not self.session.is_empty():
                self.session.check_stem(session.model_stem)
        except ValueError as e:
            QMessageBox.warning(self, "Different Model", str(e))
            return False

        # Copied in, so later edits in the tool do not change this prompt.
        offered = len(session.positives) + len(session.negatives)
        added = self.session.merge(session)
        notes = []
        if session.confidence is not None:
            self.main_window.update_uncertainty_thresh(session.confidence)
            notes.append(f"confidence threshold {session.confidence:.2f}")
        if session.imgsz and hasattr(self, 'imgsz_spinbox'):
            self.imgsz_spinbox.setValue(int(session.imgsz))
            self.get_imgsz()
            notes.append(f"image size {self.imgsz}")

        # The panel and OK button follow the session only when told to.
        self.refresh_prompt()

        if added == offered:
            message = f"Added {added} example{'s' if added != 1 else ''} from {source}."
        elif added:
            skipped = offered - added
            message = (f"Added {added} new example{'s' if added != 1 else ''} from {source}; "
                       f"{skipped} {'was' if skipped == 1 else 'were'} already in the prompt.")
        else:
            message = f"Everything from {source} is already in the prompt; nothing was added."
        if notes:
            message += " Applied its " + " and ".join(notes) + "."
        self.report(message)
        return True

    def load_session_file(self):
        """Add a saved prompt, or a plain prompt-embedding NPZ, to the prompt."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Prompt", "",
                                              "Prompt or prompt embeddings (*.npz)")
        if not path:
            return False
        try:
            session = PromptSession.from_npz(path, expected_stem=self._expected_stem())
        except Exception as e:
            QMessageBox.critical(self, "Could Not Load Prompt", str(e))
            return False
        if not self.import_session(session, source=os.path.basename(path)):
            return False
        QMessageBox.information(self, "Prompt Loaded", self._last_action)
        return True

    def session_for_export(self):
        """A copy of the prompt carrying the threshold and image size in force now."""
        session = self.session.copy()
        session.model_stem = session.model_stem or self._expected_stem()
        session.confidence = self.thresholds_widget.get_uncertainty_thresh()
        session.imgsz = self.get_imgsz()
        return session

    def save_session_file(self):
        """Save the whole prompt -- rows, roles, on/off states, sizes, origins, threshold."""
        if not self.session.positives:
            QMessageBox.information(self, "Nothing to Save", "The prompt has no positive examples yet.")
            return None
        path, _ = QFileDialog.getSaveFileName(self, "Save Prompt", "", "Prompt (*.npz)")
        if not path:
            return None
        try:
            written = self.session_for_export().to_npz(path)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Save Prompt", str(e))
            return None
        count = len(self.session.positives) + len(self.session.negatives)
        self.report(f"Prompt saved to {os.path.basename(written)}.")
        QMessageBox.information(self, "Prompt Saved",
                                f"Saved {count} example{'s' if count != 1 else ''} to:\n{written}")
        return written

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None

        # The prompt is kept: the Predictor shares it, and it still fits the next
        # model of the same kind. A different model clears it when it loads.
        self.n_positive_classes = None
        self.refresh_prompt()

        # Clean up references
        gc.collect()
        torch.cuda.empty_cache()

        # Untoggle all tools
        self.main_window.untoggle_all_tools()

        # Update status bar
        self.report("Model deactivated; the prompt is kept.")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
