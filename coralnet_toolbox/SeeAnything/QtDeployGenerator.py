import warnings

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
                             QHBoxLayout, QInputDialog, QSizePolicy)

from coralnet_toolbox.QtImageWindow import ImageWindow

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import bgr_to_qimage, decode_video_frame

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_window_icon

# Both See Anything dialogs open on the same model
from coralnet_toolbox.SeeAnything.QtDeployPredictor import DEFAULT_MODEL

from coralnet_toolbox.SeeAnything.PromptAlignment import (TextEmbedder,
                                                          ensure_vp_predictor,
                                                          to_model_space)
from coralnet_toolbox.SeeAnything.QtPromptAlignment import project_label_names
from coralnet_toolbox.SeeAnything.QtPromptSessionPanel import (PromptSessionPanel,
                                                               describe_sources,
                                                               fixed_width_text,
                                                               inspect_session)
from coralnet_toolbox.SeeAnything.PromptSession import (KIND_BOXES,
                                                        ORIGIN_ANNOTATIONS,
                                                        PromptSession,
                                                        collapse_to_one_class,
                                                        keep_positive_detections,
                                                        new_prototype,
                                                        stem_from_path)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

    The prompt is a `PromptSession`: examples added from annotations, phrases, and
    whatever the See Anything tool sends. It used to be five separate stores --
    reference images, an imported VPE file, kept phrases, an imported session and
    a legacy averaged tensor -- merged silently at run time. Now every example is a
    row in one list, and the tensor a run uses is `session.build_classes()`, the
    same function the tool predicts with.

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
        self.resize(1200, 760)  # Landscape, to fit the two panels side by side

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
        # Until the user picks an output label, it follows the annotation label,
        # which is what the one combo used to do.
        self._output_label_touched = False
        self.last_annotation_label_code = None

        # The prompt. Every example is a row in it, wherever it came from.
        self.session = PromptSession()
        self._text_embedder = None
        # Class indices at or above this are decoys (see _setup_model_with_vpes).
        self.n_positive_classes = None
        # The last thing done, shown under the model and prompt in the status.
        self._last_action = ""
        # How many images the Add from annotations table lists for the chosen label.
        self._annotation_image_count = None

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
        self.horizontal_layout.addLayout(self.left_panel)

        # Create right panel
        self.right_panel = QVBoxLayout()
        self.horizontal_layout.addLayout(self.right_panel, 1)

        # Settings on the left
        self.setup_models_layout()
        self.setup_parameters_layout()
        self.setup_sam_layout()
        self.setup_thresholds_layout()
        self.setup_output_layout()
        self.setup_model_buttons_layout()
        self.setup_status_layout()

        # Settings sit at their natural height at the top of the column rather
        # than being stretched apart down the dialog's full height.
        self.left_panel.addStretch(1)

        # The prompt on the right
        self.setup_prompt_layout()

        # Setup the buttons layout at the bottom
        self.setup_buttons_layout()

        self.refresh_prompt()

    # --- The output label, and its old name --------------------------------------------------------------------------

    @property
    def reference_label(self):
        """Old name for `output_label`, kept for anything not yet converted."""
        return self.output_label

    @reference_label.setter
    def reference_label(self, label):
        self.output_label = label

    def configure_image_window_for_dialog(self):
        """
        Disables parts of the internal ImageWindow UI to guide user selection.
        This forces the image list to only show images with annotations
        matching the selected annotation label.
        """
        iw = self.image_selection_window

        # Block signals to prevent setChecked from triggering the ImageWindow's
        # own filtering logic. We want to be in complete control.
        if hasattr(iw, 'filter_combo'):
            iw.filter_combo.blockSignals(True)

        # Disable and set filter checkboxes
        # Set only "Has Annotations" checked
        if hasattr(iw, 'filter_combo'):
            for i in range(iw.filter_combo.count()):
                item = iw.filter_combo.model().item(i)
                if item.text() == "Has Annotations":
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)

            iw.filter_combo.setEnabled(False)

        # Unblock signals now that we're done.
        if hasattr(iw, 'filter_combo'):
            iw.filter_combo.blockSignals(False)

        # Disable search UI elements
        if hasattr(iw, 'home_button'):
            iw.home_button.setEnabled(False)
        if hasattr(iw, 'search_bar_images'):
            iw.search_bar_images.setEnabled(False)
        if hasattr(iw, 'search_bar_labels'):
            iw.search_bar_labels.setEnabled(False)

        # Hide the "Current" label as it is not applicable in this dialog
        if hasattr(iw, 'current_image_index_label'):
            iw.current_image_index_label.hide()

        # Disconnect the double-click signal to prevent it from loading an image
        # in the main window, as this dialog is for selection only.
        if hasattr(iw, 'tableView'):
            try:
                iw.tableView.doubleClicked.disconnect()
            except TypeError:
                pass

        # CRITICAL: Override the load_first_filtered_image method to prevent auto-loading
        # This is the key fix to prevent unwanted load_image_by_path calls
        if hasattr(iw, 'load_first_filtered_image'):
            iw.load_first_filtered_image = lambda: None

    def showEvent(self, event):
        """
        Set up the layout when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.thresholds_widget.initialize_thresholds()

        # Update the device
        self.device = self.main_window.device
        # Configure the image window's UI elements for this specific dialog
        self.configure_image_window_for_dialog()
        # Sync with main window's images BEFORE updating labels
        self.sync_image_window()
        # Populate both label dropdowns, restore the last selections, and filter
        # the image table.
        self.update_label_combos()
        self.refresh_prompt()

    def sync_image_window(self):
        """
        Syncs by directly adopting the main manager's up-to-date raster objects,
        avoiding redundant and slow re-calculation of annotation info.
        """
        main_manager = self.main_window.image_window.raster_manager
        dialog_manager = self.image_selection_window.raster_manager

        # Since the main_manager's rasters are always up-to-date, we can
        # simply replace the dialog's raster dictionary and path list entirely.
        # This is a shallow copy of the dictionary, which is extremely fast.
        # The Raster objects themselves are not copied, just referenced.
        dialog_manager.rasters = main_manager.rasters.copy()

        # Update the path list to match the new dictionary of rasters.
        dialog_manager.image_paths = list(dialog_manager.rasters.keys())

    def filter_images_by_label_and_type(self):
        """
        Filters the image list to show only images that contain at least one
        annotation that has BOTH the selected label AND a valid type (Polygon or Rectangle).
        This uses the fast, pre-computed cache for performance.
        """
        annotation_label = self.annotation_label_combo.currentData()

        if annotation_label is not None:
            # Remembered for a better experience on re-opening.
            self.last_annotation_label_code = annotation_label.short_label_code
            if not self._output_label_touched:
                self._follow_annotation_label(annotation_label)

        table_model = self.image_selection_window.table_model
        previously_highlighted = table_model.get_highlighted_paths()

        if not annotation_label:
            # If no label is selected (e.g., during initialization), show an empty list.
            table_model.set_filtered_paths([])
            self._annotation_image_count = 0
            self.update_annotation_note()
            return

        all_paths = self.image_selection_window.raster_manager.image_paths
        final_filtered_paths = []

        valid_types = {"RectangleAnnotation", "PolygonAnnotation"}
        selected_label_code = annotation_label.short_label_code

        # Loop through paths and check the pre-computed map on each raster
        for path in all_paths:
            raster = self.image_selection_window.raster_manager.get_raster(path)
            if not raster:
                continue

            # Skip VideoRasters — VPE generation requires loading a static image
            # from the path, which doesn't work for video files. Users should
            # extract frames first if they want to use video content as references.
            if getattr(raster, 'raster_type', None) == 'VideoRaster':
                continue

            # From the cache, get the set of annotation types for the selected label,
            # and keep the image if any of them is a Polygon or Rectangle.
            types_for_this_label = raster.label_to_types_map.get(selected_label_code, set())
            if not valid_types.isdisjoint(types_for_this_label):
                final_filtered_paths.append(path)

        # Directly set the filtered list in the table model.
        table_model.set_filtered_paths(final_filtered_paths)

        # Keep highlights that are still in the list
        valid_selections = [p for p in previously_highlighted if p in final_filtered_paths]
        if valid_selections:
            table_model.set_highlighted_paths(valid_selections)

        # After filtering, update all labels with the correct counts.
        dialog_iw = self.image_selection_window
        dialog_iw.update_image_count_label(len(final_filtered_paths))  # Set "Total" to filtered count
        dialog_iw.update_current_image_index_label()
        dialog_iw.update_highlighted_count_label()

        self._annotation_image_count = len(final_filtered_paths)
        self.update_annotation_note()

    def run_blocker(self):
        """Why the prompt cannot run yet, or None when it can.

        OK is disabled while this says anything. Highlighted images used to count
        as a prompt here although nothing but "Generate VPEs" ever read them, so OK
        accepted a run that then had nothing to predict with.
        """
        if self.loaded_model is None:
            return "Load a model first."
        if self.output_label is None:
            return "Choose a label under 'Save detections as'."
        if not self.session.has_positives():
            return "Add an example or a phrase to the prompt."
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
        info_label = QLabel("Load a model, then build a prompt on the right from any of:\n"
                            "  •  Images from your annotations: choose a label, highlight images, Add highlighted.\n"
                            "  •  The prompt you tried out in the See Anything tool: Add from Tool.\n"
                            "  •  Phrases: Add phrase.\n"
                            "Untick a row to leave it out. Detections are saved under 'Save detections as'.")
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
        self.left_panel.addWidget(self.thresholds_widget)

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
        # `activated` fires for the user's own choices only, not for programmatic ones.
        self.output_label_combo.activated.connect(self._on_output_label_chosen)
        layout.addRow("Save detections as:", self.output_label_combo)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)

    def setup_model_buttons_layout(self):
        """
        Setup action buttons in a group box.
        """
        group_box = QGroupBox("Actions")
        main_layout = QVBoxLayout()

        button_row = QHBoxLayout()
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        load_button.setToolTip("Load the selected See Anything model.")
        button_row.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        deactivate_button.setToolTip("Unload the current model and clear the prompt.")
        button_row.addWidget(deactivate_button)

        main_layout.addLayout(button_row)

        group_box.setLayout(main_layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_status_layout(self):
        """
        Setup status display in a group box.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_bar = QLabel("No model loaded")
        self.status_bar.setWordWrap(True)
        # The status changes after every action; its length must not resize the columns.
        fixed_width_text(self.status_bar)
        layout.addWidget(self.status_bar)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_prompt_layout(self):
        """The prompt list above, the annotations to add examples from below, at equal heights."""
        # --- The prompt ---
        self.prompt_panel = PromptSessionPanel(
            lambda: self.session,
            stem_source=self._expected_stem,
            imgsz_source=self.get_imgsz,
            title="Prompt",
            empty_text="Empty. Add examples from annotations below, a phrase, or the prompt "
                       "built with the See Anything tool.")

        add_from_tool_button = QPushButton("Add from Tool")
        add_from_tool_button.setToolTip(
            "Add the prompt built with the See Anything tool: its examples, its negative\n"
            "examples, its confidence threshold and its image size. Adding it twice is harmless.")
        add_from_tool_button.clicked.connect(self.import_session_from_tool)
        self.prompt_panel.add_host_widget(add_from_tool_button)

        add_phrase_button = QPushButton("Add phrase...")
        add_phrase_button.setToolTip("Add a text prompt. Inspect ranks phrases against your examples.")
        add_phrase_button.clicked.connect(self.add_phrase_from_input)
        self.prompt_panel.add_host_widget(add_phrase_button)

        self.prompt_panel.edited.connect(self.refresh_prompt)
        self.prompt_panel.saveRequested.connect(self.save_session_file)
        self.prompt_panel.loadRequested.connect(self.load_session_file)
        self.prompt_panel.inspectRequested.connect(self.inspect_prompt)
        # The image-size warnings depend on the spinbox.
        self.imgsz_spinbox.valueChanged.connect(lambda _: self.refresh_prompt())

        # --- Add from annotations ---
        group_box = QGroupBox("Add from annotations")
        layout = QVBoxLayout(group_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label:"))
        self.annotation_label_combo = QComboBox()
        self.annotation_label_combo.setToolTip("Whose annotations to embed. The image table lists images that have them.")
        self.annotation_label_combo.currentIndexChanged.connect(self.filter_images_by_label_and_type)
        row.addWidget(self.annotation_label_combo, 1)
        layout.addLayout(row)

        # How many of these images the prompt already has, so they are not added again.
        self.annotation_note_label = QLabel()
        self.annotation_note_label.setWordWrap(True)
        fixed_width_text(self.annotation_note_label)
        layout.addWidget(self.annotation_note_label)

        # A full ImageWindow instance for image selection
        self.image_selection_window = ImageWindow(self.main_window)
        layout.addWidget(self.image_selection_window, 1)

        self.add_annotations_button = QPushButton("Add highlighted")
        self.add_annotations_button.setToolTip(
            "Embed the label's boxes on each highlighted image as one example per image.\n"
            "An image already in the prompt is only embedded again if its annotations changed.")
        self.add_annotations_button.clicked.connect(self.add_highlighted_annotations)
        layout.addWidget(self.add_annotations_button)

        self.remove_annotations_button = QPushButton("Remove all images from the prompt")
        self.remove_annotations_button.setToolTip(
            "Remove every example added from annotations, of every label.\n"
            "Examples from the tool and phrases stay.")
        self.remove_annotations_button.clicked.connect(self.remove_annotation_examples_from_button)
        layout.addWidget(self.remove_annotations_button)

        # Equal stretch and a size hint neither can grow past, so the two boxes
        # split the column evenly instead of by their contents.
        for widget in (self.prompt_panel, group_box):
            policy = widget.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Ignored)
            widget.setSizePolicy(policy)
            self.right_panel.addWidget(widget, 1)

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
        self.filter_images_by_label_and_type()

    def _follow_annotation_label(self, label):
        index = self.output_label_combo.findText(label.short_label_code)
        if index != -1 and index != self.output_label_combo.currentIndex():
            self.output_label_combo.setCurrentIndex(index)

    def _on_output_label_chosen(self, _index):
        """The user picked an output label, so it stops following the annotation label."""
        self._output_label_touched = True

    def _on_output_label_changed(self, _index=None):
        self.output_label = self.output_label_combo.currentData()
        if self.output_label is not None:
            self.last_output_label_code = self.output_label.short_label_code
        self.update_ok_state()

    # --- Annotations as examples --------------------------------------------------------------------------------------

    def get_label_annotations(self, image_path, label_code):
        """
        Return the bboxes and polygon points of one label's annotations on an image.

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
            if annotation.label.short_label_code != label_code:
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

    def add_highlighted_annotations(self):
        """The Add button: embed the highlighted images' annotations right away."""
        label = self.annotation_label_combo.currentData()
        paths = self.image_selection_window.table_model.get_highlighted_paths()

        if self.loaded_model is None:
            QMessageBox.warning(self, "No Model Loaded", "Load a model before adding examples.")
            return
        if label is None:
            QMessageBox.warning(self, "No Label", "Choose the label whose annotations to add.")
            return
        if not paths:
            QMessageBox.information(self, "No Images Highlighted",
                                    "Highlight one or more images in the table, then press Add highlighted.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Adding Examples")
        progress_bar.show()
        try:
            progress_bar.set_busy_mode("Embedding annotations...")
            report = self.add_from_annotations(paths, label.short_label_code)
        except Exception as e:
            report = None
            QMessageBox.critical(self, "Could Not Add Examples", str(e))
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        if report is not None:
            self.report(report.message(label.short_label_code))

    def update_annotation_note(self):
        """Say, in one short line, how many of the chosen label's images the prompt already has."""
        note = getattr(self, 'annotation_note_label', None)
        if note is None:
            return
        label = self.annotation_label_combo.currentData()
        if label is None:
            note.setText("")
            return

        code = label.short_label_code
        in_prompt = len(self.examples_from_annotations(code))
        listed = self._annotation_image_count
        of = f" of {listed}" if listed is not None and listed >= in_prompt else ""
        note.setText(f"In the prompt: {in_prompt}{of} {code} image{'s' if (listed or in_prompt) != 1 else ''}.")

    def remove_annotation_examples(self):
        """Remove every example added from annotations; tool examples and phrases stay.

        Returns:
            int: How many examples were removed.
        """
        doomed = [p for p in self.session.positives + self.session.negatives if p.source == ORIGIN_ANNOTATIONS]
        for prototype in doomed:
            self.session.remove(prototype.uid)
        self.refresh_prompt()
        return len(doomed)

    def remove_annotation_examples_from_button(self):
        """The Remove all images button, with a confirmation: embedding them again takes time."""
        count = len(self.examples_from_annotations())
        if not count:
            return
        answer = QMessageBox.question(
            self, "Remove Images from the Prompt",
            f"Remove the {count} example{'s' if count != 1 else ''} added from annotations?\n"
            f"Examples from the tool and phrases stay.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        removed = self.remove_annotation_examples()
        self.report(f"Removed {removed} image example{'s' if removed != 1 else ''}; "
                    f"examples from the tool and phrases kept.")

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
                message += ("\n\nThe prompt was cleared: its examples were made with a different "
                            "model and cannot be used with this one.")
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
        self.loaded_model = YOLOE(self.model_path)

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

        return True

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

    # --- The prompt ---------------------------------------------------------------------------------------------------

    def refresh_prompt(self):
        """Redraw the prompt panel, the note and the status; re-check whether the prompt can run."""
        panel = getattr(self, 'prompt_panel', None)
        if panel is not None:
            panel.refresh()
        remove_button = getattr(self, 'remove_annotations_button', None)
        if remove_button is not None:
            remove_button.setEnabled(bool(self.examples_from_annotations()))
        self.update_annotation_note()
        self.update_ok_state()
        self.update_status()

    def report(self, message):
        """Show what was just done, under the model and prompt lines of the status."""
        self._last_action = message or ""
        self.update_status()

    def update_status(self):
        """The model, where the prompt's examples came from, and the last thing done."""
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

    def project_label_names(self):
        """Every project label name, as candidate phrases for text alignment.

        Returns:
            list[str]: Sorted, de-duplicated names, without the "Review" label.
        """
        return project_label_names(getattr(self.main_window, 'label_window', None))

    def text_embedder(self):
        """The cached text encoder for the loaded model, rebuilt when it changes.

        Returns:
            TextEmbedder: Bound to `self.loaded_model`.
        """
        embedder = getattr(self, '_text_embedder', None)
        if embedder is None or embedder.model is not self.loaded_model:
            embedder = TextEmbedder(self.loaded_model)
            self._text_embedder = embedder
        return embedder

    def add_text_prototype(self, phrase):
        """Add a phrase to the prompt, alongside any examples and other phrases.

        A phrase and an example crop produce embeddings of the same kind, so a
        phrase is simply one more positive class, and agnostic NMS keeps one
        detection where a phrase and a crop fire on the same object.

        Args:
            phrase (str): The text prompt to add.

        Returns:
            bool: True if it is now in the prompt.

        Raises:
            RuntimeError: If the model cannot encode text.
        """
        phrase = (phrase or "").strip()
        if not phrase or self.loaded_model is None:
            return False

        if phrase not in self.session.phrases():
            self.session.add_phrase(phrase, self.text_embedder().encode([phrase]))
            if self.session.model_stem is None:
                self.session.model_stem = self._prompt_embedding_stem()
        self.refresh_prompt()
        return True

    def remove_text_prototype(self, phrase):
        """Drop one phrase."""
        self.session.remove_phrase(phrase)
        self.refresh_prompt()

    def text_prototype_phrases(self):
        """The phrases in the prompt, in the order they were added."""
        return self.session.phrases()

    def add_phrase_from_input(self):
        """The Add phrase button."""
        if self.loaded_model is None:
            QMessageBox.warning(self, "No Model Loaded", "Load a model before adding a phrase.")
            return
        phrase, ok = QInputDialog.getText(self, "Add Phrase", "Text prompt:")
        if not ok or not phrase.strip():
            return
        try:
            self.add_text_prototype(phrase)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Add Phrase", str(e))
            return
        self.report(f"Added the phrase '{phrase.strip()}'.")

    def inspect_prompt(self):
        """Plot the prompt's examples and rank phrases against them."""
        try:
            shown = inspect_session(self, self.session, self.loaded_model,
                                    label_names=self.project_label_names(),
                                    prompt_store=self)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could Not Inspect", f"An error occurred: {e}")
            return
        if not shown:
            QMessageBox.information(self, "Nothing to Inspect",
                                    "Add examples to the prompt, or load a model to work from text.")
        self.refresh_prompt()

    def _expected_stem(self):
        """The checkpoint stem a prompt must match: the loaded model's, else the selected one's."""
        stem = self._prompt_embedding_stem()
        if stem:
            return stem
        path = self.model_edit.text().strip() if hasattr(self, 'model_edit') else ""
        if not path and hasattr(self, 'model_combo'):
            path = self.model_combo.currentText()
        return stem_from_path(path)

    def import_session(self, session, source="the See Anything tool"):
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

    def import_session_from_tool(self):
        """Pull the session straight from the See Anything predictor dialog."""
        predictor = getattr(self.main_window, 'see_anything_deploy_predictor_dialog', None)
        if predictor is None or not hasattr(predictor, 'session_for_export'):
            QMessageBox.warning(self, "No Session", "The See Anything tool is not available.")
            return False
        return self.import_session(predictor.session_for_export())

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

        # Embeddings belong to the checkpoint that produced them, so the prompt
        # goes with the model.
        self._text_embedder = None
        self.session = PromptSession()
        self.n_positive_classes = None
        self.refresh_prompt()

        # Clean up references
        gc.collect()
        torch.cuda.empty_cache()

        # Untoggle all tools
        self.main_window.untoggle_all_tools()

        # Update status bar
        self.report("Model deactivated; the prompt went with it.")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
