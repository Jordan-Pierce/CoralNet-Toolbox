import os
import gc

import numpy as np

import torch
from torch.cuda import empty_cache

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSpinBox, QVBoxLayout, QGroupBox, QTabWidget,
                             QWidget, QLineEdit, QFileDialog)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_window_icon
from coralnet_toolbox.paths import resolve_weights

from coralnet_toolbox.SeeAnything.PromptAlignment import (TextEmbedder,
                                                          as_matrix,
                                                          checkpoint_stem,
                                                          promptable,
                                                          rank,
                                                          read_vocabulary_cache,
                                                          to_model_space,
                                                          vocabulary_cache_path,
                                                          vpe_from_prompts)
from coralnet_toolbox.SeeAnything.PromptSession import (ORIGIN_TOOL,
                                                        PromptSession,
                                                        keep_positive_detections,
                                                        stem_from_path)
from coralnet_toolbox.SeeAnything.QtPromptSession import (PromptButtons,
                                                          describe_sources,
                                                          fit_height,
                                                          fixed_width_text,
                                                          level_columns,
                                                          reserve_lines,
                                                          session_signature,
                                                          session_warnings,
                                                          shared_prompt,
                                                          show_warnings,
                                                          warning_label)


# Input-size granularity for YOLOE. Everything in the family is a stride-32
# model, so the letterbox pads to a multiple of this.
IMGSZ_STRIDE = 32

# Lines the status keeps room for: the model, the prompt and the last thing done.
STATUS_LINES = 3

# Shared by both See Anything dialogs so they open on the same model.
# 11s rather than 11l: on THESIA (scripts/SeeAnything_Testbed.py, held out test,
# every annotated image as a reference, 1024) 11s scored mAP50-95 0.183 and 11l
# 0.029 less, worse in 6 of 6 paired runs; 11m was lowest. 11s is also the cheapest.
DEFAULT_MODEL = 'yoloe-11s-seg.pt'

# What the interactive tool produces unless the user says otherwise.
DEFAULT_OUTPUT_TYPE = "Rectangle"


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployPredictorDialog(QDialog):
    # Emitted when the prompt changes other than through the tool (Clear or
    # Load here, or anything the Generator does to the shared prompt), so the
    # tool can re-run its work area. Edits the tool makes itself do not emit:
    # the tool re-runs those on its own and would otherwise run twice.
    sessionEdited = pyqtSignal()

    def __init__(self, main_window, parent=None):
        """Initialize the SeeAnything Deploy Model dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("eye.svg"))
        self.setWindowTitle("See Anything Deploy Model")
        # The width; the height is fitted to the contents when the dialog shows.
        self.resize(820, 600)

        # The prompt the tool builds interactively, shared with the Generator
        # (QtPromptSession.SharedPrompt). Held outside the tool so it outlives
        # tool switches.
        self.prompt = shared_prompt(main_window)
        self.prompt.changed.connect(self._on_prompt_changed)
        # What the prompt was when this dialog last looked, to tell a change
        # made in the Generator from a redraw.
        self._seen_signature = None
        # The last thing done, shown under the model and prompt in the status.
        self._last_action = ""

        # Initialize instance variables
        self.imgsz = 1024
        self.task = "detect"
        self.model_path = None
        self.loaded_model = None
        # Phrase embeddings are checkpoint-specific, so this is rebuilt whenever
        # `loaded_model` changes rather than carried across models.
        self._text_embedder = None
        self.image_path = None
        # The work-area crop exactly as the tool read it. It is handed to the
        # model unresized; ultralytics letterboxes it and inverts that transform
        # for us (see predict_from_prompts).
        self.original_image = None
        # Text prompt set from the tool (Ctrl+T), or None for box prompts
        self.text_prompt = None

        self.class_mapping = {}

        # Information across the top, then two columns (landscape);
        # self.layout is the layout being filled
        root = QVBoxLayout(self)
        self.layout = root
        # Setup the info layout
        self.setup_info_layout()

        # Equal-width columns that end level with each other
        columns = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        root.addLayout(columns)

        # What to load on the left
        self.layout = left
        self.setup_models_layout()
        self.setup_parameters_layout()
        self.setup_sam_layout()

        # What it finds on the right
        self.layout = right
        self.setup_thresholds_layout()
        level_columns(left, right)

        # The actions in one row, then where things stand, across the full width
        self.layout = root
        self.setup_buttons_layout()
        self.setup_status_layout()

        self.refresh_session_state()

    @property
    def session(self):
        """The shared prompt's session: the tool's examples, and whatever the Generator added."""
        return self.prompt.session

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.thresholds_widget.initialize_thresholds()
        # Room for the usual status lines, then no spare height.
        QWidget.layout(self).activate()
        self.reserve_space()
        fit_height(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Text wraps differently at another width, so the room it needs changes.
        self.reserve_space()

    def reserve_space(self):
        """Keep room for the usual status lines, so loading a model or adding examples changes text, not height."""
        if hasattr(self, 'status_bar'):
            reserve_lines(self.status_bar, STATUS_LINES)

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel(
            "Load a model, then select the See Anything tool:\n"
            "  •  Left-click to start a work area, move, left-click to finish (Space uses the current view).\n"
            "  •  Draw 2 or 3 boxes around examples and press Space, or press Ctrl+T to type a phrase.\n"
            "  •  Ctrl+click a detection for more like it, Ctrl+right-click for fewer; "
            "Ctrl+wheel changes the confidence.\n"
            "The See Anything Generator shares this prompt, to run it over many images."
        )
        info_label.setTextFormat(Qt.PlainText)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

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
        standard_models = [
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
            'yoloe-26x-seg.pt'
        ]

        # Add all models to combo box
        self.model_combo.addItems(standard_models)

        # Set the default model
        self.model_combo.setCurrentIndex(standard_models.index(DEFAULT_MODEL))
        self.model_combo.setToolTip("Choose a See Anything (YOLOE) model variant.\nSmall: faster. Large: slower, and not always more accurate.\nDefault: 11s, which did as well as or better than 11m and 11l on the data tested.")
        model_select_layout.addRow("Model:", self.model_combo)

        tab_widget.addTab(model_select_tab, "Select Model")

        # Tab 2: Use existing model (custom weights)
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
        self.layout.addWidget(group_box)

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

        # Output type. This replaces the old detect/segment "Task" dropdown,
        # which named the model's mode rather than what the user gets. The task
        # is derived from it (see get_task), so there is one control instead of
        # two that could disagree. Same three options as the SAM tool's dialog.
        self.output_type_dropdown = QComboBox()
        self.output_type_dropdown.addItems(["Polygon", "Rectangle", "Mask"])
        self.output_type_dropdown.setCurrentText(DEFAULT_OUTPUT_TYPE)
        self.output_type_dropdown.setToolTip(
            "Format for See Anything output annotations.\n"
            "Polygon: Free-form shapes (segmentation).\n"
            "Rectangle: Bounding boxes (detection, fastest).\n"
            "Mask: Segmentation painted into the image's raster mask.")
        layout.addRow("Output Type", self.output_type_dropdown)

        # The greyed-out "Resize Image" dropdown that used to sit here is gone.
        # It offered a choice that was never taken -- it was disabled -- and it
        # described a manual resize that no longer happens: ultralytics
        # letterboxes the work-area crop itself.

        # Image size control. The old 512-65536 range in steps of 1024 offered
        # sizes no GPU can run; YOLOE's cost grows with the square of this.
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(IMGSZ_STRIDE)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.imgsz_spinbox.setToolTip("Input image size for the model.\nLarger sizes improve accuracy but consume more GPU memory.\n"
                                      f"Rounded to a multiple of {IMGSZ_STRIDE}.")
        layout.addRow("Image Size (imgsz)", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_sam_layout(self):
        """Use SAM model for segmentation."""
        group_box = QGroupBox("Use SAM to Create Polygons")
        layout = QFormLayout()

        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        self.use_sam_dropdown.setToolTip("Use SAM to refine See Anything detections into high-quality polygons.\nRequires SAM model to be deployed first.")
        layout.addRow("Use SAM Polygons:", self.use_sam_dropdown)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_thresholds_layout(self):
        """
        Setup thresholds control section in a group box.
        """
        # Add ThresholdsWidget for all threshold controls. Boundary detections
        # is shown because the See Anything tool reads it when it maps SAM's
        # masks out of the work area; leaving it hidden meant the setting was
        # in force but not visible anywhere.
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_boundary=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
        )

        self.layout.addWidget(self.thresholds_widget)

    # --- Prompt session ------------------------------------------------------------------------------------------------

    def session_stem(self):
        """The checkpoint stem session embeddings are bound to right now."""
        if self.loaded_model is not None:
            stem = checkpoint_stem(self.loaded_model)
            if stem:
                return stem
        path = self.model_edit.text().strip() if hasattr(self, 'model_edit') else ""
        if not path and hasattr(self, 'model_combo'):
            path = self.model_combo.currentText()
        return stem_from_path(path)

    def _adopt_model_stem(self):
        """Bind the session to the model just loaded, dropping incompatible examples.

        Returns:
            bool: True if examples were dropped because they belong to another model.
        """
        stem = self.session_stem()
        dropped = (not self.session.is_empty()
                   and self.session.model_stem is not None
                   and stem != self.session.model_stem)
        if dropped:
            self.session.clear()
        self.session.model_stem = stem
        self.refresh_session_state()
        return dropped

    def refresh_session_state(self):
        """Redraw the buttons and the status, and tell the Generator the prompt may have changed."""
        self._refresh_view()
        self._seen_signature = session_signature(self.session)
        self.prompt.notify(self)

    def _refresh_view(self):
        """Redraw the buttons and the status from the shared prompt."""
        if not hasattr(self, 'prompt_buttons') or not hasattr(self, 'status_bar'):
            return
        self.prompt_buttons.refresh(self.session)
        self.update_status()

    def _on_prompt_changed(self, source):
        """The Generator changed or redrew the shared prompt: redraw, and re-run the tool if it changed."""
        if source is self:
            return
        self.text_prompt = self.session.text_phrase()
        self._refresh_view()
        signature = session_signature(self.session)
        if signature != self._seen_signature:
            self._seen_signature = signature
            self.sessionEdited.emit()

    def _claim_session(self):
        """Whether the tool may add to the shared prompt with this dialog's model.

        An empty prompt takes this model's checkpoint. One made with another
        checkpoint (the Generator loaded a different model) refuses, and says
        why: its embeddings mean nothing to this model.
        """
        stem = self.session_stem()
        if self.session.is_empty():
            self.session.model_stem = stem
            return True
        try:
            self.session.check_stem(stem)
        except ValueError:
            QMessageBox.warning(self.annotation_window, "Different Model",
                                f"The prompt is shared with the See Anything Generator and was made with "
                                f"'{self.session.model_stem}', but this dialog has '{stem}' loaded. Load the same "
                                f"model in both, or clear the prompt.")
            return False
        if self.session.model_stem is None:
            self.session.model_stem = stem
        return True

    def report(self, message):
        """Show what was just done, under the model and prompt lines of the status."""
        self._last_action = message or ""
        self.update_status()

    def update_status(self):
        """The model, where the session's examples came from, the last thing done, and any warnings.

        Refreshed on every edit, the tool's included, so it always matches the session.
        """
        if not hasattr(self, 'status_bar'):
            return
        if self.loaded_model is None:
            lines = ["No model loaded."]
        else:
            lines = [f"Model: {os.path.basename(str(self.model_path))}"]
        lines.append(f"Prompt: {describe_sources(self.session)}")
        if self._last_action:
            lines.append(self._last_action)
        self.status_bar.setText("\n".join(lines))
        # The tool uses every example as it was made, so it does not re-embed.
        show_warnings(self.warning_label, session_warnings(self.session, stem=self.session_stem(),
                                                           imgsz=self.get_imgsz(), reembeds=False))
        # Room is kept for the usual lines; anything longer (warnings) needs more
        # height, and is never squeezed.
        if self.isVisible():
            fit_height(self, grow_only=True)

    def _on_clear_requested(self):
        """The Clear Prompt button: forget every example, and let the tool re-run its work area."""
        self.session.clear()
        self.text_prompt = None
        self.refresh_session_state()
        self.sessionEdited.emit()

    def clear_session(self):
        """Forget every example (the tool's Ctrl+Shift+Backspace)."""
        self.session.clear()
        self.text_prompt = None
        self.refresh_session_state()

    def add_session_positive(self, kind, embedding, label):
        """Add a positive example from the tool. Does not emit `sessionEdited`.

        Stamped with the image size it was embedded at, so the Generator can
        warn when it runs at another.

        Returns:
            Prototype | None: The example, or None if the prompt belongs to another model.
        """
        if not self._claim_session():
            return None
        prototype = self.session.add_positive(kind, embedding, label,
                                              imgsz=self.get_imgsz(),
                                              origin={"source": ORIGIN_TOOL})
        self.refresh_session_state()
        return prototype

    def add_session_negative(self, embedding, label):
        """Add a negative example from the tool. Does not emit `sessionEdited`."""
        if not self._claim_session():
            return None
        prototype = self.session.add_negative(embedding, label,
                                              imgsz=self.get_imgsz(),
                                              origin={"source": ORIGIN_TOOL})
        self.refresh_session_state()
        return prototype

    def session_for_export(self):
        """A copy of the session carrying the threshold in force now."""
        session = self.session.copy()
        session.model_stem = session.model_stem or self.session_stem()
        session.confidence = self.main_window.get_uncertainty_thresh()
        session.imgsz = self.get_imgsz()
        return session

    def save_session(self):
        if not self.session.positives:
            QMessageBox.information(self, "Nothing to Save",
                                    "The prompt has no positive examples yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Prompt", "", "Prompt (*.npz)")
        if not path:
            return
        try:
            written = self.session_for_export().to_npz(path)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Save Prompt", str(e))
            return
        count = len(self.session.positives) + len(self.session.negatives)
        self.report(f"Prompt saved to {os.path.basename(written)}.")
        QMessageBox.information(self, "Prompt Saved",
                                f"Saved {count} example{'s' if count != 1 else ''} to:\n{written}")

    def load_session(self):
        """Load Prompt: add a saved prompt's examples to the shared prompt, as the Generator's Load does.

        Added rather than replacing, so loading here does not wipe what the
        Generator calibrated; Clear Prompt first to start over. The file's
        confidence threshold is applied.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load Prompt", "",
                                              "Prompt or prompt embeddings (*.npz)")
        if not path:
            return
        try:
            session = PromptSession.from_npz(path, expected_stem=self.session_stem())
            session.model_stem = session.model_stem or self.session_stem()
            if self.session.is_empty():
                self.session.model_stem = session.model_stem
            else:
                self.session.check_stem(session.model_stem)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Load Prompt", str(e))
            return

        offered = len(session.positives) + len(session.negatives)
        added = self.session.merge(session)
        self.text_prompt = self.session.text_phrase()
        if session.confidence is not None:
            self.main_window.update_uncertainty_thresh(session.confidence)
        self.refresh_session_state()
        self.sessionEdited.emit()
        name = os.path.basename(path)
        if added == offered:
            message = f"Added {added} example{'s' if added != 1 else ''} from {name}."
        else:
            message = (f"Added {added} of {offered} examples from {name}; "
                       f"the others were already in the prompt.")
        self.report(message)
        QMessageBox.information(self, "Prompt Loaded", message)

    def embed_boxes(self, bboxes, masks=None):
        """One embedding for a set of boxes drawn together on the current image.

        The boxes share `cls=0`, so they merge into a single example exactly as a
        box prompt always has -- this is the embedding `predict_from_prompts` would
        have used internally.

        Returns:
            torch.Tensor | None: Shape (1, 1, D).
        """
        if self.loaded_model is None or self.original_image is None:
            return None
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return None
        return vpe_from_prompts(self.loaded_model, self.original_image,
                                self.build_prompts(bboxes, masks),
                                **self._embedding_args())

    def _embedding_args(self):
        """Size, device and precision to extract embeddings at.

        They must be the ones predictions run at: `get_vpe` letterboxes at the
        predictor's own image size, and an embedding taken at 640 is not the one
        an in-image prompt produces at 1024.
        """
        return dict(imgsz=self.get_imgsz(),
                    device=self.main_window.device,
                    quantize=self.get_quantize())

    def embed_detections(self, boxes):
        """One embedding per box, in a single forward pass.

        Distinct `cls` values keep the boxes in separate channels rather than
        merging them. Measured: six boxes in 0.21 s on a CPU.

        Returns:
            list[torch.Tensor]: One (D,) vector per box, in order.
        """
        if self.loaded_model is None or self.original_image is None:
            return []
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return []
        vpe = vpe_from_prompts(self.loaded_model, self.original_image,
                               dict(bboxes=boxes, cls=np.arange(len(boxes), dtype=np.float32)),
                               **self._embedding_args())
        if vpe is None:
            return []
        return list(as_matrix([vpe]))

    def predict_from_session(self, conf=None):
        """Predict on the current image from every enabled example in the session.

        The session's positives and decoys become the model's classes, then a
        prompt-free predict runs -- the same path the Generator takes, and measured
        to produce exactly what in-image visual prompts produce. Decoy detections
        are removed before returning; class IDs are left for the caller.

        Args:
            conf (float, optional): Confidence floor for the prediction. Defaults to
                the threshold widget's value.

        Returns:
            list | None: Ultralytics Results, or None if there is nothing to run.
        """
        if not self.loaded_model or self.original_image is None:
            return None
        if not self.session.has_positives():
            return None
        try:
            self.session.check_stem(self.session_stem())
        except ValueError:
            self.main_window.status_bar.showMessage(
                f"The prompt was made with '{self.session.model_stem}', not this model: load the same "
                f"model here and in the See Anything Generator, or clear the prompt.", 8000)
            return None

        self.task = self.get_task()

        try:
            names, embeddings, n_positive = self.session.build_classes()
            self.loaded_model.is_fused = lambda: False
            self.loaded_model.set_classes(names, to_model_space(self.loaded_model, embeddings))
            if conf is None:
                conf = self.thresholds_widget.get_uncertainty_thresh()
            results = self.loaded_model.predict(self.original_image,
                                                imgsz=self.get_imgsz(),
                                                rect=True,
                                                conf=conf,
                                                iou=self.thresholds_widget.get_iou_thresh(),
                                                max_det=self.thresholds_widget.get_max_detections(),
                                                device=self.main_window.device,
                                                quantize=self.get_quantize(),
                                                retina_masks=self.task == "segment")
        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            return None

        return [keep_positive_detections(result, n_positive) for result in results]

    def setup_buttons_layout(self):
        """
        Setup the action buttons in one row across the dialog: the model, then the prompt.

        Examples are added in the tool itself (more like this, fewer like this),
        so there is no list of them here; Clear Prompt removes them all.
        """
        group_box = QGroupBox("Actions")
        layout = QHBoxLayout()

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        load_button.setToolTip("Load the selected See Anything (YOLOE) model for inference.")

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        deactivate_button.setToolTip("Unload the current See Anything model and free GPU memory.\n"
                                     "The prompt is kept.")

        self.prompt_buttons = PromptButtons("Remove every example, here and in the See Anything Generator\n"
                                            "(Ctrl+Shift+Backspace in the tool).")
        self.prompt_buttons.clearRequested.connect(self._on_clear_requested)
        self.prompt_buttons.saveRequested.connect(self.save_session)
        self.prompt_buttons.loadRequested.connect(self.load_session)

        # Equal shares of the width, however wide the dialog is.
        for button in [load_button, deactivate_button] + self.prompt_buttons.buttons:
            layout.addWidget(button, 1)

        # The image-size warnings depend on the spinbox.
        self.imgsz_spinbox.valueChanged.connect(lambda _: self.refresh_session_state())

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """
        Setup status display in a group box.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_bar = QLabel("No model loaded")
        self.status_bar.setWordWrap(True)
        # The status changes after every edit; its length must not resize the columns.
        fixed_width_text(self.status_bar)
        layout.addWidget(self.status_bar)

        self.warning_label = warning_label()
        layout.addWidget(self.warning_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        self.update_status()

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the dropdown state accordingly.

        The See Anything image size is no longer tied to SAM's. That coupling
        existed while SAM was handed this dialog's resized image; it now encodes
        the native work-area crop at its own size, so pinning the two together
        only capped YOLOE at SAM's ceiling.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            # Signals blocked: this is connected to currentIndexChanged, so
            # resetting the dropdown here re-entered the method and showed the
            # error a second time.
            self.use_sam_dropdown.blockSignals(True)
            self.use_sam_dropdown.setCurrentText("False")
            self.use_sam_dropdown.blockSignals(False)
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        # SAM refines boxes into masks, so a Rectangle output would throw away
        # the only thing SAM adds. Move to Polygon rather than running both.
        if (self.use_sam_dropdown.currentText() == "True"
                and self.get_output_type() == "Rectangle"):
            self.output_type_dropdown.setCurrentText("Polygon")

        return True

    def load_model(self):
        """
        Load the selected model (from dropdown or file).
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Obtaining model...", 3000)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
    
        try:
            # Update the task
            self.task = self.get_task()
            
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
            visuals = dict(
                bboxes=np.array(
                    [
                        [120, 425, 160, 445],  # Random box
                    ],
                ),
                cls=np.array(
                    np.zeros(1),
                ),
            )
    
            # Run a dummy prediction to load the model. Warm up at the precision
            # and with the predictor class the real calls use: ultralytics
            # rebuilds the predictor when either changes between calls.
            self.loaded_model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                visual_prompts=visuals.copy(),  # This needs to happen to properly initialize the predictor
                predictor=self.get_predictor_class(),
                imgsz=640,
                conf=0.99,
                device=self.main_window.device,
                quantize=self.get_quantize(),
            )
            # Finish the progress bar
            progress_bar.finish_progress()
            # Session embeddings belong to one checkpoint; examples collected with
            # another model are meaningless to this one.
            dropped = self._adopt_model_stem()
            # Update the status bar
            self.report("Model loaded." if not dropped else
                        "Model loaded; the prompt was cleared, its examples were made with another model.")
            message = "Model loaded successfully"
            if dropped:
                message += ("\n\nThe prompt, which the See Anything Generator shares, was cleared: its "
                            "examples were made with a different model and cannot be used with this one.")
            QMessageBox.information(self, "Model Loaded", message)
            # The dialog has done its job; leaving it up meant it reappeared
            # behind the message box and had to be dismissed a second time.
            # A failed load keeps it open instead, so the choice can be retried.
            self.accept()

        except Exception as e:
            self.loaded_model = None
            self.report(f"Error loading model: {os.path.basename(str(self.model_path))}")
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")
    
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None
            
    def get_output_type(self):
        """Return the annotation format the tool should produce."""
        return self.output_type_dropdown.currentText()

    def get_task(self):
        """Return the ultralytics task the current output type needs.

        Rectangle output only needs boxes, so it runs the detection head and
        skips mask prediction entirely. Polygon and Mask both need masks.
        """
        return "detect" if self.get_output_type() == "Rectangle" else "segment"

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

        FP16 on the CPU is slower than FP32, not faster. This is the rule the
        SAM dialogs use, so the two stay comparable.
        """
        on_cpu = str(self.main_window.device).strip().lower() == "cpu"
        return 32 if on_cpu else 16

    def get_predictor_class(self):
        """Return the visual-prompt predictor class to run with.

        Always the segmentation predictor, including for Rectangle output.
        Every YOLOE checkpoint offered here is a `-seg` model, and a `-seg`
        model's raw output is nested one level deeper than the detection
        predictor expects: `non_max_suppression` unwraps the outer tuple once
        and then hits `prediction.shape[-1]` on the inner one, raising
        "'tuple' object has no attribute 'shape'" before any result is built.
        Only `SegmentationPredictor.postprocess` unpacks it correctly.

        The mask head therefore runs whatever the output type is -- that is a
        property of the checkpoint, not a choice. What Rectangle output does
        save is `retina_masks`, which stays off for it (see predict_from_prompts).
        """
        return YOLOEVPSegPredictor

    def set_image(self, image, image_path):
        """
        Set the image in the predictor.

        The array is kept exactly as it was read. It used to be pre-resized
        here, with the prompts scaled to match. That saved nothing -- predict
        mode already letterboxes to the long side with minimum padding, so
        ultralytics would have scaled the crop to the same size by itself --
        and it cost correctness: the two axes were scaled by slightly different
        factors, each independently rounded to a multiple of 32, which every
        ultralytics inverse transform then unwound as a single uniform gain. So
        masks drifted further off the further they sat from the centre of the
        work area. Predictions now run on the native crop.
        """
        if image is None:
            # There was a fallback here that read the image itself via
            # `image_window.rasterio_images`, an attribute that exists nowhere
            # in the codebase -- so it raised AttributeError rather than
            # recovering. Nothing reaches it (every caller passes a real array),
            # and it would have supplied RGB where this predictor needs the
            # BGR ultralytics documents. Fail clearly instead of pretending.
            raise ValueError("set_image requires an image array; got None")

        self.original_image = image
        self.image_path = image_path

    def build_prompts(self, bboxes, masks=None):
        """Assemble the ultralytics visual-prompt dict from work-area coordinates.

        No coordinate scaling happens here. `YOLOEVPDetectPredictor` rasterizes
        these prompts against the letterboxed batch itself, applying the same
        gain and padding it applied to the image, so prompts must arrive in the
        source image's own pixels.

        `cls` is all zeros: See Anything is deliberately single-class.
        """
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]

        # Set the predictor task
        self.task = self.get_task()

        visual_prompts = {
            'bboxes': bboxes,
            'cls': np.zeros(len(bboxes))
        }
        if self.task == 'segment':
            if masks:
                visual_prompts['masks'] = [np.asarray(m, dtype=np.float32) for m in masks]
            else:  # Fallback to creating masks from bboxes if no masks are provided
                fallback_masks = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    fallback_masks.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                                   dtype=np.float32))
                visual_prompts['masks'] = fallback_masks

        return visual_prompts

    def set_text_prompt(self, text):
        """Set (or clear) the text prompt used in place of box prompts.

        Ultralytics turns the phrase into a class embedding with
        `YOLOE.get_text_pe`, which `set_classes` accepts exactly like a visual
        prompt embedding. One phrase only: See Anything stays single-class.

        Args:
            text (str | None): The phrase, or None/empty to go back to boxes.

        Returns:
            bool: True if the model is now prompted by this text.
        """
        text = (text or "").strip()
        self.text_prompt = text or None

        if self.loaded_model is None or self.text_prompt is None:
            # Clearing the phrase clears it from the session too, so the next
            # prediction does not quietly keep using it.
            self.session.set_text(None)
            self.refresh_session_state()
            return False

        # A fused head cannot take new prompts; unfuse before setting classes,
        # exactly as the Generator does for its VPEs.
        self.loaded_model.is_fused = lambda: False
        # Through the embedder rather than `YOLOE.get_text_pe`, which leaves
        # `cache_clip_model` False and so rebuilds the 572 MB MobileCLIP encoder
        # on every phrase -- 0.40 s a call against 0.08 s when it is kept.
        if not self._claim_session():
            self.text_prompt = None
            return False
        phrase_vector = self.text_embedder().encode([self.text_prompt])
        embeddings = to_model_space(self.loaded_model, phrase_vector.unsqueeze(0))
        self.loaded_model.set_classes([self.text_prompt], embeddings)
        # The phrase is one of the session's examples: the tool predicts from the
        # session, and the Generator receives it with everything else.
        self.session.set_text(self.text_prompt, phrase_vector)
        self.refresh_session_state()
        return True

    def text_embedder(self):
        """The cached text encoder for the loaded model, rebuilt when it changes.

        Embeddings are checkpoint-specific -- `get_tpe` runs every phrase through
        that model's own `reprta` head -- so the cache is discarded whenever the
        model is.

        Returns:
            TextEmbedder: Bound to `self.loaded_model`.
        """
        embedder = getattr(self, '_text_embedder', None)
        if embedder is None or embedder.model is not self.loaded_model:
            embedder = TextEmbedder(self.loaded_model)
            self._text_embedder = embedder
        return embedder

    def suggest_text_prompts(self, bboxes, masks=None, candidates=None, top_k=12):
        """Rank phrases by how well they match the boxes the user just drew.

        This answers "what do I type to find this?", which is not the same
        question as "what is this?" -- a crop of a bus ranks `vehicle` above
        `bus`. The winner is whatever drives the model, which is what a text
        prompt needs to be.

        Extracting a VPE runs a warm-up predict that resets the head to a single
        visual class, so any text prompt already in force is re-applied before
        returning.

        Args:
            bboxes (np.ndarray): Boxes in work-area pixels.
            masks (list, optional): Polygons, when the boxes came from them.
            candidates (list[str], optional): Phrases to rank. Defaults to the
                cached vocabulary when one has been built, and nothing otherwise.
            top_k (int): How many to return.

        Returns:
            list[dict]: As `PromptAlignment.rank` returns, best first. Empty when
            there is nothing to rank or the model cannot encode text.
        """
        if self.loaded_model is None or self.original_image is None:
            return []
        if not promptable(self.loaded_model):
            return []

        if candidates is None:
            candidates = self.cached_vocabulary()
        candidates = [c for c in dict.fromkeys(candidates or []) if c]
        if not candidates:
            return []

        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return []

        active_text = self.text_prompt
        try:
            vpe = vpe_from_prompts(self.loaded_model,
                                   self.original_image,
                                   self.build_prompts(bboxes, masks),
                                   **self._embedding_args())
            if vpe is None:
                return []
            return rank(as_matrix([vpe]),
                        self.text_embedder().encode(candidates),
                        candidates,
                        top_k=top_k)
        finally:
            if active_text:
                # The warm-up above replaced the head's classes; put the phrase back.
                try:
                    self.set_text_prompt(active_text)
                except Exception:
                    pass

    def cached_vocabulary(self):
        """The prompt-free vocabulary, but only if it is already on disk.

        Building it downloads a checkpoint and encodes 4,585 words, which is a
        reasonable thing to offer behind a button in the Generator and an
        unreasonable thing to trigger from a keyboard shortcut. So the tool gets
        the full vocabulary once the Generator has paid for it, and the project's
        own labels until then.

        Returns:
            list[str]: The vocabulary, or an empty list if it is not cached.
        """
        if self.loaded_model is None:
            return []
        stem = checkpoint_stem(self.loaded_model)
        cached = read_vocabulary_cache(vocabulary_cache_path(stem))
        if cached is None:
            return []
        names, embeddings = cached
        self.text_embedder().prime(names, embeddings)
        return names

    def predict_from_prompts(self, bboxes, masks=None):
        """
        Make predictions using the currently loaded model using prompts.

        Runs on the native work-area crop, so ultralytics letterboxes it itself
        and `scale_boxes`/`scale_masks` invert that transform exactly. Results
        therefore come back in work-area pixels and need no rescaling by the
        caller.

        `rect=True` is passed explicitly even though predict mode already
        defaults to it: minimum-padding letterboxing is what makes the geometry
        above hold, so it is stated rather than inherited.

        Args:
            bboxes (np.ndarray): The bounding boxes to use as prompts, in
                work-area pixel coordinates.
            masks (list, optional): A list of polygons to use as prompts for
                segmentation, in the same coordinates.

        Returns:
            results (Results): Ultralytics Results object
        """
        if not self.loaded_model:
            QMessageBox.critical(self.annotation_window,
                                 "Model Not Loaded",
                                 "Model not loaded, cannot make predictions")
            return None

        if self.original_image is None:
            QMessageBox.critical(self.annotation_window,
                                 "No Image Set",
                                 "No image set for the See Anything predictor.")
            return None

        if not len(bboxes):
            return None

        visual_prompts = self.build_prompts(bboxes, masks)

        try:
            # Make predictions
            results = self.loaded_model.predict(self.original_image,
                                                visual_prompts=visual_prompts,
                                                predictor=self.get_predictor_class(),
                                                imgsz=self.get_imgsz(),
                                                rect=True,
                                                conf=self.thresholds_widget.get_uncertainty_thresh(),
                                                iou=self.thresholds_widget.get_iou_thresh(),
                                                max_det=self.thresholds_widget.get_max_detections(),
                                                device=self.main_window.device,
                                                quantize=self.get_quantize(),
                                                retina_masks=self.task == "segment")

        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            results = None

        return results

    def predict_from_text(self):
        """Predict from the current text prompt instead of drawn boxes.

        `set_text_prompt` has already put the phrase's embedding on the model,
        so this is a plain prompt-free forward pass.

        Returns:
            list | None: Ultralytics Results, or None if there is nothing to run.
        """
        if not self.loaded_model or self.text_prompt is None:
            return None

        if self.original_image is None:
            QMessageBox.critical(self.annotation_window,
                                 "No Image Set",
                                 "No image set for the See Anything predictor.")
            return None

        self.task = self.get_task()

        try:
            results = self.loaded_model.predict(self.original_image,
                                                imgsz=self.get_imgsz(),
                                                rect=True,
                                                conf=self.thresholds_widget.get_uncertainty_thresh(),
                                                iou=self.thresholds_widget.get_iou_thresh(),
                                                max_det=self.thresholds_widget.get_max_detections(),
                                                device=self.main_window.device,
                                                quantize=self.get_quantize(),
                                                retina_masks=self.task == "segment")
        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            results = None

        return results

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        # Clear the model
        self.loaded_model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None
        # Phrase embeddings belong to the checkpoint that produced them
        self._text_embedder = None
        # The prompt is kept: the Generator shares it, and it still fits the next
        # model of the same kind. A different model clears it when it loads.
        self.text_prompt = self.session.text_phrase()
        self.refresh_session_state()
        # Clear the cache
        gc.collect()
        empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update the status bar
        self.report("Model deactivated; the prompt is kept.")
        QMessageBox.information(self.annotation_window, "Model Deactivated", "Model deactivated")
