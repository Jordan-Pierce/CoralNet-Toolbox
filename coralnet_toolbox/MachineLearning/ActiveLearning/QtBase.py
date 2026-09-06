"""Active Learning: annotate a little, train, predict, review, repeat.

The loop this drives is the one a person actually performs when building a
detector from scratch. Label a handful of objects, train something small, let it
propose the rest, correct what it got wrong, and train again on the corrections.
Each round the model does more of the work and the person does less.

Every piece already existed -- training, inference, the confidence window, the
Explorer -- but the round trip between them ran through several dialogs and an
exported dataset on disk. This dialog is the loop itself: it trains straight
from the project (see InPlaceTraining), predicts onto the images most worth
predicting on, and keeps a record of whether the model is still improving.

Two rules matter more than the rest:

  * Training uses **verified annotations only**. An unverified annotation is a
    model prediction, and training on those teaches the model its own guesses.
    The failure is quiet -- metrics look fine while the model converges on its
    own errors -- so it is the default rather than an option.
  * Predictions arrive **unverified**, however confident. That already falls
    out of update_machine_confidence, so this dialog does not have to arrange
    it -- but the loop depends on it, so it is pinned by a test rather than
    assumed.

The dialog is split across two tabs because it did not otherwise fit on a
1080p screen -- seven stacked group boxes wanted 1265 px and could not shrink
below 1047. Setup is what you decide before the first round; Session is what
changes as rounds run. Train and the ready line sit outside both, since a Train
button that disappears when you switch tabs would be worse than the height was.

Scope: detection and instance segmentation, plain image rasters. See
ACTIVE_LEARNING_PLAN.md.
"""

import warnings

import os
import gc
import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
                             QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from coralnet_toolbox.Common.QtThresholdsWidget import ThresholdsWidget

from coralnet_toolbox.MachineLearning import InPlaceTraining
from coralnet_toolbox.MachineLearning.TrainModel.QtBase import TrainModelWorker

from coralnet_toolbox.Results.ResultsProcessor import ResultsProcessor

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Status-bar updates are recounted rather than incremented, so they are
# debounced: a bulk label change would otherwise trigger one full recount per
# annotation.
STATUS_DEBOUNCE_MS = 300

TASK_LABELS = {'detect': "Detection", 'segment': "Segmentation"}

# Splits are fixed rather than exposed, and there is no test split.
#
# A held-out test set is right for a one-off training run and wrong here. Nothing
# in a round ever evaluates it -- Ultralytics trains on train and reports on val,
# and the post-training test pass needs a real folder on disk that in-place
# training does not have -- so a test split would only take annotations away from
# a budget that is small by definition and give nothing back. That is why the
# Test column always read zero: there was never anything to put in it.
TRAIN_RATIO = 0.80
VAL_RATIO = 1.0 - TRAIN_RATIO

# The parameters the Train Model dialog sends, with the same defaults, so a
# round trains the same way a manual run would. Hardcoded for now; the dialog
# may expose them later.
#
# workers=0 is not a preference. In-place training reads its labels from a class
# attribute on the dataset, and a DataLoader worker is a separate process that
# would not have it -- on Windows, where workers are spawned rather than forked,
# every worker would come up with an empty registry.
TRAINING_DEFAULTS = {
    'patience': 30,
    'workers': 0,
    'optimizer': 'auto',
    'cache': False,
    'save': True,
    'save_period': -1,
    'single_cls': False,
    'mask_ratio': 4,
    'dropout': 0.0,
    'freeze_layers': 0.0,
    # True to match Train Model. The in-place dataset folds weighted sampling in
    # through WeightedInMemoryDataset, which is what the MRO composition test
    # covers, so this costs nothing here.
    'weighted': True,
    'val': True,
    'verbose': True,
    'exist_ok': True,
    # Plots cost time on every round, but "is this still improving" is the
    # question the loop exists to answer, and the confusion matrix answers it
    # better than the mAP column alone.
    'plots': True,
}

# Small models by default. Rounds are meant to be cheap enough to run often;
# a large model turns a five-minute loop into an afternoon.
MODELS = {
    'detect': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo8n.pt'],
    'segment': ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt'],
}

# Nothing is unloaded before a round. Taking away a SAM or See Anything model
# the user deployed for their own work is a bigger cost than the VRAM it frees:
# it is invisible until they go to use it, and they have to redeploy by hand.
# Cached blocks are released instead, which frees memory without touching
# anyone's model, and an out-of-memory failure is met with the batch-size retry.

# Substrings that mean "ran out of memory" across torch, CUDA and the allocator.
OOM_MARKERS = ('out of memory', 'cuda oom', 'cublas_status_alloc_failed',
               'not enough memory', 'defaultcpuallocator')

# Columns of the Training Data table, following ExportDataset's.
(COL_INCLUDE, COL_LABEL, COL_VERIFIED, COL_TRAIN,
 COL_VAL, COL_IMAGES, COL_AWAITING) = range(7)

TABLE_HEADERS = ["Include", "Label", "Verified", "Train", "Val", "Images", "Awaiting"]

EMPTY_SPLIT_COLOR = QColor(255, 220, 220)
FILLED_SPLIT_COLOR = QColor(220, 255, 220)


def bool_combo(default=True, tooltip=""):
    """A True / False combo, matching how the Train Model dialog asks."""
    combo = QComboBox()
    combo.addItems(["True", "False"])
    combo.setCurrentIndex(0 if default else 1)
    if tooltip:
        combo.setToolTip(tooltip)
    return combo


class Base(QDialog):
    """Drives rounds of train-then-predict against the open project.

    ``task`` is a class attribute rather than something chosen in the dialog:
    the layout is built against it during __init__, and the menu already asks
    which task you want. Each task therefore gets its own dialog instance, and
    with it its own round history -- a detection session and a segmentation
    session are different experiments and should not share a scoreboard.
    """

    # Set by the subclasses; read during __init__, so it must be class-level.
    task = None

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle(f"Active Learning {TASK_LABELS.get(self.task, self.task)}")
        self.resize(940, 700)

        self.in_place_dataset = None
        self.worker = None
        self.round_history = []
        self.last_model_path = None
        self.disagreements = []
        self.ready_status = False
        self.last_round_outcome = None

        # Recount is debounced; see STATUS_DEBOUNCE_MS.
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(STATUS_DEBOUNCE_MS)
        self._status_timer.timeout.connect(self.update_status_message)
        self._monitoring = False
        self._populating_table = False

        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_setup_tab(), "Setup")
        self.tabs.addTab(self.create_session_tab(), "Session")
        self.tabs.setTabToolTip(0, "What to train on, and with what.")
        self.tabs.setTabToolTip(1, "What each round produced, and what needs your attention.")
        self.layout.addWidget(self.tabs, 1)

        self.setup_buttons_layout()
        self.load_models()
        self.fit_to_screen()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def fit_to_screen(self):
        """Shrink to fit a small display rather than running off the bottom."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.resize(min(self.width(), int(available.width() * 0.9)),
                    min(self.height(), int(available.height() * 0.9)))

    def create_setup_tab(self):
        """Everything decided before the first round, in two columns."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left = QVBoxLayout()
        left.addWidget(self.create_info_group())
        left.addWidget(self.create_model_group())
        left.addWidget(self.create_thresholds_group())
        left.addStretch()

        right = QVBoxLayout()
        right.addWidget(self.create_data_group(), 1)

        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        return widget

    def create_session_tab(self):
        """Everything that changes as rounds run, in two columns."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left = QVBoxLayout()
        left.addWidget(self.create_next_step_group())
        left.addWidget(self.create_round_group())
        left.addWidget(self.create_history_group(), 1)

        right = QVBoxLayout()
        right.addWidget(self.create_disagreement_group(), 1)

        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        return widget

    def create_info_group(self):
        """The Information box every dialog in this application opens with."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Train on the annotations you have already confirmed, then let the model propose "
            "more for you to review. Nothing is exported: training reads the project directly.\n"
            "Predictions always arrive unverified, so a later round never trains on an earlier "
            "round's guesses."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        info_label.setToolTip("Each round trains a model, predicts onto the images most worth\n"
                              "labelling next, and leaves those predictions for you to confirm.")
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        return group_box

    def create_data_group(self):
        """What will be trained on, laid out like the Export Dataset table.

        The table replaces a summary line and a collapsed label list. Both hid
        the thing that decides whether a round is worth running: which labels
        actually have enough confirmed examples to land in every split.
        """
        group_box = QGroupBox("Training Data")
        layout = QVBoxLayout()

        self.label_table = QTableWidget(0, len(TABLE_HEADERS))
        self.label_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.label_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.label_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.setToolTip(
            "Verified counts only -- unverified annotations are model predictions, and\n"
            "training on those would teach the model its own guesses.\n"
            "Awaiting is how many predictions of that label are still unreviewed:\n"
            "it is the answer to 'is there enough new material to train again?'.\n"
            "A red split cell means that label has no examples there.\n"
            f"Images are split {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val by a hash of "
            "each image's path, so an image never moves between them from one round to the "
            "next -- which would leak and inflate every metric after it.")
        header = self.label_table.horizontalHeader()
        header.setSectionResizeMode(COL_LABEL, QHeaderView.Stretch)
        for column in (COL_INCLUDE, COL_VERIFIED, COL_TRAIN,
                       COL_VAL, COL_IMAGES, COL_AWAITING):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.label_table, 1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Re-read the project and recount what would be trained on.")
        self.refresh_button.clicked.connect(self.refresh_dataset)
        layout.addWidget(self.refresh_button)

        group_box.setLayout(layout)
        return group_box

    def create_model_group(self):
        """The model and the few training parameters a round needs."""
        group_box = QGroupBox("Model")
        layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setToolTip(
            "A nano model is the default on purpose: rounds are only useful if they\n"
            "are cheap enough to run often. Reach for a larger one once the labels\n"
            "have settled.")
        layout.addRow("Model:", self.model_combo)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(30)
        self.epochs_spinbox.setToolTip("Epochs per round. Short rounds beat one long one early on.")
        layout.addRow("Epochs:", self.epochs_spinbox)

        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(64, 4096)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setToolTip(
            "Images are resized to this before the model sees them.\n"
            "Small objects in large images may be lost.")
        layout.addRow("Image Size:", self.imgsz_spinbox)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 256)
        self.batch_spinbox.setValue(4)
        self.batch_spinbox.setToolTip("Images per batch. Lower this first if training runs out of memory.")
        layout.addRow("Batch:", self.batch_spinbox)

        self.warm_start_combo = bool_combo(
            True,
            "Start each round from the weights the last one produced rather than from\n"
            "scratch, which makes later rounds much cheaper.\n"
            "Set to False to train from the selected model every round.")
        layout.addRow("Warm Start:", self.warm_start_combo)

        self.free_gpu_combo = bool_combo(
            True,
            "Release cached GPU memory before training.\n"
            "Deployed models are left alone -- only memory nothing is using is freed.\n"
            "If a round still runs out of memory, it offers to retry at a smaller batch.")
        layout.addRow("Free GPU:", self.free_gpu_combo)

        group_box.setLayout(layout)
        return group_box

    def create_thresholds_group(self):
        """The application-wide thresholds, set here instead of elsewhere.

        Shared with the rest of the application rather than kept as a second
        copy, so the two cannot disagree. Area is shown because it silently
        filters this dialog's predictions too -- a round that adds nothing when
        the area bounds exclude every detection should not be a mystery.
        """
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_uncertainty=True,
            show_iou=True,
            show_area=True,
            title="Thresholds",
            parent=self)
        return self.thresholds_widget

    def create_round_group(self):
        """What happens after training finishes."""
        group_box = QGroupBox("After Training")
        layout = QFormLayout()

        self.predict_combo = bool_combo(
            True,
            "Run the freshly trained model on un-reviewed images so the next round of\n"
            "review is waiting for you when training finishes.")
        layout.addRow("Predict:", self.predict_combo)

        self.budget_spinbox = QSpinBox()
        self.budget_spinbox.setRange(1, 100000)
        self.budget_spinbox.setValue(10)
        self.budget_spinbox.setToolTip(
            "How many un-reviewed images to predict on.\n"
            "Predicting on everything is rarely worth it: a handful of well-chosen images\n"
            "teaches the model more per minute of your attention than hundreds of easy ones.")
        layout.addRow("Image Budget:", self.budget_spinbox)

        self.audit_combo = bool_combo(
            True,
            "Re-run the model over images you have already reviewed and report where it\n"
            "confidently predicts a different label than the one you confirmed.\n"
            "Uses the same budget, and covers a different slice of images each round.")
        layout.addRow("Disagreements:", self.audit_combo)

        self.deploy_combo = bool_combo(
            True,
            "Load the round's weights into the matching Deploy Model dialog when it\n"
            "finishes, so the model can be used with Batch Inference and the tools\n"
            "rather than only inside this session.\n"
            "Note that the next round unloads it again if Free GPU is True.")
        layout.addRow("Deploy Model:", self.deploy_combo)

        group_box.setLayout(layout)
        return group_box

    def create_next_step_group(self):
        """What the last round produced, and the one thing to do about it.

        A round takes minutes and used to end in an eight-second status message.
        Worse, nothing said where the predictions went: the user had to guess
        which images had changed. This panel is the hand-off.
        """
        group_box = QGroupBox("Next Step")
        layout = QVBoxLayout()

        self.next_step_label = QLabel(
            "No rounds yet. Train one from the Setup tab when the data looks right.")
        self.next_step_label.setWordWrap(True)
        layout.addWidget(self.next_step_label)

        button_layout = QHBoxLayout()

        self.review_button = QPushButton("Review Predictions")
        self.review_button.setToolTip(
            "Filter the Image Window to images carrying annotations nobody has confirmed\n"
            "yet, and open the first one. Unconfirmed annotations are drawn with a black\n"
            "outline on the canvas.")
        self.review_button.clicked.connect(self.review_predictions)
        self.review_button.setEnabled(False)
        button_layout.addWidget(self.review_button)

        self.deploy_button = QPushButton("Deploy This Model")
        self.deploy_button.setToolTip(
            "Load the last round's weights into the matching Deploy Model dialog.")
        self.deploy_button.clicked.connect(self.deploy_last_model)
        self.deploy_button.setEnabled(False)
        button_layout.addWidget(self.deploy_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        group_box.setLayout(layout)
        return group_box

    def create_history_group(self):
        """The per-round record."""
        group_box = QGroupBox("Rounds")
        layout = QVBoxLayout()

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Round", "Train Images", "Annotations", "mAP50"])
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setToolTip(
            "Whether the model is still improving is the question that decides when to stop.\n"
            "A round that adds annotations but not accuracy is telling you something.")
        header = self.history_table.horizontalHeader()
        for index in range(4):
            header.setSectionResizeMode(index, QHeaderView.Stretch)
        layout.addWidget(self.history_table)

        group_box.setLayout(layout)
        return group_box

    def create_disagreement_group(self):
        """The queue of places the model and the user do not agree.

        Worth its own surface because it is the highest-value thing a round
        produces. Another easy positive confirms what is already known; a
        confident prediction of *sand* where a person confirmed *coral* is
        either a model failure or a labelling error, and both are worth more
        attention than the queue of easy ones.
        """
        group_box = QGroupBox("Disagreements")
        layout = QVBoxLayout()

        self.disagreement_label = QLabel("Nothing checked yet.")
        self.disagreement_label.setWordWrap(True)
        layout.addWidget(self.disagreement_label)

        self.disagreement_table = QTableWidget(0, 4)
        self.disagreement_table.setHorizontalHeaderLabels(
            ["Image", "You Confirmed", "Model Says", "Conf"])
        self.disagreement_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.disagreement_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.disagreement_table.verticalHeader().setVisible(False)
        self.disagreement_table.setToolTip(
            "Double-click a row to open that image with the annotation selected.")
        header = self.disagreement_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for index in (1, 2, 3):
            header.setSectionResizeMode(index, QHeaderView.ResizeToContents)
        self.disagreement_table.cellDoubleClicked.connect(self.on_disagreement_activated)
        layout.addWidget(self.disagreement_table, 1)

        group_box.setLayout(layout)
        return group_box

    def setup_buttons_layout(self):
        """The action row, outside the tabs so it never goes out of reach."""
        button_layout = QHBoxLayout()

        self.ready_label = QLabel("❌ Not Ready")
        self.ready_label.setToolTip("Whether a round can be started with the current selection.")
        button_layout.addWidget(self.ready_label)

        button_layout.addStretch()

        # Both actions in the bottom-right corner, where a dialog's actions live.
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self.train_button = QPushButton("Train Round")
        self.train_button.setToolTip("Train on the verified annotations, then predict if enabled.")
        self.train_button.clicked.connect(self.start_round)
        self.train_button.setEnabled(False)
        self.buttons.addButton(self.train_button, QDialogButtonBox.ActionRole)
        self.buttons.rejected.connect(self.reject)
        button_layout.addWidget(self.buttons)

        self.layout.addLayout(button_layout)

    def load_models(self):
        """Fill the model combo for this dialog's task."""
        self.model_combo.clear()
        self.model_combo.addItems(MODELS.get(self.task, []))
        self.model_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Reading the project
    # ------------------------------------------------------------------

    def showEvent(self, event):
        """Read the project and start reporting progress when opened."""
        super().showEvent(event)
        self.thresholds_widget.initialize_thresholds()
        self.start_monitoring()
        self.refresh_dataset()

    def closeEvent(self, event):
        """Stop reporting progress when the dialog goes away."""
        self.stop_monitoring()
        super().closeEvent(event)

    def reject(self):
        self.stop_monitoring()
        super().reject()

    def start_monitoring(self):
        """Report review progress to the status bar as annotations change.

        Connected to the manager's aggregate signals rather than to each
        annotation's verifiedChanged: a project can hold hundreds of thousands
        of annotations, and one connection each is not a reasonable price for a
        status line.
        """
        if self._monitoring:
            return
        manager = self.main_window.annotation_manager
        for signal in (manager.annotationsAdded, manager.annotationsRemoved,
                       manager.annotationLabelChanged, manager.annotationModified):
            try:
                signal.connect(self.schedule_status_update)
            except (TypeError, AttributeError):
                pass
        self._monitoring = True
        self.update_status_message()

    def stop_monitoring(self):
        """Disconnect the status-bar reporting."""
        if not self._monitoring:
            return
        manager = self.main_window.annotation_manager
        for signal in (manager.annotationsAdded, manager.annotationsRemoved,
                       manager.annotationLabelChanged, manager.annotationModified):
            try:
                signal.disconnect(self.schedule_status_update)
            except (TypeError, RuntimeError):
                pass
        self._monitoring = False

    def schedule_status_update(self, *_args):
        """Coalesce a burst of annotation changes into one recount."""
        self._status_timer.start()

    def update_status_message(self):
        """Say how much review is left before another round is worth running.

        Phrased around the decision rather than as a bare countdown: what a
        person wants to know is whether there is enough new material to justify
        training again.
        """
        verified = 0
        unverified = 0
        for annotation in self.annotation_counts():
            if getattr(annotation, 'verified', True):
                verified += 1
            else:
                unverified += 1

        if unverified:
            message = (f"Active Learning: {verified} verified · "
                       f"{unverified} awaiting review")
        else:
            message = f"Active Learning: {verified} verified · nothing awaiting review"

        try:
            self.main_window.status_bar.showMessage(message, 8000)
        except Exception:
            pass

    def annotation_counts(self):
        """Yield annotations relevant to this dialog's task."""
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        for annotation in self.annotation_window.annotations_dict.values():
            if isinstance(annotation, allowed_types):
                yield annotation

    def project_annotations(self):
        """Verified, labelled annotations for this task, grouped by image path.

        Only plain image rasters: a video frame is a virtual path the trainer
        cannot open, and an orthomosaic is one enormous sample that needs tiling
        before it means anything.
        """
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        raster_manager = self.image_window.raster_manager

        grouped = {}
        for annotation in self.annotation_window.annotations_dict.values():
            if not isinstance(annotation, allowed_types):
                continue
            if not getattr(annotation, 'verified', True):
                continue
            if annotation.label is None or annotation.label.short_label_code == 'Review':
                continue

            raster = raster_manager.get_raster(annotation.image_path)
            if raster is None or getattr(raster, 'raster_type', '') != 'ImageRaster':
                continue

            grouped.setdefault(annotation.image_path, []).append(annotation)

        return grouped

    def awaiting_counts(self):
        """Unverified annotations of this task, counted per label.

        The count that answers "is another round worth running yet?", which the
        verified columns cannot: they go up only after the review is done.
        """
        counts = {}
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        for annotation in self.annotation_window.annotations_dict.values():
            if not isinstance(annotation, allowed_types):
                continue
            if getattr(annotation, 'verified', True):
                continue
            if annotation.label is None:
                continue
            code = annotation.label.short_label_code
            counts[code] = counts.get(code, 0) + 1
        return counts

    @staticmethod
    def tally(grouped, groups):
        """Per-label counts of verified annotations, overall and per split.

        Returns (verified, images, per_split) where per_split maps a split name
        to {label: count}.
        """
        verified = {}
        images = {}
        for image_path, annotations in grouped.items():
            for annotation in annotations:
                code = annotation.label.short_label_code
                verified[code] = verified.get(code, 0) + 1
                images.setdefault(code, set()).add(image_path)

        per_split = {}
        for split, image_paths in groups.items():
            counts = {}
            for image_path in image_paths:
                for annotation in grouped.get(image_path, []):
                    code = annotation.label.short_label_code
                    counts[code] = counts.get(code, 0) + 1
            per_split[split] = counts

        return verified, {code: len(paths) for code, paths in images.items()}, per_split

    def refresh_dataset(self):
        """Rebuild the preview of what a round would train on."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            grouped = self.project_annotations()
            awaiting = self.awaiting_counts()

            # Split first, so the table can show where each label actually lands
            # rather than only how many of it there are.
            InPlaceTraining.set_split_ratios(TRAIN_RATIO, VAL_RATIO)
            groups = InPlaceTraining.group_images_by_split(
                sorted(grouped), TRAIN_RATIO, VAL_RATIO,
                overrides=self.split_overrides(grouped))

            verified, images, per_split = self.tally(grouped, groups)
            self.populate_label_table(verified, images, per_split, awaiting)

            selected = self.selected_labels()
            if not selected:
                self.set_not_ready("No verified annotations for this task yet.")
                return

            label_to_index = {code: index for index, code in enumerate(selected)}
            raster_manager = self.image_window.raster_manager

            def dimensions_for(image_path):
                raster = raster_manager.get_raster(image_path)
                return raster.height, raster.width

            records_by_split = {
                split: InPlaceTraining.build_records(
                    image_paths, grouped, label_to_index, self.task, dimensions_for)
                for split, image_paths in groups.items()
            }

            self.in_place_dataset = InPlaceTraining.InPlaceDataset(
                self.task, records_by_split, selected)

            ready, reason = self.readiness(self.in_place_dataset)
            self.ready_status = ready
            self.ready_label.setText("✅ Ready" if ready else f"❌ Not Ready - {reason}")
            self.train_button.setEnabled(ready)

        except Exception as e:
            self.set_not_ready(f"Could not read the project: {e}")
            print(f"Error reading project for Active Learning: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def split_overrides(self, grouped):
        """Per-image split assignments a user pinned on the raster."""
        raster_manager = self.image_window.raster_manager
        overrides = {}
        for image_path in grouped:
            raster = raster_manager.get_raster(image_path)
            override = getattr(raster, 'split_override', None) if raster else None
            if override:
                overrides[image_path] = override
        return overrides

    def set_not_ready(self, message):
        """Report why a round cannot run, and make sure it cannot be started.

        Every path that fails to produce a dataset goes through here: leaving
        Train enabled after one of them offers a button that can only fail.
        """
        self.in_place_dataset = None
        self.ready_status = False
        self.ready_label.setText(f"❌ Not Ready - {message}")
        self.train_button.setEnabled(False)

    def readiness(self, dataset):
        """Return (ready, reason) for a prepared dataset.

        Only hard blockers. A label missing from a split is shown in red but
        does not block: early rounds legitimately have a rare class absent from
        validation, and refusing to train then would disable the feature in
        exactly the situation it exists for.

        Reasons name a remedy, because the ones that can occur here are not
        fixed by trying again. Splits are derived from the image paths, so an
        empty split stays empty however many times Refresh is pressed -- and on
        a small project it is not a rare accident: roughly one in ten ten-image
        projects has no validation split at 70/20/10.
        """
        total = sum(dataset.image_count(split) for split in ('train', 'val', 'test'))

        if dataset.image_count('train') == 0:
            return False, self.split_advice("no training images", total)
        if VAL_RATIO > 0 and dataset.image_count('val') == 0:
            return False, self.split_advice("no validation images", total)
        if sum(dataset.annotation_count(s) for s in ('train', 'val', 'test')) == 0:
            return False, "no annotations on the included labels"
        return True, ""

    @staticmethod
    def split_advice(reason, image_count):
        """Attach a remedy to an empty-split reason.

        Splits are stable per image path on purpose -- it is what stops an image
        migrating from train into validation between rounds and quietly
        inflating every metric after it. The cost is that a small project can
        land badly and stay there, so the way out has to be stated rather than
        guessed at.
        """
        if image_count == 0:
            return "no images carry verified annotations for this task yet"
        if image_count < 20:
            return (f"{reason} ({image_count} images split by path, so Refresh will not "
                    f"change it - annotate more images)")
        return reason

    # ------------------------------------------------------------------
    # The training-data table
    # ------------------------------------------------------------------

    @staticmethod
    def order_labels(verified, awaiting):
        """The order rows appear in, which is also the order of class indices.

        Deterministic and stated once, because ``selected_labels`` reads it back
        as the class-index order the model is trained with, and
        ``trained_labels`` maps detections back through the same order. A
        reshuffle between those two points would attach predictions to the wrong
        labels, silently.
        """
        codes = set(verified) | set(awaiting)
        return sorted(codes, key=lambda code: (-verified.get(code, 0), code))

    def populate_label_table(self, verified, images, per_split, awaiting):
        """Rebuild the table, keeping anything the user has already unticked."""
        unchecked = self.unchecked_labels()

        self._populating_table = True
        self.label_table.setUpdatesEnabled(False)
        try:
            self.label_table.setRowCount(0)
            for code in self.order_labels(verified, awaiting):
                verified_count = verified.get(code, 0)
                row = self.label_table.rowCount()
                self.label_table.insertRow(row)

                # A label with nothing confirmed cannot be trained on: ticking it
                # would add a class index with no examples behind it.
                trainable = verified_count > 0
                checkbox = QCheckBox()
                checkbox.setChecked(trainable and code not in unchecked)
                checkbox.setEnabled(trainable)
                if not trainable:
                    checkbox.setToolTip(
                        "Nothing confirmed for this label yet, so there is nothing to train on.\n"
                        "Review some of its predictions first.")
                checkbox.stateChanged.connect(self.on_label_toggled)
                self.label_table.setCellWidget(row, COL_INCLUDE, self.centered(checkbox))

                self.label_table.setItem(row, COL_LABEL, self.centered_item(code))
                self.label_table.setItem(row, COL_VERIFIED, self.centered_item(verified_count))
                for column, split in ((COL_TRAIN, 'train'), (COL_VAL, 'val')):
                    self.label_table.setItem(
                        row, column, self.centered_item(per_split.get(split, {}).get(code, 0)))
                self.label_table.setItem(row, COL_IMAGES, self.centered_item(images.get(code, 0)))
                self.label_table.setItem(row, COL_AWAITING,
                                         self.centered_item(awaiting.get(code, 0)))
        finally:
            self.label_table.setUpdatesEnabled(True)
            self._populating_table = False

        self.color_split_cells()

    def color_split_cells(self):
        """Red where an included label has no examples in an active split."""
        ratios = {'train': TRAIN_RATIO, 'val': VAL_RATIO}

        for row in range(self.label_table.rowCount()):
            included = self.row_is_checked(row)
            for column, split in ((COL_TRAIN, 'train'), (COL_VAL, 'val')):
                item = self.label_table.item(row, column)
                if item is None:
                    continue
                empty = included and ratios[split] > 0 and item.text() == "0"
                item.setBackground(QBrush(EMPTY_SPLIT_COLOR if empty else FILLED_SPLIT_COLOR))
                item.setForeground(QBrush(QColor(0, 0, 0)))

    def on_label_toggled(self, *_args):
        """Re-read the project when the included labels change."""
        if self._populating_table:
            return
        self.refresh_dataset()

    @staticmethod
    def centered(widget):
        """Wrap a widget so it sits centred in its cell."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return container

    @staticmethod
    def centered_item(text):
        """A read-only, centred table item."""
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def row_is_checked(self, row):
        """Whether the Include box on this row is ticked."""
        container = self.label_table.cellWidget(row, COL_INCLUDE)
        if container is None:
            return False
        checkbox = container.findChild(QCheckBox)
        return bool(checkbox is not None and checkbox.isChecked())

    def unchecked_labels(self):
        """Label codes the user has deliberately excluded, so a refresh keeps them.

        Only rows whose checkbox is enabled count: a row disabled for having
        nothing confirmed is not a choice the user made, and treating it as one
        would leave the label excluded after they review it.
        """
        excluded = set()
        for row in range(self.label_table.rowCount()):
            container = self.label_table.cellWidget(row, COL_INCLUDE)
            item = self.label_table.item(row, COL_LABEL)
            if container is None or item is None:
                continue
            checkbox = container.findChild(QCheckBox)
            if checkbox is not None and checkbox.isEnabled() and not checkbox.isChecked():
                excluded.add(item.text())
        return excluded

    def selected_labels(self):
        """Ticked label codes, in the row order that fixes their class indices."""
        return [self.label_table.item(row, COL_LABEL).text()
                for row in range(self.label_table.rowCount())
                if self.row_is_checked(row) and self.label_table.item(row, COL_LABEL) is not None]

    # ------------------------------------------------------------------
    # GPU sequencing
    # ------------------------------------------------------------------

    @staticmethod
    def free_gpu_memory():
        """Release cached GPU memory, leaving every deployed model in place.

        An earlier version unloaded every deploy dialog before training. That
        traded a problem the user can see -- an out-of-memory error, which the
        round already offers to retry at a smaller batch -- for one they cannot:
        a SAM or See Anything model silently gone, discovered only when they
        went to use it, and needing a manual redeploy. Freeing the allocator's
        unused blocks costs nobody anything.
        """
        gc.collect()
        try:
            from torch.cuda import empty_cache
            empty_cache()
        except Exception:
            pass

    @staticmethod
    def looks_like_oom(message):
        """Whether a training failure was the card running out of memory."""
        text = str(message).lower()
        return any(marker in text for marker in OOM_MARKERS)

    # ------------------------------------------------------------------
    # A round
    # ------------------------------------------------------------------

    def start_round(self):
        """Train on the current verified annotations."""
        if self.worker is not None:
            QMessageBox.information(self, "Already Training", "A round is already running.")
            return

        self.refresh_dataset()
        if self.in_place_dataset is None:
            QMessageBox.warning(self, "Nothing to Train On",
                                "There are no verified annotations for this task yet.")
            return

        ready, reason = self.readiness(self.in_place_dataset)
        if not ready:
            QMessageBox.warning(self, "Not Ready", f"Cannot train: {reason}.")
            return

        dataset = self.in_place_dataset
        # A fresh object next round, so a second run cannot reuse scaffolding
        # this one is about to delete.
        self.in_place_dataset = None

        try:
            data_path = dataset.prepare()
        except Exception as e:
            QMessageBox.critical(self, "Failed to Prepare Dataset", f"{e}")
            return

        round_number = len(self.round_history) + 1
        run_name = f"round_{round_number:02d}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

        model = self.model_combo.currentText()
        if self.warm_start_combo.currentText() == "True" and self.last_model_path:
            if os.path.isfile(self.last_model_path):
                model = self.last_model_path

        params = self.training_params(data_path, model, run_name)
        params['in_place_dataset'] = dataset

        self._pending = {
            'round': round_number,
            'dataset': dataset,
            'run_dir': os.path.join(params['project'], run_name),
            # The model's class names are exactly these, in this order, so the
            # prediction pass needs them to map detections back onto labels.
            'labels': list(dataset.names),
        }

        if self.free_gpu_combo.currentText() == "True":
            self.free_gpu_memory()

        self.train_button.setEnabled(False)
        self.train_button.setText(f"Training round {round_number}...")
        # The rounds fill in on the other tab, so send the user there to watch.
        self.tabs.setCurrentIndex(1)
        self.show_status(f"Active Learning: training round {round_number}...")

        self.worker = TrainModelWorker(params, self.main_window.device)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def training_params(self, data_path, model, run_name):
        """The parameter set for one round.

        Deliberately the same set the Train Model dialog sends, so a round is
        not a differently-configured kind of training that happens to share a
        worker. The values are fixed for now; the ones a round actually wants to
        vary -- model, epochs, image size, batch -- are the ones on the dialog.
        """
        params = dict(TRAINING_DEFAULTS)
        params.update({
            'task': self.task,
            'data': data_path,
            'model': model,
            # Absolute on purpose. Ultralytics resolves a relative `project`
            # under its own runs directory -- the round would land in
            # runs/detect/Data/ActiveLearning/... while everything here looked
            # in Data/ActiveLearning/..., so results.csv and best.pt were never
            # found and the whole post-round chain silently did nothing.
            'project': os.path.abspath(os.path.join('Data', 'ActiveLearning')),
            'name': run_name,
            'epochs': self.epochs_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'batch': self.batch_spinbox.value(),
        })
        return params

    def on_training_error(self, message):
        """A failed round must leave the session usable.

        Out of memory is the expected failure rather than an exotic one, and
        halving the batch is what a person does next anyway -- so it is offered
        here instead of leaving them to find the spinbox.
        """
        self.finish_round()

        batch = self.batch_spinbox.value()
        if self.looks_like_oom(message) and batch > 1:
            reply = QMessageBox.question(
                self, "Out of Memory",
                f"Training ran out of memory at batch {batch}.\n\n"
                f"Retry this round at batch {batch // 2}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.batch_spinbox.setValue(batch // 2)
                self.start_round()
                return

        QMessageBox.critical(self, "Training Failed",
                             f"{message}\n\nThe round was abandoned; your annotations are unchanged.")

    def on_training_completed(self):
        """Record the round, then predict if that was asked for."""
        pending = getattr(self, '_pending', None)
        weights = None
        if pending:
            weights = os.path.join(pending['run_dir'], 'weights', 'best.pt')
            if os.path.isfile(weights):
                self.last_model_path = weights
            else:
                weights = None

            self.record_round(pending, weights)

        self.finish_round()

        # Start the outcome fresh: the passes below fill in what they produced,
        # and a stale count from the previous round would read as this one's.
        self.last_round_outcome = {
            'round': pending['round'] if pending else len(self.round_history),
            'map50': self.round_history[-1]['map50'] if self.round_history else None,
            'weights': weights,
            'predictions': 0,
            'predicted_on': 0,
            'disagreements': 0,
            'deployed': False,
        }

        if weights:
            if self.deploy_combo.currentText() == "True":
                self.last_round_outcome['deployed'] = self.deploy_model(weights, quiet=True)
            self.after_training(weights)

        self.update_next_step()
        self.update_status_message()
        # The summary is on this tab, so land the user where the answer is.
        self.tabs.setCurrentIndex(1)

    def record_outcome(self, **fields):
        """Fold what a post-round pass produced into this round's summary."""
        if self.last_round_outcome is None:
            return
        self.last_round_outcome.update(fields)

    @staticmethod
    def next_step_message(outcome):
        """What the round produced, and what to do about it.

        Written as an instruction rather than a report. The counts alone leave
        the user to work out that unconfirmed predictions are the input to the
        next round, which is the one thing the loop depends on them doing -- and
        the round that produces nothing needs to say so most of all, since that
        is the one where it is least obvious what went wrong.
        """
        parts = [f"Round {outcome['round']} finished"]
        if outcome.get('map50') is not None:
            parts.append(f"mAP50 {outcome['map50']:.3f}")
        if outcome.get('deployed'):
            parts.append("model deployed")
        headline = " · ".join(parts) + "."

        predictions = outcome.get('predictions', 0)
        disagreements = outcome.get('disagreements', 0)

        if predictions:
            detail = (f"{predictions} predictions are waiting on "
                      f"{outcome.get('predicted_on', 0)} images. ")
            action = ("Next: press Review Predictions, confirm or correct what you see, "
                      "then train another round.")
        elif outcome.get('weights'):
            detail = "No new predictions were added. "
            action = ("Next: annotate more images, or raise the Image Budget, "
                      "then train another round.")
        else:
            detail = "Training produced no weights. "
            action = "Next: check the console output for what went wrong."

        if disagreements:
            detail += (f"{disagreements} disagreements need a look - the model contradicts "
                       f"a label you confirmed. ")

        return headline + " " + detail + action

    def update_next_step(self):
        """Put the round's summary on the panel, and enable what it offers."""
        outcome = self.last_round_outcome
        if not outcome:
            return

        self.next_step_label.setText(self.next_step_message(outcome))
        self.review_button.setEnabled(bool(outcome.get('predictions')))
        self.deploy_button.setEnabled(bool(outcome.get('weights')))

    def review_predictions(self):
        """Take the user to the images that are waiting on them.

        The filter is the point: without it, "23 predictions across 10 images"
        leaves them to work out which ten out of a project of thousands.
        """
        try:
            self.image_window.filter_combo.check_item("Needs Review")
        except Exception as e:
            print(f"Warning: could not apply the Needs Review filter: {e}")

        # Synchronous on purpose: the threaded path returns before the table
        # model has been updated, so filtered_paths would still be the previous
        # filter's answer.
        self.image_window.filter_images(use_threading=False)
        paths = list(self.image_window.table_model.filtered_paths)
        if not paths:
            self.show_status("Active Learning: nothing is awaiting review.")
            return

        try:
            self.image_window.load_image_by_path(paths[0])
        except Exception as e:
            print(f"Warning: could not open the first image awaiting review: {e}")

        self.show_status(f"Active Learning: {len(paths)} images awaiting review. "
                         f"Unconfirmed annotations are drawn with a black outline.")

    def deploy_target(self):
        """The Deploy Model dialog matching this dialog's task."""
        attribute = f"{self.task}_deploy_model_dialog"
        return getattr(self.main_window, attribute, None)

    def deploy_last_model(self):
        """Deploy the most recent round's weights, reporting either way."""
        weights = self.last_model_path
        if not weights or not os.path.isfile(weights):
            QMessageBox.information(self, "No Model Yet",
                                    "No round has produced weights to deploy.")
            return
        if self.deploy_model(weights):
            self.update_next_step()

    def deploy_model(self, weights, quiet=False):
        """Load `weights` into the task's Deploy Model dialog.

        Uses the dialog's own load_model so the class-name table, the label
        mapping and the status text all end up in the state they would be in
        had the user loaded it by hand -- a half-loaded dialog claiming to hold
        a model it never mapped is worse than not deploying at all.
        """
        if not weights or not os.path.isfile(weights):
            # Not defensive padding: Ultralytics treats an unresolvable path as
            # a hub model rather than an error, and the failure that follows is
            # not one a try block catches.
            print(f"Warning: refusing to deploy a missing weights file: {weights!r}")
            if not quiet:
                QMessageBox.warning(self, "Cannot Deploy",
                                    f"The weights file no longer exists:\n{weights}")
            return False

        dialog = self.deploy_target()
        if dialog is None:
            if not quiet:
                QMessageBox.warning(self, "Cannot Deploy",
                                    "No Deploy Model dialog for this task.")
            return False

        try:
            dialog.model_path = weights
            dialog.load_model()
        except Exception as e:
            print(f"Warning: could not deploy the round's model: {e}")
            if not quiet:
                QMessageBox.warning(self, "Deploy Failed", f"Could not deploy the model: {e}")
            return False

        if getattr(dialog, 'loaded_model', None) is None:
            return False

        self.show_status(f"Active Learning: deployed {os.path.basename(weights)}.")
        return True

    def finish_round(self):
        """Release the worker and re-enable the button."""
        self.worker = None
        self._pending = None
        self.train_button.setEnabled(True)
        self.train_button.setText("Train Round")

    def record_round(self, pending, weights):
        """Add a row describing what this round trained on and how it scored."""
        dataset = pending['dataset']
        metric = self.read_metric(pending['run_dir'])

        self.round_history.append({
            'round': pending['round'],
            'train_images': dataset.image_count('train'),
            'annotations': dataset.annotation_count('train'),
            'map50': metric,
            'weights': weights,
            'labels': pending['labels'],
        })

        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        for column, value in enumerate((pending['round'],
                                        dataset.image_count('train'),
                                        dataset.annotation_count('train'),
                                        f"{metric:.3f}" if metric is not None else "-")):
            self.history_table.setItem(row, column, self.centered_item(value))

    @staticmethod
    def read_metric(run_dir):
        """Read mAP50 out of the run's results.csv, or None when unavailable."""
        results_path = os.path.join(run_dir, 'results.csv')
        if not os.path.isfile(results_path):
            return None
        try:
            with open(results_path, 'r') as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
            if len(lines) < 2:
                return None
            header = [column.strip() for column in lines[0].split(',')]
            index = next(i for i, name in enumerate(header) if 'mAP50' in name and '95' not in name)
            return float(lines[-1].split(',')[index])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def candidate_images(self, budget):
        """Choose which un-reviewed images are worth spending the budget on.

        A deliberately simple ranking for now: images with nothing on them
        first, then those with least. It needs no precomputed state, which
        matters because the alternative -- ranking by distance from the labelled
        set in the Explorer's embedding index -- only works once features have
        been extracted, and cannot be the default until that is guaranteed.
        """
        raster_manager = self.image_window.raster_manager

        scored = []
        for image_path in raster_manager.image_paths:
            raster = raster_manager.get_raster(image_path)
            if raster is None or getattr(raster, 'raster_type', '') != 'ImageRaster':
                continue

            annotations = self.annotation_window.get_image_annotations(image_path)
            unverified = sum(1 for a in annotations if not getattr(a, 'verified', True))
            if unverified:
                # Already carrying work for the user; do not pile more on.
                continue

            # Ties broken by a stable hash so the choice does not wander between
            # rounds for no reason.
            scored.append((len(annotations),
                           InPlaceTraining.stable_fraction(image_path),
                           image_path))

        scored.sort()
        return [image_path for _count, _tie, image_path in scored[:budget]]

    @property
    def trained_labels(self):
        """Class names of the most recently trained model, in class-index order."""
        return self.round_history[-1]['labels'] if self.round_history else []

    def after_training(self, weights):
        """Run the passes that need the trained model, loading it exactly once.

        Prediction and the disagreement check both want the same weights, and
        loading them twice is a second copy on the card for no reason.
        """
        predict = self.predict_combo.currentText() == "True"
        audit = self.audit_combo.currentText() == "True"
        if not (predict or audit):
            return

        model = self.load_trained_model(weights)
        if model is None:
            return

        try:
            if predict:
                self.run_predictions(model)
            if audit:
                self.run_audit(model)
        finally:
            self.release_model(model)

    def load_trained_model(self, weights):
        """Load the round's weights, or None with the reason already reported."""
        try:
            from ultralytics import YOLO
        except Exception as e:
            QMessageBox.warning(self, "Prediction Failed", f"Could not load Ultralytics: {e}")
            return None
        try:
            return YOLO(weights)
        except Exception as e:
            QMessageBox.warning(self, "Prediction Failed", f"Could not load the trained model: {e}")
            return None

    @staticmethod
    def release_model(model):
        """Give the weights back before the next round needs the card."""
        del model
        gc.collect()
        try:
            from torch.cuda import empty_cache
            empty_cache()
        except Exception:
            pass

    def predict_each(self, model, image_paths, title):
        """Yield (image_path, results) per image, behind one progress bar.

        A failure on one image is printed and skipped: losing a whole pass
        because a single file could not be read would cost the round.
        """
        progress_bar = ProgressBar(self, title=title)
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))
        try:
            for image_path in image_paths:
                try:
                    results = model.predict(
                        image_path,
                        conf=self.main_window.get_uncertainty_thresh(),
                        iou=self.main_window.get_iou_thresh(),
                        imgsz=self.imgsz_spinbox.value(),
                        device=self.main_window.device,
                        verbose=False,
                    )
                    for result in results:
                        result.path = image_path
                    yield image_path, results
                except Exception as e:
                    print(f"Warning: prediction failed for {image_path}: {e}")
                progress_bar.update_progress()
        finally:
            progress_bar.stop_progress()
            progress_bar.close()

    def results_processor(self):
        """A ResultsProcessor mapping the trained classes onto project labels.

        Predictions land unverified on their own: update_machine_confidence
        clears the flag for anything carrying a machine prediction. That is the
        invariant the loop depends on -- without it a later round would train on
        this round's guesses -- so tests/active_learning/test_session.py pins it
        rather than leaving it to be rediscovered.
        """
        class_mapping = {}
        for code in self.trained_labels:
            label = self.label_window.get_label_by_short_code(code)
            if label is not None:
                class_mapping[code] = label.to_dict()
        return ResultsProcessor(self.main_window, class_mapping)

    def run_predictions(self, model):
        """Predict onto the chosen images, leaving everything unverified.

        Blocking on purpose for now. The recommended model is nano and the
        budget is small, so the wait is short and bounded, and a modal pass is
        far simpler than reconciling predictions that land while the user is
        editing the same image.
        """
        image_paths = self.candidate_images(self.budget_spinbox.value())
        if not image_paths:
            self.show_status("Active Learning: no un-reviewed images left to predict on.")
            return

        results_processor = self.results_processor()

        added = 0
        for image_path, results in self.predict_each(model, image_paths,
                                                     "Predicting on Un-reviewed Images"):
            try:
                if self.task == 'segment':
                    annotations = results_processor.build_segmentation_annotations(results)
                else:
                    annotations = results_processor.build_detection_annotations(results)
                if annotations:
                    self.annotation_window.add_annotations(annotations)
                    added += len(annotations)
            except Exception as e:
                print(f"Warning: could not add annotations for {image_path}: {e}")

        self.image_window.filter_images()
        self.annotation_window.load_annotations()
        self.record_outcome(predictions=added, predicted_on=len(image_paths))
        self.show_status(f"Active Learning: {added} predictions added across "
                         f"{len(image_paths)} images, all awaiting review.")

    # ------------------------------------------------------------------
    # Disagreements
    # ------------------------------------------------------------------

    def is_auditable(self, annotation, allowed_types, trained_labels):
        """Whether a disagreement about this annotation would mean anything.

        The label has to be one the model was trained on. If it was not, the
        model could not have predicted it, and 'the model said something else'
        is a statement about the label list rather than about the annotation.
        """
        if not isinstance(annotation, allowed_types):
            return False
        if not getattr(annotation, 'verified', True):
            return False
        if annotation.label is None:
            return False
        return annotation.label.short_label_code in trained_labels

    @staticmethod
    def rotate(items, budget, round_number):
        """Take a window of `budget` items, advancing it each round.

        Auditing the same first N images every round would re-check work already
        checked and never reach the rest. Rotating over a stable order means
        coverage accumulates instead.
        """
        if not items or budget <= 0:
            return []
        start = ((round_number - 1) * budget) % len(items)
        window = (items + items)[start:start + budget]
        return window[:len(items)]

    def audit_images(self, budget, round_number):
        """Reviewed images worth re-checking this round."""
        trained = set(self.trained_labels)
        if not trained:
            return []
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        raster_manager = self.image_window.raster_manager

        candidates = []
        for image_path in raster_manager.image_paths:
            raster = raster_manager.get_raster(image_path)
            if raster is None or getattr(raster, 'raster_type', '') != 'ImageRaster':
                continue
            annotations = self.annotation_window.get_image_annotations(image_path)
            if any(self.is_auditable(a, allowed_types, trained) for a in annotations):
                candidates.append(image_path)

        candidates.sort(key=InPlaceTraining.stable_fraction)
        return self.rotate(candidates, budget, round_number)

    @staticmethod
    def box_iou(first, second):
        """Intersection over union of two (xmin, ymin, xmax, ymax) boxes."""
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        if right <= left or bottom <= top:
            return 0.0
        overlap = (right - left) * (bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - overlap
        return float(overlap / union) if union > 0 else 0.0

    @staticmethod
    def bounds_of(annotation):
        """An annotation's bounding box as (xmin, ymin, xmax, ymax)."""
        top_left = annotation.get_bounding_box_top_left()
        bottom_right = annotation.get_bounding_box_bottom_right()
        return (top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y())

    @staticmethod
    def class_name(result, class_id):
        """Map a class index back to its name, whether names is a dict or list."""
        names = getattr(result, 'names', None)
        if names is None:
            return None
        try:
            return names[class_id]
        except (KeyError, IndexError, TypeError):
            return None

    def find_disagreements(self, image_path, results):
        """Detections that land on a verified annotation but name a different class.

        Deliberately computed here rather than inside ResultsProcessor. The NMS
        there is class-agnostic and runs against every existing annotation, so
        its suppressed set mixes 'the model re-found something you drew' with
        'the model contradicts you'. Only the second is a signal, and only
        against annotations a person actually confirmed.
        """
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        trained = set(self.trained_labels)

        verified = [a for a in self.annotation_window.get_image_annotations(image_path)
                    if self.is_auditable(a, allowed_types, trained)]
        if not verified:
            return []

        boxes = [self.bounds_of(annotation) for annotation in verified]
        iou_thresh = self.main_window.get_iou_thresh()

        found = []
        for result in results:
            if getattr(result, 'boxes', None) is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for index in range(len(xyxy)):
                predicted = self.class_name(result, int(classes[index]))
                if predicted is None:
                    continue

                detection = (float(xyxy[index][0]), float(xyxy[index][1]),
                             float(xyxy[index][2]), float(xyxy[index][3]))

                best_iou = 0.0
                best = None
                for annotation, box in zip(verified, boxes):
                    overlap = self.box_iou(box, detection)
                    if overlap > best_iou:
                        best_iou, best = overlap, annotation

                if best is None or best_iou < iou_thresh:
                    continue
                if best.label.short_label_code == predicted:
                    continue

                found.append({
                    'image_path': image_path,
                    'annotation_id': best.id,
                    'confirmed': best.label.short_label_code,
                    'predicted': predicted,
                    'confidence': float(confidences[index]),
                })

        return found

    def run_audit(self, model):
        """Re-check reviewed images and collect where the model disagrees."""
        round_number = self.round_history[-1]['round'] if self.round_history else 1
        image_paths = self.audit_images(self.budget_spinbox.value(), round_number)
        if not image_paths:
            self.disagreement_label.setText("No reviewed images to check yet.")
            return

        found = []
        for image_path, results in self.predict_each(model, image_paths,
                                                     "Checking Reviewed Images"):
            try:
                found.extend(self.find_disagreements(image_path, results))
            except Exception as e:
                print(f"Warning: disagreement check failed for {image_path}: {e}")

        self.show_disagreements(found, round_number, len(image_paths))
        self.record_outcome(disagreements=len(found))

    def show_disagreements(self, found, round_number, checked):
        """Fill the queue, most confident disagreement first."""
        # Most confident first: that is the ordering by how likely the row is to
        # be a real problem rather than a near-miss box.
        found.sort(key=lambda entry: entry['confidence'], reverse=True)
        self.disagreements = found

        self.disagreement_table.setRowCount(0)
        for entry in found:
            row = self.disagreement_table.rowCount()
            self.disagreement_table.insertRow(row)
            values = (os.path.basename(entry['image_path']),
                      entry['confirmed'],
                      entry['predicted'],
                      f"{entry['confidence']:.2f}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column:
                    item.setTextAlignment(Qt.AlignCenter)
                self.disagreement_table.setItem(row, column, item)

        if found:
            self.disagreement_label.setText(
                f"Round {round_number}: {len(found)} disagreements across {checked} "
                f"reviewed images. Double-click a row to go there.")
        else:
            self.disagreement_label.setText(
                f"Round {round_number}: the model agreed with you on all {checked} "
                f"reviewed images it checked.")

    def on_disagreement_activated(self, row, _column):
        """Open the image behind a row with its annotation selected."""
        if not 0 <= row < len(self.disagreements):
            return
        entry = self.disagreements[row]

        try:
            self.image_window.load_image_by_path(entry['image_path'])
        except Exception as e:
            self.show_status(f"Active Learning: could not open that image: {e}")
            return

        annotation = self.annotation_window.annotations_dict.get(entry['annotation_id'])
        if annotation is None:
            self.show_status("Active Learning: that annotation has since been deleted.")
            return

        try:
            self.annotation_window.unselect_annotations()
            self.annotation_window.select_annotation(annotation)
            self.annotation_window.center_on_annotation(annotation)
        except Exception as e:
            print(f"Warning: could not select the disagreeing annotation: {e}")

    def show_status(self, message):
        """Post a message to the main window status bar."""
        try:
            self.main_window.status_bar.showMessage(message, 8000)
        except Exception:
            pass
