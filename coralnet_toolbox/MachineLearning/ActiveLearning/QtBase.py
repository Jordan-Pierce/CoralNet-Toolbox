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

Three rules matter more than the rest:

  * Training uses **verified annotations only**. An unverified annotation is a
    model prediction, and training on those teaches the model its own guesses.
    The failure is quiet -- metrics look fine while the model converges on its
    own errors -- so it is the default rather than an option.
  * Predictions arrive **unverified**, however confident. That already falls
    out of update_machine_confidence, so this dialog does not have to arrange
    it -- but the loop depends on it, so it is pinned by a test rather than
    assumed.
  * An image a person **reviewed and cleared** trains as background. Deleting a
    false positive is the only way the user can say "there is nothing there",
    and an image with no annotations is otherwise dropped from the dataset
    entirely -- so the correction would be silently discarded and the model
    would keep making it. Review state is tracked per image on the raster.

The dialog is modeless. That is load-bearing rather than a preference: Previous
and Next move the canvas from one waiting annotation to the next, and the review
buttons act on whatever is in front of the user. Behind a modal dialog none of
that can be touched.

It also deliberately does not reimplement what the rest of the application
already does. The Image Window is filtered to Needs Review, the Annotation Viewer
lists what is waiting, and the canvas draws unverified annotations with a black
outline. What was missing was a way to walk that queue without hunting for the
next one, so that is all this offers.

The dialog is split across two tabs because it did not otherwise fit on a
1080p screen -- seven stacked group boxes wanted 1265 px and could not shrink
below 1047. Setup is what you decide before the first round; Session is what
changes as rounds run. Train and the ready line sit outside both, since a Train
button that disappears when you switch tabs would be worse than the height was.

A session is ephemeral. Round history, the frozen class order and each image's
review state live for as long as the application does and are not written into
the project file. That is a decision rather than an omission: the history
describes one sitting's experiment, and a review state that outlived it would
make a mistaken "reviewed" permanent -- an image training as empty in every
future session with nothing on screen to explain why.

Scope: detection and instance segmentation, plain image rasters. See
ACTIVE_LEARNING_PLAN.md.
"""

import warnings

import os
import gc
from html import escape
import shutil
import datetime
import statistics

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
                             QFrame, QProgressBar, QScrollArea, QSpinBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from coralnet_toolbox import theme as app_theme

from coralnet_toolbox.Common.QtThresholdsWidget import ThresholdsWidget

from coralnet_toolbox.Features.FeatureMapCodec import load_feature_vector

from coralnet_toolbox.MachineLearning import InPlaceTraining
from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
from coralnet_toolbox.MachineLearning.TrainModel.QtBase import TrainModelWorker
from coralnet_toolbox.MachineLearning.TrainModel.QtDetect import STANDARD_MODELS as DETECT_MODELS
from coralnet_toolbox.MachineLearning.TrainModel.QtSegment import STANDARD_MODELS as SEGMENT_MODELS

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
    # Ten rather than the Train Model dialog's thirty: rounds here are short by
    # design, and a patience longer than the round is early stopping that can
    # never fire.
    'patience': 10,
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

# Exactly what the Train Model dialog offers, imported rather than restated. The
# short hand-written list this replaced had drifted: it contained 'yolo8n.pt',
# which is not a model Ultralytics can resolve -- the real name is 'yolov8n.pt' --
# so choosing it failed a round with a FileNotFoundError before training began.
MODELS = {
    'detect': DETECT_MODELS,
    'segment': SEGMENT_MODELS,
}

# Small models by default. Rounds are only useful if they are cheap enough to run
# often; a large model turns a five-minute loop into an afternoon.
DEFAULT_MODEL = {
    'detect': 'yolo11n.pt',
    'segment': 'yolo11n-seg.pt',
}

# Ultralytics cache modes, as the Train Model dialog presents them.
CACHE_OPTIONS = (
    ("False", False,
     "cache=False: disables image caching completely. Lowest RAM usage, slowest training."),
    ("True / ram", True,
     "cache=True or cache='ram': caches preprocessed images in system RAM. "
     "Fastest training, highest RAM usage."),
    ("disk", "disk",
     "cache='disk': caches preprocessed images on disk. Minimal RAM usage, "
     "moderate training speed."),
)

OPTIMIZERS = ("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp")

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

# Status colours for the Next Step headline. Local rather than added to the
# theme: the rest of the application has no success/warning vocabulary, and
# claiming one here would be inventing a shared convention out of a single use.
STATE_IDLE = app_theme.TEXT_SECONDARY_COLOR
STATE_RUNNING = app_theme.ACCENT_COLOR
STATE_OK = QColor("#4fbf7b")
STATE_WARN = QColor("#d9a441")
STATE_ERROR = QColor("#e0605e")

# The four numbers a person actually wants off a finished round. Buried in a
# paragraph they were read as prose; as tiles they are read at a glance, which
# is what "should I run another round?" needs.
STAT_TILES = (
    ('map50', "mAP50",
     "Validation mAP50 from the most recent round that produced one, so a round"
     "\nthat failed or was stopped early leaves the last real score standing."),
    ('delta', "Change",
     "Change in mAP50-95 against the previous comparable round, which is what"
     "\ndecides whether a round's model is adopted. mAP50 is not: it saturates,"
     "\nand a project whose objects are easy to find sits at 0.99 from round two"
     "\nonwards while the model is still getting better at placing them."
     "\nBlank when the label set changed: that is a different measurement,"
     "\nnot a worse round."),
    ('awaiting', "Awaiting",
     "Predictions nobody has confirmed yet. These are the input to the next round."),
    ('images', "Images",
     "How many images those predictions are spread across, which is how many you"
     "\nwould have to open to review them one at a time."),
)

STAT_TILE_STYLE = (
    f"QFrame#ALStatTile {{"
    f" background-color: {app_theme.SURFACE_COLOR.name()};"
    f" border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};"
    f" border-radius: 4px; }}"
)


def rgba(color, alpha):
    """`rgba(...)` for a stylesheet, since QSS has no colour-with-alpha literal."""
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def callout_style(color):
    """The guidance panel's look, tinted by the state it is reporting on.

    A left bar and a tint rather than a group box, because it is the one thing
    on the tab that is neither a control nor a number: it is the sentence that
    says what to do next, and it was previously a line of grey text among four
    other lines of grey text.

    The tint is derived from the state colour rather than being a flat elevated
    surface: at a glance the panel reads as "running" or "failed" before a word
    of it is read, which is the whole point of giving it a surface at all.
    """
    return (
        f"QFrame#ALCallout {{"
        f" background-color: {rgba(color, 26)};"
        f" border: 1px solid {rgba(color, 90)};"
        f" border-left: 4px solid {color.name()};"
        f" border-radius: 6px; }}"
        f"QFrame#ALCallout QLabel {{ background: transparent; border: none; }}"
    )


def callout_badge_style(color):
    """The state glyph, in a tinted disc rather than floating on the panel."""
    return (
        f"QLabel#ALCalloutIcon {{"
        f" color: {color.name()};"
        f" background-color: {rgba(color, 38)};"
        f" border: 1px solid {rgba(color, 110)};"
        f" border-radius: {app_theme.scale_int(15)}px; }}"
    )


def state_pill_style(color):
    """The state word, top-right of the callout: the label for its colour.

    Colour alone says a round went badly; it does not say whether badly means
    stopped or failed, and a person who cannot separate the two hues gets
    nothing from it at all.
    """
    return (
        f"QLabel#ALStatePill {{"
        f" color: {color.name()};"
        f" background-color: {rgba(color, 34)};"
        f" border: 1px solid {rgba(color, 120)};"
        f" border-radius: {app_theme.scale_int(8)}px;"
        f" padding: 1px 8px; font-weight: bold; }}"
    )


# The Review group's header band. The queue count used to be a bare line of grey
# text sitting between the tiles and the buttons, which is where a caption goes
# to be ignored: it is the sentence that says how much work is left, so it gets
# the top of the group and a surface of its own.
REVIEW_HEADER_STYLE = (
    f"QFrame#ALReviewHeader {{"
    f" background-color: {app_theme.SURFACE_COLOR.name()};"
    f" border: 1px solid {app_theme.SURFACE_BORDER_COLOR.name()};"
    f" border-radius: 4px; }}"
    f"QFrame#ALReviewHeader QLabel {{ background: transparent; border: none; }}"
)

REVIEW_EYEBROW_STYLE = (
    f"QLabel#ALReviewEyebrow {{ color: {app_theme.TEXT_MUTED_COLOR.name()};"
    f" font-weight: bold; }}"
)


def review_position_style(active):
    """The "3 of 24" badge, greyed out when the queue is empty."""
    color = app_theme.ACCENT_COLOR if active else app_theme.DISABLED_COLOR
    return (
        f"QLabel#ALReviewPosition {{"
        f" color: {color.name()};"
        f" background-color: {rgba(color, 34)};"
        f" border: 1px solid {rgba(color, 120)};"
        f" border-radius: {app_theme.scale_int(8)}px;"
        f" padding: 1px 8px; font-weight: bold; }}"
    )


PRIMARY_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {app_theme.ACCENT_COLOR.name()};"
    f" color: {app_theme.TEXT_BRIGHT_COLOR.name()};"
    f" border: 1px solid {app_theme.ACCENT_COLOR.name()};"
    f" border-radius: 4px; padding: 4px 10px; font-weight: bold; }}"
    f"QPushButton:hover {{ background-color: {app_theme.ACCENT_HOVER_COLOR.name()}; }}"
    f"QPushButton:disabled {{ background-color: {app_theme.SURFACE_COLOR.name()};"
    f" color: {app_theme.DISABLED_COLOR.name()};"
    f" border-color: {app_theme.SURFACE_BORDER_COLOR.name()}; }}"
)

BLANK_STAT = "--"

# How many newly confirmed annotations of every included label it takes before
# the next round starts on its own. Twenty is enough to move a class's weights
# and small enough to reach in one sitting.
AUTO_TRAIN_PER_LABEL = 20

# A round is in exactly one of these, and the headline is the only thing in the
# panel that carries colour -- so the state is readable before anything is read.
STATE_COLOR = {
    'idle': STATE_IDLE,
    'running': STATE_RUNNING,
    'ok': STATE_OK,
    'warn': STATE_WARN,
    'error': STATE_ERROR,
}

STATE_ICON = {
    'idle': "\u25cb",
    'running': "\u23f3",
    'ok': "\u2705",
    'warn': "\u26a0",
    'error': "\u274c",
}

# The pill beside the headline. Colour alone cannot separate "stopped" from
# "failed", and a session that has never trained looks the same as one whose
# round is still going if the only difference between them is a hue.
STATE_WORD = {
    'idle': "IDLE",
    'running': "RUNNING",
    'ok': "DONE",
    'warn': "STOPPED",
    'error': "FAILED",
}

STATE_TOOLTIP = {
    'idle': "No round has run yet in this session.",
    'running': "A round is training. The status bar tracks the epochs.",
    'ok': "The last round finished and produced a model.",
    'warn': "The last round was stopped early. What it trained was still saved.",
    'error': "The last round failed or produced no weights.",
}

# Columns of the Rounds table. The delta is the column the user actually reads:
# an absolute mAP means little on its own, and "is this still improving?" is
# the question that decides whether to run another round.
(HIST_ROUND, HIST_IMAGES, HIST_BACKGROUND,
 HIST_ANNOTATIONS, HIST_MAP, HIST_FITNESS, HIST_DELTA) = range(7)

HISTORY_HEADERS = ["Round", "Train Images", "Background", "Annotations",
                   "mAP50", "mAP50-95", "Change"]

# Per-image Active Learning review state, stored on the raster.
REVIEW_PENDING = 'pending'
REVIEW_REVIEWED = 'reviewed'

# Weights are kept for the most recent rounds only. Every round writes a full
# Ultralytics run directory; ten rounds of a nano model is a few hundred MB of
# checkpoints nobody will open again, and nothing pruned them.
KEEP_ROUND_WEIGHTS = 3

# How much of a round's budget goes to images with nothing on them at all.
# The rest goes to images the user has already worked on, which is where the
# model's mistakes are worth the most: an image with a few annotations on it is
# somewhere the user has decided is interesting, and a model that gets those
# wrong is a model the user can correct cheaply. Spending the whole budget on
# empty images was the old behaviour, and it sent every round to the corner of
# the project nobody had looked at yet.
EXPLORE_SHARE = 0.75

# Acquisition ranks by descriptor diversity only when this share of the images
# it is choosing between actually carries a pooled descriptor.
#
# A partial bake is the dangerous case, not a missing one. With three baked
# images in a project of six hundred, the diverse three would take the front of
# every round's budget forever -- not because they are worth predicting on, but
# because they are the only ones the ranking can see. Below the share the whole
# pool falls back to the counting order, which at least ranks every image by the
# same rule.
DIVERSITY_MIN_SHARE = 0.5

# The smallest set a farthest-point traversal says anything about. One vector
# has no distances in it.
DIVERSITY_MIN_VECTORS = 2

# An object smaller than this after the image is resized to imgsz is not going
# to be detected. Below it the session says so up front rather than letting the
# user find out after five rounds that the loop cannot converge.
MIN_OBJECT_PIXELS = 12



class SessionResetPrompt(QDialog):
    """What starting a new session costs, and what it can clear out.

    A QMessageBox would have done for the question, but not for the two answers
    that come with it: the weights on disk are a separate decision from the
    session, and they default differently. This session's checkpoints may be the
    model currently loaded in a Deploy dialog, so removing them is opt-in;
    weights from a session that has already ended cannot be reached by anything
    in the application, so removing them is the default.
    """

    def __init__(self, rounds, reviewed, mine, mine_count, stale, stale_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setWindowIcon(get_window_icon("coralnet.svg"))

        layout = QVBoxLayout(self)

        discarded = []
        if rounds:
            discarded.append(f"{rounds} round{'s' if rounds != 1 else ''} of history")
        if reviewed:
            discarded.append(f"the review state on {reviewed} image"
                             f"{'s' if reviewed != 1 else ''}")
        summary = ", ".join(discarded) if discarded else "nothing yet -- no rounds have run"

        heading = QLabel(
            f"<b>Start a new session?</b><br><br>"
            f"Discards {summary}, and the best model this session produced.<br><br>"
            f"Your annotations are untouched -- including predictions waiting for "
            f"review and everything you have confirmed. Images you cleared as "
            f"background stop counting as background, because that is a fact about "
            f"this session rather than about the image.")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self.mine_check = QCheckBox(
            f"Also delete this session's checkpoints "
            f"({mine_count} round folder{'s' if mine_count != 1 else ''}, "
            f"{Base.as_megabytes(mine)})")
        self.mine_check.setToolTip(
            "Off by default: the best of these may be the model currently loaded in\n"
            "a Deploy dialog, and it is the only copy.\n"
            "results.csv and the plots are kept either way.")
        self.mine_check.setChecked(False)
        self.mine_check.setEnabled(bool(mine_count))
        layout.addWidget(self.mine_check)

        self.stale_check = QCheckBox(
            f"Delete checkpoints left by earlier sessions "
            f"({stale_count} round folder{'s' if stale_count != 1 else ''}, "
            f"{Base.as_megabytes(stale)})")
        self.stale_check.setToolTip(
            "On by default: a session's round history dies with the session, so\n"
            "nothing in the application can warm start from or deploy these weights\n"
            "any more. They are simply taking up room.\n"
            "results.csv and the plots are kept either way.")
        self.stale_check.setChecked(bool(stale_count))
        self.stale_check.setEnabled(bool(stale_count))
        layout.addWidget(self.stale_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        start_button = QPushButton("Start New Session")
        buttons.addButton(start_button, QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def delete_mine(self):
        """Whether this session's own checkpoints were opted in for deletion."""
        return self.mine_check.isChecked() and self.mine_check.isEnabled()

    def delete_stale(self):
        """Whether the earlier sessions' checkpoints should go."""
        return self.stale_check.isChecked() and self.stale_check.isEnabled()


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

        # Kept above the main window, and given the buttons to get out of the
        # way with. The review pass drives the canvas -- opening an image, moving
        # the view, selecting an annotation -- and any of that activates the main
        # window, which on a single monitor puts this dialog behind it after
        # every press. The batch-inference dialog carries the same hint for the
        # same reason. Minimize is the escape hatch: on top is only tolerable if
        # it can be dismissed without being closed.
        self.setWindowFlags(Qt.Window
                            | Qt.WindowStaysOnTopHint
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowCloseButtonHint)

        self.worker = None
        self._pending = None
        self.round_history = []
        # The best round's weights, and the round that produced them. Kept
        # together because predicting with one round's model while mapping class
        # indices through another round's label order mislabels everything.
        self.last_model_path = None
        self.best_round = None

        # The images the last prediction pass ran on, so it can be run again at
        # different thresholds over exactly the same set.
        self.last_predicted_images = []

        # Every image any round has predicted on this session. Rounds prefer
        # images this does not contain, so the budget moves across the project
        # instead of landing on the same emptiest handful every time.
        self.predicted_ever = set()

        # Why images were passed over by the last acquisition pass, so a round
        # that predicted on nothing can say what it skipped rather than leaving
        # the user to guess. The open image is the one that surprises people.
        self.last_skipped = {}

        # How the last pass ranked its candidates: 'diversity' when the project
        # carries pooled descriptors and k-center greedy decided the order,
        # 'counts' when it fell back to annotation counts. Reported rather than
        # asked for -- there is no knob, because a project with no descriptors
        # has no choice to make.
        self.last_acquisition = 'counts'

        # Per-label verified counts as of the last round, which is what the
        # automatic trigger measures against. Captured when a round starts, so
        # work done while one is training still counts towards the next.
        #
        # None until the session first reads the project. Opening a dialog onto
        # a project that already has hundreds of confirmed annotations would
        # otherwise start training before the user had finished reading the tab:
        # the trigger is about new work, and nothing is new yet.
        self.baseline_counts = None

        # Set while a finished round is deploying and predicting. The automatic
        # trigger must not fire in the middle of that: the worker is already
        # released by then, so nothing else would stop it starting a round on
        # top of a prediction pass that is still adding annotations.
        self._post_round = False
        self.ready_status = False
        self.last_round_outcome = None

        # What refresh_dataset() worked out, kept so start_round() can turn it
        # into records without reading the project a second time. Building
        # records is O(annotations) and the table only needs counts, so the
        # expensive half runs once per round rather than once per checkbox.
        self.plan = None

        # Every label that has ever been included, in the order it was first
        # included. Class indices are assigned from this rather than from the
        # table's row order, which sorts by verified count and therefore
        # reshuffles as the user reviews -- silently pointing round n+1's warm
        # start at round n's weights with the classes permuted.
        self.class_memory = []

        # What the review queue looks like right now, from awaiting_summary().
        # Kept rather than recomputed because four surfaces read it on every
        # repaint, and they have to agree with each other.
        self.awaiting = {'per_label': {}, 'total': 0, 'images': 0,
                         'confidences': [], 'on_image': 0}

        # Recount is debounced; see STATUS_DEBOUNCE_MS.
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(STATUS_DEBOUNCE_MS)
        self._status_timer.timeout.connect(self.update_status_message)
        self._monitoring = False
        self._populating_table = False
        self._refreshing = False
        self._auto_shortfall = {}
        self._verified_counts = {}

        # Which of the five states the session is in, read by the callout and
        # by the status-bar line so the two cannot disagree.
        self.session_state = 'idle'
        # What the round is doing right now, in the words the worker used. Held
        # rather than only posted, because the line is rebuilt on every recount
        # and would otherwise lose the epoch note to the next annotation change.
        self._activity = ""
        # Whether the session is currently reporting to the status bar, what it
        # last put there, and a re-entrancy guard for the repost.
        self._reporting = False
        self._posted = ""
        self._reposting = False
        # Filled in by update_status_message, which is the one pass that counts
        # the project's annotations.
        self._verified_total = 0

        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_setup_tab(), "Setup")
        self.tabs.addTab(self.create_session_tab(), "Session")
        self.tabs.setTabToolTip(0, "What to train on, with what, and what a round does "
                                   "when it finishes.")
        self.tabs.setTabToolTip(1, "What each round produced, and what needs your attention.")
        self.layout.addWidget(self.tabs, 1)

        self.setup_buttons_layout()
        self.load_models()
        # After both halves exist: the panel reports on controls that live in
        # the tabs and on the Re-run button that lives outside them.
        self.update_next_step()
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
        """Everything decided before the first round, in two columns.

        "After Training" belongs here rather than on the Session tab, where it
        used to sit in the top-right corner. It is a set of switches decided
        once, before the first round -- putting it in the most prominent place
        on the tab that reports results left the queue of things actually
        needing attention below it.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left is the model and how it trains; right is the data and what
        # happens when the round finishes. Both columns lead with the thing that
        # can give up space -- the parameter list scrolls, the table shrinks --
        # so neither sets a floor the dialog cannot get under on a laptop.
        left = QVBoxLayout()
        left.addWidget(self.create_info_group())
        left.addWidget(self.create_model_group())
        left.addWidget(self.create_parameters_group(), 1)

        right = QVBoxLayout()
        right.addWidget(self.create_data_group(), 1)
        right.addWidget(self.create_thresholds_group())
        right.addWidget(self.create_round_group())

        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        return widget

    def create_session_tab(self):
        """What each round produced, laid out in the order the work happens.

        The panels used to be grouped by kind, which put "After Training" -- a
        set of switches decided once, before the first round -- in the top-right
        corner, the most prominent place in the tab, above the queue of things
        actually needing attention. It has moved to Setup, next to the other
        decisions made before pressing Train.

        What is left is a single column, read top to bottom: the sentence saying
        what to do next, the panel for doing it, then the question that decides
        whether to go round again. Two columns were only ever a way of fitting
        the disagreement queue onto a 1080p screen; without it the tab is shorter
        than the Setup tab beside it, and a column is a worse way to express a
        sequence than a list is.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.create_callout())
        layout.addWidget(self.create_review_group())
        layout.addWidget(self.create_history_group(), 1)
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

        # Background images have no label, so they cannot be a table row, and
        # they are the part of the dataset a user is most likely to think is
        # not being used.
        self.background_label = QLabel("")
        self.background_label.setWordWrap(True)
        self.background_label.setToolTip(
            "Images you reviewed that ended up with nothing on them. They train as\n"
            "background, which is how deleting a false positive teaches the model.\n"
            "An image nobody has reviewed is never counted here, however long it has\n"
            "been in the project: unannotated does not mean empty.")
        layout.addWidget(self.background_label)

        # Stated up front rather than discovered after five rounds: if objects
        # are tiny relative to the image, full-image inference at imgsz cannot
        # see them and the loop will not converge however much is annotated.
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: rgb(160, 80, 0);")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Re-read the project and recount what would be trained on.")
        self.refresh_button.clicked.connect(lambda: self.refresh_dataset(quiet=False))
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
            "The same list the Train Model dialog offers, including any community\n"
            "models. A nano model is the default on purpose: rounds are only useful\n"
            "if they are cheap enough to run often. Reach for a larger one once the\n"
            "labels have settled.\n"
            "The box is editable, so a path to your own weights can be typed in.")
        layout.addRow("Model:", self.model_combo)

        self.warm_start_combo = bool_combo(
            True,
            "Start each round from the best weights any round has produced so far,\n"
            "which makes later rounds much cheaper.\n"
            "False trains from the model above every round. That is not training from\n"
            "scratch: a .pt file is a COCO-pretrained checkpoint, so a cold round still\n"
            "begins with a backbone that has seen a million photographs.\n"
            "Either way, a round that scores worse than the best one is recorded and\n"
            "set aside -- predictions and Deploy keep using the best model, not the\n"
            "newest -- so a cold round can leave the application behaving exactly as\n"
            "it did before.")
        layout.addRow("Warm Start:", self.warm_start_combo)

        self.free_gpu_combo = bool_combo(
            True,
            "Release cached GPU memory before training.\n"
            "Deployed models are left alone -- only memory nothing is using is freed.\n"
            "If a round still runs out of memory, it offers to retry at a smaller batch.")
        layout.addRow("Free GPU:", self.free_gpu_combo)

        group_box.setLayout(layout)
        return group_box

    def create_parameters_group(self):
        """Everything Ultralytics is told about how to train, in one place.

        The same set the Train Model dialog sends, with the same labels and the
        same defaults, because a round is not a differently-configured kind of
        training that happens to share a worker. Two of them are constrained by
        how in-place training works and say so on the widget rather than being
        quietly missing.

        In a scroll area for the same reason the Train Model dialog uses one:
        sixteen rows of fixed-height widgets is taller than the tab, and a
        parameter that cannot be reached is worse than a shorter list.
        """
        group_box = QGroupBox("Training Parameters")
        group_layout = QVBoxLayout(group_box)

        form_widget = QWidget()
        layout = QFormLayout(form_widget)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        group_layout.addWidget(scroll_area, 1)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(30)
        self.epochs_spinbox.setToolTip(
            "Total number of training iterations over the dataset.\n"
            "Thirty rather than the Train Model dialog's hundred: a round is meant to\n"
            "be short enough to run often, and several short rounds beat one long one\n"
            "while the labels are still moving.")
        layout.addRow("Epochs:", self.epochs_spinbox)

        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(1, 1000)
        self.patience_spinbox.setValue(TRAINING_DEFAULTS['patience'])
        self.patience_spinbox.setToolTip(
            "Number of epochs with no improvement after which training stops.\n"
            "Keep it below Epochs or it can never fire.")
        layout.addRow("Patience:", self.patience_spinbox)

        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(16, 4096)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setToolTip(
            "Input image size for the neural network, in pixels. Must be a multiple\n"
            "of 32. Larger sizes improve accuracy but use more GPU memory.\n"
            "Images are resized to this before the model sees them, so small objects\n"
            "in large images may be lost - the Training Data panel warns when that\n"
            "looks likely.")
        layout.addRow("Image Size:", self.imgsz_spinbox)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 1024)
        self.batch_spinbox.setValue(4)
        self.batch_spinbox.setToolTip(
            "Number of images to process in each batch.\n"
            "Lower this first if a round runs out of memory - it offers to halve it\n"
            "for you when that happens.")
        layout.addRow("Batch Size:", self.batch_spinbox)

        self.single_class_combo = bool_combo(
            TRAINING_DEFAULTS['single_cls'],
            "If True, treat all objects as a single class (presence/absence).\n"
            "If False, distinguish between the labels you included.")
        layout.addRow("Single Class:", self.single_class_combo)

        self.mask_ratio_spinbox = QSpinBox()
        self.mask_ratio_spinbox.setRange(1, 32)
        self.mask_ratio_spinbox.setValue(TRAINING_DEFAULTS['mask_ratio'])
        self.mask_ratio_spinbox.setToolTip(
            "Downsample ratio for segmentation masks during training\n"
            "(1 is native resolution, 4 is a quarter of it). Segmentation only.")
        layout.addRow("Mask Ratio:", self.mask_ratio_spinbox)

        self.weighted_combo = bool_combo(
            TRAINING_DEFAULTS['weighted'],
            "Weighted sampling, to balance an uneven label distribution.\n"
            "On by default here: an Active Learning project is uneven almost by\n"
            "definition early on, when one label has been drawn far more than the rest.")
        layout.addRow("Weighted Sampling:", self.weighted_combo)

        self.freeze_layers_spinbox = QDoubleSpinBox()
        self.freeze_layers_spinbox.setRange(0.0, 1.0)
        self.freeze_layers_spinbox.setSingleStep(0.01)
        self.freeze_layers_spinbox.setValue(TRAINING_DEFAULTS['freeze_layers'])
        self.freeze_layers_spinbox.setToolTip(
            "Fraction of encoder layers to freeze for transfer learning.\n"
            "0.0 trains everything, 0.5 freezes the bottom half, 1.0 freezes all.\n"
            "Worth raising when a round has very few annotations to learn from.")
        layout.addRow("Freeze Layers:", self.freeze_layers_spinbox)

        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setRange(0.0, 1.0)
        self.dropout_spinbox.setSingleStep(0.05)
        self.dropout_spinbox.setValue(TRAINING_DEFAULTS['dropout'])
        self.dropout_spinbox.setToolTip(
            "Dropout rate for regularisation. Higher values reduce overfitting but\n"
            "may underfit on small datasets. Try 0.2-0.5 if rounds overfit.")
        layout.addRow("Dropout:", self.dropout_spinbox)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(OPTIMIZERS)
        self.optimizer_combo.setCurrentText(TRAINING_DEFAULTS['optimizer'])
        self.optimizer_combo.setToolTip(
            "Optimisation algorithm for gradient descent.\n"
            "'auto' picks one for the model; AdamW is a reasonable manual choice.")
        layout.addRow("Optimizer:", self.optimizer_combo)

        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(0, 64)
        self.workers_spinbox.setValue(TRAINING_DEFAULTS['workers'])
        self.workers_spinbox.setEnabled(False)
        self.workers_spinbox.setToolTip(
            "Fixed at 0 for in-place training, and not a preference.\n"
            "The dataset reads its labels from a class attribute in this process. A\n"
            "DataLoader worker is a separate process - spawned, not forked, on\n"
            "Windows - so it would start with an empty registry and the round would\n"
            "train on nothing, without erroring.\n"
            "Export a dataset and use Train Model if you need parallel loading.")
        layout.addRow("Workers:", self.workers_spinbox)

        self.cache_combo = QComboBox()
        for text, value, tooltip in CACHE_OPTIONS:
            self.cache_combo.addItem(text, value)
            self.cache_combo.setItemData(self.cache_combo.count() - 1, tooltip, Qt.ToolTipRole)
        self.cache_combo.setCurrentIndex(0)
        self.cache_combo.setToolTip(
            "Ultralytics cache mode. False disables caching, True/ram caches decoded\n"
            "images to RAM, disk caches them to storage.")
        layout.addRow("Cache:", self.cache_combo)

        self.save_combo = bool_combo(
            TRAINING_DEFAULTS['save'],
            "Save checkpoints during training.\n"
            "The round needs this: best.pt is what warm start, prediction and Deploy\n"
            "all read. Set to False and a round trains and then produces nothing.")
        layout.addRow("Save:", self.save_combo)

        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setRange(-1, 1000)
        self.save_period_spinbox.setValue(TRAINING_DEFAULTS['save_period'])
        self.save_period_spinbox.setToolTip(
            "Save a checkpoint every N epochs. -1 keeps only best and last, which is\n"
            "what the round reads and all it needs.")
        layout.addRow("Save Period:", self.save_period_spinbox)

        self.val_combo = bool_combo(
            TRAINING_DEFAULTS['val'],
            "Validate after each epoch.\n"
            "Turning this off leaves the Rounds table with no mAP to report, so the\n"
            "session can no longer answer whether the model is still improving.")
        layout.addRow("Validation:", self.val_combo)

        self.verbose_combo = bool_combo(
            TRAINING_DEFAULTS['verbose'],
            "Detailed training logs on the console. The per-epoch line in the Session\n"
            "tab comes from the callbacks either way.")
        layout.addRow("Verbose:", self.verbose_combo)

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
        # The ceiling is the number of images in the project, set by
        # update_budget_range whenever the project is recounted: a budget larger
        # than the project is a number that cannot mean anything, and the point
        # of a maximum is to say what the largest useful answer is.
        self.budget_spinbox.setRange(1, 100000)
        self.budget_spinbox.setValue(10)
        self.budget_spinbox.setToolTip(
            "How many un-reviewed images to predict on, at most.\n"
            "Capped at the number of images in the project.\n"
            "A round spends most of the budget on images with nothing on them and the\n"
            "rest on images you have already annotated, so it looks for new objects and\n"
            "checks itself where you are working.\n"
            "Where the project has been baked for features, the images within each of\n"
            "those groups are chosen to be as unalike as possible, so a budget of ten is\n"
            "not spent on ten pictures of the same thing.\n"
            "Images already carrying predictions you have not reviewed are skipped, and\n"
            "so is the image open on the canvas.")
        layout.addRow("Image Budget:", self.budget_spinbox)

        self.auto_train_combo = bool_combo(
            True,
            "Start the next round on its own once every included label has gained\n"
            "the number of newly confirmed annotations below.\n"
            "If a round is already training when that happens, nothing is\n"
            "interrupted - the next round starts when this one finishes, and trains\n"
            "on everything confirmed by then.")
        layout.addRow("Auto Train:", self.auto_train_combo)

        self.auto_train_spinbox = QSpinBox()
        self.auto_train_spinbox.setRange(1, 100000)
        self.auto_train_spinbox.setValue(AUTO_TRAIN_PER_LABEL)
        self.auto_train_spinbox.setToolTip(
            "How many newly confirmed annotations each included label needs before\n"
            "the next round starts by itself. Counted from the last round, so\n"
            "confirming twenty more of every label starts another one.\n"
            "A rare label can hold this up; the Training Data panel says which, and\n"
            "Train Round is always available regardless.")
        layout.addRow("New Per Label:", self.auto_train_spinbox)

        group_box.setLayout(layout)
        return group_box

    def create_callout(self):
        """The one sentence telling the user what to do, given its own surface.

        It was the last line of a five-part panel, in the same weight and colour
        as everything above it, which is a poor place for the only text on the
        tab that asks for an action.
        """
        frame = QFrame()
        frame.setObjectName("ALCallout")
        frame.setStyleSheet(callout_style(STATE_IDLE))

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(app_theme.scale_int(12), app_theme.scale_int(10),
                                  app_theme.scale_int(12), app_theme.scale_int(10))
        layout.setSpacing(app_theme.scale_int(12))

        # The glyph sits in a tinted disc of a fixed size, so the panel keeps the
        # same shape whichever state it is in: an emoji and a geometric circle
        # are different widths, and the text used to shift sideways every time a
        # round changed state.
        self.callout_icon = QLabel("")
        self.callout_icon.setObjectName("ALCalloutIcon")
        icon_font = self.callout_icon.font()
        icon_font.setPointSize(icon_font.pointSize() + 4)
        self.callout_icon.setFont(icon_font)
        self.callout_icon.setAlignment(Qt.AlignCenter)
        self.callout_icon.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        layout.addWidget(self.callout_icon, 0, Qt.AlignTop)

        text_column = QVBoxLayout()
        text_column.setSpacing(app_theme.scale_int(4))

        # The heading and the state pill share a row: the pill names the colour
        # the whole panel is wearing, and it belongs beside what it qualifies.
        heading_row = QHBoxLayout()
        heading_row.setSpacing(app_theme.scale_int(8))

        self.headline_label = QLabel("")
        headline_font = self.headline_label.font()
        headline_font.setBold(True)
        headline_font.setPointSize(headline_font.pointSize() + 2)
        self.headline_label.setFont(headline_font)
        self.headline_label.setWordWrap(True)
        heading_row.addWidget(self.headline_label, 1)

        self.state_pill = QLabel("")
        self.state_pill.setObjectName("ALStatePill")
        pill_font = self.state_pill.font()
        pill_font.setPointSize(max(6, pill_font.pointSize() - 1))
        self.state_pill.setFont(pill_font)
        self.state_pill.setAlignment(Qt.AlignCenter)
        heading_row.addWidget(self.state_pill, 0, Qt.AlignTop)

        text_column.addLayout(heading_row)

        self.next_step_label = QLabel("")
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setTextFormat(Qt.RichText)
        self.next_step_label.setStyleSheet(
            f"color: {app_theme.TEXT_SECONDARY_COLOR.name()};")
        text_column.addWidget(self.next_step_label)

        layout.addLayout(text_column, 1)
        self.callout_frame = frame
        return frame

    def create_review_group(self):
        """The round's numbers, and the four keys of the review pass.

        This absorbed the old Next Step panel, and most of what used to be here
        went with the bulk buttons. The reason is that the rest of the
        application already does the work: the Image Window is filtered to Needs
        Review, the Annotation Viewer lists what is waiting, and the canvas draws
        unverified annotations with a black outline. Duplicating any of that in
        a dialog was building a second, worse version of a window the user
        already has open.

        What was missing was a way to walk the queue one annotation at a time
        without hunting for the next one, and to say the two things a person
        actually says about a prediction: it is right, or it needs a decision
        later.
        """
        group_box = QGroupBox("Review")
        layout = QVBoxLayout()
        layout.setSpacing(app_theme.scale_int(8))

        # The queue count leads the group rather than sitting between the tiles
        # and the buttons. It is the sentence that says how much work is left,
        # and a caption floating in the middle of a group of controls is read as
        # belonging to whichever control it happens to be nearest.
        layout.addWidget(self.create_review_header())

        self.stat_values = {}
        tiles = QHBoxLayout()
        tiles.setSpacing(app_theme.scale_int(6))
        for key, title, tooltip in STAT_TILES:
            frame, value_label = self.stat_tile(title, tooltip)
            self.stat_values[key] = value_label
            tiles.addWidget(frame, 1)
        layout.addLayout(tiles)

        # Training is the longest part of a round and used to report nothing at
        # all: the button read "Training round 1..." for several minutes while
        # the worker was already emitting per-epoch losses that nobody had
        # connected. Silence for that long reads as a hang.
        self.epoch_bar = QProgressBar()
        self.epoch_bar.setTextVisible(False)
        self.epoch_bar.setFixedHeight(app_theme.scale_int(6))
        self.epoch_bar.setVisible(False)
        layout.addWidget(self.epoch_bar)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet(
            f"color: {app_theme.TEXT_SECONDARY_COLOR.name()};")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # Four keys, in the order a hand moves along them. Previous and Next
        # walk the queue; the two in the middle are the only two answers a
        # person gives to a prediction.
        button_layout = QHBoxLayout()

        self.previous_button = QPushButton("Previous")
        self.previous_button.setToolTip(
            "Go to the previous annotation still awaiting review, wherever it is.\n"
            "Opens its image if that is not the one already open.")
        self.previous_button.clicked.connect(lambda: self.step_review(-1))
        button_layout.addWidget(self.previous_button)

        self.mark_review_button = QPushButton("Mark as Review")
        self.mark_review_button.setToolTip(
            "You cannot say what this is yet. Relabels it Review, which takes it out\n"
            "of the queue without training on it - the Review label is excluded from\n"
            "every round - and moves to the next one waiting.")
        self.mark_review_button.clicked.connect(self.review_current_annotation)
        button_layout.addWidget(self.mark_review_button)

        self.verify_button = QPushButton("Mark Verified")
        self.verify_button.setToolTip(
            "The model was right. Confirms the prediction, which is what makes it\n"
            "training data for the next round, and moves to the next one waiting.")
        self.verify_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.verify_button.clicked.connect(self.verify_current_annotation)
        button_layout.addWidget(self.verify_button)

        self.next_button = QPushButton("Next")
        self.next_button.setToolTip(
            "Go to the next annotation still awaiting review, leaving this one alone.\n"
            "Opens its image if that is not the one already open.")
        self.next_button.clicked.connect(lambda: self.step_review(1))
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout)

        group_box.setLayout(layout)
        return group_box

    def begin_status_reporting(self):
        """Take the status bar over for as long as the session is open.

        Posted with no timeout, so it stands until something replaces it, and
        re-posted whenever the bar falls empty. A timed message from a tool
        replaces the session's line and then clears the bar rather than
        restoring what it interrupted -- so without the repost the loop went
        quiet every time the mouse touched anything.
        """
        if self._reporting:
            return
        try:
            self.main_window.status_bar.messageChanged.connect(self.on_status_cleared)
        except (AttributeError, TypeError):
            return
        self._reporting = True
        self.post_status()

    def end_status_reporting(self):
        """Give the status bar back, taking the session's line with it."""
        if not self._reporting:
            return
        self._reporting = False
        try:
            self.main_window.status_bar.messageChanged.disconnect(self.on_status_cleared)
        except (AttributeError, TypeError, RuntimeError):
            pass
        try:
            if self.main_window.status_bar.currentMessage() == self._posted:
                self.main_window.status_bar.clearMessage()
        except Exception:
            pass

    def on_status_cleared(self, message):
        """Put the session's line back when somebody else's expires.

        Only into an empty bar: while another message is up it stands, so this
        fills gaps rather than fighting for the space.
        """
        if message or not self._reporting or self._reposting:
            return
        self._reposting = True
        try:
            self.post_status()
        finally:
            self._reposting = False

    def post_status(self, activity=None):
        """Put everything the session has to say on the status bar, as one line.

        ``activity`` is what is happening right now in the words of whoever
        knows -- an epoch line, a prediction count, a failure. It is held rather
        than only posted, since the line is rebuilt on every recount.
        """
        if activity is not None:
            self._activity = activity.replace("Active Learning: ", "").strip()
        if not self._reporting:
            return
        self._posted = self.status_line()
        try:
            self.main_window.status_bar.showMessage(self._posted)
        except Exception:
            pass

    def status_line(self):
        """The whole session in one line: state, what is happening, what is left.

        Ordered by what a person interrupts themselves to read: which session
        this is, what state it is in, what it is doing, how much is waiting, and
        how much more confirming it takes before the next round.
        """
        # No state glyph here. The callout wears one because it has a panel to
        # colour; a status bar is a line of text, and an emoji in it renders at
        # whatever size and colour the platform feels like.
        task = TASK_LABELS.get(self.task, self.task)
        segments = ["Active Learning: %s" % task]

        headline = self.headline_label.text() if hasattr(self, 'headline_label') else ""
        if headline:
            segments.append(headline)
        if self._activity:
            segments.append(self._activity)

        total = self.awaiting.get('total', 0)
        images = self.awaiting.get('images', 0)
        plural = "image" if images == 1 else "images"
        confirmed = ("%d verified" % self._verified_total) if self._verified_total else ""
        if total:
            waiting = "%d awaiting review on %d %s" % (total, images, plural)
        else:
            waiting = "nothing awaiting review"
        segments.append(", ".join(part for part in (confirmed, waiting) if part))

        # Mid-round the trigger is about the round after this one, which is one
        # thing too many to say while epochs are going past.
        if self.worker is None:
            note = self.next_round_note()
            if note:
                segments.append(note)

        return " · ".join(segments)

    def next_round_note(self):
        """How much more confirmed work it takes before the next round.

        The one number a person in the middle of a review pass actually wants,
        and until now it existed only inside the dialog's guidance sentence --
        which is behind whatever window they are annotating in.
        """
        shortfall = getattr(self, '_auto_shortfall', None)
        auto = (getattr(self, 'auto_train_combo', None) is not None
                and self.auto_train_combo.currentText() == "True")
        if not shortfall:
            return "auto training next round" if auto else "ready to train"

        short = sorted(shortfall.items(), key=lambda item: -item[1])[:2]
        listed = ", ".join("%d %s" % (count, code) for code, count in short)
        if len(shortfall) > len(short):
            rest = len(shortfall) - len(short)
            listed += ", +%d label%s" % (rest, "" if rest == 1 else "s")
        return ("auto train in %s" % listed) if auto else ("next round wants %s" % listed)

    def next_round_detail(self):
        """The same thing at tooltip length, naming every label that is short."""
        shortfall = getattr(self, '_auto_shortfall', None)
        spinbox = getattr(self, 'auto_train_spinbox', None)
        target = spinbox.value() if spinbox is not None else 0
        auto = (getattr(self, 'auto_train_combo', None) is not None
                and self.auto_train_combo.currentText() == "True")
        if not shortfall:
            if auto:
                return ("Every included label has enough new work: the next round "
                        "starts on its own.")
            return ("Every included label has enough new work. Press Train Round "
                    "when you are ready.")

        listed = ", ".join("%s %d" % (code, count) for code, count
                           in sorted(shortfall.items(), key=lambda item: -item[1]))
        lead = ("Auto Train starts the next round once every included label has gained "
                if auto else
                "Counting towards the next round: every included label wants ")
        return "%s%d newly confirmed annotations. Still to go: %s." % (lead, target, listed)

    def create_review_header(self):
        """The queue count, at the top of the Review group and on its own surface.

        Three parts, left to right: what the band is, how much is waiting, and
        where in the queue the open annotation sits. The position used to be a
        parenthetical at the end of the same sentence, which is the least
        readable place to put the one number that changes on every key press.
        """
        frame = QFrame()
        frame.setObjectName("ALReviewHeader")
        frame.setStyleSheet(REVIEW_HEADER_STYLE)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(app_theme.scale_int(10), app_theme.scale_int(6),
                                  app_theme.scale_int(10), app_theme.scale_int(6))
        layout.setSpacing(app_theme.scale_int(10))

        eyebrow = QLabel("REVIEW QUEUE")
        eyebrow.setObjectName("ALReviewEyebrow")
        eyebrow_font = eyebrow.font()
        eyebrow_font.setPointSize(max(6, eyebrow_font.pointSize() - 1))
        eyebrow_font.setBold(True)
        # Tracked out, the way a small-caps label is set. QSS has no
        # letter-spacing, so it has to come off the font.
        eyebrow_font.setLetterSpacing(QFont.PercentageSpacing, 112)
        eyebrow.setFont(eyebrow_font)
        eyebrow.setStyleSheet(REVIEW_EYEBROW_STYLE)
        layout.addWidget(eyebrow, 0)

        self.queue_label = QLabel("")
        self.queue_label.setWordWrap(True)
        queue_font = self.queue_label.font()
        queue_font.setBold(True)
        self.queue_label.setFont(queue_font)
        self.queue_label.setStyleSheet(f"color: {app_theme.TEXT_PRIMARY_COLOR.name()};")
        layout.addWidget(self.queue_label, 1)

        self.position_label = QLabel("")
        self.position_label.setObjectName("ALReviewPosition")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setToolTip(
            "Where the selected annotation sits in the queue Previous and Next walk.")
        self.position_label.setStyleSheet(review_position_style(False))
        layout.addWidget(self.position_label, 0)

        return frame

    def rerun_predictions(self):
        """Predict again over the last round's images at the current thresholds.

        The thresholds live on the Setup tab and are read at prediction time, so
        moving them after a round has run changes nothing that is already on
        screen. This is the button that makes them mean something without paying
        for another round of training.
        """
        if self.worker is not None:
            QMessageBox.information(self, "Round Running",
                                    "Wait for the round to finish first.")
            return
        if not self.last_model_path or not os.path.isfile(self.last_model_path):
            QMessageBox.information(self, "No Model Yet",
                                    "No round has produced a model to predict with.")
            return

        image_paths = list(self.last_predicted_images)
        if not image_paths:
            # include_current: the user pressed this button, so the image they
            # are looking at is the one they most likely meant. The automatic
            # pass after a round leaves it alone; this one should not.
            image_paths = self.candidate_images(self.budget_spinbox.value(),
                                                include_current=True)
        if not image_paths:
            self.show_status("Active Learning: no images to predict on.")
            return

        stale = [annotation for path in image_paths
                 for annotation in self.unverified_annotations(path)]
        reply = QMessageBox.question(
            self, "Re-run Predictions",
            f"Predict again on {len(image_paths)} images at an uncertainty threshold "
            f"of {self.main_window.get_uncertainty_thresh():.2f}?\n\n"
            f"{len(stale)} unreviewed predictions on those images will be discarded "
            f"first. Anything you have confirmed is left alone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return

        if stale:
            self.annotation_window.delete_annotations(stale)

        model = self.load_trained_model(self.last_model_path)
        if model is None:
            return
        try:
            self.run_predictions(model, image_paths=image_paths)
        finally:
            self.release_model(model)

        self.refresh_dataset()
        self.update_next_step()

    @staticmethod
    def stat_tile(title, tooltip):
        """One number with a caption. Returns (frame, value label)."""
        frame = QFrame()
        frame.setObjectName("ALStatTile")
        frame.setStyleSheet(STAT_TILE_STYLE)
        frame.setToolTip(tooltip)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(app_theme.scale_int(6), app_theme.scale_int(4),
                                  app_theme.scale_int(6), app_theme.scale_int(4))
        layout.setSpacing(0)

        value_label = QLabel(BLANK_STAT)
        value_font = value_label.font()
        value_font.setBold(True)
        value_font.setPointSize(value_font.pointSize() + 2)
        value_label.setFont(value_font)
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)

        caption = QLabel(title)
        caption.setAlignment(Qt.AlignCenter)
        caption_font = caption.font()
        caption_font.setPointSize(max(6, caption_font.pointSize() - 1))
        caption.setFont(caption_font)
        caption.setStyleSheet(f"color: {app_theme.TEXT_MUTED_COLOR.name()};")
        layout.addWidget(caption)

        return frame, value_label

    def create_history_group(self):
        """The per-round record."""
        group_box = QGroupBox("Rounds")
        layout = QVBoxLayout()

        self.history_table = QTableWidget(0, len(HISTORY_HEADERS))
        self.history_table.setHorizontalHeaderLabels(HISTORY_HEADERS)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setToolTip(
            "Whether the model is still improving is the question that decides when to stop.\n"
            "A round that adds annotations but not accuracy is telling you something.\n"
            "Change is measured against the previous round, and is the column worth\n"
            "reading: an absolute mAP says little without one to compare it to.\n"
            "Rounds trained on different label sets are not comparable, and say so.")
        header = self.history_table.horizontalHeader()
        for index in range(len(HISTORY_HEADERS)):
            header.setSectionResizeMode(index, QHeaderView.Stretch)
        layout.addWidget(self.history_table)

        group_box.setLayout(layout)
        return group_box

    def setup_buttons_layout(self):
        """The action row, outside the tabs so it never goes out of reach."""
        button_layout = QHBoxLayout()

        # On the left, away from Train: it undoes a session rather than
        # advancing one, and a destructive action next to the one the user
        # presses every few minutes is a misclick waiting to happen.
        self.new_session_button = QPushButton("New Session")
        self.new_session_button.setToolTip(
            "Forget this session's rounds and start over.\n"
            "The round history, the best model and the review state on every image\n"
            "are discarded. Your annotations -- including predictions you have\n"
            "already confirmed -- are untouched.\n"
            "Also where round folders left on disk can be cleared out.")
        self.new_session_button.clicked.connect(self.new_session)
        button_layout.addWidget(self.new_session_button)

        button_layout.addSpacing(16)

        self.ready_label = QLabel("❌ Not Ready")
        self.ready_label.setToolTip("Whether a round can be started with the current selection.")
        button_layout.addWidget(self.ready_label)

        button_layout.addStretch()

        # Both actions in the bottom-right corner, where a dialog's actions live.
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, self)

        # Re-run Predictions is a whole-session action, not a review control: it
        # acts on the last round's images rather than on the annotation in front
        # of you, and it belongs with Train and Stop rather than in the middle of
        # the keys the review pass walks with.
        self.rerun_button = QPushButton("Re-run Predictions")
        self.rerun_button.setToolTip(
            "Predict again over the same images with the current thresholds.\n"
            "The uncertainty, IoU and area thresholds decide what a round proposes,\n"
            "and changing them afterwards otherwise does nothing until the next\n"
            "round trains.\n"
            "Unreviewed predictions on those images are cleared first, so raising\n"
            "the threshold removes what no longer qualifies rather than leaving it\n"
            "behind. Anything you have confirmed is untouched.")
        self.rerun_button.clicked.connect(self.rerun_predictions)
        self.rerun_button.setEnabled(False)
        self.buttons.addButton(self.rerun_button, QDialogButtonBox.ActionRole)

        # A round is minutes long. Abandoning one used to mean killing the
        # application, which also loses the round history.
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip(
            "End the running round after the current epoch. What has been trained so\n"
            "far is still saved, so a stopped round is a short round rather than a\n"
            "lost one.")
        self.stop_button.clicked.connect(self.stop_round)
        self.stop_button.setEnabled(False)
        self.buttons.addButton(self.stop_button, QDialogButtonBox.ActionRole)

        self.train_button = QPushButton("Train Round")
        self.train_button.setToolTip("Train on the verified annotations, then predict if enabled.")
        self.train_button.clicked.connect(self.start_round)
        self.train_button.setEnabled(False)
        self.buttons.addButton(self.train_button, QDialogButtonBox.ActionRole)
        self.buttons.rejected.connect(self.reject)
        button_layout.addWidget(self.buttons)

        self.layout.addLayout(button_layout)

    def load_models(self):
        """Fill the model combo exactly as the Train Model dialog fills its own."""
        self.model_combo.clear()
        standard = MODELS.get(self.task, [])
        self.model_combo.addItems(standard)

        community = get_available_configs(task=self.task)
        if community:
            self.model_combo.insertSeparator(len(standard))
            self.model_combo.addItems(list(community.keys()))

        default = DEFAULT_MODEL.get(self.task)
        if default in standard:
            self.model_combo.setCurrentIndex(standard.index(default))

    # ------------------------------------------------------------------
    # Reading the project
    # ------------------------------------------------------------------

    def project_root(self):
        """The directory a session writes its runs and scaffolding under.

        Anchored to the open project rather than the process working directory.
        `abspath` on a relative path only fixed the Ultralytics-nesting bug --
        it still resolves against wherever the application happened to be
        launched from, so the same project would scatter its rounds across the
        disk depending on how it was started, and a restored round history would
        point at weights that are not there.
        """
        path = getattr(self.main_window, 'current_project_path', '') or ''
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if os.path.isdir(directory):
                return directory
        return os.path.abspath(os.getcwd())

    def runs_root(self):
        """Where this session's Ultralytics run directories go."""
        return os.path.join(self.project_root(), 'Data', 'ActiveLearning')

    def cache_root(self):
        """Where the generated yaml and its empty split directories go."""
        return os.path.join(self.project_root(),
                            InPlaceTraining.CACHE_BASE,
                            InPlaceTraining.CACHE_SUBDIR)

    def showEvent(self, event):
        """Read the project and start reporting progress when opened."""
        super().showEvent(event)
        self.thresholds_widget.initialize_thresholds()
        self.begin_status_reporting()
        self.start_monitoring()
        self.refresh_dataset(quiet=False)

    def closeEvent(self, event):
        """Stop reporting progress when the dialog goes away.

        A running round is worth one question. The worker outlives the dialog
        either way -- it is a QThread with the patched dataset class installed --
        so closing does not stop it, and somebody who closes by reflex should
        know the round is still going rather than assume they cancelled it.
        """
        if self.worker is not None:
            reply = QMessageBox.question(
                self, "Round Still Running",
                "A round is still training. Closing this window does not stop it, and "
                "the results will not be recorded.\n\n"
                "Close anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        self.stop_monitoring()
        # The session keeps reporting from behind a closed dialog while a round
        # is still training. Closing does not stop the worker -- the question
        # above says so -- and the status bar is then its only surface.
        if self.worker is None:
            self.end_status_reporting()
        super().closeEvent(event)

    def reject(self):
        self.stop_monitoring()
        if self.worker is None:
            self.end_status_reporting()
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
        # The reviewed toggle is about whichever image is open, so it has to
        # follow the canvas rather than the annotations.
        try:
            self.image_window.imageSelected.connect(self.on_image_selected)
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
        try:
            self.image_window.imageSelected.disconnect(self.on_image_selected)
        except (TypeError, RuntimeError, AttributeError):
            pass
        self._monitoring = False

    def on_image_selected(self, *_args):
        """Follow the canvas: the queue position is about the open image."""
        self.update_review_controls()

    def schedule_status_update(self, *_args):
        """Coalesce a burst of annotation changes into one recount."""
        self._status_timer.start()

    def update_status_message(self):
        """Say how much review is left before another round is worth running.

        Phrased around the decision rather than as a bare countdown: what a
        person wants to know is whether there is enough new material to justify
        training again.

        The table is recounted here too. The dialog now sits open beside the
        canvas rather than in front of it, so its counts are being read while
        the project changes underneath them; a Refresh button that has to be
        remembered would leave them wrong most of the time. The pass is counts
        only -- records are built once per round -- and it is debounced, so a
        bulk relabel costs one recount rather than one per annotation.
        """
        if self.worker is None:
            self.refresh_dataset()

        verified = 0
        for annotation in self.annotation_counts():
            if getattr(annotation, 'verified', True):
                verified += 1
        self._verified_total = verified

        # One line, rebuilt here and posted from update_next_step. Two separate
        # messages about the same session would take turns overwriting each
        # other, and whichever landed second would be the only one ever read.
        self.update_next_step()

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

    def review_state(self, raster):
        """This task's Active Learning review state for one raster."""
        states = getattr(raster, 'active_learning', None)
        return states.get(self.task) if isinstance(states, dict) else None

    def set_review_state(self, raster, state):
        """Record this task's review state on a raster, creating the dict."""
        if raster is None:
            return
        if not isinstance(getattr(raster, 'active_learning', None), dict):
            raster.active_learning = {}
        raster.active_learning[self.task] = state

    def image_rasters(self):
        """Yield (image_path, raster) for the plain image rasters in the project.

        Video frames are virtual paths the trainer cannot open, and an
        orthomosaic is one enormous sample that means nothing without tiling.
        """
        raster_manager = self.image_window.raster_manager
        for image_path in raster_manager.image_paths:
            raster = raster_manager.get_raster(image_path)
            if raster is None or getattr(raster, 'raster_type', '') != 'ImageRaster':
                continue
            yield image_path, raster

    def note_predicted(self, image_path, added):
        """Record that a round put `added` predictions on this image.

        The one guard against an unannotated image being trained as empty, so it
        is a named method rather than a condition inside the prediction loop.

        Only an image that was actually given something becomes pending, and
        only a pending image can ever be promoted to reviewed. An image the model
        found nothing on has nothing on it for anyone to accept or reject, so no
        action the user could take on it would amount to saying it is empty --
        and it must not be inferred from their silence. Marking every image in
        the budget instead, which is what this replaced, turned a project's
        unannotated majority into background images after a single round.
        """
        if added <= 0:
            return
        raster = self.image_window.raster_manager.get_raster(image_path)
        if raster is not None and self.review_state(raster) is None:
            self.set_review_state(raster, REVIEW_PENDING)

    def promote_pending(self):
        """Move images the user has finished with from pending to reviewed.

        Pending means a round put predictions on this image and they are waiting
        on somebody. Once nothing unverified is left, that somebody has been
        through it -- they confirmed some, deleted others, or both -- so it is
        reviewed, and if nothing survived it is a confirmed negative. Inferring
        that beats asking: the user already said it by clearing the image, and a
        dialog asking them to say it again would be dismissed.

        What makes the inference safe is that only an image the model actually
        put something on is ever pending. An earlier version marked every image
        in the budget, so an image the model found nothing on -- which is most
        of them early on, and which nobody has looked at -- went pending,
        immediately had nothing unverified, and was promoted to a background
        image on the next recount. The model was then taught that unannotated
        images are empty, which for most projects is the opposite of true: they
        are unannotated, not empty.
        """
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        for image_path, raster in self.image_rasters():
            if self.review_state(raster) != REVIEW_PENDING:
                continue
            annotations = self.annotation_window.get_image_annotations(image_path)
            unverified = any(isinstance(a, allowed_types) and not getattr(a, 'verified', True)
                             for a in annotations)
            if not unverified:
                self.set_review_state(raster, REVIEW_REVIEWED)

    def negative_images(self, grouped):
        """Reviewed images that ended up with nothing on them.

        These are the images that make deleting a false positive mean anything.
        Without them an image the user cleared is simply absent from the
        dataset, which says nothing at all, and the model goes on predicting the
        same thing there every round.
        """
        negatives = []
        for image_path, raster in self.image_rasters():
            if image_path in grouped:
                continue
            if self.review_state(raster) != REVIEW_REVIEWED:
                continue
            # Marking an image reviewed by hand while predictions are still
            # sitting on it says "I have been here", not "there is nothing
            # here". Training it as background would contradict annotations the
            # user has not actually rejected.
            if self.unverified_annotations(image_path):
                continue
            negatives.append(image_path)
        return negatives

    def awaiting_summary(self):
        """Everything about the review queue, from one walk of the annotations.

        Four surfaces read this queue -- the Awaiting column, the two stat tiles,
        the status line, and every count in the Review group -- and each used to
        walk the project for itself. One pass, and the answers travel together
        so they cannot disagree with each other on screen.

        `confidences` is what lets the bulk buttons say how many annotations they
        are about to touch before they are pressed, rather than in the
        confirmation dialog afterwards.
        """
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        current = getattr(self.annotation_window, 'current_image_path', None)

        per_label = {}
        images = set()
        confidences = []
        on_image = []

        for annotation in self.annotation_window.annotations_dict.values():
            if not isinstance(annotation, allowed_types):
                continue
            if getattr(annotation, 'verified', True):
                continue
            if annotation.label is None:
                continue

            code = annotation.label.short_label_code
            per_label[code] = per_label.get(code, 0) + 1
            images.add(annotation.image_path)
            confidences.append(self.top_confidence(annotation) or 0.0)
            if current is not None and annotation.image_path == current:
                on_image.append(annotation)

        return {
            'per_label': per_label,
            'total': sum(per_label.values()),
            'images': len(images),
            'confidences': confidences,
            'on_image': len(on_image),
        }

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

    def auto_train_shortfall(self, verified):
        """Labels still short of the automatic trigger, as {code: how many more}.

        Measured from the counts captured when the last round started rather
        than from zero, so the question is always "how much new work since the
        model last saw the project" -- which is the only version of it that
        means anything after round one.
        """
        target = self.auto_train_spinbox.value()
        baseline = self.baseline_counts or {}
        shortfall = {}
        for code in self.selected_labels():
            gained = verified.get(code, 0) - baseline.get(code, 0)
            if gained < target:
                shortfall[code] = target - gained
        return shortfall

    def maybe_auto_train(self):
        """Start a round if every included label has had enough new work.

        Deferred to the event loop rather than run inline: it is reached from
        refresh_dataset, and start_round re-enters that. A round starting inside
        the recount that decided to start it would read a half-built plan.
        """
        if self.auto_train_combo.currentText() != "True":
            return
        if self.worker is not None or self._post_round:
            return
        if not self.ready_status or self.plan is None:
            return
        if self._auto_shortfall:
            return
        QTimer.singleShot(0, self.start_round)

    def refresh_dataset(self, quiet=True):
        """Recount what a round would train on, without building it.

        Counting and building used to be the same pass, so every checkbox click
        and every reopen paid to convert every annotation in the project into
        normalized geometry -- work only a round actually needs. The table wants
        counts; records are built once, in start_round.

        Args:
            quiet (bool): Skip the wait cursor. Set for the automatic refresh
                that follows annotation edits, where a cursor flicking on every
                keystroke is worse than no feedback at all.
        """
        if self._refreshing:
            return
        self._refreshing = True
        if not quiet:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.promote_pending()

            grouped = self.project_annotations()
            self.awaiting = self.awaiting_summary()
            awaiting = self.awaiting['per_label']
            negatives = self.negative_images(grouped)

            # Split first, so the table can show where each label actually lands
            # rather than only how many of it there are. Negatives are split the
            # same way: a background image is a training sample like any other.
            InPlaceTraining.set_split_ratios(TRAIN_RATIO, VAL_RATIO)
            all_paths = sorted(set(grouped) | set(negatives))
            groups = InPlaceTraining.group_images_by_split(
                all_paths, TRAIN_RATIO, VAL_RATIO,
                overrides=self.split_overrides(all_paths))

            verified, images, per_split = self.tally(grouped, groups)
            self._verified_counts = dict(verified)
            if self.baseline_counts is None:
                self.baseline_counts = dict(verified)
            self.populate_label_table(verified, images, per_split, awaiting)
            # After the table, not before it. The shortfall is per included
            # label and selected_labels() reads the table's rows, so computing
            # it first asks an empty table which labels are included, gets none,
            # concludes that no label is short of anything, and starts a round
            # the moment the dialog opens.
            self._auto_shortfall = self.auto_train_shortfall(verified)
            self.update_background_label(groups, negatives)
            self.update_budget_range()
            self.update_size_warning(grouped)
            self.update_review_controls()
            self.update_next_step()

            selected = self.selected_labels()
            if not selected:
                self.plan = None
                self.set_not_ready(self.nothing_yet_advice(self.awaiting['total']))
                return

            self.plan = {
                'grouped': grouped,
                'groups': groups,
                'negatives': set(negatives),
                'classes': self.class_order(selected),
            }

            ready, reason = self.readiness(groups, grouped, negatives)
            self.ready_status = ready
            self.ready_label.setText("✅ Ready" if ready else f"❌ Not Ready - {reason}")
            self.train_button.setEnabled(ready and self.worker is None)
            self.new_session_button.setEnabled(self.worker is None)
            self.maybe_auto_train()

        except Exception as e:
            self.set_not_ready(f"Could not read the project: {e}")
            print(f"Error reading project for Active Learning: {e}")
        finally:
            self._refreshing = False
            if not quiet:
                QApplication.restoreOverrideCursor()

    def class_order(self, selected):
        """Class indices for this round, in an order that survives review.

        The table sorts rows by verified count, and review is exactly what
        changes those counts -- so reading indices off the row order means class
        0 can be a different label in round 2 than it was in round 1. Nothing
        would report an error: the warm start would load the previous head onto
        permuted classes and the model would quietly relearn them.

        Indices therefore come from the order labels were first included. A new
        label lands at the end, leaving every existing index where it was.
        """
        chosen = set(selected)
        ordered = [code for code in self.class_memory if code in chosen]
        ordered += [code for code in selected if code not in self.class_memory]
        return ordered

    def remember_classes(self, ordered):
        """Fold this round's classes into the remembered order.

        Codes are never dropped from the memory, only filtered out of a round
        that excludes them: a label unticked for one round and re-ticked for the
        next gets its original index back rather than being appended at the end.
        """
        for code in ordered:
            if code not in self.class_memory:
                self.class_memory.append(code)

    def build_dataset(self):
        """Turn the current plan into an in-place dataset. Returns it, or None.

        The expensive half of what refresh_dataset used to do, kept apart so it
        runs once per round instead of once per checkbox.
        """
        if not self.plan:
            return None

        grouped = self.plan['grouped']
        groups = self.plan['groups']
        negatives = self.plan['negatives']
        classes = self.plan['classes']

        self.remember_classes(classes)
        label_to_index = {code: index for index, code in enumerate(classes)}
        raster_manager = self.image_window.raster_manager

        def dimensions_for(image_path):
            raster = raster_manager.get_raster(image_path)
            return raster.height, raster.width

        records_by_split = {
            split: InPlaceTraining.build_records(
                image_paths, grouped, label_to_index, self.task, dimensions_for,
                negatives=negatives)
            for split, image_paths in groups.items()
        }

        return InPlaceTraining.InPlaceDataset(
            self.task, records_by_split, classes, cache_root=self.cache_root())

    def update_background_label(self, groups, negatives):
        """Say how many images train as background, and where they landed."""
        if not negatives:
            self.background_label.setText(
                "No background images. An image only becomes one when you review it and "
                "leave nothing on it - an image that is simply unannotated is left out "
                "of training, not trained as empty.")
            return

        negative_set = set(negatives)
        train = sum(1 for path in groups.get('train', []) if path in negative_set)
        val = sum(1 for path in groups.get('val', []) if path in negative_set)
        plural = "image" if len(negatives) == 1 else "images"
        self.background_label.setText(
            f"{len(negatives)} background {plural} ({train} train / {val} val): you "
            f"reviewed these and left nothing on them, so they train as empty.")

    def update_budget_range(self):
        """Cap the Image Budget at the number of images the project holds.

        A budget larger than the project is a number that cannot mean anything,
        and the ceiling is what the spinbox is for. Left alone when the project
        has no images yet, so opening the dialog early does not pin the box to 1.
        """
        count = sum(1 for _path, _raster in self.image_rasters())
        if count > 0:
            self.budget_spinbox.setMaximum(count)

    def update_size_warning(self, grouped):
        """Warn when objects are too small to survive the resize to imgsz.

        Full-image inference at imgsz is the shape of v1, so a 4000 px image of
        40 px objects is downscaled roughly six times before the model sees
        anything. For those projects the loop does not converge however much is
        annotated, and there is no point letting somebody find that out over
        five rounds.
        """
        sizes = []
        longest = []
        raster_manager = self.image_window.raster_manager
        for image_path, annotations in grouped.items():
            raster = raster_manager.get_raster(image_path)
            if raster is None or not raster.width or not raster.height:
                continue
            edge = max(raster.width, raster.height)
            for annotation in annotations:
                try:
                    top_left = annotation.get_bounding_box_top_left()
                    bottom_right = annotation.get_bounding_box_bottom_right()
                except Exception:
                    continue
                extent = max(bottom_right.x() - top_left.x(), bottom_right.y() - top_left.y())
                if extent > 0:
                    sizes.append(extent)
                    longest.append(edge)

        if not sizes:
            self.warning_label.setVisible(False)
            return

        median_object = statistics.median(sizes)
        median_edge = statistics.median(longest)
        scale = self.imgsz_spinbox.value() / median_edge if median_edge else 1.0
        after = median_object * scale

        if after >= MIN_OBJECT_PIXELS:
            self.warning_label.setVisible(False)
            return

        self.warning_label.setText(
            f"Objects are small for these images: the typical one is {median_object:.0f} px "
            f"in a {median_edge:.0f} px image, so it reaches the model about "
            f"{after:.0f} px across. Detection is unreliable below roughly "
            f"{MIN_OBJECT_PIXELS} px. Raise Image Size, or wait for Work Area tiling.")
        self.warning_label.setVisible(True)

    @staticmethod
    def nothing_yet_advice(awaiting_total):
        """What to do when there is nothing to train on yet.

        This is the state the dialog opens in for the person the feature exists
        for -- somebody with no annotations and no model -- so it should say
        where to start rather than only refusing.
        """
        if awaiting_total:
            return ("nothing confirmed yet - review some of the waiting predictions "
                    "(Review Predictions) and they become training data")
        return ("nothing confirmed yet - draw a handful of examples of each label "
                "on a few images, then train a first round")

    def split_overrides(self, image_paths):
        """Per-image split assignments a user pinned on the raster."""
        raster_manager = self.image_window.raster_manager
        overrides = {}
        for image_path in image_paths:
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
        self.plan = None
        self.ready_status = False
        self.ready_label.setText(f"❌ Not Ready - {message}")
        self.train_button.setEnabled(False)

    def readiness(self, groups, grouped, negatives):
        """Return (ready, reason) for what a round would train on.

        Only hard blockers. A label missing from a split is shown in red but
        does not block: early rounds legitimately have a rare class absent from
        validation, and refusing to train then would disable the feature in
        exactly the situation it exists for.

        Reasons name a remedy, because the ones that can occur here are not
        fixed by trying again. Splits are derived from the image paths, so an
        empty split stays empty however many times Refresh is pressed -- and on
        a small project it is not a rare accident: roughly one in ten ten-image
        projects has no validation split.
        """
        total = sum(len(paths) for paths in groups.values())

        if not groups.get('train'):
            return False, self.split_advice("no training images", total)
        if VAL_RATIO > 0 and not groups.get('val'):
            return False, self.split_advice("no validation images", total)
        if not grouped:
            if negatives:
                return False, "only background images - confirm some annotations to train on"
            return False, "no annotations on the included labels"
        return True, ""

    @staticmethod
    def split_advice(reason, image_count):
        """Attach a remedy to an empty-split reason.

        Splits are stable per image path on purpose -- it is what stops an image
        migrating from train into validation between rounds and quietly
        inflating every metric after it. The cost is that a small project can
        land badly and stay there, so the way out has to be stated rather than
        guessed at, and it has to be a lever the application actually offers.
        """
        if image_count == 0:
            return "no images carry verified annotations for this task yet"
        if image_count < 20:
            return (f"{reason} ({image_count} images split by path, so Refresh will not "
                    f"change it - annotate more images, or pin one in the Image Window "
                    f"under Training Split)")
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

        self.refresh_dataset(quiet=False)
        if not self.ready_status or self.plan is None:
            QMessageBox.warning(self, "Not Ready",
                                f"Cannot train: {self.ready_label.text()}")
            return

        dataset = self.build_dataset()
        if dataset is None:
            QMessageBox.warning(self, "Nothing to Train On",
                                "There are no verified annotations for this task yet.")
            return

        try:
            data_path = dataset.prepare()
        except Exception as e:
            QMessageBox.critical(self, "Failed to Prepare Dataset", f"{e}")
            return

        # Captured before training rather than after it, so annotations confirmed
        # while this round runs count towards starting the next one.
        self.baseline_counts = dict(self._verified_counts)

        round_number = len(self.round_history) + 1
        run_name = f"round_{round_number:02d}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

        model, warm_note = self.warm_start_source(list(dataset.names))

        params = self.training_params(data_path, model, run_name)
        params['in_place_dataset'] = dataset

        self._pending = {
            'round': round_number,
            'dataset': dataset,
            'run_dir': os.path.join(params['project'], run_name),
            # The model's class names are exactly these, in this order, so the
            # prediction pass needs them to map detections back onto labels.
            'labels': list(dataset.names),
            'stopped': False,
        }

        if self.free_gpu_combo.currentText() == "True":
            self.free_gpu_memory()

        self.train_button.setEnabled(False)
        self.train_button.setText(f"Training round {round_number}...")
        self.stop_button.setEnabled(True)

        self.epoch_bar.setRange(0, self.epochs_spinbox.value())
        self.epoch_bar.setValue(0)
        self.epoch_bar.setVisible(True)
        source_note = self.model_source_note(model)
        self.progress_label.setText(
            f"Round {round_number} starting on {dataset.image_count('train')} training "
            f"images. {source_note}" + (" " + warm_note if warm_note else ""))
        self.progress_label.setVisible(True)
        self.set_headline('running', f"Round {round_number} training")
        # The rounds fill in on the other tab, so send the user there to watch.
        self.tabs.setCurrentIndex(1)
        self.show_status(f"Active Learning: training round {round_number}...")
        print(f"Active Learning: round {round_number}. {source_note}")
        if warm_note:
            print(f"Active Learning: {warm_note}")

        self.worker = TrainModelWorker(params, self.main_window.device)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.training_status.connect(self.on_training_status)
        self.worker.epoch_completed.connect(self.on_epoch_completed)
        self.worker.start()

    def warm_start_source(self, classes):
        """Return (model to train from, note about what warm starting did).

        Warm starting is only meaningful if this round's class indices agree
        with the ones the previous round's head was trained with. Two cases have
        to be told apart, and neither used to be:

          * The label set grew. Indices still line up, so the backbone is worth
            keeping -- but Ultralytics rebuilds the head for the new class count
            and loads only the weights that match, so the detection head is
            reset. Useful, and worth saying out loud rather than leaving the
            user to wonder why round 2 started worse than round 1 ended.
          * The label set changed shape. The old head would be loaded onto
            different classes, so warm starting is skipped for this round.
        """
        base = self.model_combo.currentText()
        if self.warm_start_combo.currentText() != "True":
            return base, ""
        if not self.last_model_path or not os.path.isfile(self.last_model_path):
            return base, ""

        previous = self.best_round.get('labels', []) if self.best_round else []
        if previous and classes[:len(previous)] != previous:
            return base, ("Warm start was skipped: the label set changed, so the previous "
                          "round's weights describe different classes.")

        note = ""
        if previous and len(classes) != len(previous):
            note = ("Warm start kept the backbone; the detection head was reset because "
                    "the label set grew.")
        return self.last_model_path, note

    def model_source_note(self, model):
        """One phrase naming what this round is about to train from.

        A warm-started round and a cold one looked identical on screen, and
        which of the two it was is the first thing worth knowing when reading
        the metric it produces. Naming the file also makes it plain that a cold
        round starts from a COCO-pretrained checkpoint rather than from nothing.
        """
        if self.last_model_path and model == self.last_model_path:
            previous = self.best_round.get('round') if self.best_round else None
            if previous is None:
                return "Warm starting from the best round's weights."
            return f"Warm starting from round {previous}'s weights."

        return f"Training from {os.path.basename(str(model))}."

    # ------------------------------------------------------------------
    # Starting over
    # ------------------------------------------------------------------

    @staticmethod
    def directory_size(directory):
        """Bytes held under a directory; 0 if it is missing or unreadable."""
        total = 0
        for root, _dirs, files in os.walk(directory):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total

    def session_run_dirs(self):
        """The run directories this session has written, absolute."""
        dirs = {entry['run_dir'] for entry in self.round_history if entry.get('run_dir')}
        if self._pending and self._pending.get('run_dir'):
            dirs.add(os.path.abspath(self._pending['run_dir']))
        return dirs

    def weights_on_disk(self, run_dirs):
        """[(weights directory, bytes)] for the run directories that still hold any."""
        found = []
        for run_dir in sorted(run_dirs):
            weights = os.path.join(run_dir, 'weights')
            if not os.path.isdir(weights):
                continue
            size = self.directory_size(weights)
            if size:
                found.append((weights, size))
        return found

    def stale_round_weights(self):
        """[(weights directory, bytes)] for rounds no session can reach any more.

        A session is ephemeral by design, and its round history goes with it --
        so weights from a session that has ended cannot be warm started from,
        deployed, or even named by the dialog: nothing reads a run directory
        back off disk. They are also the entire cost of keeping rounds around.
        Measured on a real project: 10.9 MB of checkpoints against 3 MB of
        results.csv and plots per round, and eight abandoned sessions had left
        260 MB of weights nothing could ever load again.

        That asymmetry is the policy. The sweep takes `weights/` and leaves the
        run directory, because results.csv and the plots are what the plan says
        survives a session and they cost almost nothing to keep.
        """
        root = self.runs_root()
        if not os.path.isdir(root):
            return []

        mine = self.session_run_dirs()
        candidates = []
        for name in os.listdir(root):
            run_dir = os.path.abspath(os.path.join(root, name))
            if not name.startswith('round_') or not os.path.isdir(run_dir):
                continue
            if run_dir in mine:
                continue
            candidates.append(run_dir)

        return self.weights_on_disk(candidates)

    @staticmethod
    def as_megabytes(size):
        """A size in MB, for a sentence rather than a table."""
        return f"{size / (1024 * 1024):.0f} MB"

    def delete_weights(self, entries):
        """Remove the weights directories in `entries`, returning what went.

        Failures are reported and skipped rather than raised: a checkpoint held
        open by something else is a reason to keep the other ones, not to
        abandon the sweep.
        """
        removed = 0
        freed = 0
        for directory, size in entries:
            try:
                shutil.rmtree(directory)
            except Exception as e:
                print(f"Warning: could not delete {directory}: {e}")
                continue
            removed += 1
            freed += size

        for entry in self.round_history:
            weights = entry.get('weights')
            if weights and not os.path.isfile(weights):
                entry['weights'] = None

        return removed, freed

    def reviewed_image_count(self):
        """How many images carry a review state for this task."""
        return sum(1 for _path, raster in self.image_rasters()
                   if self.review_state(raster) is not None)

    def new_session(self):
        """Discard this session and start a fresh one, with the disk clean-up.

        The dialog is built once per task on the MainWindow and re-shown, so
        closing it never ended a session: the round counter, the best model and
        every image's review state carried straight into what looked like a new
        sitting. This is the seam that actually ends one.

        The clean-up rides along here because this is the only moment the answer
        to "which rounds no longer matter?" is knowable: the session that owned
        them is the one being thrown away.
        """
        if self.worker is not None:
            QMessageBox.information(
                self, "Round Still Running",
                "A round is training. Stop it before starting a new session.")
            return

        rounds = len(self.round_history)
        reviewed = self.reviewed_image_count()
        mine = self.weights_on_disk(self.session_run_dirs())
        stale = self.stale_round_weights()

        prompt = SessionResetPrompt(
            rounds=rounds,
            reviewed=reviewed,
            mine=sum(size for _dir, size in mine),
            mine_count=len(mine),
            stale=sum(size for _dir, size in stale),
            stale_count=len(stale),
            parent=self)
        if prompt.exec_() != QDialog.Accepted:
            return

        entries = []
        if prompt.delete_mine():
            entries += mine
        if prompt.delete_stale():
            entries += stale

        removed, freed = self.delete_weights(entries)
        self.reset_session()

        message = "New session started."
        if removed:
            message += f" Freed {self.as_megabytes(freed)} from {removed} round folders."
        self.show_status(f"Active Learning: {message}")
        print(f"Active Learning: {message}")

    def reset_session(self):
        """Forget everything one sitting accumulated, and redraw.

        Annotations are deliberately not touched -- not the predictions waiting
        for review, and certainly not what the user has confirmed. What goes is
        the session's own bookkeeping: the rounds, the model they produced, and
        the per-image review state that says which images have been through the
        loop.
        """
        self.round_history = []
        self.best_round = None
        self.last_model_path = None
        self.last_round_outcome = None
        self.last_predicted_images = []
        self.predicted_ever = set()
        self.last_skipped = {}
        self.last_acquisition = 'counts'
        self.baseline_counts = None
        # The frozen class order goes too. It exists to keep round n+1's class
        # indices lined up with round n's weights, and there are no weights now.
        self.class_memory = []

        for _image_path, raster in self.image_rasters():
            states = getattr(raster, 'active_learning', None)
            if isinstance(states, dict):
                states.pop(self.task, None)

        self.populate_history_table()
        self.rerun_button.setEnabled(False)
        self.epoch_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.refresh_dataset(quiet=False)
        self.update_next_step()
        self.update_status_message()

    def stop_round(self):
        """Ask the running round to end after the current epoch."""
        if self.worker is None:
            return
        if self._pending:
            self._pending['stopped'] = True
        self.stop_button.setEnabled(False)
        self.worker.request_stop()
        self.progress_label.setText(
            "Stopping after this epoch. What has been trained so far is still saved.")
        self.show_status("Active Learning: stopping the round after this epoch...")

    def on_training_status(self, message):
        """Relay the worker's status to the panel that is already being watched."""
        self.progress_label.setText(message)
        self.progress_label.setVisible(True)
        self.post_status(message)

    def on_epoch_completed(self, epoch, total_epochs, losses, learning_rate):
        """Show per-epoch progress, which is the only sign a round is alive."""
        parts = [f"Epoch {epoch}/{total_epochs}"]
        if losses:
            parts.append(", ".join(f"{name} {value}" for name, value in list(losses.items())[:3]))
        parts.append(f"lr {learning_rate:.5f}")
        round_number = self._pending['round'] if self._pending else len(self.round_history) + 1
        self.epoch_bar.setValue(epoch)
        line = f"Round {round_number} · " + " · ".join(parts)
        self.progress_label.setText(line)
        self.progress_label.setVisible(True)

        # The same detail in the status bar, where it is visible with the dialog
        # behind the canvas -- which is where it will be for most of a round.
        # Without the round number: the headline segment beside it already says
        # which round this is, and saying it twice is what makes a status line
        # stop being read.
        self.post_status(" · ".join(parts))

    def training_params(self, data_path, model, run_name):
        """The parameter set for one round.

        Deliberately the same set the Train Model dialog sends, key for key, so
        a round is not a differently-configured kind of training that happens to
        share a worker -- and so a parameter added there is a visible omission
        here rather than a silent difference in behaviour.
        """
        params = {
            'exist_ok': TRAINING_DEFAULTS['exist_ok'],
            'plots': TRAINING_DEFAULTS['plots'],
            'task': self.task,
            'data': data_path,
            'model': model,
            # Absolute, and anchored to the project rather than to the working
            # directory. Ultralytics resolves a relative `project` under its own
            # runs directory -- the round would land in
            # runs/detect/Data/ActiveLearning/... while everything here looked
            # in Data/ActiveLearning/..., so results.csv and best.pt were never
            # found and the whole post-round chain silently did nothing.
            'project': os.path.abspath(self.runs_root()),
            'name': run_name,
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'single_cls': self.single_class_combo.currentText() == "True",
            'mask_ratio': self.mask_ratio_spinbox.value(),
            'weighted': self.weighted_combo.currentText() == "True",
            'freeze_layers': self.freeze_layers_spinbox.value(),
            'dropout': self.dropout_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'workers': self.workers_spinbox.value(),
            'cache': self.cache_combo.currentData(),
            'save': self.save_combo.currentText() == "True",
            'save_period': self.save_period_spinbox.value(),
            'val': self.val_combo.currentText() == "True",
            'verbose': self.verbose_combo.currentText() == "True",
        }
        return params

    def on_training_error(self, message):
        """A failed round must leave the session usable.

        Out of memory is the expected failure rather than an exotic one, and
        halving the batch is what a person does next anyway -- so it is offered
        here instead of leaving them to find the spinbox.
        """
        round_number = self._pending['round'] if self._pending else len(self.round_history) + 1
        self.finish_round()

        # Recorded as an outcome rather than only shown in a message box: the
        # panel is what the user looks at afterwards, and a dismissed dialog
        # used to leave it describing the round before last as though this one
        # had never run.
        self.last_round_outcome = {
            'round': round_number,
            'failed': True,
            'map50': None,
            'fitness': None,
            'weights': None,
            'predictions': 0,
            'predicted_on': 0,
            'candidates': 0,
            'skipped': {},
            'deployed': False,
            'stopped': False,
        }
        self.update_next_step()

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
        improved = False
        if pending:
            weights = os.path.join(pending['run_dir'], 'weights', 'best.pt')
            if not os.path.isfile(weights):
                weights = None

            self.record_round(pending, weights)
            improved = self.round_improved()
            if weights and improved:
                self.last_model_path = weights
                self.best_round = self.round_history[-1]

        self.finish_round()

        # Start the outcome fresh: the passes below fill in what they produced,
        # and a stale count from the previous round would read as this one's.
        self.last_round_outcome = {
            'round': pending['round'] if pending else len(self.round_history),
            'map50': self.round_history[-1]['map50'] if self.round_history else None,
            'fitness': self.round_history[-1]['fitness'] if self.round_history else None,
            'weights': weights,
            'predictions': 0,
            'predicted_on': 0,
            'candidates': 0,
            'skipped': {},
            'deployed': False,
            'improved': improved,
            'stopped': bool(pending.get('stopped')) if pending else False,
        }

        # Everything downstream uses the best model, not the newest one. A round
        # that came out worse is recorded and then set aside: warm-starting from
        # it, predicting with it or deploying it would all be a step backwards
        # that nothing later in the session could undo.
        self._post_round = True
        try:
            if weights and improved:
                # Always, rather than behind a switch. A round that produced a
                # better model and then left it unloaded is a round whose result
                # cannot be used anywhere else in the application, and the pair
                # of Predict / Deploy switches read as if they were alternatives
                # when they are two unrelated things.
                self.last_round_outcome['deployed'] = self.deploy_model(weights, quiet=True)
                self.after_training(self.last_model_path)
        finally:
            self._post_round = False

        self.update_next_step()
        self.update_status_message()
        # The summary is on this tab, so land the user where the answer is.
        self.tabs.setCurrentIndex(1)

    def round_improved(self):
        """Whether the round just recorded beat the best comparable one before it.

        Comparable means the same label set: a round that added a class is a
        different measurement, not a worse one, so it is always adopted. So is a
        round with no metric at all -- validation turned off, or results.csv
        unreadable -- because refusing to adopt on missing evidence would leave a
        session that can never adopt anything.
        """
        if len(self.round_history) < 2:
            return True

        latest = self.round_history[-1]
        metric = Base.round_score(latest)
        if metric is None:
            return True

        comparable = [Base.round_score(entry) for entry in self.round_history[:-1]
                      if Base.round_score(entry) is not None
                      and entry.get('labels') == latest.get('labels')]
        if not comparable:
            return True
        return metric > max(comparable)

    def record_outcome(self, **fields):
        """Fold what a post-round pass produced into this round's summary."""
        if self.last_round_outcome is None:
            return
        self.last_round_outcome.update(fields)

    @staticmethod
    def next_step_message(outcome):
        """The whole hand-off as one sentence, for the status bar and for tests."""
        headline, detail, action = Base.next_step_parts(outcome)
        return headline + " " + detail + action

    @staticmethod
    def next_step_parts(outcome):
        """(headline, detail, action) for one finished round.

        Written as an instruction rather than a report. The counts alone leave
        the user to work out that unconfirmed predictions are the input to the
        next round, which is the one thing the loop depends on them doing -- and
        the round that produces nothing needs to say so most of all, since that
        is the one where it is least obvious what went wrong.

        Split into three because the panel shows them in three different places:
        the headline is a coloured state line, the numbers inside it have their
        own tiles, and only the detail and the instruction are left as prose.
        """
        parts = [f"Round {outcome['round']} " + ("stopped early" if outcome.get('stopped')
                                                  else "finished")]
        if outcome.get('map50') is not None:
            parts.append(f"mAP50 {outcome['map50']:.3f}")
        # Both, because mAP50 alone is the number that stops moving: it is the
        # one people read, and the one that says a round changed nothing when
        # the model has in fact improved.
        if outcome.get('fitness') is not None:
            parts.append(f"mAP50-95 {outcome['fitness']:.3f}")
        if outcome.get('deployed'):
            parts.append("model deployed")
        headline = " · ".join(parts) + "."

        predictions = outcome.get('predictions', 0)

        skipped = Base.skipped_note(outcome.get('skipped'))

        if predictions:
            detail = (f"{predictions} predictions are waiting on "
                      f"{outcome.get('predicted_on', 0)} images. ")
            action = ("Next: walk the queue with Next and Previous, marking each "
                      "prediction verified or for review.")
        elif outcome.get('weights') and not outcome.get('candidates'):
            # What was skipped is the whole answer here, and it used to be
            # invisible: a project where every image already carries unreviewed
            # predictions has nothing left to spend a budget on, and so does one
            # where the only candidate was the image on screen.
            detail = "No images were left to predict on. " + skipped
            action = ("Next: review what is already waiting, raise the Image Budget "
                      "on the Setup tab, or add more images.")
        elif outcome.get('weights'):
            # The threshold is named because it is the usual answer and the
            # least visible one: the round ran, the model predicted, and every
            # detection fell below the bar the user set somewhere else.
            detail = (f"The model found nothing above the uncertainty threshold on "
                      f"{outcome.get('candidates', 0)} images "
                      f"(threshold {outcome.get('threshold', 0):.2f}). " + skipped)
            action = ("Next: lower the uncertainty threshold and press Re-run "
                      "Predictions, raise the Image Budget so the round reaches more "
                      "images, or annotate more examples and train again.")
        else:
            detail = "Training produced no weights. "
            action = "Next: check the console output for what went wrong."

        return headline, detail, action

    @staticmethod
    def skipped_note(skipped):
        """What acquisition passed over, as a sentence, or "" if it passed over nothing.

        The open image is the one worth naming: it is skipped so predictions do
        not land under the cursor mid-edit, and somebody who has just run a model
        on that image by hand and then watched a round produce nothing there has
        no way to find that out.
        """
        if not skipped:
            return ""

        parts = []
        if skipped.get('open'):
            parts.append("the image you have open")
        if skipped.get('awaiting'):
            parts.append(f"{skipped['awaiting']} already awaiting review")
        if skipped.get('background'):
            parts.append(f"{skipped['background']} confirmed empty")
        if not parts:
            return ""
        return "Skipped: " + ", ".join(parts) + ". "

    def set_headline(self, state, text):
        """Set the callout's state: its tint, its glyph, its pill and its heading."""
        colour = STATE_COLOR[state]
        self.callout_frame.setStyleSheet(callout_style(colour))
        self.callout_icon.setText(STATE_ICON[state])
        self.callout_icon.setStyleSheet(callout_badge_style(colour))
        self.headline_label.setText(text)
        self.headline_label.setStyleSheet(f"color: {colour.name()};")
        self.state_pill.setText(STATE_WORD[state])
        self.state_pill.setStyleSheet(state_pill_style(colour))
        self.state_pill.setToolTip(STATE_TOOLTIP[state])
        self.session_state = state
        self.post_status()

    @staticmethod
    def next_step_state(outcome):
        """Which of the five states a finished round left the session in.

        A round that trained and one that fell over used to read the same at a
        glance: one paragraph, same weight, same colour. The distinction that
        matters is whether there is something to do, something to fix, or
        nothing to worry about.
        """
        if not outcome:
            return 'idle', "No rounds yet"
        number = outcome.get('round', 0)
        if outcome.get('failed'):
            return 'error', f"Round {number} failed"
        if not outcome.get('weights'):
            return 'error', f"Round {number} produced no weights"
        if outcome.get('stopped'):
            return 'warn', f"Round {number} stopped early"
        return 'ok', f"Round {number} finished"

    def latest_delta(self):
        """The change the last scored round made, for the Change tile.

        Rounds with no metric are skipped rather than ending the chain: a round
        that failed or was stopped before it validated should not blank a
        comparison the two rounds either side of it can still make.

        Anything metric_delta cannot express as a signed number -- no comparable
        round, or a round that changed its label set -- comes back blank. The
        table has room to say "new labels"; a tile the width of four characters
        does not, and a tile is read as a number.
        """
        scored = [entry for entry in self.round_history
                  if Base.round_score(entry) is not None]
        if len(scored) < 2:
            return BLANK_STAT
        change = self.metric_delta(Base.round_score(scored[-1]),
                                   Base.round_score(scored[-2]),
                                   scored[-1].get('labels'), scored[-2].get('labels'))
        return change if change[:1] in "+-" and change != "-" else BLANK_STAT

    def update_stat_tiles(self):
        """Fill the four numbers, colouring the change by which way it went."""
        metric = None
        for entry in reversed(self.round_history):
            if entry.get('map50') is not None:
                metric = entry['map50']
                break

        delta = self.latest_delta()
        self.stat_values['map50'].setText(BLANK_STAT if metric is None else f"{metric:.3f}")
        self.stat_values['delta'].setText(delta)
        self.stat_values['awaiting'].setText(str(self.awaiting.get('total', 0)))
        self.stat_values['images'].setText(str(self.awaiting.get('images', 0)))

        if delta.startswith("+"):
            colour = STATE_OK
        elif delta.startswith("-") and delta != BLANK_STAT:
            colour = STATE_ERROR
        else:
            colour = app_theme.TEXT_SECONDARY_COLOR
        self.stat_values['delta'].setStyleSheet(f"color: {colour.name()};")

    def update_review_controls(self):
        """Say where the queue stands, and disable what would do nothing."""
        total = self.awaiting.get('total', 0)
        images = self.awaiting.get('images', 0)

        if total:
            plural = "image" if images == 1 else "images"
            self.queue_label.setText(
                f"{total} predictions awaiting review across {images} {plural}")
        else:
            self.queue_label.setText(
                "Nothing awaiting review - train a round to get predictions to confirm")

        position = self.review_position() if total else ""
        self.position_label.setText(position or "--")
        self.position_label.setStyleSheet(review_position_style(bool(position)))

        for button in (self.previous_button, self.next_button,
                       self.verify_button, self.mark_review_button):
            button.setEnabled(bool(total))

        self.rerun_button.setEnabled(bool(self.last_model_path) and self.worker is None)

    def keep_in_front(self):
        """Take focus back after an action that moved the canvas.

        Opening an image activates the main window, and the user pressed a
        button in here: the next press should land in the same place as the
        last one, without a trip through the taskbar. Minimized is left alone --
        that is the user saying they want the dialog out of the way.
        """
        if not self.isVisible() or self.isMinimized():
            return
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def review_position(self):
        """"3 of 24", when the selection is somewhere in the queue."""
        annotation = self.selected_review_annotation()
        if annotation is None:
            return ""
        order = self.review_order()
        try:
            return f"{order.index(annotation) + 1} of {len(order)}"
        except ValueError:
            return ""

    # ------------------------------------------------------------------
    # Walking the queue
    # ------------------------------------------------------------------

    def review_order(self):
        """The unreviewed annotations, in the order Previous and Next walk them.

        Grouped by image, then least confident first within an image. Ordering
        purely by confidence would be a better use of attention in the abstract
        and a worse one in practice: it jumps between images on every press, and
        the cost of loading a new image dwarfs the difference between the third
        and fourth most uncertain box on the one already open.
        """
        annotations = self.unverified_annotations()
        annotations.sort(key=lambda a: (a.image_path,
                                        self.top_confidence(a) or 0.0,
                                        str(a.id)))
        return annotations

    def selected_review_annotation(self):
        """The selected annotation, if it is one the queue is about."""
        selected = list(getattr(self.annotation_window, 'selected_annotations', []))
        if not selected:
            return None
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        for annotation in selected:
            if isinstance(annotation, allowed_types) and not getattr(annotation, 'verified', True):
                return annotation
        return None

    def go_to_annotation(self, annotation):
        """Open the annotation's image if needed, then select and centre it."""
        try:
            if annotation.image_path != getattr(self.annotation_window,
                                                'current_image_path', None):
                self.image_window.load_image_by_path(annotation.image_path)
            self.annotation_window.unselect_annotations()
            self.annotation_window.select_annotation(annotation)
            self.annotation_window.center_on_annotation(annotation)
        except Exception as e:
            print(f"Warning: could not open the next annotation for review: {e}")
            return False
        return True

    def step_review(self, delta):
        """Move to the next or previous annotation awaiting review.

        Wraps around, because a review pass is a loop rather than a list with
        an end: running off the last one and being told so is less useful than
        arriving back at the first one still waiting.
        """
        order = self.review_order()
        if not order:
            self.show_status("Active Learning: nothing is awaiting review.")
            return

        current = self.selected_review_annotation()
        if current is None or current not in order:
            target = order[0] if delta > 0 else order[-1]
        else:
            target = order[(order.index(current) + delta) % len(order)]

        self.go_to_annotation(target)
        self.update_review_controls()
        self.keep_in_front()

    def act_on_current_annotation(self, action, description):
        """Apply `action` to the annotation under review, then move on.

        The annotation leaves the queue as a result, so the next one to look at
        is whatever has taken its place -- not the one after it, which would skip
        an annotation on every press.
        """
        order = self.review_order()
        if not order:
            self.show_status("Active Learning: nothing is awaiting review.")
            return

        current = self.selected_review_annotation()
        index = order.index(current) if current in order else 0
        annotation = current if current in order else order[0]

        try:
            action(annotation)
        except Exception as e:
            self.show_status(f"Active Learning: could not {description}: {e}")
            return

        # The raster's unverified counters are what the Needs Review filter
        # reads, and update_user_confidence does not touch them. Refreshed here
        # rather than left to a signal, which only reaches annotations connected
        # to the image currently on the canvas.
        image_path = getattr(annotation, 'image_path', None)
        if image_path:
            try:
                self.image_window.update_image_annotations(image_path, update_counts=False)
            except Exception:
                pass

        remaining = self.review_order()
        if remaining:
            self.go_to_annotation(remaining[min(index, len(remaining) - 1)])
        self.keep_in_front()
        self.refresh_dataset()
        self.show_status(f"Active Learning: {description}. {len(remaining)} left to review.")

    def verify_current_annotation(self):
        """Confirm the prediction under review: the model was right."""
        self.act_on_current_annotation(
            lambda annotation: annotation.update_verified(True), "marked verified")

    def review_current_annotation(self):
        """Park the prediction under review as Review: no decision yet.

        Relabelling it Review takes it out of the queue without teaching the
        model anything, because project_annotations() excludes the Review label
        from every round. That is the honest answer to a prediction a person
        cannot judge, and it is otherwise unreachable without leaving the loop.
        """
        review_label = self.label_window.get_review_label()
        if review_label is None:
            self.show_status("Active Learning: this project has no Review label.")
            return
        self.act_on_current_annotation(
            lambda annotation: annotation.update_user_confidence(review_label),
            "marked for review")

    def next_step_instruction(self, outcome):
        """The one line telling the user what to do, with no numbers in it."""
        if not outcome:
            if self.awaiting.get('total'):
                return ("Next: walk the queue with Next and Previous, marking each "
                        "prediction verified or for review." + self.auto_train_note())
            return ("Next: confirm a handful of examples of each label, then train the "
                    "first round from the Setup tab.")
        if outcome.get('failed'):
            return ("Next: the round was abandoned and your annotations are unchanged. "
                    "Check the console for the error, then train again.")

        auto = self.auto_train_note()
        _headline, detail, action = self.next_step_parts(outcome)
        if outcome.get('predictions'):
            # The tiles carry the counts now. Repeating them in a sentence
            # underneath is the wall of text this panel was rebuilt to remove --
            # and it pushed the instruction, the only part that asks the user to
            # do something, to the end of a paragraph.
            return action + auto
        # The rounds that produced nothing are the exception: what went wrong is
        # not a number, and no tile can show it.
        return detail + action + auto

    def auto_train_note(self):
        """What the automatic trigger is still waiting for, if it is on.

        Named rather than left implicit: a round that starts by itself is
        surprising if you did not know it could, and one that never starts is
        worse -- a rare label short of its target will hold the trigger
        indefinitely, and nothing else on screen would say so.
        """
        if self.auto_train_combo.currentText() != "True":
            return ""
        if not self._auto_shortfall:
            return " The next round will start on its own."

        short = sorted(self._auto_shortfall.items(), key=lambda item: -item[1])[:3]
        detail = ", ".join(f"{code} {count}" for code, count in short)
        if len(self._auto_shortfall) > len(short):
            detail += f", +{len(self._auto_shortfall) - len(short)} more"
        return (f" Auto Train is waiting on {len(self._auto_shortfall)} labels "
                f"({detail} to go).")

    @staticmethod
    def instruction_html(text):
        """The instruction, with "Next:" starting a paragraph of its own.

        It is the only line on the tab that asks the user to do something, and
        as a clause at the end of a sentence it read as more of the report in
        front of it. A blank line and a bold lead-in is the difference between
        a panel that reports and one that instructs.
        """
        lead, marker, rest = text.partition("Next:")
        lead = escape(lead.strip())
        if not marker:
            return lead
        instruction = "<b>Next:</b> " + escape(rest.strip())
        return "<br><br>".join(part for part in (lead, instruction) if part)

    def update_next_step(self):
        """Put the round's summary on the panel, and enable what it offers."""
        outcome = self.last_round_outcome

        if self.worker is not None:
            number = self._pending['round'] if self._pending else len(self.round_history) + 1
            self.set_headline('running', f"Round {number} training")
        else:
            state, headline = self.next_step_state(outcome)
            self.set_headline(state, headline)

        self.update_stat_tiles()
        self.next_step_label.setText(
            self.instruction_html(self.next_step_instruction(outcome)))
        self.update_review_controls()
        self.post_status()

    # ------------------------------------------------------------------
    # The review queue
    # ------------------------------------------------------------------

    def unverified_annotations(self, image_path=None):
        """Unreviewed annotations of this task, on one image or the whole project."""
        allowed_types = InPlaceTraining.TASK_ANNOTATION_TYPES.get(self.task, ())
        if image_path is not None:
            source = self.annotation_window.get_image_annotations(image_path)
        else:
            source = list(self.annotation_window.annotations_dict.values())
        return [annotation for annotation in source
                if isinstance(annotation, allowed_types)
                and not getattr(annotation, 'verified', True)]

    @staticmethod
    def top_confidence(annotation):
        """The annotation's best machine confidence, or None if it has none."""
        confidences = getattr(annotation, 'machine_confidence', None)
        if not confidences:
            return None
        try:
            return float(max(confidences.values()))
        except Exception:
            return None

    def apply_needs_review_filter(self):
        """Point the Image Window at the images a round left work on.

        Applied for the user rather than offered as a button: a round that
        produces predictions has, by definition, made the rest of the project
        irrelevant until they are dealt with, and the alternative is working out
        which twenty images out of a thousand changed.

        Synchronous on purpose: the threaded path returns before the table model
        has been updated, so filtered_paths would still hold the previous
        filter's answer.
        """
        try:
            self.image_window.filter_combo.check_item("Needs Review")
            self.image_window.filter_images(use_threading=False)
        except Exception as e:
            print(f"Warning: could not apply the Needs Review filter: {e}")

    def deploy_target(self):
        """The Deploy Model dialog matching this dialog's task."""
        attribute = f"{self.task}_deploy_model_dialog"
        return getattr(self.main_window, attribute, None)

    def deploy_model(self, weights, quiet=False):
        """Load `weights` into the task's Deploy Model dialog.

        Uses the dialog's own load path so the class-name table, the label
        mapping and the status text all end up in the state they would be in
        had the user loaded it by hand -- a half-loaded dialog claiming to hold
        a model it never mapped is worse than not deploying at all.

        Quietly, though: a round's model has no class_mapping.json and never
        will. It was trained on this project's labels, so its class names are
        their short codes -- "no class mapping found, shall I invent generic
        labels?" is a question with one answer, asked in the middle of a round
        the user is not watching, and a "Model loaded successfully" box behind
        it for every round after that.
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
            if hasattr(dialog, 'load_model_quietly'):
                dialog.load_model_quietly()
            else:
                dialog.load_model()
        except Exception as e:
            # Out of memory is the expected failure here rather than an exotic
            # one: the round has just finished training and the card may still
            # be holding what it needed. Nothing about it should cost the round,
            # which is already recorded, so it is reported and stepped over --
            # and the cached blocks are released so the prediction pass that
            # follows has a chance of fitting.
            if self.looks_like_oom(e):
                self.free_gpu_memory()
                message = ("Ran out of memory loading the round's model into the Deploy "
                           "dialog. The round itself is unaffected; deploy it by hand "
                           "once something else has released the card.")
            else:
                message = f"Could not deploy the model: {e}"
            print(f"Warning: {message}")
            if not quiet:
                QMessageBox.warning(self, "Deploy Failed", message)
            return False
        except BaseException as e:
            # load_model reaches Ultralytics, which can raise things that are not
            # Exceptions. Letting one through here would abandon the prediction
            # pass and the round summary along with it.
            print(f"Warning: deploying the round's model failed hard: {e!r}")
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
        self.stop_button.setEnabled(False)
        # Epoch 27 of 30 is not something to leave on screen for the rest of the
        # session; the headline and the tiles say what came of it.
        self.epoch_bar.setVisible(False)
        self.progress_label.setVisible(False)
        # The epoch line describes a round that is over. What the round produced
        # arrives through show_status a moment later.
        self._activity = ""
        # A round left running behind a closed dialog keeps reporting until it
        # finishes; this is where that ends.
        if not self.isVisible():
            self.end_status_reporting()

    def record_round(self, pending, weights):
        """Add a row describing what this round trained on and how it scored."""
        dataset = pending['dataset']
        metrics = self.read_metrics(pending['run_dir']) or {}

        self.round_history.append({
            'round': pending['round'],
            # The run directory, not just the weights inside it: pruning clears
            # 'weights' and the housekeeping still has to know which folders on
            # disk belong to this session.
            'run_dir': os.path.abspath(pending['run_dir']),
            'train_images': dataset.image_count('train'),
            'background': dataset.negative_count('train'),
            'annotations': dataset.annotation_count('train'),
            'map50': metrics.get('map50'),
            'fitness': metrics.get('fitness'),
            'epoch': metrics.get('epoch'),
            'weights': weights,
            'labels': pending['labels'],
            'stopped': bool(pending.get('stopped')),
        })

        self.populate_history_table()
        self.prune_round_weights()

    def populate_history_table(self):
        """Rebuild the Rounds table from the recorded history."""
        self.history_table.setRowCount(0)
        previous = None
        previous_labels = None
        for entry in self.round_history:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            metric = entry.get('map50')
            fitness = entry.get('fitness')
            score = self.round_score(entry)
            values = {
                HIST_ROUND: entry.get('round', row + 1),
                HIST_IMAGES: entry.get('train_images', 0),
                HIST_BACKGROUND: entry.get('background', 0),
                HIST_ANNOTATIONS: entry.get('annotations', 0),
                HIST_MAP: f"{metric:.3f}" if metric is not None else "-",
                HIST_FITNESS: f"{fitness:.3f}" if fitness is not None else "-",
                # Against the score the round was judged by, not against mAP50:
                # a Change column that moves while the adopted model does not
                # would be answering a different question from the one the
                # session acts on.
                HIST_DELTA: self.metric_delta(score, previous,
                                              entry.get('labels'), previous_labels),
            }
            for column, value in values.items():
                self.history_table.setItem(row, column, self.centered_item(value))

            # The same green/red the Change tile uses. A column of signed
            # numbers is read for its direction first, and the sign alone is a
            # thin thing to read it from.
            self.color_delta_cell(row, values[HIST_DELTA])

            if entry.get('stopped'):
                item = self.history_table.item(row, HIST_ROUND)
                if item is not None:
                    item.setText(f"{values[HIST_ROUND]} (stopped)")

            if score is not None:
                previous = score
                previous_labels = entry.get('labels')

    def color_delta_cell(self, row, text):
        """Colour one Change cell by which way the round went.

        Only a signed number gets a colour: "-" means there was nothing to
        compare against and "new labels" means the comparison would be
        meaningless, and neither is a regression to paint red.
        """
        item = self.history_table.item(row, HIST_DELTA)
        if item is None:
            return
        if text.startswith("+"):
            item.setForeground(QBrush(STATE_OK))
        elif text.startswith("-") and len(text) > 1:
            item.setForeground(QBrush(STATE_ERROR))

    @staticmethod
    def metric_delta(metric, previous, labels, previous_labels):
        """The change against the last comparable round.

        Comparable means the same label set. A round that added a class is not
        a worse round because its mAP fell -- it is a different measurement, and
        reporting a drop there would be actively misleading.
        """
        if metric is None or previous is None:
            return "-"
        if labels is not None and previous_labels is not None and labels != previous_labels:
            return "new labels"
        change = metric - previous
        return f"{change:+.3f}"

    def prune_round_weights(self):
        """Delete checkpoints from rounds nobody will go back to.

        Every round writes a full Ultralytics run directory, and nothing removed
        them: ten rounds is ten copies of best.pt and last.pt. Only the weights
        go -- results.csv and the plots are what the Rounds table and any later
        look at the run are built on, and they are small.
        """
        keep = {entry.get('weights') for entry in self.round_history[-KEEP_ROUND_WEIGHTS:]}
        keep.add(self.last_model_path)

        for entry in self.round_history[:-KEEP_ROUND_WEIGHTS]:
            weights = entry.get('weights')
            if not weights or weights in keep:
                continue
            directory = os.path.dirname(weights)
            if not os.path.isdir(directory):
                continue
            try:
                shutil.rmtree(directory, ignore_errors=True)
                entry['weights'] = None
            except Exception as e:
                print(f"Warning: could not prune old round weights {directory}: {e}")

    @staticmethod
    def read_metrics(run_dir):
        """Read the round's scores out of results.csv, or None when unavailable.

        Returns {'map50', 'fitness', 'epoch'}, all of which describe **the same
        epoch**: the one with the best fitness, which is the epoch ``best.pt``
        was saved from. Reading the last row instead -- which is what this used
        to do -- reports a different model from the one the round goes on to
        deploy and warm start from, and with early stopping they are routinely
        several epochs apart.

        Fitness is Ultralytics' own, recomputed here because the trainer pops it
        out of the metrics dict before writing the csv. In 8.4.82 that is
        ``mAP50-95`` alone for detection (weights ``[0, 0, 0, 1]`` over
        ``[P, R, mAP50, mAP50-95]``) and the sum of the box and mask figures for
        segmentation, which is why every ``mAP50-95`` column present is added.

        It is the right criterion for "did this round improve the model?" and
        mAP50 is not: mAP50 saturates -- a project whose objects are easy to find
        sits at 0.99 from round two onwards -- while mAP50-95 keeps moving,
        because it also measures how well the boxes are placed.
        """
        results_path = os.path.join(run_dir, 'results.csv')
        if not os.path.isfile(results_path):
            return None
        try:
            with open(results_path, 'r') as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
            if len(lines) < 2:
                return None

            header = [column.strip() for column in lines[0].split(',')]
            map50_columns = [i for i, name in enumerate(header)
                             if 'mAP50' in name and '95' not in name]
            fitness_columns = [i for i, name in enumerate(header) if 'mAP50-95' in name]
            if not map50_columns and not fitness_columns:
                return None

            best = None
            for line in lines[1:]:
                cells = line.split(',')

                def value(index):
                    try:
                        return float(cells[index])
                    except (IndexError, ValueError):
                        return None

                fitness_parts = [value(i) for i in fitness_columns]
                fitness = (sum(part for part in fitness_parts if part is not None)
                           if any(part is not None for part in fitness_parts) else None)
                map50_parts = [value(i) for i in map50_columns]
                map50 = next((part for part in map50_parts if part is not None), None)
                if fitness is None and map50 is None:
                    continue

                # Ranked by fitness when there is one, so the row chosen is the
                # row best.pt came from. mAP50 only stands in when validation
                # produced no mAP50-95 column at all.
                rank = fitness if fitness is not None else map50
                epoch = value(0)
                if best is None or rank > best['rank']:
                    best = {'rank': rank, 'map50': map50,
                            'fitness': fitness, 'epoch': epoch}

            if best is None:
                return None
            return {'map50': best['map50'], 'fitness': best['fitness'],
                    'epoch': best['epoch']}
        except Exception:
            return None

    @staticmethod
    def read_metric(run_dir):
        """The round's mAP50, for the column that reports it."""
        metrics = Base.read_metrics(run_dir)
        return None if metrics is None else metrics.get('map50')

    @staticmethod
    def round_score(entry):
        """What a round is judged by: its fitness, or its mAP50 if it has none.

        One accessor rather than a key read in five places, because the fallback
        has to be the same everywhere: a round scored one way and compared
        another is a round that can be adopted for the wrong reason.
        """
        if entry is None:
            return None
        fitness = entry.get('fitness')
        return entry.get('map50') if fitness is None else fitness

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def candidate_images(self, budget, include_current=False):
        """Choose which un-reviewed images are worth spending the budget on.

        Two pools rather than one ranked list, because "where should the model
        look next?" has two answers and the old ranking only gave the first:

        * **untouched** -- images with nothing on them, where the model may find
          objects nobody has got to yet.
        * **working** -- images the user has already annotated. Somewhere they
          decided was worth their time, which makes the model's mistakes there
          both likelier to matter and cheaper to correct.

        Sorting purely by annotation count sent every round to the emptiest
        corner of the project and never checked the model where the user was
        actually working. The budget is now split EXPLORE_SHARE / the rest, and
        either pool takes what the other cannot fill.

        Within a pool: images no round has predicted on yet come first, so the
        budget moves across the project instead of re-picking the same handful;
        then fewest annotations; then a stable hash, so the choice does not
        wander between rounds for no reason.

        Within those tiers, when the project carries pooled per-image
        descriptors, the order is decided by **k-center greedy** rather than by
        annotation count: the image furthest from everything already covered
        first, each pick joining the covered set. That is what stops a budget of
        ten going on ten pictures of the same sand patch. It falls back to the
        counting order silently when the descriptors are not there --- see
        rank_by_diversity.

        The Explorer's FAISS index is not what this reads, and cannot be: it is
        keyed by annotation, and images with nothing on them -- exactly the ones
        being ranked -- are not in it at all. The descriptors come from the
        feature bake instead, one pooled [C] vector per image.

        :param budget: How many images to return at most.
        :param include_current: Whether the image open on the canvas may be
                                chosen. False for the automatic pass after a
                                round, where predictions would land under the
                                user's cursor mid-edit; True when the user asked
                                for this pass themselves, since the image they
                                are looking at is usually the one they meant.
        """
        current = None if include_current else getattr(self.annotation_window,
                                                       'current_image_path', None)

        skipped = {'open': 0, 'awaiting': 0, 'background': 0}
        untouched = []
        working = []
        # Every annotated image, eligible or not: the set diversity measures
        # distance *from*. An image excluded from this round still covers the
        # part of the project it sits in.
        covered_paths = []
        for image_path, raster in self.image_rasters():
            annotations = self.annotation_window.get_image_annotations(image_path)
            if annotations:
                covered_paths.append(image_path)

            if current is not None and image_path == current:
                skipped['open'] += 1
                continue

            unverified = sum(1 for a in annotations if not getattr(a, 'verified', True))
            if unverified:
                # Already carrying work for the user; do not pile more on.
                skipped['awaiting'] += 1
                continue
            if self.review_state(raster) == REVIEW_REVIEWED and not annotations:
                # A confirmed background image. Predicting on it again would
                # re-propose exactly what the user just deleted.
                skipped['background'] += 1
                continue

            entry = (1 if image_path in self.predicted_ever else 0,
                     len(annotations),
                     InPlaceTraining.stable_fraction(image_path),
                     image_path)
            (untouched if not annotations else working).append(entry)

        untouched.sort()
        working.sort()
        self.last_skipped = skipped

        untouched, working = self.rank_by_diversity(untouched, working,
                                                    covered_paths, budget)
        return self.spend_budget(untouched, working, budget)

    @staticmethod
    def spend_budget(untouched, working, budget):
        """Split `budget` across the two pools, letting either cover the other.

        Whichever pool is short, the other fills the gap: a project where every
        image has been annotated should still get a full budget, and so should
        one where none of them have.
        """
        if budget <= 0:
            return []

        explore = min(len(untouched), max(1, int(round(budget * EXPLORE_SHARE))))
        exploit = min(len(working), budget - explore)
        explore = min(len(untouched), budget - exploit)

        chosen = untouched[:explore] + working[:exploit]
        return [entry[-1] for entry in chosen]

    # ------------------------------------------------------------------
    # Acquisition: diversity
    # ------------------------------------------------------------------

    @staticmethod
    def image_descriptor(raster):
        """The pooled [C] descriptor for one raster, or None if it has none.

        `Raster.to_dict()` persists the feature map's path, model, stride and
        dimension -- but not the pooled vector, so after a project reload the
        descriptor exists only on disk. It is read back out of the sidecar it
        was already saved in, which costs a few kilobytes of JSON rather than
        the dense [h, w, C] array a full load_feature_map() would pull into
        memory once per image in the project.

        Cached back onto the raster, so a round pays for the read once.
        """
        if raster is None:
            return None

        vector = getattr(raster, 'feature_vector', None)
        if vector is None:
            path = getattr(raster, 'feature_map_path', None)
            if not path:
                return None
            vector = load_feature_vector(path)
            if vector is None:
                return None
            raster.feature_vector = vector

        vector = np.asarray(vector, dtype=np.float32).ravel()
        if vector.size == 0 or not np.isfinite(vector).all():
            return None
        return vector

    def descriptors_for(self, paths):
        """Pooled descriptors for `paths`, keyed by path, at one dimension.

        Descriptors from two different backbones are not comparable, and a
        project can be baked twice. Whichever dimension covers more images wins
        and the rest are dropped: they then rank like an image with no
        descriptor at all, which is a worse position than they deserve but an
        honest one -- the alternative is a distance between two numbers that do
        not mean the same thing.
        """
        found = {}
        raster_manager = self.image_window.raster_manager
        for image_path in paths:
            vector = self.image_descriptor(raster_manager.get_raster(image_path))
            if vector is not None:
                found[image_path] = vector

        if not found:
            return found

        counts = {}
        for vector in found.values():
            counts[vector.size] = counts.get(vector.size, 0) + 1
        dim = max(counts, key=lambda size: (counts[size], size))
        return {path: vector for path, vector in found.items() if vector.size == dim}

    def rank_by_diversity(self, untouched, working, covered_paths, budget):
        """Re-order both pools by descriptor diversity, or leave them as found.

        Diversity is an upgrade to the ranking and never a requirement of it: a
        project that has never been baked must behave exactly as it did before,
        with no dialog, no error, and no minutes silently spent baking features
        nobody asked for. Every early return here is that fallback.

        :param covered_paths: Every annotated image in the project, whether or
                              not this round may predict on it. An image
                              excluded from the budget still covers the part of
                              the project it sits in.
        """
        self.last_acquisition = 'counts'

        candidates = [entry[-1] for entry in untouched] + [entry[-1] for entry in working]
        if budget <= 0 or len(candidates) < DIVERSITY_MIN_VECTORS:
            return untouched, working

        vectors = self.descriptors_for(candidates)
        if len(vectors) < max(DIVERSITY_MIN_VECTORS,
                              DIVERSITY_MIN_SHARE * len(candidates)):
            return untouched, working

        dim = next(iter(vectors.values())).size
        covered = [vector for vector in self.descriptors_for(covered_paths).values()
                   if vector.size == dim]
        covered = np.stack(covered) if covered else None

        self.last_acquisition = 'diversity'
        # The untouched pool is measured against the annotated set, because
        # reaching the parts of the project nobody has covered is the whole
        # point of it. The working pool *is* that set: measuring it against
        # itself would make every distance zero, so it is spread out within
        # itself instead, from its most typical image outwards.
        return (self.diversify(untouched, vectors, covered, budget),
                self.diversify(working, vectors, None, budget))

    @staticmethod
    def diversify(entries, vectors, covered, budget):
        """Re-order one pool by k-center greedy, tier by tier.

        The tiers are the pool's existing first sort key -- images no round has
        predicted on come before ones that have -- and diversity re-orders
        *within* them rather than across them. Both rules are about spending the
        budget somewhere new; where they disagree the cheaper one wins, since an
        image already predicted on this session has a result waiting no matter
        how distinct it looks.

        Entries whose image has no descriptor keep their counting order and go
        behind the ranked ones in their tier. Picks made in an earlier tier join
        the covered set, so a later tier is not ranked as though the budget were
        still unspent.
        """
        if budget <= 0 or not entries:
            return list(entries)

        tiers = {}
        for entry in entries:
            tiers.setdefault(entry[0], []).append(entry)

        ordered = []
        remaining = budget
        for tier in sorted(tiers):
            tier_entries = tiers[tier]
            described = [entry for entry in tier_entries if entry[-1] in vectors]

            if remaining > 0 and len(described) >= DIVERSITY_MIN_VECTORS:
                matrix = np.stack([vectors[entry[-1]] for entry in described])
                picked = Base.kcenter_greedy(matrix, covered, remaining)
                taken = set(picked)
                tier_order = ([described[index] for index in picked]
                              + [entry for index, entry in enumerate(described)
                                 if index not in taken]
                              + [entry for entry in tier_entries if entry[-1] not in vectors])
            else:
                # One descriptor is not a ranking, and neither is a spent
                # budget. The tier keeps the order it arrived in rather than
                # being shuffled by which of its images happen to be baked.
                tier_order = list(tier_entries)

            # Whatever this tier is about to spend covers the part of the
            # project it sits in, ranked or not: a later tier that ignored it
            # would send the rest of the budget straight back to the same place.
            spent = [vectors[entry[-1]] for entry in tier_order[:max(remaining, 0)]
                     if entry[-1] in vectors]
            if spent:
                block = np.stack(spent)
                covered = block if covered is None else np.concatenate([covered, block])

            remaining -= len(tier_order)
            ordered.extend(tier_order)

        return ordered

    @staticmethod
    def kcenter_greedy(vectors, covered, k):
        """Farthest-point traversal over `vectors`, returning row indices.

        The classic diversity acquisition: repeatedly take the row furthest from
        everything covered so far, each pick joining the covered set. O(k x n)
        distances, which at project scale is milliseconds of numpy and needs no
        index -- see the plan's note on why FAISS is not reached for here.

        :param covered: Rows already accounted for, or None. With none, the
                        first pick is the pool's medoid: the most typical image
                        rather than the strangest, which is where a maximum over
                        an empty covered set lands and is usually a blurred
                        frame or a hand in the shot.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        count = len(vectors)
        if k <= 0 or count == 0:
            return []
        k = min(k, count)

        if covered is None or len(covered) == 0:
            centre = vectors.mean(axis=0, keepdims=True)
            first = int(np.argmin(Base.min_square_distance(vectors, centre)))
            running = np.full(count, np.inf, dtype=np.float32)
        else:
            running = Base.min_square_distance(vectors, covered)
            first = int(np.argmax(running))

        picked = [first]
        running = np.minimum(
            running, Base.min_square_distance(vectors, vectors[first:first + 1]))

        while len(picked) < k:
            # Excluded explicitly rather than by trusting a picked row's
            # distance to itself to be zero: two identical images could
            # otherwise take each other's place forever.
            running[picked] = -1.0
            nearest = int(np.argmax(running))
            if running[nearest] < 0.0:
                break
            picked.append(nearest)
            running = np.minimum(
                running, Base.min_square_distance(vectors, vectors[nearest:nearest + 1]))

        return picked

    @staticmethod
    def min_square_distance(vectors, others, chunk=1024):
        """Squared distance from each row of `vectors` to its nearest row of `others`.

        Chunked over `others` and expanded as |a|^2 - 2a.b + |b|^2, so the work
        is one matmul per block rather than the n x m x C broadcast a project of
        ten thousand images cannot hold.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        others = np.asarray(others, dtype=np.float32)
        if others.ndim == 1:
            others = others[None, :]

        best = np.full(len(vectors), np.inf, dtype=np.float32)
        if others.size == 0 or len(vectors) == 0:
            return best

        squared = (vectors ** 2).sum(axis=1)[:, None]
        for start in range(0, len(others), chunk):
            block = others[start:start + chunk]
            distances = squared - 2.0 * (vectors @ block.T) + (block ** 2).sum(axis=1)[None, :]
            best = np.minimum(best, distances.min(axis=1))

        # Floating point can put an identical pair slightly below zero, and a
        # negative distance reads as "already picked" to kcenter_greedy.
        return np.maximum(best, 0.0)

    @property
    def trained_labels(self):
        """Class names of the model in use, in class-index order.

        The best round's, not the newest round's, because that is the model
        predictions actually come from. Reading them off the newest round would
        map class 0 through a label list the loaded weights were never trained
        with, and mislabel everything without erroring.
        """
        if self.best_round:
            return self.best_round.get('labels', [])
        return self.round_history[-1]['labels'] if self.round_history else []

    def after_training(self, weights):
        """Predict with the round's weights, if that was asked for."""
        if self.predict_combo.currentText() != "True":
            return

        model = self.load_trained_model(weights)
        if model is None:
            return

        try:
            self.run_predictions(model)
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
        # Inference is on this thread, so the progress dialog is the only place
        # a long pass can be interrupted from.
        progress_bar.cancel_button.setEnabled(True)
        try:
            for image_path in image_paths:
                if progress_bar.canceled:
                    print("Active Learning: prediction pass cancelled.")
                    break
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

    def run_predictions(self, model, image_paths=None):
        """Predict onto the chosen images, leaving everything unverified.

        Blocking on purpose for now. The recommended model is nano and the
        budget is small, so the wait is short and bounded, and a modal pass is
        far simpler than reconciling predictions that land while the user is
        editing the same image.
        """
        if image_paths is None:
            image_paths = self.candidate_images(self.budget_spinbox.value())
        self.last_predicted_images = list(image_paths)
        self.predicted_ever.update(image_paths)
        self.record_outcome(candidates=len(image_paths),
                            skipped=dict(self.last_skipped),
                            threshold=self.main_window.get_uncertainty_thresh())
        if not image_paths:
            # What acquisition passed over is the whole explanation here.
            self.show_status("Active Learning: no images left to predict on. "
                             + self.skipped_note(self.last_skipped))
            return

        results_processor = self.results_processor()

        # The pass runs on this thread, so the bar will not repaint during it.
        # Setting it first means the bar says what the frozen window is doing
        # rather than still reporting the training that finished a moment ago.
        self.show_status(f"Active Learning: predicting on {len(image_paths)} images...")

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
                self.note_predicted(image_path, len(annotations))
            except Exception as e:
                print(f"Warning: could not add annotations for {image_path}: {e}")

        # The Image Window is pointed at what is now waiting, rather than left
        # showing the whole project with the new work hidden somewhere in it.
        self.apply_needs_review_filter()
        self.annotation_window.load_annotations()
        self.record_outcome(predictions=added, predicted_on=len(image_paths))
        self.show_status(f"Active Learning: {added} predictions added across "
                         f"{len(image_paths)} images, all awaiting review.")

    def show_status(self, message):
        """Say what just happened, as part of the session's status-bar line.

        Folded into the line rather than posted as a message of its own: a
        timed message expires and takes the whole session's state with it, and
        a prediction pass that reported "37 predictions added" and then lost it
        to a mouse-over eight seconds later was reporting nothing at all.
        """
        self.post_status(message)
        if not self._reporting:
            try:
                self.main_window.status_bar.showMessage(message, 8000)
            except Exception:
                pass
