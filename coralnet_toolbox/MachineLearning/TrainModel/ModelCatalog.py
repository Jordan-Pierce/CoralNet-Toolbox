"""The models the Train Model and Active Learning dialogs offer, grouped into families.

Both dialogs show the same two dropdowns, Family then Model, built from here. One
list for both matters: Active Learning trains through the same worker as Train
Model, and a name that exists in one dialog and not the other is a download
failure at the start of a round.
"""
import os
import re

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QComboBox

from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
from coralnet_toolbox.paths import find_weights
from coralnet_toolbox.pyspacer import CORALNET_WEIGHTS_NAME


# ----------------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------------

CLASSIFY_MODELS = [
    'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt',
    'yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolo11m-cls.pt', 'yolo11l-cls.pt', 'yolo11x-cls.pt',
    'yolo26n-cls.pt', 'yolo26s-cls.pt', 'yolo26m-cls.pt', 'yolo26l-cls.pt', 'yolo26x-cls.pt',
]

DETECT_MODELS = [
    'yolov3u.pt', 'yolov3-sppu.pt', 'yolov3-tinyu.pt',
    'yolov5nu.pt', 'yolov5su.pt', 'yolov5mu.pt', 'yolov5lu.pt', 'yolov5xu.pt',
    'yolov5n6u.pt', 'yolov5s6u.pt', 'yolov5m6u.pt', 'yolov5l6u.pt', 'yolov5x6u.pt',
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolov8n-oiv7.pt', 'yolov8s-oiv7.pt', 'yolov8m-oiv7.pt', 'yolov8l-oiv7.pt', 'yolov8x-oiv7.pt',
    'yolov9t.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9c.pt', 'yolov9e.pt',
    'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt',
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
    'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt',
    'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt',
    'rtdetr-l.pt', 'rtdetr-x.pt',
]

SEGMENT_MODELS = [
    'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt',
    'yolov9c-seg.pt', 'yolov9e-seg.pt',
    'yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt',
    'yolo26n-seg.pt', 'yolo26s-seg.pt', 'yolo26m-seg.pt', 'yolo26l-seg.pt', 'yolo26x-seg.pt',
]

SEMANTIC_MODELS = [
    'yolo26n-sem.pt', 'yolo26s-sem.pt', 'yolo26m-sem.pt', 'yolo26l-sem.pt', 'yolo26x-sem.pt',
]

STANDARD_MODELS = {
    'classify': CLASSIFY_MODELS,
    'detect': DETECT_MODELS,
    'segment': SEGMENT_MODELS,
    'semantic': SEMANTIC_MODELS,
}

# Small models by default: cheap to train, and enough to find out whether the
# labels are working before reaching for a larger one.
DEFAULT_MODELS = {
    'classify': 'yolo11n-cls.pt',
    'detect': 'yolo11n.pt',
    'segment': 'yolo11n-seg.pt',
    'semantic': 'yolo26n-sem.pt',
}

# A model's family, by its file name. First match wins, which is why Open Images
# sits ahead of plain YOLOv8.
FAMILIES = (
    (re.compile(r'coralnet', re.I), 'CoralNet'),
    (re.compile(r'^yolov3', re.I), 'YOLOv3'),
    (re.compile(r'^yolov5', re.I), 'YOLOv5'),
    (re.compile(r'^yolov8.*-oiv7', re.I), 'YOLOv8 Open Images'),
    (re.compile(r'^yolov8', re.I), 'YOLOv8'),
    (re.compile(r'^yolov9', re.I), 'YOLOv9'),
    (re.compile(r'^yolov10', re.I), 'YOLOv10'),
    (re.compile(r'^yolo11', re.I), 'YOLO11'),
    (re.compile(r'^yolo12', re.I), 'YOLO12'),
    (re.compile(r'^yolo26', re.I), 'YOLO26'),
    (re.compile(r'^rtdetr', re.I), 'RT-DETR'),
    (re.compile(r'^dinov2', re.I), 'DINOv2'),
)

# The Community configs the dialogs list. The others in Community/cfg ship no
# weights, so they would train from random initialisation; they are left out for
# now rather than deleted. efficientnet_b0.yaml is what pyspacer builds CoralNet's
# backbone from, and a hidden config's file name typed into the Model box still
# trains, since TrainModelWorker.pre_run resolves every config in the folder.
LISTED_COMMUNITY_FAMILIES = ('DINOv2',)

# Within a Community family, smallest first, so a family's first entry is its default.
_SIZE_ORDER = ('small', 'base', 'large', 'giant')

YOUR_MODEL = "Your Model"

FAMILY_TIPS = {
    'CoralNet': "CoralNet's EfficientNet-B0, trained on CoralNet's patches.\n"
                "Trained here as a new classifier on your labels.",
    'YOLOv8 Open Images': "YOLOv8 pretrained on Open Images V7 (600 classes) rather than COCO.",
    'DINOv2': "Meta's DINOv2 as a frozen backbone. It is never trained: only the\n"
              "layers after it learn, and those start untrained.\n"
              "The backbone downloads from Hugging Face the first time it is used.",
    YOUR_MODEL: "A model file you browsed to.",
}

MODEL_TIPS = {
    'dinov2_small.yaml': "DINOv2 ViT-S/14 backbone, 22M parameters, frozen.",
    'dinov2_base.yaml': "DINOv2 ViT-B/14 backbone, 86M parameters, frozen.\nSlower than Small.",
}


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def model_family(name):
    """The family a model name belongs to, or None for one this catalog does not know."""
    base = os.path.basename(str(name or "")).strip()
    for pattern, family in FAMILIES:
        if pattern.search(base):
            return family
    return None


def _size_rank(name):
    lowered = name.lower()
    for rank, size in enumerate(_SIZE_ORDER):
        if size in lowered:
            return rank, lowered
    return len(_SIZE_ORDER), lowered


def community_models(task):
    """The Community config names the dialogs list for a task, smallest first."""
    names = [name for name in get_available_configs(task=task)
             if model_family(name) in LISTED_COMMUNITY_FAMILIES]
    return sorted(names, key=_size_rank)


def coralnet_available(task):
    """CoralNet's weights are not distributed: offered only once a copy is in the weights folder."""
    return task == 'classify' and bool(find_weights(CORALNET_WEIGHTS_NAME))


def model_families(task):
    """{family: [model, ...]} for a task, in the order the Family dropdown shows them."""
    models = list(STANDARD_MODELS.get(task, []))
    if coralnet_available(task):
        models.insert(0, CORALNET_WEIGHTS_NAME)

    families = {}
    for name in models + community_models(task):
        families.setdefault(model_family(name) or 'Other', []).append(name)
    return families


def default_model(task):
    """The model a dialog starts on: CoralNet's when it is there, else a small one."""
    if coralnet_available(task):
        return CORALNET_WEIGHTS_NAME
    return DEFAULT_MODELS.get(task)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------




class ModelPicker(QObject):
    """One Model dropdown, its models grouped under a heading per family.

    A heading is a bold row that cannot be chosen; the arrow keys and the wheel
    skip it, as they skip the divider between families. The box stays editable,
    so any name Ultralytics knows can still be typed in. Each model row carries
    its name as item data, so the text can carry a marker (Active Learning's
    "(no PU)") without that marker ever being passed on; current() reads the data.

    Model's currentTextChanged fires once per change, never for the empty combo
    that rebuilding passes through, so dialogs can keep watching it directly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        # Inline completion would finish "yolo11" as the YOLO11 heading, which is
        # not a model; typed names are taken as typed.
        self.model_combo.setCompleter(None)
        self.model_combo.setMaxVisibleItems(24)
        self._families = {}
        self._mark = None

    def mark(self, model):
        """The text shown after a model's name, or ""."""
        return self._mark(model) if self._mark else ""

    def set_families(self, families, default=None, mark=None):
        """Offer these families, then select `default` (else the first model).

        Args:
            families (dict): {family: [model, ...]}, in display order.
            default (str): The model to start on.
            mark (callable): model -> text shown after its name, or "". A family
                whose models are all marked shows the mark on its heading too.
        """
        self._families = {family: list(models) for family, models in families.items() if models}
        self._mark = mark
        self._rebuild(select=default)

    def select(self, model):
        """Select `model`. False when the list does not offer it."""
        index = self.model_combo.findData(model) if model else -1
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        return index >= 0

    def add_your_model(self, path):
        """Offer a browsed model file under Your Model, at the top of the list, and select it."""
        models = self._families.pop(YOUR_MODEL, [])
        if path in models:
            models.remove(path)
        models.insert(0, path)  # most recent first
        self._families = {YOUR_MODEL: models, **self._families}
        self._rebuild(select=path)

    def current(self):
        """The chosen model name or path. Typed text comes back as it was typed."""
        text = self.model_combo.currentText().strip()
        index = self.model_combo.currentIndex()
        if index >= 0 and self.model_combo.itemText(index) == text:
            data = self.model_combo.itemData(index)
            if data:
                return data
        return text

    def _add_heading(self, family, models):
        combo = self.model_combo
        if combo.count():
            combo.insertSeparator(combo.count())
        marks = {self.mark(model) for model in models}
        combo.addItem(family + (marks.pop() if len(marks) == 1 else ""))
        heading = combo.model().item(combo.count() - 1)
        heading.setFlags(Qt.NoItemFlags)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        tip = FAMILY_TIPS.get(family)
        if tip:
            heading.setToolTip(tip)

    def _rebuild(self, select=None):
        combo = self.model_combo
        before = combo.currentText()

        combo.blockSignals(True)
        combo.clear()
        for family, models in self._families.items():
            self._add_heading(family, models)
            for model in models:
                combo.addItem(model + self.mark(model), model)
                tip = MODEL_TIPS.get(model) or (FAMILY_TIPS['CoralNet'] if model == CORALNET_WEIGHTS_NAME else None)
                if tip:
                    combo.setItemData(combo.count() - 1, tip, Qt.ToolTipRole)
        if not self.select(select):
            first = next((i for i in range(combo.count()) if combo.itemData(i)), -1)
            combo.setCurrentIndex(first)
        combo.blockSignals(False)

        if combo.currentText() != before:
            combo.currentTextChanged.emit(combo.currentText())
