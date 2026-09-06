"""Train detection and segmentation models directly from the open project.

Normally training means exporting a dataset: every image copied into an output
folder and every annotation written out as a .txt label file. For an iterative
workflow that cost is paid again on every round, and it is dominated by the
image copy -- gigabytes of pixels that already exist on disk.

This module trains from the annotation store instead. Images are read where they
already are and labels are handed to Ultralytics in memory, so nothing is copied
and no label files are written.

Two things make that possible:

  * Ultralytics builds its dataset through a module-level class attribute, which
    this toolbox already replaces for weighted sampling (TrainModel/QtBase.py).
    Swapping in a different subclass is an established pattern here, not a new
    one.
  * A YOLODataset that overrides get_img_files() and get_labels() never consults
    the filesystem for annotations at all.

That second point is what forces the in-memory design rather than the more
obvious "write labels to .cache and leave the images alone". Ultralytics locates
labels by string surgery -- img2label_paths() rsplits an image path on
os.sep + 'images' + os.sep and rejoins with 'labels'. A project image at
D:/surveys/2024/dive3/IMG_0042.jpg has no such segment, so the fallback would
have Ultralytics read and write .txt files inside the user's own image folders.
Overriding get_labels() sidesteps the rule entirely.

What still gets written, into .cache: a data.yaml (Ultralytics requires one for
'names' and 'nc') and one empty directory per split, because check_det_dataset
validates that the paths a yaml names actually exist before any dataset is
constructed. Both are removed when training finishes.

See ACTIVE_LEARNING_PLAN.md for the validation work behind these claims, and
tests/active_learning/ for the tests that hold them in place.
"""

import warnings

import os
import shutil
import hashlib
import datetime

import numpy as np
import yaml

import ultralytics.data.build as detection_build
from ultralytics.data.dataset import YOLODataset

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.MachineLearning.WeightedDataset import WeightedInstanceDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

CACHE_BASE = ".cache"
CACHE_SUBDIR = "in_place_training"

# Directory names the generated yaml points at. They exist but stay empty:
# check_det_dataset validates the paths before any dataset is built, while the
# dataset itself never reads from them.
SPLIT_SENTINELS = {
    'train': "__project_train__",
    'val': "__project_val__",
    'test': "__project_test__",
}

# Which annotation types each task can learn from. Detection derives a bounding
# box from any shape; segmentation needs real polygon geometry.
TASK_ANNOTATION_TYPES = {
    'detect': (RectangleAnnotation, PolygonAnnotation, MultiPolygonAnnotation, PatchAnnotation),
    'segment': (PolygonAnnotation, MultiPolygonAnnotation),
}

SUPPORTED_TASKS = tuple(TASK_ANNOTATION_TYPES)


# ----------------------------------------------------------------------------------------------------------------------
# Version guard
# ----------------------------------------------------------------------------------------------------------------------


def check_ultralytics_support():
    """Verify the Ultralytics internals this module reaches into still exist.

    pyproject pins ultralytics with no upper bound, and the approach depends on
    module attributes that are not public API. Failing here, before a dialog
    offers the option, is far better than failing several epochs into a run.

    Returns:
        tuple: (supported: bool, reason: str). reason is empty when supported.
    """
    missing = []

    if not hasattr(detection_build, 'YOLODataset'):
        missing.append("ultralytics.data.build.YOLODataset")

    for name in ('get_img_files', 'get_labels'):
        if not hasattr(YOLODataset, name):
            missing.append(f"YOLODataset.{name}")

    if missing:
        return False, ("This version of Ultralytics no longer exposes "
                       f"{', '.join(missing)}. Export the dataset and train from it instead.")

    return True, ""


# ----------------------------------------------------------------------------------------------------------------------
# Split assignment
# ----------------------------------------------------------------------------------------------------------------------


# The ratios splits are currently derived from. Shared because two places need
# to agree on them: the Train Model dialog, which sets them, and the Image
# Window tooltip, which reports the split an image would land in. A tooltip
# computed from different ratios than training uses would simply be wrong.
#
# Note this is the one thing that does move images between splits: the
# assignment is stable for a fixed ratio, but changing the ratio moves the
# bucket boundaries and so reassigns some images. That is inherent, and the
# reason the dialog does not re-randomize on top of it.
_SPLIT_RATIOS = (0.7, 0.2)


def get_split_ratios():
    """Return the (train, val) ratios splits are currently derived from."""
    return _SPLIT_RATIOS


def set_split_ratios(train_ratio, val_ratio):
    """Record the ratios in use, so every reader agrees on the assignment."""
    global _SPLIT_RATIOS
    _SPLIT_RATIOS = (float(train_ratio), float(val_ratio))


def stable_fraction(image_path):
    """Map an image path to a stable number in [0, 1).

    Deterministic across sessions and machines, which is the whole point:
    Python's hash() is salted per process, so it would reshuffle the splits on
    every launch.
    """
    digest = hashlib.md5(os.path.normcase(str(image_path)).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def assign_split(image_path, train_ratio, val_ratio, override=None):
    """Return 'train', 'val' or 'test' for an image.

    Derived from a hash of the path rather than a shuffle, so the assignment
    survives everything an iterative workflow does to a project:

      * deleting images shrinks a split instead of forcing a rebalance
      * re-importing an image puts it back in the split it was in before
      * repeated training rounds compare like with like, because no image ever
        migrates from train to val between rounds

    That last point is a correctness property, not a convenience. An image that
    moves into validation after being trained on leaks, and quietly inflates
    every metric that follows.

    Args:
        image_path (str): Path used as the hash key.
        train_ratio (float): Fraction assigned to train.
        val_ratio (float): Fraction assigned to validation.
        override (str, optional): A split the user pinned for this image.

    Returns:
        str: 'train', 'val' or 'test'.
    """
    if override in SPLIT_SENTINELS:
        return override

    fraction = stable_fraction(image_path)
    if fraction < train_ratio:
        return 'train'
    if fraction < train_ratio + val_ratio:
        return 'val'
    return 'test'


def group_images_by_split(image_paths, train_ratio, val_ratio, overrides=None):
    """Bucket image paths into {'train': [...], 'val': [...], 'test': [...]}."""
    overrides = overrides or {}
    groups = {split: [] for split in SPLIT_SENTINELS}
    for image_path in image_paths:
        split = assign_split(image_path, train_ratio, val_ratio, overrides.get(image_path))
        groups[split].append(image_path)
    return groups


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------


class InMemoryYOLODataset(YOLODataset):
    """A YOLODataset whose images and labels come from a registry.

    The registry is a class attribute rather than a constructor argument because
    build_yolo_dataset() constructs the dataset itself with a fixed keyword set,
    and model.train() offers no channel to pass anything extra through. Splits
    are told apart by the directory the yaml named, which arrives as img_path.
    """

    # split sentinel name -> list of Ultralytics label dicts
    REGISTRY = {}

    @classmethod
    def register(cls, key, records):
        cls.REGISTRY[key] = records

    @classmethod
    def reset(cls):
        cls.REGISTRY = {}

    @staticmethod
    def split_key(img_path):
        return os.path.basename(os.path.normpath(str(img_path)))

    def _records(self, img_path):
        key = self.split_key(img_path)
        if key not in self.REGISTRY:
            raise KeyError(f"No in-memory records registered for split {key!r}. "
                           f"Known splits: {sorted(self.REGISTRY)}")
        return self.REGISTRY[key]

    def get_img_files(self, img_path):
        """Return image paths from the registry rather than scanning a directory."""
        return [record["im_file"] for record in self._records(img_path)]

    def get_labels(self):
        """Return label dicts from the registry rather than parsing .txt files.

        Copies, deliberately. Ultralytics mutates these in place -- update_labels
        filters classes and the augmentation pipeline rewrites boxes -- so
        handing out the registry's own arrays would let one epoch corrupt the
        next, and the training split corrupt the validation one.
        """
        labels = []
        for record in self._records(self.img_path):
            labels.append({
                "im_file": record["im_file"],
                "shape": record["shape"],
                "cls": record["cls"].copy(),
                "bboxes": record["bboxes"].copy(),
                "segments": [segment.copy() for segment in record["segments"]],
                "keypoints": record["keypoints"],
                "normalized": record["normalized"],
                "bbox_format": record["bbox_format"],
            })
        return labels


class WeightedInMemoryDataset(InMemoryYOLODataset, WeightedInstanceDataset):
    """In-memory labels plus weighted sampling.

    Both features want the same patch slot, so they have to arrive as one class.
    The method resolution order does the work: InMemoryYOLODataset supplies
    get_img_files / get_labels, and WeightedInstanceDataset.__init__ then
    computes its sampling probabilities from the labels it finds already there.
    """


# ----------------------------------------------------------------------------------------------------------------------
# Record building
# ----------------------------------------------------------------------------------------------------------------------


def _parse_floats(text):
    """Parse a whitespace-separated coordinate string, skipping unparseable values."""
    values = []
    for token in text.split():
        try:
            values.append(float(token))
        except ValueError:
            return []
    return values


def _detection_rows(annotation, width, height):
    """Return [(label_code, [cx, cy, w, h])] for one annotation."""
    try:
        label_code, coords = annotation.to_yolo_detection(width, height)
    except Exception as e:
        print(f"Warning: skipping annotation {getattr(annotation, 'id', '?')} for detection: {e}")
        return []

    values = _parse_floats(coords)
    if len(values) != 4:
        return []
    return [(label_code, values)]


def _segmentation_rows(annotation, width, height):
    """Return [(label_code, [x1, y1, ...])] for one annotation.

    MultiPolygonAnnotation contributes one row per member polygon, and returns a
    different shape from the base implementation -- newline-joined lines with the
    label repeated at the front of each, rather than a (code, coords) pair.
    """
    try:
        result = annotation.to_yolo_segmentation(width, height)
    except Exception as e:
        print(f"Warning: skipping annotation {getattr(annotation, 'id', '?')} for segmentation: {e}")
        return []

    rows = []
    if isinstance(result, tuple):
        label_code, coords = result
        values = _parse_floats(coords)
        if len(values) >= 6:  # at least a triangle
            rows.append((label_code, values))
        return rows

    # MultiPolygon: "<code> x1 y1 x2 y2 ..." per line.
    for line in str(result).splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        values = _parse_floats(" ".join(parts[1:]))
        if len(values) >= 6:
            rows.append((parts[0], values))
    return rows


def _bbox_from_polygon(values):
    """Return normalized xywh enclosing a flat [x1, y1, x2, y2, ...] polygon."""
    points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return [
        float((x_min + x_max) / 2.0),
        float((y_min + y_max) / 2.0),
        float(x_max - x_min),
        float(y_max - y_min),
    ]


def build_records(image_paths, annotations_by_image, label_to_index, task, dimensions_for):
    """Build Ultralytics label dicts for one split.

    The geometry conversion is delegated to the annotations' own
    to_yolo_detection / to_yolo_segmentation methods -- the same ones the export
    path uses -- so an in-place dataset and an exported one describe identical
    objects, and any future fix to that conversion reaches both.

    Args:
        image_paths (list): Images in this split, in order.
        annotations_by_image (dict): image path -> list of annotations.
        label_to_index (dict): short_label_code -> class index.
        task (str): 'detect' or 'segment'.
        dimensions_for (callable): image path -> (height, width).

    Returns:
        list: One label dict per image that produced at least one object, in the
            format pinned by tests/active_learning (im_file, shape, cls, bboxes,
            segments, keypoints, normalized, bbox_format).
    """
    allowed_types = TASK_ANNOTATION_TYPES[task]
    records = []

    for image_path in image_paths:
        annotations = annotations_by_image.get(image_path, [])
        if not annotations:
            continue

        try:
            height, width = dimensions_for(image_path)
        except Exception as e:
            print(f"Warning: skipping {image_path}, cannot resolve dimensions: {e}")
            continue

        if not height or not width:
            continue

        classes = []
        bboxes = []
        segments = []

        for annotation in annotations:
            if not isinstance(annotation, allowed_types):
                continue
            if task == 'segment':
                rows = _segmentation_rows(annotation, width, height)
            else:
                rows = _detection_rows(annotation, width, height)

            for label_code, values in rows:
                class_index = label_to_index.get(label_code)
                if class_index is None:
                    continue
                classes.append(class_index)
                if task == 'segment':
                    segments.append(np.asarray(values, dtype=np.float32).reshape(-1, 2))
                    bboxes.append(_bbox_from_polygon(values))
                else:
                    bboxes.append(values)

        if not classes:
            continue

        records.append({
            "im_file": image_path,
            "shape": (int(height), int(width)),
            "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
            "bboxes": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
            "segments": segments,
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
        })

    return records


# ----------------------------------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------------------------------


class InPlaceDataset:
    """The scaffolding one in-place training run needs, and its cleanup.

    Owns three things with a lifetime: the generated yaml and its empty split
    directories, the registry entries, and the patched dataset class. All three
    are undone by remove(), which is safe to call more than once.
    """

    def __init__(self, task, records_by_split, names, cache_root=None):
        """
        Args:
            task (str): 'detect' or 'segment'.
            records_by_split (dict): split name -> label dicts.
            names (list): Class names, ordered by class index.
            cache_root (str, optional): Where the scaffolding is written.
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"In-place training supports {SUPPORTED_TASKS}, not {task!r}")

        self.task = task
        self.records_by_split = {split: records for split, records in records_by_split.items() if records}
        self.names = list(names)
        self.root = None
        self.yaml_path = None
        self._installed = False
        self._original_dataset = None

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = cache_root or os.path.join(os.getcwd(), CACHE_BASE, CACHE_SUBDIR)
        self.root = os.path.join(base, stamp)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def image_count(self, split):
        return len(self.records_by_split.get(split, []))

    def annotation_count(self, split):
        return sum(len(record["cls"]) for record in self.records_by_split.get(split, []))

    def summary(self):
        """One line describing what a run would train on."""
        images = sum(self.image_count(split) for split in SPLIT_SENTINELS)
        annotations = sum(self.annotation_count(split) for split in SPLIT_SENTINELS)
        return (f"{images} images · {annotations} annotations · {len(self.names)} labels · "
                f"Train {self.image_count('train')} / "
                f"Val {self.image_count('val')} / "
                f"Test {self.image_count('test')}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self):
        """Write the yaml and its empty split directories. Returns the yaml path.

        The directories are what check_det_dataset validates; nothing is ever
        read from them, and no images or label files are written anywhere.
        """
        os.makedirs(self.root, exist_ok=True)

        data = {'path': self.root}
        for split in ('train', 'val', 'test'):
            if split not in self.records_by_split:
                continue
            sentinel = SPLIT_SENTINELS[split]
            os.makedirs(os.path.join(self.root, sentinel), exist_ok=True)
            data[split] = sentinel

        # Ultralytics requires a val entry; without one it falls back to train
        # and reports meaningless numbers.
        if 'val' not in data and 'train' in data:
            data['val'] = data['train']

        data['nc'] = len(self.names)
        data['names'] = {index: name for index, name in enumerate(self.names)}

        self.yaml_path = os.path.join(self.root, 'data.yaml')
        with open(self.yaml_path, 'w') as handle:
            yaml.dump(data, handle, default_flow_style=False, sort_keys=False)

        return self.yaml_path

    def install(self, weighted=False):
        """Register the records and swap in the dataset class."""
        InMemoryYOLODataset.reset()
        for split, records in self.records_by_split.items():
            InMemoryYOLODataset.register(SPLIT_SENTINELS[split], records)

        self._original_dataset = detection_build.YOLODataset
        detection_build.YOLODataset = WeightedInMemoryDataset if weighted else InMemoryYOLODataset
        self._installed = True

    def remove(self):
        """Undo install() and delete the scaffolding. Safe to call twice."""
        if self._installed:
            detection_build.YOLODataset = self._original_dataset
            self._original_dataset = None
            self._installed = False

        InMemoryYOLODataset.reset()

        if self.root and os.path.isdir(self.root):
            try:
                shutil.rmtree(self.root, ignore_errors=True)
            except Exception as e:
                print(f"Warning: could not remove in-place training cache {self.root}: {e}")
