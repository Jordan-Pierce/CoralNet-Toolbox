import warnings

import os
import uuid
import yaml
import random
import rasterio
import shutil
import ujson as json

from PyQt5 import sip
from PyQt5.QtCore import Qt, QPointF, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QDialog, QPushButton, QDialogButtonBox,
                             QGridLayout, QScrollArea, QFrame, QCheckBox, QRadioButton,
                             QSpinBox, QHBoxLayout, QWidget, QSizePolicy)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import build_mask_annotation

from coralnet_toolbox.IO.QtImportImages import SUPPORTED_IMAGE_EXTENSIONS
from coralnet_toolbox.MachineLearning.coco_format import (DIALECT_CORALSCOP, annotation_polygons, category_counts,
                                              find_annotation_files, load_annotation_files,
                                              paint_class_mask)

# Semantic masks are single-channel class-ID rasters, so only lossless
# formats are accepted: JPEG compression would rewrite the very integers
# that carry the class.
MASK_EXTENSIONS = ('.png', '.tif', '.tiff')

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon, get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Dataset Discovery
# ----------------------------------------------------------------------------------------------------------------------


def _split_entries(data):
    """Return the raw train/val/test entries from a parsed data.yaml.

    Each key may hold a single path or a list of them, and any of them may be
    absent. Everything is normalized to a flat list of strings.
    """
    entries = []
    for key in ('train', 'val', 'valid', 'test'):
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            entries.extend(str(item) for item in value if item)
        else:
            entries.append(str(value))
    return entries


def _candidate_dirs(entry, root, yaml_dir):
    """Yield the places an image directory named by an entry could actually be.

    A data.yaml is a hint, not a guarantee. Exports from Roboflow and Colab
    routinely carry an absolute path from the machine that produced them
    (/content/datasets/...), which is dead as soon as the folder is copied
    anywhere else. The YAML's own directory is the one location known to be
    real -- the user just picked it in a file dialog -- so every candidate is
    ultimately anchored there.

    Ordering is most-trusted first: the YAML taken at its word, then the entry
    relative to the YAML, then progressively longer tails of the entry
    re-anchored at the YAML's directory, so that a stale
    /content/datasets/coco8/images/train is retried as <yaml_dir>/train, then
    <yaml_dir>/images/train, and so on.
    """
    entry = entry.strip().replace('\\', '/').rstrip('/')
    if not entry:
        return

    if root:
        yield os.path.normpath(os.path.join(root, entry))
    yield os.path.normpath(os.path.join(yaml_dir, entry))

    # Re-anchor the tail. Bounded because a single mis-rooted path never needs
    # more than a handful of segments to line up.
    parts = [part for part in entry.split('/') if part not in ('', '.', '..')]
    for depth in range(1, min(len(parts), 6) + 1):
        yield os.path.normpath(os.path.join(yaml_dir, *parts[-depth:]))


def _resolve_image_dir(entry, root, yaml_dir):
    """Return the first candidate directory that exists, or None.

    A split entry is allowed to name a .txt file listing image paths rather
    than a directory. That form is not supported here, so it resolves to None
    and the caller falls back to scanning the tree.
    """
    seen = set()
    for candidate in _candidate_dirs(entry, root, yaml_dir):
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate
    return None


def _sidecar_dir_for(image_dir, sidecar_name):
    """Map an image directory to the parallel directory holding its sidecars.

    Substitutes the LAST path segment named 'images' with the sidecar's own
    folder name, which is the rule the YOLO loaders themselves use. One rule
    covers both accepted layouts, because in each of them the two directories
    are siblings at the same depth:

        split-first    train/images  ->  train/labels,  train/masks
        images-first   images/train  ->  labels/train,  masks/train
    """
    parts = os.path.normpath(image_dir).replace('\\', '/').split('/')
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == 'images':
            parts[index] = sidecar_name
            return os.path.normpath('/'.join(parts))
    return None


def _normalize_image_dir(resolved):
    """Descend into an 'images' subfolder when a split entry names the split root.

    This toolbox's detection and instance-segmentation exports write
    ``train: train`` rather than ``train: train/images``, so the entry resolves
    to the split folder itself. Walking that folder still finds the images, but
    _sidecar_dir_for works by substituting a path segment named 'images', and
    with none present it returns None -- leaving every image unpaired. Stepping
    down to the folder the images actually live in restores the pairing.
    """
    if os.path.basename(os.path.normpath(resolved)).lower() != 'images':
        nested = os.path.join(resolved, 'images')
        if os.path.isdir(nested):
            return nested
    return resolved


def _is_excluded(path, exclude_dirs):
    """Return True if a path lies inside one of the excluded directories."""
    if not exclude_dirs:
        return False
    resolved = os.path.normcase(os.path.abspath(path))
    for excluded in exclude_dirs:
        excluded = os.path.normcase(os.path.abspath(excluded))
        if resolved == excluded or resolved.startswith(excluded + os.sep):
            return True
    return False


def _iter_images(image_dir, exclude_dirs):
    """Yield every supported image file under a directory, recursively."""
    for current_root, dir_names, file_names in os.walk(image_dir):
        if _is_excluded(current_root, exclude_dirs):
            dir_names[:] = []
            continue
        for file_name in file_names:
            if os.path.splitext(file_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield os.path.join(current_root, file_name)


def _pair(image_path, image_dir, sidecar_dir, sidecar_exts):
    """Return the sidecar file mirroring an image, or None if there is none.

    Pairing is by path relative to the image directory rather than by bare
    basename, so an img1.jpg present in several splits keeps its own sidecar
    instead of colliding with its namesakes. Only the stem has to match -- the
    mask for img1.jpg is img1.png -- so each accepted extension is tried in
    turn.
    """
    if not sidecar_dir:
        return None
    relative = os.path.relpath(image_path, image_dir)
    stem = os.path.join(sidecar_dir, os.path.splitext(relative)[0])
    for extension in sidecar_exts:
        candidate = stem + extension
        if os.path.isfile(candidate):
            return candidate
    return None


def _annotation_key(raw_ann_data):
    """Return a hashable identity for an annotation about to be imported.

    Keying on the *imported* geometry rather than the source line, because that
    is what would end up duplicated in the project: two polygons that differ
    only in vertex count still land as one identical rectangle when the import
    format is 'rectangle'.

    Coordinates are rounded to 1e-4 px so that label files writing the same box
    with different precision (0.5 vs 0.50000001) collapse together; anything
    coarser would start merging genuinely distinct small annotations.
    """
    kind = raw_ann_data["type"]
    if kind == "RectangleAnnotation":
        geometry = (round(raw_ann_data["top_left"][0], 4),
                    round(raw_ann_data["top_left"][1], 4),
                    round(raw_ann_data["bottom_right"][0], 4),
                    round(raw_ann_data["bottom_right"][1], 4))
    else:
        geometry = tuple((round(x, 4), round(y, 4)) for x, y in raw_ann_data["points"])
    return (kind, raw_ann_data["class_name"], geometry)


def _message_with_details(parent, icon, title, text, details):
    """Show a message box with a long list behind its Show Details button.

    QMessageBox's static helpers take no details text, and passing one raises
    TypeError - which, inside a slot, loses the message altogether.
    """
    box = QMessageBox(icon, title, text, QMessageBox.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.exec_()


def _size_problem(expected, actual):
    """Describe a mismatch between the size a COCO file gives an image and its real size.

    COCO coordinates are pixels, not fractions, so annotations on an image of
    another size land in the wrong place and are better skipped than imported.
    A swap is called out on its own: CoralNet-Toolbox reads pixels as stored and ignores
    EXIF orientation, so an image another tool displayed rotated - and was
    annotated that way - arrives with its width and height exchanged.

    Args:
        expected (tuple): (height, width) from the file; either may be None
            when the file leaves it out, and then nothing is checked.
        actual (tuple): (height, width) of the image.

    Returns:
        str: The problem, or None when the sizes agree.
    """
    if None in expected or tuple(expected) == tuple(actual):
        return None

    message = (f"the annotation file says it is {expected[1]}x{expected[0]}, but the image is "
               f"{actual[1]}x{actual[0]}, so its annotations were not imported.")
    if tuple(expected) == tuple(actual)[::-1]:
        message += (" The tool that made the file probably rotated the image by its EXIF "
                    "orientation, which CoralNet-Toolbox does not apply.")
    return message


def discover_dataset_files(yaml_path, image_import_policy='annotated_only', exclude_dirs=(),
                           sidecar_kind='labels'):
    """Find every image in a YOLO dataset and the sidecar file that goes with it.

    Both accepted YOLO layouts are supported, since the substitution that
    locates the sidecar directory is the same in either:

        dataset/train/images/img1.jpg     dataset/images/train/img1.jpg
        dataset/train/labels/img1.txt     dataset/labels/train/img1.txt

    Detection and instance segmentation keep their annotations in .txt label
    files; semantic segmentation keeps them in single-channel mask rasters,
    under the folder the YAML names in masks_dir.

    The data.yaml's own split entries are consulted first because they state
    the layout outright; a tree scan is the fallback for when those entries are
    missing or point somewhere that no longer exists.

    Args:
        yaml_path (str): Path to the dataset's data.yaml.
        image_import_policy (str): 'all' keeps images with no sidecar file,
            'annotated_only' keeps only images that have one.
        exclude_dirs (iterable): Directories to skip. The import's own output
            folder is passed here -- it defaults to a subdirectory of the
            dataset, so without this a second import would re-ingest the copies
            the first one made.
        sidecar_kind (str): 'labels' for the .txt annotations used by detection
            and instance segmentation, 'masks' for the class-ID rasters used by
            semantic segmentation.

    Returns:
        dict: image path -> sidecar path, or None when the image has none.
    """
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    exclude_dirs = list(exclude_dirs or ())

    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    # A relative 'path' is itself relative to the YAML, and an absolute one is
    # only usable if it survived the trip to this machine.
    root = str(data.get('path') or '').strip()
    if root:
        root = root.replace('\\', '/')
        if not os.path.isabs(root):
            root = os.path.join(yaml_dir, root)
        if not os.path.isdir(root):
            root = None
    else:
        root = None

    # Semantic datasets name their mask folder in the YAML; 'masks' is the
    # conventional default, and only the folder name is used -- the rest of the
    # path comes from the image directory it sits beside.
    if sidecar_kind == 'masks':
        sidecar_name = str(data.get('masks_dir') or 'masks').strip()
        sidecar_name = os.path.basename(sidecar_name.replace(chr(92), '/').rstrip('/')) or 'masks'
        sidecar_exts = MASK_EXTENSIONS
    else:
        sidecar_name = 'labels'
        sidecar_exts = ('.txt',)

    image_dirs = []
    for entry in _split_entries(data):
        resolved = _resolve_image_dir(entry, root, yaml_dir)
        if not resolved:
            continue
        resolved = _normalize_image_dir(resolved)
        if resolved not in image_dirs and not _is_excluded(resolved, exclude_dirs):
            image_dirs.append(resolved)

    # Fallback: no usable split entries, so scan for directories named 'images'.
    # Covers a YAML with no split keys at all, and one whose every entry points
    # at a location that does not exist here.
    if not image_dirs:
        for current_root, dir_names, _file_names in os.walk(yaml_dir):
            if _is_excluded(current_root, exclude_dirs):
                dir_names[:] = []
                continue
            for dir_name in dir_names:
                if dir_name.lower() == 'images':
                    candidate = os.path.join(current_root, dir_name)
                    if not _is_excluded(candidate, exclude_dirs):
                        image_dirs.append(candidate)

    source_map = {}
    for image_dir in image_dirs:
        sidecar_dir = _sidecar_dir_for(image_dir, sidecar_name)
        for image_path in _iter_images(image_dir, exclude_dirs):
            # Forward slashes, because a raster is looked up by the exact string
            # it was registered under and the rest of the application settled on
            # that form: Qt's file dialogs hand back forward slashes on Windows,
            # and ResultsProcessor writes `results.path.replace("\\", "/")` onto
            # every annotation it creates. os.path.normpath alone gives
            # backslashes here, so an in-place import registered
            # `C:\data\images\a.jpg` while every prediction landing on that
            # image claimed `C:/data/images/a.jpg` -- two keys for one file, and
            # `raster_manager.get_raster()` finding neither.
            #
            # The copying branch already normalized its destinations this way
            # (see _copy_files_with_progress), which is why only in-place
            # imports were affected.
            normalized = os.path.normpath(image_path).replace("\\", "/")
            # An image reachable through two entries keeps the first sidecar found.
            if normalized in source_map and source_map[normalized]:
                continue
            sidecar_path = _pair(image_path, image_dir, sidecar_dir, sidecar_exts)
            if sidecar_path is None and image_import_policy != 'all':
                continue
            source_map[normalized] = sidecar_path

    return source_map

# ----------------------------------------------------------------------------------------------------------------------
# Worker Class for Threading
# ----------------------------------------------------------------------------------------------------------------------


class DatasetProcessor(QObject):
    """
    Worker object to process a dataset in a separate thread.
    It is completely decoupled from the GUI.
    """
    status_changed = pyqtSignal(str, int)
    progress_updated = pyqtSignal(int)
    processing_complete = pyqtSignal(list, list, list, int)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, yaml_path, output_folder, task, import_as, rename_on_conflict=False,
                 excluded_classes=None, image_import_policy='annotated_only', copy_images=True,
                 sample_size=None, coco_dataset=None, parent=None):
        super().__init__(parent)
        self.yaml_path = yaml_path
        # A COCO dataset the dialog already read (see coco_format), or None for
        # a YOLO import, which reads yaml_path instead.
        self.coco_dataset = coco_dataset
        # None when importing in place, where there is no destination at all.
        self.output_folder = output_folder
        self.copy_images = copy_images
        self.task = task  # 'detect' or 'segment' (source format)
        self.import_as = import_as  # 'rectangle' or 'polygon' (target format)
        self.rename_on_conflict = rename_on_conflict
        self.excluded_classes = excluded_classes if excluded_classes is not None else set()
        self.image_import_policy = image_import_policy
        # None imports everything found; an int keeps that many images at random.
        self.sample_size = sample_size
        self.is_running = True
        self.parsing_errors = []  # To collect errors instead of printing
        # Exact-duplicate label lines dropped while parsing, reported at the end.
        self.duplicates_removed = 0

    def stop(self):
        self.is_running = False

    # In class DatasetProcessor, update this method:

    def run(self):
        """Main processing method executed in the thread."""
        try:
            # Step 1: Discover files - through the YAML for YOLO, from the
            # annotation file the dialog already read for COCO
            if self.coco_dataset is not None:
                class_names = None
                source_image_label_map = self._find_coco_source_files()
            else:
                with open(self.yaml_path, 'r') as file:
                    data = yaml.safe_load(file)
                class_names = data.get('names', [])
                source_image_label_map = self._find_source_files()

            if not source_image_label_map:
                if self.coco_dataset is not None and not any(image['path'] for image in self.coco_dataset['images']):
                    self.error.emit("None of the annotation file's images were found on disk. "
                                    "Pick the folder that holds them under Images.")
                elif self.image_import_policy == 'annotated_only':
                    self.error.emit("No valid image/label pairs found in the dataset.")
                else:
                    self.error.emit("No images found in the dataset.")
                return

            # --- Step 2: Copy files with progress reporting ---
            if self.copy_images:
                self.status_changed.emit("Copying image files...", len(source_image_label_map))
                image_label_paths = self._copy_files_with_progress(source_image_label_map)
            else:
                # Importing in place: the source paths are the ones the project
                # will reference, so there is nothing to materialize.
                image_label_paths = source_image_label_map

            if not self.is_running:
                return

            # Step 3: Parse label files and create annotation data
            self.status_changed.emit("Importing annotations...", len(image_label_paths))
            raw_annotations = self._create_raw_annotations(image_label_paths, class_names)

            if not self.is_running:
                return

            # Step 4 (REMOVED): The JSON export is no longer done here.

            # Step 5: Emit results for GUI to consume
            image_paths = list(image_label_paths.keys())
            self.processing_complete.emit(raw_annotations,
                                          image_paths,
                                          self.parsing_errors,
                                          self.duplicates_removed)

        except Exception as e:
            self.error.emit(f"An error occurred during processing: {str(e)}")
        finally:
            self.finished.emit()

    def _find_source_files(self):
        """Finds all source image and sidecar paths based on the import policy.

        A sample is drawn after discovery rather than during it, so the images
        are picked from the set the import rule already accepted: asking for 50
        images with 'only images with annotations' selected gives 50 annotated
        images, not 50 images of which a few happen to have labels.

        Each image keeps whatever sidecar it was paired with, so the annotations
        follow their image automatically.

        'images_only' discovers like 'all' and then drops every sidecar, so the
        later stages see no labels or masks at all: nothing is parsed, no mask
        is copied, and no annotation is created.
        """
        sidecar_kind = 'masks' if self.task == 'semantic' else 'labels'
        # Only a copying import has an output folder to keep out of its own scan.
        exclude_dirs = [self.output_folder] if self.output_folder else []
        images_only = self.image_import_policy == 'images_only'
        source_map = discover_dataset_files(self.yaml_path,
                                            image_import_policy='all' if images_only else self.image_import_policy,
                                            exclude_dirs=exclude_dirs,
                                            sidecar_kind=sidecar_kind)
        if images_only:
            source_map = dict.fromkeys(source_map)

        if self.sample_size and self.sample_size < len(source_map):
            # Sorted first so the draw depends only on the seed and not on the
            # order the filesystem happened to hand the directories back.
            chosen = random.sample(sorted(source_map), self.sample_size)
            source_map = {path: source_map[path] for path in chosen}

        return source_map

    def _find_coco_source_files(self):
        """Map each COCO image found on disk to its annotations.

        The same rules as _find_source_files, applied to an annotation file
        instead of a folder tree: the import rule picks the images, the sample
        is drawn from those, and each image carries its annotations along. An
        image counts as annotated by the checked classes only, so one whose
        annotations are all of unchecked classes is left out of an 'only images
        with annotations' import rather than arriving empty.

        Each value is {'image': the image entry, 'annotations': [...]}, or None
        for an images-only import. Images that were not found are skipped
        here; the dialog reported them when it read the file.
        """
        names = {category['id']: category['name'] for category in self.coco_dataset['categories']}
        annotations_by_image = {}
        for annotation in self.coco_dataset['annotations']:
            if names[annotation['category_id']] not in self.excluded_classes:
                annotations_by_image.setdefault(annotation['image_id'], []).append(annotation)

        source_map = {}
        for image in self.coco_dataset['images']:
            if image['path'] is None:
                continue
            annotations = annotations_by_image.get(image['id'], [])
            if self.image_import_policy == 'annotated_only' and not annotations:
                continue
            if self.image_import_policy == 'images_only':
                source_map[image['path']] = None
            else:
                source_map[image['path']] = {'image': image, 'annotations': annotations}

        if self.sample_size and self.sample_size < len(source_map):
            chosen = random.sample(sorted(source_map), self.sample_size)
            source_map = {path: source_map[path] for path in chosen}

        return source_map

    def _copy_files_with_progress(self, source_image_label_map):
        """Copies files and reports progress for each file.

        Semantic masks are copied too, and are renamed in lockstep with their
        image so the pair stays together in the flattened output.
        """
        img_out_dir = os.path.join(self.output_folder, "images")
        os.makedirs(img_out_dir, exist_ok=True)

        # A COCO import has no mask files: its masks are in the annotation file
        copy_masks = self.task == 'semantic' and self.coco_dataset is None
        mask_out_dir = os.path.join(self.output_folder, "masks")
        if copy_masks:
            os.makedirs(mask_out_dir, exist_ok=True)

        dest_label_map = {}
        for i, (src_image_path, label_path) in enumerate(source_image_label_map.items()):
            if not self.is_running:
                break

            original_img_basename = os.path.basename(src_image_path)

            if self.rename_on_conflict:
                base, ext = os.path.splitext(original_img_basename)
                unique_id = str(uuid.uuid4())[:8]
                new_img_basename = f"{base}_{unique_id}{ext}"
            else:
                new_img_basename = original_img_basename

            dest_image_path = os.path.join(img_out_dir, new_img_basename)
            shutil.copy(src_image_path, dest_image_path)

            if copy_masks and label_path:
                # The mask takes the image's (possibly renamed) stem and keeps
                # its own extension, which is how it is found again on load.
                mask_ext = os.path.splitext(label_path)[1]
                mask_stem = os.path.splitext(new_img_basename)[0]
                dest_mask_path = os.path.join(mask_out_dir, mask_stem + mask_ext)
                shutil.copy(label_path, dest_mask_path)
                label_path = dest_mask_path

            dest_label_map[dest_image_path.replace("\\", "/")] = label_path
            self.progress_updated.emit(i + 1)

        return dest_label_map

    def _create_raw_annotations(self, image_label_paths, class_names):
        """
        Parses label files, converts format if needed, and creates raw annotation data.
        Returns a list of annotation dictionaries.

        Exact duplicates are dropped per image: a label file that repeats the
        same class and geometry produces one annotation, not several stacked on
        top of each other. The count of what was dropped is kept in
        self.duplicates_removed. Scoping the check to a single image is what
        makes it safe -- the same box on two different images is two real
        annotations.
        """
        if self.coco_dataset is not None:
            return self._create_raw_coco_annotations(image_label_paths)
        if self.task == 'semantic':
            return self._create_raw_mask_records(image_label_paths)

        all_raw_annotations = []
        for i, (image_path, label_path) in enumerate(image_label_paths.items()):
            if not self.is_running:
                break

            # If there's no label file for this image (e.g., 'import all' policy), skip to progress update
            if not label_path:
                self.progress_updated.emit(i + 1)
                continue

            # Opened directly rather than through rasterio_open: that helper
            # caches the dataset and never closes it, which would leave one
            # GDAL handle open per image, and pops a QMessageBox on failure,
            # which must not happen off the GUI thread.
            with rasterio.open(image_path) as src:
                image_height, image_width = src.shape
            with open(label_path, 'r') as file:
                lines = file.readlines()

            seen_keys = set()
            for line_num, line in enumerate(lines):
                try:
                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])

                    class_name = class_names[class_id]
                    if class_name in self.excluded_classes:
                        continue  # Skip this annotation if its class was unchecked

                    raw_ann_data = {"image_path": image_path, "class_name": class_name}

                    parsed_data = {}
                    if self.task == 'detect':  # Source is bbox: class, x_c, y_c, w, h
                        _, x_c, y_c, w, h = parts
                        x = x_c * image_width
                        y = y_c * image_height
                        width = w * image_width
                        height = h * image_height
                        parsed_data['top_left'] = (x - width / 2, y - height / 2)
                        parsed_data['bottom_right'] = (x + width / 2, y + height / 2)
                    else:  # Source is polygon: class, x1, y1, x2, y2, ...
                        points_norm = parts[1:]
                        # Convert normalized coordinates to pixel coordinates
                        # Extract x and y coordinates from the flattened list
                        x_coords = points_norm[::2]  # Every even index (0, 2, 4...)
                        y_coords = points_norm[1::2]  # Every odd index (1, 3, 5...)
                        
                        # Scale coordinates by image dimensions
                        points = []
                        for x, y in zip(x_coords, y_coords):
                            pixel_x = x * image_width
                            pixel_y = y * image_height
                            points.append((pixel_x, pixel_y))
                        parsed_data['points'] = points

                    if self.import_as == 'rectangle':
                        raw_ann_data["type"] = "RectangleAnnotation"
                        if 'top_left' in parsed_data:
                            raw_ann_data.update(parsed_data)
                        else:
                            points = parsed_data['points']
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            raw_ann_data["top_left"] = (min(x_coords), min(y_coords))
                            raw_ann_data["bottom_right"] = (max(x_coords), max(y_coords))
                    elif self.import_as == 'polygon':
                        raw_ann_data["type"] = "PolygonAnnotation"
                        if 'points' in parsed_data:
                            raw_ann_data.update(parsed_data)
                        else:
                            tl, br = parsed_data['top_left'], parsed_data['bottom_right']
                            raw_ann_data["points"] = [(tl[0], tl[1]), (br[0], tl[1]), (br[0], br[1]), (tl[0], br[1])]

                    key = _annotation_key(raw_ann_data)
                    if key in seen_keys:
                        self.duplicates_removed += 1
                        continue
                    seen_keys.add(key)

                    all_raw_annotations.append(raw_ann_data)
                except (ValueError, IndexError) as e:
                    error_msg = (f"In file '{os.path.basename(label_path)}' on line {line_num + 1}:\n"
                                 f"Skipped malformed content: '{line.strip()}'\nReason: {e}\n")
                    self.parsing_errors.append(error_msg)

            self.progress_updated.emit(i + 1)
        return all_raw_annotations

    def _create_raw_mask_records(self, image_mask_paths):
        """Pair each copied image with its mask for the GUI thread to load.

        The masks themselves are deliberately not decoded here. A
        MaskAnnotation allocates a QImage canvas, which has to happen on the
        GUI thread, so this stage only reports which file belongs to which
        image and leaves the reading to on_processing_complete.
        """
        records = []
        for i, (image_path, mask_path) in enumerate(image_mask_paths.items()):
            if not self.is_running:
                break

            if mask_path:
                records.append({"type": "MaskAnnotation",
                                "image_path": image_path,
                                "mask_path": mask_path})

            self.progress_updated.emit(i + 1)
        return records

    def _create_raw_coco_annotations(self, image_records):
        """Turn COCO annotations into the same raw records a YOLO import produces.

        A semantic import gets one record per image carrying that image's
        annotations, and the mask is painted on the GUI thread alongside the
        MaskAnnotation it becomes, for the same reason a YOLO mask file is read
        there. Painting here would hold every image's full-size mask in memory
        until the import finished.
        """
        names = {category['id']: category['name'] for category in self.coco_dataset['categories']}

        records = []
        for i, (image_path, record) in enumerate(image_records.items()):
            if not self.is_running:
                break

            if record and record['annotations']:
                image = record['image']
                if self.task == 'semantic':
                    records.append({"type": "MaskAnnotation",
                                    "image_path": image_path,
                                    "annotations": record['annotations'],
                                    "size": (image['height'], image['width'])})
                else:
                    records.extend(self._coco_image_annotations(image_path, image, record['annotations'], names))

            self.progress_updated.emit(i + 1)
        return records

    def _coco_image_annotations(self, image_path, image, annotations, names):
        """Raw annotation records for one image's COCO annotations.

        Coordinates are already pixels, so nothing is scaled; that is also why
        an image whose size disagrees with the file has its annotations skipped
        rather than imported in the wrong place. Duplicates are dropped as
        _create_raw_annotations drops them.
        """
        image_name = os.path.basename(image_path)

        if image['height'] and image['width']:
            # Opened directly for the reasons given in _create_raw_annotations
            with rasterio.open(image_path) as src:
                problem = _size_problem((image['height'], image['width']), src.shape)
            if problem:
                self.parsing_errors.append(f"'{image_name}': {problem}")
                return []

        raw_annotations = []
        seen_keys = set()
        for annotation in annotations:
            class_name = names[annotation['category_id']]
            try:
                if self.import_as == 'rectangle':
                    x, y, w, h = annotation['bbox']
                    shapes = [{"type": "RectangleAnnotation",
                               "top_left": (x, y),
                               "bottom_right": (x + w, y + h)}]
                else:
                    # A mask is traced, and the tracing has simplified it
                    traced = isinstance(annotation['segmentation'], dict)
                    shapes = [{"type": "PolygonAnnotation", "points": exterior, "holes": holes, "traced": traced}
                              for exterior, holes in annotation_polygons(annotation)]
            except Exception as e:
                self.parsing_errors.append(f"'{image_name}': skipped a '{class_name}' annotation: {e}")
                continue

            for shape in shapes:
                shape.update(image_path=image_path, class_name=class_name)
                key = _annotation_key(shape)
                if key in seen_keys:
                    self.duplicates_removed += 1
                    continue
                seen_keys.add(key)
                raw_annotations.append(shape)

        return raw_annotations


# ----------------------------------------------------------------------------------------------------------------------
# Dialog Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    def __init__(self, main_window, parent=None):
        super(Base, self).__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Import Dataset")

        self.task = None
        self.progress_bar = None
        self.thread = None
        self.worker = None
        self.output_folder = None
        self.sample_size = None
        self.images_only = False
        self.class_checkboxes = []
        # The value each checkbox's class has in the dataset: the YOLO class
        # index, or the COCO category id. Parallel to class_checkboxes.
        self.class_values = []
        # Read when a COCO file or folder is picked, and handed to the worker
        self.coco_dataset = None
        self.coco_problems = []

        # Two columns over a full-width button box. self.layout is the left
        # column, which is where the subclasses' info and options groups land;
        # the right column holds the image rules and the class list.
        root_layout = QVBoxLayout(self)
        columns_layout = QHBoxLayout()
        root_layout.addLayout(columns_layout)

        left_widget = QWidget()
        left_widget.setMinimumWidth(480)
        self.layout = QVBoxLayout(left_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.addWidget(left_widget, 3)

        right_widget = QWidget()
        right_widget.setMinimumWidth(320)
        self.right_layout = QVBoxLayout(right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.addWidget(right_widget, 2)

        self.setup_info_layout()
        self.setup_options_layout()
        self.setup_dataset_layout()
        self.setup_output_layout()
        self.layout.addStretch()
        self.setup_import_rules_layout()
        self.setup_buttons_layout(root_layout)

        # The left column sets the height: the class list ignores its own size
        # hint, so a long list scrolls instead of stretching the dialog.
        self.resize(1200, self.sizeHint().height())

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        raise NotImplementedError("Subclasses must implement method.")

    def setup_options_layout(self):
        """
        Set up the layout and widgets for the task's import options.
        """
        raise NotImplementedError("Subclasses must implement method.")

    def setup_dataset_layout(self):
        """Set up the layout for choosing the dataset: a YOLO data.yaml, or COCO annotations."""
        group_box = QGroupBox("Dataset")
        layout = QGridLayout()
        layout.addWidget(QLabel("File:"), 0, 0)
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setPlaceholderText("Select a YOLO data.yaml, or a COCO .json file or folder...")
        self.dataset_path_edit.setToolTip("YOLO: the dataset's data.yaml, which names its classes and splits.\n"
                                          "COCO: an annotation .json, or a folder of them for tools that write\n"
                                          "one file per image.")
        layout.addWidget(self.dataset_path_edit, 0, 1)
        self.browse_dataset_button = QPushButton("File...")
        self.browse_dataset_button.clicked.connect(self.browse_dataset_file)
        self.browse_dataset_button.setToolTip("Browse for a data.yaml, or a COCO .json file.")
        layout.addWidget(self.browse_dataset_button, 0, 2)
        self.browse_dataset_folder_button = QPushButton("Folder...")
        self.browse_dataset_folder_button.clicked.connect(self.browse_dataset_folder)
        self.browse_dataset_folder_button.setToolTip("Browse for a folder of COCO .json files, read as one dataset.")
        layout.addWidget(self.browse_dataset_folder_button, 0, 3)

        # A COCO file does not say where its images are. They are looked for
        # beside it first; this row is for when they are somewhere else.
        self.images_row_label = QLabel("Images:")
        layout.addWidget(self.images_row_label, 1, 0)
        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setReadOnly(True)
        self.images_dir_edit.setPlaceholderText("Found automatically beside the annotation file")
        self.images_dir_edit.setToolTip("Folder holding the dataset's images, when they are not beside the\n"
                                        "annotation file or in an images/ or data/ folder next to it.")
        layout.addWidget(self.images_dir_edit, 1, 1)
        self.browse_images_button = QPushButton("Browse")
        self.browse_images_button.clicked.connect(self.browse_images_dir)
        self.browse_images_button.setToolTip("Browse for the folder holding the dataset's images.")
        layout.addWidget(self.browse_images_button, 1, 2, 1, 2)
        self.set_images_row_visible(False)

        # One line describing what was actually found, so the size and label
        # coverage of a dataset are known before committing to the import.
        self.dataset_summary_label = QLabel("No dataset selected.")
        self.dataset_summary_label.setWordWrap(True)
        self.dataset_summary_label.setToolTip("What was found in the selected dataset.")
        layout.addWidget(self.dataset_summary_label, 2, 0, 1, 4)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def set_images_row_visible(self, visible):
        """Show the Images row, which only a COCO dataset uses."""
        for widget in (self.images_row_label, self.images_dir_edit, self.browse_images_button):
            widget.setVisible(visible)

    def setup_output_layout(self):
        """Set up the layout for output directory and the advanced options accordion."""
        group_box = QGroupBox("Output Settings")
        layout = QGridLayout()

        # Copying makes the project self-contained; importing in place makes it
        # cheap to look at a dataset without duplicating it on disk first.
        self.copy_images_radio = QRadioButton("Copy images into a project folder")
        self.copy_images_radio.setChecked(True)
        self.copy_images_radio.setToolTip("Copy every image (and mask) into the output directory.\n"
                                          "The project is self-contained and can be moved or shared.")
        self.import_in_place_radio = QRadioButton("Import in place (no copy)")
        self.import_in_place_radio.setToolTip("Read the images where they already are; nothing is written to disk.\n"
                                              "Much faster for a large dataset, but a saved project points at the\n"
                                              "original dataset folder and breaks if it is moved or deleted.")
        self.copy_images_radio.toggled.connect(self.update_output_widgets)
        layout.addWidget(self.copy_images_radio, 0, 0, 1, 3)
        layout.addWidget(self.import_in_place_radio, 1, 0, 1, 3)

        self.output_dir_row_label = QLabel("Directory:")
        layout.addWidget(self.output_dir_row_label, 2, 0)
        self.output_dir_label = QLineEdit()
        self.output_dir_label.setPlaceholderText("Select output directory...")
        self.output_dir_label.setToolTip("Directory where the imported dataset will be saved.\nDefault: same directory as the dataset file.")
        layout.addWidget(self.output_dir_label, 2, 1)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_dir)
        self.browse_output_button.setToolTip("Browse for an output directory.")
        layout.addWidget(self.browse_output_button, 2, 2)
        self.output_name_row_label = QLabel("Folder Name:")
        layout.addWidget(self.output_name_row_label, 3, 0)
        self.output_folder_name = QLineEdit("data")
        self.output_folder_name.setPlaceholderText("data")
        self.output_folder_name.setToolTip("Name for the subdirectory containing imported images.\nFinal location: Directory / Folder Name /")
        layout.addWidget(self.output_folder_name, 3, 1, 1, 2)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_import_rules_layout(self):
        """Set up the right column: which images to import, and which classes."""
        image_rule_box = QGroupBox("Image Import Rule")
        image_rule_layout = QVBoxLayout()
        self.import_annotated_images_radio = QRadioButton("Import only images with annotations")
        self.import_annotated_images_radio.setToolTip("Only import images that have corresponding annotation label files.\nSkips images without labels.")
        self.import_all_images_radio = QRadioButton("Import all images found in dataset")
        self.import_all_images_radio.setToolTip("Import all images found in the dataset, with or without annotations.\nUseful for datasets with partially-labeled images.")
        self.import_images_only_radio = QRadioButton("Import images only, without annotations")
        self.import_images_only_radio.setToolTip("Import all images found in the dataset and ignore their label files or masks.\n"
                                                 "Useful for annotating from scratch or testing a model on unseen images.")
        self.import_images_only_radio.toggled.connect(self.update_annotation_widgets)
        self.import_annotated_images_radio.setChecked(True)
        image_rule_layout.addWidget(self.import_annotated_images_radio)
        image_rule_layout.addWidget(self.import_all_images_radio)
        image_rule_layout.addWidget(self.import_images_only_radio)

        # Sampling is applied on top of whichever rule is selected above, so the
        # subset is drawn from the images that rule already accepted.
        sample_row = QHBoxLayout()
        self.sample_subset_checkbox = QCheckBox("Import a random subset of")
        self.sample_subset_checkbox.setToolTip("Import a random selection of the images instead of all of them.\n"
                                               "Each image brings its own annotations.\n"
                                               "Useful for trying out a large dataset without copying every file.")
        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setRange(1, 1000000)
        self.sample_size_spinbox.setValue(100)
        self.sample_size_spinbox.setEnabled(False)
        self.sample_size_spinbox.setToolTip("How many images to import.\n"
                                            "If the dataset has fewer than this, all of them are imported.")
        self.sample_subset_checkbox.toggled.connect(self.sample_size_spinbox.setEnabled)
        sample_row.addWidget(self.sample_subset_checkbox)
        sample_row.addWidget(self.sample_size_spinbox)
        sample_row.addWidget(QLabel("images"))
        sample_row.addStretch()
        image_rule_layout.addLayout(sample_row)

        image_rule_box.setLayout(image_rule_layout)
        self.right_layout.addWidget(image_rule_box)

        self.class_filter_box = QGroupBox("Classes to Import")
        class_filter_layout = QVBoxLayout()

        class_buttons_row = QHBoxLayout()
        self.select_all_classes_button = QPushButton("Select All")
        self.select_all_classes_button.setToolTip("Check every class.")
        self.select_all_classes_button.clicked.connect(lambda: self.set_all_classes_checked(True))
        self.deselect_all_classes_button = QPushButton("Deselect All")
        self.deselect_all_classes_button.setToolTip("Uncheck every class.")
        self.deselect_all_classes_button.clicked.connect(lambda: self.set_all_classes_checked(False))
        class_buttons_row.addWidget(self.select_all_classes_button)
        class_buttons_row.addWidget(self.deselect_all_classes_button)
        class_filter_layout.addLayout(class_buttons_row)

        self.class_scroll_area = QScrollArea()
        self.class_scroll_area.setWidgetResizable(True)
        # Ignored vertically so the list never drives the dialog's height: it
        # takes whatever the left column leaves, and scrolls past that.
        self.class_scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.class_scroll_area.setMinimumHeight(80)
        self.class_widget = QFrame()
        self.class_layout = QVBoxLayout(self.class_widget)
        self.class_layout.setAlignment(Qt.AlignTop)
        self.class_scroll_area.setWidget(self.class_widget)
        class_filter_layout.addWidget(self.class_scroll_area)
        self.class_filter_box.setLayout(class_filter_layout)
        self.right_layout.addWidget(self.class_filter_box, 1)

    def set_all_classes_checked(self, checked):
        """Check or uncheck every class in the Classes to Import list."""
        for checkbox in self.class_checkboxes:
            checkbox.setChecked(checked)

    def update_annotation_widgets(self):
        """Grey out the annotation settings when no annotations are imported."""
        importing_annotations = not self.import_images_only_radio.isChecked()
        self.class_filter_box.setEnabled(importing_annotations)
        # Semantic dialogs do not build the format combo.
        import_as_combo = getattr(self, 'import_as_combo', None)
        if import_as_combo is not None:
            import_as_combo.setEnabled(importing_annotations)

    def update_output_widgets(self):
        """Grey out the destination fields when nothing is being written."""
        copying = self.copy_images_radio.isChecked()
        for widget in (self.output_dir_row_label,
                       self.output_dir_label,
                       self.browse_output_button,
                       self.output_name_row_label,
                       self.output_folder_name):
            widget.setEnabled(copying)

    def setup_buttons_layout(self, parent_layout):
        """Set up the OK/Cancel button box beneath both columns."""
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        ok_button.setToolTip("Import the dataset with the configured settings.")
        cancel_button.setToolTip("Close this dialog without importing.")
        self.button_box.accepted.connect(self.start_processing)
        self.button_box.rejected.connect(self.reject)
        parent_layout.addWidget(self.button_box)

    def browse_dataset_file(self):
        """Open a file dialog for a data.yaml or a COCO .json, and load it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "Datasets (*.yaml *.yml *.json);;YOLO data.yaml (*.yaml *.yml);;"
            "COCO annotations (*.json);;All Files (*)"
        )
        if not file_path:
            return

        if file_path.lower().endswith('.json'):
            self.load_coco_dataset(file_path)
        else:
            self.load_yolo_dataset(file_path)

    def browse_dataset_folder(self):
        """Open a folder dialog for a folder of COCO .json files, and load it."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder of COCO Annotation Files")
        if dir_path:
            self.load_coco_dataset(dir_path)

    def browse_images_dir(self):
        """Pick where a COCO dataset's images are, and read the dataset again to find them."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if dir_path:
            self.images_dir_edit.setText(dir_path)
            self.load_coco_dataset(self.dataset_path_edit.text(), reload=True)

    def set_class_list(self, names, values, keep_unchecked=False):
        """Rebuild the Classes to Import list.

        Args:
            names (list): Class names, in the order they are listed.
            values (list): Each class's value in the dataset, as class_values.
            keep_unchecked (bool): Leave classes the user had unchecked
                unchecked, for when the same dataset is read again.
        """
        unchecked = set()
        if keep_unchecked:
            unchecked = {checkbox.text() for checkbox in self.class_checkboxes if not checkbox.isChecked()}

        for checkbox in self.class_checkboxes:
            self.class_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.class_checkboxes.clear()

        for name in names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(name not in unchecked)
            self.class_layout.addWidget(checkbox)
            self.class_checkboxes.append(checkbox)
        self.class_values = list(values)

    def load_coco_dataset(self, path, reload=False):
        """Read COCO annotations - one file, or a folder of them - and list their classes.

        The whole dataset is read here, not just its class names as for a
        YAML: a COCO file holds its annotations, so this is the read the import
        needs anyway, and the summary can say how many images were found.

        Args:
            path (str): A .json file, or a folder of them.
            reload (bool): The same dataset read again after its images folder
                changed, which keeps the output settings and unchecked classes.
        """
        if not reload:
            self.images_dir_edit.clear()

        json_paths = find_annotation_files(path)
        if not json_paths:
            QMessageBox.warning(self, "No Annotation Files", f"There are no .json files in:\n{path}")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dataset, problems = load_annotation_files(json_paths, self.images_dir_edit.text() or None)
        finally:
            QApplication.restoreOverrideCursor()

        # A tool that writes one file per image is nearly always meant to be
        # read whole, so picking one of its files offers the rest of them
        if not reload and len(json_paths) == 1 and dataset['dialects'] == [DIALECT_CORALSCOP]:
            folder = os.path.dirname(path)
            siblings = find_annotation_files(folder)
            if len(siblings) > 1:
                reply = QMessageBox.question(self,
                                             "One File per Image",
                                             "This annotation file describes a single image, and its folder "
                                             f"holds {len(siblings)} .json files.\n\n"
                                             "Import all of them as one dataset?",
                                             QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.load_coco_dataset(folder)
                    return

        if not dataset['images']:
            _message_with_details(self,
                                  QMessageBox.Critical,
                                  "Error",
                                  f"No COCO annotations could be read from:\n{path}",
                                  '\n'.join(problems))
            return

        self.coco_dataset = dataset
        self.coco_problems = problems
        self.dataset_path_edit.setText(path)
        if not reload:
            self.output_dir_label.setText(path if os.path.isdir(path) else os.path.dirname(path))
            self.output_folder_name.setText("data")
        self.set_images_row_visible(True)

        # Only the categories something uses: an unused one is noise at best,
        # and at worst a placeholder like the one Roboflow puts at id 0
        counts = category_counts(dataset)
        used = [category for category in dataset['categories'] if counts[category['id']]]
        self.set_class_list([category['name'] for category in used],
                            [category['id'] for category in used],
                            keep_unchecked=reload)

        self.update_coco_summary()

    def update_coco_summary(self):
        """Summarize the loaded COCO dataset in a single line, with its problems on hover."""
        dataset = self.coco_dataset
        images = dataset['images']
        found = sum(1 for image in images if image['path'])
        annotated_ids = {annotation['image_id'] for annotation in dataset['annotations']}
        annotated = sum(1 for image in images if image['path'] and image['id'] in annotated_ids)
        class_count = len(self.class_checkboxes)
        plural = "es" if class_count != 1 else ""
        text = (f"COCO · {len(images)} images, {found} found · {annotated} with annotations · "
                f"{len(dataset['annotations'])} annotations · {class_count} class{plural}")
        tooltip = "What was found in the selected dataset."
        if self.coco_problems:
            text += f" · {len(self.coco_problems)} issue(s), hover for details"
            shown = self.coco_problems[:30]
            tooltip += "\n\n" + "\n".join(shown)
            if len(self.coco_problems) > len(shown):
                tooltip += f"\n... and {len(self.coco_problems) - len(shown)} more"

        self.dataset_summary_label.setText(text)
        self.dataset_summary_label.setToolTip(tooltip)

    def load_yolo_dataset(self, file_path):
        """Read a YOLO data.yaml's class names into the class list."""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            
            names_data = data.get('names')
            if not names_data:
                QMessageBox.warning(self,
                                    "Warning", 
                                    "Could not find a 'names' entry in the selected YAML file.")
                return

            names_to_display = []
            if isinstance(names_data, dict):
                names_to_display = [str(names_data[key]) for key in sorted(names_data.keys())]
            elif isinstance(names_data, list):
                names_to_display = [str(name) for name in names_data]
            else:
                QMessageBox.warning(self, 
                                    "Format Error", 
                                    f"The 'names' entry in the YAML has an unexpected format: {type(names_data)}.")
                return
            
            self.dataset_path_edit.setText(file_path)
            yaml_dir = os.path.dirname(file_path)
            self.output_dir_label.setText(yaml_dir)
            self.output_folder_name.setText("data")

            # A COCO dataset picked earlier no longer applies
            self.coco_dataset = None
            self.coco_problems = []
            self.images_dir_edit.clear()
            self.set_images_row_visible(False)

            # A YOLO class's value is its index in names
            self.set_class_list(names_to_display, range(len(names_to_display)))

            self.update_dataset_summary(file_path, len(names_to_display))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read or parse YAML file:\n{e}")
            self.dataset_summary_label.setText("No dataset selected.")

    def update_dataset_summary(self, yaml_path, class_count):
        """Summarize the selected dataset in a single line.

        The scan is the same one the import itself runs, so the counts shown are
        exactly what would be brought in. A previous import's output folder is
        skipped, otherwise re-importing a dataset would count its own copies.
        """
        sidecar_kind = 'masks' if self.task == 'semantic' else 'labels'
        exclude_dirs = []
        if self.output_dir_label.text() and self.output_folder_name.text():
            exclude_dirs.append(os.path.join(self.output_dir_label.text(),
                                             self.output_folder_name.text()))

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            discovered = discover_dataset_files(yaml_path,
                                                image_import_policy='all',
                                                exclude_dirs=exclude_dirs,
                                                sidecar_kind=sidecar_kind)
        except Exception:
            discovered = {}
        finally:
            QApplication.restoreOverrideCursor()

        self.dataset_summary_label.setToolTip("What was found in the selected dataset.\n"
                                              "Counts every split the YAML points at.")
        plural = "es" if class_count != 1 else ""
        if not discovered:
            self.dataset_summary_label.setText(
                f"{class_count} class{plural} · no images found "
                "(check the YAML's train / val / test entries)")
            return

        annotated = sum(1 for sidecar in discovered.values() if sidecar)
        noun = "masks" if self.task == 'semantic' else "annotations"
        self.dataset_summary_label.setText(
            f"{len(discovered)} images · {annotated} with {noun} · {class_count} class{plural}")

    def browse_output_dir(self):
        """Open a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def start_processing(self):
        """Validate inputs, check for duplicates, and start the worker thread."""
        copy_images = self.copy_images_radio.isChecked()
        rename_files = False

        if not self.dataset_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select a dataset: a data.yaml, or COCO annotations.")
            return

        if copy_images:
            if not all([self.output_dir_label.text(), self.output_folder_name.text()]):
                QMessageBox.warning(self, "Error", "Please fill in all fields.")
                return

            self.output_folder = os.path.join(self.output_dir_label.text(), self.output_folder_name.text())
            if os.path.exists(self.output_folder) and os.listdir(self.output_folder):
                reply = QMessageBox.question(self,
                                             'Directory Not Empty',
                                             f"The directory '{self.output_folder}' is not empty. Continue?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No: return

            if self.coco_dataset is not None:
                # Every image the annotation file resolved, which is what the
                # worker copies from
                image_paths = [image['path'] for image in self.coco_dataset['images'] if image['path']]
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    # The same discovery the worker will use, so the conflict check
                    # sees exactly the files that are about to be copied.
                    image_paths = discover_dataset_files(self.dataset_path_edit.text(),
                                                         image_import_policy='all',
                                                         exclude_dirs=[self.output_folder])
                finally:
                    QApplication.restoreOverrideCursor()

            basenames, duplicates_exist = set(), False
            for path in image_paths:
                basename_no_ext = os.path.splitext(os.path.basename(path))[0]
                if basename_no_ext in basenames:
                    duplicates_exist = True
                    break
                basenames.add(basename_no_ext)

            if duplicates_exist:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle('Duplicate Filenames Found')
                msg_box.setText(
                    "Images with the same base name exist in different subdirectories.\n"
                    "This can cause files to be overwritten in the output directory."
                )
                msg_box.setInformativeText("How would you like to handle these conflicts?")
                rename_button = msg_box.addButton("Rename Files (Safe)", QMessageBox.AcceptRole)
                overwrite_button = msg_box.addButton("Overwrite", QMessageBox.DestructiveRole)
                cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
                msg_box.setDefaultButton(rename_button)
                msg_box.exec_()
                clicked_button = msg_box.clickedButton()
                if clicked_button == cancel_button:
                    return
                elif clicked_button == rename_button:
                    rename_files = True
                elif clicked_button == overwrite_button:
                    rename_files = False
                else:
                    return
        else:
            # Nothing is written, so there is no destination to validate and no
            # flattening to collide: the images keep the distinct paths they
            # already have, which is what the project will reference.
            self.output_folder = None

        excluded_classes = {cb.text() for cb in self.class_checkboxes if not cb.isChecked()}

        if self.import_images_only_radio.isChecked():
            image_import_policy = 'images_only'
        elif self.import_all_images_radio.isChecked():
            image_import_policy = 'all'
        else:
            image_import_policy = 'annotated_only'
        # Kept on the dialog so the completion message can describe the import.
        self.images_only = image_import_policy == 'images_only'
        self.sample_size =self.sample_size_spinbox.value() if self.sample_subset_checkbox.isChecked() else None
        # Semantic imports have no geometry to choose a representation for,
        # so those dialogs do not build the combo at all.
        if getattr(self, 'import_as_combo', None) is None:
            import_as = 'mask'
        else:
            import_as = 'polygon' if 'Polygon' in self.import_as_combo.currentText() else 'rectangle'

        self.button_box.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progress_bar = ProgressBar(self, title="Preparing to Import...")
        self.progress_bar.show()

        self.thread = QThread()
        self.worker = DatasetProcessor(
            yaml_path=None if self.coco_dataset is not None else self.dataset_path_edit.text(),
            output_folder=self.output_folder,
            task=self.task,
            import_as=import_as,
            rename_on_conflict=rename_files,
            excluded_classes=excluded_classes,
            image_import_policy=image_import_policy,
            copy_images=copy_images,
            sample_size=self.sample_size,
            coco_dataset=self.coco_dataset,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_error)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.progress_updated.connect(self.on_progress_update)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.thread.start()

    def _live_progress_bar(self):
        """The worker's progress dialog, or None once it has been closed.

        ProgressBar.set_title() pumps the event loop, which drains the worker's
        queued signals re-entrantly: a fast import can run on_processing_complete
        and on_worker_finished inside a status slot, closing (and, through
        WA_DeleteOnClose, deleting) the dialog that slot is still updating.
        """
        bar = self.progress_bar
        if bar is None or sip.isdeleted(bar):
            return None
        return bar

    def on_status_changed(self, title, total):
        bar = self._live_progress_bar()
        if bar is None:
            return
        # set_title last: it pumps events, after which the bar may be gone.
        bar.start_progress(total)
        bar.set_title(title)

    def on_progress_update(self, value):
        bar = self._live_progress_bar()
        if bar is not None:
            bar.set_value(value)

    def on_processing_complete(self, raw_annotations, image_paths, parsing_errors, duplicates_removed=0):
        if self.task == 'semantic':
            # A semantic mask holds one class per pixel, so it has no duplicates
            # to remove and the count is always zero here.
            self.on_mask_processing_complete(raw_annotations, image_paths, parsing_errors)
            return

        progress_bar = ProgressBar(self, title="Adding Data to Project...")
        progress_bar.show()

        added_paths = []
        progress_bar.set_title(f"Adding {len(image_paths)} images...")
        progress_bar.start_progress(len(image_paths))
        
        for path in image_paths:
            if self.image_window.raster_manager.add_raster(path, emit_signal=False):
                added_paths.append(path)
            progress_bar.update_progress()

        newly_created_annotations = []
        progress_bar.set_title(f"Creating {len(raw_annotations)} annotation objects...")
        progress_bar.start_progress(len(raw_annotations))

        # 1. Create all annotation objects in memory first
        for raw_ann in raw_annotations:
            # Defer UI refresh; batch-refresh once after the loop to avoid flashing.
            label = self.main_window.label_window.add_label_if_not_exists(raw_ann["class_name"], refresh_ui=False)
            annotation = None
            if raw_ann["type"] == "RectangleAnnotation":
                tl, br = raw_ann["top_left"], raw_ann["bottom_right"]
                annotation = RectangleAnnotation(
                    QPointF(tl[0], tl[1]),
                    QPointF(br[0], br[1]),
                    label,
                    raw_ann["image_path"],
                    transparency=self.main_window.get_transparency_value(),
                )
            else: # PolygonAnnotation
                points = [QPointF(p[0], p[1]) for p in raw_ann["points"]]
                holes = [[QPointF(x, y) for x, y in hole] for hole in raw_ann.get("holes", [])]
                annotation = PolygonAnnotation(
                    points,
                    label,
                    raw_ann["image_path"],
                    transparency=self.main_window.get_transparency_value(),
                    holes=holes,
                    # A traced COCO mask was simplified as it was traced, so the
                    # cleanup pass would cost a shapely round-trip for nothing
                    simplify=not raw_ann.get("traced", False),
                )
            
            if annotation:
                newly_created_annotations.append(annotation)

            progress_bar.update_progress()

        # Single UI refresh for the whole batch (avoids per-label flashing).
        self.main_window.label_window.refresh_after_batch_add()

        # 2. Add all created annotations to the project in a single batch operation
        if newly_created_annotations:
            progress_bar.set_title("Adding annotations to project...")
            self.annotation_window.add_annotations(newly_created_annotations)

        # An in-place import has no folder of its own to write the sidecar into;
        # File > Export > Annotations (JSON) covers it on demand. With nothing
        # created there is nothing to write either.
        if self.output_folder and newly_created_annotations:
            progress_bar.set_title("Exporting annotations.json...")
            self._export_annotations_to_json(newly_created_annotations, self.output_folder)


        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
        
        # Manually perform a full UI update exactly once.
        self.image_window.update_search_bars()
        self.image_window.filter_images()

        if added_paths:
            last_image_path = added_paths[-1]
            self.image_window.load_image_by_path(last_image_path)
            for path in added_paths:
                self.image_window.update_image_annotations(path)
            self.annotation_window.load_annotations()

        summary_message = "Dataset has been successfully imported."
        if self.images_only:
            summary_message += f"\n\nImported {len(added_paths)} image(s) without annotations."
        if self.sample_size:
            summary_message += f"\n\nImported a random subset of {len(image_paths)} image(s)."
        if duplicates_removed:
            summary_message += (f"\n\nRemoved {duplicates_removed} exact duplicate annotation(s) "
                                f"({len(newly_created_annotations)} kept).")
        if parsing_errors:
            _message_with_details(self,
                                  QMessageBox.Warning,
                                  "Import Complete with Warnings",
                                  f"{summary_message}\n\nHowever, {len(parsing_errors)} issue(s) were found. "
                                  "Please review them below.",
                                  '\n'.join(parsing_errors))
        else:
            QMessageBox.information(self, "Dataset Imported", summary_message)
            
    def get_class_id_to_label(self):
        """Map each class's value in the dataset to the project Label it should import as.

        For YOLO the value is the class ID, the index into the YAML's names --
        which is also the pixel value in a semantic mask. For COCO it is the
        category id, which is the value paint_class_mask writes. Two kinds of
        class are deliberately absent from the result, and so stay background:
        ones the user unchecked, and a YOLO class literally named "background"
        at index 0, which is what this toolbox's own semantic export writes to
        mean "nothing here". COCO ids start at 1, so that rule never applies.
        """
        class_id_to_label = {}
        for class_id, checkbox in zip(self.class_values, self.class_checkboxes, strict=True):
            if not checkbox.isChecked():
                continue

            class_name = checkbox.text()
            if class_id == 0 and class_name.strip().lower() == 'background':
                continue

            label = self.main_window.label_window.add_label_if_not_exists(class_name,
                                                                         refresh_ui=False)
            class_id_to_label[class_id] = label

        self.main_window.label_window.refresh_after_batch_add()
        return class_id_to_label

    def on_mask_processing_complete(self, raw_records, image_paths, parsing_errors):
        """Attach a MaskAnnotation to each imported image.

        Reading happens here rather than in the worker because a
        MaskAnnotation builds a QImage canvas, which belongs on the GUI thread.
        """
        progress_bar = ProgressBar(self, title="Adding Data to Project...")
        progress_bar.show()

        added_paths = []
        progress_bar.set_title(f"Adding {len(image_paths)} images...")
        progress_bar.start_progress(len(image_paths))

        for path in image_paths:
            if self.image_window.raster_manager.add_raster(path, emit_signal=False):
                added_paths.append(path)
            progress_bar.update_progress()

        # Only create the YAML's labels when there are masks to use them;
        # an images-only import should not fill the label window.
        class_id_to_label = self.get_class_id_to_label() if raw_records else {}
        project_labels = list(self.main_window.label_window.labels)

        errors = list(parsing_errors)
        imported = 0

        progress_bar.set_title(f"Importing {len(raw_records)} masks...")
        progress_bar.start_progress(len(raw_records))

        for record in raw_records:
            image_path = record["image_path"]
            # A COCO record carries its annotations rather than a mask file
            mask_path = record.get("mask_path") or image_path
            try:
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    raise ValueError("Image was not added to the project.")

                if "annotations" in record:
                    shape = (raster.height, raster.width)
                    problem = _size_problem(record["size"], shape)
                    if problem:
                        raise ValueError(problem)
                    source_mask = paint_class_mask([(annotation['category_id'], annotation)
                                                    for annotation in record["annotations"]], *shape)
                else:
                    # Read band 1: a semantic mask is single-channel by definition,
                    # and a palette PNG still carries the class IDs as its indices.
                    with rasterio.open(mask_path) as src:
                        source_mask = src.read(1)

                mask_annotation = build_mask_annotation(
                    image_path=image_path,
                    source_mask=source_mask,
                    value_to_label=class_id_to_label,
                    project_labels=project_labels,
                    shape=(raster.height, raster.width),
                    rasterio_src=raster.rasterio_src,
                    transparency=self.main_window.get_transparency_value(),
                )

                if raster.mask_annotation is not None:
                    raster.mask_annotation.remove_from_scene()
                raster.mask_annotation = mask_annotation
                imported += 1

            except Exception as e:
                errors.append(f"Could not import mask '{os.path.basename(mask_path)}': {e}\n")

            progress_bar.update_progress()

        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

        # Manually perform a full UI update exactly once.
        self.image_window.update_search_bars()
        self.image_window.filter_images()

        if added_paths:
            last_image_path = added_paths[-1]
            self.image_window.load_image_by_path(last_image_path)
            for path in added_paths:
                self.image_window.update_image_annotations(path)
            self.annotation_window.load_annotations()
            self.annotation_window.load_mask_annotation()

        if self.images_only:
            summary_message = (f"Dataset has been successfully imported "
                               f"({len(added_paths)} image(s) without masks).")
        else:
            summary_message = f"Dataset has been successfully imported ({imported} mask(s))."
        if self.sample_size:
            summary_message += f"\n\nImported a random subset of {len(image_paths)} image(s)."
        if errors:
            _message_with_details(self,
                                  QMessageBox.Warning,
                                  "Import Complete with Warnings",
                                  f"{summary_message}\n\nHowever, {len(errors)} issue(s) were found. "
                                  "Please review them below.",
                                  '\n'.join(errors))
        else:
            QMessageBox.information(self, "Dataset Imported", summary_message)

    def _export_annotations_to_json(self, annotations_list, output_dir):
        """
        Merges the list of annotation objects into an existing annotations.json file,
        or creates a new one if it doesn't exist.
        """
        export_dict = {}
        json_path = os.path.join(output_dir, "annotations.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as file:
                    export_dict = json.load(file)
                if not isinstance(export_dict, dict):
                    raise TypeError("annotations.json is not in the expected format (dict).")
            except (json.JSONDecodeError, TypeError, IOError) as e:
                QMessageBox.warning(self, 
                                    "Read Error",
                                    f"Could not read or parse existing annotations.json:\n{e}\n\n"
                                    "A new file will be created, overwriting the old one.")
                export_dict = {}

        for annotation in annotations_list:
            image_path = annotation.image_path
            export_dict.setdefault(image_path, [])
            
            annotation_dict = annotation.to_dict()
            if isinstance(annotation, RectangleAnnotation):
                annotation_dict['type'] = 'RectangleAnnotation'
            elif isinstance(annotation, PolygonAnnotation):
                annotation_dict['type'] = 'PolygonAnnotation'
            else:
                warnings.warn(f"Unknown annotation type skipped during export: {type(annotation)}")
                continue

            export_dict[image_path].append(annotation_dict)

        try:
            with open(json_path, 'w') as file:
                json.dump(export_dict, file, indent=4)
                file.flush()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write annotations.json:\n{e}")

    def on_error(self, message):
        QMessageBox.warning(self, "Error", message)

    def on_worker_finished(self):
        # Already torn down: a duplicate delivery is a no-op.
        if self.thread is None:
            return

        bar = self._live_progress_bar()
        if bar is not None:
            bar.stop_progress()
            bar.close()
        self.progress_bar = None
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.worker = None
        self.thread = None
        QApplication.restoreOverrideCursor()
        self.button_box.setEnabled(True)
        self.accept()

    def reject(self):
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.button_box.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        self.reject()
        super().closeEvent(event)