import warnings

import os
import uuid
import yaml
import rasterio
import shutil
import ujson as json

from PyQt5.QtCore import Qt, QPointF, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QDialog, QPushButton, QDialogButtonBox,
                             QGridLayout, QScrollArea, QFrame, QCheckBox, QRadioButton,
                             QToolButton)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import build_mask_annotation

from coralnet_toolbox.IO.QtImportImages import SUPPORTED_IMAGE_EXTENSIONS

# Semantic masks are single-channel class-ID rasters, so only lossless
# formats are accepted: JPEG compression would rewrite the very integers
# that carry the class.
MASK_EXTENSIONS = ('.png', '.tif', '.tiff')

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon, get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
        if resolved and resolved not in image_dirs and not _is_excluded(resolved, exclude_dirs):
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
            normalized = os.path.normpath(image_path)
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
    processing_complete = pyqtSignal(list, list, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, yaml_path, output_folder, task, import_as, rename_on_conflict=False,
                 excluded_classes=None, image_import_policy='annotated_only', parent=None):
        super().__init__(parent)
        self.yaml_path = yaml_path
        self.output_folder = output_folder
        self.task = task  # 'detect' or 'segment' (source format)
        self.import_as = import_as  # 'rectangle' or 'polygon' (target format)
        self.rename_on_conflict = rename_on_conflict
        self.excluded_classes = excluded_classes if excluded_classes is not None else set()
        self.image_import_policy = image_import_policy
        self.is_running = True
        self.parsing_errors = []  # To collect errors instead of printing

    def stop(self):
        self.is_running = False

    # In class DatasetProcessor, update this method:

    def run(self):
        """Main processing method executed in the thread."""
        try:
            # Step 1: Read YAML and discover files
            with open(self.yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            class_names = data.get('names', [])

            source_image_label_map = self._find_source_files()

            if not source_image_label_map:
                self.error.emit("No valid image/label pairs found in the dataset.")
                return

            # --- Step 2: Copy files with progress reporting ---
            self.status_changed.emit("Copying image files...", len(source_image_label_map))
            image_label_paths = self._copy_files_with_progress(source_image_label_map)

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
            self.processing_complete.emit(raw_annotations, image_paths, self.parsing_errors)

        except Exception as e:
            self.error.emit(f"An error occurred during processing: {str(e)}")
        finally:
            self.finished.emit()

    def _find_source_files(self):
        """Finds all source image and sidecar paths based on the import policy."""
        sidecar_kind = 'masks' if self.task == 'semantic' else 'labels'
        return discover_dataset_files(self.yaml_path,
                                      image_import_policy=self.image_import_policy,
                                      exclude_dirs=[self.output_folder],
                                      sidecar_kind=sidecar_kind)

    def _copy_files_with_progress(self, source_image_label_map):
        """Copies files and reports progress for each file.

        Semantic masks are copied too, and are renamed in lockstep with their
        image so the pair stays together in the flattened output.
        """
        img_out_dir = os.path.join(self.output_folder, "images")
        os.makedirs(img_out_dir, exist_ok=True)

        copy_masks = self.task == 'semantic'
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
        """
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
        self.resize(500, 350)

        self.task = None
        self.progress_bar = None
        self.thread = None
        self.worker = None
        self.output_folder = None
        self.class_checkboxes = []

        self.layout = QVBoxLayout(self)
        self.setup_info_layout()
        self.setup_yaml_layout()
        self.setup_output_layout()
        self.setup_buttons_layout()

        self.advanced_options_toggle.setEnabled(False)
        self.advanced_options_frame.setVisible(False)

    def setup_info_layout(self):
        raise NotImplementedError("Subclasses must implement method.")

    def setup_yaml_layout(self):
        """Set up the layout for selecting the data YAML file."""
        group_box = QGroupBox("Data YAML File")
        layout = QGridLayout()
        layout.addWidget(QLabel("File:"), 0, 0)
        self.yaml_path_label = QLineEdit()
        self.yaml_path_label.setReadOnly(True)
        self.yaml_path_label.setPlaceholderText("Select Data YAML file...")
        self.yaml_path_label.setToolTip("Path to the dataset.yaml file from your YOLO dataset.\nDefines the structure and class names for import.")
        layout.addWidget(self.yaml_path_label, 0, 1)
        self.browse_yaml_button = QPushButton("Browse")
        self.browse_yaml_button.clicked.connect(self.browse_data_yaml)
        self.browse_yaml_button.setToolTip("Browse for a dataset.yaml file.")
        layout.addWidget(self.browse_yaml_button, 0, 2)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """Set up the layout for output directory and the advanced options accordion."""
        group_box = QGroupBox("Output Settings")
        layout = QGridLayout()
        layout.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_label = QLineEdit()
        self.output_dir_label.setPlaceholderText("Select output directory...")
        self.output_dir_label.setToolTip("Directory where the imported dataset will be saved.\nDefault: same directory as the YAML file.")
        layout.addWidget(self.output_dir_label, 0, 1)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_dir)
        self.browse_output_button.setToolTip("Browse for an output directory.")
        layout.addWidget(self.browse_output_button, 0, 2)
        layout.addWidget(QLabel("Folder Name:"), 1, 0)
        self.output_folder_name = QLineEdit("data")
        self.output_folder_name.setPlaceholderText("data")
        self.output_folder_name.setToolTip("Name for the subdirectory containing imported images.\nFinal location: Directory / Folder Name /")
        layout.addWidget(self.output_folder_name, 1, 1, 1, 2)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

        self.advanced_options_toggle = QToolButton()
        self.advanced_options_toggle.setText("Advanced Options")
        self.advanced_options_toggle.setCheckable(True)
        self.advanced_options_toggle.setStyleSheet("QToolButton { border: none; }")
        self.advanced_options_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_options_toggle.setArrowType(Qt.RightArrow)
        self.advanced_options_toggle.toggled.connect(self.toggle_advanced_options)
        self.advanced_options_toggle.setToolTip("Expand to access image import rules, class filtering, and annotation format options.")
        self.layout.addWidget(self.advanced_options_toggle)

        self.advanced_options_frame = QFrame()
        self.advanced_options_frame.setFrameShape(QFrame.StyledPanel)
        advanced_layout = QVBoxLayout(self.advanced_options_frame)

        image_rule_box = QGroupBox("Image Import Rule")
        image_rule_layout = QVBoxLayout()
        self.import_annotated_images_radio = QRadioButton("Import only images with annotations")
        self.import_annotated_images_radio.setToolTip("Only import images that have corresponding annotation label files.\nSkips images without labels.")
        self.import_all_images_radio = QRadioButton("Import all images found in dataset")
        self.import_all_images_radio.setToolTip("Import all images found in the dataset, with or without annotations.\nUseful for datasets with partially-labeled images.")
        self.import_annotated_images_radio.setChecked(True)
        image_rule_layout.addWidget(self.import_annotated_images_radio)
        image_rule_layout.addWidget(self.import_all_images_radio)
        image_rule_box.setLayout(image_rule_layout)
        advanced_layout.addWidget(image_rule_box)

        class_filter_box = QGroupBox("Classes to Import")
        class_filter_layout = QVBoxLayout()
        self.class_scroll_area = QScrollArea()
        self.class_scroll_area.setWidgetResizable(True)
        self.class_widget = QFrame()
        self.class_layout = QVBoxLayout(self.class_widget)
        self.class_scroll_area.setWidget(self.class_widget)
        class_filter_layout.addWidget(self.class_scroll_area)
        class_filter_box.setLayout(class_filter_layout)
        advanced_layout.addWidget(class_filter_box)
        self.layout.addWidget(self.advanced_options_frame)

    def toggle_advanced_options(self, checked):
        self.advanced_options_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.advanced_options_frame.setVisible(checked)

    def setup_buttons_layout(self):
        """Set up the OK/Cancel button box."""
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        ok_button.setToolTip("Import the dataset with the configured settings.\nImages and annotations will be copied to the output directory.")
        cancel_button.setToolTip("Close this dialog without importing.")
        self.button_box.accepted.connect(self.start_processing)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def browse_data_yaml(self):
        """Open a file dialog to select the data YAML file and populate advanced options."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Data YAML",
            "", 
            "YAML Files (*.yaml);;All Files (*)", 
            options=options
        )
        if not file_path:
            return

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
            
            self.yaml_path_label.setText(file_path)
            yaml_dir = os.path.dirname(file_path)
            self.output_dir_label.setText(yaml_dir)
            self.output_folder_name.setText("data")

            for checkbox in self.class_checkboxes:
                self.class_layout.removeWidget(checkbox)
                checkbox.deleteLater()
            self.class_checkboxes.clear()

            for name in names_to_display:
                checkbox = QCheckBox(name)
                checkbox.setChecked(True)
                self.class_layout.addWidget(checkbox)
                self.class_checkboxes.append(checkbox)

            self.advanced_options_toggle.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read or parse YAML file:\n{e}")
            self.advanced_options_toggle.setEnabled(False)

    def browse_output_dir(self):
        """Open a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def start_processing(self):
        """Validate inputs, check for duplicates, and start the worker thread."""
        if not all([self.yaml_path_label.text(), self.output_dir_label.text(), self.output_folder_name.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all fields.")
            return

        self.output_folder = os.path.join(self.output_dir_label.text(), self.output_folder_name.text())
        if os.path.exists(self.output_folder) and os.listdir(self.output_folder):
            reply = QMessageBox.question(self, 
                                         'Directory Not Empty', 
                                         f"The directory '{self.output_folder}' is not empty. Continue?", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # The same discovery the worker will use, so the conflict check
            # sees exactly the files that are about to be copied.
            image_paths = discover_dataset_files(self.yaml_path_label.text(),
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

        rename_files = False
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

        excluded_classes = set()
        if self.advanced_options_toggle.isEnabled():
            for cb in self.class_checkboxes:
                if not cb.isChecked():
                    excluded_classes.add(cb.text())
                
        image_import_policy = 'all' if self.import_all_images_radio.isChecked() else 'annotated_only'
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
            yaml_path=self.yaml_path_label.text(),
            output_folder=self.output_folder,
            task=self.task,
            import_as=import_as,
            rename_on_conflict=rename_files,
            excluded_classes=excluded_classes,
            image_import_policy=image_import_policy
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_error)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.progress_updated.connect(self.on_progress_update)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.thread.start()

    def on_status_changed(self, title, total):
        self.progress_bar.set_title(title)
        self.progress_bar.start_progress(total)

    def on_progress_update(self, value):
        self.progress_bar.set_value(value)

    def on_processing_complete(self, raw_annotations, image_paths, parsing_errors):
        if self.task == 'semantic':
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
                annotation = PolygonAnnotation(
                    points,
                    label,
                    raw_ann["image_path"],
                    transparency=self.main_window.get_transparency_value(),
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
        if parsing_errors:
            QMessageBox.warning(self, 
                                "Import Complete with Warnings", 
                                f"{summary_message}\n\nHowever, {len(parsing_errors)} issue(s) were found. "
                                "Please review them below.", 
                                details='\n'.join(parsing_errors))
        else:
            QMessageBox.information(self, "Dataset Imported", summary_message)
            
    def get_class_id_to_label(self):
        """Map each YOLO class ID to the project Label it should import as.

        The checkbox list is built from the YAML's names in order, so its index
        is the class ID -- which is also the pixel value in a semantic mask.
        Two kinds of class are deliberately absent from the result, and so stay
        background: ones the user unchecked, and a class literally named
        "background", which is what this toolbox's own semantic export writes
        at index 0 to mean "nothing here".
        """
        class_id_to_label = {}
        for class_id, checkbox in enumerate(self.class_checkboxes):
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

        class_id_to_label = self.get_class_id_to_label()
        project_labels = list(self.main_window.label_window.labels)

        errors = list(parsing_errors)
        imported = 0

        progress_bar.set_title(f"Importing {len(raw_records)} masks...")
        progress_bar.start_progress(len(raw_records))

        for record in raw_records:
            image_path = record["image_path"]
            mask_path = record["mask_path"]
            try:
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    raise ValueError("Image was not added to the project.")

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

        summary_message = f"Dataset has been successfully imported ({imported} mask(s))."
        if errors:
            QMessageBox.warning(self,
                                "Import Complete with Warnings",
                                f"{summary_message}\n\nHowever, {len(errors)} issue(s) were found. "
                                "Please review them below.",
                                details='\n'.join(errors))
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

        if self.progress_bar:
            self.progress_bar.stop_progress()
            self.progress_bar.close()
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