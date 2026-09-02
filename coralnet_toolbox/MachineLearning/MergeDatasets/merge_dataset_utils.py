import warnings

import os
import shutil
import yaml
import ujson as json

import numpy as np
from PIL import Image

from coralnet_toolbox.MachineLearning.ImportDataset.QtBase import (
    MASK_EXTENSIONS,
    _iter_images,
    _normalize_image_dir,
    _pair,
    _resolve_image_dir,
    _sidecar_dir_for,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Output splits always use the names the exporters write, so a merged dataset is
# indistinguishable from a freshly exported one.
OUTPUT_SPLITS = ('train', 'valid', 'test')

# A data.yaml may name the validation split either way; both fold onto 'valid'.
YAML_SPLIT_KEYS = (('train', 'train'), ('val', 'valid'), ('valid', 'valid'), ('test', 'test'))

# Semantic masks reserve 255 for ignore/unlabeled, which is what the exporter
# fills unannotated pixels with when 'background' is not a class of its own.
IGNORE_VALUE = 255

# 255 is spoken for by IGNORE_VALUE, so a single-channel uint8 mask can carry at
# most 255 distinct classes (0..254).
MAX_SEMANTIC_CLASSES = IGNORE_VALUE

BACKGROUND_NAME = 'background'


# ----------------------------------------------------------------------------------------------------------------------
# Dataset Discovery
# ----------------------------------------------------------------------------------------------------------------------


def load_data_yaml(yaml_path):
    """Parse a data.yaml, returning an empty dict rather than raising."""
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_dataset_names(yaml_path):
    """Return a data.yaml's class names as {index: name}.

    The 'names' key is written as a dict by this toolbox's exporters but as a
    plain list by most other YOLO tooling, and a dict loaded from YAML may key
    on either ints or strings. All three forms normalize to {int: str}.
    """
    data = load_data_yaml(yaml_path)
    names = data.get('names')

    resolved = {}
    if isinstance(names, dict):
        for key, value in names.items():
            try:
                resolved[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, (list, tuple)):
        for index, value in enumerate(names):
            resolved[index] = str(value)

    return resolved


def read_class_mapping(mapping_path):
    """Load a class_mapping.json, returning {} rather than raising."""
    if not mapping_path or not os.path.isfile(mapping_path):
        return {}
    try:
        with open(mapping_path, 'r') as file:
            data = json.load(file)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def sidecar_spec(task):
    """Return the (folder name, accepted extensions) a task's sidecars use."""
    if task == 'semantic':
        return 'masks', MASK_EXTENSIONS
    return 'labels', ('.txt',)


def _dataset_root(data, yaml_dir):
    """Resolve a data.yaml's optional 'path' key, or None if it is unusable."""
    root = str(data.get('path') or '').strip()
    if not root:
        return None
    root = root.replace(chr(92), '/')
    if not os.path.isabs(root):
        root = os.path.join(yaml_dir, root)
    return root if os.path.isdir(root) else None


def _split_from_path(path):
    """Infer a split from a path's own segments, defaulting to 'train'.

    Used only by the fallback scan, where there is no YAML entry saying which
    split a discovered folder belongs to.
    """
    parts = os.path.normpath(path).replace(chr(92), '/').lower().split('/')
    for part in reversed(parts):
        if part in ('train', 'training'):
            return 'train'
        if part in ('val', 'valid', 'validation'):
            return 'valid'
        if part in ('test', 'testing'):
            return 'test'
    return 'train'


def discover_dataset_splits(yaml_path, task):
    """Map every image in a YOLO dataset to its sidecar, split by split.

    Mirrors ``ImportDataset.discover_dataset_files`` but keeps the split each
    image came from, which merging needs: train images from one dataset must not
    land in another dataset's validation split.

    Args:
        yaml_path (str): Path to the dataset's data.yaml.
        task (str): 'detect', 'segment' or 'semantic'.

    Returns:
        dict: {'train'|'valid'|'test': {image path: sidecar path or None}}.
    """
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    data = load_data_yaml(yaml_path)
    root = _dataset_root(data, yaml_dir)

    sidecar_name, sidecar_exts = sidecar_spec(task)
    if task == 'semantic':
        # Semantic datasets name their mask folder in the YAML; only the folder
        # name is used, the rest of the path comes from the image directory.
        configured = str(data.get('masks_dir') or 'masks').strip()
        configured = os.path.basename(configured.replace(chr(92), '/').rstrip('/'))
        sidecar_name = configured or 'masks'

    splits = {split: {} for split in OUTPUT_SPLITS}

    def collect(image_dir, split):
        """Record every image under one directory against its sidecar."""
        image_dir = _normalize_image_dir(image_dir)
        sidecar_dir = _sidecar_dir_for(image_dir, sidecar_name)
        found = False
        for image_path in _iter_images(image_dir, ()):
            normalized = os.path.normpath(image_path)
            # An image reachable through two entries keeps the first sidecar found.
            if normalized in splits[split] and splits[split][normalized]:
                continue
            splits[split][normalized] = _pair(image_path, image_dir, sidecar_dir, sidecar_exts)
            found = True
        return found

    resolved_any = False
    seen_dirs = set()
    for yaml_key, split in YAML_SPLIT_KEYS:
        value = data.get(yaml_key)
        if not value:
            continue
        entries = value if isinstance(value, (list, tuple)) else [value]
        for entry in entries:
            if not entry:
                continue
            image_dir = _resolve_image_dir(str(entry), root, yaml_dir)
            if not image_dir:
                continue
            key = (os.path.normcase(os.path.normpath(image_dir)), split)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            resolved_any = collect(image_dir, split) or resolved_any

    # Fallback: no split entry pointed anywhere real, so scan for folders named
    # 'images' and read the split off their own paths.
    if not resolved_any:
        for current_root, dir_names, _file_names in os.walk(yaml_dir):
            for dir_name in dir_names:
                if dir_name.lower() != 'images':
                    continue
                candidate = os.path.join(current_root, dir_name)
                collect(candidate, _split_from_path(candidate))

    return splits


# ----------------------------------------------------------------------------------------------------------------------
# Class Vocabulary
# ----------------------------------------------------------------------------------------------------------------------


def build_union_vocabulary(name_maps, background_first=False):
    """Fold several datasets' class lists into one, with a remap for each.

    Class identity in a YOLO dataset is a bare integer whose meaning lives only
    in that dataset's own data.yaml, so the same index routinely means different
    things in two datasets being merged. Names are the only stable identity, so
    the union is taken over names and every dataset gets a lookup table from its
    own indices to the merged ones.

    Args:
        name_maps (list): One {index: name} dict per dataset, in merge order.
        background_first (bool): Pin a class named 'background' to index 0, the
            convention semantic masks follow.

    Returns:
        tuple: (ordered names list, list of {old index: new index} per dataset).
    """
    ordered = []
    seen = set()

    for name_map in name_maps:
        for index in sorted(name_map):
            name = name_map[index]
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                ordered.append(name)

    if background_first:
        for position, name in enumerate(ordered):
            if name.strip().lower() == BACKGROUND_NAME:
                ordered.insert(0, ordered.pop(position))
                break

    name_to_index = {name.strip().lower(): index for index, name in enumerate(ordered)}

    luts = []
    for name_map in name_maps:
        lut = {}
        for index, name in name_map.items():
            new_index = name_to_index.get(name.strip().lower())
            if new_index is not None:
                lut[int(index)] = new_index
        luts.append(lut)

    return ordered, luts


def merge_class_mappings(mappings, names):
    """Union several class_mapping.json dicts, keeping the first definition.

    Entries for classes that survived into the merged vocabulary are kept as
    they were written, so label colors and long codes carry through the merge.
    """
    wanted = {name.strip().lower() for name in names}
    merged = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in merged and str(key).strip().lower() in wanted:
                merged[key] = value
    return merged


def write_data_yaml(output_dir_path, names, task):
    """Write the merged dataset's data.yaml.

    Keys are emitted in reading order -- the dataset's location and shape first,
    then the class list -- rather than the alphabetical order PyYAML defaults to,
    which would bury 'names' above 'nc' and scatter the splits.

    The exporters deliberately omit 'path' so their output survives being moved
    to another machine. A merge writes to a folder the user just chose here, so
    'path' is written out in full: it is the line that gets pasted into a
    training config. Readers fall back to the YAML's own directory when the path
    does not resolve, so moving the folder still works.
    """
    yaml_path = os.path.join(output_dir_path, 'data.yaml')

    data = {'path': os.path.abspath(output_dir_path)}

    if task == 'semantic':
        data['train'] = 'train/images'
        data['val'] = 'valid/images'
        data['test'] = 'test/images'
    else:
        data['train'] = 'train'
        data['val'] = 'valid'
        data['test'] = 'test'

    data['nc'] = len(names)

    if task == 'semantic':
        data['masks_dir'] = 'masks'

    data['names'] = {index: name for index, name in enumerate(names)}

    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    return yaml_path


# ----------------------------------------------------------------------------------------------------------------------
# Sidecar Remapping
# ----------------------------------------------------------------------------------------------------------------------


def unique_stem(used_stems, stem):
    """Return a filename stem not yet used in a destination folder.

    Detection, instance and semantic datasets all write their images into one
    flat folder per split, so two datasets that both contain 'image1.jpg' would
    otherwise clobber each other. Worse, the image and its sidecar could be
    overwritten by different sources and silently desynchronize, so the stem
    chosen here is used for both halves of the pair.
    """
    candidate = stem
    suffix = 1
    while candidate.lower() in used_stems:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used_stems.add(candidate.lower())
    return candidate


def remap_label_file(src_path, dst_path, lut):
    """Copy a YOLO .txt label file, rewriting each row's class index.

    Only the leading class token changes; the geometry that follows it is written
    back verbatim, so detection boxes and segmentation polygons are handled by
    the same code. Rows naming a class that did not survive into the merged
    vocabulary are dropped, which leaves a legitimately empty label file when
    none of them did.

    Returns:
        tuple: (rows kept, rows dropped).
    """
    kept = []
    dropped = 0

    with open(src_path, 'r') as file:
        for raw_line in file:
            parts = raw_line.split()
            if not parts:
                continue
            try:
                old_index = int(float(parts[0]))
            except ValueError:
                dropped += 1
                continue
            if old_index not in lut:
                dropped += 1
                continue
            parts[0] = str(lut[old_index])
            kept.append(' '.join(parts))

    with open(dst_path, 'w') as file:
        if kept:
            file.write('\n'.join(kept) + '\n')

    return len(kept), dropped


def build_mask_lut(lut, ignore_value=IGNORE_VALUE):
    """Build the 256-entry table that rewrites a uint8 mask's class IDs.

    Every value the merge has no mapping for becomes ignore rather than passing
    through unchanged: an unmapped pixel left alone would keep an index that now
    means a different class entirely, which is worse than dropping it.
    """
    table = np.full(256, ignore_value, dtype=np.uint8)
    for old_index, new_index in lut.items():
        if 0 <= old_index < 256:
            table[old_index] = new_index
    table[ignore_value] = ignore_value
    return table


def remap_mask_file(src_path, dst_path, table, ignore_value=IGNORE_VALUE):
    """Copy a semantic mask, rewriting the class ID each pixel carries.

    Args:
        src_path (str): Source mask, single-channel or paletted.
        dst_path (str): Destination .png.
        table (np.ndarray): 256-entry lookup from build_mask_lut.
        ignore_value (int): Value standing for ignore/unlabeled.

    Raises:
        ValueError: If the mask is multi-channel, where a pixel's class cannot be
            read without knowing the color coding its producer used.
    """
    with Image.open(src_path) as image:
        # Mode 'P' is read as palette indices rather than expanded to RGB, since
        # for a mask those indices are themselves the class IDs.
        array = np.array(image)

    if array.ndim != 2:
        raise ValueError(f"Mask is not single-channel: {src_path}")

    if array.dtype == np.uint8:
        remapped = table[array]
    else:
        # A 16-bit mask cannot index a 256-entry table, so the mapping is applied
        # class by class instead. Still narrows to uint8, which is what YOLO reads.
        remapped = np.full(array.shape, ignore_value, dtype=np.uint8)
        for old_index in np.unique(array):
            if 0 <= int(old_index) < 256:
                remapped[array == old_index] = table[int(old_index)]

    Image.fromarray(remapped, mode='L').save(dst_path)


def copy_image(src_path, dst_path):
    """Copy an image file, preserving its timestamps."""
    shutil.copy2(src_path, dst_path)
