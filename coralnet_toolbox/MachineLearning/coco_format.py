"""COCO annotation files: reading them, writing them, and the dialects that call themselves COCO.

Import and export both go through here. Reading always ends in one canonical
shape - the ``instances`` layout, with ``images``, ``annotations`` and
``categories`` lists - whatever the file looked like on disk, so the dialogs
only ever handle that shape.

A dialect is a file that labels itself COCO but does not follow the schema.
Each one gets a function that rewrites it into the official layout, and nothing
past that point knows it existed. Known so far:

    coco        the official schema, read tolerantly: string ids, missing
                iscrowd / area / bbox, 0-based or gapped category ids, and
                every segmentation encoding (polygons, compressed and
                uncompressed RLE) are all accepted.
    coralscop   CoralSCOP's "COCO" export: one file per image, a singular
                ``image`` object with ``image_filename`` / ``image_width`` /
                ``image_height``, compressed RLE masks with no bbox or area,
                and 0-based categories carrying a ``color``.

Writing produces the official schema only, strictly, since an export is meant
to be readable by anything that reads COCO.
"""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import ujson as json
from pycocotools import mask as mask_utils

from coralnet_toolbox.IO.QtImportImages import SUPPORTED_IMAGE_EXTENSIONS

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

DIALECT_COCO = "coco"
DIALECT_CORALSCOP = "coralscop"

# Holes smaller than this many pixels are filled when a mask is traced into
# polygons. Tracing every speckle in a model's mask turns one object into a
# polygon with dozens of tiny rings.
MIN_HOLE_AREA = 25

# Douglas-Peucker tolerance for traced outlines, the same one the mask
# vectorizer uses, so imported and vectorized polygons look alike.
TRACE_SIMPLIFY_PX = 1.0

# -------------------------------------------------------------------------------------------------
# Dialects
# -------------------------------------------------------------------------------------------------


def sniff_dialect(data):
    """Name the dialect a parsed annotation file is written in.

    Decided from the keys alone, never from a tool name inside the file: none of
    the dialects seen so far say who wrote them.

    Args:
        data: The parsed JSON.

    Returns:
        str: DIALECT_COCO or DIALECT_CORALSCOP.

    Raises:
        ValueError: When the file is not an annotation file of any known kind.
            A bare list gets its own message, because that is the shape of a
            COCO results file - predictions that name their images only by the
            ids of another file - and it cannot be read without that file.
    """
    if isinstance(data, list):
        raise ValueError("This is a COCO results file (a list of predictions). It refers to images "
                         "only by id, so it can't be imported without the dataset file it was made from.")
    if not isinstance(data, dict):
        raise ValueError("Not a COCO annotation file: the top level is not a JSON object.")

    if isinstance(data.get('images'), list):
        return DIALECT_COCO

    image = data.get('image')
    if isinstance(image, dict) and 'image_filename' in image:
        return DIALECT_CORALSCOP

    keys = ", ".join(sorted(str(key) for key in data)) or "none"
    raise ValueError(f"Not a recognised COCO annotation file (top-level keys: {keys}).")


def _from_coralscop(data):
    """Rewrite a CoralSCOP per-image file into the official layout.

    The file describes one image, so every annotation in it belongs to that
    image whatever its own image_id says. The missing bbox, area and iscrowd are
    left for _canonicalize to fill, the same as for any official file that
    leaves them out.
    """
    image = data['image']
    image_id = image.get('id', 0)

    return {
        'images': [{
            'id': image_id,
            'file_name': image.get('image_filename'),
            'width': image.get('image_width'),
            'height': image.get('image_height'),
        }],
        'annotations': [dict(annotation, image_id=image_id)
                        for annotation in data.get('annotations') or []
                        if isinstance(annotation, dict)],
        'categories': data.get('categories') or [],
    }


def normalize(data):
    """Rewrite a parsed annotation file of any known dialect into the canonical layout.

    Ids are renumbered from 1 - images, annotations and categories alike - and
    categories are merged by name, so the result never depends on a file's own
    numbering. Every annotation leaves with a compressed-RLE or polygon-list
    segmentation (or [] for a box alone), a bbox, an area and an iscrowd.

    Args:
        data: The parsed JSON.

    Returns:
        tuple: (dataset, problems). problems lists, as readable strings, what
        was skipped and why.

    Raises:
        ValueError: When the file is not an annotation file of any known kind.
    """
    if sniff_dialect(data) == DIALECT_CORALSCOP:
        data = _from_coralscop(data)
    return _canonicalize(data)


# -------------------------------------------------------------------------------------------------
# Canonical form
# -------------------------------------------------------------------------------------------------


def _id_key(value):
    """A file's own id as a lookup key. Some tools write ids as strings, and
    some write the same id as an int in one list and a string in another."""
    return str(value)


def _positive_int(value):
    """Return value as a positive int, or None when it is not one."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _canonicalize(data):
    """Validate and renumber a file already in the official layout."""
    problems = []

    # Categories, merged by name and numbered from 1. A file's own numbering
    # varies too much to keep: 0-based, gapped, or a placeholder at 0.
    categories = []
    category_ids = {}
    old_to_new_category = {}
    for entry in data.get('categories') or []:
        if not isinstance(entry, dict) or entry.get('name') in (None, ''):
            problems.append(f"Skipped a category with no name: {entry!r}")
            continue

        name = str(entry['name'])
        if name not in category_ids:
            category_ids[name] = len(categories) + 1
            category = {'id': category_ids[name],
                        'name': name,
                        'supercategory': str(entry.get('supercategory') or '')}
            color = entry.get('color')
            if isinstance(color, str) and color:
                category['color'] = color
            categories.append(category)
        old_to_new_category[_id_key(entry.get('id'))] = category_ids[name]

    images = []
    old_to_new_image = {}
    for entry in data.get('images') or []:
        if not isinstance(entry, dict) or not entry.get('file_name'):
            problems.append(f"Skipped an image with no file name: {entry!r}")
            continue

        key = _id_key(entry.get('id'))
        if key in old_to_new_image:
            problems.append(f"Skipped image '{entry['file_name']}': its id {entry.get('id')!r} is already used.")
            continue

        image = {'id': len(images) + 1,
                 'file_name': str(entry['file_name']),
                 'width': _positive_int(entry.get('width')),
                 'height': _positive_int(entry.get('height'))}
        old_to_new_image[key] = image
        images.append(image)

    annotations = []
    for entry in data.get('annotations') or []:
        if not isinstance(entry, dict):
            problems.append(f"Skipped an annotation that is not an object: {entry!r}")
            continue

        image = old_to_new_image.get(_id_key(entry.get('image_id')))
        if image is None:
            problems.append(f"Skipped annotation {entry.get('id')!r}: "
                            f"its image_id {entry.get('image_id')!r} is not in the file.")
            continue

        category_id = old_to_new_category.get(_id_key(entry.get('category_id')))
        if category_id is None:
            problems.append(f"Skipped annotation {entry.get('id')!r} on '{image['file_name']}': "
                            f"its category_id {entry.get('category_id')!r} is not in the file.")
            continue

        try:
            annotation = _clean_annotation(entry, image)
        except (TypeError, ValueError) as e:
            problems.append(f"Skipped annotation {entry.get('id')!r} on '{image['file_name']}': {e}")
            continue

        annotation['id'] = len(annotations) + 1
        annotation['image_id'] = image['id']
        annotation['category_id'] = category_id
        annotations.append(annotation)

    return {'images': images, 'annotations': annotations, 'categories': categories}, problems


def _clean_annotation(entry, image):
    """Return one annotation with its segmentation, bbox, area and iscrowd settled.

    The bbox in the file is kept when it is usable, since for a detection
    dataset it is the thing that was drawn. The area never is: tools fill it
    with the box area, the polygon area or nothing at all, so it is always
    recomputed from the geometry.

    Raises:
        ValueError: When the annotation has no usable geometry, or its mask
            does not fit its image.
    """
    segmentation = entry.get('segmentation')

    if isinstance(segmentation, dict):
        segmentation = _clean_rle(segmentation)
        mask_height, mask_width = segmentation['size']

        # A file that leaves the image size out still says it inside every RLE
        if image['height'] is None or image['width'] is None:
            image['height'], image['width'] = mask_height, mask_width
        elif (mask_height, mask_width) != (image['height'], image['width']):
            raise ValueError(f"its mask is {mask_width}x{mask_height}, "
                             f"but the image is {image['width']}x{image['height']}.")

        derived_bbox = [float(value) for value in mask_utils.toBbox(segmentation)]
        area = float(mask_utils.area(segmentation))
    else:
        segmentation = _flat_rings(segmentation)
        derived_bbox = _rings_bbox(segmentation)
        area = sum(_ring_area(ring) for ring in segmentation)

    bbox = _clean_bbox(entry.get('bbox'))
    if bbox is None:
        bbox = derived_bbox
    if bbox is None:
        raise ValueError("it has neither a bbox nor a segmentation.")

    if not segmentation:
        area = bbox[2] * bbox[3]

    annotation = {'segmentation': segmentation,
                  'bbox': bbox,
                  'area': float(area),
                  'iscrowd': 1 if entry.get('iscrowd') in (1, True, '1') else 0}

    # Kept so predictions saved as a dataset arrive with their confidence
    score = entry.get('score')
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        annotation['score'] = float(score)

    return annotation


def _clean_rle(segmentation):
    """Return an RLE as compressed RLE with string counts.

    Uncompressed RLE (a list of counts) is converted rather than passed on, so
    the rest of the module sees one RLE form. Its counts are checked against
    its size first: pycocotools trusts them and decodes a short list into a
    mask that silently stops early.

    Raises:
        ValueError: When the RLE has no valid size or counts.
    """
    size = segmentation.get('size')
    counts = segmentation.get('counts')
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"its RLE has no valid size: {size!r}")

    height, width = int(size[0]), int(size[1])
    if isinstance(counts, list):
        counts = [int(count) for count in counts]
        if sum(counts) != height * width:
            raise ValueError(f"its RLE counts cover {sum(counts)} pixels, but its size is "
                             f"{width}x{height} ({height * width} pixels).")
        rle = mask_utils.frPyObjects({'counts': counts, 'size': [height, width]}, height, width)
    elif isinstance(counts, (str, bytes)):
        rle = {'counts': counts, 'size': [height, width]}
    else:
        raise ValueError("its RLE has no counts.")

    return _compressed(rle)


def _compressed(rle):
    """An RLE dict with str counts, ready for JSON."""
    counts = rle['counts']
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    return {'size': [int(rle['size'][0]), int(rle['size'][1])], 'counts': counts}


def _flat_rings(segmentation):
    """Return a polygon segmentation as a list of flat [x1, y1, x2, y2, ...] rings.

    Rings with fewer than three points, or an odd number of coordinates, are
    dropped: they enclose nothing. The three-point floor also matters to
    pycocotools, whose frPyObjects decides a four-number list is a bounding box
    from its length alone. A single flat list of numbers, which some tools
    write for a one-part polygon, is read as one ring.
    """
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return []

    if all(isinstance(value, (int, float)) for value in segmentation):
        segmentation = [segmentation]

    rings = []
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)):
            continue
        try:
            coords = [float(value) for value in ring]
        except (TypeError, ValueError):
            continue
        if len(coords) >= 6 and len(coords) % 2 == 0:
            rings.append(coords)
    return rings


def _rings_bbox(rings):
    """Return the [x, y, w, h] around flat rings, or None when there are none."""
    if not rings:
        return None
    xs = [x for ring in rings for x in ring[0::2]]
    ys = [y for ring in rings for y in ring[1::2]]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _ring_area(ring):
    """Shoelace area of one flat ring."""
    xs = np.asarray(ring[0::2], dtype=np.float64)
    ys = np.asarray(ring[1::2], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def _clean_bbox(value):
    """Return value as a float [x, y, w, h] with a positive size, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(number) for number in value]
    except (TypeError, ValueError):
        return None
    return bbox if bbox[2] > 0 and bbox[3] > 0 else None


def merge_datasets(datasets):
    """Merge canonical datasets into one, renumbering every id.

    Needed as soon as there is more than one file: CoralSCOP writes one file
    per image and every one of them calls its image 0, and two official split
    files each number from 1. Categories merge by name. Images merge by the
    resolved 'path' when they have one, so an image described by two files ends
    up as one image holding both files' annotations.

    Args:
        datasets (list): Canonical datasets, as normalize returns them.

    Returns:
        dict: One canonical dataset.
    """
    merged = {'images': [], 'annotations': [], 'categories': []}
    category_ids = {}
    images_by_path = {}

    for dataset in datasets:
        category_map = {}
        for category in dataset['categories']:
            name = category['name']
            if name not in category_ids:
                category_ids[name] = len(merged['categories']) + 1
                merged['categories'].append(dict(category, id=category_ids[name]))
            category_map[category['id']] = category_ids[name]

        image_map = {}
        for image in dataset['images']:
            path = image.get('path')
            if path is not None and path in images_by_path:
                image_map[image['id']] = images_by_path[path]
                continue

            image_map[image['id']] = len(merged['images']) + 1
            merged['images'].append(dict(image, id=image_map[image['id']]))
            if path is not None:
                images_by_path[path] = image_map[image['id']]

        for annotation in dataset['annotations']:
            merged['annotations'].append(dict(annotation,
                                              id=len(merged['annotations']) + 1,
                                              image_id=image_map[annotation['image_id']],
                                              category_id=category_map[annotation['category_id']]))

    return merged


def category_counts(dataset):
    """Return {category id: annotation count} for every category, zeros included.

    An importer hides the zero rows: a category nothing uses is noise at best,
    and at worst a placeholder such as the dataset-named category Roboflow
    puts at id 0.
    """
    counts = {category['id']: 0 for category in dataset['categories']}
    for annotation in dataset['annotations']:
        counts[annotation['category_id']] = counts.get(annotation['category_id'], 0) + 1
    return counts


# -------------------------------------------------------------------------------------------------
# Reading files
# -------------------------------------------------------------------------------------------------


def find_annotation_files(path):
    """Return the annotation files a picked path stands for.

    A file is itself. A folder is every .json directly inside it, which is how
    a per-image export such as CoralSCOP's arrives.
    """
    if os.path.isdir(path):
        return sorted(os.path.join(path, name) for name in os.listdir(path)
                      if name.lower().endswith('.json'))
    return [path]


def image_path_candidates(file_name, json_path, image_root=None):
    """Return, in order of preference, the paths an image's file_name may refer to.

    COCO says nothing about where the images are, and every tool puts them
    somewhere else relative to the JSON:

        official    annotations/instances_train2017.json   beside train2017/
        CVAT        annotations/instances_default.json     beside images/
        Roboflow    train/_annotations.coco.json           among its images
        FiftyOne    labels.json                            beside data/
        CoralSCOP   <image>.json                           among its images, or nowhere

    file_name may also carry its own subfolders, or be absolute. Each root is
    tried with the name as given and with its base name alone. A folder the
    user picked comes first.

    Args:
        file_name (str): The image's file_name from the JSON.
        json_path (str): The JSON the image was listed in.
        image_root (str): Folder the user picked for the images, if any.

    Returns:
        list: Candidate paths, without duplicates.
    """
    name = str(file_name).replace('\\', '/')
    base = os.path.basename(name)
    json_dir = os.path.dirname(os.path.abspath(json_path))
    parent = os.path.dirname(json_dir)
    stem = os.path.splitext(os.path.basename(json_path))[0]

    roots = [image_root] if image_root else []
    roots += [json_dir,
              os.path.join(json_dir, 'images'),
              os.path.join(json_dir, 'data'),
              parent,
              os.path.join(parent, 'images'),
              os.path.join(parent, 'data')]
    if stem.startswith('instances_'):
        roots.append(os.path.join(parent, stem[len('instances_'):]))

    candidates = [name] if os.path.isabs(name) else []
    for root in roots:
        candidates.append(os.path.join(root, name))
        candidates.append(os.path.join(root, base))

    return list(dict.fromkeys(os.path.normpath(candidate) for candidate in candidates))


def resolve_image_path(file_name, json_path, image_root=None):
    """Return the first candidate image path that exists, or None.

    Forward slashes, the form a raster is registered under everywhere else in
    the application (see discover_dataset_files).
    """
    for candidate in image_path_candidates(file_name, json_path, image_root):
        if os.path.isfile(candidate):
            return candidate.replace('\\', '/')
    return None


def _resolve_by_json_stem(json_path, image_root=None):
    """Find an image named like its JSON, for a one-image file whose file_name
    no longer matches the image beside it."""
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS):
        path = resolve_image_path(stem + extension, json_path, image_root)
        if path is not None:
            return path
    return None


def load_annotation_files(json_paths, image_root=None):
    """Read annotation files of any known dialect into one canonical dataset.

    A file that cannot be read is reported and skipped rather than failing the
    rest: a folder of per-image files should not be lost to one bad file, nor
    to an unrelated .json sitting beside them.

    Args:
        json_paths (list): Annotation files, as find_annotation_files returns them.
        image_root (str): Folder the user picked for the images, if any.

    Returns:
        tuple: (dataset, problems). Each image carries 'path', the image file
        it resolved to, or None when nothing on disk matched. The dataset also
        carries 'dialects', the sorted names of the dialects that were read.
    """
    datasets = []
    problems = []
    dialects = set()

    for json_path in json_paths:
        file_label = os.path.basename(json_path)
        try:
            with open(json_path, encoding='utf-8') as json_file:
                data = json.load(json_file)
            dialect = sniff_dialect(data)
            dataset, file_problems = normalize(data)
        except (OSError, ValueError) as e:
            problems.append(f"{file_label}: {e}")
            continue

        dialects.add(dialect)

        problems.extend(f"{file_label}: {problem}" for problem in file_problems)

        for image in dataset['images']:
            image['path'] = resolve_image_path(image['file_name'], json_path, image_root)

        if len(dataset['images']) == 1 and dataset['images'][0]['path'] is None:
            dataset['images'][0]['path'] = _resolve_by_json_stem(json_path, image_root)

        for image in dataset['images']:
            if image['path'] is None:
                problems.append(f"{file_label}: image '{image['file_name']}' was not found.")

        datasets.append(dataset)

    merged = merge_datasets(datasets)
    merged['dialects'] = sorted(dialects)
    return merged, problems


# -------------------------------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------------------------------


def polygon_rings(segmentation):
    """Return a polygon segmentation's rings as lists of (x, y) points.

    An RLE segmentation has no rings and returns []; decode it with
    annotation_mask instead. Each ring is a separate part of the object - COCO
    polygons carry no holes.
    """
    if isinstance(segmentation, dict):
        return []
    return [list(zip(ring[0::2], ring[1::2], strict=True)) for ring in _flat_rings(segmentation)]


def segmentation_to_rle(segmentation, height, width):
    """Return any segmentation as one compressed RLE, or None when it has none.

    Args:
        segmentation: Polygon rings, or an RLE dict of either kind.
        height (int): Image height, used to rasterize polygons.
        width (int): Image width, used to rasterize polygons.
    """
    if isinstance(segmentation, dict):
        return _clean_rle(segmentation)

    rings = _flat_rings(segmentation)
    if not rings:
        return None
    return _compressed(mask_utils.merge(mask_utils.frPyObjects(rings, height, width)))


def annotation_mask(annotation, height, width):
    """Decode an annotation's segmentation to a boolean (height, width) mask.

    Returns None for an annotation with a box alone.
    """
    rle = segmentation_to_rle(annotation.get('segmentation'), height, width)
    if rle is None:
        return None
    return mask_utils.decode(rle).astype(bool)


def encode_mask(mask, image_size=None, offset=(0, 0)):
    """Return a boolean mask as compressed RLE, optionally as a crop placed in a larger image.

    RLE always covers the whole image, but building a whole-image array for
    every object is what makes an orthomosaic export run out of memory. A crop
    and its offset are enough: columns outside the crop are all background, so
    only the crop's own columns are built, each at the image's full height.

    Args:
        mask (np.ndarray): Boolean mask, either the whole image or a crop of it.
        image_size (tuple): (height, width) of the image; None when mask is the
            whole image.
        offset (tuple): (x, y) of the crop's top-left corner in the image. Parts
            of the crop outside the image are clipped away.

    Returns:
        dict: Compressed RLE with str counts.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = image_size if image_size is not None else mask.shape
    x0, y0 = int(offset[0]), int(offset[1])

    if mask.shape == (height, width) and (x0, y0) == (0, 0):
        # Column-major, as pycocotools wants: cv2.transpose makes the copy ~12x
        # faster than np.asfortranarray
        return _compressed(mask_utils.encode(cv2.transpose(mask.view(np.uint8)).T))

    # Clip the crop to the image
    left, top = max(0, -x0), max(0, -y0)
    right = min(mask.shape[1], width - x0)
    bottom = min(mask.shape[0], height - y0)
    if right <= left or bottom <= top:
        counts = [height * width]
    else:
        mask = mask[top:bottom, left:right]
        x0, y0 = x0 + left, y0 + top

        strip = np.zeros((height, mask.shape[1]), dtype=np.uint8)
        strip[y0:y0 + mask.shape[0]] = mask
        flat = cv2.transpose(strip).ravel()  # column order, ~12x faster than ravel(order='F')

        # Run lengths, starting with a background run as RLE requires
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
        runs = np.diff(bounds).tolist()
        if flat[0]:
            runs.insert(0, 0)

        # Columns left of the crop extend the first run, which is background.
        # Columns right of it extend the last run if that is background too.
        runs[0] += x0 * height
        trailing = (width - x0 - mask.shape[1]) * height
        if len(runs) % 2 == 1:
            runs[-1] += trailing
        elif trailing:
            runs.append(trailing)
        counts = runs

    rle = mask_utils.frPyObjects({'counts': counts, 'size': [height, width]}, height, width)
    return _compressed(rle)


def _rasterize_rings(rings, holes, image_size):
    """Rasterize flat rings, minus flat holes, into a crop of the image.

    Each ring is filled on its own: handed to fillPoly together, overlapping
    rings would cancel where they overlap.

    Returns:
        tuple: (mask crop, (x, y) offset), or (None, None) when the rings lie
        outside the image.
    """
    height, width = image_size
    xs = [x for ring in rings for x in ring[0::2]]
    ys = [y for ring in rings for y in ring[1::2]]
    x0, y0 = max(int(np.floor(min(xs))), 0), max(int(np.floor(min(ys))), 0)
    x1, y1 = min(int(np.ceil(max(xs))) + 1, width), min(int(np.ceil(max(ys))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None, None

    def to_crop(ring):
        return np.round(np.asarray(ring, dtype=np.float64).reshape(-1, 2) - (x0, y0)).astype(np.int32)

    crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for ring in rings:
        cv2.fillPoly(crop, [to_crop(ring)], 1)
    for hole in holes:
        cv2.fillPoly(crop, [to_crop(hole)], 0)
    return crop.astype(bool), (x0, y0)


def _rle_counts(rle):
    """Return a compressed RLE's run lengths as a list of ints.

    The inverse of pycocotools' string encoding (rleFrString in maskApi.c),
    needed because pycocotools only decodes to a whole-image array. Each run is
    a little-endian sequence of 5-bit groups offset by 48: bit 0x20 says
    another group follows, and bit 0x10 of the last group is the sign. From the
    third run on, each value is stored as a delta from the run two back.
    """
    counts = rle['counts']
    if isinstance(counts, list):
        return counts
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')

    values = []
    position = 0
    while position < len(counts):
        value, shift, more = 0, 0, True
        while more:
            char = ord(counts[position]) - 48
            value |= (char & 0x1f) << shift
            more = bool(char & 0x20)
            position += 1
            shift += 5
            if not more and char & 0x10:
                value |= -1 << shift
        if len(values) > 2:
            value += values[-2]
        values.append(value)
    return values


def _rle_crop(rle, x0, y0, x1, y1):
    """Decode only the [x0, x1) x [y0, y1) window of an RLE mask.

    RLE runs down whole columns, so the window's columns are built at the full
    image height and then cut to its rows: memory follows the object's width,
    not the image's area.
    """
    height = int(rle['size'][0])
    counts = np.asarray(_rle_counts(rle), dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts

    # Odd runs are foreground; keep the parts inside the window's columns
    low, high = x0 * height, x1 * height
    fg_starts, fg_ends = starts[1::2], ends[1::2]
    inside = (fg_ends > low) & (fg_starts < high)

    # +1 where a run starts and -1 where it ends; the running sum is the mask
    edges = np.zeros(high - low + 1, dtype=np.int8)
    np.add.at(edges, np.clip(fg_starts[inside], low, high) - low, 1)
    np.add.at(edges, np.clip(fg_ends[inside], low, high) - low, -1)
    columns = np.cumsum(edges[:-1], dtype=np.int8).astype(bool).reshape(x1 - x0, height)
    # Row-major, so painting and tracing the crop do not walk it column by column
    return cv2.transpose(columns[:, y0:y1].view(np.uint8)).view(bool)


def annotation_crop(annotation, height=None, width=None):
    """Rasterize an annotation into the smallest crop of its image that holds it.

    An RLE mask is decoded only inside its own bounding box, so an object on an
    orthomosaic costs its own size rather than the mosaic's.

    Args:
        annotation (dict): A canonical annotation.
        height (int): Image height. Needed for polygons and boxes, which are
            clipped to the image; an RLE carries its own size.
        width (int): Image width, likewise.

    Returns:
        tuple: (boolean crop, (x, y) offset), or (None, None) when the
        annotation covers nothing inside the image.
    """
    segmentation = annotation.get('segmentation')
    if isinstance(segmentation, dict):
        rle = _clean_rle(segmentation)
        x, y, w, h = (float(value) for value in mask_utils.toBbox(rle))
        if w <= 0 or h <= 0:
            return None, None
        x0, y0 = int(x), int(y)
        x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
        return _rle_crop(rle, x0, y0, x1, y1), (x0, y0)

    rings = _flat_rings(segmentation)
    if rings:
        return _rasterize_rings(rings, [], (height, width))

    bbox = _clip_bbox(annotation['bbox'], (height, width))
    x0, y0 = int(np.floor(bbox[0])), int(np.floor(bbox[1]))
    x1, y1 = int(np.ceil(bbox[0] + bbox[2])), int(np.ceil(bbox[1] + bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return None, None
    return np.ones((y1 - y0, x1 - x0), dtype=bool), (x0, y0)


def _trace_crop(crop, offset, min_hole_area):
    """Trace a crop's regions into (exterior, holes) polygons in image coordinates.

    RETR_CCOMP gives exactly two levels - every region's outline, then the
    holes inside it - and an island inside a hole comes back as a region of its
    own. Outlines are simplified as the mask vectorizer simplifies them.
    """
    contours, hierarchy = cv2.findContours(crop.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    x0, y0 = offset

    def to_points(contour):
        simplified = cv2.approxPolyDP(contour, TRACE_SIMPLIFY_PX, True).reshape(-1, 2)
        return [(float(x + x0), float(y + y0)) for x, y in simplified.tolist()]

    parts = []
    for index, contour in enumerate(contours):
        # A hole; collected below, under the region it belongs to
        if hierarchy[index][3] != -1:
            continue

        exterior = to_points(contour)
        if len(exterior) < 3:
            continue

        holes = []
        child = hierarchy[index][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_hole_area:
                hole = to_points(contours[child])
                if len(hole) >= 3:
                    holes.append(hole)
            child = hierarchy[child][0]

        parts.append((exterior, holes))
    return parts


def annotation_polygons(annotation, min_hole_area=MIN_HOLE_AREA):
    """Return an annotation as polygons, each an (exterior, holes) pair of (x, y) rings.

    A polygon segmentation gives one polygon per ring: COCO splits an object
    into parts that way, and has no holes to carry. An RLE mask is traced, so
    each of its separate regions becomes a polygon and its holes survive. A box
    alone becomes its four corners.

    Args:
        annotation (dict): A canonical annotation.
        min_hole_area (int): Holes in a traced mask smaller than this many
            pixels are filled rather than kept.
    """
    segmentation = annotation.get('segmentation')
    if isinstance(segmentation, dict):
        crop, offset = annotation_crop(annotation)
        if crop is None:
            return []
        return _trace_crop(crop, offset, min_hole_area)

    rings = polygon_rings(segmentation)
    if rings:
        return [(ring, []) for ring in rings]

    x, y, w, h = annotation['bbox']
    return [([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], [])]


def paint_class_mask(items, height, width):
    """Paint annotations into one class-value mask, for semantic import.

    Larger objects are painted first, so where two overlap the smaller one
    shows: otherwise a colony drawn after the small objects on it would bury
    them.

    Args:
        items (list): (value, annotation) pairs; value is what the annotation's
            pixels are set to, typically its category id.
        height (int): Image height.
        width (int): Image width.

    Returns:
        np.ndarray: (height, width) uint16 mask, 0 where nothing was painted.
    """
    class_mask = np.zeros((height, width), dtype=np.uint16)
    for value, annotation in sorted(items, key=lambda item: item[1].get('area', 0), reverse=True):
        crop, (x0, y0) = annotation_crop(annotation, height, width)
        if crop is None:
            continue
        region = class_mask[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]]
        region[crop[:region.shape[0], :region.shape[1]]] = value
    return class_mask


def _clip_bbox(bbox, image_size):
    """Clip an [x, y, w, h] box to the image."""
    height, width = image_size
    x, y, w, h = bbox
    left, top = min(max(x, 0.0), width), min(max(y, 0.0), height)
    right, bottom = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    return [float(left), float(top), float(right - left), float(bottom - top)]


def _flatten_points(points):
    """Return (x, y) points as one flat list of floats."""
    return [float(value) for point in points for value in point]


# -------------------------------------------------------------------------------------------------
# Writing
# -------------------------------------------------------------------------------------------------


class CocoWriter:
    """Builds one COCO instances file in the official schema.

    Ids start at 1 for images, annotations and categories alike. For
    annotations that is more than convention: COCOeval records a match by
    storing the matched annotation's id and reads 0 as "no match", so a ground
    truth with id 0 can never be matched.

    Every annotation is written with iscrowd 0, its area, and a bbox clipped to
    its image. A polygon with holes is written as RLE, since a COCO polygon
    cannot hold one.
    """

    def __init__(self, category_names, colors=None, description="Exported from CoralNet-Toolbox"):
        """
        Args:
            category_names (list): Class names, in the order their ids are given.
            colors (dict): Optional {class name: '#rrggbb'}, written as each
                category's 'color'. Not part of the schema; readers ignore it.
            description (str): The file's info.description.
        """
        colors = colors or {}
        self.description = description
        self.categories = []
        self.images = []
        self.annotations = []
        self._category_ids = {}
        self._image_sizes = {}

        for name in category_names:
            if name in self._category_ids:
                continue
            category = {'id': len(self.categories) + 1, 'name': name, 'supercategory': ''}
            if colors.get(name):
                category['color'] = colors[name]
            self._category_ids[name] = category['id']
            self.categories.append(category)

    def add_image(self, file_name, width, height):
        """Add an image and return its id."""
        image_id = len(self.images) + 1
        self.images.append({'id': image_id,
                            'file_name': file_name,
                            'width': int(width),
                            'height': int(height)})
        self._image_sizes[image_id] = (int(height), int(width))
        return image_id

    def add_box(self, image_id, category, bbox, score=None):
        """Add a box-only annotation from an [x, y, w, h] box.

        Returns:
            int: The annotation id, or None when the box falls outside the image.
        """
        bbox = _clip_bbox(bbox, self._image_sizes[image_id])
        if bbox[2] <= 0 or bbox[3] <= 0:
            return None
        return self._add(image_id, category, [], bbox, bbox[2] * bbox[3], score)

    def add_polygon(self, image_id, category, exterior, holes=(), score=None):
        """Add a polygon annotation from (x, y) points.

        Returns:
            int: The annotation id, or None when the polygon is degenerate or
            falls outside the image.
        """
        image_size = self._image_sizes[image_id]
        exterior = _flatten_points(exterior)
        holes = [ring for ring in (_flatten_points(hole) for hole in holes) if len(ring) >= 6]
        if len(exterior) < 6:
            return None

        if holes:
            # A COCO polygon has no holes; a mask is the only form that keeps them
            crop, offset = _rasterize_rings([exterior], holes, image_size)
            if crop is None:
                return None
            return self.add_mask(image_id, category, crop, offset, score)

        bbox = _clip_bbox(_rings_bbox([exterior]), image_size)
        if bbox[2] <= 0 or bbox[3] <= 0:
            return None
        return self._add(image_id, category, [exterior], bbox, _ring_area(exterior), score)

    def add_mask(self, image_id, category, mask, offset=(0, 0), score=None):
        """Add a mask annotation from a boolean mask - the whole image, or a
        crop of it at offset.

        Returns:
            int: The annotation id, or None when the mask is empty.
        """
        rle = encode_mask(mask, self._image_sizes[image_id], offset)
        area = float(mask_utils.area(rle))
        if area == 0:
            return None
        bbox = [float(value) for value in mask_utils.toBbox(rle)]
        return self._add(image_id, category, rle, bbox, area, score)

    def _add(self, image_id, category, segmentation, bbox, area, score):
        annotation = {'id': len(self.annotations) + 1,
                      'image_id': image_id,
                      'category_id': self._category_ids[category],
                      'segmentation': segmentation,
                      'area': float(area),
                      'bbox': bbox,
                      'iscrowd': 0}
        if score is not None:
            annotation['score'] = float(score)
        self.annotations.append(annotation)
        return annotation['id']

    def to_dict(self):
        """Return the whole file as a dict."""
        return {'info': {'description': self.description,
                         'version': '1.0',
                         'date_created': datetime.now().isoformat(timespec='seconds')},
                'licenses': [],
                'images': self.images,
                'annotations': self.annotations,
                'categories': self.categories}

    def write(self, path):
        """Write the file, creating its folder if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as json_file:
            json.dump(self.to_dict(), json_file)
