"""Bookkeeping for Extract Work Areas: which images came from which work areas.

A *set* is the images one Extract and Import made from one parent raster. The
parent keeps the set's record in `raster.tile_sets`; each image keeps
`raster.tile_of`, pointing back at its parent and where in it the image came
from. Plain Extract writes files and a manifest but creates no set, so it
never ties the parent down.

A parent with an active set is left out of Active Learning and Export, since
its images already carry the same pixels.

No Qt here: rasters are duck-typed, so this can be tested without the app.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

MANIFEST_FILENAME = "extracted_work_areas.json"
MANIFEST_VERSION = 1

# Raster types work areas can be extracted from; videos are excluded
SOURCE_RASTER_TYPES = ("ImageRaster", "OrthoRaster")

# Reasons a raster cannot be extracted, as shown in menus and the dialog
REASON_NOT_IMAGE = "video"
REASON_EXTRACTED_IMAGE = "extracted image"
REASON_ALREADY_EXTRACTED = "already extracted"
REASON_NO_WORK_AREAS = "no work areas"

# <prefix>_<x>_<y>_<width>_<height>
_TILE_STEM = re.compile(r"^(?P<prefix>.*)_(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)_(?P<h>\d+)$")


# ----------------------------------------------------------------------------------------------------------------------
# Paths and rectangles
# ----------------------------------------------------------------------------------------------------------------------


def normalize_path(path):
    """Forward slashes, the form every raster is registered under."""
    return path.replace("\\", "/")


def work_area_rects(work_areas, image_width, image_height):
    """Integer (x, y, width, height) per work area, as the pixels will be read.

    Truncated to whole pixels the way work_area_to_numpy builds its window,
    clipped to the image, with empty and repeated rectangles dropped. Order
    follows the work areas.
    """
    rects = []
    seen = set()
    for work_area in work_areas:
        rect = getattr(work_area, 'rect', work_area)
        x, y = int(rect.x()), int(rect.y())
        right = min(image_width, x + int(rect.width()))
        bottom = min(image_height, y + int(rect.height()))
        x, y = max(0, x), max(0, y)
        if right <= x or bottom <= y:
            continue
        clipped = (x, y, right - x, bottom - y)
        if clipped in seen:
            continue
        seen.add(clipped)
        rects.append(clipped)
    return rects


def overlap_summary(rects):
    """(how many rectangles overlap another, largest overlap in pixels).

    The overlap between two rectangles is the narrow side of their
    intersection, which for neighbouring grid tiles is the overlap the grid
    was made with.
    """
    order = sorted(range(len(rects)), key=lambda i: rects[i][0])
    overlapping = set()
    max_overlap = 0
    for position, i in enumerate(order):
        ax, ay, aw, ah = rects[i]
        for j in order[position + 1:]:
            bx, by, bw, bh = rects[j]
            if bx >= ax + aw:
                break  # sorted by x: nothing further right can touch this one
            overlap_width = min(ax + aw, bx + bw) - max(ax, bx)
            overlap_height = min(ay + ah, by + bh) - max(ay, by)
            if overlap_width > 0 and overlap_height > 0:
                overlapping.update((i, j))
                max_overlap = max(max_overlap, min(overlap_width, overlap_height))
    return len(overlapping), max_overlap


def tile_filename(prefix, rect, extension):
    """<prefix>_<x>_<y>_<width>_<height>.<extension>

    The size is part of the name because hand-drawn work areas can share a
    top-left corner.
    """
    x, y, width, height = rect
    return f"{prefix}_{x}_{y}_{width}_{height}.{extension.lstrip('.')}"


def parse_tile_filename(path):
    """(prefix, (x, y, width, height)) from a tile filename, or None if it is not one."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _TILE_STEM.match(stem)
    if not match:
        return None
    rect = tuple(int(match.group(key)) for key in ('x', 'y', 'w', 'h'))
    return match.group('prefix'), rect


# ----------------------------------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------------------------------


def new_set_id():
    return uuid.uuid4().hex


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def make_tile_of(parent_path, set_id, rect):
    """The record an extracted image keeps about where it came from."""
    x, y, width, height = rect
    return {'parent_path': parent_path, 'set_id': set_id,
            'x': x, 'y': y, 'width': width, 'height': height}


def make_set_record(set_id, output_dir, manifest_path, tile_paths, annotations_included,
                    has_overlap, max_overlap_px, copies=None):
    """The record a parent keeps about one set of extracted images.

    `copies` maps each copied annotation's id to [origin_id, tile_path], so
    Merge Back can tell a copy deleted on its image from one never made.
    """
    return {
        'set_id': set_id,
        'output_dir': output_dir,
        'manifest_path': manifest_path,
        'created': _now(),
        'tile_paths': list(tile_paths),
        'annotations_included': bool(annotations_included),
        'has_overlap': bool(has_overlap),
        'max_overlap_px': int(max_overlap_px),
        'copies': dict(copies or {}),
    }


def make_manifest(parent_path, parent_width, parent_height, crs_wkt, transform, set_id,
                  tiles, image_format, annotations_included, has_overlap, max_overlap_px):
    """Manifest written next to the images, so a set can be found again without the project.

    Args:
        transform: the parent's affine transform as 6 numbers (a, b, c, d, e, f), or None.
        tiles: list of {'path', 'x', 'y', 'width', 'height'}.
        set_id: None for a plain Extract, which creates no set.
    """
    return {
        'version': MANIFEST_VERSION,
        'parent_path': parent_path,
        'parent_width': parent_width,
        'parent_height': parent_height,
        'crs': crs_wkt,
        'transform': list(transform) if transform is not None else None,
        'set_id': set_id,
        'created': _now(),
        'image_format': image_format,
        'annotations_included': bool(annotations_included),
        'has_overlap': bool(has_overlap),
        'max_overlap_px': int(max_overlap_px),
        'tiles': list(tiles),
    }


def write_manifest(path, manifest):
    """Write a manifest, replacing any previous one only once the new one is complete."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_path, path)


def read_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# Raster state
# ----------------------------------------------------------------------------------------------------------------------


def is_extracted_image(raster):
    return bool(getattr(raster, 'tile_of', None))


def active_set(raster):
    """The parent's active set record, or None. There is at most one."""
    sets = getattr(raster, 'tile_sets', None) or []
    return sets[0] if sets else None


def has_active_set(raster):
    return active_set(raster) is not None


def extraction_block_reason(raster):
    """Why work areas cannot be extracted from this raster, or None when they can.

    A parent with an active set is blocked too, files only or not: merge back
    or unlink first, so there is only ever one set of images per raster.
    """
    if getattr(raster, 'raster_type', None) not in SOURCE_RASTER_TYPES:
        return REASON_NOT_IMAGE
    if is_extracted_image(raster):
        return REASON_EXTRACTED_IMAGE
    if has_active_set(raster):
        return REASON_ALREADY_EXTRACTED
    if not raster.has_work_areas():
        return REASON_NO_WORK_AREAS
    return None


def find_set(raster_manager, path):
    """(parent_path, set_record) for a parent with an active set or one of its images, else (None, None)."""
    raster = raster_manager.get_raster(path)
    if raster is None:
        return None, None

    record = active_set(raster)
    if record is not None:
        return raster.image_path, record

    tile_of = getattr(raster, 'tile_of', None)
    if tile_of:
        parent = raster_manager.get_raster(tile_of['parent_path'])
        if parent is not None:
            for record in getattr(parent, 'tile_sets', None) or []:
                if record['set_id'] == tile_of['set_id']:
                    return parent.image_path, record
    return None, None


def tile_paths_in_project(raster_manager, record):
    return [path for path in record['tile_paths'] if raster_manager.has_image_path(path)]


def unlink_set(raster_manager, parent_path, set_id):
    """Dissolve a set: the images become independent and the parent is released.

    Returns:
        list: paths of the rasters that changed.
    """
    changed = []
    tile_paths = []

    parent = raster_manager.get_raster(parent_path)
    if parent is not None:
        for record in list(getattr(parent, 'tile_sets', None) or []):
            if record['set_id'] == set_id:
                tile_paths = record['tile_paths']
                parent.tile_sets.remove(record)
                changed.append(parent.image_path)

    for path in tile_paths:
        tile = raster_manager.get_raster(path)
        if tile is not None and (getattr(tile, 'tile_of', None) or {}).get('set_id') == set_id:
            tile.tile_of = None
            changed.append(path)

    return changed


def forget_raster(raster_manager, path, raster):
    """Keep sets consistent once a raster has left the project.

    A parent leaving makes its images independent. An image leaving is
    dropped from its set, and the last one out dissolves the set, releasing
    the parent.

    Returns:
        list: paths of the remaining rasters that changed.
    """
    changed = []

    for record in list(getattr(raster, 'tile_sets', None) or []):
        for tile_path in record['tile_paths']:
            tile = raster_manager.get_raster(tile_path)
            if tile is not None and (getattr(tile, 'tile_of', None) or {}).get('set_id') == record['set_id']:
                tile.tile_of = None
                changed.append(tile_path)

    tile_of = getattr(raster, 'tile_of', None)
    if tile_of:
        parent = raster_manager.get_raster(tile_of['parent_path'])
        if parent is not None:
            for record in list(getattr(parent, 'tile_sets', None) or []):
                if record['set_id'] != tile_of['set_id']:
                    continue
                record['tile_paths'] = [p for p in record['tile_paths'] if p != path]
                if not tile_paths_in_project(raster_manager, record):
                    parent.tile_sets.remove(record)
                changed.append(parent.image_path)

    return changed
