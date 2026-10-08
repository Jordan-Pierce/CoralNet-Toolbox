"""Clip annotation dicts to a rectangle and shift them between coordinate frames.

Works on the dicts produced by `Annotation.to_dict()` and consumed by the
subclasses' `from_dict()`: the same route Extract Frames uses to clone
annotations, kept free of Qt so it can be tested without a scene.

Clipping and shifting are separate steps. To copy a parent annotation onto a
tile at (x, y): clip to the tile's rectangle in parent coordinates, then shift
by (-x, -y). Merging back is the shift by (+x, +y).

Returned dicts are deep copies. That matters: `to_dict()` hands out the live
annotation's own `data` and `metadata` dicts, so a shallow copy would let an
edit on the clone (e.g. stamping an origin id) change the original too.

Ids are left as they are. Callers set `id`, `image_path` and any provenance
in `data`; a polygon split into parts gets part dicts without an `id`, so each
part is given a fresh one when it is rebuilt.
"""

import copy
import hashlib
import json

from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Class names, as used by Extract Frames' type map and type(annotation).__name__
PATCH = "PatchAnnotation"
RECTANGLE = "RectangleAnnotation"
POLYGON = "PolygonAnnotation"
MULTIPOLYGON = "MultiPolygonAnnotation"

SUPPORTED_TYPES = (PATCH, RECTANGLE, POLYGON, MULTIPOLYGON)

# Pieces smaller than this (square pixels) are dropped when clipping, so a
# shape that only grazes a tile does not leave a sliver annotation behind.
DEFAULT_MIN_AREA = 4.0

# Keys holding geometry; everything else is carried over unchanged
_GEOMETRY_KEYS = ('points', 'holes', 'polygons', 'top_left', 'bottom_right', 'center_xy', 'annotation_size')


# ----------------------------------------------------------------------------------------------------------------------
# Clipping
# ----------------------------------------------------------------------------------------------------------------------


def clip_annotation_dict(type_name, data, rect, min_area=DEFAULT_MIN_AREA):
    """Clip one annotation dict to a rectangle, in the annotation's own coordinates.

    Args:
        type_name: annotation class name (see SUPPORTED_TYPES).
        data: dict from `to_dict()`.
        rect: (x, y, width, height) of the clipping rectangle.
        min_area: drop clipped rectangles / polygon parts smaller than this.

    Returns:
        [] when nothing of the annotation is left inside the rectangle, else
        [(type_name, data)]. The type can change: a polygon cut into several
        parts comes back as one MultiPolygonAnnotation, and a multipolygon
        left with one part comes back as a PolygonAnnotation. An annotation
        entirely inside the rectangle comes back as an unchanged copy.

    Raises:
        ValueError: unsupported type_name.
    """
    if type_name == PATCH:
        return _clip_patch(data, rect)
    if type_name == RECTANGLE:
        return _clip_rectangle(data, rect, min_area)
    if type_name in (POLYGON, MULTIPOLYGON):
        return _clip_polygonal(type_name, data, rect, min_area)
    raise ValueError(f"Unsupported annotation type: {type_name!r}")


def _clip_patch(data, rect):
    """A patch goes wherever its center is; the half-open test puts a center on a shared edge in one tile only."""
    x, y, width, height = rect
    center_x, center_y = data['center_xy']
    if x <= center_x < x + width and y <= center_y < y + height:
        return [(PATCH, copy.deepcopy(data))]
    return []


def _clip_rectangle(data, rect, min_area):
    x, y, width, height = rect
    (x0, y0), (x1, y1) = data['top_left'], data['bottom_right']
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)

    clipped_left = max(left, x)
    clipped_top = max(top, y)
    clipped_right = min(right, x + width)
    clipped_bottom = min(bottom, y + height)

    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0 or clipped_height <= 0:
        return []

    # Entirely inside: unchanged copy, corners exactly as they were
    if (clipped_left, clipped_top, clipped_right, clipped_bottom) == (left, top, right, bottom):
        return [(RECTANGLE, copy.deepcopy(data))]

    if clipped_width * clipped_height < min_area:
        return []

    clipped = copy.deepcopy(data)
    clipped['top_left'] = (clipped_left, clipped_top)
    clipped['bottom_right'] = (clipped_right, clipped_bottom)
    return [(RECTANGLE, clipped)]


def _clip_polygonal(type_name, data, rect, min_area):
    geometry = _to_geometry(type_name, data)
    if geometry is None or geometry.is_empty:
        return []

    x, y, width, height = rect
    clip_box = box(x, y, x + width, y + height)

    # Entirely inside: unchanged copy, vertices exactly as they were
    if clip_box.covers(geometry):
        return [(type_name, copy.deepcopy(data))]

    parts = [part for part in _polygon_parts(geometry.intersection(clip_box)) if part.area >= min_area]
    if not parts:
        return []

    common = common_fields(data)
    if len(parts) == 1:
        return [(POLYGON, {**common, **_polygon_geometry(parts[0])})]

    return [(MULTIPOLYGON, {**common, 'polygons': [_part_dict(common, part) for part in parts]})]


def annotation_dict_bounds(type_name, data):
    """(min_x, min_y, max_x, max_y) of an annotation dict, or None when it has no geometry.

    Cheap, so a caller clipping many annotations to many tiles can find the
    candidates for each tile with a spatial index instead of clipping them all.
    A patch's bounds are its whole square, although only its center decides
    which tile it goes to.

    Raises:
        ValueError: unsupported type_name.
    """
    if type_name == PATCH:
        center_x, center_y = data['center_xy']
        half = (data.get('annotation_size') or 0) / 2
        return (center_x - half, center_y - half, center_x + half, center_y + half)

    if type_name == RECTANGLE:
        points = [data['top_left'], data['bottom_right']]
    elif type_name == POLYGON:
        points = data.get('points', [])
    elif type_name == MULTIPOLYGON:
        points = [p for part in data.get('polygons', []) for p in part.get('points', [])]
    else:
        raise ValueError(f"Unsupported annotation type: {type_name!r}")

    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def annotation_geometry(type_name, data):
    """Shapely geometry of an annotation dict: a Point for a patch's center, else its area.

    Returns None for a degenerate shape.

    Raises:
        ValueError: unsupported type_name.
    """
    if type_name == PATCH:
        return Point(data['center_xy'])
    if type_name == RECTANGLE:
        (x0, y0), (x1, y1) = data['top_left'], data['bottom_right']
        return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if type_name in (POLYGON, MULTIPOLYGON):
        return _to_geometry(type_name, data)
    raise ValueError(f"Unsupported annotation type: {type_name!r}")


def polygonal_dicts(common, geometry, min_area=0.0):
    """[(type_name, data)] for a polygonal geometry: one polygon, one multipolygon, or nothing.

    Args:
        common: the non-geometry fields to give the result (copied).
        geometry: any shapely geometry; only its polygons are kept.
        min_area: drop parts smaller than this.
    """
    if geometry is None:
        return []
    parts = [part for part in _polygon_parts(geometry) if part.area > 0 and part.area >= min_area]
    if not parts:
        return []
    common = common_fields(common)
    if len(parts) == 1:
        return [(POLYGON, {**common, **_polygon_geometry(parts[0])})]
    return [(MULTIPOLYGON, {**common, 'polygons': [_part_dict(common, part) for part in parts]})]


def annotation_fingerprint(type_name, data, decimals=3):
    """Short hash of everything a person can change by hand: type, label, verification, geometry.

    Two dicts of the same annotation, before and after a save and reload,
    give the same fingerprint; any edit to its shape, label or verified state
    gives a different one. Ids, confidences and data are left out.
    """
    def number(value):
        return round(float(value), decimals) + 0.0  # + 0.0 folds -0.0 into 0.0

    def points(values):
        return [[number(x), number(y)] for x, y in values]

    if type_name == PATCH:
        geometry = [points([data['center_xy']]), number(data.get('annotation_size') or 0)]
    elif type_name == RECTANGLE:
        (x0, y0), (x1, y1) = data['top_left'], data['bottom_right']
        geometry = points([(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))])
    elif type_name == POLYGON:
        geometry = [points(data.get('points', [])), [points(hole) for hole in data.get('holes', [])]]
    elif type_name == MULTIPOLYGON:
        geometry = [[points(part.get('points', [])), [points(hole) for hole in part.get('holes', [])]]
                    for part in data.get('polygons', [])]
    else:
        raise ValueError(f"Unsupported annotation type: {type_name!r}")

    payload = json.dumps([type_name, data.get('label_short_code'), bool(data.get('verified', True)), geometry],
                         separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------------
# Shifting
# ----------------------------------------------------------------------------------------------------------------------


def shift_annotation_dict(type_name, data, dx, dy):
    """Return a copy of an annotation dict with its geometry moved by (dx, dy).

    Raises:
        ValueError: unsupported type_name.
    """
    shifted = copy.deepcopy(data)

    if type_name == PATCH:
        center_x, center_y = shifted['center_xy']
        shifted['center_xy'] = (center_x + dx, center_y + dy)
    elif type_name == RECTANGLE:
        shifted['top_left'] = _shift_point(shifted['top_left'], dx, dy)
        shifted['bottom_right'] = _shift_point(shifted['bottom_right'], dx, dy)
    elif type_name == POLYGON:
        _shift_polygon_in_place(shifted, dx, dy)
    elif type_name == MULTIPOLYGON:
        for part in shifted['polygons']:
            _shift_polygon_in_place(part, dx, dy)
    else:
        raise ValueError(f"Unsupported annotation type: {type_name!r}")

    return shifted


def _shift_point(point, dx, dy):
    return (point[0] + dx, point[1] + dy)


def _shift_polygon_in_place(polygon_dict, dx, dy):
    polygon_dict['points'] = [_shift_point(p, dx, dy) for p in polygon_dict['points']]
    polygon_dict['holes'] = [[_shift_point(p, dx, dy) for p in hole]
                             for hole in polygon_dict.get('holes', [])]


# ----------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------------------------------


def _to_geometry(type_name, data):
    """Shapely geometry for a polygon or multipolygon dict, repaired if invalid; None if degenerate."""
    if type_name == POLYGON:
        geometry = _polygon_from_dict(data)
    else:
        parts = [_polygon_from_dict(part) for part in data.get('polygons', [])]
        parts = [part for part in parts if part is not None]
        geometry = unary_union(parts) if parts else None

    if geometry is None:
        return None
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    return geometry


def _polygon_from_dict(polygon_dict):
    points = polygon_dict.get('points', [])
    if len(points) < 3:
        return None
    holes = [hole for hole in polygon_dict.get('holes', []) if len(hole) >= 3]
    return Polygon(points, holes)


def _polygon_parts(geometry):
    """Polygons inside any geometry; points and lines from a touching edge are dropped."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        parts = []
        for sub_geometry in geometry.geoms:
            parts.extend(_polygon_parts(sub_geometry))
        return parts
    return []


def _polygon_geometry(polygon):
    """'points' / 'holes' for a shapely polygon, without the repeated closing vertex."""
    return {
        'points': [tuple(p) for p in polygon.exterior.coords[:-1]],
        'holes': [[tuple(p) for p in ring.coords[:-1]] for ring in polygon.interiors],
    }


def common_fields(data):
    """Deep copy of everything but the geometry."""
    return copy.deepcopy({key: value for key, value in data.items() if key not in _GEOMETRY_KEYS})


def _part_dict(common, polygon):
    """Polygon dict for one part of a multipolygon; no id, so the part gets a fresh one."""
    part = {key: value for key, value in copy.deepcopy(common).items() if key != 'id'}
    part.update(_polygon_geometry(polygon))
    return part
