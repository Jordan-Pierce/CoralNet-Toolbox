"""Geometry planning for batch overlap operations between annotation labels.

Nothing here touches Qt: callers hand over shapely geometries and get back which
ones change and what they become, so the rules can be checked without the app.
"""

from dataclasses import dataclass, field

import shapely
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

SUBTRACT = "subtract"
REMOVE = "remove"
MERGE = "merge"

# Overlap is measured as a fraction of the target's area. This slack keeps a
# target that is exactly covered from missing a 100% threshold by rounding.
_COVER_TOLERANCE = 1e-6


# ----------------------------------------------------------------------------------------------------------------------
# Plan containers
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class OverlapSpec:
    """What the user asked for, independent of any one image."""
    operation: str
    target_label_ids: frozenset
    reference_label_ids: frozenset = frozenset()
    min_overlap: float = 0.0        # fraction of the target's area, 0..1
    min_piece_area: float = 0.0     # px^2; smaller pieces left by a subtract are dropped
    include_unverified: bool = True


@dataclass
class ImagePlan:
    """What an OverlapSpec would change on one image.

    ``removed`` annotations are deleted with nothing in their place. Each entry of
    ``replaced`` is (sources, geometry, template): the sources are deleted and one
    annotation built from geometry takes their place, copying label and
    confidence from template.
    """
    image_path: str
    checked: int = 0
    removed: list = field(default_factory=list)
    replaced: list = field(default_factory=list)

    @property
    def has_changes(self):
        return bool(self.removed or self.replaced)

    @property
    def affected(self):
        """Every existing annotation this plan would delete or rebuild."""
        annotations = list(self.removed)
        for sources, _geometry, _template in self.replaced:
            annotations.extend(sources)
        return annotations


# ----------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------------------------------


def polygon_parts(geom, min_area=0.0):
    """List the Polygons in geom, dropping lines, points and pieces under min_area."""
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        candidates = [geom]
    elif hasattr(geom, 'geoms'):
        candidates = []
        for part in geom.geoms:
            candidates.extend(polygon_parts(part))
    else:
        candidates = []

    return [p for p in candidates if not p.is_empty and p.area > 0 and p.area >= min_area]


def _from_parts(parts):
    """Collapse a list of Polygons into a Polygon, a MultiPolygon, or None."""
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def clean_geometry(geom):
    """Return geom's polygonal part, repaired if invalid, or None if nothing is left."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return _from_parts(polygon_parts(geom))


def group_by_overlap(geoms):
    """Group indices of geometries that overlap, directly or through a chain of
    overlaps. ``None`` or empty geometries end up alone.
    """
    n = len(geoms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # The tree only tests pairs whose bounding boxes meet; testing every
    # pair took ~15 s for 3,000 polygons against ~10 ms this way.
    left, right = STRtree(geoms).query(geoms, predicate='intersects')
    for i, j in zip(left.tolist(), right.tolist()):
        if i < j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _hits_by_target(target_geoms, reference_geoms):
    """Map each target index to the reference indices it intersects."""
    left, right = STRtree(reference_geoms).query(target_geoms, predicate='intersects')
    hits = {}
    for t, r in zip(left.tolist(), right.tolist()):
        hits.setdefault(t, []).append(r)
    return hits


def _meets_threshold(target, cutter, min_overlap):
    """True if cutter covers a positive share of target that reaches min_overlap."""
    area = target.area
    if area <= 0:
        return False
    fraction = target.intersection(cutter).area / area
    return fraction > 0 and fraction + _COVER_TOLERANCE >= min_overlap


# ----------------------------------------------------------------------------------------------------------------------
# Planners
# ----------------------------------------------------------------------------------------------------------------------


def plan_subtract(target_geoms, reference_geoms, min_overlap=0.0, min_piece_area=0.0):
    """Cut the references out of each target they overlap.

    Returns [(target_index, new_geometry)] for every target that changes; a
    new_geometry of None means nothing of that target is left.
    """
    if not target_geoms or not reference_geoms:
        return []

    edits = []
    for t, refs in sorted(_hits_by_target(target_geoms, reference_geoms).items()):
        target = target_geoms[t]
        cutter = unary_union([reference_geoms[r] for r in refs])
        if not _meets_threshold(target, cutter, min_overlap):
            continue
        edits.append((t, _from_parts(polygon_parts(target.difference(cutter), min_piece_area))))
    return edits


def plan_remove(target_geoms, reference_geoms, min_overlap=0.0):
    """Return the indices of targets the references overlap by at least min_overlap."""
    if not target_geoms or not reference_geoms:
        return []

    removed = []
    for t, refs in sorted(_hits_by_target(target_geoms, reference_geoms).items()):
        target = target_geoms[t]
        cutter = unary_union([reference_geoms[r] for r in refs])
        if _meets_threshold(target, cutter, min_overlap):
            removed.append(t)
    return removed


def plan_merge(geoms, group_keys):
    """Union every cluster of overlapping geometries that share a group key.

    Returns [(indices, merged_geometry)] for clusters of two or more.
    """
    by_key = {}
    for i, key in enumerate(group_keys):
        by_key.setdefault(key, []).append(i)

    edits = []
    for indices in by_key.values():
        if len(indices) < 2:
            continue
        for group in group_by_overlap([geoms[i] for i in indices]):
            if len(group) < 2:
                continue
            members = [indices[g] for g in group]
            merged = clean_geometry(unary_union([geoms[i] for i in members]))
            if merged is not None:
                edits.append((members, merged))
    return edits
