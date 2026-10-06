"""Plan merging a set of extracted images back into the raster they came from.

Annotation dicts in, a plan out; the Merge Back dialog shows the plan, lets
the person settle its conflicts, and applies it. Nothing here touches the
project.

How a copy is judged
--------------------
Extract and Import stamps every annotation it copies onto an image with, in
its `data`:

    tile_origin_id           the original's id on the parent
    tile_origin_fingerprint  the original's fingerprint when it was copied
    tile_copy_fingerprint    the copy's own fingerprint when it was made

and the set record keeps {copy_id: [origin_id, tile_path]}, so a copy deleted
on its image is noticed too. For each original, then:

    the images changed it    if any copy's fingerprint moved, or a copy is gone
    the original changed     if its own fingerprint moved, or it was deleted

Changed only on the images, the change is applied. Changed only on the
original, the original is kept. Changed on both, it is a conflict. An
annotation drawn on an image (no origin) is added, unless it overlaps one of
the same label that the person could not see (added to the original after
extraction), or one of the same label drawn on another image; then it is a
conflict. Different classes are never treated as the same object, and Merge
never joins them: both are brought in.

Pieces and overlap
------------------
An original cut by an image's edge has a piece on each image; the pieces are
rejoined. With overlapping work areas the same strip is on two images, so an
image whose piece changed (or was deleted) governs its whole rectangle, and
unchanged pieces from other images are trimmed to outside it. Otherwise an
untouched copy would bring the old shape back.
"""

import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from coralnet_toolbox.Annotations.annotation_clipping import (PATCH, RECTANGLE, annotation_dict_bounds,
                                                              annotation_fingerprint, annotation_geometry,
                                                              clip_annotation_dict, common_fields,
                                                              polygonal_dicts, shift_annotation_dict)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Conflict choices
MERGE = 'merge'
KEEP_OLD = 'keep_old'
KEEP_NEW = 'keep_new'
KEEP_BOTH = 'keep_both'

CHOICE_LABELS = {
    MERGE: "Merge",
    KEEP_OLD: "Keep old",
    KEEP_NEW: "Keep new",
    KEEP_BOTH: "Keep both",
}

# Conflict kinds, as shown to the person
KIND_EDITED_BOTH = "Edited on both"
KIND_DELETED_ON_ORIGINAL = "Deleted on the original"
KIND_LABELS_DIFFER = "Relabeled differently"
KIND_OVERLAPS = "New overlaps existing"
KIND_DUPLICATED = "Drawn on two images"
# Changes moved to the conflicts list because their kind is set to Ask
KIND_EDITED_ON_IMAGES = "Edited on the images"
KIND_NEW_ON_IMAGES = "New on the images"
KIND_DELETED_ON_IMAGES = "Deleted on the images"

# The kinds of change an image can make; each has its own action in the dialog
EDITED = 'edited'
NEW = 'new'
DELETED = 'deleted'

# Actions for a kind of change
APPLY = 'apply'    # do it (apply the edit, add the new annotation, delete from the original)
ASK = 'ask'        # list each one as a conflict to decide
IGNORE = 'ignore'  # leave the original as it is, conflicts of that kind included

# Provenance written into a copy's data; dropped once it is merged back
TILE_DATA_KEYS = ('tile_origin_id', 'tile_origin_fingerprint', 'tile_copy_fingerprint')

# Intersection over union at which a new annotation is the same object as another
DEFAULT_OVERLAP_THRESHOLD = 0.5

# Masks: MaskAnnotation.LOCK_BIT, kept here so this module stays free of Qt
MASK_LOCK_BIT = 128

# The snapshot of an image's mask as extracted, written next to the image
MASK_BASE_SUFFIX = ".mask_base.npz"


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class TileState:
    """One extracted image still in the project."""
    path: str
    rect: tuple            # (x, y, width, height) in the parent's pixels
    annotations: list      # [(type_name, data)] in the image's own coordinates


@dataclass
class Conflict:
    """Something the person has to decide. Dicts are in the parent's coordinates."""
    kind: str
    existing_ids: list = field(default_factory=list)   # annotations on the original this is about
    existing: list = field(default_factory=list)       # [(type_name, data)]
    incoming: list = field(default_factory=list)       # [(type_name, data)]
    restore_id: str = None       # id a single incoming annotation takes when it replaces its original
    overlap: float = None        # how strongly the overlap that made it a conflict matched
    existing_is_new: bool = False  # the existing side was also drawn on an image, not on the original
    category: str = EDITED       # the kind of change on the images it comes from: EDITED, NEW or DELETED

    @property
    def allow_merge(self):
        """Merging joins shapes of one class: it needs something incoming, one shared label, and no patches.

        Two different classes are never joined into one; Merge keeps both instead.
        """
        members = self.existing + self.incoming
        labels = {d.get('label_short_code') for _, d in members}
        return (bool(self.incoming) and len(members) > 1 and len(labels) == 1
                and all(t != PATCH for t, _ in members))

    def bounds(self):
        """(min_x, min_y, max_x, max_y) around everything involved, or None."""
        boxes = [annotation_dict_bounds(t, d) for t, d in self.existing + self.incoming]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class Change:
    """One change made only on the images, applied as it is unless its kind is set otherwise."""
    delete_ids: list                # ids to delete from the original
    adds: list                      # [(type_name, data, id or None)]
    existing: list                  # [(type_name, data)] on the original, for display
    incoming: list                  # [(type_name, data)] from the images, for display
    restore_id: str = None

    def as_conflict(self, kind, category):
        """This change as a row to decide, for a kind set to Ask."""
        return Conflict(kind, existing_ids=list(self.delete_ids), existing=list(self.existing),
                        incoming=list(self.incoming), restore_id=self.restore_id, category=category)


@dataclass
class MergePlan:
    edits: list = field(default_factory=list)      # [Change] edited only on the images
    news: list = field(default_factory=list)       # [Change] drawn on an image, overlapping nothing
    removals: list = field(default_factory=list)   # [Change] deleted only on the images
    conflicts: list = field(default_factory=list)
    unchanged: int = 0        # copies still match their original
    kept_original: int = 0    # changed (or deleted) only on the original: the original stays as it is

    @property
    def applied(self):
        return len(self.edits)

    @property
    def added(self):
        return len(self.news)

    @property
    def deleted(self):
        return len(self.removals)

    @property
    def deletes(self):
        """Ids every automatic change deletes from the original."""
        return [i for change in self.edits + self.news + self.removals for i in change.delete_ids]

    @property
    def adds(self):
        """[(type_name, data, id or None)] every automatic change adds."""
        return [add for change in self.edits + self.news + self.removals for add in change.adds]


@dataclass
class _Piece:
    rect: tuple
    type_name: str
    data: dict              # parent coordinates
    changed: bool
    geometry: object = None  # set when trimmed


@dataclass
class _Node:
    """An annotation taking part in the overlap check."""
    is_new: bool
    type_name: str
    data: dict
    tile_path: str = None
    annotation_id: str = None


# ----------------------------------------------------------------------------------------------------------------------
# Stamping copies (used by Extract and Import)
# ----------------------------------------------------------------------------------------------------------------------


def stamp_copy(piece_data, origin_id, origin_fingerprint):
    """Record where a copy came from. The copy fingerprint is added once the copy is built."""
    piece_data.setdefault('data', {})
    piece_data['data']['tile_origin_id'] = origin_id
    piece_data['data']['tile_origin_fingerprint'] = origin_fingerprint


def clean_for_parent(data, parent_path):
    """A copy of an annotation dict ready to go onto the parent: its path, no id, no provenance."""
    cleaned = copy.deepcopy(data)
    cleaned.pop('id', None)
    cleaned['image_path'] = parent_path
    cleaned['data'] = {k: v for k, v in (cleaned.get('data') or {}).items() if k not in TILE_DATA_KEYS}
    for part in cleaned.get('polygons', []):
        part.pop('id', None)
        part['image_path'] = parent_path
        part['data'] = {k: v for k, v in (part.get('data') or {}).items() if k not in TILE_DATA_KEYS}
    return cleaned


# ----------------------------------------------------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------------------------------------------------


def plan_merge_back(parent_path, parent_annotations, tiles, copies=None,
                    overlap_threshold=DEFAULT_OVERLAP_THRESHOLD):
    """Work out what merging the images back would do.

    Args:
        parent_path: the original raster's path; merged annotations go under it.
        parent_annotations: {id: (type_name, data)} for the original's vector annotations now.
        tiles: [TileState] for the set's images still in the project.
        copies: the set record's {copy_id: [origin_id, tile_path]}; None or {} for an
            older set, in which case a copy deleted on its image goes unnoticed.
        overlap_threshold: intersection over union at which a new annotation is the
            same object as another.

    Returns:
        MergePlan
    """
    plan = MergePlan()
    copies = copies or {}
    tile_by_path = {tile.path: tile for tile in tiles}

    pieces = {}          # origin_id -> [_Piece]
    origin_fingerprints = {}
    found_ids = set()
    new_nodes = []

    for tile in tiles:
        x, y = tile.rect[0], tile.rect[1]
        for type_name, data in tile.annotations:
            found_ids.add(data.get('id'))
            stamp = data.get('data') or {}
            origin_id = stamp.get('tile_origin_id')
            on_parent = clean_for_parent(shift_annotation_dict(type_name, data, x, y), parent_path)

            if not origin_id:
                new_nodes.append(_Node(True, type_name, on_parent, tile_path=tile.path))
                continue

            changed = _copy_changed(type_name, data, stamp, parent_annotations.get(origin_id), tile.rect)
            pieces.setdefault(origin_id, []).append(_Piece(tile.rect, type_name, on_parent, changed))
            if stamp.get('tile_origin_fingerprint'):
                origin_fingerprints.setdefault(origin_id, stamp['tile_origin_fingerprint'])

    # Copies deleted on an image still in the project. One whose image has left
    # the project is not counted: that image just has nothing to say.
    gone = {}
    for copy_id, (origin_id, tile_path) in copies.items():
        tile = tile_by_path.get(tile_path)
        if tile is not None and copy_id not in found_ids:
            gone.setdefault(origin_id, []).append(tile.rect)

    for origin_id in list(dict.fromkeys(list(pieces) + list(gone))):
        _plan_original(plan, origin_id, parent_annotations.get(origin_id),
                       pieces.get(origin_id, []), gone.get(origin_id, []),
                       origin_fingerprints.get(origin_id), parent_path)

    # A new annotation is only checked against the original's annotations the
    # person could not see on the images: ones added after extraction. One whose
    # copy was on an image was in view when the new one was drawn, so drawing
    # over it was deliberate, and both are kept.
    touched = set(plan.deletes) | {i for conflict in plan.conflicts for i in conflict.existing_ids}
    seen = set(pieces) | set(gone)
    _plan_new(plan, new_nodes,
              {i: a for i, a in parent_annotations.items() if i not in touched and i not in seen},
              overlap_threshold)
    return plan


def _copy_changed(type_name, data, stamp, original, rect):
    """Whether a copy differs from how it was made."""
    stored = stamp.get('tile_copy_fingerprint')
    if stored:
        return annotation_fingerprint(type_name, data) != stored

    # A copy made before fingerprints were stored: compare with what copying
    # the original now would give. An edit on the original then reads as an
    # edit on the image, so such sets apply rather than ask.
    if original is None:
        return True
    x, y = rect[0], rect[1]
    expected = {annotation_fingerprint(t, shift_annotation_dict(t, d, -x, -y))
                for t, d in clip_annotation_dict(original[0], original[1], rect)}
    return annotation_fingerprint(type_name, data) not in expected


def _plan_original(plan, origin_id, original, pieces, gone_rects, origin_fingerprint, parent_path):
    images_changed = bool(gone_rects) or any(piece.changed for piece in pieces)
    original_changed = original is None or (
        origin_fingerprint is not None and annotation_fingerprint(*original) != origin_fingerprint)

    if not images_changed:
        if original_changed:
            plan.kept_original += 1
        else:
            plan.unchanged += 1
        return

    rebuilt, labels_differ = _rebuild(pieces, gone_rects, parent_path)
    restore_id = origin_id if len(rebuilt) == 1 else None
    # Whether the images edited it or deleted it, for the kind's action in the dialog
    category = EDITED if rebuilt else DELETED

    if original is None:
        if rebuilt:
            plan.conflicts.append(Conflict(KIND_DELETED_ON_ORIGINAL, incoming=rebuilt, restore_id=restore_id,
                                           category=EDITED))
        else:
            plan.kept_original += 1  # deleted on both
        return

    if original_changed or labels_differ:
        plan.conflicts.append(Conflict(KIND_EDITED_BOTH if original_changed else KIND_LABELS_DIFFER,
                                       existing_ids=[origin_id], existing=[original],
                                       incoming=rebuilt, restore_id=restore_id, category=category))
        return

    change = Change([origin_id], [(t, d, restore_id) for t, d in rebuilt], [original], rebuilt, restore_id)
    (plan.edits if rebuilt else plan.removals).append(change)


def _rebuild(pieces, gone_rects, parent_path):
    """The original as the images now have it: ([(type_name, data)], whether pieces disagree on label)."""
    governing = [piece.rect for piece in pieces if piece.changed] + list(gone_rects)
    governing_area = unary_union([box(x, y, x + w, y + h) for x, y, w, h in governing]) if governing else None

    kept = []
    for piece in pieces:
        if piece.changed or governing_area is None:
            kept.append(piece)
        elif piece.type_name == PATCH:
            center = piece.data['center_xy']
            if not any(_contains(rect, center) for rect in governing):
                kept.append(piece)
        else:
            trimmed = annotation_geometry(piece.type_name, piece.data).difference(governing_area)
            if not trimmed.is_empty and trimmed.area > 0:
                kept.append(_Piece(piece.rect, piece.type_name, piece.data, False, geometry=trimmed))

    groups = {}
    for piece in kept:
        groups.setdefault(piece.data.get('label_short_code'), []).append(piece)

    results = []
    for group in groups.values():
        seen_patches = set()
        for piece in group:
            if piece.type_name == PATCH:
                fingerprint = annotation_fingerprint(PATCH, piece.data)
                if fingerprint not in seen_patches:
                    seen_patches.add(fingerprint)
                    results.append((PATCH, clean_for_parent(piece.data, parent_path)))

        regions = [piece for piece in group if piece.type_name != PATCH]
        if regions:
            base = clean_for_parent(next((p for p in regions if p.changed), regions[0]).data, parent_path)
            union = unary_union([p.geometry if p.geometry is not None else annotation_geometry(p.type_name, p.data)
                                 for p in regions])
            if all(p.type_name == RECTANGLE for p in regions):
                results.append(_rectangle_dict(base, union))
            else:
                results.extend(polygonal_dicts(base, union))

    return results, len(groups) > 1


def _plan_new(plan, new_nodes, candidates, overlap_threshold):
    """Add new annotations, or make conflicts of those overlapping the original's or each other."""
    if not new_nodes:
        return

    parent_nodes = [_Node(False, t, d, annotation_id=i) for i, (t, d) in candidates.items()]
    nodes = new_nodes + parent_nodes
    boxes = []
    for node in nodes:
        bounds = annotation_dict_bounds(node.type_name, node.data) or (0, 0, 0, 0)
        boxes.append(box(*bounds))
    tree = STRtree(boxes)

    parent_of = list(range(len(nodes)))

    def find(i):
        while parent_of[i] != i:
            parent_of[i] = parent_of[parent_of[i]]
            i = parent_of[i]
        return i

    best_overlap = {}
    for i, node in enumerate(new_nodes):
        for j in tree.query(boxes[i]):
            j = int(j)
            if j == i:
                continue
            other = nodes[j]
            # Two annotations drawn on the same image were drawn on purpose
            if other.is_new and other.tile_path == node.tile_path:
                continue
            score = _overlap(node, other, overlap_threshold)
            if score is None:
                continue
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent_of[root_j] = root_i
            for k in (i, j):
                best_overlap[k] = max(best_overlap.get(k, 0.0), score)

    # Groups holding at least one new annotation; the original's annotations
    # that overlap nothing new are not part of the merge
    new_roots = {find(k) for k in range(len(new_nodes))}
    components = {}
    for i in range(len(nodes)):
        root = find(i)
        if root in new_roots:
            components.setdefault(root, []).append(i)

    for members in components.values():
        new_members = [nodes[i] for i in members if nodes[i].is_new]
        if not new_members:
            continue
        if len(members) == 1:
            node = new_members[0]
            plan.news.append(Change([], [(node.type_name, node.data, None)], [], [(node.type_name, node.data)]))
            continue

        parent_members = [nodes[i] for i in members if not nodes[i].is_new]
        overlap = max((best_overlap.get(i, 0.0) for i in members), default=None)
        if parent_members:
            plan.conflicts.append(Conflict(
                KIND_OVERLAPS,
                existing_ids=[n.annotation_id for n in parent_members],
                existing=[(n.type_name, n.data) for n in parent_members],
                incoming=[(n.type_name, n.data) for n in new_members],
                overlap=overlap,
                category=NEW,
            ))
        else:
            plan.conflicts.append(Conflict(
                KIND_DUPLICATED,
                existing=[(new_members[0].type_name, new_members[0].data)],
                incoming=[(n.type_name, n.data) for n in new_members[1:]],
                overlap=overlap,
                existing_is_new=True,
                category=NEW,
            ))


def _overlap(a, b, overlap_threshold):
    """How strongly two annotations are the same object (0 to 1), or None if they are not.

    Only annotations of the same label can match: two classes overlapping are
    both brought in. Patches match when their centers are within half a patch
    width; areas match at the intersection over union threshold. A patch and
    an area never match: a point inside a polygon is normal, not a duplicate.
    """
    if a.data.get('label_short_code') != b.data.get('label_short_code'):
        return None
    a_patch, b_patch = a.type_name == PATCH, b.type_name == PATCH
    if a_patch != b_patch:
        return None

    if a_patch:
        (ax, ay), (bx, by) = a.data['center_xy'], b.data['center_xy']
        half = min(a.data.get('annotation_size') or 0, b.data.get('annotation_size') or 0) / 2
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        if half <= 0:
            return 1.0 if distance == 0 else None
        return 1.0 - distance / half if distance <= half else None

    geometry_a = annotation_geometry(a.type_name, a.data)
    geometry_b = annotation_geometry(b.type_name, b.data)
    if geometry_a is None or geometry_b is None:
        return None
    union_area = geometry_a.union(geometry_b).area
    if union_area <= 0:
        return None
    iou = geometry_a.intersection(geometry_b).area / union_area
    return iou if iou >= overlap_threshold else None


# ----------------------------------------------------------------------------------------------------------------------
# Actions per kind of change
# ----------------------------------------------------------------------------------------------------------------------


def apply_actions(plan, edited=APPLY, new=APPLY, deleted=APPLY):
    """What the merge does once each kind of change has its action.

    APPLY  the changes of that kind happen
    ASK    each change of that kind becomes a conflict to decide
    IGNORE the original stays as it is for that kind, conflicts of that kind included

    Conflicts proper (changed on both sides, overlaps) stay conflicts under
    APPLY and ASK: applying a kind never settles a real disagreement.

    Returns:
        (ids to delete from the original, [(type_name, data, id or None)] to add, [Conflict] to decide)
    """
    actions = {EDITED: edited, NEW: new, DELETED: deleted}
    asked_kinds = {EDITED: KIND_EDITED_ON_IMAGES, NEW: KIND_NEW_ON_IMAGES, DELETED: KIND_DELETED_ON_IMAGES}

    deletes, adds = [], []
    conflicts = [conflict for conflict in plan.conflicts if actions[conflict.category] != IGNORE]
    for category, changes in ((EDITED, plan.edits), (NEW, plan.news), (DELETED, plan.removals)):
        action = actions[category]
        if action == APPLY:
            for change in changes:
                deletes.extend(change.delete_ids)
                adds.extend(change.adds)
        elif action == ASK:
            conflicts.extend(change.as_conflict(asked_kinds[category], category) for change in changes)
    return deletes, adds, conflicts


# ----------------------------------------------------------------------------------------------------------------------
# Resolving conflicts
# ----------------------------------------------------------------------------------------------------------------------


def resolve(conflict, choice, parent_path):
    """What a choice does to a conflict: (ids to delete from the original, [(type_name, data, id or None)] to add).

    keep_old   the original stays; what came from the images is dropped
    keep_new   what came from the images replaces the original
    keep_both  both are kept
    merge      shapes of one label are joined into one; with different labels
               or patches, both are kept instead
    """
    existing_adds = [(t, d, None) for t, d in conflict.existing] if conflict.existing_is_new else []

    if choice == MERGE and not conflict.allow_merge:
        choice = KEEP_BOTH

    if choice == KEEP_OLD:
        return [], existing_adds

    if choice == KEEP_NEW:
        restore = conflict.restore_id if len(conflict.incoming) == 1 else None
        return list(conflict.existing_ids), [(t, d, restore) for t, d in conflict.incoming]

    if choice == KEEP_BOTH:
        return [], existing_adds + [(t, d, None) for t, d in conflict.incoming]

    if choice == MERGE:
        return list(conflict.existing_ids), [(t, d, None) for t, d in _merged(conflict, parent_path)]

    raise ValueError(f"Unknown choice: {choice!r}")


def _merged(conflict, parent_path):
    members = conflict.existing + conflict.incoming
    base = clean_for_parent(conflict.incoming[0][1], parent_path)
    union = unary_union([g for g in (annotation_geometry(t, d) for t, d in members) if g is not None])
    if all(t == RECTANGLE for t, _ in members):
        return [_rectangle_dict(base, union)]
    return polygonal_dicts(base, union)


# ----------------------------------------------------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------------------------------------------------
#
# A mask merges pixel by pixel, the same three ways as annotations. At
# extraction each image gets a slice of the original's mask, and a snapshot of
# that slice (the base) is saved next to the image. At merge, per pixel:
#
#     changed on the image only      -> the image's value is applied
#     changed on the original only   -> the original's value stays
#     changed on both, differently   -> a conflict, settled once for the whole set
#
# Values are compared raw, lock bit included. Without a base (a set extracted
# before masks were tracked) only empty pixels on the original are filled, and
# erasing on an image cannot be told from never having painted there.


def mask_base_path(tile_path):
    """Where an image's extraction-time mask snapshot lives: next to the image."""
    return os.path.splitext(tile_path)[0] + MASK_BASE_SUFFIX


def save_mask_base(path, mask, label_codes):
    """Write a mask snapshot with the {class_id: short_code} its values mean."""
    np.savez_compressed(path, mask=np.ascontiguousarray(mask, dtype=np.uint8),
                        label_codes=json.dumps({str(k): v for k, v in label_codes.items()}))


def load_mask_base(path):
    """(mask, {class_id: short_code}) from a snapshot, or (None, None) if it cannot be read."""
    try:
        with np.load(path, allow_pickle=False) as data:
            codes = {int(k): v for k, v in json.loads(str(data['label_codes'])).items()}
            return data['mask'].astype(np.uint8), codes
    except Exception:
        return None, None


def class_lut(source_codes, target_ids, lock_bit=MASK_LOCK_BIT):
    """Lookup table moving mask values between two class-id spaces, by label code.

    Args:
        source_codes: {class_id: short_code} the values were written against.
        target_ids: {short_code: class_id} to translate them into.

    A class whose label the target lacks is cleared to 0, since its pixels no
    longer mean anything. The lock bit is carried across. Values the source
    map does not mention pass through unchanged.
    """
    lut = np.arange(256, dtype=np.uint8)
    for class_id, code in source_codes.items():
        class_id = int(class_id)
        if class_id <= 0 or class_id >= lock_bit:
            continue
        new_id = target_ids.get(code)
        if new_id is None:
            lut[class_id] = 0
            lut[class_id + lock_bit] = 0
            continue
        lut[class_id] = new_id
        if new_id + lock_bit < 256:
            lut[class_id + lock_bit] = new_id + lock_bit
    return lut


def mask_merge_regions(parent, tile, base=None, lock_bit=MASK_LOCK_BIT):
    """Which pixels of one image's region to take from the image, and which are conflicts.

    All three arrays are the same region, in the same class-id space.

    Args:
        parent: the original's mask over the image's rectangle, as it is now.
        tile: the image's mask now.
        base: the image's mask as extracted, or None when it is unknown.

    Returns:
        (apply, conflict): boolean arrays. `apply` pixels take the image's
        value; `conflict` pixels changed on both sides to different values.
    """
    if base is None:
        tile_has = tile != 0
        apply = tile_has & (parent == 0)
        conflict = tile_has & (parent != 0) & ((parent % lock_bit) != (tile % lock_bit))
        return apply, conflict

    tile_changed = tile != base
    parent_changed = parent != base
    apply = tile_changed & ~parent_changed
    conflict = tile_changed & parent_changed & (tile != parent)
    return apply, conflict


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _contains(rect, point):
    x, y, width, height = rect
    return x <= point[0] < x + width and y <= point[1] < y + height


def _rectangle_dict(base, geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    return (RECTANGLE, {**common_fields(base), 'top_left': (min_x, min_y), 'bottom_right': (max_x, max_y)})
