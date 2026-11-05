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
original, the original is kept; but deleted on the original while the images
still have it, it is a conflict (bring it back, or keep it deleted). Changed
on both, it is a conflict. An annotation drawn on an image (no origin) is
added as it is.

Annotations are never merged or joined into one. Where two versions of
something come back, both are kept, for the person to combine or delete
afterwards.

Pieces and overlap
------------------
An original cut by an image's edge has a piece on each image. An image whose
piece changed (or was deleted) governs its whole rectangle: the result is the
original outside every governing rectangle, plus the changed pieces, each as
its own annotation. Starting from the original, not from the pieces, keeps
what no image showed: parts outside every work area, in images left out as
empty, in images removed from the project, or slivers too small to copy.

A patch annotation in an overlapping strip is on two images. Changed alike on
both, it comes back once; changed differently, both versions come back,
stacked on one spot.

Masks
-----
A pixel holds one label, so overlapping images that changed it differently
must be settled: either the image the pixel lies more centrally in has the
final say (OWNER), or the pixel is relabeled Review (REVIEW). This is done
among the images first, before anything is compared with the original, so one
image's change is never mistaken for an edit on the original.
"""

import copy
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from coralnet_toolbox.Annotations.annotation_clipping import (PATCH, RECTANGLE, annotation_dict_bounds,
                                                              annotation_fingerprint, annotation_geometry,
                                                              clip_annotation_dict, common_fields,
                                                              polygonal_dicts, shift_annotation_dict)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Conflict choices. Annotations are never merged or joined into one: every
# version comes back, for the person to combine or delete afterwards.
KEEP_OLD = 'keep_old'
KEEP_NEW = 'keep_new'
KEEP_BOTH = 'keep_both'

# Conflict kinds
KIND_EDITED_BOTH = "Edited on both"
KIND_DELETED_ON_ORIGINAL = "Deleted on the original"
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

# Masks: MaskAnnotation.LOCK_BIT, kept here so this module stays free of Qt
MASK_LOCK_BIT = 128

# The snapshot of an image's mask as extracted, written next to the image
MASK_BASE_SUFFIX = ".mask_base.npz"

# Where overlapping images changed a mask pixel differently
OWNER = 'owner'    # the image the pixel lies more centrally in has the final say
REVIEW = 'review'  # the pixel is relabeled Review

# The Review label, as QtLabelWindow creates it; it can be neither edited nor deleted
REVIEW_LABEL_ID = "-1"

# Mask disagreements thinner or smaller than this go to the more central image
# instead of Review: predictions from two images differ by a pixel or two
# along every object's edge, and marking those would outline everything
REVIEW_MIN_WIDTH = 5    # pixels
REVIEW_MIN_AREA = 100   # square pixels

# How the images' mask pixels go onto the original
MASK_ADD = 'add'                  # the images' labels fill every pixel the original has no label on
MASK_REPLACE_ALL = 'replace_all'  # the images' pixels wherever they differ from the original

# Conflicts by what they are about, each asked as its own question
DELETED_KINDS = (KIND_DELETED_ON_ORIGINAL,)  # deleted on the original, still on the images
EDITED_BOTH_KINDS = (KIND_EDITED_BOTH,)      # edited on both sides


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
    category: str = EDITED       # the kind of change on the images it comes from: EDITED, NEW or DELETED

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
    news: list = field(default_factory=list)       # [Change] drawn on an image
    removals: list = field(default_factory=list)   # [Change] deleted only on the images
    conflicts: list = field(default_factory=list)
    unchanged: int = 0        # copies still match their original
    kept_original: int = 0    # changed (or deleted) only on the original: the original stays as it is
    stacked_patches: int = 0  # patches overlapping images changed differently: every version comes back
    cut_annotations: int = 0  # edited annotations that come back as more than one piece

    def conflict_count(self, kinds):
        """How many conflicts are of the given kinds, e.g. DELETED_KINDS."""
        return sum(1 for conflict in self.conflicts if conflict.kind in kinds)

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


def plan_merge_back(parent_path, parent_annotations, tiles, copies=None):
    """Work out what merging the images back would do.

    Args:
        parent_path: the original raster's path; merged annotations go under it.
        parent_annotations: {id: (type_name, data)} for the original's vector annotations now.
        tiles: [TileState] for the set's images still in the project.
        copies: the set record's {copy_id: [origin_id, tile_path]}; None or {} for an
            older set, in which case a copy deleted on its image goes unnoticed.

    Returns:
        MergePlan
    """
    plan = MergePlan()
    copies = copies or {}
    tile_by_path = {tile.path: tile for tile in tiles}

    pieces = {}          # origin_id -> [_Piece]
    origin_fingerprints = {}
    found_ids = set()

    for tile in tiles:
        x, y = tile.rect[0], tile.rect[1]
        for type_name, data in tile.annotations:
            found_ids.add(data.get('id'))
            stamp = data.get('data') or {}
            origin_id = stamp.get('tile_origin_id')
            on_parent = clean_for_parent(shift_annotation_dict(type_name, data, x, y), parent_path)

            if not origin_id:
                # Drawn on an image: added as it is, even over something alike
                plan.news.append(Change([], [(type_name, on_parent, None)], [], [(type_name, on_parent)]))
                continue

            changed = _copy_changed(type_name, data, stamp, parent_annotations.get(origin_id), tile.rect)
            pieces.setdefault(origin_id, []).append(_Piece(tile.rect, type_name, on_parent, changed))
            if stamp.get('tile_origin_fingerprint'):
                origin_fingerprints.setdefault(origin_id, stamp['tile_origin_fingerprint'])

    # The rectangles of images a copy was deleted on. One whose image has left
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


def _plan_original(plan, origin_id, original, pieces, gone, origin_fingerprint, parent_path):
    images_changed = bool(gone) or any(piece.changed for piece in pieces)
    original_changed = original is None or (
        origin_fingerprint is not None and annotation_fingerprint(*original) != origin_fingerprint)

    if not images_changed:
        if original is None and pieces:
            # Deleted on the original since extraction, yet still on the images:
            # whether the images bring it back is asked, not assumed
            rebuilt, cut, _ = _rebuild(pieces, gone, parent_path)
            if rebuilt:
                plan.cut_annotations += int(cut)
                plan.conflicts.append(Conflict(KIND_DELETED_ON_ORIGINAL, incoming=rebuilt, restore_id=origin_id,
                                               category=EDITED))
                return
        if original_changed:
            plan.kept_original += 1
        else:
            plan.unchanged += 1
        return

    # The original is only a sound starting point while it is still as it was copied
    rebuilt, cut, stacked = _rebuild(pieces, gone, parent_path, None if original_changed else original)
    plan.stacked_patches += int(stacked)
    plan.cut_annotations += int(cut)
    # Whether the images edited it or deleted it, for the kind's action in the dialog
    category = EDITED if rebuilt else DELETED

    if original is None:
        if rebuilt:
            plan.conflicts.append(Conflict(KIND_DELETED_ON_ORIGINAL, incoming=rebuilt, restore_id=origin_id,
                                           category=EDITED))
        else:
            plan.kept_original += 1  # deleted on both
        return

    if original_changed:
        plan.conflicts.append(Conflict(KIND_EDITED_BOTH, existing_ids=[origin_id], existing=[original],
                                       incoming=rebuilt, restore_id=origin_id, category=category))
        return

    change = Change([origin_id], with_restore_id(rebuilt, origin_id), [original], rebuilt, origin_id)
    (plan.edits if rebuilt else plan.removals).append(change)


def with_restore_id(items, restore_id):
    """[(type_name, data, id or None)]: the largest of items takes restore_id, the rest get new ids.

    An annotation that comes back as several keeps its id on one of them only,
    so no two annotations ever share an id.
    """
    if not items or not restore_id:
        return [(t, d, None) for t, d in items]
    largest = max(range(len(items)), key=lambda k: _area(*items[k]))
    return [(t, d, restore_id if k == largest else None) for k, (t, d) in enumerate(items)]


def _area(type_name, data):
    geometry = annotation_geometry(type_name, data)
    return geometry.area if geometry is not None else 0.0


def _rebuild(pieces, gone, parent_path, original=None):
    """The original as the images now have it.

    Args:
        pieces: [_Piece] from the images still in the project.
        gone: [rect] of the images its copy was deleted on.
        original: (type_name, data) when it is still as it was copied, else None.

    Returns:
        ([(type_name, data)], whether it came back as several pieces, whether a patch came back stacked)
    """
    patches = [piece for piece in pieces if piece.type_name == PATCH]
    regions = [piece for piece in pieces if piece.type_name != PATCH]
    is_patch = original[0] == PATCH if original is not None else bool(patches) and not regions

    if is_patch:
        results = _settle_patch(patches, bool(gone), parent_path)
        return results, False, len(results) > 1

    results = _rebuild_regions(regions, gone, parent_path, original)
    return results, len(results) > 1, False


def _rebuild_regions(regions, gone_rects, parent_path, original=None):
    """A rectangle or polygon original from its pieces: [(type_name, data)].

    Each piece comes back as its own annotation, never joined, for the person
    to combine afterwards or not.
    """
    governing = [piece.rect for piece in regions if piece.changed] + list(gone_rects)
    governing_area = unary_union([box(x, y, x + w, y + h) for x, y, w, h in governing]) if governing else None

    if original is not None:
        # Changed pieces, plus the original wherever no changed image governs
        kept = [piece for piece in regions if piece.changed]
        rest = annotation_geometry(*original)
        if rest is not None and governing_area is not None:
            rest = rest.difference(governing_area)
        if rest is not None and not rest.is_empty and rest.area > 0:
            kept.append(_Piece(None, original[0], original[1], False, geometry=rest))
    else:
        # The original changed or was deleted: only the pieces are known to be the
        # images' version. Changed pieces are kept whole; unchanged ones are trimmed
        # to outside the changed images and to outside each other, since with
        # overlapping work areas two of them hold the same strip
        kept = [piece for piece in regions if piece.changed]
        covered = governing_area
        for piece in regions:
            if piece.changed:
                continue
            if covered is None:
                kept.append(piece)
            else:
                geometry = annotation_geometry(piece.type_name, piece.data)
                trimmed = geometry.difference(covered) if geometry is not None else None
                if trimmed is not None and not trimmed.is_empty and trimmed.area > 0:
                    kept.append(_Piece(piece.rect, piece.type_name, piece.data, False, geometry=trimmed))
            x, y, w, h = piece.rect
            covered = box(x, y, x + w, y + h) if covered is None else covered.union(box(x, y, x + w, y + h))

    results = []
    for piece in kept:
        base = clean_for_parent(piece.data, parent_path)
        if piece.geometry is None:
            results.append((piece.type_name, base))  # a changed piece, exactly as the image has it
        elif piece.type_name == RECTANGLE and _is_box(piece.geometry):
            results.append(_rectangle_dict(base, piece.geometry))
        else:
            # What is left of the original, or a trimmed piece; a rectangle cut at
            # a corner is no longer a box, so it comes back as a polygon
            results.extend(polygonal_dicts(base, piece.geometry))
    return results


def _is_box(geometry):
    """Whether a geometry is one axis-aligned rectangle."""
    return (geometry.geom_type == 'Polygon' and geometry.area > 0
            and abs(geometry.envelope.area - geometry.area) <= 1e-6 * geometry.envelope.area)


def _settle_patch(patches, deleted, parent_path):
    """A patch from its copies: [(type_name, data)], empty when it was deleted.

    A patch in an overlapping strip is copied onto every image it falls in.
    Unchanged copies still match the original and have no say. Copies changed
    alike come back once; copies changed differently all come back, stacked on
    one spot, for the person to keep or delete. A deletion on one image stands
    only when no other image changed the patch.
    """
    changed = {}
    for piece in patches:
        if piece.changed:
            changed.setdefault(annotation_fingerprint(PATCH, piece.data), piece.data)
    if changed:
        return [(PATCH, clean_for_parent(data, parent_path)) for data in changed.values()]
    if deleted or not patches:
        return []
    return [(PATCH, clean_for_parent(patches[0].data, parent_path))]


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


def resolve(conflict, choice):
    """What a choice does to a conflict: (ids to delete from the original, [(type_name, data, id or None)] to add).

    keep_old   the original stays; what came from the images is dropped
    keep_new   what came from the images replaces the original
    keep_both  both are kept
    """
    if choice == KEEP_OLD:
        return [], []

    if choice == KEEP_NEW:
        return list(conflict.existing_ids), with_restore_id(conflict.incoming, conflict.restore_id)

    if choice == KEEP_BOTH:
        return [], [(t, d, None) for t, d in conflict.incoming]

    raise ValueError(f"Unknown choice: {choice!r}")


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
# Values are compared raw, lock bit included. A pixel locked on the original
# since extraction is never overwritten. Without a base (a set extracted before
# masks were tracked) only empty pixels on the original are filled, and erasing
# on an image cannot be told from never having painted there.
#
# Overlapping images are settled among themselves first (settle_mask_overlaps):
# where two images both changed a pixel to different values, one of them is
# held back from it, or both are and the pixel is marked for Review. Only then
# is each image compared with the original.


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


def mask_pixels_to_take(parent, tile, base=None, mode=MASK_ADD, lock_bit=MASK_LOCK_BIT):
    """Which pixels of one image's region go onto the original, for a mask mode.

    All three arrays are the same region, in the same class-id space; base is
    the image's mask as extracted, or None when unknown.

    MASK_ADD          every pixel the image has a label on, where the original
                      has none, including pixels erased on the original since
                      extraction; nothing on the original is overwritten or erased
    MASK_REPLACE_ALL  the image's pixels wherever they differ from the original,
                      painting and erasing, including pixels the original changed
                      since extraction; a pixel locked on the original since
                      extraction is never taken
    """
    if mode == MASK_ADD:
        return (tile != 0) & (parent == 0)
    locked_since = (parent >= lock_bit) & ((parent != base) if base is not None else True)
    return (tile != parent) & ~locked_since


def settle_mask_overlaps(jobs, image_size=None, disagreement=OWNER, lock_bit=MASK_LOCK_BIT,
                         min_width=REVIEW_MIN_WIDTH, min_area=REVIEW_MIN_AREA):
    """Settle the pixels overlapping images changed differently, before any is compared with the original.

    Where only one image changed a pixel, the others' untouched copies are held
    back from it, so they can neither fill back an erasure nor undo the paint.
    Where all that changed it agree, nothing is held back. Where two changed it
    to different values, the image it lies
    more centrally in keeps it and the other is held back (OWNER). With
    REVIEW, disagreements on class wide and large enough to be real are held
    back from both and returned as areas to mark for Review; thin or small
    ones, and those differing only in the lock bit, still go to the more
    central image.

    Args:
        jobs: [{'rect': (x, y, width, height), 'tile': array, 'base': array or None}], one per
            image with something to say, arrays (height, width) in the original's class ids.
        image_size: (width, height) of the original.
        disagreement: OWNER or REVIEW.

    Returns:
        (held_back, review):
            held_back: {job index: bool (height, width)} pixels that image must leave alone.
            review: [(x, y, area, base)] in the original's pixels: `area` a bool array whose
                top left is (x, y), `base` the snapshot under it, or None when unknown.
    """
    held_back = {}
    review = []

    def hold(index, rows, cols, pixels):
        if not pixels.any():
            return
        if index not in held_back:
            held_back[index] = np.zeros(jobs[index]['tile'].shape, dtype=bool)
        held_back[index][rows, cols] |= pixels

    for i, j, (x0, y0, x1, y1) in _overlapping_pairs([job['rect'] for job in jobs]):
        a, b = jobs[i], jobs[j]
        rows_a, cols_a = _local_slices(a['rect'], x0, y0, x1, y1)
        rows_b, cols_b = _local_slices(b['rect'], x0, y0, x1, y1)
        tile_a, tile_b = a['tile'][rows_a, cols_a], b['tile'][rows_b, cols_b]

        changed_a = _mask_changed(a, rows_a, cols_a)
        changed_b = _mask_changed(b, rows_b, cols_b)

        # Where only one image changed a pixel, the other's untouched copy has no
        # say: it must not fill back what the first erased, or undo what it painted
        hold(i, rows_a, cols_a, changed_b & ~changed_a & (tile_a != tile_b))
        hold(j, rows_b, cols_b, changed_a & ~changed_b & (tile_a != tile_b))

        differ = changed_a & changed_b & (tile_a != tile_b)
        if not differ.any():
            continue

        settle = differ
        if disagreement == REVIEW:
            area = _solid_regions(differ & ((tile_a % lock_bit) != (tile_b % lock_bit)), min_width, min_area)
            if area.any():
                base = a['base'] if a['base'] is not None else b['base']
                base_rows, base_cols = (rows_a, cols_a) if a['base'] is not None else (rows_b, cols_b)
                review.append((x0, y0, area, base[base_rows, base_cols].copy() if base is not None else None))
                hold(i, rows_a, cols_a, area)
                hold(j, rows_b, cols_b, area)
                settle = differ & ~area

        if settle.any():
            a_wins = _first_wins(a['rect'], b['rect'], x0, y0, x1, y1, image_size)
            hold(i, rows_a, cols_a, settle & ~a_wins)
            hold(j, rows_b, cols_b, settle & a_wins)

    return held_back, review


def _mask_changed(job, rows, cols):
    """Pixels an image changed since extraction; without a base, any it painted."""
    tile = job['tile'][rows, cols]
    return tile != (job['base'][rows, cols] if job['base'] is not None else 0)


def _solid_regions(pixels, min_width, min_area):
    """The parts of a bool array at least min_width across and min_area in size."""
    kept = pixels.astype(np.uint8)
    if min_width > 1:
        kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN, np.ones((min_width, min_width), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(kept, connectivity=8)
    large = np.zeros(count, dtype=bool)
    large[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return large[labels] & pixels


# ----------------------------------------------------------------------------------------------------------------------
# Which image has the final say
# ----------------------------------------------------------------------------------------------------------------------


def edge_depth(rect, xs, ys, image_size=None):
    """How far points lie inside a rectangle, from its nearest edge that is not the image's own.

    An image's edge where the original ends is no reason to distrust what is
    near it: no other image covers past it. So such edges are ignored, and a
    point is judged only by the edges it shares the original's pixels across.

    Args:
        rect: (x, y, width, height).
        xs, ys: 1-D x and y coordinates.
        image_size: (width, height) of the original, or None to count every edge.

    Returns:
        float array (len(ys), len(xs)).
    """
    x, y, width, height = rect
    image_width, image_height = image_size if image_size else (None, None)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    def side(distance, at_image_edge):
        return np.full_like(distance, np.inf) if at_image_edge else distance

    dx = np.minimum(side(xs - x, x <= 0),
                    side(x + width - xs, image_width is not None and x + width >= image_width))
    dy = np.minimum(side(ys - y, y <= 0),
                    side(y + height - ys, image_height is not None and y + height >= image_height))
    return np.minimum.outer(dy, dx)


def _first_wins(rect_a, rect_b, x0, y0, x1, y1, image_size=None):
    """Where image a has the final say over image b in their overlap: deeper inside a. Ties go to a."""
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    return edge_depth(rect_a, xs, ys, image_size) >= edge_depth(rect_b, xs, ys, image_size)


def masks_overlap(jobs):
    """Whether any two of the mask jobs overlap, so settle_mask_overlaps may have something to settle."""
    return next(_overlapping_pairs([job['rect'] for job in jobs]), None) is not None


def _overlapping_pairs(rects):
    """(i, j, (x0, y0, x1, y1)) for each pair of rectangles that overlap, i < j."""
    order = sorted(range(len(rects)), key=lambda k: rects[k][0])
    for position, i in enumerate(order):
        ax, ay, aw, ah = rects[i]
        for j in order[position + 1:]:
            bx, by, bw, bh = rects[j]
            if bx >= ax + aw:
                break  # sorted by x: nothing further right can touch this one
            x0, y0 = max(ax, bx), max(ay, by)
            x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            if x1 > x0 and y1 > y0:
                yield min(i, j), max(i, j), (x0, y0, x1, y1)


def _local_slices(rect, x0, y0, x1, y1):
    """Row and column slices of a region of the original, in an image's own pixels."""
    x, y = rect[0], rect[1]
    return slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _rectangle_dict(base, geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    return (RECTANGLE, {**common_fields(base), 'top_left': (min_x, min_y), 'bottom_right': (max_x, max_y)})
