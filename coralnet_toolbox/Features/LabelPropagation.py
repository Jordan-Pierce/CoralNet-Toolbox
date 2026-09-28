"""
LabelPropagation — densify existing labels over a dense feature grid.

Shared by the 2D FeatureSelectTool (multi-class mode) and batch densify: seed
per-class prototypes from an image's existing annotations, classify a region
into a per-pixel label map, and write it into the image's mask, optionally
replacing the patch annotations it agrees with.

A region is an image-pixel rect ``(left, top, width, height)`` covered by a
``(grid_h, grid_w)`` feature grid. The mapping between the two is proportional,
not a single stride: the extractor resizes the crop to a square before
patchifying, so the grid's aspect ratio need not match the crop's.

Batch densify can split an image into several regions (its work areas), given
as integer windows ``(x0, y0, x1, y1)``. Where windows overlap, each pixel
belongs to the window it sits most centrally in (see owned_mask), so a seed
votes once and every pixel is written once.
"""

from __future__ import annotations

import numpy as np

# The Review label marks a point nobody has classified; it is never propagated.
REVIEW_LABEL_ID = '-1'


def _has_class(label):
    return label is not None and str(label.id) != REVIEW_LABEL_ID


# ----------------------------------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------------------------------


def standardize_features(features):
    """Per-channel z-standardize [N, D] features over the region, re-L2.

    Raw ViT patch tokens carry a large channel-wise mean, so cosine similarity
    between ANY two patches of one image sits high and a fixed threshold has
    little usable range. Centering and scaling each channel by its statistics
    across the region removes that shared offset.

    Use it ONLY for thresholding similarity to positive clicks (binary mode
    without negatives), where it measured better: IoU 0.87 vs 0.80 at the best
    fixed threshold (Shoe), 0.34 vs 0.23 (Coralscapes). Do NOT use it when
    classes compete (multi-class, densify, suggestions): the region's mean is
    the dominant class, so standardizing turns that class's features into noise
    and its pixels leak to the others (mIoU 0.79 -> 0.25 on Shoe from one
    click per class).

    NOTE: dropping leading principal components is WRONG here: the
    class-discriminative signal lives in those components (removing the top 1
    or top 8 collapsed mean AUC to ~0.50, i.e. chance).
    """
    features = np.asarray(features, dtype=np.float32)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def build_query_engine(feature_map, standardize=False):
    """QueryEngine over an [h, w, C] feature map; returns ``(engine, (h, w))``.

    Raw (L2-normalized) features by default; see standardize_features for the
    one case that should standardize.
    """
    from coralnet_toolbox.Features.QueryEngine import QueryEngine

    grid_hw = (int(feature_map.shape[0]), int(feature_map.shape[1]))
    features = np.asarray(feature_map).reshape(-1, feature_map.shape[2])
    if standardize:
        features = standardize_features(features)
    valid = np.ones(features.shape[0], dtype=bool)
    return QueryEngine(features, valid), grid_hw


def upsample_field(grid, out_h, out_w):
    """Bilinearly upsample a [gh, gw] float grid to [out_h, out_w].

    Smoothing the per-patch field before thresholding / argmaxing gives a
    boundary that follows a smooth contour rather than the grid steps. Falls
    back to a numpy linear interpolation if OpenCV is unavailable.
    """
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    try:
        import cv2
        return cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        gh, gw = grid.shape
        ys = np.linspace(0, gh - 1, out_h)
        xs = np.linspace(0, gw - 1, out_w)
        y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
        y1 = np.clip(y0 + 1, 0, gh - 1)
        x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
        x1 = np.clip(x0 + 1, 0, gw - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
        bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
        return top * (1 - wy) + bot * wy


# ----------------------------------------------------------------------------------------------------------------------
# Grid mapping
# ----------------------------------------------------------------------------------------------------------------------


def cell_at(x, y, rect, grid_hw):
    """Map an image pixel (x, y) to a flat grid element id, or None if outside."""
    left, top, width, height = rect
    grid_h, grid_w = grid_hw
    rx, ry = x - left, y - top
    if rx < 0 or ry < 0 or rx >= width or ry >= height:
        return None
    gx = max(0, min(int(rx / width * grid_w), grid_w - 1))
    gy = max(0, min(int(ry / height * grid_h), grid_h - 1))
    return gy * grid_w + gx


def _grid_rings(annotation, rect, grid_hw):
    """Map a vector annotation's shapely geometry into feature-grid rings.

    Returns a list of ``(exterior_int32, [hole_int32, ...])`` rings in grid
    pixel coordinates, ready for ``cv2.fillPoly``. Vertices are shifted by -0.5
    so a cell CENTER lands on an integer coord (OpenCV samples pixels at integer
    positions), matching cell_at's proportional mapping. Returns None when the
    annotation has no rasterizable geometry.
    """
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return None
    sx = grid_hw[1] / width
    sy = grid_hw[0] / height

    def _ring(coords):
        pts = np.asarray(coords, dtype=np.float64)
        if pts.shape[0] < 3:
            return None
        pts[:, 0] = (pts[:, 0] - left) * sx - 0.5
        pts[:, 1] = (pts[:, 1] - top) * sy - 0.5
        return np.round(pts).astype(np.int32)

    # Robust geometry acquisition (mirrors MaskAnnotation): prefer the
    # shapely getter, fall back to the Qt polygon's outer ring.
    geom = None
    getter = getattr(annotation, 'get_rasterization_geometry', None)
    if callable(getter):
        try:
            geom = getter()
        except Exception:
            geom = None
    if geom is None:
        try:
            pts = [(p.x(), p.y()) for p in annotation.get_polygon()]
            if len(pts) >= 3:
                from shapely.geometry import Polygon
                geom = Polygon(pts)
        except Exception:
            geom = None
    if geom is None:
        return None

    members = geom.geoms if getattr(geom, 'geom_type', None) == 'MultiPolygon' else [geom]
    rings = []
    for poly in members:
        try:
            ext = _ring(list(poly.exterior.coords))
        except Exception:
            ext = None
        if ext is None:
            continue
        holes = []
        try:
            for r in poly.interiors:
                hole = _ring(list(r.coords))
                if hole is not None:
                    holes.append(hole)
        except Exception:
            pass
        rings.append((ext, holes))
    return rings or None


def cells_covered(annotation, rect, grid_hw):
    """Flat grid element ids covered by a vector region annotation.

    Rasterizes the annotation's geometry straight into the (small) grid with
    cv2.fillPoly — O(vertices + grid), no per-cell containment test. Holes are
    punched out and MultiPolygon islands all fill. Falls back to the centroid's
    cell for sub-cell shapes or when geometry is unavailable.
    """
    cells = []
    rings = _grid_rings(annotation, rect, grid_hw)
    if rings:
        import cv2
        grid = np.zeros(grid_hw, dtype=np.uint8)
        for ext, holes in rings:
            cv2.fillPoly(grid, [ext], 1)
            if holes:
                cv2.fillPoly(grid, holes, 0)
        ys, xs = np.nonzero(grid)
        cells = (ys.astype(np.int64) * grid_hw[1] + xs).tolist()

    if not cells:
        try:
            cid = cell_at(*annotation.get_centroid(), rect, grid_hw)
            if cid is not None:
                cells.append(cid)
        except Exception:
            pass
    return cells


def cells_by_class_from_mask(mask_annotation, rect, grid_hw):
    """Group grid element ids by the mask class occupying them, ``{label: [cells]}``.

    Crops mask_data to the rect and nearest-downsamples it to the grid in one
    cv2.resize — O(grid) regardless of the mask's full resolution. The LOCK_BIT
    is stripped so locked and unlocked pixels of a class read alike.
    """
    data = getattr(mask_annotation, 'mask_data', None)
    if data is None:
        return {}
    left, top, width, height = rect
    h, w = data.shape
    x0 = max(0, int(np.floor(left)))
    y0 = max(0, int(np.floor(top)))
    x1 = min(w, int(np.ceil(left + width)))
    y1 = min(h, int(np.ceil(top + height)))
    if x1 <= x0 or y1 <= y0:
        return {}

    import cv2
    lock = getattr(mask_annotation, 'LOCK_BIT', 128)
    crop = np.ascontiguousarray(data[y0:y1, x0:x1] & (lock - 1))
    grid = cv2.resize(crop, (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_NEAREST)

    by_label = {}
    for class_id in np.unique(grid):
        if class_id == 0:
            continue
        label = mask_annotation.class_id_to_label_map.get(int(class_id))
        if not _has_class(label):
            continue
        ys, xs = np.nonzero(grid == class_id)
        by_label[label] = (ys.astype(np.int64) * grid_hw[1] + xs).tolist()
    return by_label


# ----------------------------------------------------------------------------------------------------------------------
# Seeding and classification
# ----------------------------------------------------------------------------------------------------------------------


def iter_seeds(annotations, rect, grid_hw, mask_annotation=None):
    """Yield ``(label, cells, annotation)`` prototype seeds from existing annotations.

      - MaskAnnotation: one seed per class present in the rect (annotation None).
      - PatchAnnotation: its single center-most cell (deliberately one vote, so
        dense patches don't swamp manual refinements).
      - Polygon / Rectangle / MultiPolygon: every cell the shape covers (a
        bigger shape genuinely represents more of its class).

    The image's mask lives in the annotation manager's own registry, not in
    the image's annotation list, so it is passed separately. Review-labeled
    annotations carry no class and are skipped.
    """
    from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

    sources = list(annotations)
    if mask_annotation is not None and mask_annotation not in sources:
        sources.append(mask_annotation)

    for ann in sources:
        if getattr(ann, 'is_mask_annotation', False):
            try:
                by_label = cells_by_class_from_mask(ann, rect, grid_hw)
            except Exception:
                by_label = {}
            for label, cells in by_label.items():
                yield label, cells, None
            continue

        label = getattr(ann, 'label', None)
        if not _has_class(label):
            continue
        if isinstance(ann, PatchAnnotation):
            cid = cell_at(*ann.get_centroid(), rect, grid_hw)
            cells = [cid] if cid is not None else []
        else:
            cells = cells_covered(ann, rect, grid_hw)
        if cells:
            yield label, cells, ann


def label_map_from_scores(best, grid_hw, out_hw, reject):
    """Per-pixel label map at ``out_hw`` from per-class grid scores ``best`` [C, N].

    Bilinearly upsamples EACH class's similarity field to the target size, then
    argmaxes + applies the reject floor there, so the boundary follows a smooth
    contour at full resolution. -1 is unlabeled; k indexes the rows of ``best``.
    """
    ups = np.stack(
        [upsample_field(np.asarray(best[c], dtype=np.float32).reshape(grid_hw), *out_hw)
         for c in range(len(best))],
        axis=0,
    )  # [C, out_h, out_w]
    return np.where(ups.max(axis=0) >= reject, ups.argmax(axis=0), -1)


def classify(engine, prototypes, grid_hw, out_hw, reject):
    """Classify ``prototypes`` (element ids) into a per-pixel label map at ``out_hw``.

    Returns ``(label_map, keys)`` (see label_map_from_scores); ``(None, [])``
    without prototypes.
    """
    best, keys = engine.class_scores(prototypes)
    if not keys:
        return None, []
    return label_map_from_scores(best, grid_hw, out_hw, reject), keys


# ----------------------------------------------------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------------------------------------------------


def _centrality(window, xs, ys):
    """[len(ys), len(xs)] distance of pixels from ``window``'s center: 0 there, 1 at its edge."""
    x0, y0, x1, y1 = window
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = max((x1 - x0) / 2.0, 1e-6), max((y1 - y0) / 2.0, 1e-6)
    return np.maximum(np.abs(ys - cy)[:, None] / half_h, np.abs(xs - cx)[None, :] / half_w)


def owned_mask(index, windows):
    """Bool [h, w] over ``windows[index]``: the pixels that window owns.

    A pixel covered by several windows belongs to the one it sits most
    centrally in (ties to the lower index), so overlapping tiles each write,
    and seed, only their own share. Computed per window, never at image size.
    """
    x0, y0, x1, y1 = windows[index]
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    mine = _centrality(windows[index], xs, ys)
    owned = np.ones(mine.shape, dtype=bool)
    for other, window in enumerate(windows):
        ix0, iy0 = max(x0, window[0]), max(y0, window[1])
        ix1, iy1 = min(x1, window[2]), min(y1, window[3])
        if other == index or ix0 >= ix1 or iy0 >= iy1:
            continue
        sy, sx = slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0)
        theirs = _centrality(window, xs[sx], ys[sy])
        owned[sy, sx] &= (mine[sy, sx] <= theirs) if index < other else (mine[sy, sx] < theirs)
    return owned


def owner_of(x, y, windows):
    """Index of the window owning the pixel under (x, y) (see owned_mask), or None."""
    px, py = np.floor(x), np.floor(y)
    best, owner = None, None
    for index, window in enumerate(windows):
        x0, y0, x1, y1 = window
        if x0 <= px < x1 and y0 <= py < y1:
            c = float(_centrality(window, np.array([px + 0.5]), np.array([py + 0.5]))[0, 0])
            if best is None or c < best:
                best, owner = c, index
    return owner


def annotations_in(annotations, window):
    """Annotations whose bounding box overlaps ``window``."""
    x0, y0, x1, y1 = window
    inside = []
    for ann in annotations:
        try:
            tl, br = ann.get_bounding_box_top_left(), ann.get_bounding_box_bottom_right()
        except Exception:
            inside.append(ann)
            continue
        if tl.x() < x1 and tl.y() < y1 and br.x() >= x0 and br.y() >= y0:
            inside.append(ann)
    return inside


def iter_owned_seeds(annotations, windows, index, grid_hw, mask_annotation=None, max_cells=None):
    """iter_seeds over ``windows[index]``, keeping only the seeds that window owns.

    A patch seeds only in the window owning its center, and a region's cells
    only where the window owns the cell center, so a seed in an overlap votes
    once. ``max_cells`` caps (strided) the cells kept per seed, which bounds the
    vectors a large polygon or mask class contributes across many tiles.
    """
    from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

    x0, y0, x1, y1 = windows[index]
    grid_h, grid_w = grid_hw
    owned = owned_mask(index, windows) if len(windows) > 1 else None
    for label, cells, ann in iter_seeds(annotations, (x0, y0, x1 - x0, y1 - y0), grid_hw,
                                        mask_annotation):
        cells = np.asarray(cells, dtype=np.int64)
        if owned is not None:
            if isinstance(ann, PatchAnnotation):
                if owner_of(*ann.get_centroid(), windows) != index:
                    continue
            else:
                gy, gx = np.divmod(cells, grid_w)
                py = ((gy + 0.5) * (y1 - y0) / grid_h).astype(np.int64)
                px = ((gx + 0.5) * (x1 - x0) / grid_w).astype(np.int64)
                cells = cells[owned[py, px]]
        if max_cells and cells.size > max_cells:
            cells = cells[np.linspace(0, cells.size - 1, max_cells).astype(np.int64)]
        if cells.size:
            yield label, cells, ann


# ----------------------------------------------------------------------------------------------------------------------
# Patch replacement
# ----------------------------------------------------------------------------------------------------------------------


def patches_where(annotations, contains):
    """Classed (non-Review) PatchAnnotations whose center passes ``contains(x, y)``."""
    from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

    return [ann for ann in annotations
            if isinstance(ann, PatchAnnotation) and _has_class(ann.label)
            and contains(*ann.get_centroid())]


def partition_patches(patches, prediction, label_id_to_class_id, origin=(0, 0)):
    """Split ``patches`` by whether the prediction agrees with them.

    A patch agrees when ``prediction`` (class ids, top-left at ``origin`` in the
    image) holds the patch's own class at its center: the mask already says
    what the point says, so the patch can be replaced without losing its label.
    Patches centered outside the prediction disagree. Returns
    ``(agree, disagree)``.
    """
    x0, y0 = origin
    h, w = prediction.shape
    agree, disagree = [], []
    for ann in patches:
        cx, cy = ann.get_centroid()
        x, y = int(cx) - x0, int(cy) - y0
        class_id = label_id_to_class_id.get(ann.label.id)
        if class_id is not None and 0 <= x < w and 0 <= y < h and prediction[y, x] == class_id:
            agree.append(ann)
        else:
            disagree.append(ann)
    return agree, disagree


# ----------------------------------------------------------------------------------------------------------------------
# Writing to the mask
# ----------------------------------------------------------------------------------------------------------------------


def _occupancy(mask_annotation, annotations, origin, shape):
    """Bool ``shape`` window (top-left ``origin``) of pixels vector annotations cover, or None.

    Reuses the MaskAnnotation's own rasterization helpers so the occupancy
    matches exactly how rasterize_annotations()/bake mark the same pixels.
    Geometries are shifted into the window, so nothing is rasterized beyond it.
    """
    from shapely.affinity import translate

    x0, y0 = origin
    h, w = shape
    geometries = []
    for ann in annotations:
        try:
            geom = mask_annotation._get_annotation_rasterization_geometry(ann)
        except Exception:
            geom = None
        if geom is None or getattr(geom, 'is_empty', False):
            continue
        minx, miny, maxx, maxy = geom.bounds
        if maxx < x0 or maxy < y0 or minx > x0 + w or miny > y0 + h:
            continue
        geometries.append(translate(geom, -x0, -y0) if (x0 or y0) else geom)
    if not geometries:
        return None
    try:
        occupied = mask_annotation._fast_rasterize(geometries, w, h, mode="rasterio")
    except Exception:
        return None
    return occupied if occupied.any() else None


def write_prediction(mask_annotation, prediction, origin, annotations, replaced=(),
                     region=None, history_action=None):
    """Paint a class-id prediction window (top-left at ``origin``) into the mask.

    Enforces the app-wide invariant that a MaskAnnotation never holds a label
    behind a vector annotation (mask class A hiding under a patch/polygon of
    class B): the prediction is zeroed wherever one sits, and any mask already
    there is cleared. ``replaced`` patches (see partition_patches) count as
    gone, so the mask fills their footprint; deleting them is the caller's job,
    so it can fold the deletion into its own undo action (or not). ``region``
    (bool, window-shaped) limits the pixels this call may change, so
    overlapping tiles each write only what they own. ``prediction`` is
    modified in place.
    """
    x0, y0 = origin
    width = mask_annotation.mask_data.shape[1]
    if region is not None:
        prediction[~region] = 0

    def flat(selected):
        ys, xs = np.nonzero(selected)
        return (ys.astype(np.int64) + y0) * width + (xs + x0)

    replaced_ids = {a.id for a in replaced}
    vectors = [a for a in annotations
               if not getattr(a, 'is_mask_annotation', False) and a.id not in replaced_ids]
    occupied = _occupancy(mask_annotation, vectors, origin, prediction.shape)
    if occupied is not None:
        if region is not None:
            occupied &= region
        # Don't paint under the remaining vectors; clear any mask already there
        # (respects the LOCK_BIT, so protected pixels are left untouched).
        prediction[occupied] = 0
        mask_annotation.update_mask_at_indices(
            flat(occupied), 0, silent=True, history_action=history_action)

    freed = _occupancy(mask_annotation, replaced, origin, prediction.shape)
    if freed is not None:
        if occupied is not None:
            freed &= ~occupied
        if region is not None:
            freed &= region
        # Raw write: also lifts the LOCK_BIT rasterize_annotations() left under
        # the patch, which the lock-respecting update below would skip.
        applied = mask_annotation.apply_flat_values_at_indices(flat(freed), prediction[freed])
        if applied is not None and history_action is not None:
            history_action.add_change(applied["flat_indices"], applied["before_values"],
                                      applied["after_values"], update_rect=applied["update_rect"])

    # Merge over the prediction's bounding box only; the update skips locked pixels.
    rows, cols = np.nonzero(prediction)
    if rows.size == 0:
        return
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    tile = prediction[r0:r1, c0:c1]
    current = mask_annotation.mask_data[y0 + r0:y0 + r1, x0 + c0:x0 + c1]
    mask_annotation.update_mask_with_mask(np.where(tile > 0, tile, current),
                                          (int(x0 + c0), int(y0 + r0)),
                                          history_action=history_action)
