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

    Raw ViT patch tokens carry a large channel-wise mean plus a positional
    component, so cosine similarity between ANY two patches of one image sits
    high (~0.78 mean measured on DINOv2-with-registers) and varies with
    spatial distance even when the content is identical. Two consequences:
    the threshold has almost no usable range, and nearby-but-different
    patches can outscore far-but-identical ones.

    Centering and scaling each channel by its statistics ACROSS THE REGION
    removes that shared offset, so the remaining variation is what actually
    distinguishes patches within this crop. Measured over four content pairs:
    mean ROC-AUC 0.928 -> 0.953, correlation of similarity with spatial
    distance on a homogeneous canvas -0.60 -> -0.29, and the fraction of the
    work area passing a fixed threshold from one click tightens from a
    content-dependent 20-80% to 10-22%.

    NOTE: dropping leading principal components is the obvious next step and
    is WRONG here — the class-discriminative signal lives in those components
    (removing the top 1 or top 8 collapsed mean AUC to ~0.50, i.e. chance).

    Statistics are region-local by design; they are deliberately NOT applied
    to feature maps persisted to disk, which stay raw so they remain
    comparable across crops.
    """
    features = np.asarray(features, dtype=np.float32)
    features = features - features.mean(axis=0, keepdims=True)
    features = features / (features.std(axis=0, keepdims=True) + 1e-6)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def build_query_engine(feature_map, standardize=True):
    """QueryEngine over an [h, w, C] feature map; returns ``(engine, (h, w))``."""
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


def classify(engine, prototypes, grid_hw, out_hw, reject):
    """Classify ``prototypes`` into a per-pixel label map at ``out_hw``.

    Bilinearly upsamples EACH class's similarity field to the target size, then
    argmaxes + applies the reject floor there, so the boundary follows a smooth
    contour at full resolution.

    Returns ``(label_map, keys)``: an int [out_h, out_w] map where -1 is
    unlabeled and k indexes ``keys``; ``(None, [])`` without prototypes.
    """
    best, keys = engine.class_scores(prototypes)
    if not keys:
        return None, []
    ups = np.stack(
        [upsample_field(best[c].reshape(grid_hw).astype(np.float32), *out_hw)
         for c in range(len(keys))],
        axis=0,
    )  # [C, out_h, out_w]
    label_map = np.where(ups.max(axis=0) >= reject, ups.argmax(axis=0), -1)
    return label_map, keys


def partition_patches(annotations, prediction_mask, label_id_to_class_id, rect):
    """Split the patches centered in ``rect`` by whether the prediction agrees.

    A patch agrees when ``prediction_mask`` (full-image class ids) holds the
    patch's own class at its center: the mask already says what the point says,
    so the patch can be replaced without losing its label. Returns
    ``(agree, disagree)``.
    """
    from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

    left, top, width, height = rect
    h, w = prediction_mask.shape
    agree, disagree = [], []
    for ann in annotations:
        if not isinstance(ann, PatchAnnotation) or not _has_class(ann.label):
            continue
        cx, cy = ann.get_centroid()
        if not (left <= cx < left + width and top <= cy < top + height):
            continue
        x, y = int(cx), int(cy)
        class_id = label_id_to_class_id.get(ann.label.id)
        if class_id is not None and 0 <= x < w and 0 <= y < h and prediction_mask[y, x] == class_id:
            agree.append(ann)
        else:
            disagree.append(ann)
    return agree, disagree


# ----------------------------------------------------------------------------------------------------------------------
# Writing to the mask
# ----------------------------------------------------------------------------------------------------------------------


def occupancy_indices(mask_annotation, annotations):
    """Flat pixel indices covered by vector ``annotations``, or None if none.

    Reuses the MaskAnnotation's own rasterization helpers so the occupancy
    matches exactly how rasterize_annotations()/bake mark the same pixels.
    """
    geometries = []
    for ann in annotations:
        try:
            geom = mask_annotation._get_annotation_rasterization_geometry(ann)
        except Exception:
            geom = None
        if geom is not None and not getattr(geom, 'is_empty', False):
            geometries.append(geom)
    if not geometries:
        return None
    try:
        h, w = mask_annotation.mask_data.shape
        occ = mask_annotation._fast_rasterize(geometries, w, h, mode="rasterio")
    except Exception:
        return None
    idx = np.flatnonzero(occ.ravel())
    return idx if idx.size else None


def write_prediction(mask_annotation, prediction_mask, annotations, rect,
                     replace_patches=False, history_action=None):
    """Paint a full-image class-id prediction into the mask.

    Enforces the app-wide invariant that a MaskAnnotation never holds a label
    behind a vector annotation (mask class A hiding under a patch/polygon of
    class B): the prediction is zeroed wherever one sits, and any mask already
    there is cleared. With ``replace_patches``, the patches in ``rect`` the
    prediction agrees with (see partition_patches) are treated as gone, so the
    mask fills their footprint. ``prediction_mask`` is modified in place.

    Returns ``(replaced, kept)`` patches; deleting ``replaced`` is the caller's
    job, so it can fold the deletion into its own undo action (or not).
    """
    vectors = [a for a in annotations if not getattr(a, 'is_mask_annotation', False)]
    replaced, kept = [], []
    if replace_patches:
        replaced, kept = partition_patches(vectors, prediction_mask,
                                           mask_annotation.label_id_to_class_id_map, rect)
        replaced_ids = {a.id for a in replaced}
        vectors = [a for a in vectors if a.id not in replaced_ids]

    occupied = occupancy_indices(mask_annotation, vectors)
    if occupied is not None:
        # Don't paint under the remaining vectors; clear any mask already there
        # (respects the LOCK_BIT, so protected pixels are left untouched).
        prediction_mask.ravel()[occupied] = 0
        mask_annotation.update_mask_at_indices(
            occupied, 0, silent=True, history_action=history_action)

    freed = occupancy_indices(mask_annotation, replaced)
    if freed is not None:
        if occupied is not None:
            freed = np.setdiff1d(freed, occupied, assume_unique=True)
        # Raw write: also lifts the LOCK_BIT rasterize_annotations() left under
        # the patch, which the lock-respecting update below would skip.
        applied = mask_annotation.apply_flat_values_at_indices(freed, prediction_mask.ravel()[freed])
        if applied is not None and history_action is not None:
            history_action.add_change(applied["flat_indices"], applied["before_values"],
                                      applied["after_values"], update_rect=applied["update_rect"])

    mask_annotation.update_mask_with_prediction_mask(prediction_mask, history_action=history_action)
    return replaced, kept
