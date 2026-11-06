# Utility to refine existing bounding boxes using SAM

import os
import csv
import sys
import time
import uuid
import pickle
import argparse
import warnings
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

# Before rasterio loads GDAL: decode compressed TIFF tiles on every core
os.environ.setdefault('GDAL_NUM_THREADS', 'ALL_CPUS')

import cv2
import numpy as np
import torch

try:
    import ujson as json
except ImportError:
    import json

# Plain photos have no georeferencing; rasterio warns on every open. Set once
# here: warnings.catch_warnings() is not thread-safe, and images open on threads.
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
except ImportError:
    pass

# The SAM deploy dialog's names, so either those or a weights file work here
MODELS = {
    'mobilesam': 'mobile_sam.pt',
    'sam-base': 'sam_b.pt',
    'sam-large': 'sam_l.pt',
    'sam-huge': 'sam_h.pt',
    'sam 2 tiny': 'sam2_t.pt',
    'sam 2 small': 'sam2_s.pt',
    'sam 2 base': 'sam2_b.pt',
    'sam 2 large': 'sam2_l.pt',
    'sam 2.1 tiny': 'sam2.1_t.pt',
    'sam 2.1 small': 'sam2.1_s.pt',
    'sam 2.1 base': 'sam2.1_b.pt',
    'sam 2.1 large': 'sam2.1_l.pt',
    'sam 3': 'sam3.pt',
    'sam 3.1 multiplex': 'sam3.1_multiplex.pt',
}

CV2_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

REPORT_FIELDS = [
    'image_path', 'annotation_id', 'label', 'verified', 'encode',
    'old_x0', 'old_y0', 'old_x1', 'old_y1',
    'new_x0', 'new_y0', 'new_x1', 'new_y1',
    'iou', 'change_pct', 'growth_pct', 'sam_score', 'action', 'review_id',
]


# ----------------------------------------------------------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------------------------------------------------------


def load_project(path):
    """The project's dict, from JSON if the extension says so, else pickle (as the app does)."""
    if os.path.splitext(path)[1].lower() == '.json':
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_project(path, project):
    """Write beside path and swap in, so a failed write never truncates a project.

    Mirrors IO/QtSaveProject.write_project_file.
    """
    temp_path = f"{path}.tmp"
    try:
        if os.path.splitext(path)[1].lower() == '.json':
            with open(temp_path, 'w', encoding='utf-8') as file:
                json.dump(project, file, indent=4)
        else:
            with open(temp_path, 'wb') as file:
                pickle.dump(project, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def is_review(annotation):
    return (annotation.get('label_short_code') == 'Review'
            and annotation.get('label_long_code') == 'Review')


def review_annotation(original, box, score, change, grown, model):
    """A new, unverified RectangleAnnotation at ``box`` with ``original``'s label.

    SAM's predicted IoU is its machine confidence, so the app shows the score
    and lists it among the unverified annotations to check. data['sam_refine']
    links it to the original (see pending_pairs); a dict, so the app keeps it as
    raw data rather than promoting it to a metadata field.
    """
    return {
        'type': 'RectangleAnnotation',
        'id': str(uuid.uuid4()),
        'label_short_code': original['label_short_code'],
        'label_long_code': original['label_long_code'],
        'annotation_color': original.get('annotation_color'),
        'image_path': original['image_path'],
        'label_id': original['label_id'],
        'data': {'sam_refine': {
            'original_id': original.get('id'),
            'model': model,
            'change_pct': round(float(change), 2),
            'growth_pct': round(float(grown), 2),
        }},
        'machine_confidence': {original['label_short_code']: round(float(score), 4)},
        'verified': False,
        'top_left': (box[0], box[1]),
        'bottom_right': (box[2], box[3]),
    }


def pending_pairs(project):
    """Ids of both boxes of every refit pair from an earlier run still in the project.

    Refitting either again would add a third box for the same object, so both
    are skipped until the pair is settled by deleting one of them.
    """
    ids = set()
    for annotations in (project.get('annotations') or {}).values():
        present = {ann.get('id') for ann in annotations}
        for ann in annotations:
            link = (ann.get('data') or {}).get('sam_refine')
            if isinstance(link, dict) and link.get('original_id') in present:
                ids.update((ann.get('id'), link['original_id']))
    return ids


def resolve_image_path(path, image_root, index):
    """``path`` if it exists, else a file of the same name under ``image_root``."""
    if os.path.exists(path):
        return path
    if not image_root:
        return None
    if not index:
        for root, _, files in os.walk(image_root):
            for name in files:
                index.setdefault(name.lower(), os.path.join(root, name))
        index.setdefault('', '')  # walked, even if empty
    return index.get(os.path.basename(path).lower())


# ----------------------------------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------------------------------


def to_bgr(bands):
    """BGR uint8 [H, W, 3] from rasterio's [C, H, W] bands (gray repeated, >8-bit stretched)."""
    bands = bands[:3] if bands.shape[0] >= 3 else np.repeat(bands[:1], 3, axis=0)
    if bands.dtype != np.uint8:
        bands = bands.astype(np.float32)
        low, high = np.nanmin(bands), np.nanmax(bands)
        bands = ((bands - low) * (255.0 / max(high - low, 1e-6))).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(bands.transpose(1, 2, 0)[..., ::-1])


class ImageSource:
    """One image's pixels in BGR, whole or by window.

    Opened on a worker thread. An image up to ``max_full_pixels`` is read whole
    there; a larger one keeps no pixels and is read by window on a thread of its
    own, which also owns its rasterio handle (handles are not thread-safe).

    EXIF orientation is ignored, as the app's rasterio reads ignore it, so the
    pixels line up with the annotations.
    """

    def __init__(self, path, max_full_pixels):
        import rasterio

        self.path = path
        with rasterio.open(path) as src:
            self.width, self.height = src.width, src.height
            self.windowed = self.width * self.height > max_full_pixels
            self.full = None if self.windowed else self._read_full(src)
        self._src = None

    def _read_full(self, src):
        if os.path.splitext(self.path)[1].lower() in CV2_EXTENSIONS:
            bgr = cv2.imread(self.path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is not None and bgr.shape[:2] == (self.height, self.width):
                return bgr
        return to_bgr(src.read())

    def _read_window(self, window):
        import rasterio
        from rasterio.windows import Window

        if self._src is None:
            self._src = rasterio.open(self.path)
        x0, y0, x1, y1 = window
        indexes = [1, 2, 3] if self._src.count >= 3 else [1]
        return to_bgr(self._src.read(indexes, window=Window(x0, y0, x1 - x0, y1 - y0)))

    def _close(self):
        if self._src is not None:
            self._src.close()
            self._src = None

    def windows(self, windows, lookahead=3):
        """Yield the BGR pixels of each ``(x0, y0, x1, y1)`` in order, reading ahead."""
        if not self.windowed:
            for x0, y0, x1, y1 in windows:
                yield np.ascontiguousarray(self.full[y0:y1, x0:x1])
            return
        reader = ThreadPoolExecutor(max_workers=1)
        pending = deque()
        try:
            queue = iter(windows)
            for window in queue:
                pending.append(reader.submit(self._read_window, window))
                if len(pending) > lookahead:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
            reader.submit(self._close).result()
            reader.shutdown()


# ----------------------------------------------------------------------------------------------------------------------
# Encode planning
# ----------------------------------------------------------------------------------------------------------------------


def square_window(x0, y0, x1, y1, side, width, height):
    """A ``side``-square around the rect's center, shifted inside the image (and clipped to it)."""
    side_w, side_h = min(side, width), min(side, height)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(min(max(cx - side_w / 2.0, 0), width - side_w)))
    top = int(round(min(max(cy - side_h / 2.0, 0), height - side_h)))
    return left, top, int(left + side_w), int(top + side_h)


def pack_crops(boxes, ids, width, height, imgsz, min_object_px, context, min_crop):
    """Group boxes into shared square crops; returns ``[(window, ids)]``.

    Each box brings its context window (the box grown by ``context`` of its size
    on every side) and a cap on the crop side: the largest side at which its
    short side still spans ``min_object_px`` pixels once the crop is resized to
    ``imgsz``. A box joins the existing crop it grows least, if the grown crop
    stays within every member's cap; otherwise it starts a crop of its own.
    Crops are squared up to their long side (the encoder pads to a square
    anyway, so that context is free) and never below ``min_crop``.
    """
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    win = np.stack([
        np.clip(boxes[:, 0] - context * bw, 0, width),
        np.clip(boxes[:, 1] - context * bh, 0, height),
        np.clip(boxes[:, 2] + context * bw, 0, width),
        np.clip(boxes[:, 3] + context * bh, 0, height),
    ], axis=1)
    own = np.maximum(win[:, 2] - win[:, 0], win[:, 3] - win[:, 1])
    cap = np.maximum.reduce([imgsz * np.minimum(bw, bh) / float(min_object_px), own,
                             np.full(len(boxes), float(min_crop))])

    rects = np.zeros((len(boxes), 4))
    caps = np.zeros(len(boxes))
    members = []
    # Row-major, so a crop fills with neighbours before the next starts
    for k in np.lexsort((win[:, 0], win[:, 1])):
        g = len(members)
        if g:
            r = rects[:g]
            side = np.maximum(np.maximum(r[:, 2], win[k, 2]) - np.minimum(r[:, 0], win[k, 0]),
                              np.maximum(r[:, 3], win[k, 3]) - np.minimum(r[:, 1], win[k, 1]))
            fits = side <= np.minimum(caps[:g], cap[k])
            if fits.any():
                j = np.flatnonzero(fits)[np.argmin(side[fits])]
                rects[j, :2] = np.minimum(rects[j, :2], win[k, :2])
                rects[j, 2:] = np.maximum(rects[j, 2:], win[k, 2:])
                caps[j] = min(caps[j], cap[k])
                members[j].append(k)
                continue
        rects[g] = win[k]
        caps[g] = cap[k]
        members.append([k])

    plans = []
    for j, group in enumerate(members):
        x0, y0, x1, y1 = rects[j]
        side = int(np.ceil(max(x1 - x0, y1 - y0, min_crop)))
        plans.append((square_window(x0, y0, x1, y1, side, width, height), ids[np.asarray(group)]))
    return plans


def plan_encodes(boxes, source, imgsz, args, context):
    """``[(window, ids, kind)]``: which encode each box is decoded against."""
    width, height = source.width, source.height
    short = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    whole = np.zeros(len(boxes), dtype=bool)
    if not source.windowed:
        if args.mode == 'image':
            whole[:] = True
        elif args.mode == 'auto':
            whole = short * imgsz / float(max(width, height)) >= args.min_object_px

    plans = []
    if whole.any():
        plans.append(((0, 0, width, height), np.flatnonzero(whole), 'image'))
    rest = np.flatnonzero(~whole)
    if rest.size:
        for window, ids in pack_crops(boxes[rest], rest, width, height, imgsz,
                                      args.min_object_px, context, args.min_crop):
            plans.append((window, ids, 'crop'))
    return plans


# ----------------------------------------------------------------------------------------------------------------------
# SAM
# ----------------------------------------------------------------------------------------------------------------------


def load_predictor(model, imgsz, device, fp32):
    """An ultralytics SAM / SAM 2 / SAM 3 predictor, chosen as the app's SAM dialog does."""
    from ultralytics.models.sam import Predictor, SAM2Predictor, SAM3Predictor

    weights = MODELS.get(model.strip().lower(), model)
    try:
        from coralnet_toolbox.paths import resolve_weights
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from coralnet_toolbox.paths import resolve_weights
        except ImportError:
            resolve_weights = None
    if resolve_weights is not None:
        # The app's weights folder: reuse its downloads, and download into it
        weights = resolve_weights(weights)

    tag = f"{model} {os.path.basename(weights)}".lower()
    for char in ' _-':
        tag = tag.replace(char, '')
    family = 'sam3' if 'sam3' in tag else 'sam2' if 'sam2' in tag else 'sam'

    # SAM 3 is stride 14; SAM 2's Hiera raises on sizes that aren't a multiple of 32
    stride = 14 if family == 'sam3' else 32
    imgsz = max(stride, int(round(imgsz / stride)) * stride)
    # MobileSAM has no FP16, and SAM's FP16 on the CPU is far slower than FP32
    fp16 = not fp32 and 'mobilesam' not in tag and str(device).lower() != 'cpu'

    overrides = dict(task='segment', mode='predict', imgsz=imgsz, model=weights, conf=0.0,
                     device=device, retina_masks=False, quantize=16 if fp16 else 32,
                     save=False, show=False, save_txt=False, verbose=False)
    cls = {'sam3': SAM3Predictor, 'sam2': SAM2Predictor}.get(family, Predictor)
    return cls(overrides=overrides), imgsz, family, fp16


def encode(predictor, image):
    """Encode one BGR image; its features stay on ``predictor.features``."""
    predictor.reset_image()
    predictor.set_image(image)


@torch.inference_mode()
def decode(predictor, boxes, crop_hw, chunk):
    """Yield ``(start, logits [n, L, L], scores [n])`` for box prompts in crop pixels.

    Calls the prompt and mask decoders directly, as inference_features does,
    but stops at the low-resolution logits. inference_features would upsample
    every mask to the crop's full size to find its box (see mask_boxes). The
    chunk halves on an out-of-memory error, as predict_from_results does.
    """
    dst = tuple(predictor.imgsz)
    start = 0
    while start < len(boxes):
        try:
            prompts = predictor._prepare_prompts(dst, crop_hw, bboxes=boxes[start:start + chunk])
            logits, scores = predictor._inference_features(predictor.features, *prompts,
                                                           multimask_output=False)
        except torch.cuda.OutOfMemoryError:
            if chunk <= 1:
                raise
            chunk //= 2
            torch.cuda.empty_cache()
            continue
        yield start, logits, scores
        start += len(logits)


def _extent(profile):
    """``(lo, hi)`` of a 1D logit profile's positive run, in cell-edge units, or None.

    Cell k's logit sits at k + 0.5. Each edge is put where the logit crosses
    zero between the last negative cell and the first positive one, by linear
    interpolation, which is where a bilinear upsample puts the mask's edge.
    """
    idx = np.flatnonzero(profile > 0)
    if idx.size == 0:
        return None
    i0, i1 = idx[0], idx[-1]
    lo = 0.0 if i0 == 0 else i0 - 0.5 + profile[i0 - 1] / (profile[i0 - 1] - profile[i0])
    last = len(profile) - 1
    hi = float(len(profile)) if i1 == last else i1 + 0.5 + profile[i1] / (profile[i1] - profile[i1 + 1])
    return lo, hi


def mask_boxes(logits, crop_hw, prompts, keep_islands):
    """Refit boxes in crop pixels from SAM's low-resolution mask logits.

    Works on the decoder's grid (256 cells a side for SAM and SAM 2) instead of
    the full-size masks: those are a float per crop pixel per box, gigabytes for
    a few hundred boxes on a 24 MP image, to find four numbers each. The valid
    (unpadded) part of the grid is cut out exactly as ultralytics' scale_masks
    cuts it, so cell coordinates map to crop pixels as its upsample does.

    Unless ``keep_islands``, only the connected piece of the mask that overlaps
    the prompt box most is kept, so a stray blob elsewhere in the crop does not
    stretch the refit.

    Returns ``(boxes [n, 4], edge [n, 4])``: boxes are NaN where SAM found
    nothing; edge is True for each side (left, top, right, bottom) the mask
    reaches the crop's border on.
    """
    n, size = logits.shape[0], logits.shape[-1]
    h, w = crop_hw
    gain = min(size / h, size / w)
    vw = size - int(round(size - w * gain + 0.1))
    vh = size - int(round(size - h * gain + 0.1))
    grid = logits[:, :vh, :vw].float().contiguous().cpu().numpy()
    sx, sy = w / vw, h / vh

    boxes = np.full((n, 4), np.nan)
    edge = np.zeros((n, 4), dtype=bool)
    for i in range(n):
        cells = grid[i]
        positive = cells > 0
        if not positive.any():
            continue
        if not keep_islands:
            count, components = cv2.connectedComponents(positive.view(np.uint8), connectivity=8)
            if count > 2:
                px0, py0, px1, py1 = prompts[i]
                c0 = int(np.clip(np.floor(px0 / sx), 0, vw - 1))
                r0 = int(np.clip(np.floor(py0 / sy), 0, vh - 1))
                c1 = int(np.clip(np.ceil(px1 / sx), c0 + 1, vw))
                r1 = int(np.clip(np.ceil(py1 / sy), r0 + 1, vh))
                votes = np.bincount(components[r0:r1, c0:c1].ravel(), minlength=count)
                votes[0] = 0
                if not votes.any():
                    votes = np.bincount(components.ravel(), minlength=count)
                    votes[0] = 0
                cells = np.where(positive & (components != votes.argmax()), np.float32(-1), cells)
        x = _extent(cells.max(axis=0))
        y = _extent(cells.max(axis=1))
        if x is None or y is None:
            continue
        boxes[i] = (x[0] * sx, y[0] * sy, x[1] * sx, y[1] * sy)
        edge[i] = (x[0] < 1.0, y[0] < 1.0, x[1] > vw - 1.0, y[1] > vh - 1.0)
    return boxes, edge


def decode_boxes(predictor, prompts, crop_hw, args, timers):
    """``(boxes, scores, edge)`` for box prompts against the encoded crop; see mask_boxes."""
    boxes = np.full((len(prompts), 4), np.nan)
    scores = np.zeros(len(prompts))
    edge = np.zeros((len(prompts), 4), dtype=bool)
    tick = time.perf_counter()
    for start, logits, score in decode(predictor, prompts, crop_hw, args.chunk):
        stop = start + len(logits)
        scores[start:stop] = score.float().cpu().numpy()
        timers['sam'] += time.perf_counter() - tick
        tick = time.perf_counter()
        boxes[start:stop], edge[start:stop] = mask_boxes(logits, crop_hw, prompts[start:stop],
                                                         args.keep_islands)
        timers['post'] += time.perf_counter() - tick
        tick = time.perf_counter()
    timers['sam'] += time.perf_counter() - tick
    return boxes, scores, edge


def run_plans(predictor, source, plans, boxes, refits, scores, how, clipped, args, timers):
    """Encode each planned window and decode its boxes into ``refits`` / ``scores``.

    SAM reads a box prompt as "the object filling this box", so a loose box
    comes back with some of its surroundings. Each refit is therefore given back
    as the next prompt, up to ``--iterations`` times, against the same encoded
    features (decoder only, so nearly free); a box stops once it moves less than
    a pixel, and keeps its last refit if a later prompt finds nothing.

    A refit that reaches a crop border which is not also the image's border has
    probably been cut off by the crop; it is flagged in ``clipped``.
    """
    windows = [window for window, _, _ in plans]
    tick = time.perf_counter()
    for (window, ids, kind), pixels in zip(plans, source.windows(windows)):
        timers['read'] += time.perf_counter() - tick

        tick = time.perf_counter()
        x0, y0, x1, y1 = window
        offset = np.array([x0, y0, x0, y0], dtype=np.float32)
        crop_hw = pixels.shape[:2]
        encode(predictor, pixels)
        timers['sam'] += time.perf_counter() - tick

        prompts = boxes[ids] - offset
        found = np.full((len(ids), 4), np.nan)
        score = np.zeros(len(ids))
        edge = np.zeros((len(ids), 4), dtype=bool)
        active = np.arange(len(ids))
        for _ in range(max(1, args.iterations)):
            step, step_score, step_edge = decode_boxes(predictor, prompts[active], crop_hw, args, timers)
            ok = ~np.isnan(step).any(axis=1)
            done = active[ok]
            moved = np.abs(step[ok] - prompts[done]).max(axis=1) if ok.any() else np.zeros(0)
            found[done], score[done], edge[done] = step[ok], step_score[ok], step_edge[ok]
            prompts[done] = step[ok].astype(np.float32)
            active = done[moved >= 1.0]
            if not active.size:
                break

        at_border = np.array([x0 <= 0, y0 <= 0, x1 >= source.width, y1 >= source.height])
        refits[ids] = found + offset
        scores[ids] = score
        how[ids] = kind
        clipped[ids] = (edge & ~at_border).any(axis=1)
        tick = time.perf_counter()


def refine_image(predictor, source, boxes, imgsz, args, timers):
    """``(refits [n, 4], scores [n], how [n], clipped [n])`` for one image's boxes.

    Boxes cut off by their crop get one more try, alone in crops with three
    times the context; one still cut off then stays flagged.
    """
    n = len(boxes)
    refits = np.full((n, 4), np.nan)
    scores = np.zeros(n)
    how = np.full(n, '', dtype=object)
    clipped = np.zeros(n, dtype=bool)

    plans = plan_encodes(boxes, source, imgsz, args, args.context)
    run_plans(predictor, source, plans, boxes, refits, scores, how, clipped, args, timers)

    retry = np.flatnonzero(clipped)
    if retry.size:
        plans = [(window, ids, 'crop-retry') for window, ids in
                 pack_crops(boxes[retry], retry, source.width, source.height, imgsz,
                            args.min_object_px, 3 * args.context, args.min_crop)]
        run_plans(predictor, source, plans, boxes, refits, scores, how, clipped, args, timers)
    return refits, scores, how, clipped


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def iou(a, b):
    """IoU of matching rows of two [n, 4] xyxy arrays."""
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def select_rectangles(project, args):
    """``([(image_path, [rectangle dicts])], pending)`` to refine, after the filters.

    ``pending`` counts the rectangles skipped as part of an unsettled refit pair.
    """
    wanted = set(args.labels or [])
    unwanted = set(args.exclude_labels or [])
    video = {img.get('path') for img in project.get('images') or []
             if isinstance(img, dict) and img.get('raster_type') == 'VideoRaster'}
    pending = pending_pairs(project)
    skipped = 0

    jobs = []
    for image_path, annotations in (project.get('annotations') or {}).items():
        if '::frame_' in image_path or image_path in video:
            continue
        rects = []
        for ann in annotations:
            if ann.get('type') != 'RectangleAnnotation':
                continue
            if ann.get('id') in pending:
                skipped += 1
                continue
            codes = {ann.get('label_short_code'), ann.get('label_long_code')}
            if wanted and not codes & wanted:
                continue
            if codes & unwanted:
                continue
            if is_review(ann) and not args.include_review:
                continue
            if args.skip_verified and ann.get('verified'):
                continue
            rects.append(ann)
        if rects:
            jobs.append((image_path, rects))
    return (jobs[:args.limit] if args.limit else jobs), skipped


def growth(refits, boxes):
    """Percent of each refit's area that lies outside its original box."""
    iw = np.clip(np.minimum(refits[:, 2], boxes[:, 2]) - np.maximum(refits[:, 0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(refits[:, 3], boxes[:, 3]) - np.maximum(refits[:, 1], boxes[:, 1]), 0, None)
    area = (refits[:, 2] - refits[:, 0]) * (refits[:, 3] - refits[:, 1])
    return 100.0 * (1.0 - np.where(area > 0, iw * ih / np.maximum(area, 1e-9), 1.0))


def decide(change, grown, score, failure, args):
    """The action for one refit; see the module docstring."""
    if failure:
        return failure
    if score < args.min_score:
        return 'failed: low score'
    if change <= args.min_change:
        return 'unchanged'
    if grown > args.max_growth:
        return 'review' if change <= args.ignore_threshold else 'ignored'
    if change <= args.review_threshold:
        return 'updated'
    if change <= args.ignore_threshold:
        return 'review'
    return 'ignored'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refit a project's Rectangle annotations with SAM box prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('project', help="Project file (.bin or .json)")
    parser.add_argument('--output-dir', help="Folder for the refined project, report and figures "
                                             "(default: <project>_refined_<time> beside the project)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Write into --output-dir even if it is not empty (files of the same name "
                             "are replaced; nothing else is removed)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Write the report and figures, not the project")

    group = parser.add_argument_group('figures')
    group.add_argument('--no-figures', action='store_true', help="Skip the figures folder")
    group.add_argument('--examples', type=int, default=24, help="Boxes on each action's example sheet")
    group.add_argument('--overlays', type=int, default=12, help="Whole images drawn with every box")
    group.add_argument('--seed', type=int, default=0, help="Seed for which whole images are drawn")

    group = parser.add_argument_group('model')
    group.add_argument('--model', default='sam2.1_b.pt',
                       help="Weights file or SAM dialog name (e.g. 'SAM 2.1 Large', sam3.pt)")
    group.add_argument('--imgsz', type=int, default=1024, help="Encoder input size")
    group.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    group.add_argument('--fp32', action='store_true', help="Run in FP32 on the GPU (FP16 by default)")
    group.add_argument('--chunk', type=int, default=256, help="Box prompts decoded per batch")
    group.add_argument('--iterations', type=int, default=3,
                       help="Times each refit is fed back as the prompt (decoder only); 1 = no feedback")

    group = parser.add_argument_group('selection')
    group.add_argument('--labels', nargs='+', help="Only rectangles with these label codes (short or long)")
    group.add_argument('--exclude-labels', nargs='+', help="Skip rectangles with these label codes")
    group.add_argument('--include-review', action='store_true', help="Also refine rectangles labelled Review")
    group.add_argument('--skip-verified', action='store_true', help="Leave verified rectangles alone")
    group.add_argument('--limit', type=int, help="Only the first N images with selected rectangles")
    group.add_argument('--image-root', help="Folder to find moved images in, by file name")

    group = parser.add_argument_group('decision (percent, change = 1 - IoU(old, new))')
    group.add_argument('--min-change', type=float, default=0.0,
                       help="At or below this the box is left untouched")
    group.add_argument('--review-threshold', type=float, default=70.0,
                       help="At or below this the box is moved in place")
    group.add_argument('--max-growth', type=float, default=10.0,
                       help="Refits with more of their area than this outside the original box go "
                            "to review whatever their change (100 turns this off)")
    group.add_argument('--ignore-threshold', type=float, default=100.0,
                       help="At or below this (and above --review-threshold) the refit is added, "
                            "unverified, beside the original; "
                            "above it the refit is dropped")
    group.add_argument('--min-score', type=float, default=0.5,
                       help="Refits with a lower SAM predicted IoU (0-1) are dropped; on the loose-box "
                            "benchmark those under 0.8 were no better than the original on average")

    group = parser.add_argument_group('encoding')
    group.add_argument('--mode', choices=('auto', 'image', 'crop'), default='auto',
                       help="auto: whole-image encode for boxes it shows large enough, crops for the "
                            "rest; image: whole image only; crop: crops only")
    group.add_argument('--min-object-px', type=float, default=64,
                       help="Smallest short side (encoder pixels) a box is shown to SAM at")
    group.add_argument('--context', type=float, default=0.5,
                       help="Crop context around each box, as a fraction of its size per side")
    group.add_argument('--min-crop', type=int, default=256,
                       help="Smallest crop side (image pixels), which caps the zoom at imgsz / min-crop")
    group.add_argument('--keep-islands', action='store_true',
                       help="Fit the whole mask, not just the piece that overlaps the box")
    group.add_argument('--max-full-pixels', type=float, default=64e6,
                       help="Larger images are read by window and use crops only")
    group.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                       help="Image reading threads (reads run ahead of the GPU)")
    group.add_argument('--quiet', action='store_true', help="No per-image lines")

    args = parser.parse_args()
    if not args.min_change <= args.review_threshold <= args.ignore_threshold:
        parser.error("need --min-change <= --review-threshold <= --ignore-threshold")
    return args


def output_folder(args):
    """The run's output folder, created; exits if it holds files and --overwrite is not given."""
    root, _ = os.path.splitext(os.path.abspath(args.project))
    folder = args.output_dir or f"{root}_refined_{time.strftime('%Y%m%dT%H%M%S')}"
    if os.path.isdir(folder) and os.listdir(folder) and not args.overwrite:
        sys.exit(f"{folder} is not empty; pass --overwrite to write into it")
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    args = parse_args()
    started = time.perf_counter()
    project = load_project(args.project)
    jobs, pending = select_rectangles(project, args)
    if pending:
        print(f"Skipping {pending} rectangle(s) in refit pairs from an earlier run "
              f"(delete either box of a pair to settle it)")
    if not jobs:
        sys.exit("No Rectangle annotations match the selection")

    folder = output_folder(args)
    stem, ext = os.path.splitext(os.path.basename(args.project))
    output = os.path.join(folder, f"{stem}_refined{ext}")
    report_path = os.path.join(folder, 'report.csv')
    print(f"{sum(len(r) for _, r in jobs)} rectangle(s) on {len(jobs)} image(s)")

    index = {}
    resolved = [resolve_image_path(path, args.image_root, index) for path, _ in jobs]

    predictor, imgsz, family, fp16 = load_predictor(args.model, args.imgsz, args.device, args.fp32)
    print(f"Model {args.model} ({family}, imgsz {imgsz}, {'FP16' if fp16 else 'FP32'}, {args.device})")

    totals = Counter()
    timers = Counter()
    rows = []
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = {}
    ahead = max(1, args.workers) + 1

    def submit(k):
        if k < len(jobs) and resolved[k]:
            futures[k] = pool.submit(ImageSource, resolved[k], args.max_full_pixels)

    for k in range(ahead):
        submit(k)
    try:
        for k, (image_path, rects) in enumerate(jobs):
            submit(k + ahead)
            name = os.path.basename(image_path)
            tick = time.perf_counter()
            try:
                if not resolved[k]:
                    raise FileNotFoundError("image not found (see --image-root)")
                source = futures.pop(k).result()
            except Exception as e:
                totals['failed: image unreadable'] += len(rects)
                print(f"[{k + 1}/{len(jobs)}] {name}: skipped, {e}")
                continue
            timers['read'] += time.perf_counter() - tick

            image_tick = time.perf_counter()
            old = np.array([[*r['top_left'], *r['bottom_right']] for r in rects], dtype=np.float32)
            old = np.concatenate([np.minimum(old[:, :2], old[:, 2:]), np.maximum(old[:, :2], old[:, 2:])], axis=1)
            boxes = old.copy()
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, source.width)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, source.height)
            valid = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)

            refits = np.full((len(rects), 4), np.nan)
            scores = np.zeros(len(rects))
            how = np.full(len(rects), '', dtype=object)
            clipped = np.zeros(len(rects), dtype=bool)
            try:
                if valid.any():
                    sub = np.flatnonzero(valid)
                    found = refine_image(predictor, source, boxes[sub], imgsz, args, timers)
                    refits[sub], scores[sub], how[sub], clipped[sub] = found
            except Exception as e:
                print(f"[{k + 1}/{len(jobs)}] {name}: SAM failed, {e}")
                valid[:] = False
            finally:
                source.full = None

            overlap = iou(old, np.nan_to_num(refits))
            grown = growth(np.nan_to_num(refits), old)
            image_counts = Counter()
            added = []
            for i, rect in enumerate(rects):
                failure = None
                if not valid[i]:
                    failure = 'failed: empty or off-image box'
                elif np.isnan(refits[i]).any():
                    failure = 'failed: no mask'
                elif clipped[i]:
                    failure = 'failed: cut off by crop'
                change = 100.0 * (1.0 - overlap[i])
                action = decide(change, grown[i], scores[i], failure, args)
                new = [round(float(v), 2) for v in refits[i]] if not failure else [''] * 4
                review_id = ''
                if action == 'updated':
                    rect['top_left'] = (new[0], new[1])
                    rect['bottom_right'] = (new[2], new[3])
                elif action == 'review':
                    duplicate = review_annotation(rect, new, scores[i], change, grown[i], args.model)
                    added.append(duplicate)
                    review_id = duplicate['id']
                image_counts[action] += 1
                rows.append({
                    'image_path': image_path, 'annotation_id': rect.get('id', ''),
                    'label': rect.get('label_short_code', ''), 'verified': rect.get('verified', ''),
                    'encode': how[i],
                    'old_x0': round(float(old[i, 0]), 2), 'old_y0': round(float(old[i, 1]), 2),
                    'old_x1': round(float(old[i, 2]), 2), 'old_y1': round(float(old[i, 3]), 2),
                    'new_x0': new[0], 'new_y0': new[1], 'new_x1': new[2], 'new_y1': new[3],
                    'iou': round(float(overlap[i]), 4) if not failure else '',
                    'change_pct': round(change, 2) if not failure else '',
                    'growth_pct': round(float(grown[i]), 2) if not failure else '',
                    'sam_score': round(float(scores[i]), 4) if not failure else '',
                    'action': action, 'review_id': review_id,
                })
            project['annotations'][image_path].extend(added)
            totals.update(image_counts)

            if not args.quiet:
                encodes = Counter(h for h in how if h)
                summary = ', '.join(f"{v} {a}" for a, v in sorted(image_counts.items()))
                encoded = ', '.join(f"{v} {h}" for h, v in sorted(encodes.items()))
                print(f"[{k + 1}/{len(jobs)}] {name}: {len(rects)} box(es) [{encoded}] | "
                      f"{summary} ({time.perf_counter() - image_tick:.2f}s)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    with open(report_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if not args.dry_run:
        save_project(output, project)
    refined = time.perf_counter() - started

    settings = dict(vars(args))
    settings.update(imgsz=imgsz, family=family, fp16=fp16)
    run = {
        'project': os.path.abspath(args.project),
        'refined_project': None if args.dry_run else os.path.abspath(output),
        'command': ' '.join(sys.argv),
        'settings': settings,
        'totals': dict(sorted(totals.items())),
        'seconds': {'total': round(refined, 2), 'read_wait': round(timers['read'], 2),
                    'sam': round(timers['sam'], 2), 'post': round(timers['post'], 2)},
    }
    with open(os.path.join(folder, 'run.json'), 'w', encoding='utf-8') as file:
        json.dump(run, file, indent=2)

    print()
    for action, count in sorted(totals.items()):
        print(f"  {action:<32}{count}")
    print(f"Time {refined:.1f}s (read wait {timers['read']:.1f}s, SAM {timers['sam']:.1f}s, "
          f"post {timers['post']:.1f}s)")

    if not args.no_figures and rows:
        tick = time.perf_counter()
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data.experimental.SAM_Refine_Figures import make_figures

        paths = dict(zip((path for path, _ in jobs), resolved))
        written = make_figures(rows, os.path.join(folder, 'figures'), settings, paths.get,
                               examples=args.examples, overlays=args.overlays, seed=args.seed,
                               max_full_pixels=args.max_full_pixels)
        print(f"Figures: {len(written)} in {time.perf_counter() - tick:.1f}s")

    print(f"Output: {folder}")
    print("Dry run: project not written" if args.dry_run else f"Project: {output}")


if __name__ == '__main__':
    main()
