"""Choose a Generator prompt's examples and confidence threshold on held out annotations.

The Generator stacks one class per reference image, so the confidence a prompt
needs rises with the number of references, and which images are the references
matters as much as how many. Measured on THESIA with YOLOE 11s at 1024
(scripts/SeeAnything_Testbed.py): the best threshold went from about 0.16 for 10
images to 0.53 for all 234, against a fixed default of 0.20, and prompts of 10
random images ranged 0.06 to 0.21 mAP50-95.

Calibrating scores several candidate prompts on images none of them came from,
and keeps the best one with the threshold that suited it. This module holds the
parts that need no model and no Qt: splitting images into candidates and
validation, drawing the tries, matching detections to annotations, and reading
the threshold off the result. The dialog does the embedding and predicting.
"""

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np


# Each goal is the F-beta it maximizes. Beta above 1 weighs recall more.
GOALS = {
    "Balanced": 1.0,
    "Find more": 2.0,
    "Fewer false positives": 0.5,
}
DEFAULT_GOAL = "Balanced"

# Thresholds tried: what the threshold slider can show, away from the extremes.
# Predictions are made at the lowest one, so every higher one is a filter on them.
THRESHOLDS = np.arange(5, 96) / 100.0

# A detection finds an annotation at this IoU or more, highest confidence first.
MATCH_IOU = 0.5

# How many images to test on: this share of all the label's images, within bounds.
VALIDATION_SHARE = 0.3
MIN_VALIDATION = 5
MAX_VALIDATION = 40
# Images with verified annotations a calibration needs: enough to test on, plus one to prompt with.
MIN_IMAGES = MIN_VALIDATION + 1

DEFAULT_PER_TRY = 10
DEFAULT_TRIES = 20


def seed_for(keys):
    """A seed fixed by the images, so the same prompt calibrates the same way twice."""
    digest = hashlib.sha1("\n".join(sorted(str(k) for k in keys)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def split_images(pool, outside, rng):
    """Choose the images to test on, preferring ones whose examples are not in the prompt.

    Images outside the prompt cost nothing to test on and take nothing from the
    candidates. Only when there are too few of them are prompt images held out,
    and a held out image is never part of any try.

    Args:
        pool (list): Images whose examples are in the prompt and can be chosen.
        outside (list): Other images with verified annotations of the label.
        rng (random.Random): Draws the split.

    Returns:
        tuple[list, list]: The candidates and the validation images.

    Raises:
        ValueError: If there are too few images to hold any out.
    """
    pool, outside = sorted(pool), sorted(outside)
    total = len(pool) + len(outside)
    if not pool or total < MIN_IMAGES:
        raise ValueError(f"Calibrating needs at least {MIN_IMAGES} images with verified "
                         f"annotations of the label. There are {total}.")

    target = min(MAX_VALIDATION, max(MIN_VALIDATION, round(VALIDATION_SHARE * total)))
    target = min(target, total - 1)
    from_outside = rng.sample(outside, min(target, len(outside)))
    shortfall = target - len(from_outside)
    held = set(rng.sample(pool, shortfall)) if shortfall > 0 else set()
    candidates = [key for key in pool if key not in held]
    return candidates, from_outside + sorted(held)


def draw_tries(candidates, per_try, tries, rng):
    """The example sets to score: every candidate together first, then random sets.

    Scoring every candidate too shows whether choosing a few beats using them
    all; on a tie the earlier try wins, so a random set has to do better to be kept.

    Args:
        candidates (list): Images that can be in a try.
        per_try (int): Images per random set.
        tries (int): How many distinct random sets to draw.
        rng (random.Random): Draws the sets.

    Returns:
        list[tuple]: The tries. Just one, every candidate, when there are no more
        candidates than `per_try`.
    """
    everything = tuple(candidates)
    per_try = max(1, int(per_try))
    if len(candidates) <= per_try:
        return [everything]

    out, seen = [everything], {frozenset(everything)}
    attempts = 0
    while len(out) < tries + 1 and attempts < 50 * max(1, tries):
        attempts += 1
        pick = tuple(rng.sample(candidates, per_try))
        key = frozenset(pick)
        if key not in seen:
            seen.add(key)
            out.append(pick)
    return out


def box_iou(a, b):
    """IoU between (M, 4) and (N, 4) xyxy arrays, as (M, N)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class ImageMatch:
    """One validation image's detections, each marked as finding an annotation or not."""

    confidence: np.ndarray
    found: np.ndarray
    known: int


def match_image(boxes, confidence, annotations, ignore=None, iou=MATCH_IOU):
    """Match detections to annotations one to one, highest confidence first.

    Matching greedily in confidence order means the matches above any threshold
    are the matches that threshold would make on its own, so one matching serves
    every threshold.

    Detections left unmatched that land on an `ignore` box are dropped, neither
    found nor false: those are unverified annotations, often the very objects a
    model already proposed, so counting a detection there as a false alarm would
    punish finding them.

    Args:
        boxes: (N, 4) xyxy detections, in image pixels.
        confidence: (N,) their confidences.
        annotations: (G, 4) xyxy verified annotation boxes, in image pixels.
        ignore: (U, 4) xyxy unverified annotation boxes, or None.
        iou (float): Least IoU for a match.

    Returns:
        ImageMatch
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    annotations = np.asarray(annotations, dtype=np.float64).reshape(-1, 4)
    found = np.zeros(len(boxes), dtype=bool)
    if len(boxes) and len(annotations):
        overlap = box_iou(boxes, annotations)
        taken = np.zeros(len(annotations), dtype=bool)
        for i in np.argsort(-confidence, kind="stable"):
            row = np.where(taken, -1.0, overlap[i])
            j = int(np.argmax(row))
            if row[j] >= iou:
                taken[j] = True
                found[i] = True

    ignore = np.zeros((0, 4)) if ignore is None else np.asarray(ignore, dtype=np.float64).reshape(-1, 4)
    if len(boxes) and len(ignore):
        keep = found | (box_iou(boxes, ignore).max(axis=1) < iou)
        confidence, found = confidence[keep], found[keep]
    return ImageMatch(confidence, found, len(annotations))


@dataclass
class TryScore:
    """One try read at one threshold, for one goal."""

    keys: tuple
    score: float
    threshold: float
    precision: float
    recall: float
    false_per_image: float
    true_positives: int         # Verified boxes found
    false_positives: int        # Detections matching no verified box, ignored ones aside
    known: int                  # Verified boxes on the validation images


@dataclass
class TryCurve:
    """How one try did on the validation images, as counts at every threshold in `THRESHOLDS`.

    Counts rather than one score, so the goal can be chosen after the run: the
    detections do not depend on it, only which threshold and which try suit it.
    """

    keys: tuple
    true_positives: np.ndarray  # Verified boxes found above each threshold
    shown: np.ndarray           # Detections above each threshold, ignored ones aside
    known: int
    images: int

    def scores(self, beta):
        """F-beta at every threshold."""
        hits, shown = self.true_positives, self.shown
        precision = np.divide(hits, shown, out=np.zeros_like(hits), where=shown > 0)
        recall = hits / self.known if self.known else np.zeros_like(hits)
        b2 = beta * beta
        denominator = b2 * precision + recall
        return np.divide((1 + b2) * precision * recall, denominator,
                         out=np.zeros_like(hits), where=denominator > 0)

    def read(self, i, beta):
        """The try at threshold index `i`, scored for `beta`."""
        hits, shown = float(self.true_positives[i]), float(self.shown[i])
        return TryScore(keys=self.keys,
                        score=float(self.scores(beta)[i]),
                        threshold=float(THRESHOLDS[i]),
                        precision=hits / shown if shown else 0.0,
                        recall=hits / self.known if self.known else 0.0,
                        false_per_image=(shown - hits) / max(1, self.images),
                        true_positives=int(hits),
                        false_positives=int(shown - hits),
                        known=int(self.known))

    def best(self, beta):
        """At the highest threshold reaching the best score, so ties go to fewer detections."""
        scores = self.scores(beta)
        return self.read(int(np.flatnonzero(scores == scores.max())[-1]), beta)


def curve_for(keys, matches):
    """Count one try's detections at every threshold.

    Args:
        keys (tuple): The images whose examples were in the prompt.
        matches (list[ImageMatch]): One per validation image.

    Returns:
        TryCurve
    """
    confidence = np.concatenate([m.confidence for m in matches]) if matches else np.zeros(0)
    found = np.concatenate([m.found for m in matches]) if matches else np.zeros(0, dtype=bool)
    above = confidence[None, :] > THRESHOLDS[:, None]
    return TryCurve(keys=tuple(keys),
                    true_positives=(above & found[None, :]).sum(axis=1).astype(np.float64),
                    shown=above.sum(axis=1).astype(np.float64),
                    known=int(sum(m.known for m in matches)),
                    images=len(matches))


def score_try(keys, matches, beta):
    """The threshold where one try best meets a goal."""
    return curve_for(keys, matches).best(beta)


@dataclass
class Calibration:
    """Every try's counts; which one to keep depends on the goal, chosen afterwards."""

    candidates: list
    validation: list
    tries: list = field(default_factory=list)

    @property
    def known(self):
        return self.tries[0].known if self.tries else 0

    @property
    def everything(self):
        """The try with every candidate."""
        return self.tries[0] if self.tries else None

    def found_anything(self):
        """Whether any try found any verified box at any threshold."""
        return any(t.true_positives.max() > 0 for t in self.tries)

    def options(self, goal):
        """What the results offer for this goal, as (name, TryScore), each at its own threshold.

        Every candidate together, then the random set that best meets the goal,
        when any were drawn.
        """
        beta = GOALS[goal]
        rows = [("All images", self.everything.best(beta))]
        if len(self.tries) > 1:
            best = max(range(1, len(self.tries)), key=lambda i: (self.tries[i].best(beta).score, -i))
            rows.append((f"Best set of {len(self.tries[best].keys)}", self.tries[best].best(beta)))
        return rows

    def recommended(self, goal):
        """Index into `options(goal)` of the higher score; All images wins a tie."""
        rows = self.options(goal)
        return max(range(len(rows)), key=lambda i: (rows[i][1].score, -i))

    def best(self, goal):
        """The recommended option for this goal."""
        return self.options(goal)[self.recommended(goal)][1]


def new_rng(keys):
    """The random generator for a calibration over these images."""
    return random.Random(seed_for(keys))
