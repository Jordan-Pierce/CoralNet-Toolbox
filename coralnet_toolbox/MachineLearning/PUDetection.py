"""Positive-unlabeled (PU) detection training: the ignore-band design.

In a positive-unlabeled dataset some objects are boxed and many real ones are
not -- the normal state of an annotation project, where a reviewer marks what
they came for and leaves the rest of the frame alone. A stock detector treats
every unboxed region as background, so it is punished for correctly firing on
the objects nobody got to, and it learns to stop finding them.

This trains an EMA *teacher* alongside the student and uses it for one thing:
regions the teacher is moderately confident about, that the human did not box,
are **excluded from the classification loss** rather than called background.
Nothing is added to the batch. The labels the user drew are the only positives
the student ever sees.

    GT boxes                              -> positives, untouched
    teacher score >= ignore_conf, no GT   -> excluded from the cls loss
    everything else                       -> background, as normal

The band is the whole feature, and there is no confidence knob to get wrong.

Scope: the anchor-based v8 detection loss, and the instance segmentation loss
built on top of it. ``v8SegmentationLoss`` subclasses ``v8DetectionLoss`` and
reaches its classification term through the same
``get_assigned_targets_and_loss``, so the band is the same one line on both
tasks -- :class:`PUDetectionLoss` and :class:`PUSegmentationLoss` differ only in
which loss :class:`_PUIgnoreBand` is mixed into. The mask term is left alone,
and not by choice: it is computed from ``fg_mask``, the anchors the assigner
matched to something the user actually drew, and the band never touches those.
So a PU segmentation run spends its mAP in the classification head exactly as a
detection run does, and its masks are trained on the user's polygons and nothing
else.

Not end-to-end models (yolov10, yolo26, and the ``Segment26`` head that
yolo26-seg carries -- they use E2ELoss, a different object), not semantic
segmentation (``SemanticSegment`` is a per-pixel classifier with no anchor grid
to mask; the Export Dataset dialog's "Treat as Ignore (Index 255)" option is the
blunt, static version of this feature for that task), and not RT-DETR (Hungarian
matching over queries). ``supports_pu`` is the gate, and it reads the built model
rather than guessing from the filename.

What this is measured against, and the numbers to repeat wherever it is offered
to a user. Four arms -- {complete labels, 20% of train boxes removed} x {stock
trainer, this one} -- on African Wildlife at yolo11n/1024, 60 epochs, three
seeds, the drop resampled per seed, every comparison paired within a seed, and
scored on a test split whose labels were never touched. Harness and tables in
data/PU_results:

    20% of train boxes removed  -> +0.0222 mAP50-95, +0.0272 recall,
                                   ahead in 3 of 3 seeds
    labels already complete     -> +0.0054 mAP50-95, sd 0.0097, and *behind*
                                   its own baseline in 1 of 3 seeds

What separates those two arms is how much was actually missing, not the model.
Given a complete dataset there is nothing for the band to recover and it buys
nothing reliable -- the spread across seeds there is wider than the mean. Offer
this for a dataset that really is positive-unlabeled, not as a general
improvement.

It does not fully undo the damage, either. On the thinned arm it stayed 0.0032
mAP50-95 behind the complete-label baseline, in every seed: the band recovers
most of what the missing labels cost, never all of it. And it is not free -- the
teacher costs a forward pass per batch, a frozen copy of the model in VRAM, and
about 1.4x wall-clock time.

Those numbers are one dataset, one model size and one task: african-wildlife is
small (1052 train images, 1.8 boxes each) and 4-class, so nothing above speaks
to dense scenes or many classes. **None of it was measured on segmentation.**
The mechanism carries over exactly -- the same loss method, the same single
masked term -- but a segmentation run is selected and reported on mask mAP while
the band spends its budget in the classification head, so the size of that trade
is an open question there rather than the one measured here. Rerun the harness
before extending the claim.

**The numbers above were scored on complete labels, and a real project's val and
test splits are PU too.** That was measured separately -- the same checkpoints
re-scored against a test split thinned the same 20%, in data/PU_results/PU_VAL_BIAS.md:

    mAP50-95   -0.125   |  precision  -0.184   <- the measurement, not the model
    mAP50      -0.145   |  recall     within 0.017, no consistent sign

The damage is near-uniform across all four arms, which is why **the ranking
survives**: the thinned-label arm still beat its baseline in 3 of 3 seeds under PU
scoring (+0.0172 mAP50-95, +0.0392 recall). So comparisons on one PU split are
valid and conservative -- the real gain is larger than the measured one -- while
absolute numbers are not reportable at all. Recall is the metric to trust and to
select on; a "false positive" and a recovered object are indistinguishable on a PU
split, so precision there is a lower bound rather than an estimate. A small
exhaustively labeled val set is still worth more than any of this, because it is
the only thing that restores honest absolutes, fitness and early stopping.

Early stopping is the part of that which this module has to handle rather than
document, because the user cannot be asked to notice it. Ultralytics stops on
``fitness``, and ``Metric.fitness()`` weights are ``[0, 0, 0, 1]`` over
``[P, R, mAP50, mAP50-95]`` -- mAP50-95 alone, the number PU spends. On a PU val
split a run that is still getting better at finding unlabeled objects can have
falling fitness and be stopped while improving. :class:`_PUEarlyStopping` is the
answer: patience over the *union* of fitness and recall, so a PU run ends only
once both have stalled. It can stop later than the stock rule, never earlier.

That is not hypothetical. Three further runs trained with a PU validation split
(A5/A6 in data/PU_results) show the divergence directly -- on one seed fitness
peaked at epoch 48 while recall went on improving to epoch 59, and the Active
Learning dialog's ``patience`` of 10 would have ended that run at epoch 58, one
epoch before its best recall. The union rule does not stop it at all. Training a
PU run against a PU val split also costs about 0.014 mAP50-95 purely through
checkpoint selection, since the training data is identical and only the labels
fitness is computed against differ.

With a PU val split the ignore band's recall gain survives intact (+0.0364, 3 of 3
seeds) while its mAP gain mostly does not (+0.0053, 2 of 3) -- more evidence that
recall is the signal to steer by here, and the reason this class exists.

Two things deliberately left out, neither measured against the above. *Injection*
-- adding the teacher's confident boxes to the batch as positives, rather than
only masking uncertain regions out of the loss -- would break the contract this
module is built on, that the student's positives are exactly the boxes the user
drew, so it is not a knob here. And RT-DETR, whose Hungarian matching over
queries gives the band no anchors to mask; that port is separate work. In
previous works, both of these failed to improve over the baseline in a consistent 
manner. Only Ignore-Band PU consistently worked.

References:
    - S2Teacher (2025): https://arxiv.org/abs/2504.11111
    - Co-mining (2021): https://arxiv.org/abs/2012.01950
    - Object Detection as a Positive-Unlabeled Problem (2020):
      https://arxiv.org/abs/2002.04672
"""

import os
import re
import math
import shutil
import textwrap
from copy import deepcopy

import yaml
import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import LOGGER, RANK


# The defaults are the validated configuration, and they are constants rather
# than dialog fields on purpose: every one of them was swept, and the feature is
# offered as a single switch. Anything exposed here is another way for a run to
# differ from the one the numbers came from.
IGNORE_CONF = 0.25      # teacher score at/above which a region stops being background
EMA_DECAY = 0.999       # slow teacher; the point is stability, not tracking
DEDUP_IOU = 0.30        # an ignore box this close to a GT box is redundant
MAX_TEACHER_DET = 300   # NMS cap per image, so the band is bounded in dense frames
WARMUP_FRACTION = 0.12  # GT-only epochs before the teacher is worth listening to

# The extra checkpoint a PU run leaves beside best.pt: the epoch that recalled
# best, rather than the epoch that scored best. Ultralytics selects best.pt on
# fitness, which for detection is mAP50-95 alone -- precision and recall are both
# weighted zero -- and mAP is exactly the number PU is expected to spend.
#
# What this is NOT: the checkpoint to deploy. That was measured directly -- three
# seeds trained with a PU *validation* split, which is the case it exists for, then
# both checkpoints scored on clean test:
#
#     recall    -0.0036 mean, better in 1 of 3 seeds
#     mAP50-95  +0.0129 mean, better in 2 of 3 (one of them an exact tie, the
#               same epoch chosen twice)
#
# It does not reliably deliver more recall even when val is PU, which is the one
# thing it is for. No consistent winner, so best.pt stays what every consumer
# loads -- that also keeps it consistent with how Active Learning ranks rounds and
# with read_metrics' invariant that the reported epoch is the deployed one. This is
# kept as the record of the best-recall epoch, and it is worth something: on the
# one seed where fitness selection went wrong it was +0.0366 mAP50-95 ahead.
RECALL_CHECKPOINT = 'recall_best.pt'

# The families the anchor-based v8 loss covers, as a user reads them. Every
# community detection or segmentation config ends in that same Detect or Segment
# head, so they qualify too.
PU_FAMILIES = "YOLOv3u, YOLOv5u, YOLOv8, YOLOv9, YOLO11 or YOLO12"

# What the dialogs append to the PU Dataset tooltip, enabled or not, so the
# models it works with are stated rather than left to be found by elimination.
PU_MODELS_NOTE = ("Works with: YOLOv3u, YOLOv5u, YOLOv8, YOLOv9, YOLO11 and YOLO12, for\n"
                  "detection and for instance segmentation (-seg), plus the community\n"
                  "models.\n"
                  "Not available for: RT-DETR, YOLOv10, YOLO26 (yolo26-seg included).")

# How far back pu_supported_model follows a checkpoint's run records. A warm
# started Active Learning round is a chain of best.pt files, one per round.
MAX_LINEAGE = 16


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def pu_warmup_epochs(epochs):
    """Ground-truth-only epochs before the ignore band switches on.

    An untrained teacher's confident regions are noise, and excluding noise from
    the loss is worse than leaving it as background, so PU waits.
    """
    return max(1, int(WARMUP_FRACTION * int(epochs)))


def pu_close_mosaic(epochs):
    """The ``close_mosaic`` that leaves every PU-active epoch mosaic-free.

    Mosaic hands the model four images stitched into one canvas. The student
    copes; the teacher does not, because a 4-up collage is far off the
    distribution it learned to score, and its regions are then wrong about
    exactly the thing the ignore band acts on. Semi-supervised methods normally
    fix this by running the teacher on a separate weak view of the batch -- the
    batch here holds only the strong view, so the practical fix is to not use
    mosaic while PU is on.

    Ultralytics counts ``close_mosaic`` back from the end, so this returns the
    number of epochs from the end of warmup onward.
    """
    epochs = int(epochs)
    return max(0, epochs - pu_warmup_epochs(epochs))


def supports_pu(model):
    """Whether PU can run on this built model, and why not when it cannot.

    Returns ``(True, "")`` or ``(False, reason)``.

    Reads the model's **head**, which is the only thing that actually decides.
    Neither the filename nor the model class will do it: ``YOLO()`` loads
    RT-DETR, YOLO26 and YOLO11 weights all into a plain ``DetectionModel``, and
    a checkpoint can be named anything. What separates them is the last module
    and whether it carries a one-to-one branch --

        Detect, no one2one_cv2   -> v8DetectionLoss      (PU works)
        Segment, no one2one_cv2  -> v8SegmentationLoss   (PU works)
        Detect / Segment26 /
          v10Detect, with
          one2one_cv2            -> E2ELoss           (yolov10, yolo26)
        SemanticSegment          -> SemanticSegmentationLoss
        RTDETRDecoder            -> RTDETRDetectionLoss

    -- so those are what is asked. The final check is the same one
    ``v8DetectionLoss.__init__`` makes: it reads ``stride``, ``nc`` and
    ``reg_max`` off the head, and a head missing any of them raises inside the
    constructor rather than returning a usable loss. That is also what refuses
    ``SemanticSegment``, which has ``stride`` and ``nc`` but no ``reg_max``
    because it predicts no boxes at all -- there is no name check for it, and it
    needs none.

    What this does *not* settle is which task the run is. A ``Segment`` head
    passes here, and training it through the detection trainer would build a
    ``DetectionModel`` and never train a mask, so the caller has to pair the
    answer with :func:`pu_trainer_for`.
    """
    inner = getattr(model, 'model', model)
    layers = getattr(inner, 'model', None)
    if layers is None or not len(layers) or not hasattr(inner, 'args'):
        return False, "This does not look like an Ultralytics detection model."

    head = layers[-1]
    head_name = type(head).__name__

    if 'RTDETR' in head_name.upper():
        return False, ("RT-DETR trains through DETR-style Hungarian matching over "
                       "queries, which has no anchors for the ignore band to mask. "
                       f"Use a {PU_FAMILIES} model instead.")
    # one2one_cv2 is the test DetectionModel.init_criterion itself uses to pick
    # E2ELoss (8.4.153). The end2end flag alone is not enough: a YOLO26 head
    # reports end2end False and still trains through E2ELoss.
    if (getattr(head, 'one2one_cv2', None) is not None
            or getattr(inner, 'end2end', False) or 'v10' in head_name.lower()):
        return False, ("End-to-end models (YOLOv10, YOLO26) train through E2ELoss, "
                       f"a pair of losses this does not wrap. Use a {PU_FAMILIES} "
                       "model instead.")
    if not all(hasattr(head, attr) for attr in ('stride', 'nc', 'reg_max')):
        return False, (f"This model's head ({head_name}) is not the anchor-based "
                       "detection head the ignore band masks.")
    return True, ""


# Families whose name is enough to rule them out, for greying the control out
# before anything is downloaded or built. Matched against the file's basename.
#   yolo10 / yolov10, yolo26  -> end-to-end, E2ELoss
#   rtdetr                    -> DETR matching
_UNSUPPORTED_NAMES = (
    (re.compile(r'rtdetr', re.I),
     "RT-DETR trains through query matching, which has no anchors for the "
     "ignore band to mask."),
    (re.compile(r'yolo_?v?(10|26)', re.I),
     "YOLOv10 and YOLO26 are end-to-end models and train through a different "
     "loss, which this does not wrap."),
)


def pu_supported_name(name):
    """Whether a model *name* is known to be unusable, before anything is built.

    Returns ``(True, "")`` when PU may be offered, ``(False, reason)`` when the
    name alone settles it. This exists only so the dialogs can grey the control
    out: naming a weights file is not proof of what is inside it, so this is
    deliberately one-sided. It refuses what it can positively identify -- the
    dropdown entries, which are the case that matters -- and allows everything
    else through to :func:`supports_pu`, which reads the built model and is the
    check that actually protects the run.

    The asymmetry is the point. Greying out a browsed ``best.pt`` because its
    name is unfamiliar would block the most ordinary use there is: continuing
    from weights an earlier round produced.
    """
    base = os.path.basename(str(name or "")).strip()
    if not base:
        return True, ""
    for pattern, reason in _UNSUPPORTED_NAMES:
        if pattern.search(base):
            return False, reason
    return True, ""


def _trained_from(path):
    """The model a checkpoint was trained from, per its run's args.yaml, or None.

    Ultralytics writes args.yaml into every run folder, beside the weights/
    folder that holds best.pt, and its ``model`` entry is what the run was
    started from. That is the record of what is inside a checkpoint, and it
    costs a small YAML read rather than unpickling the model.
    """
    path = str(path or "").strip()
    if not path or not os.path.isfile(path):
        return None

    folder = os.path.dirname(os.path.abspath(path))
    # weights/best.pt keeps its record one level up; a checkpoint copied out
    # with its results (Save Session) keeps it alongside
    if os.path.basename(folder).lower() == 'weights':
        folder = os.path.dirname(folder)

    args_path = os.path.join(folder, 'args.yaml')
    if not os.path.isfile(args_path):
        return None
    try:
        with open(args_path, 'r') as file:
            args = yaml.safe_load(file)
    except Exception:
        return None

    parent = args.get('model') if isinstance(args, dict) else None
    return str(parent) if parent else None


def pu_supported_model(model):
    """:func:`pu_supported_name`, extended back through a checkpoint's lineage.

    Returns ``(supported, reason, source)``, where ``source`` is the name that
    settled it: the model itself, or the model an earlier run trained it from.

    A browsed ``best.pt`` says nothing by its name, but its run folder records
    what it was trained from, and that model may itself be an earlier run's
    checkpoint. Following those records refuses a fine tuned YOLO26 the same as
    the dropdown entry it came from. Still one-sided, like the name check: a
    checkpoint with no record is allowed, and :func:`supports_pu` decides once
    the model is built.
    """
    current = str(model or "").strip()
    seen = set()
    for _ in range(MAX_LINEAGE):
        supported, reason = pu_supported_name(current)
        if not supported:
            if current != str(model or "").strip():
                reason = "It was trained from {}. {}".format(os.path.basename(current), reason)
            return False, reason, current

        parent = _trained_from(current)
        if not parent:
            break
        key = os.path.normcase(os.path.abspath(parent)) if os.path.isfile(parent) else parent
        if key in seen:
            break
        seen.add(key)
        current = parent

    return True, "", str(model or "").strip()


# The size letter of a refused family, so the suggestion keeps the model the
# same size. YOLOv10's b (balanced) sits between m and l; l is the nearer offer.
_REFUSED_SIZE = re.compile(r'yolo_?v?(?:10|26)([nsmblx])|rtdetr-([lx])', re.I)


def pu_alternative(name):
    """The YOLO11 model of the same size and task as a refused one, or None.

    The task suffix is carried across, because the offer has to be a model the
    user can actually train: the alternative to ``yolo26n-seg.pt`` is
    ``yolo11n-seg.pt``, and suggesting ``yolo11n.pt`` there would answer a
    segmentation question with a detection model.
    """
    base = os.path.basename(str(name or ""))
    match = _REFUSED_SIZE.search(base)
    if not match:
        return None
    size = (match.group(1) or match.group(2)).lower()
    suffix = "-seg" if "-seg" in base.lower() else ""
    return "yolo11{}{}.pt".format('l' if size == 'b' else size, suffix)


def pu_unavailable_tooltip(model, reason, source):
    """The PU Dataset tooltip for a model that cannot run it: why, and what can."""
    name = os.path.basename(str(model or "")) or "this model"
    lines = ["Not available for {}.".format(name),
             textwrap.fill(reason, width=78),
             "",
             PU_MODELS_NOTE]

    alternative = pu_alternative(source)
    if alternative:
        lines.append("Try {} instead: the same size, and it supports PU Dataset.".format(alternative))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------------------------------


class _PUIgnoreBand:
    """The ignore band itself, as a mixin over an anchor-based v8 loss.

    Mixed in front of ``v8DetectionLoss`` or ``v8SegmentationLoss``. Both reach
    their classification term through ``get_assigned_targets_and_loss``, which
    is the single method this overrides, so the feature is the same code on both
    tasks rather than two copies that have to be kept in step.

    The trainer owns ``teacher`` (an ``nn.Module``, eval mode, no grad) and the
    schedule flag ``pu_enabled``. With either unset this is the stock loss --
    the same code path, not an equivalent one -- which is what makes the feature
    safe to leave switched off.

    Note what this does *not* do: it never writes to ``batch``. The student's
    positives are exactly the boxes and polygons the user drew, before and
    after. On a segmentation model the mask term is untouched for the same
    reason: ``v8SegmentationLoss`` builds it from the ``fg_mask`` this method
    hands back, and an anchor the assigner matched to a human annotation is
    never in the band.
    """

    is_pu_criterion = True  # read by the trainer's callbacks

    def __init__(self, model, *args, ignore_conf=IGNORE_CONF, dedup_iou=DEDUP_IOU,
                 max_teacher_det=MAX_TEACHER_DET, **kwargs):
        # *args/**kwargs pass tal_topk and tal_topk2 through to whichever loss
        # this is mixed into, so neither subclass has to restate that signature.
        super().__init__(model, *args, **kwargs)
        self.teacher = None
        self.pu_enabled = False
        self.ignore_conf = ignore_conf
        self.dedup_iou = dedup_iou
        self.max_teacher_det = max_teacher_det
        # Totals for the epoch in progress. The trainer copies them into
        # last_ignore_count at the epoch boundary and zeroes them, so the
        # reported figure always describes one whole epoch.
        self.epoch_ignore_count = 0
        self.last_ignore_count = 0
        # The names belonging to *this* method's three-element loss tensor,
        # which is not always self.loss_names: v8SegmentationLoss rewrites that
        # to five entries (box, seg, cls, dfl, sem), and zipping five names
        # against three values would label the cls term "seg_loss". The trainer
        # takes its progress-bar headings from the criterion's dict on the first
        # batch, so a wrong key here is a wrong column for the whole run.
        self._pu_loss_names = ("box_loss", "cls_loss",
                               "dfl_loss" if self.use_dfl else "l1_loss")

    @torch.no_grad()
    def _teacher_boxes(self, imgs):
        """Teacher detections per image as xyxy in the pixel space of ``imgs``."""
        was_training = self.teacher.training
        self.teacher.eval()
        preds = self.teacher(imgs)
        # Detect's eval forward returns (inference, raw); Segment's returns
        # ((inference, proto), raw), so one unwrap is not enough. NMS takes [0]
        # of a sequence itself, which is the only reason a single unwrap ever
        # worked -- go down to the tensor here rather than relying on that.
        while isinstance(preds, (list, tuple)):
            preds = preds[0]
        dets = non_max_suppression(
            preds,
            conf_thres=max(self.ignore_conf, 1e-3),
            iou_thres=0.5,
            max_det=self.max_teacher_det,
            # Load-bearing on a segmentation model. Left at its default, NMS
            # infers nc = channels - 4, and a Segment head's channels are
            # 4 + nc + 32 -- so the 32 mask coefficients get read as class
            # logits. They are not scores and are not bounded, so a box would
            # take its confidence from the largest coefficient and the band
            # would land wherever that fell. Detection is exactly 4 + nc, which
            # is why the default was fine there.
            nc=self.nc,
        )
        if was_training:
            self.teacher.train()
        return [d[:, :4] for d in dets]

    @staticmethod
    def _box_iou_matrix(a, b):
        """IoU between two sets of xyxy boxes -> [len(a), len(b)]."""
        if a.numel() == 0 or b.numel() == 0:
            return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
        area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
        area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(0)
        inter = wh[..., 0] * wh[..., 1]
        union = area_a[:, None] + area_b[None, :] - inter + 1e-9
        return inter / union

    def _ignore_boxes(self, batch):
        """Per-image uncertain-teacher boxes, xyxy pixels, GT overlaps removed.

        A teacher box sitting on a box the human already drew tells us nothing
        we do not know, and excluding its surroundings would only weaken a
        region that is correctly labelled.
        """
        imgs = batch["img"]
        b, _, ih, iw = imgs.shape
        device = imgs.device

        per_img = self._teacher_boxes(imgs)
        gt_idx = batch["batch_idx"].to(device).view(-1)
        gt_boxes_n = batch["bboxes"].to(device)  # normalized xywh

        out = [None] * b
        for i in range(b):
            boxes = per_img[i]
            if boxes.numel() == 0:
                continue

            gt_i_n = gt_boxes_n[gt_idx == i]
            if gt_i_n.numel():
                gt_xyxy = torch.empty_like(gt_i_n)
                gt_xyxy[:, 0] = (gt_i_n[:, 0] - gt_i_n[:, 2] / 2) * iw
                gt_xyxy[:, 1] = (gt_i_n[:, 1] - gt_i_n[:, 3] / 2) * ih
                gt_xyxy[:, 2] = (gt_i_n[:, 0] + gt_i_n[:, 2] / 2) * iw
                gt_xyxy[:, 3] = (gt_i_n[:, 1] + gt_i_n[:, 3] / 2) * ih
                boxes = boxes[self._box_iou_matrix(boxes, gt_xyxy).max(dim=1).values
                              < self.dedup_iou]

            if boxes.numel():
                out[i] = boxes
                self.epoch_ignore_count += boxes.shape[0]
        return out

    @staticmethod
    def _ignore_band_mask(ignore_boxes, anchor_points, stride_tensor, fg_mask, scores_shape):
        """(bs, num_anchors) float mask: 0 for an anchor to exclude, 1 to keep.

        Returns None when there is nothing to exclude, so the caller can take
        the stock path rather than multiplying by a mask of ones.
        """
        if not any(b is not None and b.numel() for b in ignore_boxes):
            return None

        bs, na, _ = scores_shape
        centers = anchor_points * stride_tensor  # (na, 2), pixels
        cx, cy = centers[:, 0], centers[:, 1]
        keep = torch.ones((bs, na), device=anchor_points.device)
        for i, boxes in enumerate(ignore_boxes):
            if boxes is None or boxes.numel() == 0:
                continue
            inside = (
                (cx[None, :] >= boxes[:, 0:1]) & (cx[None, :] <= boxes[:, 2:3]) &
                (cy[None, :] >= boxes[:, 1:2]) & (cy[None, :] <= boxes[:, 3:4])
            ).any(dim=0)  # (na,)
            keep[i][inside] = 0.0
        # An anchor the assigner matched to a human box is never dropped: the
        # ignore band is about unlabeled regions, and a matched anchor is not one.
        return torch.where(fg_mask.bool(), torch.ones_like(keep), keep)

    def get_assigned_targets_and_loss(self, preds, batch):
        """The stock body with one masked term.

        Mirrors ``v8DetectionLoss.get_assigned_targets_and_loss`` (checked
        against ultralytics 8.4.153 and 8.4.171, the ends of the supported
        range) because the ignore band has to be applied to ``bce_loss``
        *before* it is reduced, and the stock method reduces on the way out.
        When PU is off this delegates instead, so nothing here can affect a
        non-PU run.

        Two deliberate divergences from the newest stock body, both kept for the
        8.4.153 floor and neither of them changing the result: ``max(..., 1)``
        rather than ``.clamp_(min=1)``, and the ``if fg_mask.sum()`` guard
        around ``bbox_loss`` that stock dropped once ``BboxLoss`` learned to
        return zero on an empty foreground by itself.

        ``v8SegmentationLoss`` calls this too and takes its assigned targets
        from the first return value, so the mask term it then computes is built
        on the same ``fg_mask`` -- the user's own annotations -- and never sees
        the band.
        """
        if not (self.pu_enabled and self.teacher is not None):
            return super().get_assigned_targets_and_loss(preds, batch)

        ignore_boxes = self._ignore_boxes(batch)

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        # The one PU line. Note the divisor is left alone: dividing by the
        # surviving anchors instead would re-inflate the loss to its old scale
        # and undo the exclusion.
        cls_keep = self._ignore_band_mask(ignore_boxes, anchor_points, stride_tensor,
                                          fg_mask, pred_scores.shape)
        if cls_keep is not None:
            bce_loss = bce_loss * cls_keep.unsqueeze(-1)
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        # WARNING: line below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[0] += pred_distri[..., :0].sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # A dict, not the bare tensor: the trainer averages the epoch's losses
        # with loss_items.items(), so a tensor here ran warmup fine and then
        # crashed on the first batch the ignore band was switched on for.
        # _pu_loss_names rather than self.loss_names -- see __init__.
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self._pu_loss_names, loss.detach())),
        )  # loss(box, cls, dfl)


class PUDetectionLoss(_PUIgnoreBand, v8DetectionLoss):
    """The ignore band over the stock detection loss."""


class PUSegmentationLoss(_PUIgnoreBand, v8SegmentationLoss):
    """The ignore band over the stock instance segmentation loss.

    Nothing to add. ``v8SegmentationLoss.loss`` gets its box, cls and dfl terms
    by calling ``get_assigned_targets_and_loss``, which the mixin has already
    replaced, and then adds its mask term from the assigned targets it gets
    back. The band reaches the classification loss and stops there.
    """


# ----------------------------------------------------------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------------------------------------------------------


class _PUEarlyStopping:
    """``EarlyStopping`` over the union of fitness and validation recall.

    Ultralytics stops on ``fitness``, which for detection is mAP50-95 alone, and
    that is the number PU spends: on a positive-unlabeled val split every object
    the model newly finds counts against it, so a run can be improving at exactly
    the job it was given while its fitness falls. Stock early stopping then ends
    it mid-improvement, and the Active Learning dialog's ``patience`` of 10 makes
    that likely rather than theoretical.

    So this waits for **both** signals to stall: ``patience`` is measured from the
    more recent of the last fitness improvement and the last recall improvement.
    It can stop later than the stock rule and never earlier, which is why it is
    safe to install unconditionally on a PU run. The user's ``patience`` value is
    still the user's -- this changes what counts as "no improvement", not how long
    to wait for it.

    Deliberately not a subclass: ``EarlyStopping.__call__`` logs its own stop
    message, and delegating would print "training stopped early" on epochs this
    rule goes on to veto.

    ``possible_stop`` is as load-bearing as the return value. The trainer reads it
    to decide whether to validate at all (``if self.args.val or final_epoch or
    self.stopper.possible_stop or self.stop``), and recall only exists if
    validation ran, so it is reported from the union too.
    """

    def __init__(self, patience, recall_of):
        self.patience = patience or float('inf')
        # Called with no arguments; returns this epoch's validation recall or None.
        self._recall_of = recall_of
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.best_recall = None
        self.best_recall_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness is None:  # val=False; matches EarlyStopping's own guard
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        fitness_delta = epoch - self.best_epoch

        recall = None
        try:
            recall = self._recall_of()
        except Exception:  # a metrics dict this cannot read must not end the run
            recall = None
        if recall is not None and (self.best_recall is None or recall > self.best_recall):
            self.best_recall = recall
            self.best_recall_epoch = epoch
        # With no recall to read, fall back to fitness alone rather than to a
        # delta of zero, which would be a run that can never stop.
        recall_delta = (epoch - self.best_recall_epoch
                        if self.best_recall is not None else fitness_delta)

        # The union: stalled only where neither has improved recently.
        delta = min(fitness_delta, recall_delta)
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop and RANK in (-1, 0):
            LOGGER.info(
                f"PU early stopping: no improvement in fitness (best epoch "
                f"{self.best_epoch}, {self.best_fitness:.4f}) or recall (best epoch "
                f"{self.best_recall_epoch}"
                + (f", {self.best_recall:.4f}" if self.best_recall is not None else "")
                + f") for {self.patience} epochs. Both are required to stall, because "
                f"fitness alone is the metric PU trades away.")
        return stop


# ----------------------------------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------------------------------


class _PUTrainer:
    """The EMA teacher and the PU schedule, as a mixin over a stock trainer.

    Mixed in front of ``DetectionTrainer`` or ``SegmentationTrainer``. Which of
    those it is decides the model class, the validator and the dataset, and none
    of that is PU's business -- so this subclasses neither, and names only its
    loss, through ``pu_loss_cls``. Pairing those wrong is silent rather than
    loud: a segmentation run put through ``DetectionTrainer`` builds a
    ``DetectionModel``, trains happily, and never trains a mask. Call
    :func:`pu_trainer_for` rather than picking a class by hand.

    Passed to ``model.train(trainer=...)``, which is Ultralytics' own hook for
    this, so everything else about the run -- dataset class, callbacks, saving,
    validation, early stopping -- is untouched. It takes no configuration: the
    feature is one switch, and the constants at the top of this module are the
    validated values.

    The teacher is a frozen copy of the student updated after every optimizer
    step as ``theta_t <- d * theta_t + (1 - d) * theta_s``. It is discarded when
    training ends; ``best.pt`` is the student, which is what the rest of the
    toolbox reads.
    """

    pu_loss_cls = None  # set by the concrete trainers below

    def _setup_train(self):
        super()._setup_train()

        self.pu_warmup = pu_warmup_epochs(self.epochs)
        student = unwrap_model(self.model)

        self.teacher = deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher = self.teacher.to(self.device)
        self._teacher_updates = 0

        criterion = self.pu_loss_cls(student)
        criterion.teacher = self.teacher
        # Ultralytics builds the criterion lazily on the first loss call and
        # caches it here, so assigning it now is what swaps the loss.
        self.model.criterion = criterion

        # One row per completed epoch. Cheap, and it is the only record of what
        # the teacher was actually doing -- a band that keeps growing late in a
        # run is the teacher drifting onto the student's own predictions.
        self.pu_diag_csv = os.path.join(str(self.save_dir), "pu_diagnostics.csv")
        os.makedirs(str(self.save_dir), exist_ok=True)
        with open(self.pu_diag_csv, "w") as f:
            f.write("epoch,pu_enabled,ignore_conf,ignore_count\n")

        # Stop on the union of fitness and recall, not on fitness alone. Installed
        # after super()._setup_train(), which is what creates the stock stopper.
        self.stopper = _PUEarlyStopping(
            getattr(self.args, 'patience', 0),
            lambda: self.epoch_recall(getattr(self, 'metrics', None) or {}),
        )

        self._pu_best_recall = None
        self.add_callback("on_train_epoch_start", self._pu_epoch_start)
        self.add_callback("on_train_batch_end", self._pu_update_teacher)
        self.add_callback("on_fit_epoch_end", self._pu_track_recall)
        self.add_callback("on_train_end", self._pu_finalize)

        if RANK in (-1, 0):
            LOGGER.info(f"PU: ignore-band training. warmup={self.pu_warmup} epochs "
                        f"(GT only), ignore_conf={IGNORE_CONF}, ema_decay={EMA_DECAY}.")
            LOGGER.info(f"PU: early stopping waits for fitness AND recall to stall "
                        f"(patience={self.stopper.patience}), because mAP50-95 alone "
                        f"is the metric PU trades away.")

    def pu_epoch_note(self):
        """One short phrase describing what PU did this epoch, for the GUI log.

        Read by the training callbacks, which have the trainer but not the loss.
        Uses ``epoch_ignore_count`` rather than ``last_ignore_count``: this is
        called at ``on_fit_epoch_end``, after every training batch of the epoch
        has run, so the running total is complete and the rolled-over figure is
        still the previous epoch's.
        """
        crit = getattr(self.model, "criterion", None)
        if not getattr(crit, "is_pu_criterion", False):
            return ""
        if not getattr(crit, "pu_enabled", False):
            return f"PU warmup {self.epoch + 1}/{self.pu_warmup}"
        return f"PU band: {crit.epoch_ignore_count} regions excluded"

    def _pu_decay(self):
        """Decay with a short ramp, so an EMA that starts at the student's own
        weights is not held there by a 0.999 average over its first few steps.
        Mirrors Ultralytics' ModelEMA."""
        return EMA_DECAY * (1 - math.exp(-self._teacher_updates / 2000.0))

    @torch.no_grad()
    def _pu_update_teacher(self, *args, **kwargs):
        self._teacher_updates += 1
        d = self._pu_decay()
        msd = unwrap_model(self.model).state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach().to(v.dtype)

    @staticmethod
    def epoch_recall(metrics):
        """Validation recall out of an Ultralytics metrics dict, or None.

        Prefers the mask figure when there is one, matching how the rest of the
        toolbox reports a segmentation run, and falls back to any key that names
        recall so a future column rename degrades to "no recall" rather than to
        a wrong number.
        """
        for key in ('metrics/recall(M)', 'metrics/recall(B)'):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        for key, value in metrics.items():
            if 'recall' in key.lower():
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _pu_track_recall(self, *args, **kwargs):
        """Keep a copy of the epoch that recalled best, as recall_best.pt.

        ``best.pt`` is selected by Ultralytics on fitness, which for detection is
        mAP50-95 alone. Under PU that is the number being deliberately spent: a
        model that starts finding objects nobody labelled scores those finds as
        false positives, so it can go on improving at the job it was given while
        its fitness falls, and the epoch worth keeping is never written.

        This runs as ``on_fit_epoch_end``, which fires after Ultralytics has both
        appended the epoch's row to results.csv and written ``last.pt`` -- so
        ``last.pt`` is exactly this epoch's model, and copying it is cheaper and
        far safer than re-serialising a checkpoint by hand. Whoever reads the run
        afterwards can find the same epoch by taking the best recall in
        results.csv; both use a strict improvement, so both settle a tie on the
        earlier epoch.

        Best-effort throughout: a failure here must not end a run that is
        otherwise fine, because ``best.pt`` is still there to fall back on.
        """
        if RANK not in (-1, 0):
            return
        try:
            recall = self.epoch_recall(getattr(self, 'metrics', None) or {})
            if recall is None or (self._pu_best_recall is not None
                                  and recall <= self._pu_best_recall):
                return

            wdir = str(getattr(self, 'wdir', '') or
                       os.path.join(str(self.save_dir), 'weights'))
            last = os.path.join(wdir, 'last.pt')
            if not os.path.isfile(last):
                # save=False, or an epoch Ultralytics chose not to write.
                return

            shutil.copyfile(last, os.path.join(wdir, RECALL_CHECKPOINT))
            self._pu_best_recall = recall
            LOGGER.info(f"PU: epoch {self.epoch + 1} is the best recall so far "
                        f"({recall:.4f}); saved {RECALL_CHECKPOINT}.")
        except Exception as e:
            LOGGER.warning(f"PU: could not update {RECALL_CHECKPOINT} (non-fatal): {e}")

    def _pu_log_diagnostics(self, epoch, crit):
        if RANK not in (-1, 0):
            return
        with open(self.pu_diag_csv, "a") as f:
            f.write(f"{epoch + 1},{crit.pu_enabled},{crit.ignore_conf:.4f},"
                    f"{crit.last_ignore_count}\n")

    def _pu_epoch_start(self, *args, **kwargs):
        """Close out the epoch that just ended, then switch PU on once warm."""
        crit = getattr(self.model, "criterion", None)
        if not getattr(crit, "is_pu_criterion", False):
            return
        epoch = self.epoch  # 0-indexed

        # crit.pu_enabled still describes the epoch that just finished, so the
        # row is written before the flag is updated for the one starting.
        if epoch > 0:
            crit.last_ignore_count = crit.epoch_ignore_count
            self._pu_log_diagnostics(epoch - 1, crit)
            crit.epoch_ignore_count = 0

        if epoch < self.pu_warmup:
            crit.pu_enabled = False
            if RANK in (-1, 0):
                LOGGER.info(f"Epoch {epoch + 1}: PU warmup (ground truth only).")
            return

        crit.pu_enabled = True
        if RANK in (-1, 0):
            LOGGER.info(f"Epoch {epoch + 1}: PU ignore band active "
                        f"(last epoch excluded {crit.last_ignore_count} regions).")

    def _pu_finalize(self, *args, **kwargs):
        """Write the last epoch's row; no epoch-start call is coming to do it."""
        crit = getattr(self.model, "criterion", None)
        if not getattr(crit, "is_pu_criterion", False):
            return
        crit.last_ignore_count = crit.epoch_ignore_count
        self._pu_log_diagnostics(self.epoch, crit)


class PUDetectionTrainer(_PUTrainer, DetectionTrainer):
    """DetectionTrainer that keeps an EMA teacher and runs the PU schedule."""

    pu_loss_cls = PUDetectionLoss


class PUSegmentationTrainer(_PUTrainer, SegmentationTrainer):
    """SegmentationTrainer that keeps an EMA teacher and runs the PU schedule.

    ``SegmentationTrainer`` rather than ``DetectionTrainer`` is the whole
    difference, and it is the part that matters: it is what builds a
    ``SegmentationModel``, validates with mask metrics, and makes
    ``metrics/recall(M)`` -- the figure :meth:`_PUTrainer.epoch_recall` already
    prefers -- exist at all.
    """

    pu_loss_cls = PUSegmentationLoss


# The tasks PU can train, and the trainer each one needs. The dialogs read this
# rather than naming a class, so adding a task is one entry here instead of a
# condition in every caller -- and so no caller can pair a task with the wrong
# trainer, which trains without error and without masks.
PU_TRAINERS = {
    'detect': PUDetectionTrainer,
    'segment': PUSegmentationTrainer,
}


def pu_trainer_for(task):
    """The PU trainer for a task, or None when that task has no PU path.

    None is the answer for classify and semantic, and it is the same answer the
    dialogs use to decide whether to offer the control at all.
    """
    return PU_TRAINERS.get(str(task or "").strip().lower())
