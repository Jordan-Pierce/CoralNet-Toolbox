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

Why only the ignore band, when the teacher could also *inject* its confident
boxes as positives: injection was measured on FathomNet-2026, the African
Wildlife label-drop benchmark, and real MDBC coral data, and it is the one part
of this method that can fail badly. It teaches the student its own false
positives -- on multiclass data the teacher staples the wrong species onto a
recovered box, and on DETR-family models it collapses precision outright
(recall 0.61 at precision 0.03: over-prediction, not detection). The ignore
band never had that failure mode in any arm, so it is the whole feature here
and there is no confidence knob to get wrong.

Scope: the anchor-based v8 detection loss (yolov3u through yolo12x, and the
segmentation heads built on it). Not end-to-end models (yolov10, yolo26 -- they
use E2ELoss, a different object), and not RT-DETR (Hungarian matching over
queries; the port exists but is a separate piece of work). ``supports_pu``
is the gate, and it reads the built model rather than guessing from the
filename.

Measured caveat worth repeating wherever this is offered to a user: the edge is
capacity-gated. It showed up on medium backbones at large imgsz and did *not*
show up at nano -- an isolation run at yolo11n/1408 came out slightly behind
its own baseline. It is not free, either: the teacher costs a forward pass per
batch and a frozen copy of the model in VRAM.

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
from copy import deepcopy

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
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
# fitness, which for detection is mAP50-95 alone -- precision and recall are
# both weighted zero -- and mAP is exactly the number PU is expected to spend.
# Without this the run can train past its own best model and never save it.
RECALL_CHECKPOINT = 'recall_best.pt'


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
    and one flag --

        Detect,        end2end False -> v8DetectionLoss   (PU works)
        Detect,        end2end True  -> E2ELoss           (yolov10, yolo26)
        RTDETRDecoder                -> RTDETRDetectionLoss

    -- so those are what is asked. The final check is the same one
    ``v8DetectionLoss.__init__`` makes: it reads ``stride``, ``nc`` and
    ``reg_max`` off the head, and a head missing any of them raises inside the
    constructor rather than returning a usable loss.
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
                       "Use a YOLO detection model.")
    if getattr(inner, 'end2end', False) or 'v10' in head_name.lower():
        return False, ("End-to-end models (YOLOv10, YOLO26) train through E2ELoss, "
                       "a pair of losses this does not wrap. Use a YOLOv8, YOLOv9, "
                       "YOLO11 or YOLO12 model.")
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


# ----------------------------------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------------------------------


class PUDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss that drops uncertain teacher regions from the BCE term.

    The trainer owns ``teacher`` (an ``nn.Module``, eval mode, no grad) and the
    schedule flag ``pu_enabled``. With either unset this is the stock loss --
    the same code path, not an equivalent one -- which is what makes the feature
    safe to leave switched off.

    Note what this does *not* do: it never writes to ``batch``. The student's
    positives are exactly the boxes the user drew, before and after.
    """

    is_pu_criterion = True  # read by the trainer's callbacks

    def __init__(self, model, ignore_conf=IGNORE_CONF, dedup_iou=DEDUP_IOU,
                 max_teacher_det=MAX_TEACHER_DET):
        super().__init__(model)
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

    @torch.no_grad()
    def _teacher_boxes(self, imgs):
        """Teacher detections per image as xyxy in the pixel space of ``imgs``."""
        was_training = self.teacher.training
        self.teacher.eval()
        preds = self.teacher(imgs)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        dets = non_max_suppression(
            preds,
            conf_thres=max(self.ignore_conf, 1e-3),
            iou_thres=0.5,
            max_det=self.max_teacher_det,
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

        Mirrors ``v8DetectionLoss.get_assigned_targets_and_loss`` line for line
        (ultralytics 8.4.x) because the ignore band has to be applied to
        ``bce_loss`` *before* it is reduced, and the stock method reduces on the
        way out. When PU is off this delegates instead, so nothing here can
        affect a non-PU run.
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

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)


# ----------------------------------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------------------------------


class PUDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that keeps an EMA teacher and runs the PU schedule.

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

    def _setup_train(self):
        super()._setup_train()

        self.pu_warmup = pu_warmup_epochs(self.epochs)
        student = unwrap_model(self.model)

        self.teacher = deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher = self.teacher.to(self.device)
        self._teacher_updates = 0

        criterion = PUDetectionLoss(student)
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

        self._pu_best_recall = None
        self.add_callback("on_train_epoch_start", self._pu_epoch_start)
        self.add_callback("on_train_batch_end", self._pu_update_teacher)
        self.add_callback("on_fit_epoch_end", self._pu_track_recall)
        self.add_callback("on_train_end", self._pu_finalize)

        if RANK in (-1, 0):
            LOGGER.info(f"PU: ignore-band training. warmup={self.pu_warmup} epochs "
                        f"(GT only), ignore_conf={IGNORE_CONF}, ema_decay={EMA_DECAY}.")

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
