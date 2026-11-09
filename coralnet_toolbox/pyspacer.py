"""CoralNet's PySpacer EfficientNet-B0 weights, as an Ultralytics classification backbone.

CoralNet does not distribute these weights, so they are not downloaded; a user
who has them imports them on the Train Classification Model dialog's Use
Existing Model tab. A file whose name contains CORALNET_WEIGHTS_NAME is taken
to be that checkpoint.

PySpacer's checkpoint is not an Ultralytics model. It is a training snapshot
(`net`, `optimizer`, `scheduler`, `epoch`) of lukemelas' efficientnet_pytorch
EfficientNet-B0, saved through DataParallel (so every key starts `module.`),
with a 1275-way CoralNet label head. The community efficientnet_b0.yaml builds
torchvision's EfficientNet-B0, which has the same tensors under different
names, so the weights are copied across by structure.

Copying the weights is not enough. Torchvision's network is not PySpacer's:

* PySpacer pads like TensorFlow's 'same' (static padding for 224 px inputs),
  which on an even-sized input puts a stride-2 convolution's extra row and
  column on the bottom and right only. Torchvision pads symmetrically, so each
  of B0's five stride-2 convolutions samples a grid shifted by a pixel.
* PySpacer's BatchNorm eps is 1e-3; torchvision's is 1e-5.

With the weights copied and neither difference fixed, features on ARMS
patches had a cosine similarity of 0.54 to PySpacer's, and the two networks
agreed on CoralNet's top-1 label for only 21% of patches. A linear probe on
those features lost 6 points of accuracy against the real ones. Fixing both
differences reproduces PySpacer exactly. apply_pyspacer_architecture makes
both fixes.

Ultralytics rebuilds a model from its yaml whenever it trains one, which
discards both fixes. The converted model's yaml therefore carries
PYSPACER_MARKER, and restore_pyspacer_architecture reapplies the fixes to every
marked model the trainer builds. Deployment needs no help: a trained best.pt
is a pickled module and keeps them.

This module imports nothing from Qt. Unpickling a trained best.pt imports it
for Conv2dSamePadding, and loading a model should not require the GUI.
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# A weights file whose name contains this is taken to be CoralNet's PySpacer checkpoint
CORALNET_WEIGHTS_NAME = "coralnet_efficientnet-b0.pt"

# model.yaml key marking a model whose backbone must keep PySpacer's architecture
PYSPACER_MARKER = "coralnet_pyspacer"

# lukemelas' efficientnet-b0 global params: batch_norm_epsilon=1e-3
PYSPACER_BN_EPS = 1e-3

# Blocks per stage of EfficientNet-B0; 16 MBConv blocks in all
_B0_STAGE_DEPTHS = (1, 2, 2, 3, 3, 4, 1)
_BN_TENSORS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")

_COMMUNITY_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "MachineLearning", "Community", "cfg", "classify", "efficientnet_b0.yaml")


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Conv2dSamePadding(nn.Conv2d):
    """A Conv2d padded the way TensorFlow's 'same' padding pads, as PySpacer's EfficientNet does.

    The padding is computed from each input rather than fixed for one image
    size, so it is right at any imgsz and equals PySpacer's static padding at
    224. Swapped onto an existing nn.Conv2d by __class__ (see
    apply_pyspacer_architecture), so state-dict keys are unchanged.

    A trained model pickles this class by reference: it must stay importable
    at coralnet_toolbox.pyspacer.Conv2dSamePadding.
    """

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ph = max((-(-ih // sh) - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pw = max((-(-iw // sw) - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if ph or pw:
            x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
        return super().forward(x)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def is_coralnet_weights(path):
    """Whether a model path names CoralNet's PySpacer EfficientNet-B0 checkpoint.

    Decided by name alone: the file is assumed to be CoralNet's when its name
    contains CORALNET_WEIGHTS_NAME. The converted Ultralytics checkpoint
    ('..._ultralytics.pt') does not match, and neither does a trained best.pt.

    Args:
        path (str | Path | None): Model path or name.

    Returns:
        bool: True for a CoralNet weights file.
    """
    if not path:
        return False
    return CORALNET_WEIGHTS_NAME in os.path.basename(str(path)).lower()


def pyspacer_key_map():
    """Map each PySpacer backbone tensor name to its name in torchvision's EfficientNet-B0 `features`.

    Built from the two architectures' structure rather than by pairing the
    state dicts in order, so a layout that differs fails loudly here instead
    of silently pairing the wrong tensors. The 1275-way `_fc` head has no
    counterpart and is left out.

    Returns:
        dict: {PySpacer key (without 'module.'): torchvision features key}.
    """
    key_map = {"_conv_stem.weight": "0.0.weight"}
    for t in _BN_TENSORS:
        key_map[f"_bn0.{t}"] = f"0.1.{t}"

    block = 0
    for stage, depth in enumerate(_B0_STAGE_DEPTHS, start=1):
        for j in range(depth):
            src, dst = f"_blocks.{block}.", f"{stage}.{j}.block."
            # Stage 1 has expand ratio 1, so no expansion convolution
            idx = 0
            if stage != 1:
                key_map[src + "_expand_conv.weight"] = dst + "0.0.weight"
                for t in _BN_TENSORS:
                    key_map[src + f"_bn0.{t}"] = dst + f"0.1.{t}"
                idx = 1
            key_map[src + "_depthwise_conv.weight"] = dst + f"{idx}.0.weight"
            for t in _BN_TENSORS:
                key_map[src + f"_bn1.{t}"] = dst + f"{idx}.1.{t}"
            key_map[src + "_se_reduce.weight"] = dst + f"{idx + 1}.fc1.weight"
            key_map[src + "_se_reduce.bias"] = dst + f"{idx + 1}.fc1.bias"
            key_map[src + "_se_expand.weight"] = dst + f"{idx + 1}.fc2.weight"
            key_map[src + "_se_expand.bias"] = dst + f"{idx + 1}.fc2.bias"
            key_map[src + "_project_conv.weight"] = dst + f"{idx + 2}.0.weight"
            for t in _BN_TENSORS:
                key_map[src + f"_bn2.{t}"] = dst + f"{idx + 2}.1.{t}"
            block += 1

    key_map["_conv_head.weight"] = "8.0.weight"
    for t in _BN_TENSORS:
        key_map[f"_bn1.{t}"] = f"8.1.{t}"
    return key_map


def load_pyspacer_state_dict(weights_path):
    """Read the network weights out of a PySpacer checkpoint.

    Loaded with weights_only=True. The snapshot's `epoch` is a numpy integer,
    which torch>=2.6 refuses by default, so numpy's scalar reconstructor is
    allowlisted, under the numpy 1 name the file was pickled with. That is
    what PySpacer itself does. Nothing else in the file needs to be trusted.

    Args:
        weights_path (str): Path to the checkpoint.

    Returns:
        dict: The `net` state dict with DataParallel's 'module.' prefix removed.

    Raises:
        ValueError: If the file is not a PySpacer EfficientNet-B0 checkpoint.
    """
    import numpy as np

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        # torch < 2.5 has no allowlist, and its torch.load did not default to weights_only
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    else:
        scalar = (getattr(np, "_core", None) or np.core).multiarray.scalar
        allowed = [np.dtype, type(np.dtype(np.int64))]
        if scalar.__module__ == "numpy.core.multiarray":
            allowed.append(scalar)
        else:
            # numpy 2 moved it to numpy._core; the file names the old path
            allowed.append((scalar, "numpy.core.multiarray.scalar"))
        with safe_globals(allowed):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    net = checkpoint.get("net") if isinstance(checkpoint, dict) else None
    if not isinstance(net, dict) or not any(k.endswith("_conv_stem.weight") for k in net):
        raise ValueError(f"{os.path.basename(weights_path)} is named like CoralNet's EfficientNet-B0 weights, "
                         f"but it is not a PySpacer checkpoint (no 'net' state dict with an EfficientNet stem).")

    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in net.items()}


def apply_pyspacer_architecture(features):
    """Give torchvision EfficientNet-B0 layers PySpacer's padding and BatchNorm eps, in place.

    Only the stride-2 convolutions are swapped to Conv2dSamePadding. For B0's
    stride-1 convolutions (odd kernels, no dilation) TF 'same' padding is the
    same symmetric padding torchvision already uses. Safe to call again.

    Args:
        features (nn.Module): The torchvision backbone, e.g. a TorchVision layer's `.m`.
    """
    for m in features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = PYSPACER_BN_EPS
        elif type(m) is nn.Conv2d and tuple(m.stride) != (1, 1):
            m.__class__ = Conv2dSamePadding
            m.padding = (0, 0)


def _torchvision_layers(model):
    """The TorchVision backbone layers of an Ultralytics model."""
    from ultralytics.nn.modules.block import TorchVision
    return [layer for layer in getattr(model, "model", []) if isinstance(layer, TorchVision)]


def _apply_if_marked(model):
    """Apply PySpacer's architecture to a model whose yaml carries PYSPACER_MARKER."""
    if model is None:
        return False
    # Under DDP or torch.compile the Ultralytics model is wrapped
    model = getattr(model, "module", model)
    model = getattr(model, "_orig_mod", model)
    cfg = getattr(model, "yaml", None)
    if not isinstance(cfg, dict) or not cfg.get(PYSPACER_MARKER):
        return False
    for layer in _torchvision_layers(model):
        apply_pyspacer_architecture(layer.m)
    return True


def restore_pyspacer_architecture(trainer):
    """Ultralytics callback (on_pretrain_routine_end): reapply PySpacer's architecture after the rebuild.

    Model.train builds the model it trains from the checkpoint's yaml, so a
    CoralNet backbone arrives as plain torchvision with the right weights but
    the wrong padding and eps. This runs once the trainer has its model and
    EMA, before the first batch. The EMA is the copy that gets validated and
    saved, so it is fixed too. Models without the marker are left alone.
    """
    fixed = _apply_if_marked(getattr(trainer, "model", None))
    ema = getattr(trainer, "ema", None)
    _apply_if_marked(getattr(ema, "ema", None))
    if fixed:
        print("CoralNet EfficientNet-B0: using PySpacer's padding and BatchNorm eps for the backbone.")


def convert_coralnet_weights(weights_path, output_dir):
    """Build an Ultralytics EfficientNet-B0 classifier from CoralNet's PySpacer weights.

    The backbone comes from the community efficientnet_b0.yaml, built without
    ImageNet weights: every tensor those weights would supply is overwritten
    here, and leaving them in made every later rebuild download them, which
    fails offline. Every backbone tensor must be filled from CoralNet's
    weights. The Classify head stays freshly initialised, since CoralNet's
    1275-way head has no place in it.

    Args:
        weights_path (str): CoralNet's PySpacer checkpoint.
        output_dir (str): Folder to write the converted checkpoint to.

    Returns:
        str: Path to the converted checkpoint, '<name>_ultralytics.pt', loadable with YOLO().

    Raises:
        ValueError: If the weights do not fit EfficientNet-B0.
    """
    from ultralytics import __version__
    from ultralytics.nn.tasks import ClassificationModel, yaml_model_load

    state = load_pyspacer_state_dict(weights_path)

    cfg = yaml_model_load(_COMMUNITY_YAML)
    for layer in cfg["backbone"]:
        if layer[2] == "TorchVision":
            layer[3][2] = None  # weights: none, rather than 'DEFAULT' (ImageNet)
    cfg[PYSPACER_MARKER] = True
    model = ClassificationModel(cfg, verbose=False)

    (backbone,) = _torchvision_layers(model)
    target = backbone.m.state_dict()
    key_map = pyspacer_key_map()

    unmapped = sorted(k for k in state if k not in key_map and not k.startswith("_fc."))
    unfilled = sorted(set(target) - set(key_map.values()))
    if unmapped or unfilled:
        raise ValueError(f"{os.path.basename(weights_path)} does not match EfficientNet-B0: "
                         f"unexpected {unmapped[:5]}, missing {unfilled[:5]}.")

    loaded = {}
    for src, dst in key_map.items():
        if src not in state:
            raise ValueError(f"{os.path.basename(weights_path)} is missing {src}.")
        if state[src].shape != target[dst].shape:
            raise ValueError(f"{os.path.basename(weights_path)}: {src} is {tuple(state[src].shape)}, "
                             f"EfficientNet-B0 expects {tuple(target[dst].shape)}.")
        loaded[dst] = state[src]
    backbone.m.load_state_dict(loaded, strict=True)
    apply_pyspacer_architecture(backbone.m)
    # Without train_args, YOLO() takes the task from Ultralytics' defaults: 'detect'
    model.task = "classify"
    train_args = {"task": "classify", "imgsz": 224}  # PySpacer crops 224 px patches

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    output_path = os.path.join(output_dir, f"{stem}_ultralytics.pt").replace("\\", "/")
    # Full precision. Ultralytics' Model.save halves the model, which rounds
    # some of the backbone's BatchNorm variances to zero.
    torch.save({"model": model,
                "ema": None,
                "train_args": train_args,
                "date": datetime.now().astimezone().isoformat(),
                "version": __version__,
                "source": os.path.abspath(weights_path).replace("\\", "/")},
               output_path)
    print(f"CoralNet EfficientNet-B0: converted {os.path.basename(weights_path)} to {output_path}")
    return output_path
