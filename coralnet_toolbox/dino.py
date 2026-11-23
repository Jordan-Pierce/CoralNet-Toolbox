"""DINOv2 as a frozen Ultralytics backbone, for classify, detect, segment and semantic models.

The Community configs (MachineLearning/Community/cfg/<task>/dinov2_*.yaml) name
DINOBackbone as their first layer. Ultralytics' parse_model looks a yaml module
up by name in ultralytics.nn.tasks, so importing this module registers the
class there; no Ultralytics source is changed. The Community cfg package
imports it, so every dialog that lists those yamls has it registered.

DINOBackbone returns a list: the input image, then one feature map per
requested block. Index rows in the yaml pick from it and declare each map's
channel count, which parse_model cannot infer for a module it does not know.
This is the layout TorchVision(split=True) gives the swin_t yamls.

Pipeline details, all handled here so the yamls stay declarative:

* Ultralytics feeds RGB scaled to 0-1 with no mean/std. DINOv2 expects
  ImageNet normalisation, so the wrapper applies it.
* DINOv2 patches are 14 px, which would give non-integer strides. The input is
  resized so the token grid is H/16 x W/16, a stride of 16 that the detection,
  segmentation and semantic heads can use. A 640 px input runs DINO at 560 px.
* Feature maps get DINOv2's final LayerNorm, as in its linear evaluations. In
  pool mode (classify) the output is the CLS token concatenated with the mean
  patch token, shaped (B, 2C, 1, 1) for Ultralytics' Classify head.

DINO is never trained. The trainer re-enables gradients on every parameter
that `freeze` does not name, so the wrapper cannot freeze itself. Train Model
passes dino_freeze_list(model) as `freeze`, whatever Freeze Layers says.

Transformers 5 models cannot be pickled (they hold local hook functions), and
Ultralytics checkpoints are pickles. The wrapper therefore pickles DINO as
its config plus state dict and rebuilds the network on load. A trained best.pt
refers only to coralnet_toolbox.dino.DINOBackbone, not to transformers'
classes, so it survives transformers upgrades that keep the weight names. The
class must stay importable at that path.

This module imports nothing from Qt: unpickling a trained model imports it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# DINOv2 was trained on ImageNet-normalised RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Stride of the token grid the wrapper produces, whatever the model's patch size
OUT_STRIDE = 16


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def _pretrained_net(model_id):
    """Load a HuggingFace DINO model with its pretrained weights (cached after the first download)."""
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_id)


def _net_from_config(config):
    """Build a HuggingFace model from a config dict, with no download; its weights are loaded after."""
    from transformers import AutoConfig, AutoModel
    config = dict(config)
    return AutoModel.from_config(AutoConfig.for_model(config.pop("model_type"), **config))


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DINOBackbone(nn.Module):
    """A frozen DINOv2 encoder as an Ultralytics layer.

    Args:
        model_id (str): HuggingFace model id, e.g. 'facebook/dinov2-small'.
        indices (list[int]): Transformer blocks to return, counted from 0.
        pool (bool): Return one (B, 2C, 1, 1) CLS ‖ mean-patch vector, taken
            from the last of `indices`, instead of feature maps.
        pretrained (bool): Load the pretrained weights. A model built only to
            receive a checkpoint's weights can skip them.

    Returns from forward:
        list[Tensor]: [input image, *outputs]. Outputs are (B, C, H/16, W/16)
            maps, one per index, or a single pooled vector in pool mode.
    """

    def __init__(self, model_id="facebook/dinov2-small", indices=(11,), pool=False, pretrained=True):
        super().__init__()
        self.model_id = str(model_id)
        self.indices = [int(i) for i in indices]
        self.pool = bool(pool)
        self.net = _pretrained_net(self.model_id) if pretrained else None
        if self.net is None:
            from transformers import AutoConfig, AutoModel
            self.net = AutoModel.from_config(AutoConfig.from_pretrained(self.model_id))
        self._read_config()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _read_config(self):
        """Cache the network's patch size, prefix-token count and width."""
        config = self.net.config
        self.patch = int(config.patch_size)
        # CLS, plus register tokens for DINOv2-with-registers and DINOv3
        self.prefix = 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        self.channels = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        bad = [i for i in self.indices if not 0 <= i < num_layers]
        if bad:
            raise ValueError(f"{self.model_id} has blocks 0-{num_layers - 1}; the yaml asks for {bad}.")

    def _norm(self, tokens):
        """DINO's final LayerNorm, which its linear evaluations apply to every block they read."""
        norm = getattr(self.net, "layernorm", None) or getattr(self.net, "norm", None)
        return norm(tokens) if norm is not None else tokens

    def train(self, mode=True):
        """Keep DINO in eval mode. It has no dropout or BatchNorm, but its features should not depend on the mode."""
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x):
        b, _, h, w = x.shape
        gh, gw = max(h // OUT_STRIDE, 1), max(w // OUT_STRIDE, 1)
        z = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        size = (gh * self.patch, gw * self.patch)
        if size != (h, w):
            z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
        hidden = self.net(pixel_values=z, output_hidden_states=True).hidden_states  # [embeddings, block 0, ...]

        if self.pool:
            tokens = self._norm(hidden[self.indices[-1] + 1])
            vector = torch.cat([tokens[:, 0], tokens[:, self.prefix:].mean(1)], dim=1)
            return [x, vector[:, :, None, None]]

        maps = []
        for i in self.indices:
            tokens = self._norm(hidden[i + 1])[:, self.prefix:]
            maps.append(tokens.transpose(1, 2).reshape(b, self.channels, gh, gw))
        return [x, *maps]

    # Pickling: DINO travels as config + weights (see the module docstring)

    def __getstate__(self):
        state = self.__dict__.copy()
        modules = dict(state["_modules"])
        net = modules.pop("net")
        state["_modules"] = modules
        state["_net_config"] = net.config.to_dict()
        state["_net_state"] = net.state_dict()
        return state

    def __setstate__(self, state):
        config = state.pop("_net_config")
        weights = state.pop("_net_state")
        super().__setstate__(state)
        net = _net_from_config(config)
        first = next(iter(weights.values()), None)
        if first is not None:
            net.to(device=first.device, dtype=first.dtype)
        net.load_state_dict(weights)
        net.eval()
        self._modules["net"] = net


# ----------------------------------------------------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------------------------------------------------


def _layers(model):
    """The top-level layers of an Ultralytics model, unwrapping YOLO, DDP and torch.compile."""
    model = getattr(model, "model", model)  # YOLO -> DetectionModel etc.
    model = getattr(model, "module", model)
    model = getattr(model, "_orig_mod", model)
    layers = getattr(model, "model", model)
    return layers if isinstance(layers, nn.Sequential) else []


def dino_layer_indices(model):
    """Indices of the DINOBackbone layers in an Ultralytics model (empty for any other model)."""
    return [i for i, layer in enumerate(_layers(model)) if isinstance(layer, DINOBackbone)]


def dino_freeze_list(model):
    """Ultralytics `freeze` entries that freeze every DINO layer and nothing else.

    Ultralytics freezes a parameter whose name contains 'model.<entry>.'.

    Returns:
        list[str]: e.g. ['0'], or [] for a model without DINO.
    """
    return [str(i) for i in dino_layer_indices(model)]


def register():
    """Make DINOBackbone resolvable by name in Ultralytics model yamls."""
    import ultralytics.nn.tasks as tasks
    if getattr(tasks, "DINOBackbone", None) is not DINOBackbone:
        tasks.DINOBackbone = DINOBackbone


register()
