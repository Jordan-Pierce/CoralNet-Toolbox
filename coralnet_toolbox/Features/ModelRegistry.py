"""
Model registry for the Explorer tool.

Consolidates transformer and YOLO model configurations and helpers
(formerly split across transformer_models.py and yolo_models.py).
"""

from __future__ import annotations

import os


# ----------------------------------------------------------------------------------------------------------------------
# Transformer models
# ----------------------------------------------------------------------------------------------------------------------


TRANSFORMER_MODELS = {
    'ResNet-50': 'microsoft/resnet-50',
    'ResNet-101': 'microsoft/resnet-101',
    'Swin Transformer (Tiny)': 'microsoft/swin-tiny-patch4-window7-224',
    'Swin Transformer (Base)': 'microsoft/swin-base-patch4-window7-224',
    'ViT (Base)': 'google/vit-base-patch16-224',
    'ViT (Large)': 'google/vit-large-patch16-224',
    'DINOv2 (Small)': 'facebook/dinov2-small',
    'DINOv2 (Base)': 'facebook/dinov2-base',
    'DINOv2 (Large)': 'facebook/dinov2-large',
    'DINOv2 (Giant)': 'facebook/dinov2-giant',
    'DINOv2 (Giant ImageNet1k)': 'facebook/dinov2-giant-imagenet1k-1-layer',
    # DINOv2 with registers: extra register tokens absorb high-norm artifact
    # patches, yielding cleaner dense similarity maps. Ungated.
    'DINOv2+reg (Small)': 'facebook/dinov2-with-registers-small',
    'DINOv2+reg (Base)': 'facebook/dinov2-with-registers-base',
    'DINOv2+reg (Large)': 'facebook/dinov2-with-registers-large',
    'DINOv2+reg (Giant)': 'facebook/dinov2-with-registers-giant',
    # ConvNeXt V2: pure-convolutional backbone whose cost scales linearly with
    # resolution (no attention token² blowup), so it loads larger images closer
    # to native res. Ungated.
    'ConvNeXt V2 (Nano)': 'facebook/convnextv2-nano-22k-224',
    'ConvNeXt V2 (Tiny)': 'facebook/convnextv2-tiny-22k-224',
    'ConvNeXt V2 (Base)': 'facebook/convnextv2-base-22k-224',
}

try:
    from transformers import pipeline
    from huggingface_hub import snapshot_download

    # Check if HF_TOKEN environment variable is set
    # (if not, user can definitely not access the model)
    hf_token = os.getenv("HF_TOKEN")

    if hf_token and hf_token.strip():
        # Add the DINOv3 models if the HuggingFace token is set
        TRANSFORMER_MODELS.update({
            'DINOv3 ConvNext (Tiny)': 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
            'DINOv3 ConvNext (Small)': 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
            'DINOv3 ConvNext (Base)': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
            'DINOv3 ConvNext (Large)': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
            'DINOv3 ViT (Small/16)': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'DINOv3 ViT (Small/16+)': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
            'DINOv3 ViT (Base/16)': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'DINOv3 ViT (Large/16)': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'DINOv3 ViT (Huge/16+)': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        })

except Exception:
    pass


# ----------------------------------------------------------------------------------------------------------------------
# timm models
# ----------------------------------------------------------------------------------------------------------------------


# timm backbones, addressed by the "timm:" prefix so routing can tell them apart
# from HuggingFace ids (which the transformer path loads via the HF pipeline).
# These are loaded through TimmExtractor, which pulls the last-stage spatial map
# via forward_intermediates (timm >= 1.0) in a uniform [B, C, H, W] layout for
# both ViT and CNN backbones. All entries here are dense-capable.
TIMM_PREFIX = "timm:"

TIMM_MODELS = {
    # EVA-02: strong dense ViT features for cosine similarity (CLIP-normalized).
    'EVA-02 (Base/14)': 'timm:eva02_base_patch14_224.mim_in22k',
    'EVA-02 (Large/14)': 'timm:eva02_large_patch14_224.mim_in22k',
    # FastViT: fast hybrid conv-transformer, strong at higher resolution.
    'FastViT (SA12)': 'timm:fastvit_sa12.apple_in1k',
    'FastViT (SA24)': 'timm:fastvit_sa24.apple_in1k',
    # EfficientViT: near-linear scaling with resolution — a "near-native res"
    # workhorse alongside ConvNeXt V2.
    'EfficientViT (B1)': 'timm:efficientvit_b1.r224_in1k',
    'EfficientViT (B2)': 'timm:efficientvit_b2.r224_in1k',
}


def is_timm_model(model_name: str) -> bool:
    """Return True for a timm-backed model id (carries the ``timm:`` prefix)."""
    return isinstance(model_name, str) and model_name.startswith(TIMM_PREFIX)


def strip_timm_prefix(model_name: str) -> str:
    """Strip the ``timm:`` prefix, yielding the bare timm model name."""
    return model_name[len(TIMM_PREFIX):] if is_timm_model(model_name) else model_name


# ----------------------------------------------------------------------------------------------------------------------
# open_clip models
# ----------------------------------------------------------------------------------------------------------------------


# open_clip backbones, addressed by the "openclip:" prefix so routing can
# distinguish them from HuggingFace ids (which go through the HF pipeline).
# Loaded via OpenCLIPExtractor using open_clip.create_model_and_transforms().
OPENCLIP_PREFIX = "openclip:"

OPENCLIP_MODELS = {
    # BioCLIP: biology-domain CLIP (imageomics).
    'BioCLIP (ViT-B/16)': 'openclip:hf-hub:imageomics/bioclip',
    'BioCLIP v2': 'openclip:hf-hub:imageomics/bioclip-2',
    'BioCLIP v2.5 (ViT-H/14)': 'openclip:hf-hub:imageomics/bioclip-2.5-vith14',
}


def is_openclip_model(model_name: str) -> bool:
    """Return True for an open_clip-backed model id (carries the ``openclip:`` prefix)."""
    return isinstance(model_name, str) and model_name.startswith(OPENCLIP_PREFIX)


def strip_openclip_prefix(model_name: str) -> str:
    """Strip the ``openclip:`` prefix, yielding the bare open_clip model name."""
    return model_name[len(OPENCLIP_PREFIX):] if is_openclip_model(model_name) else model_name


# ----------------------------------------------------------------------------------------------------------------------
# Transformer models (HuggingFace pipeline)
# ----------------------------------------------------------------------------------------------------------------------


def is_transformer_model(model_name: str) -> bool:
    """
    Determine if a model name refers to a transformer model.

    Checks if the model name indicates a HuggingFace transformer model
    that should be handled by the transformer feature extraction pipeline.

    Args:
        model_name: The model name to check.

    Returns:
        True if this is a transformer model, False otherwise.
    """
    if not model_name or not isinstance(model_name, str):
        return False

    # timm / open_clip models have their own backends; never claim them here.
    if is_timm_model(model_name):
        return False
    if is_openclip_model(model_name):
        return False

    # Check if it's one of our known transformer model IDs
    if model_name in TRANSFORMER_MODELS.values():
        return True

    # Check for common HuggingFace model naming patterns
    # Models from HuggingFace typically contain '/' in their names
    if "/" in model_name:
        known_prefixes = (
            "facebook/",
            "microsoft/",
            "google/",
            "openai/",
            "imageomics/",
        )
        if model_name.startswith(known_prefixes):
            return True
        # Any model with '/' is likely a HuggingFace model
        return True

    return False


# ----------------------------------------------------------------------------------------------------------------------
# YOLO models
# ----------------------------------------------------------------------------------------------------------------------


# Dictionary mapping display names to model file names (classification models only)
YOLO_MODELS = {
    # YOLOv8 classification models
    'YOLOv8 (Nano)': 'yolov8n-cls.pt',
    'YOLOv8 (Small)': 'yolov8s-cls.pt',
    'YOLOv8 (Medium)': 'yolov8m-cls.pt',
    'YOLOv8 (Large)': 'yolov8l-cls.pt',
    'YOLOv8 (X-Large)': 'yolov8x-cls.pt',

    # YOLOv11 classification models
    'YOLOv11 (Nano)': 'yolo11n-cls.pt',
    'YOLOv11 (Small)': 'yolo11s-cls.pt',
    'YOLOv11 (Medium)': 'yolo11m-cls.pt',
    'YOLOv11 (Large)': 'yolo11l-cls.pt',
    'YOLOv11 (X-Large)': 'yolo11x-cls.pt',

    # YOLOv12 classification models
    # 'YOLOv12 (Nano)': 'yolo12n-cls.pt',
    # 'YOLOv12 (Small)': 'yolo12s-cls.pt',
    # 'YOLOv12 (Medium)': 'yolo12m-cls.pt',
    # 'YOLOv12 (Large)': 'yolo12l-cls.pt',
    # 'YOLOv12 (X-Large)': 'yolo12x-cls.pt',

    # YOLO26 classification models
    'YOLO26 (Nano)': 'yolo26n-cls.pt',
    'YOLO26 (Small)': 'yolo26s-cls.pt',
    'YOLO26 (Medium)': 'yolo26m-cls.pt',
    'YOLO26 (Large)': 'yolo26l-cls.pt',
    'YOLO26 (X-Large)': 'yolo26x-cls.pt',
}

LIVE_YOLO_MODEL_PREFIX = "live_yolo::"


def is_live_yolo_model(model_name: str) -> bool:
    """
    Determine if a model name refers to a live YOLO model source.

    Args:
        model_name: The model name to check.

    Returns:
        True if this is a live YOLO model source, False otherwise.
    """
    return isinstance(model_name, str) and model_name.startswith(LIVE_YOLO_MODEL_PREFIX)


def get_community_models(task: str = 'classify') -> dict:
    """
    Get available community models for a specific task.

    Args:
        task: The task type, default is 'classify'.

    Returns:
        Dictionary of community models.
    """
    try:
        from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
        return get_available_configs(task=task)
    except Exception:
        return {}


def is_yolo_model(model_name: str) -> bool:
    """
    Determine if a model name refers to a YOLO model.

    Args:
        model_name: The model name to check.

    Returns:
        True if this is a YOLO model, False otherwise.
    """
    if not model_name or not isinstance(model_name, str):
        return False

    if is_live_yolo_model(model_name):
        return True

    if model_name in YOLO_MODELS.values():
        return True

    community_models = get_community_models()
    if community_models:
        if model_name in community_models or model_name in community_models.values():
            return True

    if model_name.lower().endswith('.pt'):
        return True

    return False


def get_yolo_model_task(model_name: str) -> str:
    """
    Determine the task type of a YOLO model based on its name.

    Args:
        model_name: The model name or path.

    Returns:
        One of 'classify', 'detect', 'segment', or 'unknown'.
    """
    if not model_name or not isinstance(model_name, str):
        return 'unknown'

    if is_live_yolo_model(model_name):
        parts = model_name.split('::', 3)
        if len(parts) >= 4 and parts[2]:
            return parts[2]
        return 'unknown'

    community_models = get_community_models()
    if community_models:
        if model_name in community_models or model_name in community_models.values():
            return 'classify'

    filename = model_name.split('/')[-1].split('\\')[-1].lower()

    if '-cls' in filename:
        return 'classify'
    elif '-seg' in filename:
        return 'segment'
    elif filename.endswith('.pt'):
        return 'detect'

    return 'unknown'
