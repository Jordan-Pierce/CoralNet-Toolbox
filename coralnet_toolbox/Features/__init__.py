"""
Features module: Unified feature extraction and management system.

Provides dense per-image feature maps as a reusable product, with model registry,
extraction pipelines, and integration with the Raster product system.
"""

from coralnet_toolbox.Features.ModelRegistry import (
    TRANSFORMER_MODELS,
    TIMM_MODELS,
    OPENCLIP_MODELS,
    YOLO_MODELS,
    is_transformer_model,
    is_timm_model,
    is_openclip_model,
    is_yolo_model,
)
from coralnet_toolbox.Features.QueryEngine import QueryEngine

__all__ = [
    'TRANSFORMER_MODELS',
    'TIMM_MODELS',
    'OPENCLIP_MODELS',
    'YOLO_MODELS',
    'is_transformer_model',
    'is_timm_model',
    'is_openclip_model',
    'is_yolo_model',
    'QueryEngine',
]
