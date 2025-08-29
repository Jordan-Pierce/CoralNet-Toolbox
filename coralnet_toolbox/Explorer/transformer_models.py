"""
Transformer models configuration for the Explorer tool.

This module contains the transformer models dictionary used in the Explorer tool.
It's extracted into a separate module to allow easy importing in tests without
Qt dependencies.
"""

TRANSFORMER_MODELS = {
    'ResNet-50': 'microsoft/resnet-50',
    'ResNet-101': 'microsoft/resnet-101',
    'DINOv2 (Small)': 'facebook/dinov2-small',
    'DINOv2 (Base)': 'facebook/dinov2-base',
    'DINOv2 (Large)': 'facebook/dinov2-large',
    'DINOv2 (Giant)': 'facebook/dinov2-giant',
    'DINOv2 (Giant ImageNet1k)': 'facebook/dinov2-giant-imagenet1k-1-layer',
    'DINOv3 ConvNext (Tiny)': 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
    'DINOv3 ConvNext (Small)': 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
    'DINOv3 ConvNext (Base)': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
    'DINOv3 ConvNext (Large)': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
    'DINOv3 ViT (Small/16)': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'DINOv3 ViT (Small/16+)': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
    'DINOv3 ViT (Base/16)': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'DINOv3 ViT (Large/16)': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'DINOv3 ViT (Huge/16+)': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
    'DINOv3 ViT (7B/16)': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    'Swin Transformer (Tiny)': 'microsoft/swin-tiny-patch4-window7-224',
    'Swin Transformer (Base)': 'microsoft/swin-base-patch4-window7-224',
    'ViT (Base)': 'google/vit-base-patch16-224',
    'ViT (Large)': 'google/vit-large-patch16-224',
}


def is_transformer_model(model_name):
    """
    Determine if a model name refers to a transformer model.
    
    This function checks if the model name indicates a HuggingFace transformer model
    that should be handled by the transformer feature extraction pipeline.
    
    Args:
        model_name (str): The model name to check
        
    Returns:
        bool: True if this is a transformer model, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False
        
    # Check if it's one of our known transformer model IDs
    if model_name in TRANSFORMER_MODELS.values():
        return True
    
    # Check for common HuggingFace model naming patterns
    # Models from HuggingFace typically contain '/' in their names
    if "/" in model_name:
        # Check for known organization prefixes
        known_prefixes = ("facebook/", 
                          "microsoft/", 
                          "google/", 
                          "openai/", 
                          "imageomics/")
        
        if model_name.startswith(known_prefixes):
            return True
        
        # Any model with '/' is likely a HuggingFace model
        return True
    
    return False