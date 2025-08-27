"""
YOLO models configuration for the Explorer tool.

This module contains the YOLO models dictionary used in the Explorer tool.
It's extracted into a separate module to allow easy importing in tests without
Qt dependencies.
"""

from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs

# Dictionary mapping display names to model file names (classification models only)
YOLO_MODELS = {
    # YOLOv8 classification models
    'YOLOv8 (Nano)': 'yolov8n-cls.pt',
    'YOLOv8 (Small)': 'yolov8s-cls.pt',
    'YOLOv8 (Medium)': 'yolov8m-cls.pt',
    'YOLOv8 (Large)': 'yolov8l-cls.pt',
    'YOLOv8 (X-Large)': 'yolov8x-cls.pt',
    
    # YOLOv11 classification models
    'YOLOv11 (Nano)': 'yolov11n-cls.pt',
    'YOLOv11 (Small)': 'yolov11s-cls.pt',
    'YOLOv11 (Medium)': 'yolov11m-cls.pt',
    'YOLOv11 (Large)': 'yolov11l-cls.pt',
    'YOLOv11 (X-Large)': 'yolov11x-cls.pt',
    
    # YOLOv12 classification models
    'YOLOv12 (Nano)': 'yolov12n-cls.pt',
    'YOLOv12 (Small)': 'yolov12s-cls.pt',
    'YOLOv12 (Medium)': 'yolov12m-cls.pt',
    'YOLOv12 (Large)': 'yolov12l-cls.pt',
    'YOLOv12 (X-Large)': 'yolov12x-cls.pt',
}


def get_community_models(task='classify'):
    """
    Get available community models for a specific task.
    
    Args:
        task (str): The task type, default is 'classify'
        
    Returns:
        dict: Dictionary of community models
    """
    return get_available_configs(task=task)


def is_yolo_model(model_name):
    """
    Determine if a model name refers to a YOLO model.
    
    This function checks if the model name indicates a YOLO model
    that should be handled by the YOLO feature extraction pipeline.
    
    Args:
        model_name (str): The model name to check
        
    Returns:
        bool: True if this is a YOLO model, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False
        
    # Check if it's one of our known YOLO model IDs
    if model_name in YOLO_MODELS.values():
        return True
    
    # Check if it's a community model (check both keys and values)
    community_models = get_community_models()
    if community_models:
        if model_name in community_models or model_name in community_models.values():
            return True
    
    # Check for .pt file extension (any PyTorch model file)
    if model_name.lower().endswith('.pt'):
        return True
    
    return False


def get_yolo_model_task(model_name):
    """
    Determine the task type of a YOLO model based on its name.
    
    Args:
        model_name (str): The model name or path
        
    Returns:
        str: One of 'classify', 'detect', 'segment', or 'unknown'
    """
    if not model_name or not isinstance(model_name, str):
        return 'unknown'
    
    # Check if it's a community model - all community models are classify
    community_models = get_community_models()
    if community_models:
        if model_name in community_models or model_name in community_models.values():
            return 'classify'
    
    # Extract just the filename if a full path is provided
    filename = model_name.split('/')[-1].split('\\')[-1].lower()
    
    if '-cls' in filename:
        return 'classify'
    elif '-seg' in filename:
        return 'segment'
    elif filename.endswith('.pt'):
        # Default YOLO models without specific suffixes are detection models
        return 'detect'
    
    return 'unknown'