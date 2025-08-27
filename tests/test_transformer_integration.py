#!/usr/bin/env python3
"""
Test script for transformer model integration in CoralNet-Toolbox Explorer.

This script tests the new transformer feature extraction functionality
without launching the full GUI.
"""

import sys
import numpy as np
from PIL import Image
import tempfile
import os

# Test transformer import
try:
    from transformers import pipeline
    print("✅ Transformers library imported successfully")
except ImportError:
    print("❌ Transformers library not found. Please install with: pip install transformers")
    sys.exit(1)

# Import the transformer models dictionary and utility function from the actual implementation
try:
    from coralnet_toolbox.Explorer.transformer_models import TRANSFORMER_MODELS, is_transformer_model
    transformer_models = TRANSFORMER_MODELS
    print("✅ Successfully imported transformer models and utilities from implementation")
except ImportError as e:
    print(f"❌ FAILED: Could not import transformer models from implementation: {e}")
    print("   Tests cannot proceed without access to the actual transformer model configuration.")
    print("   Please ensure coralnet_toolbox.Explorer.transformer_models module is available.")
    sys.exit(1)

def test_transformer_pipeline():
    """Test basic transformer pipeline functionality."""
    print("\n=== Testing Transformer Pipeline ===")
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # Test models to try
    test_models = [
        "facebook/dinov2-small",
        "microsoft/resnet-50",
    ]
    
    for model_name in test_models:
        print(f"\nTesting model: {model_name}")
        try:
            # Initialize pipeline
            feature_extractor = pipeline(
                model=model_name,
                task="image-feature-extraction",
                device=-1  # CPU only for testing
            )
            
            # Extract features
            features = feature_extractor(test_image)
            
            # Check output format
            if isinstance(features, list):
                if len(features) > 0 and hasattr(features[0], 'shape'):
                    feature_shape = features[0].shape
                    print(f"  ✅ Features extracted: shape {feature_shape}")
                else:
                    print(f"  ✅ Features extracted: list of length {len(features)}")
            elif hasattr(features, 'shape'):
                print(f"  ✅ Features extracted: shape {features.shape}")
            else:
                print(f"  ✅ Features extracted: type {type(features)}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_model_list():
    """Test that the model list is properly defined."""
    print("\n=== Testing Model List Configuration ===")
    
    # transformer_models is already imported at the module level
    print(f"✅ {len(transformer_models)} transformer models configured")
    for display_name, model_id in transformer_models.items():
        print(f"  - {display_name}: {model_id}")

def test_feature_extraction_integration():
    """Test the integration with the existing feature extraction pipeline."""
    print("\n=== Testing Feature Extraction Integration ===")
    
    # Test the dispatcher logic using the actual utility function
    test_model_names = [
        ("Color Features", False),
        ("yolov8n-cls.pt", False),
        ("facebook/dinov2-small", True),
        ("microsoft/resnet-50", True),
        ("imageomics/bioclip", True),
        ("adriansaavedraa/MariCLIP", True),
        ("some-org/some-model", True),  # Any model with '/' should be detected as transformer
        ("", False),  # Empty string
        (None, False),  # None value
    ]
    
    for model_name, expected_result in test_model_names:
        # Use the actual utility function from the implementation
        actual_result = is_transformer_model(model_name)
        
        if actual_result == expected_result:
            print(f"✅ {model_name}: Correctly identified as {'transformer' if actual_result else 'non-transformer'}")
        else:
            print(f"❌ {model_name}: Expected {'transformer' if expected_result else 'non-transformer'}, got {'transformer' if actual_result else 'non-transformer'}")

if __name__ == "__main__":
    print("=" * 50)
    print("CoralNet-Toolbox Transformer Integration Test")
    print("=" * 50)
    
    test_model_list()
    test_feature_extraction_integration()
    test_transformer_pipeline()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)