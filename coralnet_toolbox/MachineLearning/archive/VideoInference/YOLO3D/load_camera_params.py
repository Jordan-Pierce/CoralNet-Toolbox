import os
import json
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def load_camera_params(params_file):
    """
    Load camera parameters from a JSON file.
    
    Args:
        params_file (str): Path to the JSON file containing camera parameters
        
    Returns:
        dict: Dictionary containing camera parameters
    """
    if not os.path.exists(params_file):
        print(f"Warning: Camera parameters file {params_file} not found. Using default parameters.")
        return None
    
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Convert lists to numpy arrays
        params['camera_matrix'] = np.array(params['camera_matrix'])
        params['dist_coeffs'] = np.array(params['dist_coeffs'])
        params['projection_matrix'] = np.array(params['projection_matrix'])
        
        print(f"Loaded camera parameters from {params_file}")
        print(f"Camera matrix:\n{params['camera_matrix']}")
        print(f"Projection matrix:\n{params['projection_matrix']}")
        
        return params
    
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return None


def create_projection_matrix(camera_matrix, R=None, t=None):
    """
    Create a projection matrix from camera intrinsic and extrinsic parameters.
    
    Args:
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
        R (numpy.ndarray): Rotation matrix (3x3)
        t (numpy.ndarray): Translation vector (3x1)
        
    Returns:
        numpy.ndarray: Projection matrix (3x4)
    """
    if R is None:
        R = np.eye(3)
    
    if t is None:
        t = np.zeros((3, 1))
    
    # Combine rotation and translation
    RT = np.hstack((R, t))
    
    # Create projection matrix
    projection_matrix = camera_matrix @ RT
    
    return projection_matrix


def apply_camera_params_to_estimator(bbox3d_estimator, params):
    """
    Apply camera parameters to a 3D bounding box estimator.
    
    Args:
        bbox3d_estimator: BBox3DEstimator instance
        params (dict): Dictionary containing camera parameters
        
    Returns:
        bbox3d_estimator: Updated BBox3DEstimator instance
    """
    if params is None:
        print("Warning: No camera parameters provided. Using default parameters.")
        return bbox3d_estimator
    
    # Update camera matrix
    if 'camera_matrix' in params:
        bbox3d_estimator.K = params['camera_matrix']
    
    # Update projection matrix
    if 'projection_matrix' in params:
        bbox3d_estimator.P = params['projection_matrix']
    
    print("Applied camera parameters to 3D bounding box estimator")
    
    return bbox3d_estimator


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """Example usage of the camera parameter functions."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input file
    params_file = "camera_params.json"  # Path to camera parameters JSON file
    
    # Camera position (for example purposes)
    camera_height = 1.65  # Camera height above ground in meters
    # ===============================================
    
    # Load camera parameters
    params = load_camera_params(params_file)
    
    if params:
        print("\nCamera Parameters:")
        print(f"Image dimensions: {params['image_width']}x{params['image_height']}")
        print(f"Reprojection error: {params['reprojection_error']}")
        
        # Example of creating a projection matrix with different extrinsic parameters
        print(f"\nExample: Creating a projection matrix with camera raised {camera_height}m above ground")
        R = np.eye(3)
        t = np.array([[0], [camera_height], [0]])  # Camera above ground
        
        projection_matrix = create_projection_matrix(params['camera_matrix'], R, t)
        print(f"New projection matrix:\n{projection_matrix}")


if __name__ == "__main__":
    main() 