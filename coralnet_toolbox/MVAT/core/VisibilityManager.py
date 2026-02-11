"""
Visibility Manager for MVAT

A stateless engine that determines which 3D points are visible to a specific camera.
It generates 'Index Maps' (Pixel -> PointID) and sets of visible Point IDs.

Hardware Acceleration:
- Uses PyTorch (CUDA or CPU) as the primary compute engine.
- Falls back to NumPy if PyTorch is unavailable.
"""

import time
import warnings
import numpy as np

# Try importing torch, but handle the case where it's not installed
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VisibilityManager:
    """
    Stateless engine for computing point visibility and generating index maps.
    """

    @classmethod
    def compute_visibility(cls, 
                           points_world: np.ndarray, 
                           K: np.ndarray, 
                           R: np.ndarray, 
                           t: np.ndarray, 
                           width: int, 
                           height: int,
                           point_ids: np.ndarray = None) -> dict:
        """
        Compute visibility for a cloud of points given camera parameters.

        Args:
            points_world (np.ndarray): (N, 3) array of 3D points in World Coordinates.
            K (np.ndarray): (3, 3) Intrinsic matrix.
            R (np.ndarray): (3, 3) Rotation matrix (World -> Camera).
            t (np.ndarray): (3,) Translation vector (World -> Camera).
            width (int): Image width.
            height (int): Image height.
            point_ids (np.ndarray, optional): (N,) array of global IDs. 
                                              If None, indices 0..N-1 are used.

        Returns:
            dict: {
                'index_map': (H, W) int32 array. Pixel value is Point ID or -1.
                'visible_indices': (M,) int32 array. Unique IDs of visible points.
            }
        """
        start_time = time.time()
        
        # Default point IDs if not provided
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)

        # 1. Prefer PyTorch (CUDA or CPU)
        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            result = cls._compute_torch(points_world, point_ids, K, R, t, width, height, device)
        else:
            # 2. Fallback to NumPy if Torch is missing
            device = 'numpy'
            result = cls._compute_numpy(points_world, point_ids, K, R, t, width, height)
        
        compute_time = time.time() - start_time
        visible_count = len(result['visible_indices'])
        print(f"⏱️ Computed index map ({height}x{width}) for {len(points_world):,} points using {device.upper()}: "
              f"{visible_count:,} visible points in {compute_time:.3f}s")
        
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # PyTorch Implementation (CUDA & CPU)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _compute_torch(points_np, ids_np, K_np, R_np, t_np, width, height, device):
        """
        PyTorch-based visibility computation.
        Uses scatter_reduce_ for efficient Z-buffering.
        Works on 'cuda' (fastest) and 'cpu' (via PyTorch tensors).
        """
        # 1. Transfer Data to Device (GPU or CPU)
        # We assume input is float32 for geometry, int32 for IDs
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids = torch.as_tensor(ids_np, dtype=torch.int32, device=device)
        
        K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
        R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_np, dtype=torch.float32, device=device)

        # 2. Transform World -> Camera
        # X_cam = R * X_world + t
        # Shape logic: (N, 3) @ (3, 3).T + (3,)
        points_cam = points @ R.T + t

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # 3. Project to Image Plane (Vectorized)
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # Note: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        u = (K[0, 0] * x_cam / z_cam) + K[0, 2]
        v = (K[1, 1] * y_cam / z_cam) + K[1, 2]

        # 4. Bounds & Depth Check
        u_idx = u.round().long()
        v_idx = v.round().long()

        valid_mask = (u_idx >= 0) & (u_idx < width) & \
                     (v_idx >= 0) & (v_idx < height) & \
                     (z_cam > 0)

        # Filter to keep only potentially visible points
        valid_u = u_idx[valid_mask]
        valid_v = v_idx[valid_mask]
        valid_z = z_cam[valid_mask]
        valid_ids = p_ids[valid_mask]

        if valid_ids.numel() == 0:
            # Nothing visible
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32)
            }

        # 5. Z-Buffering (Scatter Reduce)
        # Flatten indices: idx = y * width + x
        flat_indices = valid_v * width + valid_u

        # Initialize Z-Buffer with Infinity
        z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)

        # A. Find minimum depth at every pixel
        # Note: scatter_reduce_ is available in PyTorch 1.12+
        try:
            z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce="amin", include_self=True)
        except AttributeError:
            # Fallback for older torch versions lacking scatter_reduce_
            warnings.warn("PyTorch version too old for scatter_reduce_. Falling back to NumPy implementation.")
            return VisibilityManager._compute_numpy(points_np, ids_np, K_np, R_np, t_np, width, height)

        # B. Identify which points 'won' the Z-buffer test
        # Get the min_z recorded at the projected location for each point
        min_z_at_pixel = z_buffer[flat_indices]
        
        # Check if point's depth matches the min depth (with epsilon for float precision)
        is_closest = torch.abs(valid_z - min_z_at_pixel) < 1e-4

        # C. Filter final winners
        final_pixel_indices = flat_indices[is_closest]
        final_ids = valid_ids[is_closest]

        # 6. Construct Outputs
        # Create blank index map
        index_map_tensor = torch.full((height * width,), -1, device=device, dtype=torch.int32)
        
        # Assign IDs to map
        # Note: If multiple points are within epsilon at the same pixel, last one writes.
        index_map_tensor[final_pixel_indices] = final_ids
        
        # Reshape to 2D
        index_map_2d = index_map_tensor.view(height, width)

        # Extract unique visible IDs
        visible_indices = torch.unique(final_ids, sorted=True)

        return {
            'index_map': index_map_2d.cpu().numpy(),
            'visible_indices': visible_indices.cpu().numpy()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # NumPy Implementation (Legacy / Fallback)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _compute_numpy(points, ids, K, R, t, width, height):
        """
        CPU-based visibility computation (Legacy / Fallback).
        Uses 'Sort by Depth' optimization to handle occlusion efficiently without loops.
        """
        # 1. Transform World -> Camera
        # X_cam = R * X_world + t
        points_cam = points @ R.T + t

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # 2. Project to Image Plane
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (K[0, 0] * x_cam / z_cam) + K[0, 2]
            v = (K[1, 1] * y_cam / z_cam) + K[1, 2]

        # 3. Bounds Check & Integer Cast
        u_idx = np.rint(u).astype(np.int32)
        v_idx = np.rint(v).astype(np.int32)

        valid_mask = (u_idx >= 0) & (u_idx < width) & \
                     (v_idx >= 0) & (v_idx < height) & \
                     (z_cam > 0)

        # Filter invalid points
        u_valid = u_idx[valid_mask]
        v_valid = v_idx[valid_mask]
        z_valid = z_cam[valid_mask]
        id_valid = ids[valid_mask]

        if len(id_valid) == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32)
            }

        # 4. Z-Buffering (The Sorting Trick)
        # To handle occlusion efficiently in NumPy, we sort points by depth (Descending).
        # When we perform array assignment index_map[y, x] = id, the LAST value overwrites previous ones.
        # By sorting High -> Low, the Low-Z (closest) points are written last, correctly "winning" the pixel.
        sort_order = np.argsort(z_valid)[::-1]  # Descending order

        u_sorted = u_valid[sort_order]
        v_sorted = v_valid[sort_order]
        id_sorted = id_valid[sort_order]

        # 5. Create Outputs
        # Initialize map with -1
        index_map = np.full((height, width), -1, dtype=np.int32)

        # Bulk assignment handles the "last write wins" logic
        index_map[v_sorted, u_sorted] = id_sorted

        # Extract unique IDs from the final map
        visible_indices = np.unique(index_map[index_map != -1])

        return {
            'index_map': index_map,
            'visible_indices': visible_indices
        }