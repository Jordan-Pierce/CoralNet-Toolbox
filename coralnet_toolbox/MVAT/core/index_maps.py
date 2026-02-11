import time

import numpy as np
import torch


def generate_mock_data_torch(n_points=1_000_000, width=3840, height=2160, device='cuda'):
    """
    Generates data directly on the GPU to simulate a loaded dataset.
    """
    # FIX: Cast device to string for printing
    print(f"--- Generating {n_points} points on {str(device).upper()} ---")
    
    # 1. Random Points (N, 3)
    points_cam = torch.zeros((n_points, 3), device=device, dtype=torch.float32)
    points_cam[:, 0].uniform_(-20, 20) # X
    points_cam[:, 1].uniform_(-10, 10) # Y
    points_cam[:, 2].uniform_(1, 50)   # Z (Depth)
    
    # Unique IDs
    point_ids = torch.arange(n_points, device=device, dtype=torch.int32)

    # 2. Intrinsics
    fx, fy = 3000.0, 3000.0
    cx, cy = width / 2.0, height / 2.0
    
    # We store intrinsics as scalar tensors or a small tensor for math
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
    
    return points_cam, point_ids, K, width, height


def run_torch_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == 'cuda'
    N_POINTS = 24_000_000
    W, H = 3840, 2160
    
    # Prepare data on GPU
    points_c, p_ids, K, w, h = generate_mock_data_torch(N_POINTS, W, H, device)

    # Warmup
    if is_cuda:
        print("\n--- Warming up CUDA kernels... ---")
        z_buffer = torch.full((H * W,), float('inf'), device=device, dtype=torch.float32)
        torch.cuda.synchronize()
    else:
        print("\n--- Warming up CPU... ---")
        z_buffer = torch.full((H * W,), float('inf'), device=device, dtype=torch.float32)
    
    print(f"\n--- Starting Benchmark (Image: {W}x{H}) ---")
    
    # START TIMER
    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    # --- STEP 1: PROJECTION ---
    x, y, z = points_c[:, 0], points_c[:, 1], points_c[:, 2]
    
    u = (K[0,0] * x / z) + K[0,2]
    v = (K[1,1] * y / z) + K[1,2]
    
    # --- STEP 2: BOUNDS CHECK ---
    u_idx = u.round().long()
    v_idx = v.round().long()
    
    valid_mask = (u_idx >= 0) & (u_idx < w) & \
                 (v_idx >= 0) & (v_idx < h) & \
                 (z > 0)
                 
    valid_u = u_idx[valid_mask]
    valid_v = v_idx[valid_mask]
    valid_z = z[valid_mask]
    valid_ids = p_ids[valid_mask]
    
    # --- STEP 3: GPU Z-BUFFERING (Scatter Reduce) ---
    flat_pixel_indices = valid_v * w + valid_u
    
    # Reset Z-buffer for the real run
    z_buffer.fill_(float('inf'))
    
    # A. Find the MINIMUM depth at every pixel
    z_buffer.scatter_reduce_(0, flat_pixel_indices, valid_z, reduce="amin", include_self=True)
    
    # B. Match points to the Z-Buffer
    min_z_at_points = z_buffer[flat_pixel_indices]
    
    # Epsilon check for float equality
    is_closest_mask = torch.abs(valid_z - min_z_at_points) < 1e-4
    
    # C. Write Final IDs
    final_indices = flat_pixel_indices[is_closest_mask]
    final_ids = valid_ids[is_closest_mask]
    
    # Create Final Index Map (-1 for empty)
    index_map = torch.full((h * w,), -1, device=device, dtype=torch.int32)
    index_map[final_indices] = final_ids
    
    # STOP TIMER
    if is_cuda:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_sec = elapsed_time_ms / 1000.0
    else:
        end_time = time.time()
        elapsed_time_sec = end_time - start_time
    
    # REPORT
    print(f"--- Benchmark Complete ---")
    print(f"Total Time: {elapsed_time_sec:.4f} seconds")
    print(f"Points Processed: {N_POINTS}")
    print(f"Points Visible on Screen: {final_ids.shape[0]}")
    if elapsed_time_sec > 0:
        print(f"FPS Equivalent: {1.0/elapsed_time_sec:.2f} Hz")
    else:
        print(f"FPS Equivalent: > 1000 Hz")
        

def generate_mock_data(n_points=1_000_000, width=3840, height=2160):
    """
    Generates random 3D points and camera parameters.
    Points are generated inside the camera frustum to ensure the test is realistic
    (processing points that actually hit pixels is heavier than discarding them).
    """
    print(f"--- Generatings {n_points} random points ---")
    
    # 1. Random Points in Camera Coordinates
    # Spread X and Y to fill a 4K frame roughly
    # Z (depth) from 1m to 50m
    points_cam = np.zeros((n_points, 3), dtype=np.float32)
    points_cam[:, 0] = np.random.uniform(-20, 20, n_points) # X
    points_cam[:, 1] = np.random.uniform(-10, 10, n_points) # Y
    points_cam[:, 2] = np.random.uniform(1, 50, n_points)   # Z (Depth)
    
    # IDs corresponding to the points (0 to N-1)
    point_ids = np.arange(n_points, dtype=np.int32)

    # 2. Simple Intrinsics (Pinhole) for 4K
    fx = fy = 3000
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    return points_cam, point_ids, K, width, height


def run_projection_benchmark():
    # SETUP
    N_POINTS = 10_000_000  # Size of your dense cloud
    W, H = 3840, 2160     # 4K Resolution
    
    points_c, p_ids, K, w, h = generate_mock_data(N_POINTS, W, H)

    print(f"\n--- Starting Benchmark (Image: {W}x{H}) ---")
    start_time = time.time()

    # STEP 1: PROJECTION (The Math)
    # Extract Depth
    z = points_c[:, 2]
    
    # Project to Pixel Coordinates (Vectorized)
    # u = (fx * x / z) + cx
    u = (K[0, 0] * points_c[:, 0] / z) + K[0, 2]
    v = (K[1, 1] * points_c[:, 1] / z) + K[1, 2]

    # STEP 2: FILTERING (Bounds Check)
    # Round to integers
    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)
    
    # Find indices that are actually inside the image
    valid_mask = (u_idx >= 0) & (u_idx < w) & \
                 (v_idx >= 0) & (v_idx < h) & \
                 (z > 0)
                 
    # Filter arrays to keep only valid points
    u_valid = u_idx[valid_mask]
    v_valid = v_idx[valid_mask]
    z_valid = z[valid_mask]
    id_valid = p_ids[valid_mask]

    # STEP 3: Z-BUFFERING (The "Sort" Trick)
    # To handle occlusion without a slow loop, we sort by depth (Descending).
    # When we assign to the grid, the LAST value written "wins".
    # By sorting High-Z to Low-Z, the closest points (Low-Z) are written last.
    
    sort_order = np.argsort(z_valid)[::-1] # Descending order
    
    # Reorder our valid points
    u_sorted = u_valid[sort_order]
    v_sorted = v_valid[sort_order]
    id_sorted = id_valid[sort_order]
    
    # Create the Index Map (-1 = empty)
    index_map = np.full((h, w), -1, dtype=np.int32)
    
    # NumPy Magic: This assigns all points at once.
    # Because of our sort, the closest points overwrite the further ones.
    index_map[v_sorted, u_sorted] = id_sorted

    end_time = time.time()
    
    # REPORT
    duration = end_time - start_time
    print(f"--- Benchmark Complete ---")
    print(f"Total Time: {duration:.4f} seconds")
    print(f"Points Processed: {N_POINTS}")
    print(f"Points Visible on Screen: {len(id_sorted)}")
    print(f"FPS Equivalent: {1.0/duration:.2f} Hz")
    

if __name__ == "__main__":
    run_torch_benchmark()