"""
Visibility Manager for MVAT

A stateless engine that determines which 3D elements are visible to a specific camera.
It generates 'Index Maps' (Pixel -> ElementID) and sets of visible ElementIDs.

Supports heterogeneous scene products:
- Point Clouds: Scatter-reduce Z-buffering (point IDs)
- Meshes: Ray-casting/Rasterization (face IDs) [Placeholder]
- DEMs: Direct projection (cell IDs) [Placeholder]

Hardware Acceleration:
- Uses PyTorch (CUDA or CPU) as the primary compute engine.
- Falls back to NumPy if PyTorch is unavailable.
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from coralnet_toolbox.MVAT.core.SceneContext import SceneContext
    from coralnet_toolbox.MVAT.core.SceneProduct import AbstractSceneProduct

# Try importing torch, but handle the case where it's not installed
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
    
# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VisibilityManager:
    """
    Stateless engine for computing element visibility and generating index maps.
    
    Strategy Pattern Implementation:
    - Point Cloud Target: Scatter-reduce Z-buffering (existing algorithm)
    - Mesh Target: Ray-casting/rasterization (placeholder - falls back to point sampling)
    - DEM Target: Affine projection (for orthographic cameras)
    
    Results include 'element_type' metadata ('point', 'face', 'cell') for downstream
    annotation engines to properly interpret index map values.
    """

    @classmethod
    def compute_visibility_from_scene(cls,
                                      scene_context: 'SceneContext',
                                      K: np.ndarray,
                                      R: np.ndarray,
                                      t: np.ndarray,
                                      width: int,
                                      height: int,
                                      compute_depth_map: bool = True) -> dict:
        """
        Strategy dispatcher: compute visibility based on scene context.
        
        Queries the scene for the primary target and dispatches to the 
        appropriate visibility algorithm based on the target's element type.
        
        Args:
            scene_context: SceneContext containing loaded products.
            K: (3, 3) Intrinsic matrix.
            R: (3, 3) Rotation matrix (World -> Camera).
            t: (3,) Translation vector (World -> Camera).
            width: Image width in pixels.
            height: Image height in pixels.
            compute_depth_map: Whether to generate depth map.
            
        Returns:
            dict: {
                'index_map': (H, W) int32 array. Pixel value is Element ID or -1.
                'visible_indices': (M,) int32 array. Unique IDs of visible elements.
                'depth_map': (H, W) float32 array. Camera-space depth per pixel (optional).
                'element_type': str. One of 'point', 'face', 'cell'.
            }
        """
        primary_target = scene_context.get_primary_target()
        
        if primary_target is None:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'element_type': 'point'
            }
        
        element_type = primary_target.get_element_type()
        
        # Strategy dispatch based on element type
        if element_type == 'point':
            # Strategy A: Point Cloud - existing scatter-reduce algorithm
            from coralnet_toolbox.MVAT.core.Model import PointCloudProduct
            if isinstance(primary_target, PointCloudProduct):
                points = primary_target.get_points_array()
                if points is not None:
                    result = cls.compute_visibility(points, K, R, t, width, height, 
                                                    compute_depth_map=compute_depth_map)
                    result['element_type'] = 'point'
                    return result
                    
        elif element_type == 'face':
            # Strategy B: Mesh - ray-casting/rasterization
            result = cls._compute_mesh_visibility(primary_target, K, R, t, width, height, 
                                                  compute_depth_map=compute_depth_map)
            result['element_type'] = 'face'
            return result
            
        elif element_type == 'cell':
            # Strategy C: DEM - affine projection (for orthographic cameras)
            # Note: For DEM, orthographic projection is handled separately
            # This is a fallback for perspective cameras looking at DEM
            result = cls._compute_dem_visibility(primary_target, K, R, t, width, height)
            result['element_type'] = 'cell'
            return result
        
        # Fallback: empty result
        return {
            'index_map': np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'element_type': element_type
        }

    @classmethod
    def _compute_mesh_visibility(cls,
                                 mesh_product: 'AbstractSceneProduct',
                                 K: np.ndarray,
                                 R: np.ndarray,
                                 t: np.ndarray,
                                 width: int,
                                 height: int,
                                 compute_depth_map: bool = True) -> dict:
        """
        Strategy B: Compute visibility for mesh products.
        Attempts Open3D raycasting first, falls back to VTK rasterization,
        and finally falls back to face-center point sampling.
        """
        try:
            # First Choice: Open3D (Thread-safe, fast, no OpenGL context required)
            import open3d
            return cls._compute_mesh_visibility_open3d(
                mesh_product, K, R, t, width, height, compute_depth_map
            )
        except ImportError:
            print("⚠️ Open3D not found. Falling back to VTK mesh rasterization (Requires Main Thread).")
            try:
                # Second Choice: VTK (Requires GUI thread OpenGL context)
                return cls._compute_mesh_visibility_vtk(
                    mesh_product, K, R, t, width, height, compute_depth_map
                )
            except Exception as e:
                print(f"⚠️ VTK mesh rasterization failed: {e}, falling back to face-center sampling")
                # Last Resort: Point Sampling
                return cls._compute_mesh_visibility_fallback(
                    mesh_product, K, R, t, width, height, compute_depth_map
                )
                
    @classmethod
    def _build_subset_bvh(cls, mesh_product, camera_params_list):
        import open3d as o3d
        import time
        import numpy as np
        import torch
        
        start_time = time.time()
        mesh_product.prepare_geometry() 
        
        # Grab tensors and the designated device from the model
        centers = mesh_product._cached_face_centers_pt
        centers_sq_norm = mesh_product._cached_centers_sq_norm_pt
        triangles = mesh_product._cached_triangles_pt
        device = mesh_product.device
        
        # Initialize the global mask directly on the target device
        global_mask = torch.zeros(len(centers), dtype=torch.bool, device=device)
        
        ANGLE_THRESHOLD = 0.6
        angle_sq_threshold = ANGLE_THRESHOLD ** 2

        for K, R, t, w, h in camera_params_list:
            cam_pos = -R.T @ t
            cam_dir = R.T @ np.array([0, 0, 1]) 
            cam_dir = cam_dir / np.linalg.norm(cam_dir)
            
            # Pack the small camera matrix and push to the device
            cam_matrix = np.column_stack((cam_dir, -2.0 * cam_pos)).astype(np.float32)
            cam_matrix_pt = torch.tensor(cam_matrix, device=device)
            
            # Massive Matrix Multiplication (GPU or CPU!)
            proj = torch.matmul(centers, cam_matrix_pt)
            
            # Calculate standard scalars in python to avoid tensor device mismatches
            cam_pos_dir_dot = float(np.dot(cam_pos, cam_dir))
            cam_pos_sq_norm = float(np.dot(cam_pos, cam_pos))
            
            dot_prods = proj[:, 0] - cam_pos_dir_dot
            sq_dists = centers_sq_norm + cam_pos_sq_norm + proj[:, 1]
            
            front_mask = dot_prods > 0
            valid_mask = front_mask & (sq_dists > 1e-6)
            camera_mask = valid_mask & ((dot_prods**2) > (sq_dists * angle_sq_threshold))
            
            global_mask |= camera_mask

        # Pull ONLY the surviving tiny subset back to the CPU for Open3D
        subset_triangles = triangles[global_mask].cpu().numpy().astype(np.uint32)
        
        subset_cell_ids = None
        if getattr(mesh_product, '_original_cell_ids_pt', None) is not None:
            subset_cell_ids = mesh_product._original_cell_ids_pt[global_mask].cpu().numpy()
            
        cull_time = time.time() - start_time
        print(f"✂️ {device.upper()} Frustum Cull: Kept {len(subset_triangles):,} faces in {cull_time:.3f}s")
        
        # Build Open3D Scene
        scene = o3d.t.geometry.RaycastingScene()
        
        if len(subset_triangles) > 0:
            build_start = time.time()
            v_tensor = o3d.core.Tensor(mesh_product._cached_vertices)
            t_tensor = o3d.core.Tensor(subset_triangles)
            scene.add_triangles(v_tensor, t_tensor)
            
            dummy_ray = o3d.core.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=o3d.core.Dtype.Float32)
            scene.cast_rays(dummy_ray)
            print(f"🎯 Built Sub-BVH in {time.time() - build_start:.3f}s")
            
        return scene, subset_cell_ids, len(subset_triangles)
            
    @classmethod
    def compute_batch_mesh_visibility_open3d(cls, 
                                             mesh_product, 
                                             camera_params_list, 
                                             compute_depth_maps=True) -> list:
        """
        Batched Open3D raycasting with Dynamic Frustum Culling and Downsampling.
        """
        import open3d as o3d
        import time
        import cv2
        
        start_time = time.time()
        
        # 1. Dynamically build the culled BVH
        scene, original_cell_ids, num_faces = cls._build_subset_bvh(mesh_product, camera_params_list)

        results = []
        
        # If the culler removed everything, return empty maps
        if num_faces == 0:
            return [{
                'index_map': np.full((h, w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((h, w), np.nan, dtype=np.float32) if compute_depth_maps else None
            } for _, _, _, w, h in camera_params_list]

        # 2. Fast Downsampled Raycasting
        SCALE_FACTOR = 0.25  # 1/4 resolution raycasting
        
        for K, R, t, width, height in camera_params_list:
            E = np.eye(4, dtype=np.float64)
            E[:3, :3] = R
            E[:3, 3] = t
            
            small_w = int(width * SCALE_FACTOR)
            small_h = int(height * SCALE_FACTOR)
            
            K_small = K.copy()
            K_small[0, :3] *= SCALE_FACTOR
            K_small[1, :3] *= SCALE_FACTOR

            K_tensor = o3d.core.Tensor(K_small, dtype=o3d.core.Dtype.Float64)
            E_tensor = o3d.core.Tensor(E, dtype=o3d.core.Dtype.Float64)

            rays = scene.create_rays_pinhole(
                intrinsic_matrix=K_tensor,
                extrinsic_matrix=E_tensor,
                width_px=small_w,
                height_px=small_h
            )
            ans = scene.cast_rays(rays)

            # Extract Maps
            index_map_raw = ans['primitive_ids'].numpy()
            invalid_mask = (index_map_raw == 4294967295)
            
            index_map_small = index_map_raw.astype(np.int64)
            index_map_small[invalid_mask] = -1
            index_map_small = index_map_small.astype(np.int32)

            # Re-map triangle IDs
            if original_cell_ids is not None:
                valid_mask = (index_map_small != -1)
                index_map_small[valid_mask] = original_cell_ids[index_map_small[valid_mask]]

            # Upsample
            index_map = cv2.resize(index_map_small, (width, height), interpolation=cv2.INTER_NEAREST)
            visible_indices = np.unique(index_map_small[index_map_small != -1]).astype(np.int32)

            if compute_depth_maps:
                depth_map_small = ans['t_hit'].numpy().astype(np.float32)
                depth_map_small[invalid_mask] = np.nan
                depth_map = cv2.resize(depth_map_small, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                depth_map = None

            results.append({
                'index_map': index_map,
                'visible_indices': visible_indices,
                'depth_map': depth_map
            })
            
        print(f"✅ Full Open3D cycle for {len(camera_params_list)} cameras completed in {time.time() - start_time:.3f}s")
        return results

    @classmethod
    def _compute_mesh_visibility_vtk(cls,
                                     mesh_product: 'AbstractSceneProduct',
                                     K: np.ndarray,
                                     R: np.ndarray,
                                     t: np.ndarray,
                                     width: int,
                                     height: int,
                                     compute_depth_map: bool = True) -> dict:
        """
        VTK-based mesh rasterization for pixel-perfect face ID and depth maps.
        
        Workflow:
        1. Create off-screen plotter matching image dimensions
        2. Configure VTK camera from K, R, t (OpenCV conventions)
        3. Assign face IDs as cell scalars with RGB encoding
        4. Render and decode face IDs from screenshot
        5. Extract depth buffer and convert to camera-space depth
        """
        import pyvista as pv
        
        mesh = mesh_product.get_mesh()
        n_faces = mesh.n_cells
        
        if n_faces == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None
            }
        
        print(f"🎨 Mesh visibility: VTK rasterization for {n_faces:,} faces at {width}x{height}")
        
        # --- 1. Encode face IDs as RGB colors ---
        # Use 24-bit encoding: R + G*256 + B*65536 = face_id
        # This supports up to 16.7M faces
        face_ids = np.arange(n_faces, dtype=np.int32)
        
        # Encode face_id into RGB (shifted by 1 so face_id=0 maps to RGB(1,0,0))
        # We reserve RGB(0,0,0) for background
        encoded_ids = face_ids + 1
        r = (encoded_ids % 256).astype(np.uint8)
        g = ((encoded_ids // 256) % 256).astype(np.uint8)
        b = ((encoded_ids // 65536) % 256).astype(np.uint8)
        
        # Create RGB array for cell data
        rgb_colors = np.column_stack([r, g, b])
        
        # Clone mesh and assign face ID colors as cell data
        mesh_with_ids = mesh.copy()
        mesh_with_ids.cell_data['FaceID_RGB'] = rgb_colors
        
        # --- 2. Create off-screen plotter ---
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background('black')  # Background = RGB(0,0,0) = "no face"
        
        # Add mesh with RGB scalars, no lighting/interpolation
        plotter.add_mesh(
            mesh_with_ids,
            scalars='FaceID_RGB',
            rgb=True,
            lighting=False,
            interpolate_before_map=False,
            show_edges=False,
            style='surface'
        )
        
        # --- 3. Configure VTK camera from K, R, t ---
        cls._configure_vtk_camera(plotter, K, R, t, width, height, mesh.bounds)
        
        # --- 4. Render and extract face IDs ---
        plotter.render()
        
        # Get screenshot (RGB image)
        screenshot = plotter.screenshot(return_img=True)  # Shape: (H, W, 3) or (H, W, 4)
        
        if screenshot.shape[2] == 4:
            screenshot = screenshot[:, :, :3]  # Drop alpha channel
        
        # Decode RGB back to face IDs
        # face_id = R + G*256 + B*65536 - 1  (subtract 1 because we added 1 during encoding)
        decoded = (screenshot[:, :, 0].astype(np.int32) +
                   screenshot[:, :, 1].astype(np.int32) * 256 +
                   screenshot[:, :, 2].astype(np.int32) * 65536)
        
        # Background (0,0,0) decodes to 0, subtract 1 to get -1 for no-face
        index_map = decoded - 1
        index_map = index_map.astype(np.int32)
        
        # --- 5. Extract depth buffer ---
        depth_map = None
        if compute_depth_map:
            try:
                # Get VTK depth buffer
                # PyVista's get_image_depth() returns actual Z-coordinates in camera space
                # VTK convention: camera looks down -Z, so visible objects have negative Z
                # OpenCV convention: camera looks down +Z, depth is positive
                vtk_depth = plotter.get_image_depth(fill_value=np.nan)
                
                # Negate to convert from VTK (-Z forward) to OpenCV (+Z forward) convention
                depth_map = -vtk_depth.astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ Failed to extract depth buffer: {e}")
                depth_map = np.full((height, width), np.nan, dtype=np.float32)
        
        # --- 6. Extract visible face IDs ---
        visible_indices = np.unique(index_map[index_map >= 0]).astype(np.int32)
        
        # Cleanup
        plotter.close()
        
        n_visible = len(visible_indices)
        coverage = np.sum(index_map >= 0) / (width * height) * 100
        print(f"✅ Mesh visibility: {n_visible:,} visible faces, {coverage:.1f}% pixel coverage")
        
        return {
            'index_map': index_map,
            'visible_indices': visible_indices,
            'depth_map': depth_map
        }

    @classmethod
    def _configure_vtk_camera(cls, plotter, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                              width: int, height: int, bounds: tuple) -> None:
        """
        Configure VTK camera from OpenCV-style intrinsics and extrinsics.
        
        OpenCV conventions:
        - Camera looks down +Z axis
        - +X is right, +Y is down
        - R, t transform world points to camera space: X_cam = R @ X_world + t
        
        VTK conventions:
        - Camera looks down -Z axis (toward focal point)
        - +X is right, +Y is up
        - Position/FocalPoint/ViewUp define camera pose
        
        Args:
            plotter: PyVista plotter instance
            K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
            width, height: Image dimensions
            bounds: Scene bounding box for clipping plane estimation
        """
        # --- Camera position in world coordinates ---
        # X_world = R^T @ (X_cam - t)
        # Camera origin in camera space is (0, 0, 0)
        # So camera position in world space is: -R^T @ t
        position = -R.T @ t
        
        # --- Focal point: a point along the camera's viewing direction ---
        # In camera space, the camera looks toward +Z (OpenCV convention)
        # So a point at (0, 0, 1) in camera space, transformed to world space:
        forward_cam = np.array([0.0, 0.0, 1.0])
        forward_world = R.T @ forward_cam
        focal_point = position + forward_world
        
        # --- View up vector ---
        # In OpenCV, +Y is down in the image
        # In VTK, +Y is up
        # So view_up in world space is -R^T @ (0, 1, 0)
        up_cam = np.array([0.0, -1.0, 0.0])  # Flip Y for VTK
        view_up = R.T @ up_cam
        
        # --- Set camera pose ---
        camera = plotter.camera
        camera.position = position.tolist()
        camera.focal_point = focal_point.tolist()
        camera.up = view_up.tolist()
        
        # --- Compute view angle from intrinsics ---
        # For a pinhole camera: tan(view_angle/2) = (height/2) / fy
        # VTK uses vertical view angle in degrees
        fy = K[1, 1]
        view_angle_rad = 2.0 * np.arctan(height / (2.0 * fy))
        view_angle_deg = np.degrees(view_angle_rad)
        camera.view_angle = view_angle_deg
        
        # --- Set clipping range based on scene bounds ---
        # Compute distance from camera to scene center
        scene_center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
        scene_radius = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ]) / 2
        
        dist_to_center = np.linalg.norm(scene_center - position)
        
        # Set conservative clipping range
        near_clip = max(0.01, dist_to_center - scene_radius * 2)
        far_clip = dist_to_center + scene_radius * 2
        camera.clipping_range = (near_clip, far_clip)
        
        # --- Handle principal point offset ---
        # If cx, cy are not at image center, we need window center offset
        # VTK uses normalized window center: (0, 0) = center, (-1, -1) = bottom-left
        cx, cy = K[0, 2], K[1, 2]
        fx = K[0, 0]
        
        # Offset from image center in pixels
        dx = cx - width / 2
        dy = cy - height / 2
        
        # Convert to normalized coordinates (relative to half-width/height)
        # VTK window_center is fraction of half-window
        if abs(dx) > 1 or abs(dy) > 1:
            wx = -2.0 * dx / width  # Negate because VTK uses opposite sign
            wy = 2.0 * dy / height  # Y is already flipped
            camera.SetWindowCenter(wx, wy)

    @classmethod
    def _compute_mesh_visibility_fallback(cls,
                                          mesh_product: 'AbstractSceneProduct',
                                          K: np.ndarray,
                                          R: np.ndarray,
                                          t: np.ndarray,
                                          width: int,
                                          height: int,
                                          compute_depth_map: bool = True) -> dict:
        """
        Fallback: Compute mesh visibility using face-center point sampling.
        
        Used when VTK rasterization fails. Less accurate (sparse) but reliable.
        """
        print("⚠️ Mesh visibility: Using face-center sampling (fallback)")
        
        try:
            face_centers = mesh_product.get_face_centers()
            face_ids = np.arange(len(face_centers), dtype=np.int32)
            
            result = cls.compute_visibility(
                face_centers, K, R, t, width, height,
                point_ids=face_ids,
                compute_depth_map=compute_depth_map
            )
            return result
            
        except Exception as e:
            print(f"⚠️ Mesh visibility fallback failed: {e}")
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None
            }

    @classmethod
    def _compute_dem_visibility(cls,
                                dem_product: 'AbstractSceneProduct',
                                K: np.ndarray,
                                R: np.ndarray,
                                t: np.ndarray,
                                width: int,
                                height: int) -> dict:
        """
        Strategy C: Compute visibility for DEM products (perspective camera).
        
        PLACEHOLDER IMPLEMENTATION: Samples grid cell centers and uses point-based
        projection. For orthographic cameras viewing DEMs, use direct affine mapping
        via compute_orthographic_visibility() instead.
        
        Args:
            dem_product: DEMProduct instance.
            K, R, t: Camera parameters (perspective).
            width, height: Image dimensions.
            
        Returns:
            dict with 'index_map', 'visible_indices'.
        """
        print("⚠️ DEM visibility: Using cell-center sampling (placeholder for perspective cameras)")
        
        try:
            # Generate 3D points from DEM grid
            dem_height, dem_width = dem_product.elevation.shape
            rows, cols = np.mgrid[0:dem_height, 0:dem_width]
            
            # Convert pixel coords to world coords
            transform = dem_product.transform
            x_world = transform[0, 0] * cols + transform[0, 1] * rows + transform[0, 2]
            y_world = transform[1, 0] * cols + transform[1, 1] * rows + transform[1, 2]
            z_world = dem_product.elevation
            
            # Flatten to point array
            points = np.column_stack([
                x_world.flatten(),
                y_world.flatten(),
                z_world.flatten()
            ])
            
            # Cell IDs: row * width + col  
            cell_ids = np.arange(dem_height * dem_width, dtype=np.int32)
            
            # Filter out NaN elevations
            valid_mask = ~np.isnan(points[:, 2])
            points = points[valid_mask]
            cell_ids = cell_ids[valid_mask]
            
            # Use point-based visibility
            result = cls.compute_visibility(
                points, K, R, t, width, height,
                point_ids=cell_ids,
                compute_depth_map=False
            )
            return result
            
        except Exception as e:
            print(f"⚠️ DEM visibility computation failed: {e}")
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32)
            }

    @classmethod
    def compute_visibility(cls, 
                           points_world: np.ndarray, 
                           K: np.ndarray, 
                           R: np.ndarray, 
                           t: np.ndarray, 
                           width: int, 
                           height: int,
                           point_ids: np.ndarray = None,
                           compute_depth_map: bool = True) -> dict:
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
            result = cls._compute_torch(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width,
                                        height, 
                                        device, 
                                        compute_depth_map=compute_depth_map)
        else:
            # 2. Fallback to NumPy if Torch is missing
            device = 'numpy'
            result = cls._compute_numpy(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width, 
                                        height, 
                                        compute_depth_map=compute_depth_map)
        
        compute_time = time.time() - start_time
        visible_count = len(result['visible_indices'])
        print(f"⏱️ Computed index map ({height}x{width}) for {len(points_world):,} points using {device.upper()}: "
              f"{visible_count:,} visible points in {compute_time:.3f}s")
        
        return result

    @classmethod
    def compute_batch_visibility(cls, 
                                 points_world: np.ndarray, 
                                 camera_params_list: list,
                                 point_ids: np.ndarray = None,
                                 compute_depth_map: bool = True) -> list:
        """
        Compute visibility for multiple cameras in batch using GPU acceleration.
        
        Args:
            points_world (np.ndarray): (N, 3) array of 3D points in World Coordinates.
            camera_params_list (list): List of (K, R, t, width, height) tuples, where:
                - K (np.ndarray): (3, 3) Intrinsic matrix.
                - R (np.ndarray): (3, 3) Rotation matrix (World -> Camera).
                - t (np.ndarray): (3,) Translation vector (World -> Camera).
                - width (int): Image width.
                - height (int): Image height.
            point_ids (np.ndarray, optional): (N,) array of global IDs. 
                                              If None, indices 0..N-1 are used.

        Returns:
            list: List of dicts, one per camera, each containing:
                {
                    'index_map': (H, W) int32 array. Pixel value is Point ID or -1.
                    'visible_indices': (M,) int32 array. Unique IDs of visible points.
                }
        """
        start_time = time.time()
        
        # Default point IDs if not provided
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)

        if not HAS_TORCH:
            # Fallback to sequential NumPy computation
            results = []
            for K, R, t, width, height in camera_params_list:
                result = cls._compute_numpy(points_world, 
                                            point_ids,
                                            K, R, t, width, height, 
                                            compute_depth_map=compute_depth_map)
                results.append(result)
            compute_time = time.time() - start_time
            print(f"⏱️ Computed batch visibility for {len(camera_params_list)} cameras, "
                  f"{len(points_world):,} points using NUMPY: {compute_time:.3f}s")
            return results

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Transfer points to device once
        points = torch.as_tensor(points_world, dtype=torch.float32, device=device)
        p_ids = torch.as_tensor(point_ids, dtype=torch.int32, device=device)
        
        M = len(camera_params_list)
        
        # Extract and transfer camera parameters
        K_list = [torch.as_tensor(K, dtype=torch.float32, device=device) for K, R, t, w, h in camera_params_list]
        R_list = [torch.as_tensor(R, dtype=torch.float32, device=device) for K, R, t, w, h in camera_params_list]
        t_list = [torch.as_tensor(t, dtype=torch.float32, device=device) for K, R, t, w, h in camera_params_list]
        widths = [w for K, R, t, w, h in camera_params_list]
        heights = [h for K, R, t, w, h in camera_params_list]
        
        # Batch camera parameters
        R_batch = torch.stack(R_list)  # (M, 3, 3)
        t_batch = torch.stack(t_list)  # (M, 3)
        K_batch = torch.stack(K_list)  # (M, 3, 3)
        
        # Batch points for all cameras
        points_batch = points.unsqueeze(0).expand(M, -1, -1)  # (M, N, 3)
        
        # Batch transform: World -> Camera for all cameras
        # points_cam_batch = points_batch @ R_batch.transpose(1, 2) + t_batch.unsqueeze(1) (M, N, 3)
        points_cam_batch = torch.einsum('mni,mij->mnj', points_batch, R_batch.transpose(1, 2)) + t_batch.unsqueeze(1)  
        
        # Extract coordinates
        x_batch = points_cam_batch[:, :, 0]
        y_batch = points_cam_batch[:, :, 1]
        z_batch = points_cam_batch[:, :, 2]
        
        # Batch projection
        u_batch = (K_batch[:, 0, 0].unsqueeze(1) * x_batch / z_batch) + K_batch[:, 0, 2].unsqueeze(1)
        v_batch = (K_batch[:, 1, 1].unsqueeze(1) * y_batch / z_batch) + K_batch[:, 1, 2].unsqueeze(1)
        
        # Process each camera (since widths/heights may differ, z-buffering is per-camera)
        results = []
        for i in range(M):
            width = widths[i]
            height = heights[i]
            u = u_batch[i]
            v = v_batch[i]
            z = z_batch[i]
            ids = p_ids
            
            # Bounds and depth check
            u_idx = u.round().long()
            v_idx = v.round().long()
            valid_mask = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height) & (z > 0)
            
            valid_u = u_idx[valid_mask]
            valid_v = v_idx[valid_mask]
            valid_z = z[valid_mask]
            valid_ids = ids[valid_mask]
            
            if valid_ids.numel() == 0:
                results.append({
                    'index_map': np.full((height, width), -1, dtype=np.int32),
                    'visible_indices': np.array([], dtype=np.int32),
                    'depth_map': np.full((height, width), np.nan, dtype=np.float32)
                })
                continue
            
            # Z-buffering
            flat_indices = valid_v * width + valid_u
            z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)
            
            try:
                z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce="amin", include_self=True)
            except AttributeError:
                # Fallback for older PyTorch versions
                warnings.warn("PyTorch version too old for scatter_reduce_. Using CPU fallback for this camera.")
                result = cls._compute_numpy(points_world, point_ids, 
                                            camera_params_list[i][0], 
                                            camera_params_list[i][1], 
                                            camera_params_list[i][2], 
                                            width, height, 
                                            compute_depth_map=compute_depth_map)
                results.append(result)
                continue
            
            min_z_at_pixel = z_buffer[flat_indices]
            is_closest = torch.abs(valid_z - min_z_at_pixel) < 1e-4
            
            final_pixel_indices = flat_indices[is_closest]
            final_ids = valid_ids[is_closest]
            
            # Construct index map
            index_map_tensor = torch.full((height * width,), -1, device=device, dtype=torch.int32)
            index_map_tensor[final_pixel_indices] = final_ids
            index_map_2d = index_map_tensor.view(height, width)
            
            visible_indices = torch.unique(final_ids, sorted=True)

            if compute_depth_map:
                # Extract depth map from z_buffer and convert inf -> nan
                try:
                    z_buffer[z_buffer == float('inf')] = float('nan')
                    depth_map_2d = z_buffer.view(height, width)
                    depth_map_np = depth_map_2d.cpu().numpy()
                except Exception:
                    # Fallback: create nan-filled depth map
                    depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
            else:
                depth_map_np = None

            results.append({
                'index_map': index_map_2d.cpu().numpy(),
                'visible_indices': visible_indices.cpu().numpy(),
                'depth_map': depth_map_np
            })
        
        compute_time = time.time() - start_time
        print(f"⏱️ Computed batch visibility for {M} cameras, {len(points_world):,} "
              f"points using {device.upper()}: {compute_time:.3f}s")
        
        return results

    @staticmethod
    def _compute_torch(points_np, ids_np, K_np, R_np, t_np, width, height, device, compute_depth_map: bool = True):
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
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None
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

        # Extract the depth map from the Z-buffer if requested
        if compute_depth_map:
            try:
                z_buffer[z_buffer == float('inf')] = float('nan')
                depth_map_2d = z_buffer.view(height, width)
                depth_map_np = depth_map_2d.cpu().numpy()
            except Exception:
                depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
        else:
            depth_map_np = None

        return {
            'index_map': index_map_2d.cpu().numpy(),
            'visible_indices': visible_indices.cpu().numpy(),
            'depth_map': depth_map_np
        }

    @staticmethod
    def _compute_numpy(points, ids, K, R, t, width, height, compute_depth_map: bool = True):
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
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None
            }

        # 4. Z-Buffering (The Sorting Trick)
        # To handle occlusion efficiently in NumPy, we sort points by depth (Descending).
        # When we perform array assignment index_map[y, x] = id, the LAST value overwrites previous ones.
        # By sorting High -> Low, the Low-Z (closest) points are written last, correctly "winning" the pixel.
        sort_order = np.argsort(z_valid)[::-1]  # Descending order

        u_sorted = u_valid[sort_order]
        v_sorted = v_valid[sort_order]
        id_sorted = id_valid[sort_order]
        z_sorted = z_valid[sort_order]

        # 5. Create Outputs
        # Initialize map with -1 for IDs and NaN for depth
        index_map = np.full((height, width), -1, dtype=np.int32)
        depth_map = np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None

        # Bulk assignment handles the "last write wins" logic
        index_map[v_sorted, u_sorted] = id_sorted
        if compute_depth_map:
            depth_map[v_sorted, u_sorted] = z_sorted.astype(np.float32)

        # Extract unique IDs from the final map
        visible_indices = np.unique(index_map[index_map != -1])

        return {
            'index_map': index_map,
            'visible_indices': visible_indices,
            'depth_map': depth_map
        }
        
    @classmethod
    def compute_orthographic_visibility(cls, 
                                        points_world: np.ndarray,
                                        transform_matrix_inv: np.ndarray,
                                        width: int,
                                        height: int,
                                        point_ids: np.ndarray = None) -> dict:
        """
        Compute visibility for orthographic camera using affine transform.
        Groups points by [u,v] pixel, keeps highest Z per pixel.
        """
        start_time = time.time()
        
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)

        # 1. Prefer PyTorch (CUDA or CPU)
        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            result = cls._compute_ortho_torch(points_world, point_ids, transform_matrix_inv, width, height, device)
        else:
            # 2. Fallback to highly optimized Vectorized NumPy
            device = 'numpy'
            result = cls._compute_ortho_numpy(points_world, point_ids, transform_matrix_inv, width, height)

        compute_time = time.time() - start_time
        visible_count = len(result['visible_indices'])
        print(
            f"⏱️ Computed orthographic index map ({height}x{width}) for {len(points_world):,} points using "
            f"{device.upper()}: {visible_count:,} visible in {compute_time:.3f}s"
        )
        
        return result

    @staticmethod
    def _compute_ortho_torch(points_np, ids_np, transform_inv_np, width, height, device):
        """PyTorch-accelerated orthographic visibility computation."""
        # 1. Transfer to device
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids = torch.as_tensor(ids_np, dtype=torch.int32, device=device)
        T_inv = torch.as_tensor(transform_inv_np, dtype=torch.float32, device=device)

        # 2. Extract X, Y, Z and make X, Y homogeneous
        # points[:, :2] gets X, Y. We add a column of 1s for affine transform.
        N = points.shape[0]
        points_xy_hom = torch.cat([points[:, :2], torch.ones((N, 1), dtype=torch.float32, device=device)], dim=1)
        z = points[:, 2]

        # 3. Apply inverse affine transform: pixels_hom = T_inv @ points_hom.T
        pixels_hom = torch.matmul(T_inv, points_xy_hom.T).T
        
        u = pixels_hom[:, 0]
        v = pixels_hom[:, 1]

        # 4. Bounds Check & Integer Cast
        u_idx = u.round().long()
        v_idx = v.round().long()

        valid_mask = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)

        valid_u = u_idx[valid_mask]
        valid_v = v_idx[valid_mask]
        valid_z = z[valid_mask]
        valid_ids = p_ids[valid_mask]

        if valid_ids.numel() == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32)
            }

        # 5. Z-Buffering (Scatter Reduce for Max Z)
        flat_indices = valid_v * width + valid_u
        
        # Initialize with -infinity (we want the highest Z to win)
        z_buffer = torch.full((height * width,), float('-inf'), device=device, dtype=torch.float32)

        try:
            # reduce="amax" gets the maximum Z value per pixel
            z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce="amax", include_self=True)
        except AttributeError:
            # Fallback for old PyTorch versions
            return VisibilityManager._compute_ortho_numpy(points_np, ids_np, transform_inv_np, width, height)

        max_z_at_pixel = z_buffer[flat_indices]
        is_closest = torch.abs(valid_z - max_z_at_pixel) < 1e-4

        final_pixel_indices = flat_indices[is_closest]
        final_ids = valid_ids[is_closest]

        # 6. Construct Outputs
        index_map_tensor = torch.full((height * width,), -1, device=device, dtype=torch.int32)
        index_map_tensor[final_pixel_indices] = final_ids
        index_map_2d = index_map_tensor.view(height, width)

        visible_indices = torch.unique(final_ids, sorted=True)

        return {
            'index_map': index_map_2d.cpu().numpy(),
            'visible_indices': visible_indices.cpu().numpy()
        }

    @staticmethod
    def _compute_ortho_numpy(points, ids, transform_inv, width, height):
        """Vectorized NumPy fallback for orthographic visibility."""
        N = len(points)
        points_hom = np.column_stack([points[:, 0], points[:, 1], np.ones(N)])
        pixels_hom = (transform_inv @ points_hom.T).T
        
        u = np.rint(pixels_hom[:, 0]).astype(np.int32)
        v = np.rint(pixels_hom[:, 1]).astype(np.int32)
        z = points[:, 2]

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        id_valid = ids[valid_mask]

        if len(id_valid) == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32)
            }

        # Vectorized Z-Buffering (The Sorting Trick)
        # Sort ASCENDING by Z. Since index_map[y, x] = id overwrites previous values,
        # the highest Z (which comes last in ascending order) will "win" the pixel.
        sort_order = np.argsort(z_valid)

        u_sorted = u_valid[sort_order]
        v_sorted = v_valid[sort_order]
        id_sorted = id_valid[sort_order]

        index_map = np.full((height, width), -1, dtype=np.int32)
        index_map[v_sorted, u_sorted] = id_sorted

        visible_indices = np.unique(index_map[index_map != -1])

        return {
            'index_map': index_map,
            'visible_indices': visible_indices
        }
