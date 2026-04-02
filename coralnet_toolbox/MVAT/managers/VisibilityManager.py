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

    @staticmethod
    def _build_inverted_index(index_map: np.ndarray):
        """
        Build a CSR-style inverted index from a 2D index map.

        Maps each visible element ID back to the set of flat pixel indices
        (row-major: flat_idx = v * width + u) where it appears.

        Returns:
            dict with keys:
                'inv_ids'     (int32): sorted unique element IDs with >= 1 pixel
                'inv_offsets' (int64): length len(inv_ids)+1; offsets into inv_pixels
                'inv_pixels'  (int32): concatenated flat pixel indices
            or None if no valid pixels exist.
        """
        flat = index_map.ravel()
        flat_pixel_positions = np.where(flat >= 0)[0].astype(np.int32)
        if len(flat_pixel_positions) == 0:
            return None
        
        element_ids_at_pixels = flat[flat_pixel_positions].astype(np.int32)
        sort_order = np.argsort(element_ids_at_pixels, kind='stable')
        sorted_ids    = element_ids_at_pixels[sort_order]
        sorted_pixels = flat_pixel_positions[sort_order]
        unique_ids, start_positions, _ = np.unique(sorted_ids, return_index=True, return_counts=True)
        offsets = np.empty(len(unique_ids) + 1, dtype=np.int64)
        offsets[:-1] = start_positions
        offsets[-1]  = len(sorted_pixels)
        
        return {
            'inv_ids':     unique_ids.astype(np.int32),
            'inv_offsets': offsets,
            'inv_pixels':  sorted_pixels,
        }

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
                'element_type': 'point',
                'inverted_index': None,
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
        
        # Fallback: empty result
        return {
            'index_map': np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'element_type': element_type,
            'inverted_index': None,
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
        Attempts VTK rasterization first, falls back to Open3D raycasting,
        and finally falls back to face-center point sampling.
        """
        try:
            # First Choice: VTK (Pixel-perfect, supports all mesh types)
            return cls._compute_mesh_visibility_vtk(
                mesh_product, K, R, t, width, height, compute_depth_map
            )
        except Exception as e:
            print(f"⚠️ VTK mesh rasterization failed: {e}. Trying Open3D raycasting...")
            try:
                # Second Choice: Open3D (Thread-safe, fast, no OpenGL context required)
                import open3d
                return cls._compute_mesh_visibility_open3d(
                    mesh_product, K, R, t, width, height, compute_depth_map
                )
            except Exception as o3d_err:
                print(f"⚠️ Open3D raycasting failed: {o3d_err}. Falling back to face-center sampling")
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
    def _compute_mesh_visibility_open3d(cls,
                                        mesh_product: 'AbstractSceneProduct',
                                        K: np.ndarray,
                                        R: np.ndarray,
                                        t: np.ndarray,
                                        width: int,
                                        height: int,
                                        compute_depth_map: bool = True) -> dict:
        """
        Single-camera wrapper for Open3D mesh visibility computation.
        Delegates to the optimized batched method.
        """
        # Package the single camera parameters into the expected list format
        camera_params_list = [(K, R, t, width, height)]
        
        # Call the existing batched Open3D method
        results = cls.compute_batch_mesh_visibility_open3d(
            mesh_product, 
            camera_params_list, 
            compute_depth_maps=compute_depth_map
        )
        
        # Return the single result dictionary
        if results and len(results) > 0:
            return results[0]
            
        # Fallback empty dictionary if something goes wrong
        return {
            'index_map': np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'inverted_index': None,
        }
            
    @classmethod
    def compute_batch_mesh_visibility_open3d(cls, 
                                             mesh_product, 
                                             camera_params_list, 
                                             compute_depth_maps=True,
                                             use_global_bvh=False) -> list:
        """
        Batched Open3D raycasting with dynamic options for Frustum Culling or Global BVH.
        Includes detailed timing metrics for performance comparison.
        """
        import open3d as o3d
        import time
        import cv2
        
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f"🚀 STARTING BATCH VISIBILITY FOR {len(camera_params_list)} CAMERAS")
        print(f"{'='*50}")
        
        if use_global_bvh:
            # ---------------------------------------------------------
            # STRATEGY 1: Global BVH (Build once, cast deeper tree)
            # ---------------------------------------------------------
            print("🌐 STRATEGY: Global BVH")
            mesh_product.prepare_geometry()
            
            # If the persistent BVH hasn't been built yet, build it now
            if not getattr(mesh_product, '_o3d_raycasting_scene', None):
                print("   -> Building Global BVH from scratch...")
                build_start = time.time()
                scene = o3d.t.geometry.RaycastingScene()
                v_tensor = o3d.core.Tensor(mesh_product._cached_vertices, dtype=o3d.core.Dtype.Float32)
                
                # Get the full list of triangles (not culled)
                triangles = mesh_product._cached_triangles_pt.cpu().numpy().astype(np.uint32)
                t_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
                
                scene.add_triangles(v_tensor, t_tensor)
                mesh_product._o3d_raycasting_scene = scene
                bvh_build_time = time.time() - build_start
                print(f"   ✅ Global BVH built in {bvh_build_time:.4f}s (Faces: {len(triangles):,})")
            else:
                print("   ✅ Using cached Global BVH")
                bvh_build_time = 0.0
            
            scene = mesh_product._o3d_raycasting_scene
            original_cell_ids = getattr(mesh_product, '_original_cell_ids', None)
            num_faces = len(mesh_product._cached_triangles_pt)
            
        else:
            # ---------------------------------------------------------
            # STRATEGY 2: Dynamic Sub-BVH (Cull on GPU, build shallow tree)
            # ---------------------------------------------------------
            print("✂️ STRATEGY: Culled Sub-BVH")
            bvh_build_start = time.time()
            scene, original_cell_ids, num_faces = cls._build_subset_bvh(mesh_product, camera_params_list)
            bvh_build_time = time.time() - bvh_build_start
            print(f"   ✅ Sub-BVH prep & build completed in {bvh_build_time:.4f}s")

        results = []
        
        # If the culler removed everything (or empty mesh), return empty maps
        if num_faces == 0:
            print("   ⚠️ No faces visible. Returning empty maps.")
            return [{
                'index_map': np.full((h, w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((h, w), np.nan, dtype=np.float32) if compute_depth_maps else None,
                'inverted_index': None,
            } for _, _, _, w, h in camera_params_list]

        # 2. Fast Downsampled Raycasting
        SCALE_FACTOR = 0.25  # 1/4 resolution raycasting
        print(f"\n   -> Starting Raycasting (Scale: {SCALE_FACTOR}x)...")
        
        raycast_start_time = time.time()
        
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
                'depth_map': depth_map,
                'inverted_index': None,
            })
            
        raycast_time = time.time() - raycast_start_time
        total_time = time.time() - start_time
        
        print(f"   ✅ Raycasting finished in {raycast_time:.4f}s (Avg: {raycast_time/len(camera_params_list):.4f}s per camera)")
        print(f"\n📊 SUMMARY: {'Global BVH' if use_global_bvh else 'Sub-BVH'}")
        print(f"   - BVH Build/Prep : {bvh_build_time:.4f}s")
        print(f"   - Raycast Loop   : {raycast_time:.4f}s")
        print(f"   - Total Time     : {total_time:.4f}s")
        print(f"{'='*50}\n")
        
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
        import time
        
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f"🎨 VTK MESH VISIBILITY RASTERIZATION")
        print(f"{'='*50}")
        
        mesh = mesh_product.get_mesh()
        n_cells = mesh.n_cells
        
        if n_cells == 0:
            print("   ⚠️ No cells in mesh. Returning empty maps.")
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }
        
        print(f"   Mesh: {n_cells:,} cells | Render: {width}x{height} pixels")
        
        # --- 1. Encode face IDs as RGB colors ---
        print("   -> Encoding face IDs as RGB colors...")
        encode_start = time.time()
        
        # Use 24-bit encoding: R + G*256 + B*65536 = face_id
        # This supports up to 16.7M faces
        face_ids = np.arange(n_cells, dtype=np.int32)
        
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
        encode_time = time.time() - encode_start
        print(f"   ✅ RGB encoding completed in {encode_time:.4f}s")
        
        # --- 2. Create off-screen plotter ---
        print("   -> Creating off-screen plotter...")
        plotter_start = time.time()
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

        # Force Python to garbage collect the copy immediately!
        # VTK already has what it needs in its internal C++ pipeline.
        del mesh_with_ids
        import gc
        gc.collect()

        plotter_time = time.time() - plotter_start
        print(f"   ✅ Plotter setup completed in {plotter_time:.4f}s")
        
        # --- 3. Configure VTK camera from K, R, t ---
        print("   -> Configuring VTK camera from OpenCV intrinsics/extrinsics...")
        camera_start = time.time()
        cls._configure_vtk_camera(plotter, K, R, t, width, height, mesh.bounds)
        camera_time = time.time() - camera_start
        print(f"   ✅ Camera configuration completed in {camera_time:.4f}s")
        
        # --- 4. Render and extract face IDs ---
        print("   -> Rendering and extracting index map...")
        render_start = time.time()
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
        render_time = time.time() - render_start
        print(f"   ✅ Rendering and decoding completed in {render_time:.4f}s")
        
        # --- 5. Extract depth buffer ---
        depth_map = None
        if compute_depth_map:
            print("   -> Extracting depth buffer...")
            depth_start = time.time()
            try:
                # Get VTK depth buffer
                # PyVista's get_image_depth() returns actual Z-coordinates in camera space
                # VTK convention: camera looks down -Z, so visible objects have negative Z
                # OpenCV convention: camera looks down +Z, depth is positive
                vtk_depth = plotter.get_image_depth(fill_value=np.nan)
                
                # Negate to convert from VTK (-Z forward) to OpenCV (+Z forward) convention
                depth_map = -vtk_depth.astype(np.float16)
                depth_time = time.time() - depth_start
                print(f"   ✅ Depth buffer extracted in {depth_time:.4f}s")
                
            except Exception as e:
                print(f"   ⚠️ Failed to extract depth buffer: {e}")
                depth_map = np.full((height, width), np.nan, dtype=np.float16)
        
        # --- 6. Extract visible face IDs ---
        visible_indices = np.unique(index_map[index_map >= 0]).astype(np.int32)
        
        # Cleanup
        plotter.close()
        
        n_visible = len(visible_indices)
        coverage = np.sum(index_map >= 0) / (width * height) * 100
        total_time = time.time() - start_time
        
        print(f"\n📊 SUMMARY: VTK Rasterization")
        print(f"   - RGB Encoding   : {encode_time:.4f}s")
        print(f"   - Plotter Setup  : {plotter_time:.4f}s")
        print(f"   - Camera Config  : {camera_time:.4f}s")
        print(f"   - Render & Decode: {render_time:.4f}s")
        if compute_depth_map:
            print(f"   - Depth Extraction: {depth_time:.4f}s")
        print(f"   - Total Time     : {total_time:.4f}s")
        print(f"   - Result: {n_visible:,} visible faces, {coverage:.1f}% pixel coverage")
        print(f"{'='*50}\n")
        
        return {
            'index_map': index_map,
            'visible_indices': visible_indices,
            'depth_map': depth_map,
            'inverted_index': None,
        }
    
    @classmethod
    def compute_batch_mesh_visibility_vtk(cls,
                                          mesh_product: 'AbstractSceneProduct',
                                          camera_params_list: list,
                                          compute_depth_map: bool = True,
                                          scale_factor: float = 1.0,
                                          progress_callback=None) -> list:
        """
        Batched VTK-based mesh rasterization.
        Performs RGB encoding and Plotter setup ONCE, then iterates through cameras.
        Yields to the Qt event loop between cameras to keep the UI responsive.
        """
        import pyvista as pv
        import time
        
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f"🎨 STARTING BATCH VTK RASTERIZATION FOR {len(camera_params_list)} CAMERAS AT {scale_factor}x SCALE")
        print(f"{'='*50}")
        
        mesh = mesh_product.get_mesh()
        n_cells = mesh.n_cells
        
        if n_cells == 0:
            print("   ⚠️ No cells in mesh. Returning empty maps.")
            return [{
                'index_map': np.full((h, w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((h, w), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            } for _, _, _, w, h in camera_params_list]
            
        print(f"   Mesh: {n_cells:,} cells")
        
        # --- 1. SETUP PHASE (Done Once) ---
        setup_start = time.time()
        print("   -> Encoding face IDs as RGB colors...")
        face_ids = np.arange(n_cells, dtype=np.int32)
        encoded_ids = face_ids + 1
        r = (encoded_ids % 256).astype(np.uint8)
        g = ((encoded_ids // 256) % 256).astype(np.uint8)
        b = ((encoded_ids // 65536) % 256).astype(np.uint8)
        rgb_colors = np.column_stack([r, g, b])
        
        mesh_with_ids = mesh.copy()
        mesh_with_ids.cell_data['FaceID_RGB'] = rgb_colors
        encode_time = time.time() - setup_start
        
        print("   -> Creating off-screen plotter...")
        
        # FIX: Grab scaled dimensions from the first camera to set initial plotter size
        first_w = max(1, int(camera_params_list[0][3] * scale_factor))
        first_h = max(1, int(camera_params_list[0][4] * scale_factor))
        plotter = pv.Plotter(off_screen=True, window_size=(first_w, first_h))
        plotter.set_background('black') 
        
        plotter.add_mesh(
            mesh_with_ids,
            scalars='FaceID_RGB',
            rgb=True,
            lighting=False,
            interpolate_before_map=False,
            show_edges=False,
            style='surface'
        )

        # Delete the Python-side copy immediately
        del mesh_with_ids
        import gc
        gc.collect()

        plotter_setup_time = time.time() - setup_start - encode_time
        print(f"   ✅ Setup completed in {encode_time + plotter_setup_time:.4f}s")
        
        # --- 2. RENDER LOOP ---
        print(f"\n   -> Rendering {len(camera_params_list)} cameras at {scale_factor}x scale...")
        render_start_time = time.time()
        results = []
        
        for i, (K, R, t, width, height) in enumerate(camera_params_list):
            cam_start = time.time()

            # Progress callback (thread-safe status bar update)
            if progress_callback is not None:
                progress_callback(i + 1, len(camera_params_list))

            # Calculate scaled render dimensions
            render_w = max(1, int(width * scale_factor))
            render_h = max(1, int(height * scale_factor))

            # 1. Resize check (to scaled dimensions)
            current_size = plotter.window_size
            if current_size[0] != render_w or current_size[1] != render_h:
                plotter.window_size = (render_w, render_h)

            # Scale the intrinsic matrix
            K_scaled = K.copy()
            K_scaled[0, :3] *= scale_factor
            K_scaled[1, :3] *= scale_factor

            # 2. Config & Render using scaled parameters
            t0 = time.time()
            cls._configure_vtk_camera(plotter, K_scaled, R, t, render_w, render_h, mesh.bounds)
            plotter.render()
            t_render = time.time() - t0
            
            # 3. Screenshot transfer (GPU -> CPU)
            t0 = time.time()
            screenshot = plotter.screenshot(return_img=True)
            if screenshot.shape[2] == 4:
                screenshot = screenshot[:, :, :3]
            t_screenshot = time.time() - t0
                
            # 4. Decoding (Numpy Math)
            t0 = time.time()
            decoded = (screenshot[:, :, 0].astype(np.int32) +
                       screenshot[:, :, 1].astype(np.int32) * 256 +
                       screenshot[:, :, 2].astype(np.int32) * 65536)
            small_index_map = decoded - 1

            # Upsample to native resolution using nearest-neighbour interpolation
            if scale_factor != 1.0:
                import cv2
                index_map = cv2.resize(small_index_map, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                index_map = small_index_map
            t_decode = time.time() - t0
            
            # 5. Depth Extraction
            t0 = time.time()
            depth_map = None
            if compute_depth_map:
                try:
                    vtk_depth = plotter.get_image_depth(fill_value=np.nan)
                    small_depth = -vtk_depth.astype(np.float32)

                    if scale_factor != 1.0:
                        import cv2
                        depth_map = cv2.resize(small_depth, (width, height), interpolation=cv2.INTER_NEAREST)
                    else:
                        depth_map = small_depth
                except Exception as e:
                    print(f"   ⚠️ Failed to extract depth for camera {i}: {e}")
                    depth_map = np.full((height, width), np.nan, dtype=np.float32)
            t_depth = time.time() - t0
            
            # 6. Extract unique visible IDs (No CSR index stored, will use on-the-fly np.where() if needed)
            t0 = time.time()
            valid_mask = index_map >= 0
            visible_indices = np.unique(index_map[valid_mask]).astype(np.int32)
            t_unique = time.time() - t0
            
            results.append({
                'index_map': index_map,
                'visible_indices': visible_indices,
                'depth_map': depth_map,
                'inverted_index': None,
            })
            
            # 7. UI Yielding
            t0 = time.time()
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.processEvents()
            except ImportError:
                pass
            t_events = time.time() - t0
            
            cam_time = time.time() - cam_start
            print(f"      Cam {i+1}: {cam_time:.4f}s | Render: {t_render:.3f} | Snap: {t_screenshot:.3f} | Decode: {t_decode:.3f} | Depth: {t_depth:.3f} | Index: {t_unique:.3f} | UI Events: {t_events:.3f}")

        plotter.close()
        
        total_render_time = time.time() - render_start_time
        total_time = time.time() - start_time
        
        print(f"\n📊 SUMMARY: Batch VTK Rasterization")
        print(f"   - Setup Time     : {encode_time + plotter_setup_time:.4f}s")
        print(f"   - Render Loop    : {total_render_time:.4f}s")
        print(f"   - Total Time     : {total_time:.4f}s (Avg: {total_time/len(camera_params_list):.4f}s per camera)")
        print(f"{'='*50}\n")
        
        return results

    @classmethod
    def _configure_vtk_camera(cls, plotter, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                              width: int, height: int, bounds: tuple) -> None:
        
        position = -R.T @ t
        forward_cam = np.array([0.0, 0.0, 1.0])
        forward_world = R.T @ forward_cam
        focal_point = position + forward_world
        
        up_cam = np.array([0.0, -1.0, 0.0])  
        view_up = R.T @ up_cam
        
        camera = plotter.camera
        camera.position = position.tolist()
        camera.focal_point = focal_point.tolist()
        camera.up = view_up.tolist()
        
        # --- Set clipping range based on scene bounds ---
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
        near_clip = max(0.01, dist_to_center - scene_radius * 2)
        far_clip = dist_to_center + scene_radius * 2
        camera.clipping_range = (near_clip, far_clip)
        
        # ---> EXPLICIT PROJECTION MATRIX <---
        # Bypasses VTK's aspect ratio assumptions by explicitly mapping
        # the true horizontal (fx) and vertical (fy) OpenCV focal lengths.
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        import vtk
        mat = vtk.vtkMatrix4x4()
        mat.Zero()
        
        # Map X
        mat.SetElement(0, 0, 2.0 * fx / width)
        mat.SetElement(0, 2, (width - 2.0 * cx) / width)
        
        # Map Y (Y is flipped between OpenCV and OpenGL)
        mat.SetElement(1, 1, 2.0 * fy / height)
        mat.SetElement(1, 2, (2.0 * cy - height) / height)
        
        # Map Z (Clipping Range)
        mat.SetElement(2, 2, -(far_clip + near_clip) / (far_clip - near_clip))
        mat.SetElement(2, 3, -2.0 * far_clip * near_clip / (far_clip - near_clip))
        
        # W (Perspective divide)
        mat.SetElement(3, 2, -1.0)
        
        # Force VTK to use our raw matrix
        camera.SetUseExplicitProjectionTransformMatrix(True)
        camera.SetExplicitProjectionTransformMatrix(mat)

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
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
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
        print(f"\n{'='*50}")
        print(f"👁️  POINT CLOUD VISIBILITY COMPUTATION")
        print(f"{'='*50}")
        
        # Default point IDs if not provided
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)
        
        print(f"   Points: {len(points_world):,} | Render: {width}x{height} pixels")

        # 1. Prefer PyTorch (CUDA or CPU)
        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   Using {device.upper()} backend")
            compute_start = time.time()
            result = cls._compute_torch(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width,
                                        height, 
                                        device, 
                                        compute_depth_map=compute_depth_map)
            compute_time = time.time() - compute_start
        else:
            # 2. Fallback to NumPy if Torch is missing
            device = 'numpy'
            print(f"   Using NUMPY backend (PyTorch not available)")
            compute_start = time.time()
            result = cls._compute_numpy(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width, 
                                        height, 
                                        compute_depth_map=compute_depth_map)
            compute_time = time.time() - compute_start
        
        total_time = time.time() - start_time
        visible_count = len(result['visible_indices'])
        coverage = np.sum(result['index_map'] >= 0) / (width * height) * 100
        
        print(f"\n📊 SUMMARY: Point Cloud Visibility")
        print(f"   - Computation (Z-buffer): {compute_time:.4f}s")
        print(f"   - Total Time            : {total_time:.4f}s")
        print(f"   - Result: {visible_count:,} visible points, {coverage:.1f}% pixel coverage")
        print(f"{'='*50}\n")
        
        return result

    @classmethod
    def compute_batch_visibility(cls, 
                                 points_world: np.ndarray, 
                                 camera_params_list: list,
                                 point_ids: np.ndarray = None,
                                 compute_depth_map: bool = True) -> list:
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f"👁️  BATCH POINT CLOUD VISIBILITY COMPUTATION (STREAMING MODE)")
        print(f"{'='*50}")
        
        N_total = len(points_world)
        if point_ids is None:
            point_ids = np.arange(N_total, dtype=np.int32)
        
        print(f"   Points: {N_total:,} | Cameras: {len(camera_params_list)}")

        if not HAS_TORCH:
            # Fallback to numpy (omitted for brevity, keep your existing numpy fallback)
            pass 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using {device.upper()} backend")
        
        M = len(camera_params_list)
        results = []

        # Stream 100 million points (~1.2 GB) to the GPU at a time. 
        # Change this based on your hardware, but 100M is a very safe sweet spot.
        CHUNK_SIZE = 100_000_000 

        for i in range(M):
            cam_start = time.time()
            K_np, R_np, t_np, width, height = camera_params_list[i]
            
            # Load camera matrices to GPU
            K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
            R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
            t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
            
            # Initialize the MASTER Z-buffer and Index Map for this camera
            # This takes very little memory (O(width * height))
            global_z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)
            global_index_map = torch.full((height * width,), -1, device=device, dtype=torch.int32)

            print(f"   -> Processing Camera {i+1}/{M} in chunks...")

            for start_idx in range(0, N_total, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, N_total)
                
                # Stream just this chunk to the GPU
                chunk_pts = torch.as_tensor(points_world[start_idx:end_idx], dtype=torch.float32, device=device)
                chunk_ids = torch.as_tensor(point_ids[start_idx:end_idx], dtype=torch.int32, device=device)

                # 1. Transform World -> Camera
                points_cam = chunk_pts @ R.T + t
                x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
                
                # 2. Project to Image Plane
                u = (K[0, 0] * x / z) + K[0, 2]
                v = (K[1, 1] * y / z) + K[1, 2]
                
                # 3. Bounds check
                u_idx, v_idx = u.round().long(), v.round().long()
                valid_mask = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height) & (z > 0)
                
                valid_u, valid_v, valid_z = u_idx[valid_mask], v_idx[valid_mask], z[valid_mask]
                valid_ids = chunk_ids[valid_mask]
                
                if valid_ids.numel() == 0:
                    del chunk_pts, chunk_ids, points_cam, x, y, z, u, v, u_idx, v_idx, valid_mask
                    continue
                
                flat_indices = valid_v * width + valid_u
                
                # 4. Local Z-buffering (Who won inside this specific chunk?)
                local_z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)
                try:
                    local_z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce="amin", include_self=True)
                except AttributeError:
                    raise RuntimeError("PyTorch version too old for scatter_reduce_.")
                
                # 5. Resolve IDs for this chunk
                is_closest = torch.abs(valid_z - local_z_buffer[flat_indices]) < 1e-4
                local_index_map = torch.full((height * width,), -1, device=device, dtype=torch.int32)
                local_index_map[flat_indices[is_closest]] = valid_ids[is_closest]

                # 6. MERGE WITH MASTER: Did this chunk beat the global record?
                won_mask = local_z_buffer < global_z_buffer
                global_z_buffer[won_mask] = local_z_buffer[won_mask]
                global_index_map[won_mask] = local_index_map[won_mask]

                # Free VRAM immediately for the next chunk
                del chunk_pts, chunk_ids, points_cam, x, y, z, u, v, u_idx, v_idx, valid_mask
                del valid_u, valid_v, valid_z, valid_ids, flat_indices, local_z_buffer, is_closest, local_index_map, won_mask

            # 7. Finalize outputs for this camera
            visible_indices = torch.unique(global_index_map[global_index_map != -1], sorted=True)
            
            if compute_depth_map:
                try:
                    global_z_buffer[global_z_buffer == float('inf')] = float('nan')
                    depth_map_np = global_z_buffer.view(height, width).cpu().numpy()
                except Exception:
                    depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
            else:
                depth_map_np = None

            index_map_np = global_index_map.view(height, width).cpu().numpy()
            
            results.append({
                'index_map': index_map_np,
                'visible_indices': visible_indices.cpu().numpy(),
                'depth_map': depth_map_np,
                'inverted_index': VisibilityManager._build_inverted_index(index_map_np),
            })
            
            print(f"   ✅ Camera {i+1} completed in {time.time() - cam_start:.2f}s")

            # Free global buffers
            del K, R, t, global_z_buffer, global_index_map, visible_indices

        if device == 'cuda':
            torch.cuda.empty_cache()
            
        print(f"\n   - Total Time: {time.time() - start_time:.4f}s")
        return results

    @staticmethod
    def _compute_torch(points_np, ids_np, K_np, R_np, t_np, width, height, device, compute_depth_map: bool = True):
        """
        PyTorch-based visibility computation.
        Uses scatter_reduce_ for efficient Z-buffering.
        Works on 'cuda' (fastest) and 'cpu' (via PyTorch tensors).
        """
        stage_times = {}
        overall_start = time.time()
        
        # 1. Transfer Data to Device (GPU or CPU)
        # We assume input is float32 for geometry, int32 for IDs
        xfer_start = time.time()
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids = torch.as_tensor(ids_np, dtype=torch.int32, device=device)
        
        K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
        R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
        stage_times['transfer'] = time.time() - xfer_start

        # 2. Transform World -> Camera
        # X_cam = R * X_world + t
        # Shape logic: (N, 3) @ (3, 3).T + (3,)
        transform_start = time.time()
        points_cam = points @ R.T + t
        stage_times['transform'] = time.time() - transform_start

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # 3. Project to Image Plane (Vectorized)
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # Note: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        proj_start = time.time()
        u = (K[0, 0] * x_cam / z_cam) + K[0, 2]
        v = (K[1, 1] * y_cam / z_cam) + K[1, 2]
        stage_times['projection'] = time.time() - proj_start

        # 4. Bounds & Depth Check
        bounds_start = time.time()
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
        stage_times['bounds'] = time.time() - bounds_start

        if valid_ids.numel() == 0:
            # Nothing visible
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }

        # 5. Z-Buffering (Scatter Reduce)
        # Flatten indices: idx = y * width + x
        zbuf_start = time.time()
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
        stage_times['zbuffer'] = time.time() - zbuf_start

        # 6. Construct Outputs
        output_start = time.time()
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

        index_map_np = index_map_2d.cpu().numpy()
        stage_times['output'] = time.time() - output_start

        # Free GPU memory back to OS/other applications
        if str(device) == 'cuda':
            torch.cuda.empty_cache()

        return {
            'index_map': index_map_np,
            'visible_indices': visible_indices.cpu().numpy(),
            'depth_map': depth_map_np,
            'inverted_index': None,
        }

    @staticmethod
    def _compute_numpy(points, ids, K, R, t, width, height, compute_depth_map: bool = True):
        """
        CPU-based visibility computation (Legacy / Fallback).
        Uses 'Sort by Depth' optimization to handle occlusion efficiently without loops.
        """
        stage_times = {}
        overall_start = time.time()
        
        # 1. Transform World -> Camera
        # X_cam = R * X_world + t
        transform_start = time.time()
        points_cam = points @ R.T + t
        stage_times['transform'] = time.time() - transform_start

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # 2. Project to Image Plane
        proj_start = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (K[0, 0] * x_cam / z_cam) + K[0, 2]
            v = (K[1, 1] * y_cam / z_cam) + K[1, 2]
        stage_times['projection'] = time.time() - proj_start

        # 3. Bounds Check & Integer Cast
        bounds_start = time.time()
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
        stage_times['bounds'] = time.time() - bounds_start

        if len(id_valid) == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }

        # 4. Z-Buffering (The Sorting Trick)
        # To handle occlusion efficiently in NumPy, we sort points by depth (Descending).
        # When we perform array assignment index_map[y, x] = id, the LAST value overwrites previous ones.
        # By sorting High -> Low, the Low-Z (closest) points are written last, correctly "winning" the pixel.
        zbuf_start = time.time()
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
        
        stage_times['zbuffer'] = time.time() - zbuf_start

        # Extract unique IDs from the final map
        visible_indices = np.unique(index_map[index_map != -1])

        return {
            'index_map': index_map,
            'visible_indices': visible_indices,
            'depth_map': depth_map,
            'inverted_index': None,
        }
        
    @classmethod
    def compute_orthographic_visibility(cls, 
                                        points_world: np.ndarray,
                                        transform_matrix_inv: np.ndarray,
                                        width: int,
                                        height: int,
                                        point_ids: np.ndarray = None,
                                        chunk_transform_inv: np.ndarray = None) -> dict:
        """
        Compute visibility for orthographic camera using affine transform.
        Groups points by [u,v] pixel, keeps highest Z per pixel.
        
        Args:
            points_world: (N, 3) array of point coordinates (may be in local or world space)
            transform_matrix_inv: (3, 3) inverse georeferencing matrix (world -> pixel)
            width: Image width
            height: Image height
            point_ids: Optional (N,) array of point IDs
            chunk_transform_inv: Optional (4, 4) inverse chunk transform (world -> local).
                                If provided, points_world are assumed to be in local space,
                                and this transform converts them to geo-world space before projection.
        """
        start_time = time.time()
        
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)

        # 1. Prefer PyTorch (CUDA or CPU)
        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            result = cls._compute_ortho_torch(points_world, point_ids, transform_matrix_inv, width, height, device, chunk_transform_inv)
        else:
            # 2. Fallback to highly optimized Vectorized NumPy
            device = 'numpy'
            result = cls._compute_ortho_numpy(points_world, point_ids, transform_matrix_inv, width, height, chunk_transform_inv)

        compute_time = time.time() - start_time
        visible_count = len(result['visible_indices'])
        print(
            f"⏱️ Computed orthographic index map ({height}x{width}) for {len(points_world):,} points using "
            f"{device.upper()}: {visible_count:,} visible in {compute_time:.3f}s"
        )
        
        return result

    @staticmethod
    def _compute_ortho_torch(points_np, ids_np, transform_inv_np, width, height, device, chunk_transform_inv_np=None):
        """PyTorch-accelerated orthographic visibility computation."""
        # 1. Transfer to device
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids = torch.as_tensor(ids_np, dtype=torch.int32, device=device)
        T_inv = torch.as_tensor(transform_inv_np, dtype=torch.float32, device=device)

        N = points.shape[0]
        z = points[:, 2]

        # 2. Apply chunk_transform if provided (local -> world)
        if chunk_transform_inv_np is not None:
            chunk_T_inv = torch.as_tensor(chunk_transform_inv_np, dtype=torch.float32, device=device)
            # Convert local points to world: world_point = inv(chunk_T_inv) @ local_point
            chunk_T = torch.linalg.inv(chunk_T_inv)  # local -> world transform
            points_hom = torch.cat([points, torch.ones((N, 1), dtype=torch.float32, device=device)], dim=1)
            world_hom = torch.matmul(chunk_T, points_hom.T).T  # (N, 4)
            points_xy_hom = torch.cat([world_hom[:, :2], torch.ones((N, 1), dtype=torch.float32, device=device)], dim=1)
        else:
            # Assume points are already in world space: extract X, Y and make homogeneous
            points_xy_hom = torch.cat([points[:, :2], torch.ones((N, 1), dtype=torch.float32, device=device)], dim=1)

        # 3. Apply geospatial inverse affine transform: pixels_hom = T_inv @ world_xy_hom.T
        pixels_hom = torch.matmul(T_inv, points_xy_hom.T).T
        
        u = pixels_hom[:, 0]
        v = pixels_hom[:, 1]

        # 4. Bounds Check & Integer Cast
        u_idx = u.floor().long()
        v_idx = v.floor().long()

        valid_mask = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)

        valid_u = u_idx[valid_mask]
        valid_v = v_idx[valid_mask]
        valid_z = z[valid_mask]
        valid_ids = p_ids[valid_mask]

        if valid_ids.numel() == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'inverted_index': None,
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

        index_map_np = index_map_2d.cpu().numpy()

        # Free GPU memory back to OS/other applications
        if str(device) == 'cuda':
            torch.cuda.empty_cache()

        return {
            'index_map': index_map_np,
            'visible_indices': visible_indices.cpu().numpy(),
            'inverted_index': None,
        }

    @staticmethod
    def _compute_ortho_numpy(points, ids, transform_inv, width, height, chunk_transform_inv=None):
        """Vectorized NumPy fallback for orthographic visibility."""
        N = len(points)
        z = points[:, 2]
        
        # Apply chunk_transform if provided (local -> world)
        if chunk_transform_inv is not None:
            chunk_T = np.linalg.inv(chunk_transform_inv)  # local -> world transform
            points_hom = np.column_stack([points[:, 0], points[:, 1], points[:, 2], np.ones(N)])
            world_hom = (chunk_T @ points_hom.T).T  # (N, 4)
            points_xy_hom = np.column_stack([world_hom[:, 0], world_hom[:, 1], np.ones(N)])
        else:
            # Assume points are already in world space
            points_xy_hom = np.column_stack([points[:, 0], points[:, 1], np.ones(N)])
        
        # Apply geospatial inverse affine transform
        pixels_hom = (transform_inv @ points_xy_hom.T).T
        
        u = np.floor(pixels_hom[:, 0]).astype(np.int32)
        v = np.floor(pixels_hom[:, 1]).astype(np.int32)

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        id_valid = ids[valid_mask]

        if len(id_valid) == 0:
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'inverted_index': None,
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
            'visible_indices': visible_indices,
            'inverted_index': VisibilityManager._build_inverted_index(index_map),
        }
