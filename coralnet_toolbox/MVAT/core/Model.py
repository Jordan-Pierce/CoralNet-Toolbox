import os
import time

import torch
import numpy as np

import pyvista as pv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PointCloud():
    def __init__(self, file_path, point_size=1):
        """
        Initialize PointCloud from file.
        
        Args:
            file_path (str): Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size (int): Size of points when rendered
        """
        self.file_path = file_path
        self.label = os.path.basename(file_path)
        
        self.point_size = point_size
        self.mesh = None
        self.actor = None
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        load_time = time.time() - start_time
        
        print(f"⏱️ Loaded PointCloud: {self.label} with {self.mesh.n_points:,} points in {load_time:.3f}s")
        
        # Add a cache storage for GPU tensors
        self.gpu_cache = None 
        
        # Pre-load to GPU if possible (Optional: can also be lazy)
        if torch.cuda.is_available():
            self._ensure_gpu_cache()
        
    @classmethod
    def from_file(cls, file_path, point_size=1):
        """
        Load a point cloud from a file.
        
        Args:
            file_path (str): Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size (int): Size of points when rendered
            
        Returns:
            PointCloud: Loaded point cloud instance
        """
        return cls(file_path=file_path, point_size=point_size)
            
    def get_label(self):
        return self.label
    
    def get_mesh(self):
        """Returns the underlying PyVista mesh."""
        return self.mesh
    
    def _ensure_gpu_cache(self):
        """
        Uploads the entire point cloud to VRAM once.
        """
        if self.gpu_cache is not None:
            return True
            
        try:
            print("⚡ Uploading Point Cloud to GPU VRAM...")
            start_time = time.time()
            
            # 1. Upload Points
            points_tensor = torch.from_numpy(self.mesh.points).to('cuda')
            
            # 2. Upload Data (Colors, etc.)
            data_cache = {}
            for name, data in self.mesh.point_data.items():
                data_cache[name] = torch.from_numpy(data).to('cuda')
                
            self.gpu_cache = {
                'points': points_tensor,
                'data': data_cache
            }
            
            print(f"⚡ Upload complete in {time.time() - start_time:.3f}s")
            return True
        except Exception as e:
            print(f"⚠️ GPU Upload Failed: {e}")
            self.gpu_cache = None
            return False

    def add_to_plotter(self, plotter):
        """
        Add this point cloud to a PyVista plotter.
        
        Args:
            plotter: PyVista plotter instance
            
        Returns:
            The actor created by the plotter
        """
        # Remove existing actor if any
        if self.actor is not None:
            try:
                plotter.remove_actor(self.actor)
            except:
                pass
        
        # Handle styling for point cloud vs meshes
        if 'RGB' in self.mesh.point_data:
            self.actor = plotter.add_mesh(self.mesh, 
                                          scalars='RGB', 
                                          rgb=True, 
                                          point_size=self.point_size,
                                          style='points',
                                          render_points_as_spheres=False,
                                          lighting=False)
        else:
            point_size = self.point_size if self.mesh.n_cells == 0 else None
            self.actor = plotter.add_mesh(self.mesh, 
                                          color='black', 
                                          point_size=point_size,
                                          style='points',
                                          render_points_as_spheres=False,
                                          lighting=False)
        
        return self.actor
    
    def set_visible(self, visible):
        """Set visibility of the point cloud actor."""
        if self.actor is not None:
            self.actor.SetVisibility(visible)
    
    def set_point_size(self, size):
        """Update the point size for the point cloud."""
        self.point_size = size
        if self.actor is not None:
            self.actor.GetProperty().SetPointSize(size)

    def get_actor(self, **kwargs):
        """Returns the point cloud data as a PyVista PolyData object, ready to be added to a plotter."""
        # The "actor" for a point cloud is just the data itself, which add_points/add_mesh turns into an actor.
        return self.mesh
    
    def get_points_array(self):
        """
        Get the raw point coordinates as a numpy array for efficient processing.
        
        Returns:
            np.ndarray: (N, 3) array of point coordinates, or None if no mesh
        """
        if self.mesh is None:
            return None
        return self.mesh.points
    
    def get_subset_data(self, indices, use_torch=True, use_optimized=True):
        """
        Extract a subset of point cloud data.
        
        Args:
            indices: Point indices to extract
            use_torch: Whether to use PyTorch for extraction
            use_optimized: If True, use optimized GPU caching version; if False, use PyVista-based version
            
        Returns:
            If use_optimized=True: (points_array, point_data_dict)
            If use_optimized=False: pv.PolyData mesh
        """
        if not use_optimized:
            # Version 1: PyVista-based extraction returning PolyData
            if self.mesh is None:
                return None
            
            start_time = time.time()
            
            # Convert indices to numpy array if needed
            if not isinstance(indices, np.ndarray):
                indices = np.array(indices, dtype=np.int32)
            
            # Handle empty indices
            if len(indices) == 0:
                return pv.PolyData()
            
            # Ensure indices are within valid range
            indices = indices[indices < self.mesh.n_points]
            
            if use_torch:
                # PyTorch-based extraction
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                try:
                    # Convert points to torch tensor on device
                    points_tensor = torch.from_numpy(self.mesh.points).to(device)
                    indices_tensor = torch.from_numpy(indices).to(device)
                    
                    # Extract subset points
                    subset_points = points_tensor[indices_tensor].cpu().numpy()
                    
                    # Extract all point data arrays
                    subset_point_data = {}
                    for name, data in self.mesh.point_data.items():
                        data_tensor = torch.from_numpy(data).to(device)
                        subset_point_data[name] = data_tensor[indices_tensor].cpu().numpy()
                    
                    # Create new PolyData
                    subset_mesh = pv.PolyData(subset_points)
                    # Add point data
                    for name, data in subset_point_data.items():
                        subset_mesh.point_data[name] = data
                    
                    extract_time = time.time() - start_time
                    print(f"⏱️ get_subset_data (PyVista): Extracted {len(indices):,} points from {self.mesh.n_points:,} total in {extract_time:.3f}s (using PyTorch on {device.upper()})")
                    return subset_mesh
                
                except Exception as e:
                    print(f"Error extracting point subset with PyTorch: {e}")
                    # Fallback to PyVista if PyTorch fails
                    use_torch = False
            
            if not use_torch:
                # Original PyVista-based extraction
                try:
                    subset_mesh = self.mesh.extract_points(indices, adjacent_cells=False)
                    extract_time = time.time() - start_time
                    print(f"⏱️ get_subset_data (PyVista): Extracted {len(indices):,} points from {self.mesh.n_points:,} total in {extract_time:.3f}s (using PyVista)")
                    return subset_mesh
                except Exception as e:
                    print(f"Error extracting point subset: {e}")
                    return None
        else:
            # Version 2: Optimized GPU caching version
            if self.mesh is None:
                return None, None
                
            start_time = time.time()
            
            # --- 1. PRE-PROCESS INDICES (CPU Flattening) ---
            if indices is None:
                return self.mesh.points, dict(self.mesh.point_data)
                
            if isinstance(indices, (list, tuple)):
                if len(indices) > 0 and isinstance(indices[0], (np.ndarray, list)):
                    try:
                        indices = np.concatenate(indices)
                    except:
                        indices = np.hstack(indices)
                else:
                    indices = np.array(indices, dtype=np.int32)
            
            if len(indices) == 0:
                return np.empty((0, 3)), {}

            # --- 2. GPU PATH (The Speedup) ---
            if use_torch and torch.cuda.is_available() and self._ensure_gpu_cache():
                try:
                    indices_tensor = torch.from_numpy(indices).to('cuda')
                    indices_tensor = torch.unique(indices_tensor)
                    
                    subset_points_tensor = self.gpu_cache['points'][indices_tensor]
                    
                    subset_data = {}
                    for name, data_tensor in self.gpu_cache['data'].items():
                        subset_data[name] = data_tensor[indices_tensor].cpu().numpy()
                    
                    subset_points = subset_points_tensor.cpu().numpy()
                    
                    extract_time = time.time() - start_time
                    print(f"⚡ get_subset_data (Optimized GPU): {len(indices_tensor):,} pts in {extract_time:.3f}s")
                    
                    return subset_points, subset_data
                    
                except Exception as e:
                    print(f"GPU Error, falling back: {e}")
            
            # --- 3. CPU FALLBACK ---
            indices = np.unique(indices)
            indices = indices[indices < self.mesh.n_points]
            
            subset_points = self.mesh.points[indices]
            subset_data = {n: d[indices] for n, d in self.mesh.point_data.items()}
            
            print(f"⏱️ get_subset_data (Optimized CPU): {len(indices):,} pts in {time.time() - start_time:.3f}s")
            
            return subset_points, subset_data