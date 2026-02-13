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
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        load_time = time.time() - start_time
        
        print(f"⏱️ Loaded PointCloud: {self.label} with {self.mesh.n_points:,} points in {load_time:.3f}s")
        
        # Add a cache storage for GPU tensors (Compute Space)
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

    def get_points_array(self):
        """
        Get the raw point coordinates as a numpy array for efficient processing.
        
        Returns:
            np.ndarray: (N, 3) array of point coordinates, or None if no mesh
        """
        if self.mesh is None:
            return None
        return self.mesh.points
    
    def get_subset_data(self, indices):
        """
        Extract a subset of point cloud data using GPU-accelerated slicing.
        
        This method uses the hybrid "Compute vs. Render" architecture:
        1. Slice raw Torch tensors on GPU (fast CUDA indexing)
        2. Download only the subset to CPU (minimal PCIe transfer)
        3. Wrap in PyVista PolyData for rendering
        
        Args:
            indices: Array/List of point indices to extract.
                     - None: return full cloud
                     - Empty: return empty PolyData
                     - Array: return filtered subset
            
        Returns:
            pv.PolyData: PyVista mesh containing only the subset points
        """
        if self.mesh is None:
            return pv.PolyData()
            
        start_time = time.time()
        
        # --- 1. PRE-PROCESS INDICES (CPU Flattening) ---
        if indices is None:
            # Return full cloud
            return self.mesh.copy()
            
        if isinstance(indices, (list, tuple)):
            if len(indices) > 0 and isinstance(indices[0], (np.ndarray, list)):
                try:
                    indices = np.concatenate(indices)
                except Exception:
                    indices = np.hstack(indices)
            else:
                indices = np.array(indices, dtype=np.int32)
        
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype=np.int32)
        
        if len(indices) == 0:
            # Return empty mesh
            return pv.PolyData()

        # --- 2. GPU PATH (The Speedup) ---
        if torch.cuda.is_available() and self._ensure_gpu_cache():
            try:
                # Convert indices to GPU tensor
                indices_tensor = torch.from_numpy(indices).to('cuda')
                indices_tensor = torch.unique(indices_tensor)
                
                # Perform GPU slicing (FAST - this is the key optimization)
                subset_points_tensor = self.gpu_cache['points'][indices_tensor]
                
                subset_data = {}
                for name, data_tensor in self.gpu_cache['data'].items():
                    subset_data[name] = data_tensor[indices_tensor]
                
                # Download only the subset to CPU (minimal transfer)
                subset_points = subset_points_tensor.cpu().numpy()
                subset_data_cpu = {name: data.cpu().numpy() for name, data in subset_data.items()}
                
                # Wrap in PyVista PolyData
                subset_mesh = pv.PolyData(subset_points)
                for name, data in subset_data_cpu.items():
                    subset_mesh.point_data[name] = data
                
                extract_time = time.time() - start_time
                print(f"⚡ get_subset_data (GPU): Extracted {len(indices_tensor):,} pts in {extract_time:.3f}s")
                
                return subset_mesh
                
            except Exception as e:
                print(f"⚠️ GPU extraction failed, falling back to CPU: {e}")
        
        # --- 3. CPU FALLBACK ---
        indices = np.unique(indices)
        indices = indices[indices < self.mesh.n_points]
        
        subset_points = self.mesh.points[indices]
        
        # Create PolyData
        subset_mesh = pv.PolyData(subset_points)
        
        # Copy point data
        for name, data in self.mesh.point_data.items():
            subset_mesh.point_data[name] = data[indices]
        
        extract_time = time.time() - start_time
        print(f"⏱️ get_subset_data (CPU): Extracted {len(indices):,} pts in {extract_time:.3f}s")
        
        return subset_mesh
