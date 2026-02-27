import os
import time

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

    def get_points_array(self):
        """
        Get the raw point coordinates as a numpy array for efficient processing.
        
        Returns:
            np.ndarray: (N, 3) array of point coordinates, or None if no mesh
        """
        if self.mesh is None:
            return None
        return self.mesh.points
    
