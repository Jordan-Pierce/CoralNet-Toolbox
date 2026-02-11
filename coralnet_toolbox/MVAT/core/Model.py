import time
import pyvista as pv
import numpy as np


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
        import os
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
    
    def extract_subset(self, indices):
        """
        Create a filtered point cloud containing only points with specified indices.
        Preserves all point data (RGB, scalars, etc.) from the original mesh.
        
        Args:
            indices (np.ndarray or list): 1D array/list of point indices to extract
            
        Returns:
            pv.PolyData: New PyVista mesh containing only the specified points
        """
        if self.mesh is None:
            return None
        
        start_time = time.time()
        
        # Convert to numpy array if needed
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype=np.int32)
        
        # Handle empty indices
        if len(indices) == 0:
            # Return empty PolyData
            return pv.PolyData()
        
        # Ensure indices are within valid range
        indices = indices[indices < self.mesh.n_points]
        
        # Use PyVista's extract_points method
        # This preserves all point data arrays
        try:
            subset_mesh = self.mesh.extract_points(indices, adjacent_cells=False)
            extract_time = time.time() - start_time
            print(f"⏱️ Extracted {len(indices):,} points from {self.mesh.n_points:,} total in {extract_time:.3f}s")
            return subset_mesh
        except Exception as e:
            print(f"Error extracting point subset: {e}")
            return None