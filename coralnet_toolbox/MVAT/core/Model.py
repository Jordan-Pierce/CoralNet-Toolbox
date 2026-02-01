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
        import os
        self.file_path = file_path
        self.label = os.path.basename(file_path)
        self.point_size = point_size
        self.mesh = None
        self.actor = None
        
        # Load from file
        self.mesh = pv.read(file_path)
        self.array_names = self.mesh.array_names
        
        print("Loaded PointCloud:", self.label, "with", self.mesh.n_points, "points")
    
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
                                          point_size=self.point_size)
        else:
            point_size = self.point_size if self.mesh.n_cells == 0 else None
            self.actor = plotter.add_mesh(self.mesh, 
                                          color='black', 
                                          point_size=point_size)
        
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