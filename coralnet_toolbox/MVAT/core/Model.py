import pyvista as pv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PointCloud():
    def __init__(self, file_path=None, points=None, colors=None, name="PointCloud"):
        """
        Initialize PointCloud from file or pre-parsed data.
        
        Args:
            file_path (str): Path to .ply file (optional if points provided)
            points (np.ndarray): Nx3 array of 3D points (optional if file_path provided)
            colors (np.ndarray): Nx3 uint8 array of RGB colors (optional)
            name (str): A display name for this point cloud (e.g., "Dense Cloud").
        """
        self.label = name
        if file_path is not None:
            # Load from file
            self.point_cloud = pv.read(file_path)
        elif points is not None:
            # Create from parsed data
            self.point_cloud = pv.PolyData(points)
            if colors is not None and len(colors) == len(points):
                # Add RGB colors
                self.point_cloud.point_data['RGB'] = colors
        else:
            raise ValueError("Either file_path or points must be provided")
            
    def get_label(self):
        return self.label

    def get_actor(self, **kwargs):
        """Returns the point cloud data as a PyVista PolyData object, ready to be added to a plotter."""
        # The "actor" for a point cloud is just the data itself, which add_points/add_mesh turns into an actor.
        return self.point_cloud