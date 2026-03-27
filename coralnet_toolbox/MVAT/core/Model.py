"""
3D Scene Product Implementations for MVAT

Provides concrete implementations of AbstractSceneProduct for:
- PointCloudProduct: Point cloud data (.ply, .pcd, etc.)
- MeshProduct: Surface meshes with faces (.obj, .stl, .ply with faces)

Backward Compatibility:
- PointCloud class is preserved as an alias for PointCloudProduct
"""
import os
import time
from typing import Optional

import numpy as np
import pyvista as pv

from coralnet_toolbox.MVAT.core.SceneProduct import (
    AbstractSceneProduct,
    BoundsType,
    ElementType,
    RenderStyle,
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PointCloudProduct(AbstractSceneProduct):
    """
    Scene product for point cloud data.
    
    Wraps PyVista point cloud meshes and provides the interface required by
    the MVAT visibility engine. Point clouds are NOT solid (is_solid=False)
    since they cannot provide accurate occlusion without splatting.
    
    Attributes:
        mesh (pv.PolyData): The underlying PyVista point cloud mesh.
        point_size (int): Rendering point size.
        array_names (list): Names of point data arrays (RGB, normals, etc.)
        available_arrays (list): All selectable arrays including built-in "Labels"
        selected_array (str): Currently selected array for visualization
    """
    
    def __init__(self, file_path: str, point_size: int = 1, product_id: Optional[str] = None):
        """
        Initialize PointCloudProduct from file.
        
        Args:
            file_path: Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size: Size of points when rendered
            product_id: Optional unique ID (defaults to filename)
        """
        # Generate product_id from filename if not provided
        if product_id is None:
            product_id = os.path.basename(file_path)
        
        super().__init__(product_id=product_id, file_path=file_path)
        
        self.point_size = point_size
        self.mesh: Optional[pv.PolyData] = None
        self.array_names = []
        self.available_arrays = []  # Will be built after loading
        self.selected_array = "RGB"  # Default to RGB
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        
        # Build available arrays in priority order:
        # 1. RGB (first, created/synthesized if needed with Metashape purple as default)
        # 2. Labels (always available, all white by default)
        # 3. Everything else from the mesh
        other_arrays = [arr for arr in self.array_names if arr.lower() not in ('rgb', 'labels')]
        self.available_arrays = ["RGB", "Labels"] + other_arrays
        
        load_time = time.time() - start_time
        
        print(f"⏱️ Loaded PointCloudProduct: {self.label} with {self.mesh.n_points:,} points in {load_time:.3f}s")
        print(f"   Available arrays for visualization: {self.available_arrays}")
    
    @classmethod
    def from_file(cls, file_path: str, point_size: int = 1) -> 'PointCloudProduct':
        """
        Load a point cloud from a file.
        
        Args:
            file_path: Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size: Size of points when rendered
            
        Returns:
            PointCloudProduct instance
        """
        return cls(file_path=file_path, point_size=point_size)
    
    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------
    
    def get_render_mesh(self) -> pv.PolyData:
        """Get the PyVista mesh for rendering."""
        return self.mesh
    
    def get_render_style(self) -> RenderStyle:
        """Get preferred rendering style based on selected array."""
        style = {
            'style': 'points',
            'point_size': self.point_size,
            'render_points_as_spheres': False,
            'lighting': False,
        }
        
        # Apply rendering based on selected array
        if self.selected_array == "RGB":
            # RGB: use actual RGB if available, otherwise solid black as fallback
            # (RGB itself is not a synthesized channel for point clouds; use black as default)
            if 'RGB' in self.array_names:
                style['scalars'] = 'RGB'
                style['rgb'] = True
            else:
                # No RGB data - render as Metashape purple
                style['color'] = '#8d8cc4'
        elif self.selected_array == "Labels":
            # Labels: render as white points (all same color)
            style['color'] = 'white'
        elif self.selected_array in self.array_names:
            # Use selected array as scalars (applies colormap)
            style['scalars'] = self.selected_array
        else:
            # Fallback to Metashape purple
            style['color'] = '#8d8cc4'
        
        return style
    
    def get_bounds(self) -> BoundsType:
        """Get 3D bounding box."""
        if self.mesh is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return self.mesh.bounds
    
    def is_solid(self) -> bool:
        """Point clouds are not solid surfaces."""
        return False
    
    def supports_index_mapping(self) -> bool:
        """Point clouds support visibility index mapping."""
        return True
    
    def get_element_type(self) -> ElementType:
        """Element type is 'point'."""
        return 'point'
    
    def get_element_count(self) -> int:
        """Number of points."""
        if self.mesh is None:
            return 0
        return self.mesh.n_points
    
    # --------------------------------------------------------------------------
    # Point Cloud Specific Methods
    # --------------------------------------------------------------------------
    
    def get_mesh(self) -> pv.PolyData:
        """
        Returns the underlying PyVista mesh.
        
        Backward-compatible alias for get_render_mesh().
        """
        return self.mesh
    
    def get_points_array(self) -> Optional[np.ndarray]:
        """
        Get the raw point coordinates as a numpy array for efficient processing.
        
        Returns:
            np.ndarray: (N, 3) array of point coordinates, or None if no mesh
        """
        if self.mesh is None:
            return None
        return self.mesh.points
    
    def has_rgb(self) -> bool:
        """Check if point cloud has RGB color data."""
        return 'RGB' in self.array_names
    
    def has_normals(self) -> bool:
        """Check if point cloud has normal vectors."""
        return 'Normals' in self.array_names or 'normals' in self.array_names

    def get_element_coordinate(self, element_id: int):
        """Return the 3D world coordinate of a single point by its sequential ID."""
        if self.mesh is None:
            return None
        pts = self.mesh.points
        if element_id < 0 or element_id >= len(pts):
            return None
        return pts[element_id].astype(np.float64)

    # --------------------------------------------------------------------------
    # Array Management for Visualization
    # --------------------------------------------------------------------------

    def get_available_arrays(self) -> list:
        """Get all available arrays for visualization (including 'Labels')."""
        return self.available_arrays
    
    def get_selected_array(self) -> str:
        """Get the currently selected array for visualization."""
        return self.selected_array
    
    def set_selected_array(self, array_name: str) -> bool:
        """
        Set the array to use for visualization.
        
        Args:
            array_name: Name of the array to select (must be in available_arrays)
            
        Returns:
            True if successful, False if array not found
        """
        if array_name not in self.available_arrays:
            print(f"⚠️ Array '{array_name}' not available in {self.label}")
            return False
        
        self.selected_array = array_name
        print(f"✓ Selected array '{array_name}' for {self.label}")
        return True


class MeshProduct(AbstractSceneProduct):
    """
    Scene product for surface mesh data.
    
    Wraps PyVista mesh with faces (triangles/polygons) for solid surface rendering
    and accurate occlusion testing. Meshes ARE solid (is_solid=True).
    
    Attributes:
        mesh (pv.PolyData): The underlying PyVista surface mesh.
        opacity (float): Default rendering opacity.
    """
    
    def __init__(self, file_path: str, opacity: float = 1.0, product_id: Optional[str] = None):
        """
        Initialize MeshProduct from file.
        
        Args:
            file_path: Path to mesh file (.obj, .stl, .ply, .vtk)
            opacity: Default rendering opacity (0.0-1.0)
            product_id: Optional unique ID (defaults to filename)
        """
        if product_id is None:
            product_id = os.path.basename(file_path)
        
        super().__init__(product_id=product_id, file_path=file_path)
        
        self.opacity = opacity
        self.mesh: Optional[pv.PolyData] = None
        self.array_names = []
        self.available_arrays = []  # Will be built after loading
        self.selected_array = "RGB"  # Default to RGB
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        
        # Build available arrays in priority order:
        # 1. RGB (first, defaults to Metashape purple if not in data)
        # 2. Labels (always available, all white by default)
        # 3. Everything else from the mesh
        other_arrays = [arr for arr in self.array_names if arr.lower() not in ('rgb', 'labels')]
        self.available_arrays = ["RGB", "Labels"] + other_arrays
        
        print(f"Array names in mesh: {self.array_names}")
        print(f"Available arrays for visualization (priority order): {self.available_arrays}")
        
        load_time = time.time() - start_time
        
        # Validate it has cells (faces/triangles)
        if self.mesh.n_cells == 0:
            raise ValueError(f"File '{file_path}' has no cells/faces - use PointCloudProduct instead")
        
        print(f"⏱️ Loaded MeshProduct: {self.label} with {self.mesh.n_cells:,} faces in {load_time:.3f}s")
    
    @classmethod
    def from_file(cls, file_path: str, opacity: float = 1.0) -> 'MeshProduct':
        """Load a mesh from file."""
        return cls(file_path=file_path, opacity=opacity)
    
    # Custom method to build and cache Open3D RaycastingScene for this mesh
    def prepare_geometry(self):
        if hasattr(self, '_cached_triangles_pt'):
            return

        print(f"📐 Extracting raw geometry arrays for {self.label}...")
        import time
        import numpy as np
        import torch
        
        start_time = time.time()

        if not self.mesh.is_all_triangles:
            tri_mesh = self.mesh.triangulate()
            raw_ids = tri_mesh.cell_data.get('vtkOriginalCellIds', None)
            self._original_cell_ids = np.asarray(raw_ids) if raw_ids is not None else None
        else:
            tri_mesh = self.mesh
            self._original_cell_ids = None

        # Keep vertices in numpy for Open3D
        self._cached_vertices = np.asarray(tri_mesh.points, dtype=np.float32)
        
        # Extract the rest to numpy temporarily
        triangles_np = np.asarray(tri_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.uint32)
        centers_np = np.asarray(tri_mesh.cell_centers().points, dtype=np.float32)
        centers_sq_norm_np = np.sum(centers_np**2, axis=1)
        
        # Cache geometry arrays in PyTorch tensors for fast GPU processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        self._cached_face_centers_pt = torch.tensor(centers_np, device=self.device)
        self._cached_centers_sq_norm_pt = torch.tensor(centers_sq_norm_np, device=self.device)
        self._cached_triangles_pt = torch.tensor(triangles_np.astype(np.int64), device=self.device)
        
        if self._original_cell_ids is not None:
            self._original_cell_ids_pt = torch.tensor(self._original_cell_ids.astype(np.int64), device=self.device)
            # Index map stores original (pre-triangulation) cell IDs; use original mesh cell centers
            self._element_centers_np = np.asarray(self.mesh.cell_centers().points, dtype=np.float32)
        else:
            # Index map stores triangulated face IDs
            self._element_centers_np = centers_np.copy()
        
        print(f"✅ Geometry extracted and cached in {time.time() - start_time:.3f}s")
    
    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------
    
    def get_render_mesh(self) -> pv.PolyData:
        """Get the PyVista mesh for rendering."""
        return self.mesh
    
    def get_render_style(self) -> RenderStyle:
        """Get preferred rendering style based on selected array."""
        style = {
            'style': 'surface',
            'opacity': self.opacity,
            'show_edges': False,
            'lighting': True,
        }
        
        # Apply rendering based on selected array
        if self.selected_array == "RGB":
            # RGB: use actual RGB if available, otherwise Metashape purple as default
            if 'RGB' in self.array_names:
                style['scalars'] = 'RGB'
                style['rgb'] = True
            else:
                # Default Metashape purple color when no RGB vertex colors
                style['color'] = '#8d8cc4'
        elif self.selected_array == "Labels":
            # Labels: render as white meshes (all same color)
            style['color'] = 'white'
        elif self.selected_array in self.array_names:
            # Use selected array as scalars (applies colormap)
            style['scalars'] = self.selected_array
        else:
            # Fallback to Metashape purple
            style['color'] = '#8d8cc4'
        
        return style
    
    def get_bounds(self) -> BoundsType:
        """Get 3D bounding box."""
        if self.mesh is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return self.mesh.bounds
    
    def is_solid(self) -> bool:
        """Meshes are solid surfaces."""
        return True
    
    def supports_index_mapping(self) -> bool:
        """Meshes support face-based index mapping."""
        return True
    
    def get_element_type(self) -> ElementType:
        """Element type is 'face'."""
        return 'face'
    
    def get_element_count(self) -> int:
        """Number of faces/cells."""
        if self.mesh is None:
            return 0
        return self.mesh.n_cells
    
    # --------------------------------------------------------------------------
    # Mesh Specific Methods
    # --------------------------------------------------------------------------
    
    def get_mesh(self) -> pv.PolyData:
        """Returns the underlying PyVista mesh."""
        return self.mesh
    
    def get_face_centers(self) -> np.ndarray:
        """
        Get the center points of all faces.
        
        Returns:
            np.ndarray: (N_faces, 3) array of face center coordinates.
        """
        return self.mesh.cell_centers().points
    
    def get_face_normals(self) -> np.ndarray:
        """
        Get normal vectors for all faces.
        
        Returns:
            np.ndarray: (N_faces, 3) array of face normal vectors.
        """
        self.mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        return self.mesh.cell_data['Normals']

    def get_element_coordinate(self, element_id: int):
        """Return the 3D center coordinate of a mesh face by its ID."""
        if not hasattr(self, '_element_centers_np') or self._element_centers_np is None:
            self.prepare_geometry()
        centers = self._element_centers_np
        if centers is None or element_id < 0 or element_id >= len(centers):
            return None
        return centers[element_id].astype(np.float64)

    # --------------------------------------------------------------------------
    # Array Management for Visualization
    # --------------------------------------------------------------------------

    def get_available_arrays(self) -> list:
        """Get all available arrays for visualization (including 'Labels')."""
        return self.available_arrays
    
    def get_selected_array(self) -> str:
        """Get the currently selected array for visualization."""
        return self.selected_array
    
    def set_selected_array(self, array_name: str) -> bool:
        """
        Set the array to use for visualization.
        
        Args:
            array_name: Name of the array to select (must be in available_arrays)
            
        Returns:
            True if successful, False if array not found
        """
        if array_name not in self.available_arrays:
            print(f"⚠️ Array '{array_name}' not available in {self.label}")
            return False
        
        self.selected_array = array_name
        print(f"✓ Selected array '{array_name}' for {self.label}")
        return True

    
