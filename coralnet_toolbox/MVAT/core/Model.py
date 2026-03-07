"""
3D Scene Product Implementations for MVAT

Provides concrete implementations of AbstractSceneProduct for:
- PointCloudProduct: Point cloud data (.ply, .pcd, etc.)
- MeshProduct: Surface meshes with faces (.obj, .stl, .ply with faces)
- DEMProduct: Digital Elevation Models (GeoTIFF rasters)

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
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        load_time = time.time() - start_time
        
        print(f"⏱️ Loaded PointCloudProduct: {self.label} with {self.mesh.n_points:,} points in {load_time:.3f}s")
    
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
        """Get preferred rendering style."""
        style = {
            'style': 'points',
            'point_size': self.point_size,
            'render_points_as_spheres': False,
            'lighting': False,
        }
        # Use RGB scalars if available
        if 'RGB' in self.array_names:
            style['scalars'] = 'RGB'
            style['rgb'] = True
        else:
            style['color'] = 'black'
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


class MeshProduct(AbstractSceneProduct):
    """
    Scene product for surface mesh data.
    
    Wraps PyVista mesh with faces (triangles/polygons) for solid surface rendering
    and accurate occlusion testing. Meshes ARE solid (is_solid=True).
    
    Attributes:
        mesh (pv.PolyData): The underlying PyVista surface mesh.
        opacity (float): Default rendering opacity.
    """
    
    def __init__(self, file_path: str, opacity: float = 0.8, product_id: Optional[str] = None):
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
        
        # Load from file with timing
        start_time = time.time()
        self.mesh = pv.read(file_path, progress_bar=True)
        self.array_names = self.mesh.array_names
        print(f"Array names in mesh: {self.array_names}")
        
        # Attributes to hold the cached data
        self._o3d_scene = None
        self._original_cell_ids = None
        
        load_time = time.time() - start_time
        
        # Validate it has cells (faces/triangles)
        if self.mesh.n_cells == 0:
            raise ValueError(f"File '{file_path}' has no cells/faces - use PointCloudProduct instead")
        
        print(f"⏱️ Loaded MeshProduct: {self.label} with {self.mesh.n_cells:,} faces in {load_time:.3f}s")
    
    @classmethod
    def from_file(cls, file_path: str, opacity: float = 0.8) -> 'MeshProduct':
        """Load a mesh from file."""
        return cls(file_path=file_path, opacity=opacity)
    
    # Custom method to build and cache Open3D RaycastingScene for this mesh
    def get_o3d_scene(self):
        """
        Lazily builds and caches the Open3D RaycastingScene.
        This ensures triangulation and BVH building only happens ONCE per file.
        """
        if self._o3d_scene is not None:
            return self._o3d_scene, self._original_cell_ids

        import open3d as o3d
        import time
        import numpy as np
        
        start_time = time.time()
        
        # 1. Triangulate
        if not self.mesh.is_all_triangles:
            tri_mesh = self.mesh.triangulate()
            # 🔥 FIX: Wrap in np.asarray to prevent slow VTK indexing loops later!
            raw_ids = tri_mesh.cell_data.get('vtkOriginalCellIds', None)
            self._original_cell_ids = np.asarray(raw_ids) if raw_ids is not None else None
        else:
            tri_mesh = self.mesh
            self._original_cell_ids = None

        # 2. Extract Geometry
        vertices = np.asarray(tri_mesh.points, dtype=np.float32)
        triangles = np.asarray(tri_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.uint32)

        # 3. Add to Open3D
        self._o3d_scene = o3d.t.geometry.RaycastingScene()
        v_tensor = o3d.core.Tensor(vertices)
        t_tensor = o3d.core.Tensor(triangles)
        self._o3d_scene.add_triangles(v_tensor, t_tensor)
        
        # 🔥 THE FIX: Open3D is "lazy". It doesn't actually build the BVH until the first ray is cast.
        # We cast a single dummy ray right now to force it to build the spatial index immediately.
        # Format: [px, py, pz, dx, dy, dz]
        dummy_ray = o3d.core.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=o3d.core.Dtype.Float32)
        self._o3d_scene.cast_rays(dummy_ray)
        
        print(f"🎯 Built and cached Open3D BVH for {tri_mesh.n_cells:,} faces in {time.time() - start_time:.3f}s")
        return self._o3d_scene, self._original_cell_ids
    
    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------
    
    def get_render_mesh(self) -> pv.PolyData:
        """Get the PyVista mesh for rendering."""
        return self.mesh
    
    def get_render_style(self) -> RenderStyle:
        """Get preferred rendering style."""
        style = {
            'style': 'surface',
            'opacity': self.opacity,
            'show_edges': False,
            'lighting': True,
        }
        if 'RGB' in self.array_names:
            style['scalars'] = 'RGB'
            style['rgb'] = True
        else:
            # Default Metashape purple color when no vertex colors
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


class DEMProduct(AbstractSceneProduct):
    """
    Scene product for Digital Elevation Model (DEM) data.
    
    Handles georeferenced raster elevation data (GeoTIFF) and provides 
    visualization as a PyVista StructuredGrid surface.
    
    DEMs ARE solid (is_solid=True) and support cell-based index mapping where
    each cell corresponds to a pixel/grid position in the elevation raster.
    
    Attributes:
        elevation (np.ndarray): 2D elevation array (height x width).
        transform (np.ndarray): Affine transform matrix (3x3) for geo-projection.
        nodata (float): NoData value for invalid cells.
        crs: Coordinate reference system information.
    """
    
    def __init__(self, 
                 file_path: str,
                 opacity: float = 0.8,
                 product_id: Optional[str] = None):
        """
        Initialize DEMProduct from GeoTIFF file.
        
        Args:
            file_path: Path to GeoTIFF elevation file.
            opacity: Default rendering opacity.
            product_id: Optional unique ID (defaults to filename).
        """
        if product_id is None:
            product_id = os.path.basename(file_path)
        
        super().__init__(product_id=product_id, file_path=file_path)
        
        self.opacity = opacity
        self.elevation: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.transform_inv: Optional[np.ndarray] = None
        self.nodata: float = np.nan
        self.crs = None
        self._structured_grid: Optional[pv.StructuredGrid] = None
        
        # Load the DEM
        self._load_from_file(file_path)
    
    def _load_from_file(self, file_path: str) -> None:
        """Load DEM from GeoTIFF file."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for DEMProduct - install with: pip install rasterio")
        
        start_time = time.time()
        
        with rasterio.open(file_path) as src:
            # Read elevation data (first band)
            self.elevation = src.read(1).astype(np.float32)
            
            # Get affine transform
            self.transform = np.array([
                [src.transform.a, src.transform.b, src.transform.c],
                [src.transform.d, src.transform.e, src.transform.f],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Compute inverse transform
            self.transform_inv = np.linalg.inv(self.transform)
            
            # Store nodata and CRS
            self.nodata = src.nodata if src.nodata is not None else np.nan
            self.crs = src.crs
            
            # Get bounds
            self._bounds = src.bounds
        
        # Replace nodata with NaN for consistency
        if not np.isnan(self.nodata):
            self.elevation[self.elevation == self.nodata] = np.nan
        
        load_time = time.time() - start_time
        height, width = self.elevation.shape
        print(f"⏱️ Loaded DEMProduct: {self.label} ({width}x{height}) in {load_time:.3f}s")
    
    @classmethod
    def from_file(cls, file_path: str, opacity: float = 0.8) -> 'DEMProduct':
        """Load a DEM from file."""
        return cls(file_path=file_path, opacity=opacity)
    
    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------
    
    def get_render_mesh(self) -> pv.StructuredGrid:
        """
        Get PyVista StructuredGrid for rendering.
        
        Lazily generates the grid on first access.
        """
        if self._structured_grid is None:
            self._structured_grid = self._create_structured_grid()
        return self._structured_grid
    
    def _create_structured_grid(self) -> pv.StructuredGrid:
        """Create PyVista StructuredGrid from elevation array."""
        height, width = self.elevation.shape
        
        # Create coordinate grids
        cols = np.arange(width)
        rows = np.arange(height)
        xx, yy = np.meshgrid(cols, rows)
        
        # Transform pixel coordinates to world coordinates
        # world_coords = transform @ [col, row, 1]^T
        x_world = self.transform[0, 0] * xx + self.transform[0, 1] * yy + self.transform[0, 2]
        y_world = self.transform[1, 0] * xx + self.transform[1, 1] * yy + self.transform[1, 2]
        z_world = self.elevation.copy()
        
        # Replace NaN with 0 for grid creation (will be masked in rendering)
        z_world_clean = np.nan_to_num(z_world, nan=0.0)
        
        # Create StructuredGrid
        grid = pv.StructuredGrid(x_world, y_world, z_world_clean)
        
        # Add elevation as scalar data for coloring
        grid.point_data['Elevation'] = z_world.flatten()
        
        return grid
    
    def get_render_style(self) -> RenderStyle:
        """Get preferred rendering style."""
        return {
            'style': 'surface',
            'opacity': self.opacity,
            'scalars': 'Elevation',
            'cmap': 'terrain',
            'show_edges': False,
            'lighting': True,
        }
    
    def get_bounds(self) -> BoundsType:
        """Get 3D bounding box (in world coordinates)."""
        if self.elevation is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # X and Y bounds from rasterio bounds
        xmin, ymin = self._bounds.left, self._bounds.bottom
        xmax, ymax = self._bounds.right, self._bounds.top
        
        # Z bounds from elevation data
        z_valid = self.elevation[~np.isnan(self.elevation)]
        if len(z_valid) == 0:
            zmin, zmax = 0.0, 0.0
        else:
            zmin, zmax = float(z_valid.min()), float(z_valid.max())
        
        return (xmin, xmax, ymin, ymax, zmin, zmax)
    
    def is_solid(self) -> bool:
        """DEMs are solid surfaces."""
        return True
    
    def supports_index_mapping(self) -> bool:
        """DEMs support cell-based index mapping."""
        return True
    
    def get_element_type(self) -> ElementType:
        """Element type is 'cell' (grid cells)."""
        return 'cell'
    
    def get_element_count(self) -> int:
        """Number of grid cells (width * height)."""
        if self.elevation is None:
            return 0
        return self.elevation.size
    
    # --------------------------------------------------------------------------
    # DEM Specific Methods
    # --------------------------------------------------------------------------
    
    def pixel_to_world(self, u: int, v: int) -> tuple:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            u: Column (x pixel coordinate)
            v: Row (y pixel coordinate)
            
        Returns:
            (x, y, z) world coordinates
        """
        x = self.transform[0, 0] * u + self.transform[0, 1] * v + self.transform[0, 2]
        y = self.transform[1, 0] * u + self.transform[1, 1] * v + self.transform[1, 2]
        
        if 0 <= v < self.elevation.shape[0] and 0 <= u < self.elevation.shape[1]:
            z = self.elevation[v, u]
        else:
            z = np.nan
        
        return (x, y, z)
    
    def world_to_pixel(self, x: float, y: float) -> tuple:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            
        Returns:
            (u, v) pixel coordinates (column, row)
        """
        point = np.array([x, y, 1.0])
        pixel = self.transform_inv @ point
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        return (u, v)
    
    def get_elevation_at(self, x: float, y: float) -> float:
        """
        Get elevation at world coordinates.
        
        Args:
            x: World X coordinate
            y: World Y coordinate
            
        Returns:
            Elevation value or NaN if outside bounds/nodata.
        """
        u, v = self.world_to_pixel(x, y)
        if 0 <= v < self.elevation.shape[0] and 0 <= u < self.elevation.shape[1]:
            return float(self.elevation[v, u])
        return np.nan
    
    @property
    def width(self) -> int:
        """DEM width in pixels."""
        return self.elevation.shape[1] if self.elevation is not None else 0
    
    @property
    def height(self) -> int:
        """DEM height in pixels."""
        return self.elevation.shape[0] if self.elevation is not None else 0


# ----------------------------------------------------------------------------------------------------------------------
# Backward Compatibility Aliases
# ----------------------------------------------------------------------------------------------------------------------

# PointCloud is now an alias for PointCloudProduct
PointCloud = PointCloudProduct
    
