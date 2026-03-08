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
        
        print(f"✅ Geometry extracted and cached in {time.time() - start_time:.3f}s")
    
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
    
    Acts as a 3D wrapper around the OrthographicCamera. It has been upgraded 
    to behave exactly like a MeshProduct, outputting explicit triangles and faces 
    for dense, watertight visibility raycasting, and natively supports orthomosaic draping.
    """
    
    def __init__(self, ortho_camera, opacity: float = 1.0, product_id: Optional[str] = None):
        """
        Initialize DEMProduct from an OrthographicCamera.
        """
        self.camera = ortho_camera
        
        if product_id is None:
            product_id = f"Elevation_{self.camera.label}"
            
        # Use the z_channel_path as the file_path for cache keys, fallback to image path
        file_path = self.camera._raster.z_channel_path or self.camera.image_path
        
        super().__init__(product_id=product_id, file_path=file_path)
        
        self.opacity = opacity
        self._mesh: Optional[pv.PolyData] = None
        self._texture: Optional[pv.Texture] = None  # Cache for the draped imagery
        
        print(f"🎬 Initialized dynamic DEMProduct from camera: {self.camera.label}")

    @classmethod
    def from_file(cls, file_path: str, opacity: float = 1.0):
        """
        Overrides the legacy file loader. DEMs must now be loaded via an OrthographicCamera
        to ensure perfect coordinate alignment. 
        """
        raise NotImplementedError(
            "Standalone DEM loading is deprecated. Please attach the DEM to an "
            "orthomosaic raster to ensure perfect 3D alignment."
        )

    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------
    
    def get_render_mesh(self) -> pv.PolyData:
        """
        Get PyVista PolyData for rendering.
        Lazily asks the camera to generate the smooth elevation mesh on first access.
        """
        if self._mesh is None:
            self._mesh = self.camera.get_elevation_mesh()
        return self._mesh
    
    def prepare_geometry(self):
        """
        Extracts raw geometry into PyTorch tensors for the GPU visibility engine.
        This makes the DEM physically identical to a loaded MeshProduct!
        """
        if hasattr(self, '_cached_triangles_pt'):
            return

        print(f"📐 Extracting DEM faces into GPU tensors for {self.label}...")
        import time
        import numpy as np
        import torch
        
        start_time = time.time()
        
        mesh = self.get_render_mesh()
        if mesh is None:
            return

        # Keep vertices in numpy for Open3D
        self._cached_vertices = np.asarray(mesh.points, dtype=np.float32)
        
        # Extract faces (PyVista pads faces with the number of points, so we slice [:, 1:])
        triangles_np = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.uint32)
        centers_np = np.asarray(mesh.cell_centers().points, dtype=np.float32)
        centers_sq_norm_np = np.sum(centers_np**2, axis=1)
        
        # Cache geometry arrays in PyTorch tensors for fast GPU processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        self._cached_face_centers_pt = torch.tensor(centers_np, device=self.device)
        self._cached_centers_sq_norm_pt = torch.tensor(centers_sq_norm_np, device=self.device)
        self._cached_triangles_pt = torch.tensor(triangles_np.astype(np.int64), device=self.device)
        
        # Fulfills the PyTorch GPU raycaster contract
        self._original_cell_ids = None 
        self._original_cell_ids_pt = None 
        
        print(f"✅ DEM Geometry cached in {time.time() - start_time:.3f}s")

    def get_render_style(self) -> RenderStyle:
        """Get preferred rendering style with the safely draped orthomosaic texture."""
        style = {
            'style': 'surface',
            'opacity': self.opacity,
            'show_edges': False,
            'lighting': True,
            'scalars': None, # CRITICAL: Force PyVista to ignore any lingering data arrays
        }
        
        # Check if the mesh has texture coordinates before attempting to apply texture
        mesh = self.get_render_mesh()
        
        # Robustly check for texture coordinates
        has_tcoords = False
        if mesh is not None:
            try:
                tcoords = mesh.active_texture_coordinates
                has_tcoords = (tcoords is not None and len(tcoords) > 0)
            except Exception as e:
                print(f"⚠️ Error checking texture coordinates for {self.label}: {e}")
                has_tcoords = False
        
        if not has_tcoords:
            # No texture coordinates - use fallback color
            print(f"⚠️ DEM {self.label} has no texture coordinates, using fallback color")
            style['color'] = '#8d8cc4'
            return style
        
        if self._texture is None:
            try:
                import pyvista as pv
                from coralnet_toolbox.utilities import pixmap_to_numpy
                
                # Cap the texture size to 4096 to prevent GPU crashes
                pixmap = self.camera._raster.get_pixmap(longest_edge=4096)
                
                if pixmap is not None and not pixmap.isNull():
                    img_data = pixmap_to_numpy(pixmap)
                    
                    # CRITICAL: Flip the texture vertically to match UV coordinate convention
                    # OpenGL textures have origin at bottom-left, but images have origin at top-left
                    # Our UV coordinates are computed with v = 1.0 - (pixel_y / height)
                    # so the texture must be flipped to align properly
                    img_data = np.flipud(img_data).copy()
                    
                    self._texture = pv.Texture(img_data)
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not create texture for DEM {self.label}: {e}")
        
        if self._texture is not None:
            style['texture'] = self._texture
            style['color'] = 'white' # Prevents the texture from being tinted by a default color
        else:
            style['color'] = '#8d8cc4'  # Fallback to standard mesh purple
            
        return style
    
    def get_bounds(self) -> BoundsType:
        """Get 3D bounding box directly from the generated mesh."""
        mesh = self.get_render_mesh()
        if mesh is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return mesh.bounds
    
    def is_solid(self) -> bool:
        """DEMs are solid triangulated surfaces."""
        return True
    
    def supports_index_mapping(self) -> bool:
        """DEMs support face-based index mapping."""
        return True
    
    def get_element_type(self) -> ElementType:
        """Element type is 'face' (tells the engine to treat this as a solid mesh!)."""
        return 'face'
    
    def get_element_count(self) -> int:
        """Number of faces."""
        mesh = self.get_render_mesh()
        if mesh is None:
            return 0
        return mesh.n_cells

    # --------------------------------------------------------------------------
    # Backward Compatibility & Legacy Interfaces
    # --------------------------------------------------------------------------
    
    def get_points_array(self) -> Optional[np.ndarray]:
        """Return raw vertices to satisfy legacy point-based interfaces."""
        mesh = self.get_render_mesh()
        return np.asarray(mesh.points) if mesh else None
        
    def get_face_centers(self) -> Optional[np.ndarray]:
        """Return physical triangle centers."""
        mesh = self.get_render_mesh()
        return np.asarray(mesh.cell_centers().points) if mesh else None
        
    def get_face_normals(self) -> Optional[np.ndarray]:
        """Calculate and return surface normals for the terrain."""
        mesh = self.get_render_mesh()
        if mesh is None: 
            return None
        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        return mesh.cell_data['Normals']

    @property
    def elevation(self):
        """Backward compatibility for legacy code looking for the raw DEM array."""
        return self.camera.z_channel
        
    @property
    def transform(self):
        """Backward compatibility for legacy code looking for the matrix."""
        return self.camera.transform_matrix


# ----------------------------------------------------------------------------------------------------------------------
# Backward Compatibility Aliases
# ----------------------------------------------------------------------------------------------------------------------

# PointCloud is now an alias for PointCloudProduct
PointCloud = PointCloudProduct
    
