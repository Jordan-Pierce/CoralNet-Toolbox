"""
3D Scene Products for MVAT

Defines the abstract product interface and all concrete implementations:
- AbstractSceneProduct: ABC that all 3D products must implement
- PointCloudProduct:    Point cloud data (.ply, .pcd, etc.)
- MeshProduct:          Surface meshes with faces (.obj, .stl, .ply with faces)
- GaussianSplattingProduct: 3D Gaussian Splatting scenes (.ply with 3DGS fields)

Backward Compatibility:
- PointCloud class is preserved as an alias for PointCloudProduct
"""
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pyvista as pv


# ----------------------------------------------------------------------------------------------------------------------
# Type Aliases
# ----------------------------------------------------------------------------------------------------------------------

ElementType = Literal['point', 'face', 'cell']
BoundsType = Tuple[float, float, float, float, float, float]  # xmin, xmax, ymin, ymax, zmin, zmax
RenderStyle = Dict[str, Union[str, int, float, bool]]


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AbstractSceneProduct(ABC):
    """
    Abstract base class for all 3D scene products.
    
    Defines the contract that point clouds, meshes, and DEMs must implement
    to participate in the MVAT scene rendering and visibility computation.
    
    Attributes:
        product_id (str): Unique identifier for this product instance.
        file_path (str): Path to the source file, if loaded from disk.
        label (str): Human-readable display name.
    """
    
    def __init__(self, product_id: str, file_path: Optional[str] = None, label: Optional[str] = None):
        """
        Initialize the base scene product.
        
        Args:
            product_id: Unique identifier for this product.
            file_path: Optional path to the source file.
            label: Optional display name (defaults to filename if file_path provided).
        """
        self.product_id = product_id
        self.file_path = file_path
        
        if label is not None:
            self.label = label
        elif file_path is not None:
            import os
            self.label = os.path.basename(file_path)
        else:
            self.label = product_id
    
    # --------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # --------------------------------------------------------------------------
    
    @abstractmethod
    def get_render_mesh(self) -> Union[pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid]:
        """
        Get the PyVista mesh geometry for rendering.
        
        Returns:
            PyVista mesh object suitable for adding to a plotter.
        """
        pass
    
    @abstractmethod
    def get_render_style(self) -> RenderStyle:
        """
        Get the preferred rendering style for this product.
        
        Returns:
            Dictionary of PyVista add_mesh() keyword arguments, e.g.:
            {'style': 'points', 'point_size': 2, 'render_points_as_spheres': False}
            {'style': 'surface', 'opacity': 0.8, 'show_edges': False}
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> BoundsType:
        """
        Get the 3D bounding box of the product.
        
        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        pass
    
    @abstractmethod
    def is_solid(self) -> bool:
        """
        Whether this product represents a solid surface suitable for occlusion testing.
        
        Point clouds return False (require splatting for pseudo-occlusion).
        Meshes and DEMs return True.
        
        Returns:
            True if suitable for accurate occlusion/ray-casting tests.
        """
        pass
    
    @abstractmethod
    def supports_index_mapping(self) -> bool:
        """
        Whether this product can generate visibility index maps.
        
        Returns:
            True if the product has addressable elements that can be indexed.
        """
        pass
    
    @abstractmethod
    def get_element_type(self) -> ElementType:
        """
        Get the type of addressable elements in this product.
        
        Returns:
            'point' for point clouds (index = point ID)
            'face' for meshes (index = face/polygon ID)
            'cell' for DEMs/grids (index = grid cell linear index)
        """
        pass
    
    @abstractmethod
    def get_element_count(self) -> int:
        """
        Get the total number of addressable elements.
        
        Returns:
            Number of elements (points, faces, or cells).
        """
        pass

    @abstractmethod
    def get_element_coordinate(self, element_id: int) -> Optional[np.ndarray]:
        """
        Return the 3D world coordinate [X, Y, Z] for the given element as a (3,) float64 array.

        For point clouds: the point's XYZ.
        For meshes / DEMs: the face / cell center.

        Element IDs must match values stored in the visibility index_map.

        Returns:
            np.ndarray of shape (3,) or None if element_id is out of range or
            geometry is not yet loaded.
        """
        pass

    @abstractmethod
    def apply_labels(self, element_ids: np.ndarray, class_id: int, color_rgb: tuple) -> None:
        """
        Update the class ID and visual color for specific elements.
        
        Args:
            element_ids: 1D array of element indices (points, faces, or cells) to update.
            class_id: The semantic integer ID of the label.
            color_rgb: Tuple of (R, G, B) values (0-255) for visualization.
        """
        pass

    # --------------------------------------------------------------------------
    # Optional Methods - Override in subclasses if needed
    # --------------------------------------------------------------------------
    
    def get_label(self) -> str:
        """Get the human-readable display name."""
        return self.label
    
    def get_center(self) -> np.ndarray:
        """
        Get the center point of the product's bounding box.
        
        Returns:
            np.ndarray of shape (3,) with (x, y, z) center coordinates.
        """
        bounds = self.get_bounds()
        return np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
    
    def get_diagonal_length(self) -> float:
        """
        Get the diagonal length of the bounding box.
        
        Useful for computing scene scale factors.
        
        Returns:
            Length of the bounding box diagonal.
        """
        bounds = self.get_bounds()
        return np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
    
    def can_cast_rays(self) -> bool:
        """
        Whether ray-casting is supported for this product.
        
        Default implementation returns is_solid(). Override if different.
        
        Returns:
            True if ray-mesh intersection is supported.
        """
        return self.is_solid()
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id='{self.product_id}', "
                f"label='{self.label}', elements={self.get_element_count():,})")


import os
import time
from typing import Optional

import numpy as np
import pyvista as pv


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
    
    def __init__(self, file_path: str, point_size: int = 1, product_id: Optional[str] = None,
                 sort_data: bool = True, simplification_ratio: float = 0.0):
        """
        Initialize PointCloudProduct from file.

        Args:
            file_path: Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size: Size of points when rendered
            product_id: Optional unique ID (defaults to filename)
            sort_data: When True, spatially sort points via Morton Z-order after
                       loading so point IDs are spatially coherent (lower index-map
                       entropy → better cache compression). Mirrors MeshProduct.
            simplification_ratio: Fraction of points to *remove* at load via
                       uniform random decimation (0.0 = keep all, 0.9 = keep 10%).
                       Produces a lighter "proxy" cloud used for all visualization
                       and index-map creation, mirroring MeshProduct decimation.
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

        # Step 1: optional decimation (fraction of points removed). Mirrors the
        # mesh proxy: a single lighter cloud used for everything downstream.
        if simplification_ratio and simplification_ratio > 0.0 and self.mesh is not None:
            self.mesh = self._simplify_point_cloud(self.mesh, float(simplification_ratio))

        # Step 2: optional spatial sort (before scalar synthesis so every per-point
        # array, including the ones we are about to create, ends up in sorted order).
        if sort_data and self.mesh is not None and self.mesh.n_points > 1:
            self.mesh = self._spatially_sort_point_cloud(self.mesh)
            print(f"⏱️ Spatially sorted point cloud for {self.label} in {time.time() - start_time:.3f}s")

        self.array_names = self.mesh.array_names
        
        # Synthesize missing scalar arrays for consistent visualization
        self._ensure_scalar_arrays()
        
        # Build available arrays in priority order
        other_arrays = [arr for arr in self.array_names if arr.lower() not in ('rgb', 'labels', 'normals', 'class_ids')]
        self.available_arrays = ["RGB", "Labels"]
        self.available_arrays.extend(other_arrays)
        
        load_time = time.time() - start_time
        
        print(f"⏱️ Loaded PointCloudProduct: {self.label} with {self.mesh.n_points:,} points in {load_time:.3f}s")
        print(f"   Available arrays for visualization: {self.available_arrays}")
    
    @classmethod
    def from_file(cls, file_path: str, point_size: int = 1,
                  sort_data: bool = True, simplification_ratio: float = 0.0) -> 'PointCloudProduct':
        """
        Load a point cloud from a file.

        Args:
            file_path: Path to 3D file (.ply, .stl, .obj, .vtk, .pcd)
            point_size: Size of points when rendered
            sort_data: When True, Morton-sort the points at load (see __init__).
            simplification_ratio: Fraction of points to remove at load (see __init__).

        Returns:
            PointCloudProduct instance
        """
        return cls(file_path=file_path, point_size=point_size,
                   sort_data=sort_data, simplification_ratio=simplification_ratio)

    def _simplify_point_cloud(self, mesh: 'pv.PolyData', ratio: float) -> 'pv.PolyData':
        """Uniformly decimate the cloud by removing a fraction of points.

        Point-cloud analogue of mesh face decimation: keeps ``1 - ratio`` of the
        points (chosen uniformly at random with a fixed seed for reproducibility)
        and carries every per-point array across so RGB / Normals stay aligned.
        Random sampling preserves the cloud's overall density distribution; a
        voxel-grid downsample could be swapped in later for uniform spacing.
        """
        ratio = float(np.clip(ratio, 0.0, 0.999))
        if ratio <= 0.0:
            return mesh

        pts = np.asarray(mesh.points)
        n = pts.shape[0]
        keep = max(1, int(round(n * (1.0 - ratio))))
        if keep >= n:
            return mesh

        start_time = time.time()
        rng = np.random.default_rng(0)
        # Sorted indices keep the relative point order stable (the Morton sort
        # reorders afterwards anyway, but this keeps un-sorted clouds tidy).
        idx = np.sort(rng.choice(n, size=keep, replace=False))

        out = pv.PolyData(pts[idx])
        for name in list(mesh.point_data.keys()):
            out.point_data[name] = np.asarray(mesh.point_data[name])[idx]

        print(f"⏱️ Decimated point cloud {n:,} → {keep:,} points "
              f"({ratio:.0%} removed) in {time.time() - start_time:.3f}s")
        return out

    def _spatially_sort_point_cloud(self, mesh: 'pv.PolyData') -> 'pv.PolyData':
        """Spatially sort points along a Morton (Z-order) curve.

        Point-cloud analogue of MeshProduct._spatially_sort_mesh. Reorders the
        point coordinates AND every per-point data array by the same permutation
        so a point's sequential ID (gl_VertexID) becomes spatially coherent.

        Benefit: adjacent pixels in a rendered index map then reference points
        with numerically close IDs, which lowers the index map's local entropy
        and markedly improves DEFLATE (.npz) compression of the cached visibility
        maps and visible-index arrays. Unlike meshes there is no shared-vertex
        cache to exploit (GL_POINTS draws each point once), so this is a storage/
        I-O optimization rather than a rasterization-speed one.
        """
        pts = np.asarray(mesh.points)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return mesh

        # Normalize XYZ into a 10-bit integer grid (0..1023).
        c_min = pts.min(axis=0)
        c_max = pts.max(axis=0)
        extent = np.maximum(c_max - c_min, 1e-8)
        normalized = ((pts - c_min) / extent * 1023).astype(np.uint32)
        x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

        # Interleave X, Y, Z bits into a 30-bit Morton code (same as the mesh path).
        def expand_bits(v):
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v

        morton_codes = (expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2))
        sort_idx = np.argsort(morton_codes)

        # Rebuild a points-only PolyData and carry every per-point array across in
        # the SAME order — otherwise RGB / Normals / Labels would be scrambled.
        sorted_cloud = pv.PolyData(pts[sort_idx])
        for name in list(mesh.point_data.keys()):
            sorted_cloud.point_data[name] = np.asarray(mesh.point_data[name])[sort_idx]

        # Keep the permutation for any future proof/debug visualization.
        self._sort_idx = sort_idx
        return sorted_cloud
    
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
        
        # All arrays (RGB, Labels, and data arrays) are now real scalars in the mesh
        # Use them directly via the mapper
        if self.selected_array in self.array_names:
            style['scalars'] = self.selected_array
            # RGB, Labels, and Normals_RGB are all Nx3 uint8, so they need direct RGB mode
            if self.selected_array in ("RGB", "Labels", "Normals_RGB"):
                style['rgb'] = True
        else:
            # Fallback: render as metashape purple
            style['color'] = '#8d8cc4'  # Metashape purple
        
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

    def prepare_geometry(self):
        """Populate the element-center cache used by the spatial (KD-tree) query stack.

        For a point cloud the "elements" are the points themselves, so the element
        centers are simply the point coordinates. This mirrors MeshProduct.prepare_geometry
        so the shared brush-volume / hover-overlay machinery (which only consumes
        ``_element_centers_np``) works for point clouds without any special-casing.
        """
        if self.mesh is None:
            self._element_centers_np = None
            return
        self._element_centers_np = np.asarray(self.mesh.points, dtype=np.float32)
    
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
        """
        if array_name not in self.available_arrays:
            print(f"⚠️ Array '{array_name}' not available in {self.label}")
            return False
        
        self.selected_array = array_name
        
        # FIX: Explicitly force the underlying PyVista mesh to switch active scalars
        if self.mesh is not None and array_name in self.array_names:
            try:
                # Point clouds use point_data, so we set the preference to point
                self.mesh.set_active_scalars(array_name, preference='point')
            except Exception:
                pass
                
        return True
    
    def _ensure_scalar_arrays(self):
        """
        Ensure required scalar arrays exist in the mesh.
        """
        if self.mesh is None:
            return
        
        n_points = self.mesh.n_points

        labels_preexisting = "Labels" in self.mesh.array_names
        if not labels_preexisting:
            labels_array = np.ones((n_points, 3), dtype=np.uint8) * 255  # White
            self.mesh.point_data["Labels"] = labels_array
        # Create a Python-owned cache detached from the VTK array for background painting
        # This avoids writing into VTK memory from worker threads.
        self._labels_cache = np.asarray(self.mesh.point_data["Labels"]).copy()
        # Deferred-flush bookkeeping: dirty = cache diverged from the VTK array;
        # have_paint = the VTK array may contain non-white labels (so erasing
        # later requires a real flush to avoid stale colors under the overlay).
        self._labels_dirty = False
        self._vtk_labels_have_paint = labels_preexisting

        # Keep a readonly reference to the PyVista view for eventual flush, but
        # do NOT write to it directly during painting.
        self._labels_view = self.mesh.point_data["Labels"]

        # --- THE FIX: Decouple Class_IDs from the GPU ---
        if not hasattr(self, 'class_ids'):
            self.class_ids = np.zeros(n_points, dtype=np.int32)
        
        if "RGB" not in self.mesh.array_names:
            rgb_array = np.ones((n_points, 3), dtype=np.uint8)
            rgb_array[:, 0] = 141  
            rgb_array[:, 1] = 140  
            rgb_array[:, 2] = 196  
            self.mesh.point_data["RGB"] = rgb_array

        self.array_names = self.mesh.array_names

    def apply_labels(self, element_ids: np.ndarray, class_id: int, color_rgb: tuple) -> None:
        """ Update the class ID and visual color for specific points. """
        if self.mesh is None or len(element_ids) == 0:
            return

        # Optimization: Instant Python-side numpy check
        if np.all(self.class_ids[element_ids] == class_id):
            return

        # 1. Update the semantic data in pure Python RAM
        self.class_ids[element_ids] = class_id

        # 2. Update the Python-owned labels cache (no VTK/GL work here)
        if not hasattr(self, '_labels_cache'):
            # Fallback: materialize cache if missing
            self._labels_cache = np.asarray(self.mesh.point_data["Labels"]).copy()
        self._labels_cache[element_ids] = color_rgb
        self._labels_dirty = True

        # NOTE: Do NOT call Modified() here. The expensive GPU upload is deferred
        # and performed once by `flush_labels_to_gpu()` on the main thread.

    def flush_labels_to_gpu(self) -> None:
        """One full GPU upload. Call only on the GUI/main thread, and only at
        rare barriers (scene rebuild, array switch, erase-after-flush) — this
        triggers a full VTK mapper rebuild on the next render."""
        if self.mesh is None or not hasattr(self, '_labels_cache'):
            return
        if not getattr(self, '_labels_dirty', True):
            return
        try:
            # Overwrite the PyVista/VTK array from our python cache, then mark modified
            self.mesh.point_data["Labels"] = self._labels_cache
            v_arr = self.mesh.GetPointData().GetArray("Labels")
            if v_arr:
                v_arr.Modified()
            self._labels_dirty = False
            self._vtk_labels_have_paint = True
        except Exception as e:
            print(f"⚠️ flush_labels_to_gpu (PointCloud) failed: {e}")


class MeshProduct(AbstractSceneProduct):
    """
    Scene product for surface mesh data.

    Wraps PyVista mesh with faces (triangles/polygons) for solid surface rendering
    and accurate occlusion testing. Meshes ARE solid (is_solid=True).

    Texture Support:
        If a texture image (texture.png, texture.jpg, etc.) is present in the same
        directory as the mesh file, it will be auto-loaded and added as a "Texture"
        layer in the viewer's array dropdown. Texture rendering requires UV coordinates
        to be present in the mesh (commonly texture_u/texture_v or u/v arrays from
        Metashape and other photogrammetry software). If a texture image exists but
        the mesh lacks UV coordinates, the texture will be disabled to prevent crashes.

    Simplification Caveat:
        When simplification_ratio > 0.0, UV coordinates may be corrupted due to
        nearest-neighbor transfer to the decimated mesh. Textures will likely tear
        or stretch. For textured meshes, keep simplification_ratio = 0.0.

        Spatial sorting (sort_data=True) is safe for textures — it only reorders
        faces, not vertex data or UV mappings.

    Attributes:
        mesh (pv.PolyData): The underlying PyVista surface mesh.
        opacity (float): Default rendering opacity.
        texture (pv.Texture): Loaded image texture, or None.
    """
    
    def __init__(self, file_path: str, opacity: float = 1.0, product_id: Optional[str] = None,
                 sort_data: bool = True, simplification_ratio: float = 0.0):
        """
        Initialize MeshProduct from file.

        Args:
            sort_data: When True, spatially sort faces via Morton Z-order after
                       loading for better GPU cache coherence.
            simplification_ratio: Fraction of faces to *remove* via
                fast-simplification (0.0 = no simplification, 1.0 = remove all
                faces).  Applied before sorting so the sort operates on the
                already-reduced mesh.  Values above 0.99 are clamped to 0.99 to
                ensure a non-degenerate result.
        """
        if product_id is None:
            product_id = os.path.basename(file_path)

        super().__init__(product_id=product_id, file_path=file_path)

        self.opacity = opacity
        self.mesh: Optional[pv.PolyData] = None
        self.array_names = []
        self.available_arrays = []
        self.selected_array = "RGB"

        start_time = time.time()
        raw_mesh = pv.read(file_path, progress_bar=True)
        print(f"⏱️ Loaded raw mesh for {self.label} with {raw_mesh.n_cells:,} faces in {time.time() - start_time:.3f}s")

        # Step 1: optional simplification (decimate before sorting so the sort
        # operates on the already-reduced mesh — smaller work, better locality).
        working_mesh = raw_mesh
        if simplification_ratio and simplification_ratio > 0.0:
            ratio = min(float(simplification_ratio), 0.99)
            faces_before = working_mesh.n_cells
            try:
                import fast_simplification
                if not working_mesh.is_all_triangles:
                    working_mesh = working_mesh.triangulate()
                tris = working_mesh.faces.reshape(-1, 4)[:, 1:]

                # Snapshot point arrays before simplification changes the topology.
                src_points = np.asarray(working_mesh.points, dtype=np.float32)
                pd = working_mesh.point_data
                array_names = list(pd.keys())
                src_arrays  = [np.asarray(pd[n]) for n in array_names]

                pts_out, tris_out = fast_simplification.simplify(
                    src_points, tris, target_reduction=ratio,
                )

                padding = np.full((len(tris_out), 1), 3, dtype=np.uint32)
                simplified = pv.PolyData(pts_out, np.hstack([padding, tris_out]).flatten())

                # Transfer point arrays to output vertices via nearest-neighbour
                # lookup — works regardless of fast_simplification version.
                if array_names:
                    from scipy.spatial import cKDTree
                    _, nn_idx = cKDTree(src_points).query(pts_out, k=1, workers=-1)
                    for name, arr in zip(array_names, src_arrays):
                        simplified.point_data[name] = arr[nn_idx]

                working_mesh = simplified
                print(f"⏱️ (Fast) Simplification: {faces_before:,} → {working_mesh.n_cells:,} faces "
                      f"({ratio:.0%} reduction) in {time.time() - start_time:.3f}s")
            except ImportError:
                # PyVista's decimate preserves point/cell data automatically.
                working_mesh = raw_mesh.decimate(ratio)
                print(f"⏱️ PyVista decimate (fast_simplification not installed): "
                      f"{faces_before:,} → {working_mesh.n_cells:,} faces in {time.time() - start_time:.3f}s")

        # Step 2: optional spatial sort for GPU cache coherence.
        if sort_data:
            self.mesh = self._spatially_sort_mesh(working_mesh)
            print(f"⏱️ Spatially sorted mesh for {self.label} in {time.time() - start_time:.3f}s")
        else:
            self.mesh = working_mesh

        self.array_names = self.mesh.array_names
        self._ensure_scalar_arrays()

        # --- TEXTURE IMPORT LOGIC ---
        self.texture = None
        dir_name = os.path.dirname(self.file_path)
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            tex_path = os.path.join(dir_name, f"texture{ext}")
            if os.path.exists(tex_path):
                try:
                    self.texture = pv.Texture(tex_path)
                    print(f"🖼️ Loaded texture from {tex_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load texture {tex_path}: {e}")

        # Check and bind UV coordinates — they may have been lost during simplification/sorting
        has_uvs = False
        if self.mesh is not None:
            # Check if PyVista already mapped the texture coordinates
            if getattr(self.mesh, 't_coords', None) is not None:
                has_uvs = True
            else:
                pd = self.mesh.point_data
                # TCoords is PyVista's default name; it may have survived simplification as a normal array
                if "TCoords" in pd:
                    self.mesh.t_coords = pd["TCoords"]
                    has_uvs = True
                else:
                    # Check common PLY UV array names
                    for u_name, v_name in [("texture_u", "texture_v"), ("u", "v"), ("s", "t")]:
                        if u_name in pd and v_name in pd:
                            self.mesh.t_coords = np.column_stack((pd[u_name], pd[v_name]))
                            has_uvs = True
                            break

        # If we loaded an image but the mesh has no UVs, discard the texture to prevent rendering crashes
        if self.texture is not None and not has_uvs:
            print(f"⚠️ Texture image found, but mesh lacks UV coordinates. Disabling texture support.")
            self.texture = None

        # Update array names in case binding UVs added a new array
        self.array_names = self.mesh.array_names
        # --------------------------------

        other_arrays = [arr for arr in self.array_names if arr.lower() not in ('rgb', 'labels', 'normals', 'class_ids', 'texture coordinates', 'tcoords')]

        # Register Texture as a valid dropdown array choice ONLY if we loaded one AND have UVs
        self.available_arrays = ["RGB", "Labels"]
        if self.texture is not None and has_uvs:
            self.available_arrays.append("Texture")
        self.available_arrays.extend(other_arrays)

        if self.mesh.n_cells == 0:
            raise ValueError(f"File '{file_path}' has no cells/faces - use PointCloudProduct instead")

        print(f"⏱️ Loaded MeshProduct: {self.label} with {self.mesh.n_cells:,} faces in {time.time() - start_time:.3f}s")

    def _spatially_sort_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Spatially sorts mesh faces using a pure NumPy Z-Order (Morton) Curve.
        Requires zero C++ dependencies while providing massive GPU cache hits 
        and index map entropy reduction.
        """
        # Ensure geometry is triangulated before attempting to reshape the face array
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
            
        # Extract raw faces and vertices
        faces_nx3 = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)
        pts = mesh.points
        
        # Calculate the centroid (center point) of every face
        centroids = pts[faces_nx3].mean(axis=1)
        
        # Normalize the 3D space into a 10-bit integer grid (0 to 1023)
        c_min = centroids.min(axis=0)
        c_max = centroids.max(axis=0)
        extent = np.maximum(c_max - c_min, 1e-8) 
        
        normalized = ((centroids - c_min) / extent * 1023).astype(np.uint32)
        
        x = normalized[:, 0]
        y = normalized[:, 1]
        z = normalized[:, 2]
        
        # Interleave the X, Y, Z bits to create a 30-bit Morton Code
        def expand_bits(v):
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v <<  8)) & 0x0300F00F
            v = (v | (v <<  4)) & 0x030C30C3
            v = (v | (v <<  2)) & 0x09249249
            return v
            
        morton_codes = (expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2))
        
        # Get the sorted indices and reorder the faces
        sort_idx = np.argsort(morton_codes)
        sorted_faces = faces_nx3[sort_idx]

        # Keep proof data for debug visualizations.
        self._sort_idx = sort_idx
        self._sort_centroids = centroids
        self._sort_morton_codes = morton_codes

        if len(morton_codes) > 1:
            original_steps = np.abs(np.diff(morton_codes.astype(np.int64, copy=False)))
            sorted_steps = np.abs(np.diff(np.sort(morton_codes.astype(np.int64, copy=False))))
            self._sort_original_mean_step = float(np.mean(original_steps)) if len(original_steps) else 0.0
            self._sort_sorted_mean_step = float(np.mean(sorted_steps)) if len(sorted_steps) else 0.0
        else:
            self._sort_original_mean_step = 0.0
            self._sort_sorted_mean_step = 0.0
        
        # Pack back into VTK's expected format (padding with 3)
        padding = np.full((len(sorted_faces), 1), 3, dtype=np.uint32)
        vtk_optimized_faces = np.hstack((padding, sorted_faces)).flatten()
        
        mesh.faces = vtk_optimized_faces
        
        return mesh

    def export_sort_proof(self, output_dir: str, canvas_size: int = 1024) -> Optional[str]:
        """
        Export a side-by-side proof image showing the effect of Morton sorting.

        The left panel colors faces by their original cell order. The right
        panel colors the same mesh by post-sort cell order, which should show
        smoother spatial grouping when sorting is effective.
        """
        if self.mesh is None:
            return None

        centroids = getattr(self, '_sort_centroids', None)
        sort_idx = getattr(self, '_sort_idx', None)
        if centroids is None or sort_idx is None:
            return None

        centroids = np.asarray(centroids, dtype=np.float32)
        sort_idx = np.asarray(sort_idx, dtype=np.int64)
        if centroids.ndim != 2 or centroids.shape[0] < 2 or sort_idx.size != centroids.shape[0]:
            return None

        try:
            import cv2
        except Exception as exc:
            print(f"⚠️ Mesh sort proof export skipped (OpenCV unavailable): {exc}")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Project the face centers onto their dominant 2D plane so the ordering
        # patterns are visible without a specific camera viewpoint.
        centered = centroids - centroids.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T

        proj_min = projected.min(axis=0)
        proj_max = projected.max(axis=0)
        span = np.maximum(proj_max - proj_min, 1e-8)
        projected = (projected - proj_min) / span

        panel_size = max(512, int(canvas_size))
        separator = 36
        footer_h = 104
        canvas = np.zeros((panel_size + footer_h, panel_size * 2 + separator, 3), dtype=np.uint8)
        canvas[:] = (10, 10, 14)

        def build_panel(ordered_indices: np.ndarray) -> np.ndarray:
            coords = projected[ordered_indices]
            values = np.linspace(0, 255, len(ordered_indices), dtype=np.float32)

            x = np.clip(np.round(coords[:, 0] * (panel_size - 1)).astype(np.int32), 0, panel_size - 1)
            y = np.clip(np.round((1.0 - coords[:, 1]) * (panel_size - 1)).astype(np.int32), 0, panel_size - 1)

            accum = np.zeros((panel_size, panel_size), dtype=np.float32)
            counts = np.zeros((panel_size, panel_size), dtype=np.int32)
            np.add.at(accum, (y, x), values)
            np.add.at(counts, (y, x), 1)

            gray = np.zeros((panel_size, panel_size), dtype=np.uint8)
            mask = counts > 0
            gray[mask] = np.clip(accum[mask] / counts[mask], 0, 255).astype(np.uint8)

            panel = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            panel[~mask] = (0, 0, 0)
            return panel

        original_panel = build_panel(np.arange(len(centroids), dtype=np.int64))
        sorted_panel = build_panel(sort_idx)

        canvas[:panel_size, :panel_size] = original_panel
        canvas[:panel_size, panel_size + separator:panel_size + separator + panel_size] = sorted_panel

        border_color = (90, 90, 96)
        cv2.rectangle(canvas, (0, 0), (panel_size - 1, panel_size - 1), border_color, 1)
        cv2.rectangle(
            canvas,
            (panel_size + separator, 0),
            (panel_size + separator + panel_size - 1, panel_size - 1),
            border_color,
            1,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_color = (235, 235, 235)
        body_color = (210, 210, 210)
        cv2.putText(canvas, "Original cell order", (24, panel_size + 34), font, 0.9, title_color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Morton-sorted cell order",
            (panel_size + separator + 24, panel_size + 34),
            font,
            0.9,
            title_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Color = face position in that order (early -> late)",
            (24, panel_size + 68),
            font,
            0.65,
            body_color,
            1,
            cv2.LINE_AA,
        )

        improvement = 0.0
        if getattr(self, '_sort_sorted_mean_step', 0.0) > 0:
            improvement = float(getattr(self, '_sort_original_mean_step', 0.0) / self._sort_sorted_mean_step)

        metric_text = (
            f"Mean consecutive Morton-code delta: raw={getattr(self, '_sort_original_mean_step', 0.0):.4g} | "
            f"sorted={getattr(self, '_sort_sorted_mean_step', 0.0):.4g} | "
            f"improvement={improvement:.2f}x"
        )
        cv2.putText(canvas, metric_text, (24, panel_size + 96), font, 0.58, body_color, 1, cv2.LINE_AA)

        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_sort_proof.png")
        cv2.imwrite(output_path, canvas)
        print(f"🧪 Exported mesh sort proof: {output_path}")
        return output_path
    
    @classmethod
    def from_file(cls, file_path: str, opacity: float = 1.0,
                  sort_data: bool = True, simplification_ratio: float = 0.0) -> 'MeshProduct':
        """Load a mesh from file."""
        return cls(file_path=file_path, opacity=opacity,
                   sort_data=sort_data, simplification_ratio=simplification_ratio)
    
    # Custom method to build and cache Open3D RaycastingScene for this mesh
    def prepare_geometry(self):

        print(f"📐 Extracting raw geometry arrays for {self.label}...")
        import time
        import numpy as np
        import torch
        
        if not self.mesh.is_all_triangles:
            tri_mesh = self.mesh.triangulate()
            raw_ids = tri_mesh.cell_data.get('vtkOriginalCellIds', None)
            self._original_cell_ids = np.asarray(raw_ids) if raw_ids is not None else None
        else:
            tri_mesh = self.mesh
            self._original_cell_ids = None
        
        # Extract the rest to numpy temporarily
        triangles_np = np.asarray(tri_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.uint32)
        # Cache a CPU-side copy of the triangle indices to avoid costly
        # GPU->CPU transfers during repeated hit-tests.
        self._cached_triangles_np = triangles_np
        centers_np = np.asarray(tri_mesh.cell_centers().points, dtype=np.float32)
        centers_sq_norm_np = np.sum(centers_np**2, axis=1)
        
        # Cache geometry arrays in PyTorch tensors for fast GPU processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        self._cached_face_centers_pt = torch.tensor(centers_np, device=self.device)
        self._cached_centers_sq_norm_pt = torch.tensor(centers_sq_norm_np, device=self.device)
        
        if self._original_cell_ids is not None:
            self._original_cell_ids_pt = torch.tensor(self._original_cell_ids.astype(np.int64), device=self.device)
            self._element_centers_np = np.asarray(self.mesh.cell_centers().points, dtype=np.float32)
        else:
            self._element_centers_np = centers_np.copy()
    
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

        # If the user selected the Texture layer, apply the pyvista.Texture
        if self.selected_array == "Texture" and getattr(self, 'texture', None) is not None:
            style['texture'] = self.texture
            style['color'] = 'white'
        # All arrays (RGB, Labels, and data arrays) are now real scalars in the mesh
        # Use them directly via the mapper
        elif self.selected_array in self.array_names:
            style['scalars'] = self.selected_array
            # RGB, Labels, and Normals_RGB are all Nx3 uint8, so they need direct RGB mode
            if self.selected_array in ("RGB", "Labels", "Normals_RGB"):
                style['rgb'] = True
        else:
            # Fallback: render as metashape purple
            style['color'] = '#8d8cc4'  # Metashape purple

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

    def _get_cached_face_normals(self) -> Optional[np.ndarray]:
        """Return cached per-face normals, computing them once per product.

        ``get_face_normals`` recomputes normals on every call; the densify
        gather needs them per stroke, so cache the result and invalidate only
        when the product identity changes (mirrors the KD-tree caching).
        """
        cached = getattr(self, '_face_normals_np', None)
        cached_pid = getattr(self, '_face_normals_product_id', None)
        if cached is not None and cached_pid == getattr(self, 'product_id', None):
            return cached
        try:
            normals = np.asarray(self.get_face_normals(), dtype=np.float32)
        except Exception:
            return None
        self._face_normals_np = normals
        self._face_normals_product_id = getattr(self, 'product_id', None)
        return normals

    def gather_dense_face_ids(self, seed_ids, *, camera_position=None,
                              radius_mult: float = 1.5, normal_dot_min: float = 0.3,
                              max_expansion: float = 40.0) -> np.ndarray:
        """Expand sparse seed face IDs into the dense set covering the same patch.

        Faces sampled from a low-resolution index map are sparse — many faces
        between samples never get a pixel.  This uses the prewarmed face-center
        KD-tree (``_hover_face_kdtree``) to gather every face whose centroid lies
        within a self-calibrating radius of any seed, then filters out occluded
        / back-facing candidates so the result tracks the visible surface rather
        than geometry behind it.

        Args:
            seed_ids: Sparse face IDs sampled from a (low-res) index map.
            camera_position: Source camera world position.  Enables the
                camera-facing cull (Stage 1); skipped when None (e.g. ortho).
            radius_mult: Gather radius as a multiple of the median
                nearest-neighbor spacing among the seed centroids.
            normal_dot_min: Minimum dot product between a candidate's normal and
                its nearest seed's normal to keep the candidate (Stage 2).  Set
                to None to disable.
            max_expansion: Circuit breaker — if the gathered set exceeds this
                multiple of the seed count, the seeds are returned unchanged.

        Returns:
            np.ndarray[int64] dense face IDs (always a superset of the surviving
            seeds), or the original seeds on any failure / no-op.
        """
        seeds = np.asarray(seed_ids, dtype=np.int64).ravel()
        seeds = np.unique(seeds[seeds >= 0])
        if seeds.size == 0:
            return seeds

        tree = getattr(self, '_hover_face_kdtree', None)
        centers = getattr(self, '_element_centers_np', None)
        if tree is None or centers is None:
            return seeds

        centers = np.asarray(centers, dtype=np.float32)
        n_faces = len(centers)
        seeds = seeds[seeds < n_faces]
        if seeds.size == 0:
            return np.asarray(seed_ids, dtype=np.int64)

        pts = centers[seeds]

        # --- Self-calibrating gather radius from seed spacing ---
        radius = 0.0
        if seeds.size >= 2:
            try:
                nn_dist, _ = tree.query(pts, k=2, workers=-1)
                nn_dist = nn_dist[:, 1]
                nn_dist = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
                if nn_dist.size:
                    radius = float(np.median(nn_dist)) * float(radius_mult)
            except Exception:
                radius = 0.0
        if radius <= 0.0:
            diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
            radius = diag * 1e-3
        if radius <= 0.0:
            return seeds

        # --- Thin query centers to ~half the gather radius -------------------
        # query_ball_point cost (and the volume of duplicate output we then
        # dedupe) scales with the number of ball centers.  Seeds sampled from a
        # dense mesh heavily overlap, so collapse them to one representative per
        # voxel; the union of the (unchanged-radius) balls is virtually identical
        # but far cheaper.  On sparse low-res seeds this is a no-op.
        query_pts = pts
        if pts.shape[0] > 1:
            voxel = radius * 0.5
            if voxel > 0.0:
                cell_keys = np.floor(pts / voxel).astype(np.int64)
                _, rep_idx = np.unique(cell_keys, axis=0, return_index=True)
                query_pts = pts[rep_idx]

        # --- Radius-union gather (conforms to the stroke shape) --------------
        try:
            neighbor_lists = tree.query_ball_point(query_pts, radius, workers=-1)
        except TypeError:
            neighbor_lists = tree.query_ball_point(query_pts, radius)
        # C-level concat of per-center arrays, not a Python generator flatten.
        sub_arrays = [np.asarray(sub, dtype=np.int64) for sub in neighbor_lists if len(sub)]
        if not sub_arrays:
            return seeds
        candidates = np.unique(np.concatenate(sub_arrays))
        if candidates.size == 0:
            return seeds

        # --- Back-face / occlusion filtering ---
        normals = self._get_cached_face_normals()
        if normals is not None and normals.shape[0] == n_faces:
            cand_centers = centers[candidates]
            cand_normals = normals[candidates]
            keep = np.ones(candidates.size, dtype=bool)

            # Stage 1: camera-facing cull. A front face points back toward the
            # camera, so dot(normal, view_dir) <= 0.
            if camera_position is not None:
                cam = np.asarray(camera_position, dtype=np.float32).reshape(3)
                view = cand_centers - cam
                vn = np.linalg.norm(view, axis=1, keepdims=True)
                vn[vn < 1e-8] = 1.0
                view = view / vn
                facing = np.einsum('ij,ij->i', cand_normals, view)
                keep &= facing <= 0.0

            # Stage 2: normal consistency vs the nearest seed.
            if normal_dot_min is not None:
                try:
                    from scipy.spatial import cKDTree
                    seed_tree = cKDTree(pts)
                    _, nearest_local = seed_tree.query(cand_centers, k=1, workers=-1)
                    ref_normals = normals[seeds][nearest_local]
                    consistency = np.einsum('ij,ij->i', cand_normals, ref_normals)
                    keep &= consistency >= float(normal_dot_min)
                except Exception:
                    pass

            candidates = candidates[keep]

        # Always retain the original (visible, confirmed) seeds.
        dense = np.union1d(candidates, seeds)

        # Circuit breaker: a stroke straddling a discontinuity can balloon the
        # gather — fall back to the seeds rather than mislabel distant geometry.
        if dense.size > max_expansion * seeds.size:
            return seeds

        return dense.astype(np.int64, copy=False)

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
        return True
    
    def _ensure_scalar_arrays(self):
        """
        Ensure required scalar arrays exist in the mesh.
        """
        if self.mesh is None:
            return
        
        n_faces = self.mesh.n_cells

        # 1. Create "Labels" array if it doesn't exist - uniform white
        labels_preexisting = "Labels" in self.mesh.array_names
        if not labels_preexisting:
            labels_array = np.ones((n_faces, 3), dtype=np.uint8) * 255  # White
            self.mesh.cell_data["Labels"] = labels_array

        # Create a Python-owned labels cache detached from VTK for background painting
        self._labels_cache = np.asarray(self.mesh.cell_data["Labels"]).copy()
        # Deferred-flush bookkeeping (see PointCloudProduct._ensure_scalar_arrays)
        self._labels_dirty = False
        self._vtk_labels_have_paint = labels_preexisting

        # Keep a readonly reference to the PyVista view for flush operations only
        self._labels_view = self.mesh.cell_data["Labels"]

        # Keep this entirely in Python RAM. Do NOT attach class_ids to self.mesh!
        if not hasattr(self, 'class_ids'):
            self.class_ids = np.zeros(n_faces, dtype=np.int32)
        
        # 2. Create "RGB" array if it doesn't exist - Metashape purple
        if "RGB" not in self.mesh.array_names:
            rgb_array = np.ones((n_faces, 3), dtype=np.uint8)
            rgb_array[:, 0] = 141  
            rgb_array[:, 1] = 140  
            rgb_array[:, 2] = 196  
            self.mesh.cell_data["RGB"] = rgb_array
        
        # Ensure raw float normals exist for default VTK lighting, 
        # but DO NOT generate the heavy Normals_RGB array.
        if "Normals" not in self.mesh.array_names and "normals" not in self.mesh.array_names:
            self.mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        self.array_names = self.mesh.array_names

    def apply_labels(self, element_ids: np.ndarray, class_id: int, color_rgb: tuple) -> None:
        """
        Update the class ID and visual color for specific mesh faces.
        """
        if self.mesh is None or len(element_ids) == 0:
            return
        # 1. Update semantic data in Python RAM
        self.class_ids[element_ids] = class_id

        # 2. Update the python-owned labels cache (no VTK writes here)
        if not hasattr(self, '_labels_cache'):
            self._labels_cache = np.asarray(self.mesh.cell_data["Labels"]).copy()
        self._labels_cache[element_ids] = color_rgb
        self._labels_dirty = True

        # NOTE: No immediate VTK Modified() here. The GUI thread should call
        # `flush_labels_to_gpu()` once painting activity settles.

    def flush_labels_to_gpu(self) -> None:
        """One full GPU upload. Call only on the GUI/main thread, and only at
        rare barriers (scene rebuild, array switch, erase-after-flush) — this
        triggers a full VTK mapper rebuild on the next render."""
        if self.mesh is None or not hasattr(self, '_labels_cache'):
            return
        if not getattr(self, '_labels_dirty', True):
            return
        try:
            self.mesh.cell_data["Labels"] = self._labels_cache
            labels_array = self.mesh.GetCellData().GetArray("Labels")
            if labels_array:
                labels_array.Modified()
            self._labels_dirty = False
            self._vtk_labels_have_paint = True
        except Exception as e:
            print(f"⚠️ flush_labels_to_gpu (Mesh) failed: {e}")


class GaussianSplattingProduct(AbstractSceneProduct):
    """
    Scene product for 3D Gaussian Splatting (3DGS) data.

    Loads a 3DGS-format PLY file (which carries per-splat SH coefficients,
    quaternion rotations, anisotropic scales and opacities) via pyvista_gs
    and wraps the resulting GaussianActor.

    Rendering is handled entirely by the GaussianActor's own OpenGL pipeline,
    which is injected into the PyVista plotter via bind_to_plotter().  The
    pv.PolyData of splat centres is kept as a lightweight scene anchor that
    provides correct bounding-box information to the viewer without involving
    the splat renderer.

    Index mapping and annotation are not supported for this product type.
    """

    def __init__(self, file_path: str, product_id: Optional[str] = None):
        """
        Load a 3DGS PLY file and initialise the GaussianActor.

        Args:
            file_path:  Path to the 3DGS .ply file.
            product_id: Optional unique ID (defaults to filename).
        """
        if product_id is None:
            product_id = os.path.basename(file_path)

        super().__init__(product_id=product_id, file_path=file_path)

        start_time = time.time()

        from pyvista_gs.data import load_ply
        from pyvista_gs.actor import GaussianActor

        gaussian_data = load_ply(file_path)
        self.gaussian_actor = GaussianActor(gaussian_data)

        load_time = time.time() - start_time
        print(
            f"⏱️ Loaded GaussianSplattingProduct: {self.label} "
            f"with {self.gaussian_actor.point_count:,} splats in {load_time:.3f}s"
        )

    @classmethod
    def from_file(cls, file_path: str) -> 'GaussianSplattingProduct':
        """Load a 3DGS scene from a PLY file."""
        return cls(file_path=file_path)

    # --------------------------------------------------------------------------
    # AbstractSceneProduct Implementation
    # --------------------------------------------------------------------------

    def get_render_mesh(self) -> pv.PolyData:
        """
        Return the PolyData of Gaussian splat centres.

        This mesh is used only as a scene anchor for bounding-box queries.
        Actual rendering is performed by the GaussianActor via its OpenGL
        pipeline; the actor's pv.Actor has opacity=0 so nothing is visible
        through the standard PyVista path.
        """
        return self.gaussian_actor._mesh

    def get_render_style(self) -> RenderStyle:
        """
        Return a minimal invisible style.

        The GaussianActor renders via its own OpenGL observer registered in
        bind_to_plotter().  The pv.Actor added by that call has opacity=0
        and only serves as a handle for the plotter's actor registry.
        """
        return {
            'style': 'points',
            'point_size': 1,
            'opacity': 0.0,
            'lighting': False,
        }

    def get_bounds(self) -> BoundsType:
        """Get 3D bounding box from the splat centre positions."""
        mesh = self.gaussian_actor._mesh
        if mesh is None or mesh.n_points == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return mesh.bounds

    def is_solid(self) -> bool:
        """3DGS scenes are not solid surfaces."""
        return False

    def supports_index_mapping(self) -> bool:
        """Index-map annotation is not supported for Gaussian splats."""
        return False

    def get_element_type(self) -> ElementType:
        """Splat centres are treated as points."""
        return 'point'

    def get_element_count(self) -> int:
        """Number of Gaussian splats."""
        return self.gaussian_actor.point_count

    def get_element_coordinate(self, element_id: int) -> Optional[np.ndarray]:
        """Return the world-space centre of a single splat."""
        pts = self.gaussian_actor._mesh.points
        if element_id < 0 or element_id >= len(pts):
            return None
        return pts[element_id].astype(np.float64)

    def apply_labels(self, element_ids: np.ndarray, class_id: int, color_rgb: tuple) -> None:
        """No-op — splat annotation is not yet implemented."""
        pass

    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release the OpenGL resources held by the GaussianActor."""
        try:
            self.gaussian_actor.cleanup()
        except Exception as e:
            print(f"⚠️ GaussianSplattingProduct.cleanup failed: {e}")

