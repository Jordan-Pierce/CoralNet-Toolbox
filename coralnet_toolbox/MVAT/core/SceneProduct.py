"""
Scene Product Base Class for MVAT

Defines the abstract interface that all 3D data products (point clouds, meshes, DEMs) 
must implement to integrate with the heterogeneous scene architecture.
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
