"""
Scene Context Container for MVAT

The SceneContext is the single source of truth for the 3D scene.
It holds a collection of scene products (point clouds, meshes, DEMs) and provides
capability queries for the visibility engine and rendering pipeline.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from coralnet_toolbox.MVAT.core.SceneProduct import (
    AbstractSceneProduct,
    BoundsType,
    ElementType,
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SceneContext:
    """
    Container for heterogeneous 3D scene products.
    
    Manages multiple scene products (point clouds, meshes, DEMs) and provides:
    - Unified bounding box calculation
    - Best occluder selection for visibility computation
    - Primary target designation for annotation indexing
    - Fallback depth calculation for ray casting
    
    Attributes:
        products (Dict[str, AbstractSceneProduct]): All loaded products keyed by ID.
        primary_target_id (Optional[str]): ID of the designated annotation target.
    """
    
    def __init__(self):
        """Initialize an empty scene context."""
        self._products: Dict[str, AbstractSceneProduct] = {}
        self._primary_target_id: Optional[str] = None
        
        # Cached unified bounds (invalidated on add/remove)
        self._cached_bounds: Optional[BoundsType] = None
    
    # --------------------------------------------------------------------------
    # Product Management
    # --------------------------------------------------------------------------
    
    def add_product(self, product: AbstractSceneProduct) -> None:
        """
        Add a scene product to the context.
        
        Args:
            product: The scene product to add.
            
        Raises:
            ValueError: If a product with the same ID already exists.
        """
        if product.product_id in self._products:
            raise ValueError(f"Product with ID '{product.product_id}' already exists")
        
        self._products[product.product_id] = product
        self._invalidate_caches()
        
        # Auto-select primary target if this is the first indexable product
        if self._primary_target_id is None and product.supports_index_mapping():
            self._primary_target_id = product.product_id
        
        print(f"🎬 SceneContext: Added {product}")
    
    def remove_product(self, product_id: str) -> Optional[AbstractSceneProduct]:
        """
        Remove a product from the context.
        
        Args:
            product_id: ID of the product to remove.
            
        Returns:
            The removed product, or None if not found.
        """
        product = self._products.pop(product_id, None)
        
        if product is not None:
            self._invalidate_caches()
            
            # Clear primary target if it was removed
            if self._primary_target_id == product_id:
                self._primary_target_id = None
                # Auto-select next available indexable product
                for p in self._products.values():
                    if p.supports_index_mapping():
                        self._primary_target_id = p.product_id
                        break
            
            print(f"🎬 SceneContext: Removed {product}")
        
        return product
    
    def get_product(self, product_id: str) -> Optional[AbstractSceneProduct]:
        """Get a product by ID."""
        return self._products.get(product_id)
    
    def clear(self) -> None:
        """Remove all products from the context."""
        self._products.clear()
        self._primary_target_id = None
        self._invalidate_caches()
        print("🎬 SceneContext: Cleared all products")
    
    @property
    def products(self) -> Dict[str, AbstractSceneProduct]:
        """Read-only access to products dictionary."""
        return self._products
    
    def __len__(self) -> int:
        """Number of products in the scene."""
        return len(self._products)
    
    def __iter__(self):
        """Iterate over products."""
        return iter(self._products.values())
    
    def __contains__(self, product_id: str) -> bool:
        """Check if a product ID exists."""
        return product_id in self._products
    
    # --------------------------------------------------------------------------
    # Capability Queries
    # --------------------------------------------------------------------------
    
    def has_any_product(self) -> bool:
        """Check if any products are loaded."""
        return len(self._products) > 0
    
    def has_solid_occluder(self) -> bool:
        """Check if any solid occluder (mesh or DEM) is available."""
        return any(p.is_solid() for p in self._products.values())
    
    def get_products_by_type(self, element_type: ElementType) -> List[AbstractSceneProduct]:
        """
        Get all products of a specific element type.
        
        Args:
            element_type: 'point', 'face', or 'cell'
            
        Returns:
            List of products matching the element type.
        """
        return [p for p in self._products.values() if p.get_element_type() == element_type]
    
    def get_products_by_class(self, cls) -> List[AbstractSceneProduct]:
        """
        Get all products that are instances of a specific class.
        
        Args:
            cls: Class type to filter by (e.g., PointCloudProduct)
            
        Returns:
            List of products that are instances of cls.
        """
        return [p for p in self._products.values() if isinstance(p, cls)]
    
    # --------------------------------------------------------------------------
    # Occluder & Target Selection
    # --------------------------------------------------------------------------
    
    def get_best_occluder(self) -> Optional[AbstractSceneProduct]:
        """
        Get the best available product for occlusion testing.
        
        Priority order:
        1. MeshProduct (solid, accurate ray-casting)
        2. DEMProduct (solid, grid-based occlusion)
        3. PointCloudProduct (fallback, requires splatting)
        
        Returns:
            Best available occluder, or None if scene is empty.
        """
        # Priority 1: Find any solid occluder (mesh or DEM)
        for product in self._products.values():
            if product.is_solid() and product.get_element_type() == 'face':
                return product
        
        # Priority 2: DEM (cell-based solid)
        for product in self._products.values():
            if product.is_solid() and product.get_element_type() == 'cell':
                return product
        
        # Priority 3: Point cloud (fallback)
        for product in self._products.values():
            if product.get_element_type() == 'point':
                return product
        
        return None
    
    def get_primary_target(self) -> Optional[AbstractSceneProduct]:
        """
        Get the designated primary target for annotation indexing.
        
        Returns:
            The primary target product, or None if not set.
        """
        if self._primary_target_id is None:
            return None
        return self._products.get(self._primary_target_id)
    
    def set_primary_target(self, product_id: Optional[str]) -> bool:
        """
        Set the primary target for annotation indexing.
        
        Args:
            product_id: ID of the product to designate, or None to clear.
            
        Returns:
            True if successful, False if product not found or doesn't support indexing.
        """
        if product_id is None:
            self._primary_target_id = None
            return True
        
        product = self._products.get(product_id)
        if product is None:
            print(f"⚠️ SceneContext: Cannot set primary target - product '{product_id}' not found")
            return False
        
        if not product.supports_index_mapping():
            print(f"⚠️ SceneContext: Cannot set primary target - product '{product_id}' doesn't support indexing")
            return False
        
        self._primary_target_id = product_id
        print(f"🎯 SceneContext: Primary target set to '{product_id}'")
        return True
    
    @property
    def primary_target_id(self) -> Optional[str]:
        """ID of the current primary target."""
        return self._primary_target_id
    
    # --------------------------------------------------------------------------
    # Bounding Box & Geometry
    # --------------------------------------------------------------------------
    
    def unified_bounds(self) -> Optional[BoundsType]:
        """
        Get the unified bounding box encompassing all products.
        
        Returns:
            Tuple (xmin, xmax, ymin, ymax, zmin, zmax), or None if scene is empty.
        """
        if not self._products:
            return None
        
        if self._cached_bounds is not None:
            return self._cached_bounds
        
        # Initialize with first product's bounds
        first_bounds = next(iter(self._products.values())).get_bounds()
        xmin, xmax = first_bounds[0], first_bounds[1]
        ymin, ymax = first_bounds[2], first_bounds[3]
        zmin, zmax = first_bounds[4], first_bounds[5]
        
        # Expand to include all other products
        for product in self._products.values():
            bounds = product.get_bounds()
            xmin = min(xmin, bounds[0])
            xmax = max(xmax, bounds[1])
            ymin = min(ymin, bounds[2])
            ymax = max(ymax, bounds[3])
            zmin = min(zmin, bounds[4])
            zmax = max(zmax, bounds[5])
        
        self._cached_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        return self._cached_bounds
    
    def get_center(self) -> Optional[np.ndarray]:
        """
        Get the center of the unified bounding box.
        
        Returns:
            np.ndarray of shape (3,), or None if scene is empty.
        """
        bounds = self.unified_bounds()
        if bounds is None:
            return None
        
        return np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
    
    def get_diagonal_length(self) -> float:
        """
        Get the diagonal length of the unified bounding box.
        
        Returns:
            Diagonal length, or 0.0 if scene is empty.
        """
        bounds = self.unified_bounds()
        if bounds is None:
            return 0.0
        
        return np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
    
    def get_fallback_depth(self, camera_position: np.ndarray) -> float:
        """
        Calculate a fallback depth estimate for ray casting.
        
        Used when z-channel is not available. Returns the distance from the
        camera to the scene center.
        
        Args:
            camera_position: 3D position of the camera.
            
        Returns:
            Estimated depth to scene center, or 10.0 if scene is empty.
        """
        center = self.get_center()
        if center is None:
            return 10.0
        
        depth = float(np.linalg.norm(center - camera_position))
        return depth if depth > 0 else 10.0
    
    # --------------------------------------------------------------------------
    # Internal Methods
    # --------------------------------------------------------------------------
    
    def _invalidate_caches(self) -> None:
        """Invalidate all cached computations."""
        self._cached_bounds = None
    
    def __repr__(self) -> str:
        product_summary = ", ".join(
            f"{p.get_element_type()}:{p.get_element_count():,}" 
            for p in self._products.values()
        )
        return f"SceneContext({len(self._products)} products: [{product_summary}])"
