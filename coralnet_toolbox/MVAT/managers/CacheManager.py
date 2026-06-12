"""
CacheManager for MVAT Visibility Data

Handles persistent caching of index maps and visible indices to disk.
Uses MD5 hashing of camera parameters and point cloud paths for cache keys.
"""
import os
import json
import hashlib
import functools
import numpy as np
from typing import Optional, Tuple, Dict

from coralnet_toolbox.MVAT.utils.IndexMapCodec import (
    load_index_map_archive,
    save_index_map_archive,
)


# ---------------------------------------------------------------------------
# Module-level LRU-cached MD5 helper — avoid recomputing hashes on every
# load/save call for the same camera + geometry combination.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=2048)
def _cached_md5(extrinsics_bytes: bytes, path: str, element_type: str,
                extra: Optional[bytes],
                pixel_budget: Optional[int] = None) -> str:
    """Return the hex MD5 digest for a (extrinsics, path, element_type, extra)
    tuple.

    ``pixel_budget`` is intentionally ignored in the current cache key so the
    same scene reuses one cache file across quality settings. The parameter is
    retained for call-site compatibility.
    """
    h = hashlib.md5()
    h.update(extrinsics_bytes)
    h.update(path.encode('utf-8'))
    h.update(element_type.encode('utf-8'))
    if extra is not None:
        h.update(extra)
    return h.hexdigest()


@functools.lru_cache(maxsize=256)
def _cached_ortho_md5(ortho_path: str, mesh_path: str,
                      chunk_transform_bytes: bytes,
                      proj_mat_bytes: Optional[bytes],
                      native_size_bytes: bytes,
                      element_type: str) -> str:
    """Return the hex MD5 digest for an orthomosaic cache key."""
    h = hashlib.md5()
    h.update(ortho_path.encode('utf-8'))
    h.update(mesh_path.encode('utf-8'))
    h.update(chunk_transform_bytes)
    if proj_mat_bytes is not None:
        h.update(proj_mat_bytes)
    h.update(native_size_bytes)
    h.update(element_type.encode('utf-8'))
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CacheManager:
    """
    Manages persistent caching of visibility data (index maps and visible indices).
    
    Cache files are stored in `.cache/mvat/` relative to the project root.
    Each cache file is named using an MD5 hash of the camera extrinsics and point cloud path.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the CacheManager.

        Args:
            project_root (str): Root directory of the project where cache will be stored.

        Note: Caching is controlled at the VisibilityWorker level via the dialog,
              not at the CacheManager level.
        """
        self.project_root = project_root
        self.cache_dir = os.path.join(project_root, '.cache', 'mvat')

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, extrinsics: np.ndarray, point_cloud_path: str,
                             element_type: str = 'point',
                             extra_hash_data: Optional[bytes] = None,
                             pixel_budget: Optional[int] = None) -> str:
        """
        Generate a unique cache key based on camera extrinsics, geometry path,
        element type, and optional extra hash data.

        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix (4x4)
            point_cloud_path (str): Path to the geometry file (point cloud, mesh, or DEM)
            element_type (str): Type of indexed elements ('point', 'face', or 'cell')
            extra_hash_data (bytes, optional): Additional bytes mixed into the hash.
                Pass ``raster.dist_coeffs.tobytes()`` for distorted cameras so their
                warped maps never collide with undistorted maps from the same camera.
            pixel_budget (int, optional): Retained for call-site compatibility.
                The current cache key ignores it so a scene reuses the same cache
                file across quality settings.

        Returns:
            str: MD5 hash string to use as cache key
        """
        # Delegate to the module-level LRU-cached function so repeated calls
        # for the same camera skip the MD5 computation entirely.
        return _cached_md5(
            extrinsics.tobytes(),
            point_cloud_path,
            element_type,
            extra_hash_data,
            pixel_budget,
        )

    def get_cache_path(self, extrinsics: np.ndarray, point_cloud_path: str,
                        element_type: str = 'point',
                        extra_hash_data: Optional[bytes] = None,
                        pixel_budget: Optional[int] = None) -> str:
        """
        Get the full path to the cache file for given parameters.

        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the geometry file
            element_type (str): Type of indexed elements ('point', 'face', or 'cell')
            extra_hash_data (bytes, optional): Additional bytes mixed into the hash
                (see _generate_cache_key).
            pixel_budget (int, optional): Retained for compatibility; ignored by
                the current cache key (see _generate_cache_key).

        Returns:
            str: Canonical cache archive path.
        """
        cache_key = self._generate_cache_key(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        )
        return os.path.join(self.cache_dir, f"{cache_key}.npz")

    def _cache_path_exists(self, cache_path: str) -> bool:
        """Return True if the cache archive exists on disk."""
        return os.path.exists(cache_path)

    def _resolve_existing_cache_path(self,
                                     extrinsics: np.ndarray,
                                     point_cloud_path: str,
                                     element_type: str,
                                     extra_hash_data: Optional[bytes],
                                     pixel_budget: Optional[int]) -> Optional[str]:
        """Locate a cache file on disk using the current quality-agnostic key."""
        primary = self.get_cache_path(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        )
        if self._cache_path_exists(primary):
            return primary

        return None

    def has_visibility_cache(self, extrinsics: np.ndarray, point_cloud_path: str,
                             element_type: str = 'point',
                             extra_hash_data: Optional[bytes] = None,
                             pixel_budget: Optional[int] = None) -> bool:
        """Return True when a current cache file exists on disk."""
        return self._resolve_existing_cache_path(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        ) is not None

    def _generate_ortho_cache_key(self,
                                  ortho_image_path: str,
                                  mesh_path: str,
                                  chunk_transform: np.ndarray,
                                  ortho_projection_matrix: Optional[np.ndarray],
                                  native_size: Tuple[int, int],
                                  element_type: str = 'face') -> str:
        """Generate a cache key for an orthomosaic index map (LRU-cached)."""
        proj_bytes = (np.asarray(ortho_projection_matrix, dtype=np.float64).tobytes()
                      if ortho_projection_matrix is not None else None)
        return _cached_ortho_md5(
            str(ortho_image_path),
            str(mesh_path),
            np.asarray(chunk_transform, dtype=np.float64).tobytes(),
            proj_bytes,
            np.asarray(native_size, dtype=np.int32).tobytes(),
            element_type,
        )

    def get_ortho_index_map_cache_path(self,
                                       ortho_image_path: str,
                                       mesh_path: str,
                                       chunk_transform: np.ndarray,
                                       ortho_projection_matrix: Optional[np.ndarray],
                                       scale_factor: float,
                                       native_size: Tuple[int, int],
                                       element_type: str = 'face') -> str:
        """Get the cache path for an orthomosaic index map."""
        cache_key = self._generate_ortho_cache_key(
            ortho_image_path,
            mesh_path,
            chunk_transform,
            ortho_projection_matrix,
            native_size,
            element_type,
        )
        return os.path.join(self.cache_dir, f"ortho_{cache_key}.npz")

    def load_ortho_index_map(self,
                             ortho_image_path: str,
                             mesh_path: str,
                             chunk_transform: np.ndarray,
                             ortho_projection_matrix: Optional[np.ndarray],
                             scale_factor: float,
                             native_size: Tuple[int, int],
                             element_type: str = 'face') -> Optional[Dict]:
        """Load a cached orthomosaic index map if present."""
        cache_path = self.get_ortho_index_map_cache_path(
            ortho_image_path,
            mesh_path,
            chunk_transform,
            ortho_projection_matrix,
            scale_factor,
            native_size,
            element_type,
        )
        if not os.path.exists(cache_path):
            return None

        try:
            result = load_index_map_archive(cache_path)
            result['scale_factor'] = float(result.get('scale_factor', scale_factor))
            result['element_type'] = str(result.get('element_type', element_type))
            result['cache_path'] = cache_path
            return result
        except Exception as e:
            print(f"Warning: Failed to load ortho index map cache from {cache_path}: {e}")
            return None

    def save_ortho_index_map(self,
                             ortho_image_path: str,
                             mesh_path: str,
                             chunk_transform: np.ndarray,
                             ortho_projection_matrix: Optional[np.ndarray],
                             scale_factor: float,
                             native_size: Tuple[int, int],
                             index_map: np.ndarray,
                             visible_indices: np.ndarray,
                             element_type: str = 'face',
                             compressed: bool = False) -> Optional[str]:
        """Save an orthomosaic index map to cache."""
        cache_path = self.get_ortho_index_map_cache_path(
            ortho_image_path,
            mesh_path,
            chunk_transform,
            ortho_projection_matrix,
            scale_factor,
            native_size,
            element_type,
        )

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create cache directory {self.cache_dir}: {e}")
            return None

        try:
            index_map = np.asarray(index_map).astype(np.int32, copy=False)
            visible_indices = np.asarray(visible_indices).astype(np.int32, copy=False)
        except Exception:
            pass

        try:
            save_index_map_archive(
                cache_path,
                index_map,
                visible_indices,
                element_type=element_type,
                scale_factor=float(scale_factor),
            )
            return cache_path
        except Exception as e:
            print(f"Warning: Failed to save ortho index map cache to {cache_path}: {e}")
            return None
    
    def load_visibility(self, extrinsics: np.ndarray, point_cloud_path: str,
                         element_type: str = 'point',
                         extra_hash_data: Optional[bytes] = None,
                         pixel_budget: Optional[int] = None) -> Optional[Dict]:
        """
        Load visibility data from cache if it exists.

        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the geometry file
            element_type (str): Type of indexed elements ('point', 'face', or 'cell')
            extra_hash_data (bytes, optional): Additional bytes mixed into the hash
                (see _generate_cache_key).
            pixel_budget (int, optional): Render pixel budget for the desired quality.
                See _generate_cache_key for backward-compatibility semantics.

        Returns:
            dict or None: Dictionary with 'index_map', 'visible_indices', 'depth_map',
                         and 'element_type' if cache exists, None otherwise
        """
        # Resolve the actual on-disk cache file using the quality-agnostic key.
        cache_path = self._resolve_existing_cache_path(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        )
        if cache_path is None:
            return None
        try:
            result = load_index_map_archive(cache_path)
            result['element_type'] = str(result.get('element_type', element_type))
            result['cache_path'] = cache_path
            return result
        except Exception as e:
            print(f"Warning: Failed to load visibility cache from {cache_path}: {e}")
            return None
    
    def save_visibility(self, extrinsics: np.ndarray, point_cloud_path: str,
                        index_map: np.ndarray, visible_indices: np.ndarray,
                        depth_map: Optional[np.ndarray] = None,
                        element_type: str = 'point',
                        inverted_index: Optional[Dict] = None,
                        compressed: bool = True,
                        extra_hash_data: Optional[bytes] = None,
                        pixel_budget: Optional[int] = None) -> str:
        """
        Save visibility data to cache.

        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the geometry file
            index_map (np.ndarray): 2D index map (H x W)
            visible_indices (np.ndarray): 1D array of visible element IDs
            depth_map (np.ndarray, optional): Accepted for backward compatibility;
                ignored by the current cache format.
            element_type (str): Type of indexed elements ('point', 'face', or 'cell')
            inverted_index (dict, optional): CSR inverted index with keys
                'inv_ids', 'inv_offsets', 'inv_pixels'.
            compressed (bool): Whether to use compressed .npz format (default: False)
            extra_hash_data (bytes, optional): Additional bytes mixed into the hash
                (see _generate_cache_key).
            pixel_budget (int, optional): Render pixel budget that produced this map.
                Ignored by the cache key (see _generate_cache_key) but persisted in
                the archive metadata (0 = Native) so later sessions can recover the
                quality the map was rendered at.

        Returns:
            str: Path to the saved cache file
        """
        cache_path = self.get_cache_path(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        )

        # Ensure cache directory exists
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create cache directory {self.cache_dir}: {e}")
            return None

        # Enforce canonical dtypes to reduce RAM/disk usage
        try:
            index_map       = index_map.astype(np.int32, copy=False)
            visible_indices = np.asarray(visible_indices).astype(np.int32, copy=False)
        except Exception:
            pass

        try:
            save_index_map_archive(
                cache_path,
                index_map,
                visible_indices,
                element_type=element_type,
                compress=compressed,
                # Render quality that produced this map. 0 = Native (no budget);
                # absent in legacy archives, which loaders treat as unknown.
                pixel_budget=int(pixel_budget) if pixel_budget else 0,
            )
            return cache_path
        except Exception as e:
            print(f"Warning: Failed to save visibility cache to {cache_path}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached visibility data (both .npz and new .npy / .json formats)."""
        _CACHE_EXTS = ('.npz', '.idx.npy', '.vis.npy', '.dep.npy', '.meta.json')
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if any(filename.endswith(ext) for ext in _CACHE_EXTS):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        print(f"Warning: Failed to remove cache file {filename}: {e}")
    
    def get_cache_size(self) -> Tuple[int, int]:
        """
        Get the size of the cache directory.
        
        Returns:
            tuple: (number_of_files, total_size_in_bytes)
        """
        if not os.path.exists(self.cache_dir):
            return 0, 0
        
        _CACHE_EXTS = ('.npz', '.idx.npy', '.vis.npy', '.dep.npy', '.meta.json')
        file_count = 0
        total_size = 0

        for filename in os.listdir(self.cache_dir):
            if any(filename.endswith(ext) for ext in _CACHE_EXTS):
                file_count += 1
                file_path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(file_path)

        return file_count, total_size
