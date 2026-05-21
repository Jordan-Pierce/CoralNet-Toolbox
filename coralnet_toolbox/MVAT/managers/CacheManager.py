"""
CacheManager for MVAT Visibility Data

Handles persistent caching of index maps and visible indices to disk.
Uses MD5 hashing of camera parameters and point cloud paths for cache keys.
"""
import io
import os
import json
import hashlib
import functools
import numpy as np
from typing import Optional, Tuple, Dict

# Optional fast compression backend.  blosc2 with the lz4 codec is typically
# 3-5× faster than gzip at similar compression ratios.  Falls back gracefully
# when the package is not installed.
try:
    import blosc2 as _blosc2
    _HAS_BLOSC2 = True
except ImportError:
    _blosc2 = None
    _HAS_BLOSC2 = False


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


# ---------------------------------------------------------------------------
# Fast array I/O helpers
# ---------------------------------------------------------------------------

def _save_npy_fast(arr: np.ndarray, path: str) -> None:
    """Save a numpy array to *path* using blosc2/lz4 if available, else plain .npy."""
    if _HAS_BLOSC2:
        # blosc2 with lz4 is typically 3-5× faster than gzip at similar ratios.
        buf = io.BytesIO()
        np.save(buf, arr)
        compressed = _blosc2.compress(buf.getvalue(), codec=_blosc2.Codec.LZ4, clevel=1)
        with open(path, 'wb') as f:
            f.write(compressed)
    else:
        np.save(path, arr)


def _load_npy_fast(path: str, mmap_mode: Optional[str] = None) -> np.ndarray:
    """Load an array saved by _save_npy_fast, decompressing blosc2 if needed."""
    if _HAS_BLOSC2:
        with open(path, 'rb') as f:
            raw = f.read()
        try:
            decompressed = _blosc2.decompress(raw)
            return np.load(io.BytesIO(decompressed))
        except Exception:
            # Not a blosc2 file — fall through to plain numpy load
            pass
    return np.load(path, mmap_mode=mmap_mode)


# ---------------------------------------------------------------------------
# New-format helpers: save/load as separate .npy + .json files
# (faster than .npz for random access; supports OS mmap when uncompressed)
# ---------------------------------------------------------------------------

def _npy_paths(base: str) -> dict:
    """Return the canonical file paths for a given base (no extension)."""
    return {
        'meta': base + '.meta.json',
        'idx':  base + '.idx.npy',
        'vis':  base + '.vis.npy',
        'dep':  base + '.dep.npy',
    }


def _tmp_path(p: str) -> str:
    """
    Return an atomic-write temp path for *p* that preserves the file extension.

    np.save() auto-appends '.npy' when the destination doesn't already end in
    '.npy', which breaks the rename step.  Keeping the extension in the temp
    name avoids this on both Windows and POSIX.
    """
    if p.endswith('.npy'):
        return p[:-4] + '_tmp.npy'
    if p.endswith('.json'):
        return p[:-5] + '_tmp.json'
    return p + '_tmp'


def _save_npy_format(base: str, index_map: np.ndarray,
                     visible_indices: np.ndarray,
                     depth_map: Optional[np.ndarray],
                     element_type: str,
                     **extra_meta) -> bool:
    """
    Atomically write index_map, visible_indices, optional depth_map, and metadata 
    as separate files rooted at *base*. Returns True on success.
    """
    paths = _npy_paths(base)
    tmp = {k: _tmp_path(v) for k, v in paths.items()}
    try:
        _save_npy_fast(index_map, tmp['idx'])
        _save_npy_fast(visible_indices, tmp['vis'])
        
        meta = {'element_type': element_type, 'has_depth_map': False}
        if depth_map is not None:
            _save_npy_fast(depth_map, tmp['dep'])
            meta['has_depth_map'] = True
            
        # Merge in any extra metadata (e.g., ortho scale_factor)
        meta.update(extra_meta)
        
        with open(tmp['meta'], 'w') as f:
            json.dump(meta, f)
            
        # Atomic rename
        os.replace(tmp['idx'],  paths['idx'])
        os.replace(tmp['vis'],  paths['vis'])
        if depth_map is not None:
            os.replace(tmp['dep'], paths['dep'])
        os.replace(tmp['meta'], paths['meta'])
        return True
    except Exception as e:
        print(f"Warning: fast-format save failed ({e}); tmp files will be cleaned up")
        for p in tmp.values():
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        return False


def _load_npy_format(base: str) -> Optional[Dict]:
    """
    Load from the fast .npy format if the sentinel meta file exists.
    Returns None if not found or on error.
    """
    paths = _npy_paths(base)
    if not os.path.exists(paths['meta']):
        return None
    try:
        with open(paths['meta']) as f:
            meta = json.load(f)
            
        mmap = None if _HAS_BLOSC2 else 'r'
        index_map = _load_npy_fast(paths['idx'], mmap_mode=mmap).astype(np.int32, copy=False)
        visible_indices = _load_npy_fast(paths['vis'], mmap_mode=mmap).astype(np.int32, copy=False)
        
        depth_map = None
        if meta.get('has_depth_map') and os.path.exists(paths['dep']):
            depth_map = _load_npy_fast(paths['dep'], mmap_mode=mmap).astype(np.float16, copy=False)
            
        result = {
            'index_map':      index_map,
            'visible_indices': visible_indices,
            'depth_map':      depth_map,
            'element_type':   meta.get('element_type', 'point'),
            'inverted_index': None,
        }
        
        # Inject any additional metadata stored (like scale_factor)
        for k, v in meta.items():
            if k not in result and k != 'has_depth_map':
                result[k] = v
                
        return result
    except Exception as e:
        print(f"Warning: fast-format load failed for {base}: {e}")
        return None


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
            str: Canonical cache key path (.npz anchor used by both legacy and split formats)
        """
        cache_key = self._generate_cache_key(
            extrinsics, point_cloud_path, element_type, extra_hash_data, pixel_budget,
        )
        return os.path.join(self.cache_dir, f"{cache_key}.npz")

    def _cache_path_exists(self, cache_path: str) -> bool:
        """Return True if either the .npz or the split .npy form for *cache_path* is on disk."""
        if os.path.exists(cache_path):
            return True

        npy_base = os.path.splitext(cache_path)[0]
        paths = _npy_paths(npy_base)
        return (
            os.path.exists(paths['meta'])
            and os.path.exists(paths['idx'])
            and os.path.exists(paths['vis'])
        )

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

        npy_base = os.path.splitext(cache_path)[0]

        # 1. Try the fast .npy format first
        result = _load_npy_format(npy_base)
        if result is not None:
            result['cache_path'] = cache_path
            return result

        # 2. Fall back to legacy .npz
        if not os.path.exists(cache_path):
            return None

        try:
            try:
                data = np.load(cache_path, allow_pickle=True, mmap_mode='r')
            except Exception:
                data = np.load(cache_path, allow_pickle=True)
            result = {
                'index_map': data['index_map'].astype(np.int32, copy=False),
                'visible_indices': np.asarray(data['visible_indices']).astype(np.int32, copy=False),
                'scale_factor': float(data['scale_factor']) if 'scale_factor' in data else float(scale_factor),
                'element_type': str(data['element_type']) if 'element_type' in data else element_type,
                'cache_path': cache_path,
            }
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

        npy_base = os.path.splitext(cache_path)[0]

        # 1. Primary: fast .npy format (blosc2/lz4 if available, else plain npy)
        if not compressed:
            ok = _save_npy_format(
                npy_base, 
                index_map, 
                visible_indices, 
                depth_map=None, 
                element_type=element_type,
                scale_factor=float(scale_factor)
            )
            if ok:
                return cache_path

        # 2. Legacy / compressed .npz fallback
        save_dict = {
            'index_map': index_map,
            'visible_indices': visible_indices,
            'scale_factor': float(scale_factor),
            'element_type': element_type,
        }

        temp_path = os.path.splitext(cache_path)[0] + '_tmp.npz'
        try:
            if compressed:
                np.savez_compressed(temp_path, **save_dict)
            else:
                np.savez(temp_path, **save_dict)
            os.replace(temp_path, cache_path)
            return cache_path
        except Exception as e:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
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
        npy_base   = os.path.splitext(cache_path)[0]  # strip .npz for new-format paths

        # ------------------------------------------------------------------
        # 1. Try the fast .npy format first (written by new saves)
        # ------------------------------------------------------------------
        result = _load_npy_format(npy_base)
        if result is not None:
            result['cache_path'] = cache_path
            return result

        # ------------------------------------------------------------------
        # 2. Fall back to legacy .npz (mmap_mode='r' works for uncompressed
        #    zip entries, deferring actual disk reads until array access)
        # ------------------------------------------------------------------
        if not os.path.exists(cache_path):
            return None

        try:
            try:
                data = np.load(cache_path, allow_pickle=True, mmap_mode='r')
            except Exception:
                # Some numpy versions / platforms reject mmap on npz — fall back
                data = np.load(cache_path, allow_pickle=True)

            result = {
                'index_map': data['index_map'].astype(np.int32, copy=False),
                'visible_indices': np.asarray(data['visible_indices']).astype(np.int32, copy=False)
            }
            # depth_map is optional in older caches
            if 'depth_map' in data:
                result['depth_map'] = data['depth_map'].astype(np.float16, copy=False)
            else:
                result['depth_map'] = None

            # element_type: load from file or use provided parameter for backward compat
            if 'element_type' in data:
                result['element_type'] = str(data['element_type'])
            else:
                result['element_type'] = element_type

            # CSR inverted index (new; absent in old cache files)
            if 'inv_ids' in data and 'inv_offsets' in data and 'inv_pixels' in data:
                result['inverted_index'] = {
                    'inv_ids':     data['inv_ids'],
                    'inv_offsets': data['inv_offsets'],
                    'inv_pixels':  data['inv_pixels'],
                }
            else:
                result['inverted_index'] = None  # backward compat: regenerated on next compute

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
                        compressed: bool = False,
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
                See _generate_cache_key for backward-compatibility semantics.

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

        npy_base = os.path.splitext(cache_path)[0]  # strip .npz

        # ------------------------------------------------------------------
        # Primary: fast .npy format (blosc2/lz4 if available, else plain npy
        # with OS mmap support).  Falls back to legacy .npz on error.
        # ------------------------------------------------------------------
        if not compressed:
            ok = _save_npy_format(npy_base, index_map, visible_indices, depth_map, element_type)
            if ok:
                return cache_path  # canonical path unchanged for external callers

        # ------------------------------------------------------------------
        # Legacy / compressed .npz fallback
        # NOTE: Inverted index and depth maps are no longer saved to disk.
        # ------------------------------------------------------------------
        save_dict = {
            'index_map':      index_map,
            'visible_indices': visible_indices,
            'element_type':   element_type,
        }

        # Atomic write: temp file → rename
        temp_path = os.path.splitext(cache_path)[0] + '_tmp.npz'
        try:
            if compressed:
                np.savez_compressed(temp_path, **save_dict)
            else:
                np.savez(temp_path, **save_dict)
            os.replace(temp_path, cache_path)
            return cache_path
        except Exception as e:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
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
