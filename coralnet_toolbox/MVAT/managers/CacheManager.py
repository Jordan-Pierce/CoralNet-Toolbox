"""
CacheManager for MVAT Visibility Data

Handles persistent caching of index maps and visible indices to disk.
Uses MD5 hashing of camera parameters and point cloud paths for cache keys.
"""
import os
import hashlib
import numpy as np
from typing import Optional, Tuple, Dict


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CacheManager:
    """
    Manages persistent caching of visibility data (index maps and visible indices).
    
    Cache files are stored in `.mvat_cache/visibility/` relative to the project root.
    Each cache file is named using an MD5 hash of the camera extrinsics and point cloud path.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the CacheManager.
        
        Args:
            project_root (str): Root directory of the project where cache will be stored.
        """
        self.project_root = project_root
        self.cache_dir = os.path.join(project_root, '.mvat_cache', 'visibility')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, extrinsics: np.ndarray, point_cloud_path: str) -> str:
        """
        Generate a unique cache key based on camera extrinsics and point cloud path.
        
        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix (4x4)
            point_cloud_path (str): Path to the point cloud file
            
        Returns:
            str: MD5 hash string to use as cache key
        """
        # Combine extrinsics and path into a single string for hashing
        extrinsics_bytes = extrinsics.tobytes()
        path_bytes = point_cloud_path.encode('utf-8')
        
        # Create MD5 hash
        hash_obj = hashlib.md5()
        hash_obj.update(extrinsics_bytes)
        hash_obj.update(path_bytes)
        
        return hash_obj.hexdigest()
    
    def get_cache_path(self, extrinsics: np.ndarray, point_cloud_path: str) -> str:
        """
        Get the full path to the cache file for given parameters.
        
        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the point cloud file
            
        Returns:
            str: Full path to the cache file (.npz)
        """
        cache_key = self._generate_cache_key(extrinsics, point_cloud_path)
        return os.path.join(self.cache_dir, f"{cache_key}.npz")
    
    def load_visibility(self, extrinsics: np.ndarray, point_cloud_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load visibility data from cache if it exists.
        
        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the point cloud file
            
        Returns:
            dict or None: Dictionary with 'index_map' and 'visible_indices' if cache exists, None otherwise
        """
        cache_path = self.get_cache_path(extrinsics, point_cloud_path)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Load compressed numpy archive
            data = np.load(cache_path)
            
            result = {
                'index_map': data['index_map'],
                'visible_indices': data['visible_indices']
            }
            # depth_map is optional in older caches
            if 'depth_map' in data:
                result['depth_map'] = data['depth_map']
            else:
                result['depth_map'] = None

            return result
        except Exception as e:
            print(f"Warning: Failed to load visibility cache from {cache_path}: {e}")
            return None
    
    def save_visibility(self, extrinsics: np.ndarray, point_cloud_path: str, 
                        index_map: np.ndarray, visible_indices: np.ndarray,
                        depth_map: Optional[np.ndarray] = None) -> str:
        """
        Save visibility data to cache.
        
        Args:
            extrinsics (np.ndarray): Camera extrinsic matrix
            point_cloud_path (str): Path to the point cloud file
            index_map (np.ndarray): 2D index map (H x W)
            visible_indices (np.ndarray): 1D array of visible point IDs
            
        Returns:
            str: Path to the saved cache file
        """
        cache_path = self.get_cache_path(extrinsics, point_cloud_path)
        
        try:
            # Save as compressed numpy archive
            if depth_map is None:
                np.savez_compressed(
                    cache_path,
                    index_map=index_map,
                    visible_indices=visible_indices
                )
            else:
                np.savez_compressed(
                    cache_path,
                    index_map=index_map,
                    visible_indices=visible_indices,
                    depth_map=depth_map
                )

            return cache_path
        except Exception as e:
            print(f"Warning: Failed to save visibility cache to {cache_path}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached visibility data."""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.npz'):
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
        
        file_count = 0
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npz'):
                file_count += 1
                file_path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(file_path)
        
        return file_count, total_size
