import os

import numpy as np

from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager


def test_visibility_cache_exists_for_split_format(tmp_path):
    cache_manager = CacheManager(str(tmp_path))
    extrinsics = np.eye(4, dtype=np.float64)
    point_cloud_path = os.path.join(str(tmp_path), "scene", "points.ply")

    index_map = np.array([[1, -1], [2, 3]], dtype=np.int32)
    visible_indices = np.array([1, 2, 3], dtype=np.int32)

    cache_path = cache_manager.save_visibility(
        extrinsics,
        point_cloud_path,
        index_map,
        visible_indices,
    )

    assert cache_path is not None

    npz_path = cache_manager.get_cache_path(extrinsics, point_cloud_path)
    npy_base = os.path.splitext(npz_path)[0]

    assert not os.path.exists(npz_path)
    assert os.path.exists(npy_base + ".meta.json")
    assert os.path.exists(npy_base + ".idx.npy")
    assert os.path.exists(npy_base + ".vis.npy")
    assert cache_manager.has_visibility_cache(extrinsics, point_cloud_path)

    loaded = cache_manager.load_visibility(extrinsics, point_cloud_path)
    assert loaded is not None
    assert np.array_equal(loaded["index_map"], index_map)
    assert np.array_equal(loaded["visible_indices"], visible_indices)
    assert loaded["depth_map"] is None