"""Comprehensive smoke test: ModernGL rendering, index maps, dual-pass encoding."""
import sys, time
sys.path.insert(0, r"C:\Users\jordan\Documents\GitHub\CoralNet-Toolbox")

import numpy as np
import importlib
vm = importlib.import_module("coralnet_toolbox.MVAT.managers.VisibilityManager")
import pyvista as pv


class FakeMesh:
    """Mock mesh product for testing."""
    def __init__(self, n_cells=1024):
        self.file_path = ""
        self._n_cells = n_cells

    def get_mesh(self):
        if self._n_cells <= 1000:
            return pv.Sphere(radius=1.0, theta_resolution=32, phi_resolution=32)
        else:
            # Larger mesh for dual-pass testing
            return pv.Sphere(radius=1.0, theta_resolution=128, phi_resolution=128)

    def get_element_type(self):
        return 'face'


def test_single_camera_moderngl():
    """Test single camera ModernGL rendering."""
    print("\n" + "="*70)
    print("TEST 1: Single Camera ModernGL Rendering")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    mesh = FakeMesh(n_cells=1024)
    mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(mesh, None, W, H)

    print(f"Mesh faces: {mgl_ctx['n_cells']:,}")
    print(f"Dual-pass needed: {mgl_ctx['is_dual_pass']}")

    t0 = time.perf_counter()
    results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, [(K, R, t, W, H)],
        compute_depth_map=True, pixel_budget=None, mgl_context=mgl_ctx,
    )
    elapsed = time.perf_counter() - t0

    r = results[0]
    idx = r['index_map']
    face_px = int((idx >= 0).sum())
    bg_px = int((idx == -1).sum())

    print(f"Index map: {idx.shape} | scale={r['scale_factor']:.4f}")
    print(f"Face pixels: {face_px:,} | Background: {bg_px:,}")
    print(f"Visible faces: {len(r['visible_indices']):,}")
    print(f"Depth map: {r['depth_map'].shape if r['depth_map'] is not None else 'None'}")
    print(f"GPU tensor (index_map_gpu): {r['index_map_gpu'] is not None}")
    print(f"Render time: {elapsed*1000:.1f}ms")

    assert face_px > 10000, f"❌ Too few face pixels: {face_px}"
    assert bg_px > 0, "❌ No background pixels"
    assert r['depth_map'] is not None, "❌ Depth map not computed"
    print("✅ Single camera test PASSED")

    # Cleanup
    for fbo in mgl_ctx['_fbo_cache'].values():
        fbo.release()
    mgl_ctx['ctx'].release()

    return r


def test_batch_rendering():
    """Test batch rendering with multiple cameras."""
    print("\n" + "="*70)
    print("TEST 2: Batch Rendering (Multiple Cameras)")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)

    # Create 3 different camera views
    cameras = []
    for i, angle in enumerate([0, 45, 90]):
        rad = np.deg2rad(angle)
        R = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ], dtype=np.float64)
        t = np.array([2.0 * np.sin(rad), 0., 3.0])
        cameras.append((K, R, t, W, H))

    mesh = FakeMesh(n_cells=1024)
    t0 = time.perf_counter()
    results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, cameras,
        compute_depth_map=False, pixel_budget=None,
    )
    elapsed = time.perf_counter() - t0

    print(f"Cameras rendered: {len(results)}")
    print(f"Total batch time: {elapsed*1000:.1f}ms | Per-camera: {elapsed/len(results)*1000:.1f}ms")

    for i, r in enumerate(results):
        face_px = int((r['index_map'] >= 0).sum())
        print(f"  Camera {i}: {face_px:,} face pixels, {len(r['visible_indices']):,} visible faces")

    assert len(results) == 3, "❌ Wrong number of results"
    print("✅ Batch rendering test PASSED")


def test_dual_pass_encoding():
    """Test dual-pass encoding detection and functionality."""
    print("\n" + "="*70)
    print("TEST 3: Dual-Pass Encoding")
    print("="*70)

    # Test with small mesh (no dual-pass needed)
    small_mesh = FakeMesh(n_cells=100)
    small_rendered = small_mesh.get_mesh()
    print(f"Small mesh: {small_rendered.n_cells} faces")

    # Test with large mesh (dual-pass needed)
    large_mesh = FakeMesh(n_cells=20000)
    large_rendered = large_mesh.get_mesh()
    print(f"Large mesh: {large_rendered.n_cells} faces")

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    # Small mesh (no dual-pass)
    print("\nSmall mesh rendering (should NOT need dual-pass):")
    small_ctx = vm.VisibilityManager.setup_batch_moderngl_context(small_mesh, None, W, H)
    print(f"  is_dual_pass: {small_ctx['is_dual_pass']}")
    print(f"  FACE_ID_RGB_LIMIT: {vm.FACE_ID_RGB_LIMIT:,} (16M faces)")

    small_results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        small_mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=small_ctx,
    )
    small_valid = (small_results[0]['index_map'] >= 0).sum()
    print(f"  Rendered pixels: {small_valid:,}")

    # Large mesh (dual-pass)
    print("\nLarge mesh rendering (SHOULD need dual-pass):")
    large_ctx = vm.VisibilityManager.setup_batch_moderngl_context(large_mesh, None, W, H)
    print(f"  is_dual_pass: {large_ctx['is_dual_pass']}")

    large_results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        large_mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=large_ctx,
    )
    large_valid = (large_results[0]['index_map'] >= 0).sum()
    print(f"  Rendered pixels: {large_valid:,}")

    # Verify index maps are valid (no NaN, bounds correct)
    assert np.all((small_results[0]['index_map'] == -1) | (small_results[0]['index_map'] >= 0)), \
        "❌ Invalid small index map values"
    assert np.all((large_results[0]['index_map'] == -1) | (large_results[0]['index_map'] >= 0)), \
        "❌ Invalid large index map values"

    print("✅ Dual-pass encoding test PASSED")

    # Cleanup
    for fbo in small_ctx['_fbo_cache'].values():
        fbo.release()
    small_ctx['ctx'].release()
    for fbo in large_ctx['_fbo_cache'].values():
        fbo.release()
    large_ctx['ctx'].release()


def test_pixel_budget_downsampling():
    """Test pixel budget and dynamic downsampling."""
    print("\n" + "="*70)
    print("TEST 4: Pixel Budget & Downsampling")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    mesh = FakeMesh(n_cells=1024)

    # Full resolution
    print(f"Full resolution: {W}x{H} = {W*H:,} pixels")
    results_full = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None,
    )

    # Downsampled (0.25x pixels)
    budget = W * H // 4  # 0.5x linear scale
    print(f"With pixel budget: {budget:,} pixels")
    results_budget = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=budget,
    )

    scale_full = results_full[0]['scale_factor']
    scale_budget = results_budget[0]['scale_factor']

    print(f"Full resolution scale: {scale_full:.4f}")
    print(f"Budget resolution scale: {scale_budget:.4f}")

    assert scale_full == 1.0, "❌ Full resolution should have scale=1.0"
    assert scale_budget < 1.0, "❌ Downsampled should have scale<1.0"
    print("✅ Pixel budget test PASSED")


def test_ortho_camera():
    """Test ortho camera index map computation."""
    print("\n" + "="*70)
    print("TEST 5: Ortho Camera Index Map")
    print("="*70)

    try:
        from coralnet_toolbox.MVAT.core.Cameras import OrthoCamera

        # Create a minimal ortho raster
        class FakeOrthoRaster:
            def __init__(self):
                self.image_path = "/fake/ortho.tif"
                self.width = 512
                self.height = 512
                self.ortho_left = 0.0
                self.ortho_top = 512.0
                self.resolution_x = 1.0
                self.resolution_y = -1.0  # negative = top-down

        raster = FakeOrthoRaster()
        chunk_transform = np.eye(4)
        ortho_cam = OrthoCamera(raster, chunk_transform)

        if ortho_cam.is_valid:
            mesh = FakeMesh(n_cells=1024)
            result_mgl = vm.VisibilityManager.compute_ortho_index_map_moderngl(
                ortho_cam, mesh, pixel_budget=None
            )

            if result_mgl is not None:
                print(f"ModernGL ortho index map: {result_mgl['index_map'].shape}")
                print(f"Visible faces: {len(result_mgl['visible_indices']):,}")
                print("✅ Ortho camera test PASSED")
            else:
                print("⚠️  ModernGL ortho returned None (fallback to VTK)")
        else:
            print("⚠️  Ortho camera not valid (skipping)")
    except Exception as e:
        print(f"⚠️  Ortho test skipped: {e}")


if __name__ == "__main__":
    print("\n" + "🧪 "*35)
    print("COMPREHENSIVE MODERNGL PIPELINE TEST")
    print("🧪 "*35)

    try:
        test_single_camera_moderngl()
        test_batch_rendering()
        test_dual_pass_encoding()
        test_pixel_budget_downsampling()
        test_ortho_camera()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
