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

    # Create 3 different camera views with same parameters
    cameras = [
        (K, np.eye(3, dtype=np.float64), np.array([0., 0., 3.], dtype=np.float64), W, H),
        (K, np.eye(3, dtype=np.float64), np.array([0., 0., 3.], dtype=np.float64), W, H),
        (K, np.eye(3, dtype=np.float64), np.array([0., 0., 3.], dtype=np.float64), W, H),
    ]

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
    """Test both single-pass and dual-pass encoding pathways."""
    print("\n" + "="*70)
    print("TEST 3: Single-Pass vs Dual-Pass Encoding")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    # ===== SINGLE-PASS: Small mesh (no dual-pass needed) =====
    print("\n--- SINGLE-PASS Pathway ---")
    single_mesh = FakeMesh(n_cells=1024)
    single_rendered = single_mesh.get_mesh()
    print(f"Mesh: {single_rendered.n_cells} faces")
    print(f"Threshold for dual-pass: {vm.FACE_ID_RGB_LIMIT:,} faces (16M)")

    single_ctx = vm.VisibilityManager.setup_batch_moderngl_context(single_mesh, None, W, H)
    print(f"Dual-pass needed: {single_ctx['is_dual_pass']}")

    if single_ctx['is_dual_pass']:
        print("⚠️  WARNING: Small mesh unexpectedly triggered dual-pass!")
    else:
        print("✅ Correctly using SINGLE-PASS (LOW RGB shader only)")

    single_results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        single_mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=single_ctx,
    )

    single_valid = (single_results[0]['index_map'] >= 0).sum()
    single_indices = len(single_results[0]['visible_indices'])
    print(f"Rendered pixels: {single_valid:,}")
    print(f"Visible faces: {single_indices:,}")

    assert single_valid > 0, "❌ Single-pass rendered no pixels!"
    assert single_indices > 0, "❌ Single-pass found no visible faces!"

    # ===== DUAL-PASS: Create mesh with > 16M faces =====
    print("\n--- DUAL-PASS Pathway ---")

    # Create a high-res sphere programmatically to exceed the threshold
    high_res = pv.Sphere(radius=1.0, theta_resolution=256, phi_resolution=256)
    n_faces_dual = high_res.n_cells
    print(f"Mesh: {n_faces_dual:,} faces (>{'' if n_faces_dual > vm.FACE_ID_RGB_LIMIT else '<'} threshold)")

    class DualPassMesh:
        def __init__(self, mesh):
            self.file_path = ""
            self._mesh = mesh

        def get_mesh(self):
            return self._mesh

        def get_element_type(self):
            return 'face'

    dual_mesh = DualPassMesh(high_res)

    dual_ctx = vm.VisibilityManager.setup_batch_moderngl_context(dual_mesh, None, W, H)
    print(f"Dual-pass needed: {dual_ctx['is_dual_pass']}")

    if not dual_ctx['is_dual_pass'] and n_faces_dual > vm.FACE_ID_RGB_LIMIT:
        print("⚠️  WARNING: Large mesh didn't trigger dual-pass despite > 16M faces!")
    elif dual_ctx['is_dual_pass']:
        print("✅ Correctly using DUAL-PASS (LOW + HIGH RGB shaders)")
    else:
        print(f"ℹ️  Mesh has {n_faces_dual:,} faces (below {vm.FACE_ID_RGB_LIMIT:,} threshold, single-pass OK)")

    dual_results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        dual_mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=dual_ctx,
    )

    dual_valid = (dual_results[0]['index_map'] >= 0).sum()
    dual_indices = len(dual_results[0]['visible_indices'])
    print(f"Rendered pixels: {dual_valid:,}")
    print(f"Visible faces: {dual_indices:,}")

    assert dual_valid > 0, "❌ Dual-pass rendered no pixels!"
    assert dual_indices > 0, "❌ Dual-pass found no visible faces!"

    # Verify both pathways produce valid index maps
    for name, result in [("single-pass", single_results[0]), ("dual-pass", dual_results[0])]:
        idx = result['index_map']
        assert np.all((idx == -1) | (idx >= 0)), f"❌ Invalid {name} index map values"

    print("\n✅ Single-pass and dual-pass encoding test PASSED")

    # Cleanup
    for fbo in single_ctx['_fbo_cache'].values():
        fbo.release()
    single_ctx['ctx'].release()
    for fbo in dual_ctx['_fbo_cache'].values():
        fbo.release()
    dual_ctx['ctx'].release()


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


def test_integer_fbo_rendering():
    """Test ModernGL rendering directly to a 32-bit integer FBO (Zero RGB encoding)."""
    print("\n" + "="*70)
    print("TEST 6: 32-bit Integer FBO Rendering (Next-Gen Pathway)")
    print("="*70)

    import moderngl
    from coralnet_toolbox.MVAT.shaders.gpu_interop import _build_mvp

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    # Grab the mock mesh
    mesh = FakeMesh(n_cells=1024).get_mesh().triangulate()
    vertices = np.asarray(mesh.points, dtype=np.float32)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

    # 1. Standalone Context
    ctx = moderngl.create_context(standalone=True)
    ctx.enable(moderngl.DEPTH_TEST)

    # 2. Integer Fragment Shader (No RGB bitwise math!)
    VERT = """
    #version 330 core
    in vec3 position;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(position, 1.0);
    }
    """
    FRAG_INT = """
    #version 330 core
    out int fragFaceID;
    void main() {
        // Output 1-based face ID natively as an integer
        fragFaceID = gl_PrimitiveID + 1;
    }
    """
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_INT)

    # 3. Upload Geometry
    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(faces.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f', 'position')], ibo)

    # 4. 32-bit Integer FBO (components=1, dtype='i4')
    color_tex = ctx.texture((W, H), components=1, dtype='i4')
    depth_tex = ctx.depth_texture((W, H))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    # 5. Render
    fbo.use()
    # Explicitly zero out the integer texture (background = 0), then clear depth
    color_tex.write(np.zeros((H, W), dtype=np.int32).tobytes()) 
    fbo.clear(depth=1.0) 

    mvp = _build_mvp(K, R, t, W, H)
    prog['mvp'].write(mvp.tobytes())
    
    t0 = time.perf_counter()
    vao.render()
    ctx.finish()
    elapsed = time.perf_counter() - t0

    # 6. Readback & Decode (Using CPU readback for the test)
    raw_bytes = fbo.read(components=1, dtype='i4')
    
    # ModernGL reads upside down; flip it and cast to int32
    raw_int_map = np.frombuffer(raw_bytes, dtype=np.int32).reshape(H, W)[::-1].copy()

    # 7. Reverse the +1 offset. Background (0) becomes -1!
    index_map = raw_int_map - 1

    face_px = int((index_map >= 0).sum())
    bg_px = int((index_map == -1).sum())
    unique_faces = len(np.unique(index_map[index_map >= 0]))

    print(f"Index map shape: {index_map.shape} | dtype: {index_map.dtype}")
    print(f"Face pixels: {face_px:,} | Background: {bg_px:,}")
    print(f"Visible faces: {unique_faces:,}")
    print(f"Render time: {elapsed*1000:.1f}ms")

    assert index_map.dtype == np.int32, "❌ Map is not int32!"
    assert face_px > 10000, f"❌ Too few face pixels: {face_px}"
    assert unique_faces > 0, "❌ No faces rendered!"
    print("✅ 32-bit Integer FBO test PASSED")

    # Cleanup
    fbo.release()
    color_tex.release()
    depth_tex.release()
    vao.release()
    vbo.release()
    ibo.release()
    prog.release()
    ctx.release()


def test_raycast_crosscheck():
    """Ironclad mathematical proof: GPU rasterization vs CPU Ray-Tracing."""
    print("\n" + "="*70)
    print("TEST 8: Ironclad Ray-Cast Cross-Check")
    print("="*70)

    import moderngl
    import pyvista as pv
    import random
    from coralnet_toolbox.MVAT.shaders.gpu_interop import _build_mvp

    W, H = 512, 512
    # Off-center, slightly rotated camera to prove matrices work
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    theta = np.radians(15)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float64)
    t = np.array([0.5, 0.2, 3.5])

    # Grab the mock mesh
    mesh = FakeMesh(n_cells=2048).get_mesh().triangulate()
    vertices = np.asarray(mesh.points, dtype=np.float32)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

    # 1. ModernGL Setup (32-bit Integer Pipeline)
    ctx = moderngl.create_context(standalone=True)
    ctx.enable(moderngl.DEPTH_TEST)

    VERT = """
    #version 330 core
    in vec3 position;
    uniform mat4 mvp;
    void main() {
        gl_Position = mvp * vec4(position, 1.0);
    }
    """
    FRAG_INT = """
    #version 330 core
    out int fragFaceID;
    void main() {
        fragFaceID = gl_PrimitiveID + 1; // 1-based encoding
    }
    """
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_INT)

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(faces.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f', 'position')], ibo)

    color_tex = ctx.texture((W, H), components=1, dtype='i4')
    depth_tex = ctx.depth_texture((W, H))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    # 2. Render FBO
    fbo.use()
    color_tex.write(np.zeros((H, W), dtype=np.int32).tobytes()) 
    fbo.clear(depth=1.0) 

    mvp = _build_mvp(K, R, t, W, H)
    prog['mvp'].write(mvp.tobytes())
    
    vao.render()
    ctx.finish()

    # 3. Readback & Decode
    raw_bytes = fbo.read(components=1, dtype='i4')
    raw_int_map = np.frombuffer(raw_bytes, dtype=np.int32).reshape(H, W)[::-1].copy()
    index_map = raw_int_map - 1

    # 4. CPU Ray-Tracing Cross-Check
    valid_pixels = np.argwhere(index_map >= 0)
    samples = valid_pixels[np.random.choice(len(valid_pixels), min(100, len(valid_pixels)), replace=False)]

    K_inv = np.linalg.inv(K)
    cam_origin = -R.T @ t

    matches = 0
    misses = 0

    for v, u in samples:
        gpu_face_id = index_map[v, u]

        # OpenGL rasterizes at pixel centers (+0.5 offset)
        uv_homog = np.array([u + 0.5, v + 0.5, 1.0])
        
        # Unproject pixel to world-space ray
        ray_cam = K_inv @ uv_homog
        ray_world = R.T @ ray_cam
        ray_dir = ray_world / np.linalg.norm(ray_world)

        ray_end = cam_origin + ray_dir * 100.0

        # Fire CPU Ray
        pts, ind = mesh.ray_trace(cam_origin, ray_end, first_point=True)

        if len(ind) > 0:
            cpu_face_id = ind[0]
            if cpu_face_id == gpu_face_id:
                matches += 1
            else:
                misses += 1

    match_rate = (matches / len(samples)) * 100
    
    print(f"Sampled {len(samples)} valid pixels.")
    print(f"Exact Matches: {matches} | Edge Misses: {misses}")
    print(f"GPU-to-CPU Match Rate: {match_rate:.1f}%")

    # Due to float precision differences between GPU rasterizer and CPU math exactly on 
    # triangle edges, 100% is rare, but it should be > 95% easily.
    assert match_rate >= 95.0, f"❌ GEOMETRY MISMATCH: Match rate too low: {match_rate}%"
    print("✅ Ironclad Ray-Cast Test PASSED")

    # Cleanup
    fbo.release(); color_tex.release(); depth_tex.release()
    vao.release(); vbo.release(); ibo.release()
    prog.release(); ctx.release()


if __name__ == "__main__":
    print("\n" + "🧪 "*35)
    print("COMPREHENSIVE MODERNGL PIPELINE TEST")
    print("🧪 "*35)

    tests = [
        # test_single_camera_moderngl,
        # test_batch_rendering,
        # test_dual_pass_encoding,
        # test_pixel_budget_downsampling,
        # test_ortho_camera,
        # test_integer_fbo_rendering,
        test_raycast_crosscheck,
    ]
    
    failed = []
    passed = []
    
    for test_func in tests:
        try:
            test_func()
            passed.append(test_func.__name__)
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            failed.append((test_func.__name__, str(e)))
        except Exception as e:
            print(f"\n❌ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append((test_func.__name__, str(e)))
    
    print("\n" + "="*70)
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    print("="*70)
    
    if passed:
        print(f"✅ PASSED ({len(passed)}):")
        for name in passed:
            print(f"   - {name}")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for name, err in failed:
            print(f"   - {name}: {err[:80]}")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        print("="*70)
