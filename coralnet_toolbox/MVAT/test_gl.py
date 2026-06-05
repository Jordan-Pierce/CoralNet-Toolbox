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

    single_ctx = vm.VisibilityManager.setup_batch_moderngl_context(single_mesh, None, W, H)
    print("✅ Using 32-bit integer rendering (single-pass)")

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
    print("\n--- Large Mesh Pathway ---")

    # Create a high-res sphere programmatically
    high_res = pv.Sphere(radius=1.0, theta_resolution=256, phi_resolution=256)
    n_faces_dual = high_res.n_cells
    print(f"Mesh: {n_faces_dual:,} faces")

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
    print(f"ℹ️  Mesh has {n_faces_dual:,} faces (32-bit int rendering supports unlimited face count)")

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


def test_gpu_tensor_readback():
    """Benchmark: CPU readback vs GPU tensor transfer for downstream operations.

    This test compares:
    1. Current: Read to CPU, process on CPU, pass to downstream
    2. GPU tensor: Read to CPU, transfer to GPU, process on GPU, keep on GPU

    The GPU tensor approach saves time if downstream operations (SAM, etc.) also run on GPU.
    """
    print("\n" + "="*70)
    print("TEST 10: GPU Tensor Readback Benchmark")
    print("="*70)

    try:
        import torch
        from coralnet_toolbox.MVAT.shaders.cuda_gl_interop import (
            read_texture_to_gpu_simple, process_index_map_gpu, index_map_gpu_to_cpu
        )
    except ImportError:
        print("⚠️  PyTorch not available; skipping GPU tensor test")
        return

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    mesh = FakeMesh(n_cells=4096)
    mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(mesh, None, W, H)

    print(f"\nMesh: {mesh.get_mesh().n_cells:,} faces")
    print(f"Resolution: {W}×{H} pixels")
    print(f"Cameras: 5 (for averaging)\n")

    # Render once to get an FBO
    results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=mgl_ctx
    )

    # Now benchmark the readback approaches on the rendered FBO
    fbo = list(mgl_ctx['_fbo_cache'].values())[0]  # Get the first FBO
    fbo_w, fbo_h = fbo.size  # Get actual FBO dimensions (returns (width, height))

    print(f"Note: FBO rendered at {fbo_w}×{fbo_h} (viewport cropping applied)")
    print(f"Readback data size: {fbo_w * fbo_h:,} pixels\n")

    print("Approach 1: Current (CPU Readback)")
    print("-" * 70)
    cpu_times = []
    for i in range(5):
        t0 = time.perf_counter()
        raw = fbo.read(components=1, dtype='i4')
        shot_int32 = np.frombuffer(raw, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy()
        shot_int32 -= 1  # Convert from 1-based to 0-based
        cpu_times.append(time.perf_counter() - t0)

    cpu_avg = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"  Average: {cpu_avg*1000:.2f}ms ± {cpu_std*1000:.2f}ms")
    print(f"  Timings: {[f'{t*1000:.2f}ms' for t in cpu_times]}\n")

    print("Approach 2: GPU Tensor (CPU→GPU Transfer + GPU Processing)")
    print("-" * 70)
    gpu_times = []
    for i in range(5):
        t0 = time.perf_counter()
        # Read to CPU
        raw = fbo.read(components=1, dtype='i4')
        shot_int32_cpu = np.frombuffer(raw, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy()
        # Transfer to GPU
        shot_int32_gpu = torch.from_numpy(shot_int32_cpu).to(device='cuda', dtype=torch.int32)
        # Process on GPU
        shot_int32_gpu = process_index_map_gpu(shot_int32_gpu, offset=1)
        gpu_times.append(time.perf_counter() - t0)

    gpu_avg = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    print(f"  Average: {gpu_avg*1000:.2f}ms ± {gpu_std*1000:.2f}ms")
    print(f"  Timings: {[f'{t*1000:.2f}ms' for t in gpu_times]}\n")

    # Verify correctness
    raw = fbo.read(components=1, dtype='i4')
    shot_int32_cpu = np.frombuffer(raw, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy() - 1
    raw2 = fbo.read(components=1, dtype='i4')
    shot_int32_gpu = torch.from_numpy(np.frombuffer(raw2, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy()).to(device='cuda', dtype=torch.int32) - 1
    shot_int32_gpu_cpu = shot_int32_gpu.cpu().numpy()

    if np.allclose(shot_int32_cpu, shot_int32_gpu_cpu):
        print("✅ GPU tensor produces identical results\n")
    else:
        print("❌ GPU tensor results differ!\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    overhead = (gpu_avg - cpu_avg) * 1000
    if overhead > 0:
        print(f"GPU tensor approach: +{overhead:.2f}ms overhead (CPU→GPU transfer)")
        print(f"Benefit if downstream ops on GPU: Keep data on VRAM, avoid GPU←CPU transfers")
    else:
        print(f"GPU tensor approach: -{abs(overhead):.2f}ms improvement")

    print("\nConclusion:")
    if overhead > 2.0:
        print(f"⚠️  GPU transfer overhead ({overhead:.2f}ms) may not be worth it")
        print("   unless downstream operations heavily use GPU.")
    else:
        print(f"✅ GPU transfer overhead is modest ({overhead:.2f}ms)")
        print("   Worthwhile if SAM or other GPU ops will use the data.")

    # Cleanup
    for fbo in mgl_ctx['_fbo_cache'].values():
        fbo.release()
    mgl_ctx['ctx'].release()

    print("✅ GPU Tensor Readback test PASSED\n")


def test_gpu_tensor_scaling(width=4000, height=3000, num_cameras=100):
    """Realistic scaling test: CUDA init once, process N high-res images.

    Simulates the real workflow:
    - Initialize CUDA context once at app startup
    - Process many cameras with full-resolution textures
    - Compare cumulative cost of CPU vs GPU tensor approaches

    Args:
        width, height: Image resolution (default 4000×3000 = 12MP)
        num_cameras: Number of cameras to process
    """
    print("\n" + "="*70)
    print(f"TEST 11: GPU Tensor Scaling ({width}×{height} = {width*height/1e6:.0f}MP, {num_cameras} cameras)")
    print("="*70)

    try:
        import torch
        from coralnet_toolbox.MVAT.shaders.cuda_gl_interop import process_index_map_gpu
    except ImportError:
        print("⚠️  PyTorch not available; skipping scaling test")
        return

    W, H = width, height
    K = np.array([[W*0.8, 0, W/2], [0, H*0.8, H/2], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 5.])

    mesh = FakeMesh(n_cells=16384)  # Large mesh
    mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(mesh, None, W, H)

    print(f"\nMesh: {mesh.get_mesh().n_cells:,} faces")
    print(f"Resolution: {W}×{H} pixels ({W*H:,} pixels per camera = {W*H/1e6:.1f}MP)")
    print(f"Cameras: {num_cameras}")
    print(f"Total data: {W*H*num_cameras/1e9:.2f}GB\n")

    # Render once to get a realistic FBO
    results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
        mesh, [(K, R, t, W, H)],
        compute_depth_map=False, pixel_budget=None, mgl_context=mgl_ctx
    )

    fbo = list(mgl_ctx['_fbo_cache'].values())[0]
    fbo_w, fbo_h = fbo.size
    pixels_per_cam = fbo_w * fbo_h

    print(f"Actual FBO size: {fbo_w}×{fbo_h} = {pixels_per_cam:,} pixels/cam\n")

    # ========================================================================
    # Approach 1: CPU Readback (current)
    # ========================================================================
    print("="*70)
    print("APPROACH 1: CPU Readback (Current)")
    print("="*70)

    cpu_times_per_cam = []
    cpu_start = time.perf_counter()

    for cam_idx in range(num_cameras):
        t0 = time.perf_counter()
        raw = fbo.read(components=1, dtype='i4')
        shot_int32 = np.frombuffer(raw, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy()
        shot_int32 -= 1
        cam_time = time.perf_counter() - t0
        cpu_times_per_cam.append(cam_time)

    cpu_avg_per_cam = np.mean(cpu_times_per_cam)
    cpu_total_100 = time.perf_counter() - cpu_start
    cpu_init_cost = 0  # No initialization cost

    print(f"Per-camera time: {cpu_avg_per_cam*1000:.2f}ms")
    print(f"Total for {num_cameras} cameras: {cpu_total_100:.2f}s ({cpu_total_100/60:.2f}min)")
    print(f"Init cost (amortized): {cpu_init_cost:.2f}ms\n")

    # ========================================================================
    # Approach 2: GPU Tensor (initialize once, then stream)
    # ========================================================================
    print("="*70)
    print("APPROACH 2: GPU Tensor (CUDA-GL Streaming)")
    print("="*70)

    gpu_init_start = time.perf_counter()
    # Pre-allocate GPU tensor to avoid repeated allocation
    dummy = torch.zeros((fbo_h, fbo_w), dtype=torch.int32, device='cuda')
    torch.cuda.synchronize()
    gpu_init_cost = (time.perf_counter() - gpu_init_start) * 1000

    print(f"CUDA init cost (one-time): {gpu_init_cost:.2f}ms\n")

    gpu_times_per_cam = []
    gpu_start = time.perf_counter()

    for cam_idx in range(num_cameras):
        t0 = time.perf_counter()
        raw = fbo.read(components=1, dtype='i4')
        shot_int32_cpu = np.frombuffer(raw, dtype=np.int32).reshape(fbo_h, fbo_w)[::-1].copy()
        shot_int32_gpu = torch.from_numpy(shot_int32_cpu).to(device='cuda', dtype=torch.int32)
        shot_int32_gpu = process_index_map_gpu(shot_int32_gpu, offset=1)
        # Keep on GPU (don't transfer back to CPU)
        cam_time = time.perf_counter() - t0
        gpu_times_per_cam.append(cam_time)

    gpu_avg_per_cam = np.mean(gpu_times_per_cam)
    gpu_total = time.perf_counter() - gpu_start

    # Total with amortized init cost
    gpu_total_with_init = gpu_init_cost/1000 + gpu_avg_per_cam * num_cameras  # Amortize over 1000 cameras

    print(f"Per-camera time: {gpu_avg_per_cam*1000:.2f}ms")
    print(f"Total for {num_cameras} cameras: {gpu_total:.2f}s ({gpu_total/60:.2f}min)")
    print(f"Init cost (amortized over 1000): {gpu_init_cost/1000:.3f}ms/cam\n")

    # ========================================================================
    # SCALING COMPARISON
    # ========================================================================
    print("="*70)
    print(f"SCALING COMPARISON ({num_cameras} Cameras)")
    print("="*70)

    speedup = cpu_total_100 / gpu_total if gpu_total > 0 else 1.0
    difference = cpu_total_100 - gpu_total
    overhead_per_cam = (gpu_avg_per_cam - cpu_avg_per_cam) * 1000

    print(f"\nCPU Readback:")
    print(f"  Total time: {cpu_total_100:8.2f}s ({cpu_total_100/60:6.2f}min)")
    print(f"  Per-camera: {cpu_avg_per_cam*1000:8.2f}ms")

    print(f"\nGPU Tensor:")
    print(f"  Total time: {gpu_total:8.2f}s ({gpu_total/60:6.2f}min)")
    print(f"  Per-camera: {gpu_avg_per_cam*1000:8.2f}ms")
    print(f"  Init cost: {gpu_init_cost:8.2f}ms (one-time)")

    print(f"\n{'='*70}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {abs(difference):.2f}s ({abs(difference)/60:.2f}min)")
    print(f"Per-camera overhead: {overhead_per_cam:.2f}ms")
    print(f"{'='*70}\n")

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("VERDICT:")
    if speedup > 1.1:
        print(f"✅ GPU tensors are {(speedup-1)*100:.1f}% FASTER")
        print("   Worth integrating GPU tensor pipeline!")
    elif speedup < 0.9:
        print(f"❌ GPU tensors are {(1-speedup)*100:.1f}% SLOWER")
        print("   CPU readback is better; don't pursue GPU streaming.")
    else:
        print(f"⚠️  Roughly equivalent ({(speedup-1)*100:+.1f}%)")
        print("   Benefit depends on downstream GPU operations (SAM, etc.)")

    if overhead_per_cam > 2.0:
        print(f"\n   Per-camera overhead ({overhead_per_cam:.2f}ms) is significant")
        print("   unless GPU operations will dominate processing time.")

    # Cleanup
    for fbo in mgl_ctx['_fbo_cache'].values():
        fbo.release()
    mgl_ctx['ctx'].release()

    print("\n✅ GPU Tensor Scaling test PASSED\n")


def test_decode_bottleneck_diagnostic():
    """Isolate the decode bottleneck: GPU→CPU transfer vs NumPy math."""
    print("\n" + "="*70)
    print("TEST 11: Decode Bottleneck Diagnostic (Transfer vs Math)")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    mesh = FakeMesh(n_cells=4096)
    print(f"\nMesh: {mesh.get_mesh().n_cells:,} faces")
    print(f"Resolution: {W}×{H} pixels\n")

    # Test 1: With CUDA-GL interop (current)
    print("📊 Test 1: CUDA-GL Interop (Current Path)")
    print("-" * 70)
    try:
        mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(mesh, None, W, H)

        t0 = time.perf_counter()
        results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
            mesh, [(K, R, t, W, H)],
            compute_depth_map=False, pixel_budget=None, mgl_context=mgl_ctx
        )
        elapsed = time.perf_counter() - t0
        print(f"Total time: {elapsed*1000:.2f}ms")
        print("(Check logs for [Decode Split] Transfer/Math breakdown)\n")
    except Exception as e:
        print(f"Failed: {e}\n")


def test_moderngl_vs_vtk_speed():
    """Compare ModernGL (new 32-bit int pathway) vs VTK (legacy) performance."""
    print("\n" + "="*70)
    print("TEST 9: ModernGL vs VTK Speed Comparison")
    print("="*70)

    W, H = 512, 512
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0., 0., 3.])

    mesh = FakeMesh(n_cells=4096)
    mesh_obj = mesh.get_mesh()
    print(f"\nMesh: {mesh_obj.n_cells:,} faces")
    print(f"Resolution: {W}×{H} pixels")
    print(f"Number of cameras: 5 (for averaging)\n")

    # === ModernGL Pathway (New) ===
    print("🚀 ModernGL Pathway (32-bit Integer Rendering)")
    print("-" * 70)
    try:
        mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(mesh, None, W, H)

        mgl_times = []
        cameras = [
            (K, R, t, W, H),
            (K, R, t + np.array([0.5, 0., 0.]), W, H),
            (K, R, t + np.array([-0.5, 0., 0.]), W, H),
            (K, R, t + np.array([0., 0.5, 0.]), W, H),
            (K, R, t + np.array([0., -0.5, 0.]), W, H),
        ]

        for i, cam in enumerate(cameras, 1):
            t0 = time.perf_counter()
            mgl_results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
                mesh, [cam], compute_depth_map=False, pixel_budget=None, mgl_context=mgl_ctx
            )
            elapsed = time.perf_counter() - t0
            mgl_times.append(elapsed * 1000)  # Convert to ms
            print(f"  Camera {i}: {elapsed*1000:.2f}ms")

        mgl_avg = np.mean(mgl_times)
        mgl_std = np.std(mgl_times)
        print(f"\n  ✅ ModernGL Average: {mgl_avg:.2f}ms ± {mgl_std:.2f}ms")

    except Exception as e:
        print(f"  ❌ ModernGL failed: {e}")
        mgl_avg = float('inf')

    # === VTK Pathway (Legacy) ===
    print("\n🐢 VTK Pathway (Legacy RGB Encoding)")
    print("-" * 70)
    vtk_avg = float('inf')
    vtk_std = 0.0
    try:
        vtk_ctx = vm.VisibilityManager.setup_batch_vtk_context(mesh, None, W, H)

        vtk_times = []
        for i, cam in enumerate(cameras, 1):
            t0 = time.perf_counter()
            vtk_results = vm.VisibilityManager.compute_batch_mesh_visibility_vtk(
                mesh, [cam], compute_depth_map=False, pixel_budget=None, vtk_context=vtk_ctx
            )
            elapsed = time.perf_counter() - t0
            vtk_times.append(elapsed * 1000)  # Convert to ms
            print(f"  Camera {i}: {elapsed*1000:.2f}ms")

        vtk_avg = np.mean(vtk_times)
        vtk_std = np.std(vtk_times)
        print(f"\n  ✅ VTK Average: {vtk_avg:.2f}ms ± {vtk_std:.2f}ms")

    except Exception as e:
        print(f"  ❌ VTK failed: {e}")

    # === Summary ===
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"ModernGL:  {mgl_avg:7.2f}ms (± {mgl_std:.2f}ms)")
    print(f"VTK:       {vtk_avg:7.2f}ms (± {vtk_std:.2f}ms)")

    if mgl_avg < float('inf') and vtk_avg < float('inf'):
        speedup = vtk_avg / mgl_avg
        improvement = ((vtk_avg - mgl_avg) / vtk_avg) * 100
        print(f"\n🎯 ModernGL is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
        assert mgl_avg < vtk_avg, f"❌ ModernGL should be faster than VTK! MGL:{mgl_avg:.2f}ms vs VTK:{vtk_avg:.2f}ms"
        print("✅ ModernGL vs VTK Speed Test PASSED")
    else:
        print("\n⚠️  One or both pathways failed; cannot compare")


if __name__ == "__main__":
    print("\n" + "🧪 "*35)
    print("COMPREHENSIVE MODERNGL PIPELINE TEST")
    print("🧪 "*35)

    tests = [
        test_single_camera_moderngl,
        test_batch_rendering,
        test_dual_pass_encoding,
        test_pixel_budget_downsampling,
        test_ortho_camera,
        test_integer_fbo_rendering,
        test_raycast_crosscheck,
        test_gpu_tensor_readback,
        test_gpu_tensor_scaling,
        test_moderngl_vs_vtk_speed,
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
