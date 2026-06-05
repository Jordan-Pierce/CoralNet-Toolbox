"""Final smoke test — viewport fix + full pipeline."""
import sys, time
sys.path.insert(0, r"C:\Users\jordan\Documents\GitHub\CoralNet-Toolbox")

import numpy as np
import importlib
vm = importlib.import_module("coralnet_toolbox.MVAT.managers.VisibilityManager")
import pyvista as pv

class FakeMesh:
    file_path = ""
    def get_mesh(self): return pv.Sphere(radius=1.0, theta_resolution=32, phi_resolution=32)
    def get_element_type(self): return 'face'

W, H = 512, 512
K = np.array([[400,0,256],[0,400,256],[0,0,1]], dtype=np.float64)
R = np.eye(3, dtype=np.float64)
t = np.array([0., 0., 3.])

mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(FakeMesh(), None, W, H)
print(f"Mesh faces: {mgl_ctx['n_cells']:,}  dual_pass={mgl_ctx['is_dual_pass']}")

t0 = time.perf_counter()
results = vm.VisibilityManager.compute_batch_mesh_visibility_moderngl(
    FakeMesh(), [(K, R, t, W, H)],
    compute_depth_map=False, pixel_budget=None, mgl_context=mgl_ctx,
)
elapsed = time.perf_counter() - t0

r = results[0]
idx = r['index_map']
face_px = int((idx >= 0).sum())
bg_px   = int((idx == -1).sum())
print(f"\nIndex map  : {idx.shape}  scale={r['scale_factor']}")
print(f"Face pixels: {face_px:,}  background: {bg_px:,}")
print(f"Unique faces: {len(r['visible_indices']):,}")
print(f"Render time: {elapsed*1000:.1f}ms")

assert face_px > 10000, f"❌ Too few face pixels: {face_px}"
assert bg_px   > 0,     "❌ No background pixels"
print("\n✅ Smoke test passed")

for fbo in mgl_ctx['_fbo_cache'].values(): fbo.release()
mgl_ctx['ctx'].release()
