"""
Diagnose the depth test / projection issue.
"""
import sys
sys.path.insert(0, r"C:\Users\jordan\Documents\GitHub\CoralNet-Toolbox")

import numpy as np
import moderngl
import importlib
vm = importlib.import_module("coralnet_toolbox.MVAT.managers.VisibilityManager")

ctx = moderngl.create_context(standalone=True)
W, H = 512, 512

# ── Test 1: fullscreen quad with no depth test ────────────────────────────────
print("=== Test 1: fullscreen quad (no depth test, hardcoded color) ===")
ctx.disable(moderngl.DEPTH_TEST)

prog_flat = ctx.program(
    vertex_shader='''
        #version 330 core
        in vec2 pos;
        void main() { gl_Position = vec4(pos, 0.0, 1.0); }
    ''',
    fragment_shader='''
        #version 330 core
        out vec4 color;
        void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
    ''',
)
quad_vbo = ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4').tobytes())
quad_vao = ctx.vertex_array(prog_flat, [(quad_vbo, '2f', 'pos')])

fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4)],
                      depth_attachment=ctx.depth_texture((W, H)))
fbo.use()
ctx.clear(0,0,0,0, depth=1.0)
quad_vao.render(moderngl.TRIANGLE_STRIP)
ctx.finish()

raw = np.frombuffer(fbo.read(components=3, dtype='u1'), dtype=np.uint8).reshape(H,W,3)
print(f"  non-zero pixels: {(raw>0).any(2).sum()} / {W*H}  (expect {W*H})")
print(f"  center pixel   : {raw[H//2, W//2]}")

# ── Test 2: sphere WITHOUT depth test ────────────────────────────────────────
import pyvista as pv
print("\n=== Test 2: sphere, NO depth test ===")

class Fake:
    file_path = ""
    def get_mesh(self): return pv.Sphere(radius=1.0, theta_resolution=32, phi_resolution=32)
    def get_element_type(self): return 'face'

mgl_ctx = vm.VisibilityManager.setup_batch_moderngl_context(Fake(), None, W, H)
ctx2 = mgl_ctx['ctx']
ctx2.disable(moderngl.DEPTH_TEST)   # <-- NO depth test

K = np.array([[400,0,256],[0,400,256],[0,0,1]], dtype=np.float64)
R = np.eye(3); t = np.array([0.,0.,3.])
mvp = vm._build_mvp(K, R, t, W, H)

fbo2 = ctx2.framebuffer(color_attachments=[ctx2.texture((W,H),4)],
                        depth_attachment=ctx2.depth_texture((W,H)))
fbo2.use()
ctx2.clear(0,0,0,0, depth=1.0)
mgl_ctx['prog_low']['mvp'].write(mvp.tobytes())
mgl_ctx['vao_low'].render()
ctx2.finish()

raw2 = np.frombuffer(fbo2.read(components=3, dtype='u1'), dtype=np.uint8).reshape(H,W,3)
print(f"  non-zero pixels: {(raw2>0).any(2).sum()} / {W*H}")
print(f"  center pixel   : {raw2[H//2, W//2]}")

# ── Test 3: sphere WITH depth test but tighter near/far ───────────────────────
print("\n=== Test 3: sphere, depth test, near=0.1 far=100 ===")
ctx2.enable(moderngl.DEPTH_TEST)

mvp_tight = vm._build_mvp(K, R, t, W, H, near=0.1, far=100.0)
fbo3 = ctx2.framebuffer(color_attachments=[ctx2.texture((W,H),4)],
                        depth_attachment=ctx2.depth_texture((W,H)))
fbo3.use()
ctx2.clear(0,0,0,0, depth=1.0)
mgl_ctx['prog_low']['mvp'].write(mvp_tight.tobytes())
mgl_ctx['vao_low'].render()
ctx2.finish()

raw3 = np.frombuffer(fbo3.read(components=3, dtype='u1'), dtype=np.uint8).reshape(H,W,3)
print(f"  non-zero pixels: {(raw3>0).any(2).sum()} / {W*H}")
print(f"  center pixel   : {raw3[H//2, W//2]}")
