"""GPU interop utilities: CUDA-GL, OpenGL function loading, MVP matrix building."""

from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# CUDA-GL interop state
# None = untested, True = working, False = unavailable/failed
# ---------------------------------------------------------------------------
_CUDA_GL_INTEROP_OK: Optional[bool] = None
_cudart = None  # cached ctypes handle to libcudart


def _load_cudart():
    """Load libcudart via ctypes.

    Search order:
      1. PyTorch's bundled lib directory — the most reliable source since we know
         CUDA is already working through torch (conda/pip installs keep the DLL here).
      2. System PATH / ldconfig via ctypes.util.find_library.
      3. Hard-coded common names as a last resort.
    """
    global _cudart
    if _cudart is not None:
        return _cudart

    import ctypes, ctypes.util, os, platform

    candidates = []

    # 1. PyTorch's own lib directory (works on conda/pip Windows and Linux)
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            for fname in os.listdir(torch_lib):
                if 'cudart' in fname.lower():
                    candidates.append(os.path.join(torch_lib, fname))
    except Exception:
        pass

    # 2. System-level search
    sys_name = ctypes.util.find_library('cudart')
    if sys_name:
        candidates.append(sys_name)

    # 3. Hard-coded fallbacks (CUDA 12 changed naming: cudart64_12.dll not cudart64_120.dll)
    if platform.system() == 'Windows':
        candidates += [
            'cudart64_12.dll', 'cudart64_11.dll',
            'cudart64_120.dll', 'cudart64_110.dll', 'cudart64_100.dll',
            'cudart64.dll',
        ]
    else:
        candidates += [
            'libcudart.so', 'libcudart.so.12', 'libcudart.so.11', 'libcudart.so.10',
        ]

    for c in candidates:
        try:
            _cudart = ctypes.CDLL(c)
            return _cudart
        except OSError:
            pass

    raise OSError("Could not find libcudart. CUDA-GL interop unavailable.")


def _resolve_gl_fns():
    """Resolve OpenGL extension function pointers via the platform proc-address loader.

    Returns a namespace object with callable GL functions.  Raises on failure.
    Must be called while the target GL context is current.
    """
    import ctypes, platform, types

    ns = types.SimpleNamespace()

    if platform.system() == 'Windows':
        _gl32 = ctypes.WinDLL('opengl32')
        _get_proc = _gl32.wglGetProcAddress
        _get_proc.restype  = ctypes.c_void_p
        _get_proc.argtypes = [ctypes.c_char_p]
        ns.glReadPixels = _gl32.glReadPixels
        ns.glFinish     = _gl32.glFinish
    else:
        import ctypes.util
        _libgl = ctypes.CDLL(ctypes.util.find_library('GL') or 'libGL.so')
        _get_proc = _libgl.glXGetProcAddressARB
        _get_proc.restype  = ctypes.c_void_p
        _get_proc.argtypes = [ctypes.c_char_p]
        ns.glReadPixels = _libgl.glReadPixels
        ns.glFinish     = _libgl.glFinish

    def _ext(name, restype, *argtypes):
        ptr = _get_proc(name.encode())
        if not ptr:
            raise RuntimeError(f"proc-address lookup failed for '{name}'")
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)

    ns.glGenBuffers    = _ext('glGenBuffers',    None, ctypes.c_int,    ctypes.POINTER(ctypes.c_uint))
    ns.glBindBuffer    = _ext('glBindBuffer',    None, ctypes.c_uint,   ctypes.c_uint)
    ns.glBufferData    = _ext('glBufferData',    None, ctypes.c_uint,   ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint)
    ns.glDeleteBuffers = _ext('glDeleteBuffers', None, ctypes.c_int,    ctypes.POINTER(ctypes.c_uint))

    ns.glReadPixels.restype  = None
    ns.glReadPixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    ns.glFinish.restype  = None
    ns.glFinish.argtypes = []

    # GL constants
    ns.GL_PIXEL_PACK_BUFFER = 0x88EB
    ns.GL_STREAM_READ       = 0x88E1
    ns.GL_RED_INTEGER       = 0x8D94
    ns.GL_INT               = 0x1404

    return ns


def _pbo_cuda_readback(gl: 'types.SimpleNamespace', cudart, width: int, height: int,
                       flip: bool = True) -> Optional['torch.Tensor']:
    """Read the current GL read framebuffer into a CUDA int32 tensor via PBO.

    VRAM → PBO (GPU-side) → CUDA map → D2D copy → torch tensor.  No PCIe.
    Returns None on any error (caller should fall back to CPU screenshot).

    ``flip`` controls the GL bottom-to-top → top-to-bottom correction. Pass
    ``flip=False`` when the vertex shader already bakes the vertical flip into
    clip space (the MVAT rasterizer does), otherwise the image is double-flipped.
    """
    import ctypes
    import torch

    n_bytes = width * height * 4
    pbo      = ctypes.c_uint(0)
    resource = ctypes.c_void_p(0)
    mapped   = False
    try:
        gl.glGenBuffers(1, ctypes.byref(pbo))
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo.value)
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, n_bytes, None, gl.GL_STREAM_READ)
        gl.glReadPixels(0, 0, width, height, gl.GL_RED_INTEGER, gl.GL_INT, ctypes.c_void_p(0))
        gl.glFinish()

        err = cudart.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), pbo.value, 1)
        if err: raise RuntimeError(f"cudaGraphicsGLRegisterBuffer err={err}")
        err = cudart.cudaGraphicsMapResources(1, ctypes.byref(resource), ctypes.c_void_p(0))
        if err: raise RuntimeError(f"cudaGraphicsMapResources err={err}")
        mapped = True

        dev_ptr = ctypes.c_void_p(0); sz = ctypes.c_size_t(0)
        err = cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(sz), resource)
        if err: raise RuntimeError(f"cudaGraphicsGetMappedPointer err={err}")

        out = torch.empty(height, width, dtype=torch.int32, device='cuda')
        err = cudart.cudaMemcpy(ctypes.c_void_p(out.data_ptr()), dev_ptr, ctypes.c_size_t(n_bytes), 3)
        if err: raise RuntimeError(f"cudaMemcpy(D2D) err={err}")

        cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), ctypes.c_void_p(0))
        mapped = False
        cudart.cudaGraphicsUnregisterResource(resource); resource = ctypes.c_void_p(0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glDeleteBuffers(1, ctypes.byref(pbo)); pbo.value = 0

        # GL bottom-to-top → top-to-bottom. Skipped when the shader already flips.
        return torch.flip(out, [0]) if flip else out

    except Exception:
        try:
            if mapped:
                cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), ctypes.c_void_p(0))
            if resource.value:
                cudart.cudaGraphicsUnregisterResource(resource)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            if pbo.value:
                gl.glDeleteBuffers(1, ctypes.byref(pbo))
        except Exception:
            pass
        return None


def _build_mvp(K: np.ndarray, R: np.ndarray, t: np.ndarray,
               width: int, height: int,
               near: float = 0.01, far: float = 100_000.0) -> np.ndarray:
    """Build a column-major 4×4 MVP matrix for OpenGL from CV camera params.

    Converts from computer-vision convention (Y-down, Z-forward) to OpenGL
    convention (Y-up, Z-backward) via a flip on Y and Z before projection.
    The returned matrix is already transposed for direct upload to a GLSL mat4
    uniform (which is column-major).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    W, H = float(width), float(height)

    # View: world → CV camera, then flip Y and Z to GL camera convention
    V = np.eye(4, dtype=np.float64)
    V[:3, :3] = R
    V[:3,  3] = t
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    V = flip_yz @ V

    # Projection: CV camera space → OpenGL clip space
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fx / W
    P[0, 2] = 1.0 - 2.0 * cx / W
    P[1, 1] = 2.0 * fy / H
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0

    mvp = (P @ V).astype(np.float32)
    # Transpose: numpy is row-major; GLSL mat4 is column-major.
    # Writing mvp.T means GLSL sees the correct matrix for `mvp * vec4(pos, 1)`.
    return mvp.T
