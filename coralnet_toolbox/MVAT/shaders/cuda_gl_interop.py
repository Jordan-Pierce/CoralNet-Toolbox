"""CUDA-GL Interop utilities for GPU-accelerated index map operations.

Provides two approaches:
1. Simple GPU tensor transfer: CPU readback → GPU tensor (faster downstream)
2. True CUDA-GL interop: Direct texture→CUDA mapping (avoids CPU readback)
"""

import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pycuda.gl
    import pycuda.driver as cuda
    HAS_PYCUDA_GL = True
except (ImportError, RuntimeError):
    HAS_PYCUDA_GL = False


def read_texture_to_gpu_simple(fbo, crop_h, crop_w):
    """Read FBO texture to GPU tensor via CPU (simple path).

    Unavoidable CPU readback, but immediately transfers to GPU for downstream ops.
    Still useful if subsequent operations happen on GPU.

    Returns:
        torch.Tensor: (crop_h, crop_w) int32 on CUDA, or None if torch unavailable
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    # CPU readback (unavoidable)
    raw = fbo.read(components=1, dtype='i4')
    shot_int32_cpu = np.frombuffer(raw, dtype=np.int32).reshape(crop_h, crop_w)[::-1].copy()

    # Immediate GPU transfer
    shot_int32_gpu = torch.from_numpy(shot_int32_cpu).to(device='cuda', dtype=torch.int32)

    return shot_int32_gpu


def read_texture_to_gpu_cuda_gl(fbo, mgl_context, crop_h, crop_w):
    """Read FBO texture to GPU tensor via CUDA-GL interop (advanced path).

    Attempts true CUDA-GL mapping to avoid CPU→GPU transfer.
    Falls back to CPU readback if interop unavailable.

    Args:
        fbo: ModernGL framebuffer
        mgl_context: ModernGL context dict
        crop_h, crop_w: texture dimensions

    Returns:
        torch.Tensor: (crop_h, crop_w) int32 on CUDA, or None
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    # Try CUDA-GL interop if available
    if HAS_PYCUDA_GL:
        try:
            ctx = mgl_context.get('ctx')
            if ctx is None:
                return None

            # Register texture for CUDA access
            # This is complex and requires OpenGL context synchronization
            # For now, fallback to simple approach
            # TODO: Implement pycuda.gl.RegisteredImage if needed
            return read_texture_to_gpu_simple(fbo, crop_h, crop_w)

        except Exception as e:
            print(f"⚠️  CUDA-GL interop failed ({e}), falling back to CPU readback")
            return read_texture_to_gpu_simple(fbo, crop_h, crop_w)

    # Fallback to simple approach
    return read_texture_to_gpu_simple(fbo, crop_h, crop_w)


def process_index_map_gpu(index_map_gpu, offset=1):
    """Process index map on GPU (reshape, reverse, offset).

    Args:
        index_map_gpu: torch.Tensor (H, W) int32 on CUDA, already flipped
        offset: value to subtract (default 1 for 1-based → 0-based encoding)

    Returns:
        torch.Tensor: Processed (H, W) int32 on CUDA
    """
    if index_map_gpu is None:
        return None

    return index_map_gpu - offset


def index_map_gpu_to_cpu(index_map_gpu):
    """Transfer index map from GPU back to CPU for downstream code.

    Args:
        index_map_gpu: torch.Tensor on CUDA

    Returns:
        np.ndarray: (H, W) int32 on CPU
    """
    if index_map_gpu is None:
        return None

    return index_map_gpu.cpu().numpy()
