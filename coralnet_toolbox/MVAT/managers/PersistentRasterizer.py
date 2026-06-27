"""
PersistentRasterizer — warm GL context service for MVAT visibility.

OpenGL contexts are thread-affine: a moderngl context (and every GL / CUDA-GL
interop call made through it) is only valid on the single thread that created it.
The visibility pipeline previously created and destroyed a fresh standalone
context on each background ``VisibilityWorker`` run, which re-uploaded the entire
mesh/point geometry to the GPU every time (~19M faces ≈ hundreds of MB) and
re-paid context + FBO setup. For incremental camera adds that fixed cost
dominated wall time (~1–2 s/cam) while the actual rasterization was ~20 ms/cam.

This service owns one long-lived "owner" thread that holds the GL context(s) for
the scene's lifetime. All render requests are marshalled to that thread through a
job queue and return CPU numpy arrays, so callers (the background worker thread,
the main thread) never touch the GL context directly. Contexts are cached per
geometry product (LRU-bounded) so switching between a couple of meshes stays warm.

Threading contract
------------------
* The owner thread is the ONLY thread that ever touches a moderngl context, its
  GL objects, or the CUDA-GL interop. Everything it returns is host-side numpy.
* ``render()`` / ``invalidate()`` block the calling thread until the owner thread
  finishes the job (via a ``concurrent.futures.Future``). Call them from worker
  threads, not directly from a latency-sensitive UI handler.
* GL object lifetime is owned here: per-call transient buffers are released inside
  ``compute_batch_visibility_moderngl``; whole contexts are released only on LRU
  eviction or ``shutdown()`` — always on the owner thread. The worker must NOT
  release the context it renders through.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Optional

from coralnet_toolbox.MVAT.utils.MVATLogger import get_visibility_logger

logger = get_visibility_logger()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PersistentRasterizer:
    """Owns warm moderngl contexts on a dedicated thread and serves render jobs.

    Lifecycle is managed by the owner (typically ``MVATManager``): construct once,
    submit renders via ``render()``, and call ``shutdown()`` on teardown. The
    background owner thread is started lazily on the first submitted job, so an
    instance that is never used costs nothing.
    """

    # Number of distinct geometry contexts kept warm at once. Switching between
    # this many meshes/point clouds stays warm; older ones are released (on the
    # owner thread) when a new geometry is rendered. Kept small because each
    # context retains uploaded geometry + CUDA face-centers (hundreds of MB).
    MAX_WARM_CONTEXTS = 2

    _SENTINEL = object()

    def __init__(self, max_warm_contexts: Optional[int] = None):
        self._max_warm = int(max_warm_contexts) if max_warm_contexts else self.MAX_WARM_CONTEXTS
        # key -> mgl_context dict. Only ever mutated on the owner thread.
        self._contexts: "OrderedDict[tuple, dict]" = OrderedDict()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Owner-thread plumbing
    # ------------------------------------------------------------------
    def _ensure_thread(self):
        """Start the owner thread on first use (caller holds ``self._lock``)."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run_loop,
                name='mvat_persistent_rasterizer',
                daemon=True,
            )
            self._thread.start()

    def _run_loop(self):
        """Owner-thread main loop: run queued jobs, then release everything."""
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._release_all_contexts()
                return
            job, future = item
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(job())
                except BaseException as exc:  # noqa: BLE001 — propagate to caller
                    future.set_exception(exc)

    def _submit(self, job):
        """Marshal ``job`` to the owner thread and block for its result."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("PersistentRasterizer has been shut down")
            self._ensure_thread()
        future: Future = Future()
        self._queue.put((job, future))
        return future.result()

    # ------------------------------------------------------------------
    # Context management (all of this runs on the owner thread)
    # ------------------------------------------------------------------
    @staticmethod
    def _key_for(product, splat_radius, splat_round) -> tuple:
        """Cache key for a geometry product's warm context.

        Geometry upload is pixel-budget-agnostic (the budget only scales per
        camera at render time), so the budget is intentionally excluded. Point
        clouds include their splat parameters because those are baked into the
        context's program uniforms at setup.
        """
        from coralnet_toolbox.MVAT.core.Products import (
            MeshProduct,
            PointCloudProduct,
            GaussianSplattingProduct,
        )
        path = getattr(product, 'file_path', None) or id(product)
        if isinstance(product, MeshProduct):
            return ('mesh', path)
        if isinstance(product, GaussianSplattingProduct):
            return ('splat', path)
        if isinstance(product, PointCloudProduct):
            return ('point', path, float(splat_radius), bool(splat_round))
        # Unknown product type — fall back to a per-identity key so it still works.
        return ('other', path)

    def _build_context(self, product, key, pixel_budget, sample_w, sample_h,
                       splat_radius, splat_round) -> dict:
        """Build the moderngl context for ``product`` (owner thread only)."""
        from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
        from coralnet_toolbox.MVAT.core.Products import (
            MeshProduct,
            PointCloudProduct,
            GaussianSplattingProduct,
        )
        kind = key[0]
        if kind == 'mesh' or isinstance(product, MeshProduct):
            return VisibilityManager.setup_batch_mesh_moderngl_context(
                product, pixel_budget, sample_w, sample_h,
            )
        if kind == 'splat' or isinstance(product, GaussianSplattingProduct):
            return VisibilityManager.setup_batch_splat_moderngl_context(
                product, sample_w, sample_h,
            )
        if kind == 'point' or isinstance(product, PointCloudProduct):
            return VisibilityManager.setup_batch_point_moderngl_context(
                product, pixel_budget, sample_w, sample_h,
                splat_radius=splat_radius, splat_round=splat_round,
            )
        raise TypeError(f"Unsupported product type for rasterization: {type(product).__name__}")

    def _ensure_context(self, product, pixel_budget, sample_w, sample_h,
                        splat_radius, splat_round) -> dict:
        """Return the warm context for ``product``, building it if absent.

        Runs on the owner thread. Touching the key marks it most-recently-used;
        building a new context past the cap evicts (and releases) the LRU one.
        """
        key = self._key_for(product, splat_radius, splat_round)
        ctx = self._contexts.get(key)
        if ctx is not None:
            self._contexts.move_to_end(key)
            return ctx

        ctx = self._build_context(
            product, key, pixel_budget, sample_w, sample_h, splat_radius, splat_round,
        )
        self._contexts[key] = ctx
        self._contexts.move_to_end(key)
        # Evict least-recently-used contexts beyond the cap. Safe because
        # background visibility batches are serialized by the caller, so no other
        # in-flight render job references the context being released here.
        while len(self._contexts) > self._max_warm:
            old_key, old_ctx = self._contexts.popitem(last=False)
            logger.debug(f"   ♻️ Releasing warm rasterizer context: {old_key}")
            self._release_context(old_ctx)
        return ctx

    @staticmethod
    def _release_context(ctx: dict):
        """Release a context's GL objects (owner thread only).

        Releasing the moderngl context frees all GL objects created through it;
        the explicit buffer/FBO releases mirror the previous worker cleanup and
        are belt-and-suspenders for drivers that don't cascade.
        """
        if not ctx:
            return
        try:
            for buf in ctx.get('buffers_to_release', []) or []:
                try:
                    buf.release()
                except Exception:
                    pass
            for fbo in (ctx.get('_fbo_cache', {}) or {}).values():
                try:
                    fbo.release()
                except Exception:
                    pass
            gl_ctx = ctx.get('ctx')
            if gl_ctx is not None:
                gl_ctx.release()
        except Exception as exc:
            logger.warning(f"Warm rasterizer context release failed: {exc}")

    def _release_all_contexts(self):
        """Release every cached context (owner thread only)."""
        for ctx in self._contexts.values():
            self._release_context(ctx)
        self._contexts.clear()

    # ------------------------------------------------------------------
    # Public API (callable from any thread)
    # ------------------------------------------------------------------
    def render(self, product, camera_params_list, *,
               compute_depth_map: bool = True,
               compute_visible_indices: bool = True,
               pixel_budget: Optional[int] = None,
               upsample_to_native: bool = False,
               progress_callback=None,
               camera_index_offset: int = 0,
               warp_maps_list=None,
               splat_radius: float = 1.0,
               splat_round: bool = False) -> list:
        """Rasterize ``camera_params_list`` against ``product`` on the warm context.

        Blocks the calling thread until the owner thread finishes and returns the
        list of result dicts (CPU numpy arrays) from
        ``VisibilityManager.compute_batch_visibility_moderngl``. The geometry is
        uploaded once and reused across calls for the same product.
        """
        if not camera_params_list:
            return []
        sample_w = camera_params_list[0][3]
        sample_h = camera_params_list[0][4]

        def job():
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            key = self._key_for(product, splat_radius, splat_round)
            ctx = self._ensure_context(
                product, pixel_budget, sample_w, sample_h, splat_radius, splat_round,
            )
            try:
                return VisibilityManager.compute_batch_visibility_moderngl(
                    product, camera_params_list,
                    compute_depth_map=compute_depth_map,
                    compute_visible_indices=compute_visible_indices,
                    pixel_budget=pixel_budget,
                    upsample_to_native=upsample_to_native,
                    progress_callback=progress_callback,
                    mgl_context=ctx,
                    camera_index_offset=camera_index_offset,
                    warp_maps_list=warp_maps_list,
                )
            except Exception:
                # A failed render may have left transient GL state on the warm
                # context; drop the whole context so the next render rebuilds a
                # clean one rather than reusing a possibly-corrupt / leaking one.
                stale = self._contexts.pop(key, None)
                if stale is not None:
                    self._release_context(stale)
                raise

        return self._submit(job)

    def invalidate(self, file_path: Optional[str] = None):
        """Release cached contexts (all, or only those for ``file_path``).

        Use when a geometry product is reloaded/edited in place. Normal target
        switching does NOT need this — the LRU keeps recently used contexts warm.
        No-op (swallowed) if already shut down.
        """
        def job():
            if file_path is None:
                self._release_all_contexts()
                return None
            for key in [k for k in list(self._contexts) if len(k) > 1 and k[1] == file_path]:
                self._release_context(self._contexts.pop(key))
            return None

        try:
            self._submit(job)
        except RuntimeError:
            pass

    def shutdown(self, timeout: float = 5.0):
        """Release all contexts and stop the owner thread. Idempotent."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            thread = self._thread
        if thread is not None:
            # The sentinel runs after any already-queued jobs, releasing all GL
            # objects on the owner thread before it exits.
            self._queue.put(self._SENTINEL)
            thread.join(timeout)
