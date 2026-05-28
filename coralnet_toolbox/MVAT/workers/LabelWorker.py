"""
Background Label Worker

Pre-allocated-buffer implementation that keeps appends O(M_new) and
in-place updates for re-paints. Emits `pyvista.PolyData` overlays at a
rate-limited cadence so the main thread can swap a tiny actor.
"""
import time
from time import perf_counter
from queue import Queue, Empty

import numpy as np
import pyvista as pv
from PyQt5.QtCore import QThread, pyqtSignal


# ----------------------------------------------------------------------------------------------------------------------
# Label painter with pre-allocated buffers
# ----------------------------------------------------------------------------------------------------------------------


"""
Background thread for building overlay PolyData from painted face IDs.

Runs entirely in numpy (O(M) per stroke), signals the main thread
with a ready-to-render tiny PolyData. Never touches VTK/OpenGL directly.
"""
import numpy as np
import pyvista as pv
from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal

from coralnet_toolbox.MVAT.utils.MVATLogger import get_visibility_logger


class LabelWorker(QThread):
    """
    Consumes (face_ids, color_rgb, class_id) work items from a queue.
    
    For each item:
      1. Applies the color to the shared numpy labels buffer (O(M)).
      2. Builds a tiny PolyData from scratch — NO extract_cells — (O(M)).
      3. Emits overlay_ready so the main thread can swap the actor cheaply.
    
    If work items arrive faster than they're processed, the queue coalesces
    them so the thread always works on the latest state.
    """
    
    # Emits the tiny PolyData to swap into the plotter
    overlay_ready = pyqtSignal(object)

    def __init__(self, mesh_points: np.ndarray, mesh_faces_flat: np.ndarray,
                 labels_view: np.ndarray, class_ids: np.ndarray, parent=None):
        """
        Args:
            mesh_points:     (V, 3) float32 — shared numpy view of mesh vertices.
            mesh_faces_flat: Flat VTK face array (the raw mesh.faces), already in
                             (N_cells, 4) shape where col 0 = 3 (triangle count).
                             Pass mesh.faces.reshape(-1, 4) from the main thread.
            labels_view:     (N_cells, 3) uint8 — shared numpy view of Labels cell data.
            class_ids:       (N_cells,) int32  — shared numpy semantic class array.
        """
        super().__init__(parent)

        self._points = mesh_points          # read-only during painting
        self._faces4 = mesh_faces_flat      # (N, 4): col0=3, cols1-3=vertex IDs
        self._labels_view = labels_view     # written by this thread
        self._class_ids = class_ids         # written by this thread

        self._face_to_buf = np.full(len(class_ids), -1, dtype=np.int32)
        self._buf_face_ids = np.full(len(class_ids), -1, dtype=np.int32)
        self._buf_colors = np.zeros((len(class_ids), 3), dtype=np.uint8)
        self._n_faces = 0
        self._cached_painted_faces = np.empty(0, dtype=np.int32)
        self._last_emit_time = 0
        self._min_emit_interval = 0.016
        self._overlay_state_dirty = False

        self._queue: Queue = Queue()
        self._running = True

    def submit(self, face_ids: np.ndarray, color_rgb: tuple, class_id: int):
        """
        Non-blocking. Called from the main thread on every brush tick.
        Drains any stale pending item first so we never fall behind.
        """
        # Drain stale items — only the latest stroke matters for the overlay
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._queue.put((face_ids.copy(), color_rgb, class_id))

    def mark_state_dirty(self):
        """Flag that shared label buffers changed outside the painter thread."""
        self._overlay_state_dirty = True

    def stop(self):
        self._running = False
        self._queue.put(None)  # unblock the get()

    def run(self):
        start_time = perf_counter()
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            try:
                # -----------------------------------------------------------------
                # 1. Update RAM buffers in pure C (Instant)
                # -----------------------------------------------------------------
                face_ids, color_rgb, class_id = item
                self._class_ids[face_ids] = class_id
                self._labels_view[face_ids] = color_rgb

                # 2. Flag that we need to rebuild the snapshot
                self._overlay_state_dirty = True
            except Exception as e:
                print(f"⚠️ LabelWorker processing error: {e}")
                # Thread stays alive — don't re-raise

            # Always emit when queue drains — guarantees final stroke is never dropped
            if self._queue.empty():
                overlay = self._snapshot_overlay()
                if overlay is not None:
                    self.overlay_ready.emit(overlay)
                self._last_emit_time = time.monotonic()
                continue

            # Rate-limited emit for intermediate strokes only
            now = time.monotonic()
            if now - self._last_emit_time >= self._min_emit_interval:
                self._last_emit_time = now
                overlay = self._snapshot_overlay()
                if overlay is not None:
                    self.overlay_ready.emit(overlay)
        get_visibility_logger().info(f"LabelWorker.run: {perf_counter() - start_time:.4f}s")

    def _snapshot_overlay(self):
        if self._overlay_state_dirty:
            painted_faces = np.flatnonzero(self._class_ids != 0).astype(np.int32, copy=False)
            self._cached_painted_faces = painted_faces
            self._overlay_state_dirty = False

        if self._cached_painted_faces.size == 0:
            return None

        colors = self._labels_view[self._cached_painted_faces]

        return self._build_overlay(
            self._cached_painted_faces,
            colors,
        )

    def _build_overlay(self, face_ids: np.ndarray,
                       color_rgb) -> pv.PolyData | None:
        """
        Build a tiny PolyData containing only the painted faces.
        
        Complexity: O(M) where M = len(face_ids). Does NOT traverse the full mesh.
        """
        return self.build_overlay(self._points, self._faces4, face_ids, color_rgb)

    @staticmethod
    def build_overlay(mesh_points: np.ndarray, mesh_faces_flat: np.ndarray,
                      face_ids: np.ndarray, color_rgb, attach_colors: bool = True) -> pv.PolyData | None:
        """Build a tiny colored PolyData from a subset of face IDs."""
        try:
            face_ids = np.asarray(face_ids, dtype=np.int32)
            if face_ids.size == 0:
                return None

            selected = mesh_faces_flat[face_ids, 1:]

            # FAST PATH: Directly copy vertices without sorting/welding
            overlay_points = mesh_points[selected.ravel()]

            # Build a naive face array where every 3 vertices make a new triangle
            vtk_faces = np.empty((len(face_ids), 4), dtype=np.int32)
            vtk_faces[:, 0] = 3
            vtk_faces[:, 1:] = np.arange(len(face_ids) * 3).reshape(-1, 3)

            overlay = pv.PolyData(overlay_points.astype(np.float32), vtk_faces.ravel())

            if not attach_colors:
                return overlay

            colors = np.asarray(color_rgb, dtype=np.uint8)
            if colors.ndim == 1:
                if colors.size < 3:
                    return None
                colors = np.tile(colors[:3], (len(face_ids), 1))
            elif colors.shape[0] != len(face_ids):
                colors = np.tile(colors[0, :3], (len(face_ids), 1))

            overlay.cell_data['OverlayColors'] = colors[:, :3]
            return overlay

        except Exception as e:
            print(f"⚠️ LabelWorker.build_overlay failed: {e}")
            return None