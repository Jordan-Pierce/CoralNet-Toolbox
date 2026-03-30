"""
Background thread for building overlay geometry from painted face IDs.

Runs entirely in numpy (O(M) per stroke) and emits small numpy payloads
to the main thread for assembly into a `pyvista.PolyData`. This keeps
all VTK/PyVista work on the GUI thread (safer) while the worker only
touches plain CPU memory.
"""
import numpy as np
from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class LabelPainterThread(QThread):
    """
    Consumes (face_ids, color_rgb, class_id) work items from an internal queue.

     For each item:
        1. Updates the python-side `class_ids` and `labels_cache` (O(M)).
        2. Builds small numpy arrays (points, faces, colors) describing only
            the painted faces (O(M)).
        3. Emits `overlay_ready` with the numpy payload. The main thread
            will assemble a `pyvista.PolyData` from those arrays.

    The queue is coalesced on `submit()` so the worker never falls behind.
    """

    overlay_ready = pyqtSignal(object)
    flush_requested = pyqtSignal()

    def __init__(self, mesh_points: np.ndarray, mesh_faces_flat: np.ndarray,
                 labels_view: np.ndarray, class_ids: np.ndarray, parent=None):
        super().__init__(parent)

        # Read-only geometry references (numpy)
        self._points = mesh_points          # (V, 3) float32
        self._faces4 = mesh_faces_flat      # (N_cells, 4) int32: [3, v0, v1, v2]

        # Writable python-owned buffers
        # `labels_view` is expected to be a numpy array owned by Python (uint8, (N,3))
        self._labels_cache = labels_view
        self._class_ids = class_ids

        self._queue: Queue = Queue()
        self._running = True

    def submit(self, face_ids: np.ndarray, color_rgb: tuple, class_id: int):
        """
        Non-blocking call from the main thread on every brush tick.
        Coalesces pending items so the worker always works on the latest state.
        """
        # Drain stale items — only the latest stroke state matters
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        # Copy face_ids to avoid referencing mutable caller memory
        self._queue.put((np.array(face_ids, copy=True), tuple(color_rgb), int(class_id)))

    def stop(self):
        self._running = False
        # Unblock queue.get()
        self._queue.put(None)

    def run(self):
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            face_ids, color_rgb, class_id = item

            if face_ids is None or len(face_ids) == 0:
                continue

            # 1. Update python-side semantic + colour caches (O(M))
            try:
                if self._class_ids is not None:
                    self._class_ids[face_ids] = class_id

                color_np = np.array(color_rgb, dtype=np.uint8)
                # Broadcast assignment (N,3) = (3,)
                self._labels_cache[face_ids] = color_np
            except Exception as e:
                print(f"⚠️ LabelPainterThread: failed to update caches: {e}")
                continue

            # 2. Build a tiny numpy overlay payload containing only the painted faces
            overlay = self._build_overlay(face_ids, color_np)
            if overlay is not None:
                try:
                    pts, faces, colors = overlay
                    # Emit copies to ensure stable ownership across threads
                    self.overlay_ready.emit((pts.copy(), faces.copy(), colors.copy()))
                except Exception as e:
                    print(f"⚠️ LabelPainterThread: failed to emit overlay: {e}")
    def _build_overlay(self, face_ids: np.ndarray, color_rgb: np.ndarray) -> tuple | None:
        """
        Build numpy arrays describing only the painted faces.

        Returns a tuple: (points, vtk_faces_flat, colors) where `vtk_faces_flat`
        uses the VTK face layout ([3,v0,v1,v2,...]). Complexity is O(M).
        """
        try:
            # selected: (M, 3) vertex indices per triangle
            selected = self._faces4[face_ids, 1:]

            # Unique vertex IDs and inverse mapping to remap to local indexing
            unique_vids, inverse = np.unique(selected, return_inverse=True)

            # Local vertex positions
            overlay_points = self._points[unique_vids].astype(np.float32)

            # Remap global vertex ids -> local ids
            remapped = inverse.reshape(selected.shape)

            # VTK face layout: [3, v0, v1, v2, 3, v0, v1, v2, ...]
            vtk_faces = np.hstack([
                np.full((len(face_ids), 1), 3, dtype=np.int32),
                remapped.astype(np.int32)
            ]).ravel().astype(np.int32)

            # Uniform colour per cell for this stroke
            colors = np.tile(np.array(color_rgb, dtype=np.uint8), (len(face_ids), 1)).astype(np.uint8)

            return overlay_points, vtk_faces, colors

        except Exception as e:
            print(f"⚠️ LabelPainterThread._build_overlay failed: {e}")
            return None
