"""
Background Label Painter

Pre-allocated-buffer implementation that keeps appends O(M_new) and
in-place updates for re-paints. Emits `pyvista.PolyData` overlays at a
rate-limited cadence so the main thread can swap a tiny actor.
"""
import time
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


class LabelPainterThread(QThread):
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
        self._last_emit_time = 0
        self._min_emit_interval = 0.016
        self._overlay_state_dirty = False

        self._queue: Queue = Queue()
        self._running = True
        self._rebuild_overlay_buffer_from_state()

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

    def _clear_overlay_buffer(self):
        self._n_faces = 0
        self._face_to_buf.fill(-1)
        self._buf_face_ids.fill(-1)
        self._buf_colors.fill(0)
        self._overlay_state_dirty = False

    def _rebuild_overlay_buffer_from_state(self):
        painted_faces = np.flatnonzero(self._class_ids != 0)
        self._clear_overlay_buffer()
        if painted_faces.size == 0:
            return

        n_faces = int(painted_faces.size)
        self._buf_face_ids[:n_faces] = painted_faces.astype(np.int32, copy=False)
        self._buf_colors[:n_faces] = self._labels_view[painted_faces]
        self._face_to_buf[painted_faces] = np.arange(n_faces, dtype=np.int32)
        self._n_faces = n_faces

    def mark_state_dirty(self):
        """Flag that shared label buffers changed outside the painter thread."""
        self._overlay_state_dirty = True

    def _remove_face_from_buffer(self, face_id: int):
        buf_index = int(self._face_to_buf[face_id])
        if buf_index < 0:
            return

        last_index = self._n_faces - 1
        if buf_index != last_index:
            last_face_id = int(self._buf_face_ids[last_index])
            self._buf_face_ids[buf_index] = last_face_id
            self._buf_colors[buf_index] = self._buf_colors[last_index]
            self._face_to_buf[last_face_id] = buf_index

        self._buf_face_ids[last_index] = -1
        self._buf_colors[last_index] = 0
        self._face_to_buf[face_id] = -1
        self._n_faces -= 1

    def _upsert_face_in_buffer(self, face_id: int, color_rgb):
        buf_index = int(self._face_to_buf[face_id])
        if buf_index < 0:
            buf_index = self._n_faces
            self._buf_face_ids[buf_index] = face_id
            self._face_to_buf[face_id] = buf_index
            self._n_faces += 1

        color_arr = np.asarray(color_rgb, dtype=np.uint8).reshape(-1)
        if color_arr.size >= 3:
            self._buf_colors[buf_index, :3] = color_arr[:3]

    def _apply_item_to_overlay_buffer(self, face_ids, color_rgb, class_id: int):
        face_ids_arr = np.asarray(face_ids, dtype=np.int32).ravel()
        if face_ids_arr.size == 0:
            return

        limit = self._face_to_buf.size
        if class_id == 0:
            for face_id in face_ids_arr:
                face_id = int(face_id)
                if 0 <= face_id < limit:
                    self._remove_face_from_buffer(face_id)
            return

        for face_id in face_ids_arr:
            face_id = int(face_id)
            if 0 <= face_id < limit:
                self._upsert_face_in_buffer(face_id, color_rgb)

    @staticmethod
    def build_overlay(mesh_points: np.ndarray, mesh_faces_flat: np.ndarray,
                      face_ids: np.ndarray, color_rgb, attach_colors: bool = True) -> pv.PolyData | None:
        """Build a tiny colored PolyData from a subset of face IDs."""
        try:
            face_ids = np.asarray(face_ids, dtype=np.int32)
            if face_ids.size == 0:
                return None

            selected = mesh_faces_flat[face_ids, 1:]
            unique_vids, inverse = np.unique(selected, return_inverse=True)
            overlay_points = mesh_points[unique_vids]

            remapped = inverse.reshape(selected.shape)
            vtk_faces = np.hstack([
                np.full((len(face_ids), 1), 3, dtype=np.int64),
                remapped.astype(np.int64),
            ]).ravel()

            overlay = pv.PolyData(overlay_points.astype(np.float32), vtk_faces)

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
            print(f"⚠️ LabelPainterThread.build_overlay failed: {e}")
            return None

    def stop(self):
        self._running = False
        self._queue.put(None)  # unblock the get()

    def run(self):
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break
            if item == 'clear':
                self._clear_overlay_buffer()
                continue

            try:
                face_ids, color_rgb, class_id = item
                # -----------------------------------------------------------------
                # 1. Update RAM buffers (O(M), pure numpy, safe off main thread
                #    because the main thread never reads these during painting)
                # -----------------------------------------------------------------
                self._class_ids[face_ids] = class_id
                self._labels_view[face_ids] = color_rgb
                self._apply_item_to_overlay_buffer(face_ids, color_rgb, class_id)
            except Exception as e:
                print(f"⚠️ LabelPainterThread processing error: {e}")
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

    def _snapshot_overlay(self):
        if self._overlay_state_dirty:
            self._rebuild_overlay_buffer_from_state()

        if self._n_faces == 0:
            return None

        return self._build_overlay(
            self._buf_face_ids[:self._n_faces],
            self._buf_colors[:self._n_faces],
        )

    def _build_overlay(self, face_ids: np.ndarray,
                       color_rgb) -> pv.PolyData | None:
        """
        Build a tiny PolyData containing only the painted faces.
        
        Complexity: O(M) where M = len(face_ids). Does NOT traverse the full mesh.
        """
        return self.build_overlay(self._points, self._faces4, face_ids, color_rgb)