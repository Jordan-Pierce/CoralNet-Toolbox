"""
Background thread for building overlay PolyData from painted face IDs.

Runs entirely in numpy (O(M) per stroke), signals the main thread
with a ready-to-render tiny PolyData. Never touches VTK/OpenGL directly.
"""
import time
from time import perf_counter
from queue import Queue, Empty

import numpy as np
import pyvista as pv
from PyQt5.QtCore import QThread, pyqtSignal

from coralnet_toolbox.MVAT.utils.MVATLogger import get_visibility_logger


class LabelWorker(QThread):
    """
    Consumes (cmd, face_ids, color_rgb, class_id) work items from a queue.
    
    For each item:
      1. Applies the color to the shared numpy labels buffer instantly.
      2. Accumulates the faces actively painted during this stroke.
      3. Builds a tiny PolyData from scratch for only the current stroke.
      4. Emits overlay_ready so the main thread can swap the actor cheaply.
    """
    
    # Emits the tiny PolyData to swap into the plotter
    overlay_ready = pyqtSignal(object)
    # Emits the persistent "all painted faces" overlay payload at stroke end.
    committed_overlay_ready = pyqtSignal(object)

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

        # Tracks only the faces painted during the current drag event.
        # Mask + chunk list keeps per-chunk dedupe O(chunk) instead of the
        # O(S log S) union1d re-sort the previous implementation paid per move.
        self._stroke_mask = np.zeros(len(class_ids), dtype=bool)
        self._stroke_chunks: list = []
        self._last_emit_time = 0
        self._min_emit_interval = 0.016
        self._overlay_state_dirty = False

        self._queue: Queue = Queue()
        self._running = True

    def submit(self, face_ids: np.ndarray, color_rgb: tuple, class_id: int):
        """
        Queue paint commands. Do not drain the queue here - every face chunk
        from the brush must be processed to keep the RAM array accurate.
        """
        self._queue.put(('paint', face_ids.copy(), color_rgb, class_id))

    def finish_stroke(self, discard: bool = False):
        """Queue a signal that the user released the mouse.

        Args:
            discard: When True, only clear the stroke state and live overlay —
                     skip building the committed overlay (used when the caller
                     is about to hard-flush labels into the base actor instead).
        """
        self._queue.put(('finish', bool(discard), None, None))

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

            cmd = item[0]

            if cmd == 'paint':
                _, face_ids, color_rgb, class_id = item
                try:
                    # 1. Update RAM buffers in pure C (Instant)
                    self._class_ids[face_ids] = class_id
                    self._labels_view[face_ids] = color_rgb

                    # 2. Accumulate the current stroke geometry (O(chunk) dedupe)
                    new_faces = face_ids[~self._stroke_mask[face_ids]]
                    if new_faces.size:
                        self._stroke_mask[new_faces] = True
                        self._stroke_chunks.append(new_faces)
                    self._overlay_state_dirty = True
                except Exception as e:
                    print(f"⚠️ LabelWorker processing error: {e}")
            elif cmd == 'finish':
                # Stroke is over. Promote the painted state to the persistent
                # committed overlay (unless discarded), then clear the live one.
                discard = bool(item[1])
                self._reset_stroke_state()
                self._overlay_state_dirty = False
                if not discard:
                    self.committed_overlay_ready.emit(self._build_committed_overlay())
                self.overlay_ready.emit(None)
                continue

            # Always emit when queue drains — guarantees final stroke state is rendered
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

    def _reset_stroke_state(self):
        """Clear the stroke mask by touched indices only (O(S), not O(N))."""
        for chunk in self._stroke_chunks:
            self._stroke_mask[chunk] = False
        self._stroke_chunks = []

    def _current_stroke_faces(self) -> np.ndarray:
        if not self._stroke_chunks:
            return np.empty(0, dtype=np.int32)
        if len(self._stroke_chunks) == 1:
            return self._stroke_chunks[0]
        return np.concatenate(self._stroke_chunks)

    def _snapshot_overlay(self):
        if not self._overlay_state_dirty:
            return None
        self._overlay_state_dirty = False

        stroke_faces = self._current_stroke_faces()
        if stroke_faces.size == 0:
            return None

        # Build a tiny geometry containing only the current stroke.
        colors = self._labels_view[stroke_faces]
        return self._build_overlay(
            stroke_faces,
            colors,
        )

    def _build_committed_overlay(self):
        """
        Build the persistent overlay payload covering ALL painted faces.

        Runs on the worker thread (O(P log P) for the vertex weld, where
        P = total painted faces). Returns an (points, flat_faces, colors)
        tuple for the main thread to assemble, or None when nothing is painted.
        """
        try:
            painted = np.flatnonzero(self._class_ids != 0).astype(np.int32)
            if painted.size == 0:
                return None

            selected = self._faces4[painted, 1:]
            # Weld vertices: the committed overlay persists, so trade worker-side
            # sort time for a ~6x smaller upload than the unwelded fast path.
            unique_vids, inverse = np.unique(selected, return_inverse=True)
            overlay_points = self._points[unique_vids].astype(np.float32, copy=False)

            vtk_faces = np.empty((painted.size, 4), dtype=np.int32)
            vtk_faces[:, 0] = 3
            vtk_faces[:, 1:] = inverse.reshape(-1, 3).astype(np.int32, copy=False)

            colors = np.asarray(self._labels_view[painted, :3], dtype=np.uint8)
            return overlay_points, vtk_faces.ravel(), colors
        except Exception as e:
            print(f"⚠️ LabelWorker._build_committed_overlay failed: {e}")
            return None

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