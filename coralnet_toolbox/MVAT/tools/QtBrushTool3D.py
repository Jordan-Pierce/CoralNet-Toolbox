"""
Brush3DTool — brush-specific 3D tool logic built on top of Tool3D.

The shared preview sphere, hover batching, and label-colored highlight are
owned by Tool3D.  Brush3DTool keeps only the brush-specific stroke plumbing,
including the KD-tree lookup and the optional paint projection path.

The tool is camera-independent: the VTK cell picker works against the rendered
mesh geometry regardless of whether any annotation cameras are loaded.
"""

import warnings

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.MVAT.tools.QtTool3D import Tool3D

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Brush3DTool(Tool3D):
    """
    Brush-specific 3D tool logic.

    Attributes:
        brush_size (float): World-space radius of the brush sphere.
                            Calibrated to the scene on first activate(); resizable
                            with Ctrl+wheel — mirrors BrushTool.brush_size.
        painting (bool):    True while the left mouse button is held down.
                            Mirrors BrushTool.painting.
    """

    # Default radius as a fraction of the mesh bounding-box diagonal.
    _DEFAULT_RADIUS_FRACTION = 0.015
    # Preview appearance (overridden by Erase3DTool).
    _PREVIEW_COLOR   = 'white'
    _PREVIEW_OPACITY = 0.35

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)

        self.painting:   bool  = False

        # KD-tree over face centres — rebuilt only when the primary target changes.
        self._kdtree              = None
        self._kdtree_product_id   = None

        # Accumulates face IDs painted in the current stroke.
        self._stroke_face_ids: set = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop_current_drawing(self):
        if self.painting:
            self._finish_stroke()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def mousePressEvent(self, event, face_id: int, world_pos):
        if event.button() != Qt.LeftButton:
            return

        if self.preview_only:
            return

        if not self._has_selected_label():
            QMessageBox.warning(
                self.mvat_viewer,
                "No Label Selected",
                "A label must be selected before using the brush tool.",
            )
            return

        if face_id < 0 or world_pos is None:
            return

        self.painting = not self.painting

        if self.painting:
            self._stroke_face_ids.clear()
            primary = self._get_primary_mesh()
            if primary is not None:
                self.mvat_manager._ensure_label_painter(primary)
            self._apply_brush(world_pos)
        else:
            self._finish_stroke()

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        super().mouseMoveEvent(event, face_id, world_pos)
        if not self.preview_only and self.painting and world_pos is not None:
            self._apply_brush(world_pos)

    def mouseReleaseEvent(self, event):
        if self.preview_only:
            return
        if self.painting:
            self._finish_stroke()

    def wheelEvent(self, event, delta_y: int):
        super().wheelEvent(event, delta_y)

    # ------------------------------------------------------------------
    # Core brush logic
    # ------------------------------------------------------------------

    def _apply_brush(self, world_pos: np.ndarray):
        primary = self._get_primary_mesh()
        if primary is None:
            return

        if not self._ensure_kdtree():
            return

        selected_label = self._get_selected_label()
        if selected_label is None:
            return

        class_id, color_rgb = self._resolve_label(selected_label)
        if class_id is None:
            return

        face_ids = self._get_face_ids_in_brush_volume(world_pos)
        if face_ids is None or len(face_ids) == 0:
            return

        face_ids_arr = np.asarray(face_ids, dtype=np.int32)
        self._stroke_face_ids.update(int(face_id) for face_id in face_ids_arr.tolist())

        painter = self.mvat_manager._label_painter_thread
        if painter is not None and painter.isRunning():
            painter.submit(face_ids_arr, color_rgb, class_id)

    def _get_face_ids_in_brush_volume(self, world_pos: np.ndarray):
        primary = self._get_primary_mesh()
        if primary is None or not self._ensure_kdtree():
            return None

        centers = getattr(primary, '_element_centers_np', None)
        if centers is None or len(centers) == 0:
            return None

        try:
            world_pos = np.asarray(world_pos, dtype=np.float64).reshape(-1)
        except Exception:
            return None

        if world_pos.size < 3:
            return None

        shape = str(getattr(self, 'brush_shape', 'circle')).strip().lower()
        radius = float(getattr(self, 'brush_size', 0.0))
        if radius <= 0.0:
            return None

        if shape == 'square':
            candidate_radius = radius * np.sqrt(3.0)
            try:
                candidate_ids = self._kdtree.query_ball_point(world_pos[:3], candidate_radius)
            except Exception:
                candidate_ids = []
            if not candidate_ids:
                return np.empty(0, dtype=np.int32)

            candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
            candidate_centers = np.asarray(centers, dtype=np.float32)[candidate_ids]
            deltas = np.abs(candidate_centers - world_pos[:3].astype(np.float32))
            within = np.max(deltas, axis=1) <= radius
            return candidate_ids[within]

        try:
            face_ids = self._kdtree.query_ball_point(world_pos[:3], radius)
        except Exception:
            return None
        return np.asarray(face_ids, dtype=np.int32)

    def _finish_stroke(self):
        """
        Finalise the current stroke.  If multi-annotate is on, delegates to
        MVATManager._on_3d_brush_stroke_applied which propagates the painted
        face IDs to all visible camera masks via their index maps.
        """
        self.painting = False

        if not self._stroke_face_ids:
            return

        if self.mvat_manager.multi_annotate_enabled:
            selected_label = self._get_selected_label()
            if selected_label is not None:
                try:
                    self.mvat_manager._on_3d_brush_stroke_applied(
                        self._stroke_face_ids, selected_label
                    )
                except Exception as e:
                    print(f"⚠️  Brush3DTool: could not propagate stroke: {e}")

        self._stroke_face_ids.clear()

    # ------------------------------------------------------------------
    # KD-tree management
    # ------------------------------------------------------------------

    def _ensure_kdtree(self) -> bool:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("⚠️  Brush3DTool requires scipy (pip install scipy).")
            return False

        primary = self._get_primary_mesh()
        if primary is None:
            return False

        product_id = primary.product_id
        if self._kdtree is not None and self._kdtree_product_id == product_id:
            return True

        try:
            primary.prepare_geometry()
        except Exception as e:
            print(f"⚠️  Brush3DTool: prepare_geometry() failed: {e}")
            return False

        centers = getattr(primary, '_element_centers_np', None)
        if centers is None or len(centers) == 0:
            return False

        self._kdtree            = cKDTree(centers)
        self._kdtree_product_id = product_id
        print(f"🌳 Brush3DTool: KD-tree built over {len(centers):,} face centres "
              f"for '{product_id}'")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_primary_mesh(self):
        try:
            from coralnet_toolbox.MVAT.core.Model import MeshProduct
            product = self.mvat_viewer.scene_context.get_primary_target()
            if isinstance(product, MeshProduct):
                return product
        except Exception:
            pass
        return None

    def _has_selected_label(self) -> bool:
        return self._get_selected_label() is not None

    def _get_selected_label(self):
        try:
            return self.mvat_manager.annotation_window.selected_label
        except Exception:
            return None

    def _resolve_label(self, label):
        try:
            mask_annotation = (
                self.mvat_manager.annotation_window.current_mask_annotation
            )
            if mask_annotation is None:
                return None, None

            class_id = mask_annotation.label_id_to_class_id_map.get(label.id)
            if class_id is None:
                mask_annotation.sync_label_map([label])
                class_id = mask_annotation.label_id_to_class_id_map.get(label.id)

            if class_id is None:
                return None, None

            from PyQt5.QtGui import QColor
            c = QColor(label.color)
            return class_id, (c.red(), c.green(), c.blue())

        except Exception:
            return None, None

    def _calibrate_brush_size(self):
        try:
            primary = self._get_primary_mesh()
            if primary is None:
                return
            mesh = primary.get_render_mesh()
            if mesh is None:
                return
            b = mesh.bounds
            diag = np.sqrt(
                (b[1] - b[0]) ** 2 +
                (b[3] - b[2]) ** 2 +
                (b[5] - b[4]) ** 2
            )
            self.brush_size = max(1e-6, diag * self._DEFAULT_RADIUS_FRACTION)
        except Exception:
            pass
