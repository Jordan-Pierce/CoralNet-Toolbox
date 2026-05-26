"""
BrushTool3D — brush-specific 3D tool logic built on top of Tool3D.

The shared preview sphere, hover batching, and label-colored highlight are
owned by Tool3D.  BrushTool3D keeps only the brush-specific stroke plumbing,
including the brush-volume lookup and the optional paint projection path.

The tool is camera-independent: the VTK cell picker works against the rendered
mesh geometry regardless of whether any annotation cameras are loaded.
"""

import warnings

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.MVAT.tools.Tool3D import Tool3D

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BrushTool3D(Tool3D):
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
    tool_kind = 'brush'

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)

        self.painting:   bool  = False
        self._stroke_label = None

        # Accumulates face IDs painted in the current stroke.
        self._stroke_face_ids: set = set()
        self._last_brush_volume_state = None

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
        button = Qt.LeftButton
        try:
            event_button = getattr(event, 'button', None)
            if callable(event_button):
                button = event_button()
        except Exception:
            button = Qt.LeftButton

        if button != Qt.LeftButton:
            return

        if self.preview_only:
            return

        if self.painting:
            self._finish_stroke()
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

        self.painting = True
        self._stroke_label = self._get_selected_label()
        self._stroke_face_ids.clear()
        self._last_brush_volume_state = None
        primary = self._get_primary_mesh()
        if primary is not None:
            self.mvat_manager._ensure_label_painter(primary)
        self._apply_brush(world_pos)

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        super().mouseMoveEvent(event, face_id, world_pos)
        if not self.preview_only and self.painting and world_pos is not None:
            self._apply_brush(world_pos)

    def mouseReleaseEvent(self, event):
        return

    def wheelEvent(self, event, delta_y: int):
        super().wheelEvent(event, delta_y)

    # ------------------------------------------------------------------
    # Core brush logic
    # ------------------------------------------------------------------

    def _apply_brush(self, world_pos: np.ndarray):
        primary = self._get_primary_mesh()
        if primary is None:
            return

        selected_label = self._stroke_label or self._get_selected_label()
        if selected_label is None:
            return

        class_id, color_rgb = self._resolve_label(selected_label)
        if class_id is None:
            return

        try:
            brush_shape = str(getattr(self, 'brush_shape', 'circle')).strip().lower()
        except Exception:
            brush_shape = 'circle'

        try:
            radius = float(getattr(self, 'brush_size', 0.0))
        except Exception:
            radius = 0.0

        if radius <= 0.0:
            return

        try:
            world_pos = np.asarray(world_pos, dtype=np.float64).reshape(-1)
        except Exception:
            return

        if world_pos.size < 3:
            return

        product_id = getattr(primary, 'product_id', None)
        label_id = getattr(selected_label, 'id', None)
        current_center = world_pos[:3].copy()

        if self._should_skip_brush_volume_update(
            product_id=product_id,
            label_id=label_id,
            brush_shape=brush_shape,
            radius=radius,
            center=current_center,
        ):
            return

        face_ids = self._get_face_ids_in_brush_volume(world_pos)
        if face_ids is None or len(face_ids) == 0:
            return

        face_ids_arr = np.asarray(face_ids, dtype=np.int32)
        new_face_ids = [int(face_id) for face_id in face_ids_arr.tolist() if int(face_id) not in self._stroke_face_ids]
        if not new_face_ids:
            self._last_brush_volume_state = (
                product_id,
                label_id,
                brush_shape,
                radius,
                current_center.copy(),
            )
            return

        self._stroke_face_ids.update(new_face_ids)

        submit_3d_face_paint = getattr(self.mvat_manager, 'submit_3d_face_paint', None)
        if callable(submit_3d_face_paint):
            submit_3d_face_paint(
                np.asarray(new_face_ids, dtype=np.int32),
                color_rgb,
                class_id,
                primary_target=primary,
            )

        self._last_brush_volume_state = (
            product_id,
            label_id,
            brush_shape,
            radius,
            current_center.copy(),
        )

    def _get_face_ids_in_brush_volume(self, world_pos: np.ndarray):
        primary = self._get_primary_mesh()
        if primary is None:
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

        try:
            manager = getattr(self, 'mvat_manager', None)
            query_faces = getattr(manager, '_get_faces_within_sphere', None)
            if not callable(query_faces):
                return None
            face_ids = query_faces(primary, world_pos[:3], radius, shape=shape)
        except Exception:
            return None

        if face_ids is None:
            return None
        return np.asarray(face_ids, dtype=np.int32)

    def _finish_stroke(self):
        """
        Finalise the current stroke.  If multi-annotate is on, delegates to
        MVATManager._on_3d_brush_stroke_applied which propagates the painted
        face IDs to all visible camera masks via their index maps.
        """
        self.painting = False
        self._last_brush_volume_state = None

        try:
            if self._stroke_face_ids and self.mvat_manager.multi_annotate_enabled:
                selected_label = self._stroke_label or self._get_selected_label()
                current_kind = str(getattr(self, 'tool_kind', 'brush')).strip().lower()
                handler_name = '_on_3d_erase_stroke_applied' if current_kind == 'erase' else '_on_3d_brush_stroke_applied'
                handler = getattr(self.mvat_manager, handler_name, None)
                if callable(handler):
                    try:
                        handler(self._stroke_face_ids, selected_label)
                    except Exception as e:
                        print(f"⚠️  BrushTool3D: could not propagate stroke: {e}")
        finally:
            self._stroke_face_ids.clear()
            self._stroke_label = None
            self._refresh_hover_overlay_after_stroke()

    def _refresh_hover_overlay_after_stroke(self):
        manager = getattr(self, 'mvat_manager', None)
        center = getattr(self, '_last_hover_world_pos', None)
        if manager is None or center is None:
            return

        try:
            center = np.asarray(center, dtype=np.float64).reshape(-1)
            if center.size < 3 or not np.all(np.isfinite(center[:3])):
                return
            manager.update_sphere_hover_overlay(center[:3], render=True)
        except Exception:
            pass

    def _should_skip_brush_volume_update(self, product_id, label_id, brush_shape, radius, center):
        previous_state = self._last_brush_volume_state
        if previous_state is None:
            return False

        prev_product_id, prev_label_id, prev_shape, prev_radius, prev_center = previous_state
        if prev_product_id != product_id or prev_label_id != label_id or prev_shape != brush_shape:
            return False

        try:
            if not np.isclose(float(prev_radius), float(radius)):
                return False

            prev_center = np.asarray(prev_center, dtype=np.float64).reshape(-1)
            center = np.asarray(center, dtype=np.float64).reshape(-1)
            if prev_center.size < 3 or center.size < 3:
                return False

            center_delta = float(np.linalg.norm(center[:3] - prev_center[:3]))
        except Exception:
            return False

        # Ignore tiny jitter that is very unlikely to change the brush volume.
        return center_delta <= max(1e-6, float(radius) * 0.02)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_primary_mesh(self):
        try:
            from coralnet_toolbox.MVAT.core.Products import MeshProduct
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
