"""
DropperTool3D — label-picker tool for 3D scene products.

Mirrors Tools/QtDropperTool.py but operates on the painted elements of the
primary target in the MVATViewer.  A single left-click reads the class ID
painted on the clicked face (mesh) or point (point cloud) and selects the
matching label in the LabelWindow via the annotation window's labelSelected
signal — the 3D analogue of the 2D dropper.
"""

import numpy as np
from PyQt5.QtCore import Qt

from coralnet_toolbox.MVAT.tools.Tool3D import Tool3D


class DropperTool3D(Tool3D):
    """
    Dropper (eyedropper) tool for 3D scene products.

    A single left-click reads the class ID painted on the clicked element and
    emits the matching label via the annotation window's ``labelSelected``
    signal.  Unlike the Fill tool it requires no texture/UV data — any painted
    mesh or point cloud works, as long as its spatial KD-tree has been prewarmed
    (which happens automatically when it becomes the primary target).

    Attributes:
        tool_kind (str): 'dropper' — identifies this as the dropper tool.
    """

    _PREVIEW_COLOR = 'cyan'
    _PREVIEW_OPACITY = 0.5
    tool_kind = 'dropper'

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)
        self.preview_only = False
        # Keep the aiming preview small — the dropper picks a single element.
        self.brush_size = 0.02

    def _use_active_label_preview_color(self) -> bool:
        """Use a neutral picker color; the dropper does not paint the active label."""
        return False

    def activate(self):
        """Activate the dropper with a tiny aiming dot."""
        super().activate()
        cursor = getattr(self.mvat_viewer, '_cursor_preview', None)
        if cursor is not None:
            try:
                cursor.update_transform(
                    center=np.asarray(self.mvat_viewer.plotter.camera.focal_point),
                    radius=self.brush_size,
                    shape='circle',
                    color_rgb_float=self._preview_color_rgb_float(),
                    opacity=self._PREVIEW_OPACITY,
                )
            except Exception:
                pass

    def mousePressEvent(self, event, _face_id: int, world_pos):
        """
        Handle a left-click by reading the painted class at the clicked element
        and selecting the matching label.

        Args:
            event:     The original QMouseEvent.
            face_id:   VTK cell ID under the cursor, or -1 if no mesh face.
            world_pos: np.ndarray (3,) world coordinate of the pick, or None.
        """
        button = Qt.LeftButton
        try:
            event_button = getattr(event, 'button', None)
            if callable(event_button):
                button = event_button()
        except Exception:
            button = Qt.LeftButton

        if button != Qt.LeftButton or world_pos is None:
            return

        if self.preview_only:
            return

        primary_target = self._get_primary_target()
        if primary_target is None:
            return

        class_ids = getattr(primary_target, 'class_ids', None)
        if class_ids is None:
            return

        tree = getattr(primary_target, '_hover_face_kdtree', None)
        if tree is None:
            return

        # Find the closest element centroid to the clicked world position.
        try:
            _, closest_idx = tree.query(world_pos, k=1)
        except Exception:
            return

        closest_idx = int(closest_idx)
        if closest_idx < 0 or closest_idx >= len(class_ids):
            return

        class_id = int(class_ids[closest_idx])

        status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)

        # Background / unpainted element — nothing to pick.
        if class_id == 0:
            if status_bar is not None:
                status_bar.showMessage("Picked background (no label).", 2000)
            return

        # Resolve the class ID back to a project label widget.
        label = self._resolve_label_from_class_id(class_id)
        if label is None:
            if status_bar is not None:
                status_bar.showMessage("No label found for the picked element.", 2000)
            return

        # Select the label in the LabelWindow, mirroring the 2D dropper.
        try:
            self.mvat_manager.annotation_window.labelSelected.emit(label.id)
        except Exception:
            return

        if status_bar is not None:
            label_name = getattr(label, 'short_label_code', None) or label.id
            status_bar.showMessage(f"Selected label: {label_name}", 2000)

    def _get_primary_target(self):
        """Return the current primary scene product, or None."""
        try:
            return self.mvat_viewer.scene_context.get_primary_target()
        except Exception:
            return None

    def _resolve_label_from_class_id(self, class_id: int):
        """Map a 3D class ID back to a project label widget.

        The mesh cid is the canonical class-id (project order, 'Review' excluded),
        so resolve it against that canonical space first. Two channels carry that
        mapping: the propagation engine's class->label-id registry (populated on
        every paint) and the canonical order itself. The mask annotation's
        discovery-order class->label map is a different scheme and is only a
        last-resort fallback.
        """
        engine = getattr(self.mvat_manager, 'propagation_engine', None)
        label_window = getattr(self.mvat_manager.main_window, 'label_window', None)

        # Primary path: engine registry (canonical class-id -> label UUID).
        try:
            label_id = engine._mesh_class_label_ids.get(class_id) if engine is not None else None
            if label_id is not None and label_window is not None:
                for lbl in getattr(label_window, 'labels', []):
                    if lbl.id == label_id:
                        return lbl
        except Exception:
            pass

        # Secondary path: derive the label directly from the canonical order.
        try:
            if engine is not None:
                real = engine._canonical_real_labels()
                if 1 <= int(class_id) <= len(real):
                    return real[int(class_id) - 1]
        except Exception:
            pass

        # Last-resort fallback: the mask annotation's discovery-order registry.
        try:
            mask_annotation = self.mvat_manager.annotation_window.current_mask_annotation
            if mask_annotation is not None:
                label = mask_annotation.class_id_to_label_map.get(class_id)
                if label is not None:
                    return label
        except Exception:
            pass

        return None

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        """Allow the aiming dot to follow the mouse."""
        if not self.active:
            return

        if world_pos is not None:
            self._last_hover_world_pos = np.asarray(world_pos, dtype=np.float64)
            self._update_preview_sphere(world_pos)
        else:
            self._last_hover_world_pos = None
            self._hide_preview_sphere()

    def wheelEvent(self, event, delta_y: int):
        """Dropper doesn't use wheel for resizing."""
        pass
