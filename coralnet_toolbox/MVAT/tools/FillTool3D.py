"""
FillTool3D — flood-fill tool for 3D mesh UV segments.

Uses pre-computed UV texture segments (superpixels) to instantly fill entire
connected regions with a single click. Segments are computed via PyVista's
connectivity filter over vertices that share UV coordinates — the UV seams
create natural segment boundaries.
"""

import numpy as np
from PyQt5.QtCore import Qt

from coralnet_toolbox.MVAT.tools.Tool3D import Tool3D


class FillTool3D(Tool3D):
    """
    Fill tool for UV texture segments on a 3D mesh.

    A single left-click floods the entire connected UV segment at the clicked
    position with the currently active label color and class ID. Segments are
    pre-computed by connectivity analysis during mesh load when a texture is
    present.

    Attributes:
        tool_kind (str): 'fill' — identifies this as the fill tool.
    """

    _PREVIEW_COLOR = 'yellow'
    _PREVIEW_OPACITY = 0.5
    tool_kind = 'fill'

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)
        self.preview_only = False

    def activate(self):
        """Activate the fill tool with a tiny aiming sphere."""
        super().activate()
        # Shrink the cursor preview to a dot for aiming
        cursor = getattr(self.mvat_viewer, '_cursor_preview', None)
        if cursor is not None:
            try:
                cursor.update_transform(
                    center=np.asarray(self.mvat_viewer.plotter.camera.focal_point),
                    radius=0.02,
                    shape='circle',
                    color_rgb_float=self._preview_color_rgb_float(),
                    opacity=self._PREVIEW_OPACITY,
                )
            except Exception:
                pass

    def mousePressEvent(self, event, _face_id: int, world_pos):
        """
        Handle a left-click by finding the clicked UV segment and filling it.

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

        primary_target = self.mvat_manager._get_primary_mesh_target()
        if primary_target is None:
            return

        # Check if the mesh has pre-computed texture segments
        texture_segment_ids = getattr(primary_target, 'texture_segment_ids', None)
        if texture_segment_ids is None:
            status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)
            if status_bar is not None:
                status_bar.showMessage(
                    "Fill tool requires a mesh with textures/UV coordinates.", 3000
                )
            return

        tree = getattr(primary_target, '_hover_face_kdtree', None)
        if tree is None:
            return

        # Find the closest face centroid to the clicked world position
        try:
            _, closest_idx = tree.query(world_pos, k=1)
        except Exception:
            return

        clicked_face_id = closest_idx

        # Get all faces in the same UV segment
        segment_id = texture_segment_ids[clicked_face_id]
        faces_in_segment = np.flatnonzero(texture_segment_ids == segment_id)

        if len(faces_in_segment) == 0:
            return

        # Get the active semantic label and resolve it to a mesh class_id/color
        selected_label = self._get_selected_label()
        if selected_label is None:
            status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)
            if status_bar is not None:
                status_bar.showMessage("Select a label first.", 2000)
            return

        class_id, color_rgb = self._resolve_label(selected_label)

        if class_id is None or color_rgb is None:
            return

        # Paint the entire segment instantly
        self.mvat_manager.submit_3d_face_paint(
            faces_in_segment,
            color_rgb,
            class_id,
            primary_target=primary_target
        )

        # Trigger the visual update
        self.mvat_manager.request_lazy_flush()

        # Provide visual feedback
        status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)
        if status_bar is not None:
            status_bar.showMessage(
                f"Filled {len(faces_in_segment):,} faces in UV segment {segment_id}.",
                2000
            )

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
        """Fill tool doesn't use wheel for resizing."""
        pass