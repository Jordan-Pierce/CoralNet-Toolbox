"""
Dropper3DTool — picks the label of the face under the cursor and sets it as
the active label in the LabelWindow.

Analogous to Tools/QtDropperTool.DropperTool but operates on mesh face class_ids
instead of 2D mask pixels.  A single left-click reads class_ids[face_id] from
the primary MeshProduct, resolves the label via the mask annotation's
class_id_to_label_map, and emits labelSelected so the LabelWindow updates.
"""

import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.MVAT.tools.QtTool3D import Tool3D

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Dropper3DTool(Tool3D):
    """
    Reads the label painted on the clicked mesh face and selects it in the UI.

    Mirrors DropperTool.mousePressEvent which calls
    mask_annotation.get_class_at_point() → annotation_window.labelSelected.emit().
    Here the equivalent is:
        primary.class_ids[face_id] → class_id_to_label_map → labelSelected.emit()
    """

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self):
        super().activate()

    def deactivate(self):
        super().deactivate()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def mousePressEvent(self, event, face_id: int, world_pos):
        if event.button() != Qt.LeftButton:
            return

        if face_id < 0:
            return

        primary = self._get_primary_mesh()
        if primary is None:
            QMessageBox.warning(
                self.mvat_viewer,
                "No Mesh Available",
                "A mesh must be loaded as the primary target to use the dropper tool.",
            )
            return

        class_ids = getattr(primary, 'class_ids', None)
        if class_ids is None or face_id >= len(class_ids):
            return

        class_id = int(class_ids[face_id])

        if class_id == 0:
            QMessageBox.information(
                self.mvat_viewer,
                "Background Selected",
                "The selected face has no label (background).",
            )
            return

        # Look up the label via the current mask annotation's reverse map —
        # same source of truth that BrushTool uses for the forward direction.
        label = self._resolve_class_id_to_label(class_id)
        if label is None:
            QMessageBox.warning(
                self.mvat_viewer,
                "Label Not Found",
                "No label found for the class ID on this face.",
            )
            return

        # Emit via annotation_window so the LabelWindow updates —
        # mirrors DropperTool: self.annotation_window.labelSelected.emit(label.id)
        try:
            self.mvat_manager.annotation_window.labelSelected.emit(label.id)
        except Exception as e:
            print(f"⚠️  Dropper3DTool: could not emit labelSelected: {e}")

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        pass

    def mouseReleaseEvent(self, event):
        pass

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

    def _resolve_class_id_to_label(self, class_id: int):
        """
        Reverse-look up a Label from class_id using the current mask annotation.
        Mirrors DropperTool which uses mask_annotation.class_id_to_label_map.
        """
        try:
            mask_annotation = (
                self.mvat_manager.annotation_window.current_mask_annotation
            )
            if mask_annotation is None:
                return None
            return mask_annotation.class_id_to_label_map.get(class_id)
        except Exception:
            return None
