"""
Fill3DTool — flood-fills connected mesh faces with the active label.

Analogous to Tools/QtFillTool.FillTool but operates on mesh face adjacency
rather than pixel connectivity.

Current state: skeleton with correct interface.  The flood-fill itself
(_apply_fill) is stubbed — implementation notes are inline.

Algorithm sketch (for future implementation):
  1. Pick the seed face under the cursor (face_id from VTK cell picker).
  2. Read the seed face's current class_id.
  3. BFS / DFS over face-adjacency graph:
       - pv.PolyData.cell_neighbors(face_id, connections='edges') gives adjacent
         face IDs in PyVista ≥ 0.39.
       - Stop expansion at faces whose class_id != seed_class_id (fill only
         connected same-label / background region, matching 2D FillTool semantics).
  4. Submit the collected face IDs to LabelPainterThread — same path as BrushTool.
  5. Optionally project to cameras when multi_annotate_enabled.
"""

import warnings

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from coralnet_toolbox.MVAT.tools.QtTool3D import Tool3D

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Fill3DTool(Tool3D):
    """
    Fills a connected region of mesh faces with the active label.

    A single left-click seeds the fill at the face under the cursor.  The fill
    expands across face-adjacent neighbours that share the same current class_id
    as the seed face (matching the 2D FillTool which fills pixels of the same
    class in a connected region).

    Attributes:
        (none beyond Tool3D.active — fill is stateless between clicks)
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

        if not self._has_selected_label():
            QMessageBox.warning(
                self.mvat_viewer,
                "No Label Selected",
                "A label must be selected before using the fill tool.",
            )
            return

        if face_id < 0 or world_pos is None:
            return

        self._apply_fill(face_id)

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        # Fill tool has no live cursor preview in 3D (matching 2D FillTool which
        # uses only a small dot cursor).  No-op on move.
        pass

    def mouseReleaseEvent(self, event):
        pass

    # ------------------------------------------------------------------
    # Fill logic  (stubbed — see module docstring for algorithm sketch)
    # ------------------------------------------------------------------

    def _apply_fill(self, seed_face_id: int):
        """
        Flood-fill connected faces from seed_face_id and paint them with the
        active label.

        TODO: implement BFS over face adjacency using
              pv.PolyData.cell_neighbors(face_id, connections='edges').
              For now this is a no-op placeholder so the tool can be registered
              and the interface exercised without errors.
        """
        primary = self._get_primary_mesh()
        if primary is None:
            return

        selected_label = self._get_selected_label()
        if selected_label is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # ----------------------------------------------------------
            # Phase 1: collect connected faces with matching class_id
            # ----------------------------------------------------------
            mesh       = primary.get_render_mesh()
            class_ids  = getattr(primary, 'class_ids', None)
            if mesh is None or class_ids is None:
                return

            seed_class = int(class_ids[seed_face_id])
            filled_ids = self._bfs_fill(mesh, class_ids, seed_face_id, seed_class)

            if not filled_ids:
                return

            # ----------------------------------------------------------
            # Phase 2: resolve class_id / color for the selected label
            # ----------------------------------------------------------
            class_id, color_rgb = self._resolve_label(selected_label)
            if class_id is None:
                return

            # ----------------------------------------------------------
            # Phase 3: paint via label painter thread (same as BrushTool)
            # ----------------------------------------------------------
            self.mvat_manager._ensure_label_painter(primary)
            painter = self.mvat_manager._label_painter_thread
            if painter is not None and painter.isRunning():
                face_ids_arr = np.array(list(filled_ids), dtype=np.int32)
                painter.submit(face_ids_arr, color_rgb, class_id)

            # ----------------------------------------------------------
            # Phase 4: project to cameras if multi-annotate is on
            # ----------------------------------------------------------
            if self.mvat_manager.multi_annotate_enabled:
                self._project_faces(primary, list(filled_ids))

        except Exception as e:
            print(f"⚠️  Fill3DTool._apply_fill error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _bfs_fill(self, mesh, class_ids: np.ndarray,
                  seed_id: int, target_class: int) -> set:
        """
        BFS flood-fill over face adjacency.

        Expands from seed_id to all connected faces whose class_id equals
        target_class.  Uses PyVista's cell_neighbors() when available;
        falls back to a numpy edge-adjacency build for older PyVista versions.

        Returns:
            Set of face IDs that were filled (including the seed).
        """
        visited = set()
        queue   = [seed_id]

        while queue:
            fid = queue.pop()
            if fid in visited:
                continue
            if class_ids[fid] != target_class:
                continue
            visited.add(fid)

            # Prefer PyVista's built-in cell-neighbour query (PyVista ≥ 0.39).
            try:
                neighbours = mesh.cell_neighbors(fid, connections='edges')
                queue.extend(n for n in neighbours if n not in visited)
            except AttributeError:
                # Fallback: build shared-edge adjacency from face connectivity.
                # This is expensive on first call but only runs when PyVista is old.
                if not hasattr(self, '_adj'):
                    self._adj = self._build_adjacency(mesh)
                queue.extend(n for n in self._adj.get(fid, []) if n not in visited)

        return visited

    def _build_adjacency(self, mesh) -> dict:
        """
        Build a face-adjacency dict keyed by face ID.
        Used as fallback when PyVista.cell_neighbors() is unavailable.
        O(F) construction; cached on self._adj after first call.
        """
        adj: dict = {}
        faces_raw = np.asarray(mesh.faces)
        i = 0
        face_id = 0
        edge_to_faces: dict = {}
        while i < len(faces_raw):
            n = int(faces_raw[i])
            verts = tuple(sorted(faces_raw[i + 1: i + 1 + n]))
            # Register each edge of this face
            for j in range(n):
                edge = (verts[j], verts[(j + 1) % n])
                edge_to_faces.setdefault(edge, []).append(face_id)
            i += n + 1
            face_id += 1

        for neighbours in edge_to_faces.values():
            if len(neighbours) == 2:
                a, b = neighbours
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        return adj

    # ------------------------------------------------------------------
    # Post-fill projection (mirrors Brush3DTool._project_painted_faces)
    # ------------------------------------------------------------------

    def _project_faces(self, primary, face_ids: list):
        """Project the mean of filled face centres to all context cameras."""
        cameras        = list(self.mvat_manager.cameras.values())
        context_matrix = self.mvat_manager.context_matrix
        if not cameras or context_matrix is None:
            return

        centers = [
            primary.get_element_coordinate(fid)
            for fid in face_ids
            if primary.get_element_coordinate(fid) is not None
        ]
        if not centers:
            return

        mean_center       = np.mean(centers, axis=0)
        projections       = {}
        accuracies        = {}
        visibility_status = {}

        for cam in cameras:
            try:
                proj = cam.project(mean_center)
                if proj is None or np.any(np.isnan(proj)):
                    continue
                u, v      = float(proj[0]), float(proj[1])
                in_bounds = 0 <= u < cam.width and 0 <= v < cam.height
                projections[cam.image_path]       = (u, v, in_bounds)
                accuracies[cam.image_path]         = True
                visibility_status[cam.image_path]  = False
            except Exception:
                pass

        try:
            context_matrix.update_dynamic_markers(
                projections, accuracies, visibility_status
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers  (mirrors Brush3DTool helpers)
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
