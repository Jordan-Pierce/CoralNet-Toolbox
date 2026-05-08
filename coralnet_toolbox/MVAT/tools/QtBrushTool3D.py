"""
Brush3DTool — paints labels directly onto mesh faces in world space.

Analogous to Tools/QtBrushTool.BrushTool but operates on 3D geometry:

  Picking:    VTK cell picker (viewport → face_id) inside MVATViewer.eventFilter.
  Brush:      scipy.spatial.cKDTree radius query on cached face centres.
  Painting:   MVATManager._label_painter_thread (already used by propagation code).
  Preview:    A persistent VTK wireframe sphere actor updated in-place each frame
              (no actor creation/destruction on mouse move).
  Projection: On stroke commit, painted face IDs are forwarded to
              MVATManager._on_3d_brush_stroke_applied which propagates them to
              all visible camera masks via their index maps — but only when
              MVATManager.multi_annotate_enabled is True.

The tool is camera-independent: the VTK cell picker works against the rendered
mesh geometry regardless of whether any annotation cameras are loaded.
"""

import warnings

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.MVAT.tools.QtTool3D import Tool3D

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Brush3DTool(Tool3D):
    """
    Paints the active label onto all mesh faces within a world-space brush radius.

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

        self.brush_size: float = 0.1   # world units; calibrated on activate()
        self.painting:   bool  = False

        # KD-tree over face centres — rebuilt only when the primary target changes.
        self._kdtree              = None
        self._kdtree_product_id   = None

        # Persistent VTK preview actor (created once, updated in-place).
        self._sphere_source = None
        self._preview_actor = None

        # Accumulates face IDs painted in the current stroke.
        self._stroke_face_ids: set = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self):
        super().activate()
        self._calibrate_brush_size()
        self._ensure_kdtree()
        self._init_preview_actor()
        try:
            fp = np.array(self.mvat_viewer.plotter.camera.focal_point)
            self._update_preview_sphere(fp)
        except Exception:
            pass

    def deactivate(self):
        if self.painting:
            self._finish_stroke()
        self._remove_preview_sphere()
        super().deactivate()

    def stop_current_drawing(self):
        if self.painting:
            self._finish_stroke()

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
        if world_pos is not None:
            self._update_preview_sphere(world_pos)

        if self.painting and world_pos is not None:
            self._apply_brush(world_pos)

    def mouseReleaseEvent(self, event):
        if self.painting:
            self._finish_stroke()

    def wheelEvent(self, event, delta_y: int):
        if not (event.modifiers() & Qt.ControlModifier):
            return
        notches = delta_y / 120.0
        factor  = 1.15 ** notches
        self.brush_size = max(1e-6, self.brush_size * factor)
        try:
            fp = np.array(self.mvat_viewer.plotter.camera.focal_point)
            self._update_preview_sphere(fp)
        except Exception:
            pass

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

        face_ids = self._kdtree.query_ball_point(world_pos, self.brush_size)
        if not face_ids:
            return

        face_ids_arr = np.array(face_ids, dtype=np.int32)
        self._stroke_face_ids.update(face_ids)

        painter = self.mvat_manager._label_painter_thread
        if painter is not None and painter.isRunning():
            painter.submit(face_ids_arr, color_rgb, class_id)

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
    # Preview sphere — persistent VTK actor updated in-place each frame
    # ------------------------------------------------------------------

    def _preview_color_rgb_float(self):
        """Return (r, g, b) in [0.0, 1.0] from the class-level color name."""
        try:
            import pyvista as pv
            c = pv.Color(self._PREVIEW_COLOR)
            return c.float_rgb   # (r, g, b) each in [0,1]
        except Exception:
            return (1.0, 1.0, 1.0)

    def _init_preview_actor(self):
        """
        Create the wireframe sphere actor once and add it to the renderer.
        Subsequent calls to _update_preview_sphere only mutate the source — zero
        actor allocation overhead on every mouse-move frame.
        """
        # Skip if already initialised.
        if self._preview_actor is not None:
            return
        try:
            try:
                from vtkmodules.vtkFiltersSources import vtkSphereSource
                from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
            except ImportError:
                from vtk import vtkSphereSource, vtkPolyDataMapper, vtkActor

            src = vtkSphereSource()
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            src.SetRadius(self.brush_size)

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetRepresentationToWireframe()
            r, g, b = self._preview_color_rgb_float()
            prop.SetColor(r, g, b)
            prop.SetOpacity(self._PREVIEW_OPACITY)
            actor.VisibilityOff()   # hidden until first mouse move

            self.mvat_viewer.plotter.renderer.AddActor(actor)

            self._sphere_source = src
            self._preview_actor = actor
        except Exception as e:
            print(f"⚠️  Brush3DTool: could not create preview actor: {e}")
            self._sphere_source = None
            self._preview_actor = None

    def _update_preview_sphere(self, center: np.ndarray):
        """
        Move/resize the existing actor in-place.  No Python object creation,
        no actor removal — O(1) per frame.
        """
        if self._sphere_source is None or self._preview_actor is None:
            self._init_preview_actor()
        if self._sphere_source is None:
            return
        try:
            c = center.tolist() if hasattr(center, 'tolist') else list(center)
            self._sphere_source.SetCenter(c[0], c[1], c[2])
            self._sphere_source.SetRadius(self.brush_size)
            self._sphere_source.Modified()
            self._preview_actor.VisibilityOn()
            self.mvat_viewer.plotter.render()
        except Exception:
            pass

    def _remove_preview_sphere(self):
        """Detach the persistent actor from the renderer on deactivation."""
        if self._preview_actor is not None:
            try:
                self.mvat_viewer.plotter.renderer.RemoveActor(self._preview_actor)
                self.mvat_viewer.plotter.render()
            except Exception:
                pass
            self._preview_actor = None
            self._sphere_source = None

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
