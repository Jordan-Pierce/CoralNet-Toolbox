from PyQt5.QtCore import QPointF

from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Tools.QtHandleLayer import HandleLayerItem
from coralnet_toolbox.QtActions import AnnotationGeometryEditAction

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.utilities import get_view_scale


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResizeSubTool(SubTool):
    """SubTool for resizing a single annotation using handles.

    Owns the handle layer for whichever annotation currently has one, whether or
    not a drag is in progress: SelectTool shows handles for as long as a single
    annotation is selected, and only enters this sub-tool once a handle is
    actually grabbed.
    """

    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.target_annotation = None
        self.resize_handle_name = None
        self._orig_geom = None
        self.handle_layer = None
        self._current_annotation = None  # Annotation the layer belongs to
        self._muted_annotation = None    # Annotation whose signals we suspended

    # --- Drag lifecycle ---

    def activate(self, event, **kwargs):
        """
        Activates the resizing operation.
        Expects 'annotation' and 'handle_name' in kwargs.
        """
        super().activate(event)
        self.target_annotation = kwargs.get('annotation')
        self.resize_handle_name = kwargs.get('handle_name')

        if not self.target_annotation or not self.resize_handle_name:
            # Invalid activation, immediately deactivate.
            self.parent_tool.deactivate_subtool()
            return

        self._orig_geom = self._capture_geometry(self.target_annotation)

        # A drag emits annotationUpdated on every mouse-move, and the listeners
        # are not cheap: ConfidenceWindow.display_cropped_image clears its
        # scene, destroys and rebuilds five bar rows with a 500 ms animation
        # apiece and recomputes a tooltip; MetaDataWindow.on_annotation_updated
        # rebuilds its whole property grid, recomputing morphology, scaled area
        # and perimeter, and z-volume; AnnotationWindow.on_annotation_updated
        # rebuilds the graphics item a second time. At 60 frames a second the
        # event loop cannot keep up and the window manager shows a busy cursor.
        #
        # Disconnecting known slots one at a time only fixes the listeners you
        # happened to think of, so block the source instead: nothing is
        # interested in the intermediate states of a drag, only in where it
        # ends up. mouseReleaseEvent unblocks and emits exactly once.
        self._suspend_annotation_signals(self.target_annotation)

        layer = self.live_layer()
        if layer is not None:
            layer.set_active_handle(self.resize_handle_name)

    def deactivate(self):
        super().deactivate()
        # Unconditional: an aborted drag must not leave an annotation mute for
        # the rest of the session.
        self._resume_annotation_signals()
        layer = self.live_layer()
        if layer is not None:
            layer.set_active_handle(None)
        self.target_annotation = None
        self.resize_handle_name = None
        self._orig_geom = None
        # Note: the parent SelectTool owns when handles are shown or hidden.

    def mouseMoveEvent(self, event):
        """Perform the resize operation."""
        if not self.is_active or not self.target_annotation:
            return

        # use_status_bar, or the drag simply stops dead: an unverified
        # annotation cannot be edited, and MoveSubTool has always said so while
        # resizing said nothing at all.
        if not self.annotation_window.is_annotation_moveable(self.target_annotation,
                                                             use_status_bar=True):
            self.parent_tool.deactivate_subtool()
            return

        current_pos = self.annotation_window.mapToScene(event.pos())
        self.target_annotation.resize(self.resize_handle_name, current_pos)

        # Move the handles with the geometry. This is a list assignment and a
        # repaint request -- no scene items are created or destroyed, which is
        # what stops the handles from losing their stacking order to the
        # annotation group that resize() just rebuilt.
        self.refresh_handle_positions()

        # Live size readout: the annotation already knows its own bounding box
        # and unit scale, so this is a status-bar string, not a measurement.
        self.parent_tool.show_drag_size(self.target_annotation)

        # Force the scene to update immediately
        self.annotation_window.scene.update()

    def mouseReleaseEvent(self, event):
        """Finalize the resize, update related windows, and deactivate."""
        annotation = self.target_annotation
        if annotation:
            # Normalize the coordinates after resize is complete
            if hasattr(annotation, 'normalize_coordinates'):
                annotation.normalize_coordinates()

            # Re-crop while still blocked; the crop is what every listener
            # will want to read once they are woken up.
            annotation.create_cropped_image(self.annotation_window.rasterio_image)

            # Signals back on, then exactly one update for the whole drag.
            # update_user_confidence emits annotationUpdated and
            # verifiedChanged, which is what refreshes the ConfidenceWindow,
            # the MetaData grid and the Explorer -- so it is deliberately the
            # only thing here that emits. An edited shape is a user assertion
            # about it, so it is verified and its user confidence re-pinned to
            # its label; resize() used to do that on every frame, and once,
            # here, is what it always meant.
            self._resume_annotation_signals()
            try:
                annotation.update_user_confidence(annotation.label)
            except Exception:
                pass

            new_geom = self._capture_geometry(annotation)

            # Push undo action and emit signals
            if self._orig_geom is not None and new_geom is not None:
                try:
                    action = AnnotationGeometryEditAction(self.annotation_window,
                                                          annotation.id,
                                                          self._orig_geom, new_geom)
                    self.annotation_window.action_stack.push(action)
                except Exception:
                    pass

                # Always emit geometry edited signal (critical for viewer updates)
                self.annotation_window.annotationGeometryEdited.emit(
                    annotation.id,
                    {'old_geom': self._orig_geom, 'new_geom': new_geom}
                )

            # Also emit the general modified signal for backwards compatibility
            self.annotation_window.annotationModified.emit(annotation.id)

            self.refresh_handle_positions()

        self.parent_tool.deactivate_subtool()

    # --- Signal suspension ---

    def _suspend_annotation_signals(self, annotation):
        """Silence the annotation for the duration of a drag."""
        self._resume_annotation_signals()
        try:
            annotation.blockSignals(True)
        except Exception:
            return
        self._muted_annotation = annotation

    def _resume_annotation_signals(self):
        """Undo _suspend_annotation_signals. Safe to call when nothing is blocked."""
        annotation = self._muted_annotation
        self._muted_annotation = None
        if annotation is None:
            return
        try:
            annotation.blockSignals(False)
        except Exception:
            pass

    # --- Geometry capture ---

    @staticmethod
    def _capture_geometry(annotation):
        """Snapshot an annotation's geometry for the undo stack."""
        try:
            if hasattr(annotation, 'points'):
                points = [QPointF(p.x(), p.y()) for p in annotation.points]
                holes = []
                if getattr(annotation, 'holes', None):
                    for hole in annotation.holes:
                        holes.append([QPointF(p.x(), p.y()) for p in hole])
                return (points, holes)
            top_left = QPointF(annotation.top_left.x(), annotation.top_left.y())
            bottom_right = QPointF(annotation.bottom_right.x(), annotation.bottom_right.y())
            return (top_left, bottom_right)
        except Exception:
            return None

    # --- Handle Management ---

    def display_resize_handles(self, annotation):
        """Show the handle layer for ``annotation``, creating it if needed."""
        handles = self._get_handles(annotation)
        if not handles:
            self.remove_resize_handles()
            return

        # If we're currently resizing and the handle no longer exists, deactivate
        if self.is_active and self.resize_handle_name not in handles:
            self.parent_tool.deactivate_subtool()
            return

        if self._current_annotation is not annotation or self.live_layer() is None:
            self.remove_resize_handles()

            # Refresh handles automatically whenever the annotation changes
            # under us (a label recolour, an undo, a nudge from another tool).
            if hasattr(annotation, 'annotationUpdated'):
                try:
                    annotation.annotationUpdated.disconnect(self._on_annotation_updated)
                except Exception:
                    pass  # Connection didn't exist
                annotation.annotationUpdated.connect(self._on_annotation_updated)

            self.handle_layer = HandleLayerItem(
                annotation.label.color,
                is_polygon=isinstance(annotation, (PolygonAnnotation, MultiPolygonAnnotation)),
            )
            self.annotation_window.scene.addItem(self.handle_layer)
            self._current_annotation = annotation
        else:
            self.handle_layer.set_color(annotation.label.color)
            self.handle_layer.setPos(0, 0)

        self.sync_view_scale()
        self.handle_layer.set_handles(handles)

    def live_layer(self):
        """The handle layer, or None if there isn't a usable one.

        Loading or deleting an image tears the whole QGraphicsScene down and
        builds a new one, which destroys the C++ half of every item in it and
        leaves this attribute holding a dangling wrapper. Touching that wrapper
        raises RuntimeError from somewhere far away from the cause, so every
        access goes through here and a dead layer is quietly forgotten.
        """
        layer = self.handle_layer
        if layer is None:
            return None
        try:
            scene = layer.scene()
        except RuntimeError:
            self.handle_layer = None
            self._current_annotation = None
            return None
        if scene is not self.annotation_window.scene:
            # Outlived its scene without being destroyed with it.
            self.handle_layer = None
            self._current_annotation = None
            return None
        return layer

    def refresh_handle_positions(self):
        """Re-read handle positions from the annotation without rebuilding."""
        layer = self.live_layer()
        if layer is None or self._current_annotation is None:
            return
        handles = self._get_handles(self._current_annotation)
        if handles:
            layer.set_handles(handles)
        else:
            self.remove_resize_handles()

    def sync_view_scale(self):
        """Tell the layer the view's current zoom so its margins stay correct."""
        layer = self.live_layer()
        if layer is None:
            return
        scale = get_view_scale(self.annotation_window.transform())
        if scale:
            layer.set_scene_scale(scale)

    def handle_at(self, scene_pos):
        """Name of the handle under ``scene_pos``, or None."""
        layer = self.live_layer()
        if layer is None:
            return None
        return layer.handle_at(scene_pos)

    def set_cursor_scene_pos(self, scene_pos):
        """Feed the hover position so nearby handles wake up."""
        layer = self.live_layer()
        if layer is not None:
            layer.set_cursor_scene_pos(scene_pos)

    def set_show_all(self, show_all):
        """Toggle the undecimated, fully-opaque override."""
        layer = self.live_layer()
        if layer is not None:
            layer.set_show_all(show_all)

    def offset_handles(self, dx, dy):
        """Shift the whole layer, for the duration of a move drag."""
        layer = self.live_layer()
        if layer is not None:
            layer.moveBy(dx, dy)

    def clear_handle_offset(self):
        """Drop any move-drag offset. True when there was one to drop."""
        layer = self.live_layer()
        if layer is None or layer.pos().isNull():
            return False
        layer.setPos(0, 0)
        return True

    def handle_offset(self):
        """Current move-drag offset, for callers that mirror it."""
        layer = self.live_layer()
        return layer.pos() if layer is not None else None

    def _on_annotation_updated(self, annotation):
        """Handle annotation updates by refreshing the resize handles."""
        if annotation is not self._current_annotation:
            return
        # During an active resize the drag loop already keeps them in step.
        if not self.is_active:
            self.refresh_handle_positions()

    def remove_resize_handles(self):
        """Remove the handle layer from the scene."""
        annotation = self._current_annotation
        if annotation is not None and hasattr(annotation, 'annotationUpdated'):
            try:
                annotation.annotationUpdated.disconnect(self._on_annotation_updated)
            except Exception:
                pass  # Connection didn't exist
        self._current_annotation = None

        if self.handle_layer is not None:
            try:
                scene = self.handle_layer.scene()
                if scene is not None:
                    scene.removeItem(self.handle_layer)
            except RuntimeError:
                pass  # scene teardown already destroyed it
            self.handle_layer = None

    def _get_handles(self, annotation):
        """Return the handles based on the annotation type."""
        if isinstance(annotation, RectangleAnnotation):
            return self._get_rectangle_handles(annotation)
        if isinstance(annotation, MultiPolygonAnnotation):
            return self._get_multipolygon_handles(annotation)
        if isinstance(annotation, PolygonAnnotation):
            return self._get_polygon_handles(annotation)
        return {}

    def _get_rectangle_handles(self, annotation):
        """Return resize handles for a rectangle annotation."""
        top_left, bottom_right = annotation.top_left, annotation.bottom_right
        return {
            "left": QPointF(top_left.x(), (top_left.y() + bottom_right.y()) / 2),
            "right": QPointF(bottom_right.x(), (top_left.y() + bottom_right.y()) / 2),
            "top": QPointF((top_left.x() + bottom_right.x()) / 2, top_left.y()),
            "bottom": QPointF((top_left.x() + bottom_right.x()) / 2, bottom_right.y()),
            "top_left": QPointF(top_left.x(), top_left.y()),
            "top_right": QPointF(bottom_right.x(), top_left.y()),
            "bottom_left": QPointF(top_left.x(), bottom_right.y()),
            "bottom_right": QPointF(bottom_right.x(), bottom_right.y()),
        }

    def _get_polygon_handles(self, annotation):
        """
        Return resize handles for a polygon, including its outer boundary and all holes.
        Uses the handle format: 'point_{poly_index}_{vertex_index}'.
        """
        handles = {}

        # 1. Create handles for the outer boundary using the 'outer' keyword.
        for i, p in enumerate(annotation.points):
            handle_name = f"point_outer_{i}"
            handles[handle_name] = QPointF(p.x(), p.y())

        # 2. Create handles for each of the inner holes using their index.
        if hasattr(annotation, 'holes'):
            for hole_index, hole in enumerate(annotation.holes):
                for vertex_index, p in enumerate(hole):
                    handle_name = f"point_{hole_index}_{vertex_index}"
                    handles[handle_name] = QPointF(p.x(), p.y())

        return handles

    def _get_multipolygon_handles(self, annotation):
        """
        Return resize handles for every constituent polygon of a multi-polygon.
        Uses the handle format: 'mpoint_{polygon_index}_{ring}_{vertex_index}',
        whose tail is exactly a PolygonAnnotation handle name.

        Handles were previously unavailable for this type entirely -- _get_handles
        returned an empty dict and the annotation had no resize() at all -- so a
        multi-polygon could only be edited by exploding it first.
        """
        handles = {}
        for poly_index, polygon in enumerate(annotation.polygons):
            for vertex_index, p in enumerate(polygon.points):
                handles[f"mpoint_{poly_index}_outer_{vertex_index}"] = QPointF(p.x(), p.y())
            for hole_index, hole in enumerate(getattr(polygon, 'holes', []) or []):
                for vertex_index, p in enumerate(hole):
                    name = f"mpoint_{poly_index}_{hole_index}_{vertex_index}"
                    handles[name] = QPointF(p.x(), p.y())
        return handles
