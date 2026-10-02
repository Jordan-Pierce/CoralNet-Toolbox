from PyQt5.QtCore import Qt, QRectF, QPointF, QPoint, QTimer
from PyQt5.QtGui import QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem

from coralnet_toolbox.Annotations.QtAnnotation import FloatingTagItem

from coralnet_toolbox.Tools.QtSubTool import SubTool

from coralnet_toolbox.utilities import convert_measurement
from coralnet_toolbox.utilities import format_measurement
from coralnet_toolbox.utilities import is_length_unit


SELECTION_COLOR = QColor(0, 168, 230)

# How long the cursor has to rest during a box select before the box's size is
# shown, and how far it may then drift, in screen pixels, before the readout is
# taken down again. The tolerance keeps hand tremor from flickering it.
DIMENSION_REST_MS = 1000
DIMENSION_MOVE_TOLERANCE_PX = 8
# Closest the readout may sit to the edge of the view, in screen pixels.
DIMENSION_VIEW_MARGIN_PX = 4


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectSubTool(SubTool):
    """SubTool for selecting multiple annotations with a rectangle."""

    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.selection_rectangle = None
        self.selection_start_pos = None
        # Track marquee selection state while dragging
        self._marquee_current_ids: set = set()
        self._marquee_preexisting_ids: set = set()
        # Size readout shown once the cursor rests mid-drag
        self._dimension_label = None
        self._rest_anchor = None
        self._rest_timer = QTimer()
        self._rest_timer.setSingleShot(True)
        self._rest_timer.setInterval(DIMENSION_REST_MS)
        self._rest_timer.timeout.connect(self._show_dimensions)

    def activate(self, event, **kwargs):
        super().activate(event)
        self.selection_start_pos = self.annotation_window.mapToScene(event.pos())
        
        # Create and style the selection rectangle (dashed blue, light blue fill)
        self.selection_rectangle = QGraphicsRectItem()
        pen = QPen(SELECTION_COLOR, 3, Qt.DashLine)
        pen.setCosmetic(True)
        self.selection_rectangle.setPen(pen)
        fill = QColor(SELECTION_COLOR)
        fill.setAlpha(30)  # Light blue transparent fill
        self.selection_rectangle.setBrush(QBrush(fill))
        self.selection_rectangle.setRect(QRectF(self.selection_start_pos, self.selection_start_pos))
        self.annotation_window.scene.addItem(self.selection_rectangle)
        # Capture the set of annotation ids that were selected before starting the marquee
        try:
            self._marquee_preexisting_ids = set(a.id for a in self.parent_tool.selected_annotations)
        except Exception:
            self._marquee_preexisting_ids = set()

        self._rest_anchor = event.pos()
        self._rest_timer.start()

    def deactivate(self):
        super().deactivate()
        self._rest_timer.stop()
        self._rest_anchor = None
        self._hide_dimensions()
        if self.selection_rectangle:
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None
        self.selection_start_pos = None
        # Cleanup: deselect any annotations that were temporarily selected by the marquee
        if self._marquee_current_ids:
            # Build id -> annotation map
            id_map = {ann.id: ann for ann in self.annotation_window.get_image_annotations()}
            # Determine the set of ids that should remain selected after commit
            try:
                final_selected_ids = set(a.id for a in self.parent_tool.selected_annotations)
            except Exception:
                final_selected_ids = set()

            for ann_id in list(self._marquee_current_ids):
                # If this id was not selected before the marquee and is not in the final selection, deselect it
                if ann_id not in self._marquee_preexisting_ids and ann_id not in final_selected_ids:
                    ann = id_map.get(ann_id)
                    if ann:
                        ann.deselect()

        # Reset marquee tracking
        self._marquee_current_ids.clear()
        self._marquee_preexisting_ids.clear()

    def mouseMoveEvent(self, event):
        """Update the selection rectangle while dragging."""
        if not self.is_active or not self.selection_rectangle:
            return
            
        current_pos = self.annotation_window.mapToScene(event.pos())
        if self.annotation_window.cursorInWindow(event.pos()):
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_rectangle.setRect(rect)
            # --- Live marquee selection: consider annotations inside rect as selected while dragging ---
            locked_label = self.parent_tool.get_locked_label()
            # Build id map and compute new set
            new_ids: set = set()
            id_map = {}
            for annotation in self.annotation_window.get_image_annotations():
                id_map[annotation.id] = annotation
                try:
                    center = annotation.center_xy
                except Exception:
                    center = None
                if center is None:
                    continue
                if rect.contains(center):
                    if locked_label and annotation.label.id != locked_label.id:
                        continue
                    new_ids.add(annotation.id)

            # Compute additions and removals relative to current marquee set
            additions = new_ids - self._marquee_current_ids
            removals = self._marquee_current_ids - new_ids

            # Select newly included annotations (but don't disturb those that were selected before drag)
            for ann_id in additions:
                if ann_id in self._marquee_preexisting_ids:
                    # already selected before marquee; leave it alone
                    continue
                ann = id_map.get(ann_id)
                if ann:
                    ann.select()

            # Deselect annotations that left the marquee (but keep those that were selected before drag)
            for ann_id in removals:
                if ann_id in self._marquee_preexisting_ids:
                    continue
                ann = id_map.get(ann_id)
                if ann:
                    ann.deselect()

            # Store new marquee set
            self._marquee_current_ids = new_ids

        self._track_rest(event.pos())

    # --- Size readout ------------------------------------------------------

    def _track_rest(self, view_pos):
        """Restart the rest countdown once the cursor has really moved.

        Moves inside the tolerance neither hide the readout nor restart the
        countdown, so a hand resting on the mouse still counts as resting; the
        readout just follows the box while it is up.
        """
        if self._rest_anchor is not None:
            drift = (view_pos - self._rest_anchor).manhattanLength()
            if drift <= DIMENSION_MOVE_TOLERANCE_PX:
                if self._dimension_label is not None:
                    self._show_dimensions()
                return

        self._hide_dimensions()
        self._rest_anchor = view_pos
        self._rest_timer.start()

    def _current_raster(self):
        """The raster on display, for its scale."""
        try:
            aw = self.annotation_window
            return aw.main_window.image_window.raster_manager.get_raster(aw.current_image_path)
        except Exception:
            return None

    def _dimension_text(self, rect):
        """Width x height in pixels, then in real-world units when the image has a scale.

        The real-world line uses the unit picked on the annotation toolbar. A
        scale that is not a length (a world file in degrees) is left out rather
        than shown: degrees of longitude and latitude are not the same size, so
        a width and height in them would mislead.
        """
        width_px = rect.width()
        height_px = rect.height()
        lines = [f"{width_px:,.0f} × {height_px:,.0f} px"]

        raster = self._current_raster()
        scale_x = getattr(raster, 'scale_x', None)
        scale_y = getattr(raster, 'scale_y', None)
        units = getattr(raster, 'scale_units', None)
        if scale_x and scale_y and units and is_length_unit(units):
            target = getattr(self.annotation_window, 'current_unit_scale', None) or units
            width, unit, _ = convert_measurement(width_px * scale_x, units, target)
            height, _, _ = convert_measurement(height_px * scale_y, units, target)
            area, area_unit, _ = convert_measurement(
                width_px * scale_x * height_px * scale_y, units, target, squared=True)
            lines.append(f"{format_measurement(width)} × {format_measurement(height)} {unit}"
                         f"   ({format_measurement(area)} {area_unit}²)")

        return "\n".join(lines)

    def _show_dimensions(self):
        """Show or refresh the size readout. Never raises.

        The rest timer calls this, and an exception escaping a timer slot
        aborts the application.
        """
        try:
            self._place_dimensions()
        except RuntimeError:
            # The scene was cleared under the drag (an image switch), taking the
            # readout's items with it.
            self._dimension_label = None
        except Exception:
            self._hide_dimensions()

    def _place_dimensions(self):
        """Pin the box's size under its bottom-left corner, as a rectangle annotation's tag is.

        The same badge as RectangleAnnotation's dimension tag, in the box's
        colour, so it ignores the view's zoom the same way. Unlike that tag it is
        kept inside the view: a box dragged to the bottom edge would otherwise
        hide its own readout.
        """
        if not self.is_active or not self.selection_rectangle:
            return
        rect = self.selection_rectangle.rect()
        if rect.width() < 1 or rect.height() < 1:
            return

        if self._dimension_label is None:
            # Never suppressed for size: it is only up because the user asked by resting.
            self._dimension_label = FloatingTagItem("", SELECTION_COLOR)
            self._dimension_label.setZValue(1000)
            self.annotation_window.scene.addItem(self._dimension_label)

        tag = self._dimension_label
        tag.setText(self._dimension_text(rect))

        view = self.annotation_window
        anchor = view.mapFromScene(QPointF(rect.left(), rect.bottom()))
        size = tag.boundingRect()
        viewport = view.viewport().rect()
        margin = DIMENSION_VIEW_MARGIN_PX
        x = min(max(anchor.x(), viewport.left() + margin),
                viewport.right() - int(size.width()) - margin)
        y = min(max(anchor.y(), viewport.top() + margin),
                viewport.bottom() - int(size.height()) - margin)
        tag.setPos(view.mapToScene(QPoint(x, y)))

    def _hide_dimensions(self):
        """Take the readout down."""
        if self._dimension_label is None:
            return
        try:
            scene = self._dimension_label.scene()
            if scene is not None:
                scene.removeItem(self._dimension_label)
        except RuntimeError:
            pass  # scene teardown already destroyed it
        self._dimension_label = None

    def mouseReleaseEvent(self, event):
        """Finalize the selection and then deactivate."""
        self.finalize_selection()
        self.parent_tool.deactivate_subtool()

    def finalize_selection(self):
        """Select annotations contained within the drawn rectangle."""
        if not self.selection_rectangle:
            return

        rect = self.selection_rectangle.rect()
        locked_label = self.parent_tool.get_locked_label()

        # Gather what needs to be selected
        annotations_to_select = []
        for annotation in self.annotation_window.get_image_annotations():
            if rect.contains(annotation.center_xy):
                if locked_label and annotation.label.id != locked_label.id:
                    continue  
                if annotation not in self.parent_tool.selected_annotations:
                    annotations_to_select.append(annotation)

        # Apply in bulk
        if annotations_to_select:
            self.annotation_window._syncing_selection = True
            
            for ann in annotations_to_select:
                self.annotation_window.select_annotation(ann, multi_select=True, bulk_mode=True)
                
            self.annotation_window._syncing_selection = False
            
            # Fire the UI updates exactly once
            if len(self.annotation_window.selected_annotations) > 1:
                self.annotation_window.main_window.label_window.deselect_active_label()
                self.annotation_window.main_window.confidence_window.clear_display()
                
            self.annotation_window.viewport().update()
            self.annotation_window._emit_selection_changed()