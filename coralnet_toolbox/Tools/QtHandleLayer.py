import math

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter
from PyQt5.QtWidgets import QGraphicsItem


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Everything the layer draws is specified in device pixels and divided by the
# current scene->device scale at paint time. A handle therefore keeps a constant
# on-screen size at every zoom level without the item ever being rebuilt, which
# is what makes an always-on handle layer affordable.
RESTING_RADIUS_PX = 3.5        # at rest, out of reach of the cursor
RESTING_RADIUS_RECT_PX = 5.0   # rectangles have 8 handles, not 500; show more
POPPED_RADIUS_PX = 8.0         # fully woken
GRAB_RADIUS_PX = 16.0          # inside this, the handle is what a click hits
POP_RADIUS_PX = 60.0           # inside this, the handle starts waking up
# Decimation floor for dense polygon rings. Tied to the drawn size: at a
# spacing below roughly 1.5x the woken diameter the handles visibly merge into
# a bead of solid colour along the outline, which is what the thinning exists
# to prevent.
MIN_SPACING_PX = 12.0

RESTING_OPACITY = 0.55

# Above the phantom layer (10) and a selected annotation's group (20) so a
# handle is never swallowed by the shape it edits, below the cursor-tracking
# markers (100) and crosshair (1000) so those still read on top.
HANDLE_Z_VALUE = 60


# A polygon vertex moves freely, so it gets the omnidirectional cursor; a
# rectangle handle is constrained to its own axis or diagonal and says so.
_RECT_CURSORS = {
    "left": Qt.SizeHorCursor,
    "right": Qt.SizeHorCursor,
    "top": Qt.SizeVerCursor,
    "bottom": Qt.SizeVerCursor,
    "top_left": Qt.SizeFDiagCursor,
    "bottom_right": Qt.SizeFDiagCursor,
    "top_right": Qt.SizeBDiagCursor,
    "bottom_left": Qt.SizeBDiagCursor,
}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class HandleLayerItem(QGraphicsItem):
    """One scene item that draws every resize handle for a single annotation.

    Replaces the previous one-QGraphicsEllipseItem-per-vertex arrangement. That
    was tolerable while handles only existed for as long as Ctrl+Shift was held;
    it is not tolerable when they are always present, because a SAM-derived
    polygon carries a few hundred vertices and the old code tore down and
    rebuilt every one of those items on every mouse-move of a resize drag.

    Collapsing them into a single item buys three things beyond the item count:

    * Zoom is free. Radii are derived from the painter's level-of-detail, so
      changing zoom is a repaint, never a rebuild.
    * Resizing no longer churns the scene. Moving a vertex updates a list and
      calls update(); nothing is added to or removed from the scene, so the
      handles can no longer lose a Z tie to their own annotation mid-drag.
    * Dense rings stay legible. Vertices closer together than MIN_SPACING_PX on
      screen are dropped until the user zooms in far enough to separate them.

    Hover is driven by the tool rather than by Qt hover events: the tool already
    receives every mouse-move, and one distance test against a cached point list
    is cheaper than routing hover enter/leave through hundreds of items.
    """

    def __init__(self, color, is_polygon=False):
        super().__init__()
        self.setZValue(HANDLE_Z_VALUE)
        # The tool feeds us the cursor and performs the hit-test; taking mouse
        # or hover events here would only compete with it.
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)

        self.base_color = QColor(color)
        self.is_polygon = is_polygon

        self._names = []
        self._points = []
        self._bounds = QRectF()
        self._version = 0

        # Scene units per device pixel. Only ever written through
        # set_scene_scale(), which brackets the write with prepareGeometryChange
        # because boundingRect() depends on it.
        self._px = 1.0
        self._scale_sync_pending = False

        self._cursor_pos = None
        self._active_name = None
        self._show_all = False

        self._visible_cache = None
        self._visible_cache_key = None

    # --- Content ---

    def set_color(self, color):
        """Adopt the annotation's current label colour."""
        new_color = QColor(color)
        if new_color != self.base_color:
            self.base_color = new_color
            self.update()

    def set_handles(self, handles):
        """Replace the handle set. ``handles`` maps handle name -> scene QPointF."""
        names = list(handles.keys())
        points = [QPointF(p) for p in handles.values()]

        bounds = QRectF()
        if points:
            xs = [p.x() for p in points]
            ys = [p.y() for p in points]
            bounds = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        self.prepareGeometryChange()
        self._names = names
        self._points = points
        self._bounds = bounds
        self._version += 1
        self._visible_cache = None
        self.update()

    def has_handle(self, name):
        """True when ``name`` is still part of the current handle set."""
        return name in self._names

    # --- View coupling ---

    def set_scene_scale(self, scale):
        """Tell the layer the view's current scene->screen scale.

        boundingRect() is padded by a screen-space margin, so the scale has to
        arrive through prepareGeometryChange rather than being read inside
        paint(). The tool calls this whenever the view changes.
        """
        if scale is None or scale <= 0:
            return
        px = 1.0 / scale
        if abs(px - self._px) <= self._px * 1e-3:
            return
        self.prepareGeometryChange()
        self._px = px
        self._visible_cache = None
        self.update()

    # --- Interaction state ---

    def set_cursor_scene_pos(self, pos):
        """Feed the hover position (scene coords, or None when the pointer left).

        Returns True when this changed what should be on screen.
        """
        old = self._cursor_pos
        if old is None and pos is None:
            return False
        self._cursor_pos = QPointF(pos) if pos is not None else None

        # Only repaint when the cursor is close enough to be changing something.
        reach = POP_RADIUS_PX * self._px
        rect = self._bounds.adjusted(-reach, -reach, reach, reach)
        if (old is not None and rect.contains(old)) or (pos is not None and rect.contains(pos)):
            self.update()
            return True
        return False

    def set_active_handle(self, name):
        """Pin one handle to its woken appearance for the duration of a drag."""
        if name == self._active_name:
            return
        self._active_name = name
        self._visible_cache = None
        self.update()

    def set_show_all(self, show_all):
        """Power override: draw every vertex, undecimated and unfaded."""
        show_all = bool(show_all)
        if show_all == self._show_all:
            return
        self._show_all = show_all
        self._visible_cache = None
        self.update()

    # --- Hit testing ---

    def handle_at(self, scene_pos):
        """Return the name of the handle under ``scene_pos``, or None.

        Only handles that are actually drawn are candidates: a vertex that
        decimation dropped must not be grabbable, or the user would snag an
        invisible point.
        """
        if not self._points:
            return None

        grab = GRAB_RADIUS_PX * self._px
        best_sq = grab * grab
        best = None
        for idx in self._visible_indices():
            pt = self._points[idx]
            dx = pt.x() - scene_pos.x()
            dy = pt.y() - scene_pos.y()
            dist_sq = dx * dx + dy * dy
            if dist_sq <= best_sq:
                best_sq = dist_sq
                best = self._names[idx]
        return best

    def cursor_for(self, name):
        """Return the Qt cursor shape appropriate to a handle, or None."""
        if name is None:
            return None
        return _RECT_CURSORS.get(name, Qt.SizeAllCursor)

    # --- Painting ---

    def boundingRect(self):
        if self._bounds.isNull() and not self._points:
            return QRectF()
        # Doubled so a stale scale (a wheel zoom that repaints before the tool
        # syncs us) cannot clip a handle; an over-large rect only costs repaint
        # area, an under-large one leaves artefacts behind.
        margin = GRAB_RADIUS_PX * self._px * 2.0
        return self._bounds.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter, option, widget=None):
        if not self._points:
            return

        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        if lod <= 0:
            return
        px = 1.0 / lod

        # paint() may not touch geometry, so a drifted scale is corrected on the
        # next event-loop turn instead of here.
        if abs(px - self._px) > self._px * 0.01:
            self._schedule_scale_sync(lod)

        margin = GRAB_RADIUS_PX * px
        exposed = option.exposedRect.adjusted(-margin, -margin, margin, margin)

        painter.setRenderHint(QPainter.Antialiasing, True)

        resting_px = RESTING_RADIUS_PX if self.is_polygon else RESTING_RADIUS_RECT_PX
        wake_span = POP_RADIUS_PX - GRAB_RADIUS_PX
        cursor = self._cursor_pos

        ring_pen = QPen(self.base_color, 2.0)
        ring_pen.setCosmetic(True)
        white_brush = QBrush(Qt.white)
        flat_brush = QBrush(self.base_color)

        for idx in self._visible_indices():
            point = self._points[idx]
            if not exposed.contains(point):
                continue

            if self._names[idx] == self._active_name:
                radius_px, opacity, woken = POPPED_RADIUS_PX, 1.0, True
            elif self._show_all:
                radius_px, opacity, woken = POPPED_RADIUS_PX * 0.8, 1.0, True
            elif cursor is None:
                radius_px, opacity, woken = resting_px, RESTING_OPACITY, False
            else:
                distance_px = math.hypot(point.x() - cursor.x(), point.y() - cursor.y()) * lod
                if distance_px <= GRAB_RADIUS_PX:
                    radius_px, opacity, woken = POPPED_RADIUS_PX, 1.0, True
                elif distance_px < POP_RADIUS_PX:
                    # Squared falloff so the pop reads as local to the cursor
                    # rather than as a broad glow over the whole shape.
                    t = 1.0 - (distance_px - GRAB_RADIUS_PX) / wake_span
                    t *= t
                    radius_px = resting_px + (POPPED_RADIUS_PX - resting_px) * t
                    opacity = RESTING_OPACITY + (1.0 - RESTING_OPACITY) * t
                    woken = t > 0.5
                else:
                    radius_px, opacity, woken = resting_px, RESTING_OPACITY, False

            painter.setOpacity(opacity)
            if woken:
                painter.setPen(ring_pen)
                painter.setBrush(white_brush)
            else:
                painter.setPen(Qt.NoPen)
                painter.setBrush(flat_brush)

            radius = radius_px * px
            painter.drawEllipse(point, radius, radius)

        painter.setOpacity(1.0)

    # --- Internals ---

    def _schedule_scale_sync(self, lod):
        """Re-sync the cached scale outside of paint()."""
        if self._scale_sync_pending:
            return
        self._scale_sync_pending = True

        def apply():
            self._scale_sync_pending = False
            try:
                self.set_scene_scale(lod)
            except RuntimeError:
                pass  # item was destroyed before the timer fired

        QTimer.singleShot(0, apply)

    def _visible_indices(self):
        """Indices of the handles that are drawn (and therefore grabbable).

        Rectangles keep all eight. Polygon rings are thinned so no two drawn
        vertices sit closer than MIN_SPACING_PX apart on screen -- zooming in
        reveals the rest. The first vertex of every ring and the handle
        currently being dragged are always kept.
        """
        key = (self._version, round(self._px, 9), self._show_all, self._active_name)
        if self._visible_cache_key == key and self._visible_cache is not None:
            return self._visible_cache

        count = len(self._points)
        if self._show_all or not self.is_polygon or count <= 12:
            indices = list(range(count))
        else:
            spacing = MIN_SPACING_PX * self._px
            spacing_sq = spacing * spacing
            indices = []
            last_point = None
            last_ring = None
            for i, name in enumerate(self._names):
                point = self._points[i]
                # Handle names are "point_<ring>_<vertex>"; the ring key is
                # everything up to the vertex index.
                ring = name.rsplit("_", 1)[0]
                if ring != last_ring or name == self._active_name:
                    last_ring = ring
                    last_point = point
                    indices.append(i)
                    continue
                dx = point.x() - last_point.x()
                dy = point.y() - last_point.y()
                if dx * dx + dy * dy >= spacing_sq:
                    last_point = point
                    indices.append(i)

        self._visible_cache = indices
        self._visible_cache_key = key
        return indices
