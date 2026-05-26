"""
Data Item classes for the Explorer.

Contains AnnotationDataItem (the annotation ViewModel) and ScatterPlotItem
(the graphics object that renders it), plus confidence display and gallery
sorting helpers (formerly confidence_sorting.py).
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter, QBrush, QPixmap, QImage
from PyQt5.QtWidgets import QGraphicsItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


POINT_SIZE = 15
POINT_WIDTH = 2
SPRITE_SIZE = 48

ANNOTATION_WIDTH = 4


# ----------------------------------------------------------------------------------------------------------------------
# Confidence helpers (formerly confidence_sorting.py)
# ----------------------------------------------------------------------------------------------------------------------


def confidence_value(annotation) -> float:
    """Return the confidence used for display/sorting.

    Verified annotations use user confidence. Unverified annotations use the
    first machine confidence value.
    """
    confidence_source = annotation.user_confidence if annotation.verified else annotation.machine_confidence

    if confidence_source:
        try:
            return float(next(iter(confidence_source.values())))
        except (TypeError, ValueError, StopIteration):
            return 0.0

    return 0.0


def confidence_bucket_start(confidence) -> int:
    """Map confidence to a 10% bucket start (0, 10, ..., 90)."""
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))
    bucket_start = int(confidence * 10) * 10
    return min(bucket_start, 90)


def confidence_bucket_label(annotation) -> str:
    """Return the display label for a confidence bucket."""
    if annotation.verified:
        return "Verified"

    bucket_start = confidence_bucket_start(confidence_value(annotation))
    if bucket_start >= 90:
        return "90-100%"
    return f"{bucket_start}-{bucket_start + 9}%"


def confidence_bucket_sort_key(annotation):
    """Return a sort key that places numeric buckets before Verified."""
    if annotation.verified:
        return (1, 0, 0.0)

    confidence = max(0.0, min(1.0, confidence_value(annotation)))
    bucket_start = confidence_bucket_start(confidence)
    return (0, bucket_start, confidence)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ScatterPlotItem(QGraphicsItem):
    """Single QGraphicsItem renderer for the embedding scatter plot."""

    def __init__(self, viewer, coords_2d=None, colors=None, selected_mask=None, depth_values=None):
        super().__init__()
        self.viewer = viewer

        self.coords_2d = np.empty((0, 2), dtype=np.float32)
        self.colors = np.empty((0, 4), dtype=np.uint8)
        self.selected_mask = np.empty((0,), dtype=bool)
        self.depth_values = np.empty((0,), dtype=np.float32)
        self.pixmaps = []

        # Cache of already-scaled pixmaps: {(index, target_px): QPixmap}
        # Avoids re-scaling on every paint call; cleared whenever set_arrays() is called.
        self._scaled_pixmap_cache = {}

        self.set_arrays(coords_2d, colors, selected_mask, depth_values, pixmaps=None)

    def set_arrays(self, coords_2d=None, colors=None, selected_mask=None, depth_values=None, pixmaps=None):
        """Replace the backing arrays used for painting."""
        self.prepareGeometryChange()

        if coords_2d is None:
            coords_2d = np.empty((0, 2), dtype=np.float32)
        if colors is None:
            colors = np.empty((0, 4), dtype=np.uint8)
        if selected_mask is None:
            selected_mask = np.zeros((len(coords_2d),), dtype=bool)
        if depth_values is None:
            depth_values = np.zeros((len(coords_2d),), dtype=np.float32)

        self.coords_2d = np.asarray(coords_2d, dtype=np.float32)
        if self.coords_2d.ndim == 1:
            self.coords_2d = self.coords_2d.reshape(-1, 2)

        self.colors = np.asarray(colors, dtype=np.uint8)
        if self.colors.ndim == 1:
            self.colors = self.colors.reshape(-1, 4)

        self.selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        self.depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
        self.pixmaps = list(pixmaps) if pixmaps is not None else []

        # New embedding data — any cached scaled pixmaps are now stale.
        self._scaled_pixmap_cache = {}

        self.update()

    def _current_point_diameter(self):
        if self.viewer is None:
            return float(POINT_SIZE)
        return float(getattr(self.viewer, 'point_size', POINT_SIZE))

    def _current_sprite_extent(self):
        if self.viewer is None:
            return float(SPRITE_SIZE)
        return float(getattr(self.viewer, 'sprite_size', SPRITE_SIZE))

    def _current_sprite_render_extent(self):
        # Sprites are drawn in scene coordinates; Qt's view transform handles
        # the screen-space scaling automatically.  Multiplying by the view scale
        # here inflated the exposed-rect culling check to cover the entire scene,
        # causing every point to be drawn on every paint call.
        return self._current_sprite_extent()

    def _depth_alpha(self, index):
        if self.viewer is None or not getattr(self.viewer, 'is_3d_data', False):
            return 255
        if self.depth_values.size <= index or getattr(self.viewer, 'z_range', 0.0) <= 0:
            return 255

        try:
            z_normalized = (float(self.depth_values[index]) - float(self.viewer.min_z)) / float(self.viewer.z_range)
        except Exception:
            return 255

        z_normalized = max(0.0, min(1.0, z_normalized))
        return int(128 + 127 * z_normalized)

    def _color_at(self, index, alpha_override=None):
        if self.colors.size == 0 or index >= len(self.colors):
            return QColor(0, 0, 0, 255)

        rgba = self.colors[index]
        alpha = int(rgba[3]) if len(rgba) > 3 else 255
        if alpha_override is not None:
            alpha = int(alpha * alpha_override / 255)
        return QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), max(0, min(255, alpha)))

    def boundingRect(self):
        if self.coords_2d.size == 0:
            return QRectF(-1.0, -1.0, 2.0, 2.0)

        coords = self.coords_2d
        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_y = float(np.max(coords[:, 1]))

        if self.viewer and getattr(self.viewer, 'display_mode', 'dots') == 'sprites':
            point_diameter = self._current_sprite_render_extent()
        else:
            point_diameter = self._current_point_diameter()
            if self.viewer and getattr(self.viewer, 'is_3d_data', False):
                point_diameter *= 1.5

        margin = (point_diameter / 2.0) + 12.0
        return QRectF(min_x - margin, min_y - margin, (max_x - min_x) + 2 * margin, (max_y - min_y) + 2 * margin)

    def paint(self, painter, option, widget):
        if self.coords_2d.size == 0:
            return

        painter.setRenderHint(QPainter.Antialiasing, True)

        coords = self.coords_2d
        selected_mask = self.selected_mask if self.selected_mask.size == len(coords) else np.zeros(len(coords), dtype=bool)
        visible_mask = np.isfinite(coords).all(axis=1)

        if self.viewer is not None and getattr(self.viewer, 'isolated_mode', False):
            visible_mask &= selected_mask

        if not np.any(visible_mask):
            return

        display_mode = getattr(self.viewer, 'display_mode', 'dots') if self.viewer else 'dots'

        # LOD: only draw sprites when they'll be large enough on screen to be useful.
        # When zoomed out, fall back to dots so we don't pay the per-pixmap overhead
        # for thousands of postage-stamp thumbnails nobody can see.
        # We compute the screen-space size of one sprite: scene_size * view_scale_factor.
        # If it's below the threshold, treat this paint call as dots mode.
        _LOD_SPRITE_MIN_SCREEN_PX = 20  # sprites smaller than this on screen → draw as dots
        if display_mode == 'sprites' and self.viewer is not None:
            try:
                gv = self.viewer.graphics_view
                view_scale = abs(gv.transform().m11())
                screen_sprite_px = self._current_sprite_extent() * view_scale
                if screen_sprite_px < _LOD_SPRITE_MIN_SCREEN_PX:
                    display_mode = 'dots'  # LOD downgrade for this paint call only
            except Exception:
                pass

        point_diameter = self._current_sprite_render_extent() if display_mode == 'sprites' else self._current_point_diameter()
        radius = point_diameter / 2.0
        is_sprites = display_mode == 'sprites'
        dot_halo_margin = max(4.0, point_diameter * 0.20)
        sprite_outline_margin = max(2.0, self._current_sprite_extent() * 0.08) if is_sprites else 0.0

        # Compute the true visible scene rect by asking the QGraphicsView directly.
        # option.exposedRect equals the full bounding rect whenever update() is
        # called on the item (which happens after every rotation), so it cannot
        # be used for culling.  painter.worldTransform() inside QGraphicsItem::paint()
        # is in item-local coordinates, not scene coordinates, so it also cannot be
        # used reliably.  The correct approach is to map the viewport's device rect
        # through the view's current scene transform via mapToScene().
        try:
            if self.viewer is not None:
                gv = self.viewer.graphics_view
                vp_poly = gv.mapToScene(gv.viewport().rect())
                scene_visible = vp_poly.boundingRect()
            else:
                scene_visible = option.exposedRect
        except Exception:
            scene_visible = option.exposedRect

        exposed = scene_visible.adjusted(-point_diameter, -point_diameter, point_diameter, point_diameter)
        visible_mask &= (
            (coords[:, 0] >= exposed.left()) & (coords[:, 0] <= exposed.right()) &
            (coords[:, 1] >= exposed.top()) & (coords[:, 1] <= exposed.bottom())
        )

        if not np.any(visible_mask):
            return

        normal_indices = np.flatnonzero(visible_mask & ~selected_mask)
        selected_indices = np.flatnonzero(visible_mask & selected_mask)

        if is_sprites:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # --- Pre-extract arrays once to avoid per-index attribute lookups in the loop ---
        colors_arr = self.colors          # (N, 4) uint8
        depth_arr = self.depth_values     # (N,)  float32
        is_3d = self.viewer is not None and getattr(self.viewer, 'is_3d_data', False)
        min_z = float(getattr(self.viewer, 'min_z', 0.0)) if self.viewer else 0.0
        z_range = float(getattr(self.viewer, 'z_range', 0.0)) if self.viewer else 0.0
        pixmaps = self.pixmaps
        n_pixmaps = len(pixmaps)
        cache = self._scaled_pixmap_cache
        target_px = max(1, int(round(point_diameter)))  # sprites are square in scene space

        def _alpha(idx):
            if not is_3d or z_range <= 0 or depth_arr.size <= idx:
                return 255
            z_n = max(0.0, min(1.0, (float(depth_arr[idx]) - min_z) / z_range))
            return int(128 + 127 * z_n)

        def _qcolor(idx, alpha_override=None):
            rgba = colors_arr[idx]
            a = int(rgba[3]) if len(rgba) > 3 else 255
            if alpha_override is not None:
                a = int(a * alpha_override / 255)
            return QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), max(0, min(255, a)))

        # -------------------------------------------------------------------------
        # Fast dot path: rasterize all normal (unselected) dots into a QImage via
        # NumPy, then blit it in one drawImage() call.  This avoids N individual
        # drawEllipse() QPainter calls (each ~0.01ms) and is ~10-20× faster for
        # large N when zoomed out.
        #
        # We only use this path when:
        #   • mode is dots (not sprites)
        #   • there are no selected points that need halo rings (those still need
        #     the QPainter loop so we can draw halos on top)
        #   • the viewport rect is available to size the raster image
        # -------------------------------------------------------------------------
        _used_fast_dot_path = False
        if not is_sprites and len(normal_indices) > 0:
            try:
                gv = self.viewer.graphics_view if self.viewer is not None else None
                if gv is not None:
                    vp = gv.viewport()
                    W, H = vp.width(), vp.height()
                    if W > 0 and H > 0:
                        # scene→viewport (device pixel) transform.
                        # gv.transform() only has scale/rotation — it omits the scroll
                        # translation.  gv.viewportTransform() is the full mapping that
                        # includes the current pan offset, so dots land in the right place.
                        scene_to_dev = gv.viewportTransform()  # QTransform

                        # Collect RGBA colors for visible normal points; apply depth alpha
                        vis_coords = coords[normal_indices]  # (M, 2)
                        vis_colors = colors_arr[normal_indices].copy()  # (M, 4) uint8

                        if is_3d and z_range > 0 and depth_arr.size > 0:
                            vis_depth = depth_arr[normal_indices]
                            z_n = np.clip((vis_depth - min_z) / z_range, 0.0, 1.0).astype(np.float32)
                            alphas = (128 + 127 * z_n).astype(np.uint8)
                            # Scale existing alpha by depth alpha
                            vis_colors[:, 3] = ((vis_colors[:, 3].astype(np.float32) * alphas) / 255).astype(np.uint8)

                        # Map scene coords → device (pixel) coords using the full viewport transform
                        # QTransform: x' = m11*x + m21*y + dx,  y' = m12*x + m22*y + dy
                        m11 = scene_to_dev.m11(); m12 = scene_to_dev.m12()
                        m21 = scene_to_dev.m21(); m22 = scene_to_dev.m22()
                        dx  = scene_to_dev.dx();  dy  = scene_to_dev.dy()
                        px = (m11 * vis_coords[:, 0] + m21 * vis_coords[:, 1] + dx).astype(np.int32)
                        py = (m12 * vis_coords[:, 0] + m22 * vis_coords[:, 1] + dy).astype(np.int32)

                        # Screen-space dot radius: scene radius × view scale (from viewport transform)
                        view_scale = abs(m11) if abs(m11) > 1e-6 else 1.0
                        dot_r = max(1, int(round(radius * view_scale)))

                        # Build RGBA image buffer
                        buf = np.zeros((H, W, 4), dtype=np.uint8)

                        # Fully-vectorized circle rasterization.
                        # Build a (2r+1)² disc mask once, then scatter all point colors
                        # into the buffer using clipped index arrays — no Python loop over points.
                        r = dot_r
                        D = 2 * r + 1
                        # Offset grid for the disc (D, D)
                        og = np.arange(-r, r + 1, dtype=np.int32)
                        oy, ox = np.meshgrid(og, og, indexing='ij')   # (D, D)
                        disc = (ox ** 2 + oy ** 2) <= r * r            # (D, D) bool
                        disc_oy = oy[disc]   # (K,) offsets for pixels inside the disc
                        disc_ox = ox[disc]   # (K,)

                        # For each point i and each disc pixel k:
                        #   row = py[i] + disc_oy[k],  col = px[i] + disc_ox[k]
                        # Shape after broadcasting: (M, K)
                        rows = py[:, np.newaxis] + disc_oy[np.newaxis, :]   # (M, K)
                        cols = px[:, np.newaxis] + disc_ox[np.newaxis, :]   # (M, K)

                        # Clip to viewport and build a validity mask
                        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

                        # Flatten for indexing; repeat colors to match (M*K) shape.
                        # np.repeat expands each point index once per disc pixel (K times),
                        # giving a flat (M*K,) array we can mask with valid.ravel().
                        r_flat = rows.ravel()
                        c_flat = cols.ravel()
                        K = disc_oy.shape[0]
                        point_idx_flat = np.repeat(np.arange(len(px), dtype=np.int32), K)
                        valid_flat = valid.ravel()
                        buf[r_flat[valid_flat], c_flat[valid_flat]] = vis_colors[point_idx_flat[valid_flat]]

                        # QImage from RGBA buffer — Format_RGBA8888
                        img = QImage(buf.data, W, H, W * 4, QImage.Format_RGBA8888)
                        img = img.copy()  # detach from NumPy buffer lifetime

                        # Draw the image in device (pixel) coordinates by temporarily
                        # resetting the painter transform
                        painter.save()
                        painter.resetTransform()
                        painter.drawImage(0, 0, img)
                        painter.restore()
                        _used_fast_dot_path = True
            except Exception as _e:
                pass  # fall through to per-point loop on any failure

        # Shared pen/color objects re-used across iterations to reduce object churn
        _faint_pen = QPen()
        _faint_pen.setCosmetic(True)
        _point_pen = QPen()
        _point_pen.setCosmetic(True)
        _point_pen.setWidthF(max(1.0, point_diameter * 0.08))
        _halo_pen = QPen()
        _halo_pen.setCosmetic(True)
        _halo_pen.setStyle(Qt.DashLine)

        if not _used_fast_dot_path:
            for index in normal_indices:
                alpha = _alpha(index)
                x, y = float(coords[index, 0]), float(coords[index, 1])
                if is_sprites:
                    target_rect = QRectF(x - radius, y - radius, point_diameter, point_diameter)
                    pixmap = pixmaps[index] if index < n_pixmaps else None
                    if pixmap is not None and not pixmap.isNull():
                        ck = (index, target_px)
                        sp = cache.get(ck)
                        if sp is None:
                            sp = pixmap.scaled(target_px, target_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            cache[ck] = sp
                        painter.drawPixmap(
                            int(target_rect.left() + (target_rect.width() - sp.width()) / 2.0),
                            int(target_rect.top() + (target_rect.height() - sp.height()) / 2.0),
                            sp,
                        )
                    faint_color = _qcolor(index, alpha_override=alpha).darker(150)
                    faint_color.setAlpha(90)
                    _faint_pen.setColor(faint_color)
                    _faint_pen.setWidthF(0.85)
                    painter.setPen(_faint_pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(target_rect.adjusted(0.5, 0.5, -0.5, -0.5))
                else:
                    color = _qcolor(index, alpha_override=alpha)
                    _point_pen.setColor(QColor(color).darker(140))
                    painter.setPen(_point_pen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(QRectF(x - radius, y - radius, point_diameter, point_diameter))

        for index in selected_indices:
            alpha = _alpha(index)
            x, y = float(coords[index, 0]), float(coords[index, 1])
            color = _qcolor(index, alpha_override=alpha)
            if is_sprites:
                halo_diameter = point_diameter + (sprite_outline_margin * 2.0)
                _halo_pen.setColor(QColor(color))
                _halo_pen.setWidthF(1.0)
                painter.setPen(_halo_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(
                    x - halo_diameter / 2.0, y - halo_diameter / 2.0,
                    halo_diameter, halo_diameter,
                ).adjusted(0.5, 0.5, -0.5, -0.5))

                target_rect = QRectF(x - radius, y - radius, point_diameter, point_diameter)
                pixmap = pixmaps[index] if index < n_pixmaps else None
                if pixmap is not None and not pixmap.isNull():
                    ck = (index, target_px)
                    sp = cache.get(ck)
                    if sp is None:
                        sp = pixmap.scaled(target_px, target_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        cache[ck] = sp
                    painter.drawPixmap(
                        int(target_rect.left() + (target_rect.width() - sp.width()) / 2.0),
                        int(target_rect.top() + (target_rect.height() - sp.height()) / 2.0),
                        sp,
                    )
                else:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(target_rect)
            else:
                _halo_pen.setColor(QColor(color))
                _halo_pen.setWidthF(max(1.5, point_diameter * 0.15))
                painter.setPen(_halo_pen)
                painter.setBrush(Qt.NoBrush)
                halo_diameter = point_diameter + (dot_halo_margin * 2.0)
                painter.drawEllipse(QRectF(
                    x - halo_diameter / 2.0, y - halo_diameter / 2.0,
                    halo_diameter, halo_diameter,
                ))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QRectF(x - radius, y - radius, point_diameter, point_diameter))



class AnnotationDataItem:
    """
    Holds all annotation state information for consistent display across viewers.
    This acts as the "ViewModel" for a single annotation, serving as the single
    source of truth for its state in the UI.
    """

    def __init__(self, annotation, embedding_x=None, embedding_y=None, embedding_id=None):
        self.annotation = annotation

        self.embedding_x = embedding_x if embedding_x is not None else 0.0
        self.embedding_y = embedding_y if embedding_y is not None else 0.0

        self.embedding_z = 0.0  # This will store the rotated Z-value (depth)

        # Store the original, un-rotated 3D coordinates from the embedding
        self.embedding_x_3d = 0.0
        self.embedding_y_3d = 0.0
        self.embedding_z_3d = 0.0

        self.embedding_id = embedding_id

        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label

        # Calculate and store aspect ratio on initialization
        self.aspect_ratio = self._calculate_aspect_ratio()

    def _calculate_aspect_ratio(self):
        """Calculate and return the annotation's aspect ratio."""
        annotation = self.annotation

        if hasattr(annotation, 'cropped_bbox'):
            min_x, min_y, max_x, max_y = annotation.cropped_bbox
            width = max_x - min_x
            height = max_y - min_y
            if height > 0:
                return width / height

        try:
            top_left = annotation.get_bounding_box_top_left()
            bottom_right = annotation.get_bounding_box_bottom_right()
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                if height > 0:
                    return width / height
        except (AttributeError, TypeError):
            pass

        try:
            pixmap = annotation.get_cropped_image()
            if pixmap and not pixmap.isNull() and pixmap.height() > 0:
                return pixmap.width() / pixmap.height()
        except (AttributeError, TypeError):
            pass

        return 1.0  # Default to square

    @property
    def effective_label(self):
        """Get the current effective label (preview if it exists, otherwise original)."""
        return self._preview_label if self._preview_label else self.annotation.label

    @property
    def effective_color(self):
        """Get the effective color for this annotation based on the effective label."""
        return self.effective_label.color

    @property
    def is_selected(self):
        """Check if this annotation is selected."""
        return self._is_selected

    def set_selected(self, selected):
        """Set the selection state. This is the single point of control."""
        self._is_selected = selected

    def set_preview_label(self, label):
        """Set a preview label for this annotation."""
        self._preview_label = label

    def clear_preview_label(self):
        """Clear the preview label and revert to the original."""
        self._preview_label = None

    def has_preview_changes(self):
        """Check if this annotation has a temporary preview label assigned."""
        return self._preview_label is not None

    def apply_preview_permanently(self):
        """Apply the preview label permanently to the underlying annotation object."""
        if self._preview_label:
            self.annotation.update_label(self._preview_label)
            self.annotation.update_user_confidence(self._preview_label)
            self._original_label = self._preview_label
            self._preview_label = None
            return True
        return False

    def get_display_info(self):
        """Get display information for this annotation."""
        return {
            'id': self.annotation.id,
            'label': self.effective_label.short_label_code,
            'confidence': self.get_effective_confidence(),
            'type': type(self.annotation).__name__,
            'image': os.path.basename(self.annotation.image_path),
            'embedding_id': self.embedding_id,
            'color': self.effective_color
        }

    def get_tooltip_text(self):
        """Generates a rich HTML-formatted tooltip with all relevant information."""
        info = self.get_display_info()

        tooltip_parts = [
            f"<b>ID:</b> {info['id']}",
            f"<b>Image:</b> {info['image']}",
            f"<b>Label:</b> {info['label']}",
            f"<b>Type:</b> {info['type']}"
        ]

        return "<br>".join(tooltip_parts)

    def get_confidence_value(self) -> float:
        """Return the annotation confidence used for display and sorting.

        Verified annotations use user confidence. Unverified annotations use
        machine confidence.
        """
        return confidence_value(self.annotation)

    def get_confidence_bucket_label(self) -> str:
        """Return the gallery bucket label for the annotation confidence."""
        return confidence_bucket_label(self.annotation)

    def get_confidence_bucket_sort_key(self):
        """Return a sort key that puts confidence bins before Verified."""
        return confidence_bucket_sort_key(self.annotation)

    def get_effective_confidence(self):
        """Return the annotation confidence value used by the gallery."""
        return self.get_confidence_value()
