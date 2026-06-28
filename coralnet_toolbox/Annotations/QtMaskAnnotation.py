import warnings

import base64
import rasterio

import numpy as np

from shapely.geometry import Polygon, Point

from rasterio.features import rasterize

from pycocotools import mask

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QApplication
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush, QPolygonF

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MaskGraphicsItem(QGraphicsItem):
    def __init__(self, mask_annotation):
        super().__init__()
        self.mask_annotation = mask_annotation
        # Displaying a mask requires its canvas; build it now if it was lazy.
        try:
            mask_annotation._ensure_canvas()
        except Exception:
            pass
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def boundingRect(self):
        height, width = self.mask_annotation.mask_data.shape
        return QRectF(0, 0, width, height)

    def paint(self, painter, option, widget):
        qimg = self.mask_annotation.qimage
        if not qimg:
            return
        # Apply transparency at render time for instant updates
        transparency = self.mask_annotation.get_current_transparency()
        if transparency < 255:
            painter.setOpacity(transparency / 255.0)
        # Only draw the exposed region (ItemUsesExtendedStyleOption is set in
        # __init__, so exposedRect is the real dirty rect, not the full bounds).
        # 1px outset avoids seams from rect alignment.
        exposed = option.exposedRect.toAlignedRect().adjusted(-1, -1, 1, 1)
        exposed = exposed.intersected(qimg.rect())
        if exposed.isEmpty():
            if transparency < 255:
                painter.setOpacity(1.0)
            return
        # Class-ID masks must not be bilinearly blended: nearest-neighbor is both
        # correct (no mixed colors at label boundaries) and much faster.
        prev_smooth = painter.testRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(exposed, qimg, exposed)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, prev_smooth)
        if transparency < 255:
            painter.setOpacity(1.0)  # Reset opacity for other drawing operations


class MaskAnnotation(Annotation):
    LOCK_BIT = 2**7  # For uint8, this is 128
    is_mask_annotation = True

    def __init__(self,
                 image_path: str,
                 mask_data: np.ndarray,
                 initial_labels: list,
                 transparency: int = 128,
                 rasterio_src=None):
        """
        Initialize a full-image semantic segmentation annotation.
        There should only be one MaskAnnotation per image.
        """
        if not initial_labels:
            raise ValueError("initial_labels cannot be empty.")
        placeholder_label = initial_labels[0]

        # Pass the existing Label instance to the base class to avoid creating UI widgets
        super().__init__(
            label=placeholder_label,
            image_path=image_path,
            transparency=transparency,
        )
        
        self.mask_data = mask_data.astype(np.uint8)
        
        self.class_id_to_label_map = {}
        self.label_id_to_class_id_map = {}
        self.visible_label_ids = set()
        
        self.next_class_id = 1
        
        # NEW: Initialize the color map cache before syncing labels
        self._cached_color_map = None
        
        self.sync_label_map(initial_labels)
        self.visible_label_ids = set(self.label_id_to_class_id_map.keys())

        self.offset = QPointF(0, 0)
        self.rasterio_src = rasterio_src
        
        # Lazy canvas: colored_mask/qimage are allocated on first display via
        # _ensure_canvas(). Masks that exist only as background propagation
        # targets never pay the RGBA cost; _initialize_canvas builds colors
        # from mask_data, which already reflects every silent write.
        self.colored_mask = None
        self.qimage = None
        
        self.set_centroid()
        self.set_cropped_bbox()
        self._stats_cache = None

    # --- COLOR MAP CACHING METHODS ---

    def invalidate_color_map(self):
        """Clears the cached color map so it rebuilds on the next stroke."""
        self._cached_color_map = None

    def _get_color_map(self):
        """Returns the cached color map, building it only if necessary."""
        if self._cached_color_map is None:
            self._cached_color_map = self._build_color_map()
        return self._cached_color_map

    def _get_annotation_rasterization_geometry(self, annotation):
        """Return a shapely geometry for an annotation if it can be rasterized."""
        geometry_getter = getattr(annotation, 'get_rasterization_geometry', None)
        if callable(geometry_getter):
            try:
                geometry = geometry_getter()
                if geometry is not None:
                    return geometry
            except Exception:
                pass

        try:
            qt_polygon = annotation.get_polygon()
            points = [(p.x(), p.y()) for p in qt_polygon]
            if len(points) >= 3:
                return Polygon(points)
        except Exception:
            pass

        return None
    
    def sync_label_map(self, current_labels_in_project: list):
        """Ensures the internal maps are synced with the project's labels."""
        for label in current_labels_in_project:
            if label.id not in self.label_id_to_class_id_map:
                new_id = self.next_class_id
                self.class_id_to_label_map[new_id] = label
                self.label_id_to_class_id_map[label.id] = new_id
                self.next_class_id += 1
                self.visible_label_ids.add(label.id)
                
        # Invalidate cache since label definitions changed
        self.invalidate_color_map()
                
    def _build_color_map(self):
        """Builds a numpy array mapping class IDs to RGBA colors."""
        # Ensure the map is large enough to handle locked class IDs
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        map_size = max(256, max_id + self.LOCK_BIT + 1)
        color_map = np.zeros((map_size, 4), dtype=np.uint8)

        for class_id, label in self.class_id_to_label_map.items():
            color = label.color
            # Always use full alpha in the array - transparency applied at render time
            alpha = 255 if label.id in self.visible_label_ids else 0
            
            # Set the color for both the normal and the locked version of the class ID
            rgba = [color.red(), color.green(), color.blue(), alpha]
            color_map[class_id] = rgba
            if class_id + self.LOCK_BIT < map_size:
                color_map[class_id + self.LOCK_BIT] = rgba

        return color_map
    
    def _initialize_canvas(self):
        """
        Creates the initial color canvas and QImage. This is the one-time expensive
        operation that happens when the mask is first loaded.
        """
        height, width = self.mask_data.shape
        
        # Use the cached color map method so we initialize the cache 
        # on the very first load, preventing lag on the first brush stroke.
        color_map = self._get_color_map()
        
        # Create the full-size 4-channel RGBA numpy array
        self.colored_mask = color_map[self.mask_data]
        
        # Create a QImage that is a VIEW on the numpy array's data buffer.
        # Modifying the numpy array will now automatically update the QImage.
        self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)

    def _ensure_canvas(self):
        """Allocate colored_mask/qimage on first display (no-op when present)."""
        if self.colored_mask is None:
            self._initialize_canvas()

    def _update_full_canvas(self):
        """Regenerates the entire color canvas."""
        if self.colored_mask is None:
            self._initialize_canvas()
            return
        # Use the cached map instead of building a new one
        color_map = self._get_color_map()
        np.copyto(self.colored_mask, color_map[self.mask_data])

    def _update_canvas_slice(self, update_rect):
        """Efficiently updates only a small rectangular slice of the color canvas."""
        if self.colored_mask is None:
            self._initialize_canvas()
            return
        x1, y1, x2, y2 = update_rect
        data_slice = self.mask_data[y1:y2, x1:x2]

        # Use the cached map instead of building a new one
        color_map = self._get_color_map()

        color_slice = color_map[data_slice]
        self.colored_mask[y1:y2, x1:x2] = color_slice

    def set_centroid(self):
        """Set the centroid to the center of the image."""
        height, width = self.mask_data.shape
        self.center_xy = QPointF(width / 2.0, height / 2.0)

    def set_cropped_bbox(self):
        """Set the bounding box to the full dimensions of the image."""
        height, width = self.mask_data.shape
        self.cropped_bbox = (0, 0, width, height)
        self.annotation_size = int(max(width, height))
                
    def contains_point(self, point: QPointF) -> bool:
        """Check if a point is within the mask's classified area."""
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if 0 <= y < height and 0 <= x < width:
            return self.mask_data[y, x] > 0
        return False

    def get_area(self):
        """Return the total number of non-background pixels."""
        return np.count_nonzero(self.mask_data)

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box (always 0,0)."""
        return QPointF(0, 0)

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        height, width = self.mask_data.shape
        return QPointF(width, height)

    def create_graphics_item(self, scene: QGraphicsScene, force_hydrate: bool = False):
        """Create a QGraphicsPixmapItem to display the mask.

        Accepts `force_hydrate` for API compatibility; masks are always heavy widgets
        so the flag is ignored.
        """
        self.graphics_item = MaskGraphicsItem(self)
        scene.addItem(self.graphics_item)

    def refresh_graphics(self):
        """Recreate QImage to bust Qt's OpenGL texture cache and schedule a repaint.

        Call this whenever colored_mask has been updated in-place (e.g. silent brush
        strokes) and Qt needs to re-upload the texture without a full canvas rebuild.
        """
        if self.graphics_item is None:
            return
        try:
            if self.graphics_item.scene() is None:
                return
        except RuntimeError:
            return
        self._ensure_canvas()
        height, width = self.mask_data.shape
        self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)
        self.graphics_item.update()

    def update_graphics_item(self, update_rect=None):
        """Update the colored canvas / qimage when mask data has changed.

        IMPORTANT: the colored_mask recompute and qimage rebuild must run even
        when this annotation has no AnnotationWindow ``graphics_item`` yet.
        Context-matrix cameras that were never opened in the AnnotationWindow
        (e.g. masks pre-allocated for cache-loaded cameras during a
        Mesh -> All Cameras projection) carry a read-only overlay item on their
        matrix canvas but no primary graphics_item.  Returning early here left
        their qimage stale, so the matrix thumbnail never reflected the new
        labels until the camera was activated as the primary view.  Only the
        ``graphics_item.update()`` call is gated on graphics_item existing.
        """
        if self.colored_mask is None:
            # First display request: build the full canvas from mask_data (it
            # already includes every silent write made before now).
            self._initialize_canvas()
            if self.graphics_item is not None:
                self.graphics_item.update()
            return
        height, width = self.mask_data.shape

        if update_rect:
            # Localized update for brush strokes - update only the changed area
            self._update_canvas_slice(update_rect)
            # Recreate QImage so Qt's OpenGL texture cache sees a new cacheKey
            self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)
            if self.graphics_item is not None:
                qt_rect = QRectF(update_rect[0],
                                 update_rect[1],
                                 update_rect[2] - update_rect[0],
                                 update_rect[3] - update_rect[1])
                self.graphics_item.update(qt_rect)
        else:
            # Full update for global changes (e.g., label color changes)
            self._update_full_canvas()
            # Recreate QImage so Qt's OpenGL texture cache sees a new cacheKey
            self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)
            if self.graphics_item is not None:
                self.graphics_item.update()

    def apply_flat_values_at_indices(self, flat_indices, class_values, silent: bool = False, update_rect=None):
        """Apply raw mask values at exact flat indices without lock filtering.

        This is the low-level replay path used by undo/redo actions. The caller
        is responsible for any lock validation or history capture.

        Returns:
            dict | None: Change metadata containing the applied indices and values,
            or None if no pixels changed.
        """
        if flat_indices is None:
            return None

        flat_indices = np.asarray(flat_indices, dtype=np.int64).ravel()
        if flat_indices.size == 0:
            return None

        height, width = self.mask_data.shape
        flat_view = self.mask_data.ravel()
        valid_mask = (flat_indices >= 0) & (flat_indices < flat_view.size)
        if not np.any(valid_mask):
            return None

        target_indices = flat_indices[valid_mask]
        if target_indices.size == 0:
            return None

        new_values = np.asarray(class_values, dtype=flat_view.dtype).ravel()
        if new_values.size == 1:
            new_values = np.full(target_indices.size, new_values.item(), dtype=flat_view.dtype)
        elif new_values.size != target_indices.size:
            raise ValueError("Raw mask edit values must match the number of target pixels.")

        before_values = flat_view[target_indices].copy()
        changed_mask = before_values != new_values
        if not np.any(changed_mask):
            return None

        target_indices = target_indices[changed_mask]
        before_values = before_values[changed_mask]
        new_values = new_values[changed_mask]

        flat_view[target_indices] = new_values

        if self.colored_mask is not None:
            color_map = self._get_color_map()
            colored_flat = self.colored_mask.reshape(-1, 4)
            colored_flat[target_indices] = color_map[new_values]

        if update_rect is None:
            y_coords, x_coords = np.divmod(target_indices, width)
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            update_rect = (
                max(0, x_min - 1),
                max(0, y_min - 1),
                min(width, x_max + 2),
                min(height, y_max + 2),
            )

        if self.graphics_item is not None and not silent:
            if target_indices.size < 250000:
                qt_rect = QRectF(
                    update_rect[0],
                    update_rect[1],
                    update_rect[2] - update_rect[0],
                    update_rect[3] - update_rect[1],
                )
                self.graphics_item.update(qt_rect)
            else:
                self.graphics_item.update()

        self._invalidate_stats_cache()
        if not silent:
            self.annotationUpdated.emit(self)

        return {
            "flat_indices": target_indices,
            "before_values": before_values,
            "after_values": new_values,
            "update_rect": update_rect,
        }

    def update_mask(self, brush_location: QPointF, brush_mask: np.ndarray, new_class_id: int, silent: bool = False, use_new_method: bool = True, history_action=None):
        """
        Modify the mask data based on a brush stroke, respecting pre-locked pixels.
        Includes A/B testing for Diff-Filtering to skip redundant GPU/Qt updates.
        """

        x_start, y_start = int(brush_location.x()), int(brush_location.y())
        brush_h, brush_w = brush_mask.shape
        mask_h, mask_w = self.mask_data.shape

        # Define the update area and clip it to the mask's bounds
        x_end = min(x_start + brush_w, mask_w)
        y_end = min(y_start + brush_h, mask_h)
        clipped_x_start = max(x_start, 0)
        clipped_y_start = max(y_start, 0)

        if clipped_x_start >= x_end or clipped_y_start >= y_end:
            return
            
        # Get the slice of the main mask data we will be updating
        target_slice = self.mask_data[clipped_y_start:y_end, clipped_x_start:x_end]
        
        # Clip the user's brush mask to match the on-screen portion
        brush_x_offset = clipped_x_start - x_start
        brush_y_offset = clipped_y_start - y_start
        clipped_brush_mask = brush_mask[brush_y_offset:brush_y_offset + target_slice.shape[0],
                                        brush_x_offset:brush_x_offset + target_slice.shape[1]]
        
        unlocked_pixels_mask = target_slice < self.LOCK_BIT

        if use_new_method:
            # --- NEW: DIFF FILTER & DIRECT CANVAS INJECTION ---
            pixels_to_change = clipped_brush_mask & unlocked_pixels_mask & (target_slice != new_class_id)
            pixels_updated = np.count_nonzero(pixels_to_change)

            # FAST EXIT: If nothing actually changed, don't trigger a Qt redraw!
            if pixels_updated == 0:
                return

            local_ys, local_xs = np.where(pixels_to_change)
            flat_indices = ((clipped_y_start + local_ys) * mask_w + (clipped_x_start + local_xs)).astype(np.int64)
            before_values = target_slice[pixels_to_change].copy()

            # 1. Apply to semantic data
            target_slice[pixels_to_change] = new_class_id
            
            # 2. Update visual canvas directly (bypassing _update_canvas_slice memory allocation)
            if self.colored_mask is not None:
                color_map = self._get_color_map()
                target_colored_slice = self.colored_mask[clipped_y_start:y_end, clipped_x_start:x_end]
                target_colored_slice[pixels_to_change] = color_map[new_class_id]
            
            # 3. Trigger localized Qt repaint (respect `silent`)
            if self.graphics_item is not None and not silent:
                qt_rect = QRectF(clipped_x_start, clipped_y_start, x_end - clipped_x_start, y_end - clipped_y_start)
                self.graphics_item.update(qt_rect)

            if history_action is not None:
                history_action.add_change(
                    flat_indices,
                    before_values,
                    np.full(pixels_updated, new_class_id, dtype=self.mask_data.dtype),
                    update_rect=(clipped_x_start, clipped_y_start, x_end, y_end),
                )
            
        else:
            # --- OLD: RAW APPLY & CANVAS SLICE REBUILD ---
            final_brush_mask = clipped_brush_mask & unlocked_pixels_mask
            pixels_updated = np.count_nonzero(final_brush_mask)

            if pixels_updated == 0:
                return

            local_ys, local_xs = np.where(final_brush_mask)
            flat_indices = ((clipped_y_start + local_ys) * mask_w + (clipped_x_start + local_xs)).astype(np.int64)
            before_values = target_slice[final_brush_mask].copy()
            
            target_slice[final_brush_mask] = new_class_id
            
            changed_rect_coords = (clipped_x_start, clipped_y_start, x_end, y_end)
            if not silent:
                self.update_graphics_item(update_rect=changed_rect_coords)

            if history_action is not None:
                history_action.add_change(
                    flat_indices,
                    before_values,
                    np.full(pixels_updated, new_class_id, dtype=self.mask_data.dtype),
                    update_rect=changed_rect_coords,
                )
            
        self._invalidate_stats_cache()
        if not silent:
            self.annotationUpdated.emit(self)
        
    def update_mask_at_indices(self, flat_indices: np.ndarray, class_id: int, silent: bool = False, method: str = "hybrid_diff", history_action=None):
        """
        Paint ``class_id`` at the exact pixel positions given by ``flat_indices``.
        Respects the LOCK_BIT: locked pixels are never overwritten.
        Includes a multi-method router and benchmark timers to test memory injection speeds.
        """

        if flat_indices is None or len(flat_indices) == 0:
            return

        height, width = self.mask_data.shape
        max_idx = height * width - 1

        # Bounds guard: discard any index outside the image
        valid = flat_indices[(flat_indices >= 0) & (flat_indices <= max_idx)]
        if len(valid) == 0:
            return

        # ravel() returns a C-contiguous view so mutations write through to mask_data
        flat_view = self.mask_data.ravel()

        if method in ["flat_raw", "bbox_raw"]:
            # --- OLD: Raw Lock Check Only ---
            unlocked = valid[flat_view[valid] < self.LOCK_BIT]
            if len(unlocked) == 0:
                return
            target_indices = unlocked
        else:
            # --- NEW: Diff-Only Write Filter ---
            current_vals = flat_view[valid]
            # Only target pixels that are unlocked AND actually need to change color
            mask_to_update = (current_vals < self.LOCK_BIT) & (current_vals != class_id)
            target_indices = valid[mask_to_update]
            
            if len(target_indices) == 0:
                return

        pixels_updated = len(target_indices)
        before_values = flat_view[target_indices].copy()

        # 1. Update the semantic mask data
        flat_view[target_indices] = class_id

        # 2. Update the visual canvas — always write colors flat (O(changed
        # pixels), never a rectangular slice recompute). The bbox is computed
        # only to limit the Qt repaint region when the touch count is small.
        if self.colored_mask is not None:
            color_map = self._get_color_map()
            colored_flat = self.colored_mask.reshape(-1, 4)
            colored_flat[target_indices] = color_map[class_id]

        if self.graphics_item is not None and not silent:
            if pixels_updated < 250000:
                y_coords, x_coords = np.divmod(target_indices, width)
                qt_rect = QRectF(max(0, int(x_coords.min()) - 1),
                                 max(0, int(y_coords.min()) - 1),
                                 int(x_coords.max()) - int(x_coords.min()) + 3,
                                 int(y_coords.max()) - int(y_coords.min()) + 3)
                self.graphics_item.update(qt_rect)
            else:
                self.graphics_item.update()

        # 3. Post-update cleanup
        self._invalidate_stats_cache()
        if not silent:
            self.annotationUpdated.emit(self)

        if history_action is not None and pixels_updated > 0:
            y_coords, x_coords = np.divmod(target_indices, width)
            update_rect = (
                max(0, int(x_coords.min()) - 1),
                max(0, int(y_coords.min()) - 1),
                min(width, int(x_coords.max()) + 2),
                min(height, int(y_coords.max()) + 2),
            )
            history_action.add_change(
                target_indices,
                before_values,
                np.full(pixels_updated, class_id, dtype=flat_view.dtype),
                update_rect=update_rect,
            )

    def update_mask_with_mask(self, subset_mask: np.ndarray, top_left: tuple[int, int], silent: bool = False, use_new_method: bool = True, history_action=None):
        """
        Updates a subset area of the mask with a provided mask containing multiple labels.
        Includes A/B testing for Diff-Filtering to skip redundant GPU/Qt updates.
        """

        x, y = top_left
        h, w = subset_mask.shape
        
        # Calculate the region in the full mask where the subset will be applied
        x_end = min(x + w, self.mask_data.shape[1])
        y_end = min(y + h, self.mask_data.shape[0])
        x_start = max(x, 0)
        y_start = max(y, 0)
        
        # Calculate the corresponding region in the subset mask
        sx_start = x_start - x
        sy_start = y_start - y
        sx_end = sx_start + (x_end - x_start)
        sy_end = sy_start + (y_end - y_start)
        
        # Get the target slices
        target_slice = self.mask_data[y_start:y_end, x_start:x_end]
        subset_slice = subset_mask[sy_start:sy_end, sx_start:sx_end]
        
        unlocked_pixels_mask = target_slice < self.LOCK_BIT

        if use_new_method:
            # --- NEW: DIFF FILTER & DIRECT CANVAS INJECTION ---
            # Only update pixels that are unlocked AND actually differ from the subset mask
            pixels_to_change = unlocked_pixels_mask & (target_slice != subset_slice)
            pixels_updated = np.count_nonzero(pixels_to_change)
            
            # FAST EXIT: If the predicted mask matches what is already there, skip redraw!
            if pixels_updated == 0:
                return

            before_values = target_slice[pixels_to_change].copy()
            local_rows, local_cols = np.where(pixels_to_change)
            flat_indices = ((y_start + local_rows) * self.mask_data.shape[1] + (x_start + local_cols)).astype(np.int64)
                
            # 1. Apply the subset mask only to pixels that need changing
            target_slice[pixels_to_change] = subset_slice[pixels_to_change]
            
            # 2. Update visual canvas directly
            if self.colored_mask is not None:
                color_map = self._get_color_map()
                target_colored_slice = self.colored_mask[y_start:y_end, x_start:x_end]
                new_class_ids = subset_slice[pixels_to_change]
                target_colored_slice[pixels_to_change] = color_map[new_class_ids]
            
            # 3. Trigger localized Qt repaint (respect `silent`)
            if self.graphics_item is not None and not silent:
                qt_rect = QRectF(x_start, y_start, x_end - x_start, y_end - y_start)
                self.graphics_item.update(qt_rect)
        else:
            # --- OLD: RAW APPLY & CANVAS SLICE REBUILD ---
            pixels_to_change = unlocked_pixels_mask
            pixels_updated = np.count_nonzero(pixels_to_change)
            
            if pixels_updated == 0:
                return

            before_values = target_slice[pixels_to_change].copy()
            local_rows, local_cols = np.where(pixels_to_change)
            flat_indices = ((y_start + local_rows) * self.mask_data.shape[1] + (x_start + local_cols)).astype(np.int64)
            
            target_slice[pixels_to_change] = subset_slice[pixels_to_change]
            
            update_rect = (x_start, y_start, x_end, y_end)
            if not silent:
                self.update_graphics_item(update_rect=update_rect)
        
        if history_action is not None and pixels_updated > 0:
            history_action.add_change(
                flat_indices,
                before_values,
                subset_slice[pixels_to_change].copy(),
                update_rect=(x_start, y_start, x_end, y_end),
            )
        
        self._invalidate_stats_cache()
        if not silent:
            self.annotationUpdated.emit(self)
        
    def update_mask_with_prediction_mask(self, prediction_mask, top_left=(0, 0), history_action=None):
        """
        Updates a full-size prediction mask with the current mask data.

        This method is non-destructive; it only updates pixels where the
        prediction_mask has a valid (non-background) prediction.
        The final update respects any locked pixels.

        Args:
            prediction_mask (np.ndarray): A full-size (H, W) mask containing
                                          the new predictions.
        """
        if prediction_mask.shape != self.mask_data.shape:
            print(f"Error: Prediction mask shape {prediction_mask.shape} "
                  f"does not match annotation shape {self.mask_data.shape}.")
            return

        # 1. Get (row, col) indices of all valid predictions (class > 0)
        #    np.nonzero is highly efficient for sparse arrays.
        pred_rows, pred_cols = np.nonzero(prediction_mask)
        
        if pred_rows.size == 0:
            return  # No predictions, nothing to do.

        # 2. Find the bounding box (Region of Interest) of the predictions
        min_row, max_row = np.min(pred_rows), np.max(pred_rows)
        min_col, max_col = np.min(pred_cols), np.max(pred_cols)
        
        # 3. Define the top-left (x, y) corner for pasting
        #    Note: col = x, row = y
        paste_top_left = (min_col, min_row)
        
        # 4. Extract the *small tile* from the prediction mask
        prediction_tile = prediction_mask[min_row: max_row + 1, min_col: max_col + 1]

        # 5. Extract the corresponding *small tile* from the *current* mask data
        original_tile = self.mask_data[min_row: max_row + 1, min_col: max_col + 1]

        # 6. Create the *merged tile*
        #    Use np.where: if pred_tile > 0, use its value, else use original value.
        merged_tile = np.where(
            prediction_tile > 0,
            prediction_tile,
            original_tile
        )
        
        # 7. Call the existing update method with the *small merged tile*
        #    This reuses all your existing logic for locked pixels and
        #    graphics updates, but now only on a small, efficient region.
        self.update_mask_with_mask(merged_tile, paste_top_left, history_action=history_action)
        
    def get_current_transparency(self):
        """Get the current transparency value for rendering."""
        # Try to get the active label's transparency from the application
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                main_window = app.activeWindow()
                if hasattr(main_window, 'label_window') and hasattr(main_window.label_window, 'active_label'):
                    active_label = main_window.label_window.active_label
                    if active_label:
                        return active_label.transparency
        except Exception:
            pass
        return self.transparency
    
    def update_transparency(self, transparency):
        """Update transparency instantly using render-time application."""
        transparency = max(0, min(255, transparency))  # Clamp to valid range
        if self.transparency != transparency:
            self.transparency = transparency
            # Update transparency on all visible labels
            for label_id in self.visible_label_ids:
                label = next((lbl for lbl in self.class_id_to_label_map.values() if lbl.id == label_id), None)
                if label:
                    label.transparency = transparency
            
            # Trigger repaint - transparency applied during rendering
            if self.graphics_item:
                self.graphics_item.update()
            
    def update_visible_labels(self, visible_ids: set):
        """Updates the set of visible label IDs and triggers a redraw of the mask."""
        self.visible_label_ids = visible_ids
        # Invalidate cache since visibility dictates alpha channels in the map
        self.invalidate_color_map()
        self.update_graphics_item()
            
    def remove_from_scene(self):
        """Removes the graphics item from its scene, if it exists."""
        if self.graphics_item:
            try:
                scene = self.graphics_item.scene()
                if scene is not None:
                    scene.removeItem(self.graphics_item)
            except RuntimeError:
                # The underlying C++ item may already have been deleted by a scene clear.
                pass

        # Remove the graphics item reference
        self.graphics_item = None

    # --- Data Manipulation & Editing Methods ---

    def fill_region(self, point: QPointF, new_class_id: int, history_action=None, silent: bool = False, return_update_rect: bool = False):
        """
        Fills a contiguous region with a new class ID using optimized OpenCV floodFill, 
        respecting pre-locked pixels.
        
        Returns:
            numpy.ndarray or None: Boolean mask of pixels that were filled, or None if fill failed
        """
        def _maybe_return(fill_result, rect=None):
            if return_update_rect:
                return fill_result, rect
            return fill_result

        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if not (0 <= y < height and 0 <= x < width):
            return _maybe_return(None)

        # Check if the starting pixel is locked. If so, we cannot fill from it.
        if self.mask_data[y, x] >= self.LOCK_BIT:
            return _maybe_return(None)

        old_class_id = self.mask_data[y, x]
        if old_class_id == new_class_id:
            return _maybe_return(None)

        import cv2
        if not silent:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import Qt
            QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create a mask padded by 1 pixel on all sides (required by OpenCV)
        cv_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

        # 4-way connectivity, fill the cv_mask with 255
        flags = 4 | (255 << 8)

        # We pass a copy of the mask_data to floodFill so we don't accidentally overwrite locked pixels.
        # This extremely fast operation just generates the boolean fill shape for us.
        _, _, _, rect = cv2.floodFill(
            image=self.mask_data.copy(),
            mask=cv_mask, 
            seedPoint=(x, y), 
            newVal=new_class_id, 
            loDiff=0, 
            upDiff=0, 
            flags=flags
        )

        # Extract the unpadded boolean mask where pixels were successfully filled
        fill_mask = cv_mask[1:-1, 1:-1] == 255

        flat_indices = np.flatnonzero(fill_mask.ravel())
        before_values = self.mask_data.ravel()[flat_indices].copy()
        
        # Apply the fill to the actual data. Locked pixels inherently stopped the floodFill
        # because they have a different class ID (e.g., old_class_id + 128), so they are safe.
        self.mask_data[fill_mask] = new_class_id
        after_values = self.mask_data.ravel()[flat_indices].copy()
        
        # Use OpenCV's exact bounding box (x, y, w, h) for a lightning-fast localized update
        x_min, y_min, fill_w, fill_h = rect
        
        # Add 1px padding to the slice bounds to ensure edge anti-aliasing renders cleanly
        update_rect = (
            max(0, x_min - 1), 
            max(0, y_min - 1), 
            min(width, x_min + fill_w + 1), 
            min(height, y_min + fill_h + 1)
        )
        if not silent:
            self.update_graphics_item(update_rect=update_rect)

        if history_action is not None and flat_indices.size > 0:
            history_action.add_change(
                flat_indices,
                before_values,
                after_values,
                update_rect=update_rect,
            )

        self._invalidate_stats_cache()
        if not silent:
            QApplication.restoreOverrideCursor()
            self.annotationUpdated.emit(self)
        
        # Return the fill_mask so callers can know which pixels were filled
        return _maybe_return(fill_mask, update_rect)

    def _fast_rasterize(self, geometries, width, height, mode="rasterio"):
        """
        Fast vectorized rasterization using either Shapely or Rasterio.
        
        Args:
            geometries: List of Shapely Polygon geometries
            width, height: Dimensions of the output mask
            mode: "shapely" for Shapely vectorized method, "rasterio" for rasterio.features.rasterize
        
        Returns:
            Boolean numpy array where True indicates pixels covered by geometries
        """
        if mode == "shapely":
            # Use Shapely vectorized method (very fast for many points)
            from shapely.vectorized import contains
            
            # Create coordinate grids - much faster than manual iteration
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Create empty mask
            raster_mask = np.zeros((height, width), dtype=bool)
            
            # Process each geometry
            for geom in geometries:
                if geom.is_valid:
                    # Vectorized point-in-polygon check - VERY fast
                    mask = contains(geom, x_coords, y_coords)
                    raster_mask = raster_mask | mask
                    
            return raster_mask
            
        elif mode == "rasterio":
            # Use rasterio.features.rasterize (more robust for complex geometries)
            raster_mask = rasterize(
                geometries,
                out_shape=(height, width),
                fill=0,
                default_value=1,
                dtype=np.uint8
            ).astype(bool)
            
            return raster_mask
        
        else:
            raise ValueError(f"Unknown rasterization mode: {mode}. Use 'shapely' or 'rasterio'.")
        
    def rasterize_annotations(self, all_annotations: list, history_action=None):
        """
        Unified method to sync vector annotations with the mask.
        
        Args:
            all_annotations: List of vector annotations to process
        """
        if not all_annotations:
            return  # Nothing to do if no annotations

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            height, width = self.mask_data.shape

            # Section 1: Build geometries list AND filter annotations that actually need processing
            geometries = []
            annotations_to_process = []
            all_bounds = []

            for annotation in all_annotations:
                if getattr(annotation, 'is_mask_annotation', False):
                    continue
                try:
                    geometry = self._get_annotation_rasterization_geometry(annotation)
                    if geometry is None or getattr(geometry, 'is_empty', False):
                        continue

                    # Get the class ID for this annotation's label
                    class_id = self.label_id_to_class_id_map.get(annotation.label.id)
                    if class_id is None:
                        continue  # Skip if label not in mask

                    geometries.append(geometry)
                    annotations_to_process.append(annotation)
                    all_bounds.append(geometry.bounds)

                except Exception as e:
                    print(f"Warning: Could not process annotation {annotation.id}: {e}")
                    continue

            if not geometries:
                # No valid geometries to rasterize
                return

            # Section 2: Rasterize geometries
            annotation_mask = self._fast_rasterize(geometries, width, height, mode="rasterio")
            flat_indices = np.flatnonzero(annotation_mask.ravel())
            before_values = self.mask_data.ravel()[flat_indices].copy()

            # Section 3: Clear mask pixels under annotations FIRST
            if np.any(annotation_mask):
                # Clear mask pixels under annotations (set to 0)
                self.mask_data[annotation_mask] = 0

            # Section 4: Apply locking to the cleared areas
            # Only lock pixels that are NOT already locked
            to_lock = annotation_mask & (self.mask_data < self.LOCK_BIT)
            if np.any(to_lock):
                self.mask_data[to_lock] += self.LOCK_BIT

            # Invalidate the cache since we modified the mask data
            if np.any(annotation_mask) or np.any(to_lock):
                self._invalidate_stats_cache()

            # Calculate smart combined bounding box
            if all_bounds:
                min_x = min(b[0] for b in all_bounds)
                min_y = min(b[1] for b in all_bounds)
                max_x = max(b[2] for b in all_bounds)
                max_y = max(b[3] for b in all_bounds)

                # Add padding (5 pixels) to handle anti-aliasing edges
                x_min = max(0, int(min_x) - 5)
                y_min = max(0, int(min_y) - 5)
                x_max = min(width, int(max_x) + 6)  # +1 for inclusive, +5 for padding
                y_max = min(height, int(max_y) + 6)

                update_rect = (x_min, y_min, x_max, y_max)
            else:
                update_rect = None

            if history_action is not None and flat_indices.size > 0:
                after_values = self.mask_data.ravel()[flat_indices].copy()
                history_action.add_change(
                    flat_indices,
                    before_values,
                    after_values,
                    update_rect=update_rect,
                )
        finally:
            # Always restore the cursor, even on early return / exception.
            QApplication.restoreOverrideCursor()

    def bake_annotations(self, all_annotations: list, history_action=None):
        """Bake vector annotations into the semantic mask and return a summary.

        Unlike rasterize_annotations(), this permanently writes the annotation
        classes into the mask rather than locking the covered pixels. Existing
        vector annotations should be removed by the caller as part of the same
        undoable user action.
        """
        summary = {
            "baked_annotations": [],
            "skipped_annotations": [],
            "changed_pixels": 0,
            "update_rect": None,
        }

        if not all_annotations:
            return summary

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            height, width = self.mask_data.shape
            shapes = []
            all_bounds = []

            for annotation in all_annotations:
                if getattr(annotation, 'is_mask_annotation', False):
                    continue

                try:
                    geometry = self._get_annotation_rasterization_geometry(annotation)
                    if geometry is None or getattr(geometry, 'is_empty', False):
                        summary["skipped_annotations"].append(annotation)
                        continue

                    class_id = self.label_id_to_class_id_map.get(annotation.label.id)
                    if class_id is None:
                        summary["skipped_annotations"].append(annotation)
                        continue

                    shapes.append((geometry, class_id))
                    summary["baked_annotations"].append(annotation)
                    all_bounds.append(geometry.bounds)

                except Exception as e:
                    print(f"Warning: Could not bake annotation {annotation.id}: {e}")
                    summary["skipped_annotations"].append(annotation)
                    continue

            if not shapes:
                return summary

            baked_values = rasterize(
                shapes,
                out_shape=(height, width),
                fill=0,
                default_value=0,
                dtype=np.uint8,
            ).astype(np.uint8)

            target_mask = baked_values != 0
            flat_indices = np.flatnonzero(target_mask.ravel())
            if flat_indices.size == 0:
                return summary

            flat_values = baked_values.ravel()[flat_indices]

            if all_bounds:
                min_x = min(b[0] for b in all_bounds)
                min_y = min(b[1] for b in all_bounds)
                max_x = max(b[2] for b in all_bounds)
                max_y = max(b[3] for b in all_bounds)

                x_min = max(0, int(min_x) - 5)
                y_min = max(0, int(min_y) - 5)
                x_max = min(width, int(max_x) + 6)
                y_max = min(height, int(max_y) + 6)
                update_rect = (x_min, y_min, x_max, y_max)
            else:
                update_rect = None

            applied = self.apply_flat_values_at_indices(
                flat_indices,
                flat_values,
                silent=False,
                update_rect=update_rect,
            )

            if applied is None:
                summary["update_rect"] = update_rect
                return summary

            if history_action is not None:
                history_action.add_change(
                    applied["flat_indices"],
                    applied["before_values"],
                    applied["after_values"],
                    update_rect=applied["update_rect"],
                )

            summary["changed_pixels"] = int(applied["flat_indices"].size)
            summary["update_rect"] = applied["update_rect"]
            return summary

        finally:
            QApplication.restoreOverrideCursor()

        # Section 5: Smart localized repaint
        if np.any(to_lock) or np.any(annotation_mask):
            if update_rect and (x_max - x_min) * (y_max - y_min) < width * height * 0.3:  # If < 30% of image
                # Use localized update - much faster
                self.update_graphics_item(update_rect=update_rect)
            else:
                # Fall back to full update if annotations cover too much area
                self.update_graphics_item()

            self.annotationUpdated.emit(self)

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def unrasterize_annotations(self, history_action=None):
        """
        Remove lock protection from all pixels that were marked as locked.
        This allows mask editing over previously protected vector annotation areas.
        """
        # Find all pixels that have the lock bit set and remove it
        locked_pixels = self.mask_data >= self.LOCK_BIT
        flat_indices = np.flatnonzero(locked_pixels.ravel())
        before_values = self.mask_data.ravel()[flat_indices].copy()

        # Remove the lock bit from these pixels, keeping their original class
        if np.any(locked_pixels):
            self.mask_data[locked_pixels] = self.mask_data[locked_pixels] - self.LOCK_BIT

        if history_action is not None and flat_indices.size > 0:
            after_values = self.mask_data.ravel()[flat_indices].copy()
            history_action.add_change(
                flat_indices,
                before_values,
                after_values,
                update_rect=None,
            )

        if self.graphics_item is not None and np.any(locked_pixels):
            self.update_graphics_item()

        if np.any(locked_pixels):
            self.annotationUpdated.emit(self)

    def clear_pixels_for_class(self, class_id: int, history_action=None):
        """Finds all pixels matching a class ID (both locked and unlocked) and resets them to 0."""
        if class_id == 0:  # Cannot clear background class
            return

        # Create a boolean mask of all pixels whose real class ID matches.
        pixels_to_clear = (self.mask_data % self.LOCK_BIT) == class_id
        flat_indices = np.flatnonzero(pixels_to_clear.ravel())
        before_values = self.mask_data.ravel()[flat_indices].copy()

        # Set these pixels back to 0 (unclassified).
        self.mask_data[pixels_to_clear] = 0
        if history_action is not None and flat_indices.size > 0:
            history_action.add_change(
                flat_indices,
                before_values,
                np.zeros(flat_indices.size, dtype=self.mask_data.dtype),
                update_rect=None,
            )

        if np.any(pixels_to_clear):
            coords = np.where(pixels_to_clear)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            update_rect = (max(0, x_min - 1), max(0, y_min - 1), min(self.mask_data.shape[1], x_max + 2), min(self.mask_data.shape[0], y_max + 2))
            self.update_graphics_item(update_rect=update_rect)

        self._invalidate_stats_cache()
        if np.any(pixels_to_clear):
            self.annotationUpdated.emit(self)

    def clear_pixels_for_annotations(self, annotations_to_clear: list, history_action=None):
        """
        Rasterizes a list of vector annotations and sets the corresponding
        pixels in the mask_data to 0 (unclassified).
        """
        if not annotations_to_clear:
            return

        # 1. Build a precise mask from any stored source indices, then fall back
        #    to rasterizing the annotation geometry when no exact indices exist.
        exact_flat_indices = []
        geometries = []
        for anno in annotations_to_clear:
            source_indices = getattr(anno, '_source_clear_indices', None)
            if source_indices is not None:
                indices_array = np.asarray(source_indices, dtype=np.int64).ravel()
                if indices_array.size:
                    exact_flat_indices.append(indices_array)
                    continue

            geometry = None

            geometry_getter = getattr(anno, 'get_rasterization_geometry', None)
            if callable(geometry_getter):
                try:
                    geometry = geometry_getter()
                except Exception:
                    geometry = None

            if geometry is None and hasattr(anno, 'get_polygon'):
                try:
                    qt_polygon = anno.get_polygon()
                    points = [(p.x(), p.y()) for p in qt_polygon]
                    if len(points) >= 3:
                        geometry = Polygon(points)
                except Exception:
                    geometry = None

            if geometry is not None and not getattr(geometry, 'is_empty', False):
                geometries.append(geometry)

        height, width = self.mask_data.shape

        clear_mask = None
        if exact_flat_indices:
            flat_mask = np.zeros(height * width, dtype=bool)
            flat_mask[np.unique(np.concatenate(exact_flat_indices))] = True
            clear_mask = flat_mask.reshape((height, width))

        if geometries:
            geometry_mask = rasterize(
                geometries,
                out_shape=(height, width),
                fill=0,
                default_value=1,
                all_touched=True,
                dtype=np.uint8,
            ).astype(bool)
            clear_mask = geometry_mask if clear_mask is None else (clear_mask | geometry_mask)

        if clear_mask is None or not np.any(clear_mask):
            return

        flat_indices = np.flatnonzero(clear_mask.ravel())
        before_values = self.mask_data.ravel()[flat_indices].copy()

        # 3. Apply the mask to the data, setting pixels to 0.
        self.mask_data[clear_mask] = 0
        self._invalidate_stats_cache()

        if history_action is not None and flat_indices.size > 0:
            history_action.add_change(
                flat_indices,
                before_values,
                np.zeros(flat_indices.size, dtype=self.mask_data.dtype),
                update_rect=None,
            )

        # 4. Trigger a localized repaint of the mask to show the changes efficiently.
        coords = np.where(clear_mask)
        if len(coords[0]) > 0:
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            
            # Add a 1px padding to the slice bounds and clamp to image dimensions
            update_rect = (
                max(0, x_min - 1), 
                max(0, y_min - 1), 
                min(width, x_max + 2), 
                min(height, y_max + 2)
            )
            self.update_graphics_item(update_rect=update_rect)

        if np.any(clear_mask):
            self.annotationUpdated.emit(self)

    # --- Analysis & Information Retrieval Methods ---

    def _invalidate_stats_cache(self):
        """Invalidates the statistics cache.
        This should be called by any method that modifies self.mask_data.
        """
        self._stats_cache = None

    def recalculate_class_statistics(self) -> dict:
        """
        Performs the expensive calculation of class statistics and updates the cache.
        This is called proactively when mask editing is finished.
        """
        stats = {}
        
        # 1. Use modulus to get the "unlocked" class ID for every pixel.
        # This is a single, fast, vectorized operation.
        unlocked_mask_data = self.mask_data % self.LOCK_BIT

        # 2. Get total pixels in the entire image.
        total_image_pixels = self.mask_data.size

        # 3. Handle division by zero for an empty/invalid image
        if total_image_pixels == 0:
            self._stats_cache = stats
            return self._stats_cache

        # 4. Determine the minimum length for the bincount array
        # to ensure it's large enough for all known class IDs.
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        min_len = max_id + 1

        # 5. Get counts for ALL class IDs in one O(N) pass.
        bin_counts = np.bincount(unlocked_mask_data.ravel(), minlength=min_len)

        # 6. Populate stats by iterating only through the labels we care about.
        for class_id, label in self.class_id_to_label_map.items():
            # class_id is already guaranteed to be < min_len from step 4
            count = bin_counts[class_id]
            if count > 0:
                stats[label.short_label_code] = {
                    "pixel_count": int(count),
                    # FIXED: Percentage is (count / total_image_pixels)
                    "percentage": (count / total_image_pixels) * 100
                }
        
        self._stats_cache = stats
        return self._stats_cache

    def get_class_statistics(self) -> dict:
        """
        Gets class statistics from the cache.
        If the cache is empty, it triggers a one-time recalculation.
        """
        # If cache is empty, calculate it
        if self._stats_cache is None:
            # This is the "lazy" part, acting as a safety net
            self.recalculate_class_statistics()
        
        return self._stats_cache
    
    @property
    def cached_statistics(self) -> dict | None:
        """
        Safely returns the cached statistics dictionary, or None if not cached.
        This method will NOT trigger a recalculation.
        """
        return self._stats_cache

    def get_class_at_point(self, point: QPointF) -> int:
        """Returns the class ID at a specific point."""
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if 0 <= y < height and 0 <= x < width:
            return self.mask_data[y, x]
        return 0  # Return background class if outside bounds

    # --- Conversion & Exporting Methods ---
    
    def get_binary_mask(self, class_id: int) -> np.ndarray:
        """Returns a boolean numpy array where True corresponds to the given class ID."""
        return self.mask_data == class_id

    def to_instance_polygons(self, class_id: int) -> list:
        """Converts all contiguous regions of a class ID into PolygonAnnotations.

        Uses cv2.findContours with CHAIN_APPROX_TC89_KCOS (replaces skimage
        find_contours) for consistency with to_vector_annotations and better
        performance. No padding needed — OpenCV handles image-border contours.
        """
        import cv2

        label = self.class_id_to_label_map.get(class_id)
        if label is None:
            return []

        binary_mask = self.get_binary_mask(class_id).astype(np.uint8)

        # TC89_KCOS pre-filters dominant points at extraction time,
        # reducing vertex count before approxPolyDP even runs.
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )

        annotations = []
        for contour in contours:
            simplified = cv2.approxPolyDP(contour, 1.0, closed=True)
            points = [QPointF(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]
            if len(points) >= 3:
                anno = PolygonAnnotation(
                    points=points,
                    label=label,
                    image_path=self.image_path,
                )
                annotations.append(anno)
        return annotations

    def to_vector_annotations(self, transparency=None, show_confidence: bool = False,
                               min_hole_area: int = 500, min_component_area: int = 5) -> list:
        """Convert all labeled regions in this mask into vector annotations.

        Disconnected regions become separate annotations. Four-point, axis-aligned
        square regions become PatchAnnotation objects; four-point, axis-aligned
        non-squares become RectangleAnnotation objects; everything else becomes a
        PolygonAnnotation.

        Holes (interior voids) are handled selectively: holes whose area in pixels
        is at least *min_hole_area* are preserved as interior rings in the resulting
        PolygonAnnotation, giving an accurate representation of large voids (e.g. a
        sand patch inside a coral colony). Holes smaller than *min_hole_area* are
        silently filled, avoiding the vertex explosion that comes from tracing every
        noise-level gap while retaining meaningful structure.

        Performance notes
        -----------------
        * ``cv2.connectedComponentsWithStats`` replaces ``scipy.ndimage.label`` —
          ~2-4x faster and returns bounding-box + area for every component for free,
          enabling early noise rejection before any contour work is done.
        * Contour extraction is ROI-based: each component is cropped to its own
          bounding rect before ``findContours`` runs, so a 200×200 px object in a
          4000×4000 image operates on a ~200×200 patch rather than the full image.
        * ``CHAIN_APPROX_TC89_KCOS`` pre-filters dominant points at extraction
          time, reducing the vertex count fed to ``approxPolyDP``.
        * ``_source_clear_indices`` is derived by filtering the class's pre-built
          flat-index array rather than running a full O(H×W) ``flatnonzero`` scan
          inside the inner loop.
        Args:
            transparency: Alpha value for created annotations (0-255).
            show_confidence: Whether to display confidence scores.
            min_hole_area: Minimum hole area in pixels to preserve as an interior
                ring. Holes smaller than this threshold are filled. Default 500.
            min_component_area: Minimum component area in pixels. Components
                smaller than this are skipped as noise. Default 5.
        """
        try:
            import cv2
            from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
            from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
        except Exception:
            return []

        if transparency is None:
            transparency = getattr(self, 'transparency', 128)

        def _contour_to_points(contour_array):
            contour_array = np.asarray(contour_array)
            if contour_array.size == 0:
                return []
            return [QPointF(float(x), float(y)) for x, y in contour_array.reshape(-1, 2)]

        def _is_axis_aligned_quad(points, tolerance=1.0):
            if len(points) != 4:
                return False
            coordinates = [(point.x(), point.y()) for point in points]
            for index in range(4):
                x1, y1 = coordinates[index]
                x2, y2 = coordinates[(index + 1) % 4]
                if not (abs(x2 - x1) <= tolerance or abs(y2 - y1) <= tolerance):
                    return False
            return True

        def _build_annotation(label, exterior_points, hole_points_list):
            if len(exterior_points) == 4 and not hole_points_list and _is_axis_aligned_quad(exterior_points):
                min_x = min(point.x() for point in exterior_points)
                min_y = min(point.y() for point in exterior_points)
                max_x = max(point.x() for point in exterior_points)
                max_y = max(point.y() for point in exterior_points)
                width = abs(max_x - min_x)
                height = abs(max_y - min_y)
                center_xy = QPointF((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

                if abs(width - height) <= 1.0:
                    return PatchAnnotation(
                        center_xy=center_xy,
                        annotation_size=max(1, int(round(max(width, height)))),
                        label=label,
                        image_path=self.image_path,
                        transparency=transparency,
                        show_confidence=show_confidence,
                    )

                return RectangleAnnotation(
                    top_left=QPointF(min_x, min_y),
                    bottom_right=QPointF(max_x, max_y),
                    label=label,
                    image_path=self.image_path,
                    transparency=transparency,
                    show_confidence=show_confidence,
                )

            return PolygonAnnotation(
                points=exterior_points,
                holes=hole_points_list,
                label=label,
                image_path=self.image_path,
                transparency=transparency,
                show_confidence=show_confidence,
            )

        # Strip lock bits to recover the true class IDs for every pixel.
        class_mask = np.where(
            self.mask_data >= self.LOCK_BIT,
            self.mask_data % self.LOCK_BIT,
            self.mask_data,
        )

        mask_h, mask_w = class_mask.shape
        vector_annotations = []
        for class_id in [int(value) for value in np.unique(class_mask) if int(value) != 0]:
            label = self.class_id_to_label_map.get(class_id)
            if label is None:
                continue

            binary_mask = (class_mask == class_id).astype(np.uint8)
            if not np.any(binary_mask):
                continue

            # --- Fast connected-component analysis with free per-component stats ---
            # cv2.connectedComponentsWithStats is ~2-4x faster than scipy.ndimage.label
            # and returns bounding-box + area for every component at no extra cost,
            # enabling early noise rejection before any contour work is done.
            num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=4
            )
            if num_labels <= 1:
                continue  # nothing but background

            # Pre-compute flat indices for ALL pixels of this class in one O(N) pass.
            # Per-component sets are derived by filtering this sparse array, avoiding
            # a full O(H×W) flatnonzero scan inside the inner loop.
            all_class_flat_indices = np.flatnonzero(binary_mask.ravel())
            label_flat = label_img.ravel()

            for comp_id in range(1, num_labels):

                # --- Early noise reject (bounding-box stats are free from above) ---
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < min_component_area:
                    continue

                # --- ROI-based contour extraction ---
                # Crop to this component's bounding rect so that findContours operates
                # on a small patch rather than the full image. For a 200×200 px object
                # in a 4000×4000 image, that is roughly a 400x reduction in pixels
                # scanned.
                cx = int(stats[comp_id, cv2.CC_STAT_LEFT])
                cy = int(stats[comp_id, cv2.CC_STAT_TOP])
                cw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                ch = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

                # 1 px padding ensures contours that touch the crop edge are captured.
                x1 = max(0, cx - 1)
                y1 = max(0, cy - 1)
                x2 = min(mask_w, cx + cw + 1)
                y2 = min(mask_h, cy + ch + 1)

                roi = (label_img[y1:y2, x1:x2] == comp_id).astype(np.uint8)

                # TC89_KCOS applies the Teh-Chin dominant-point algorithm at
                # extraction time, pre-thinning the contour before approxPolyDP
                # runs — fewer input vertices means faster simplification.
                contours, hierarchy = cv2.findContours(
                    roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
                )
                if not contours or hierarchy is None:
                    continue

                hier = hierarchy[0]

                # Efficient per-component flat-index set: filter the pre-built
                # class index array by this component's label ID — no full image scan.
                component_flat_indices = all_class_flat_indices[
                    label_flat[all_class_flat_indices] == comp_id
                ]

                for i, contour in enumerate(contours):
                    # Only process top-level (exterior) contours; holes are
                    # collected below via the child-index chain.
                    if hier[i][3] != -1:
                        continue

                    # Offset contour points from ROI-local space back to full-image
                    # space before running approxPolyDP.
                    contour_offset = contour.copy()
                    contour_offset[:, :, 0] += x1
                    contour_offset[:, :, 1] += y1

                    simplified_contour = cv2.approxPolyDP(contour_offset, 1.0, closed=True)
                    exterior_points = _contour_to_points(simplified_contour)
                    if len(exterior_points) < 3:
                        continue

                    # Walk the child chain to collect significant holes.
                    interior_rings = []
                    child_idx = hier[i][2]  # index of first hole, -1 if none
                    while child_idx != -1:
                        hole_contour = contours[child_idx]
                        if cv2.contourArea(hole_contour) >= min_hole_area:
                            hole_offset = hole_contour.copy()
                            hole_offset[:, :, 0] += x1
                            hole_offset[:, :, 1] += y1
                            simplified_hole = cv2.approxPolyDP(hole_offset, 1.0, closed=True)
                            hole_points = _contour_to_points(simplified_hole)
                            if len(hole_points) >= 3:
                                interior_rings.append(hole_points)
                        child_idx = hier[child_idx][0]  # next sibling hole

                    annotation = _build_annotation(label, exterior_points, interior_rings)
                    if annotation is not None:
                        annotation._source_clear_indices = component_flat_indices
                        vector_annotations.append(annotation)

        return vector_annotations


    def export_as_png(self, path: str, use_label_colors: bool = True):
        """Saves the mask to a PNG file."""
        if use_label_colors:
            # Use current colored image for export
            self._ensure_canvas()
            self.qimage.save(path)
        else:
            # Save the raw class IDs as a grayscale image
            height, width = self.mask_data.shape
            # Ensure data is in a format QImage can handle (e.g., 8-bit grayscale)
            if self.mask_data.max() < 256:
                img_data = self.mask_data.astype(np.uint8)
                q_image = QImage(img_data.data, width, height, QImage.Format_Grayscale8)
                q_image.save(path)
            else:
                warnings.warn("Mask contains class IDs > 255; cannot save as 8-bit grayscale PNG.")

    def export_as_raster(self, path: str):
        """Saves the mask data to a raster file (e.g., GeoTIFF) using rasterio."""
        profile = {
            'driver': 'GTiff',
            'height': self.mask_data.shape[0],
            'width': self.mask_data.shape[1],
            'count': 1,
            'dtype': self.mask_data.dtype
        }
        
        # If the original image was opened with rasterio, copy its spatial metadata
        if self.rasterio_src:
            profile['crs'] = self.rasterio_src.crs
            profile['transform'] = self.rasterio_src.transform

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(self.mask_data, 1)

    # --- Serialization & Deserialization ---

    def to_dict(self):
        """Serialize the annotation to a dictionary, with crop-RLE for the mask.

        Each class is cropped to its bounding box before RLE encoding.  This
        is far more compact than full-image RLE for small or sparse classes
        (e.g. a 200×200 object in a 4000×4000 image).  The bbox offset stored
        alongside the RLE lets ``from_dict`` reconstruct the full mask exactly.
        Old project files that lack the ``bbox`` key are still handled correctly.
        """
        base_dict = super().to_dict()

        # Encode each class's binary mask using crop-RLE (compact format)
        rle_list = []
        unique_classes = np.unique(self.mask_data)
        for class_id in unique_classes:
            if class_id == 0:
                continue
            binary_mask = (self.mask_data == class_id).astype(np.uint8)

            # Find the tight bounding box of this class's pixels.
            rows_any = binary_mask.any(axis=1)
            cols_any = binary_mask.any(axis=0)
            y_idx = np.where(rows_any)[0]
            x_idx = np.where(cols_any)[0]
            if y_idx.size == 0:
                continue  # class present in map but not in mask data
            y1, y2 = int(y_idx[0]), int(y_idx[-1])
            x1, x2 = int(x_idx[0]), int(x_idx[-1])

            # RLE-encode only the crop, not the full image.
            crop = binary_mask[y1:y2 + 1, x1:x2 + 1]
            rle = mask.encode(np.asfortranarray(crop))
            rle['counts'] = base64.b64encode(rle['counts']).decode('ascii')
            rle_list.append({
                'class_id': int(class_id),
                'rle': rle,
                'bbox': [x1, y1, x2, y2],  # inclusive xyxy offset for decode
            })
        
        # Convert the label map to a serializable format
        serializable_label_map = {}
        for cid, label in self.class_id_to_label_map.items():
            serializable_label_map[cid] = label.short_label_code

        base_dict.update({
            'shape': self.mask_data.shape,
            'rle_masks': rle_list,
            'label_map': serializable_label_map
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Instantiate a MaskAnnotation from a dictionary."""
        # Get all labels currently in the project. This is needed for the constructor.
        all_project_labels = list(label_window.labels)
        if not all_project_labels:
            raise ValueError("Cannot import a MaskAnnotation without any labels loaded in the project.")

        # Decode the RLE mask data (supports both crop-RLE and legacy full-image RLE)
        shape = tuple(data['shape'])
        mask_data = np.zeros(shape, dtype=np.uint8)
        for item in data['rle_masks']:
            class_id = item['class_id']
            rle = item['rle']
            try:
                rle['counts'] = base64.b64decode(rle['counts'])
                if 'bbox' in item:
                    # Compact format: RLE covers the bounding-box crop only.
                    x1, y1, x2, y2 = item['bbox']
                    crop_mask = mask.decode(rle).astype(bool)
                    sub = mask_data[y1:y2 + 1, x1:x2 + 1]
                    sub[crop_mask] = class_id
                else:
                    # Legacy format: RLE covers the full image.
                    binary_mask = mask.decode(rle).astype(bool)
                    if binary_mask.shape != shape:
                        print(f"Warning: RLE decoded shape {binary_mask.shape} does not match expected shape {shape}")
                        continue
                    mask_data[binary_mask] = class_id
            except Exception as e:
                print(f"Error decoding RLE for class {class_id}: {e}")
                continue

        # Create the base annotation instance. It will have a generic label map initially.
        annotation = cls(
            image_path=data['image_path'],
            mask_data=mask_data,
            initial_labels=all_project_labels
        )
        
        # Clear the generic maps created by the constructor.
        annotation.class_id_to_label_map.clear()
        annotation.label_id_to_class_id_map.clear()
        
        max_id_found = 0
        if 'label_map' in data and data['label_map']:
            # Iterate through the saved map {class_id: short_code}
            for cid_str, short_code in data['label_map'].items():
                class_id = int(cid_str)
                label = label_window.get_label_by_short_code(short_code)
                
                if label:
                    # Rebuild the maps with the correct associations
                    annotation.class_id_to_label_map[class_id] = label
                    annotation.label_id_to_class_id_map[label.id] = class_id
                    if class_id > max_id_found:
                        max_id_found = class_id
                else:
                    print(f"Warning: Label with short code '{short_code}' not found in project during mask import.")

        # Ensure the next class ID is set correctly to avoid future conflicts.
        annotation.next_class_id = max_id_found + 1

        # Restore other annotation properties
        annotation.id = data.get('id', annotation.id)
        annotation.data = data.get('data', {})
        
        return annotation

    @classmethod
    def from_rasterio(cls, file_path: str, image_path: str, all_labels: list):
        """Creates a MaskAnnotation instance by loading data from a raster file."""
        with rasterio.open(file_path) as src:
            mask_data = src.read(1)
            return cls(
                image_path=image_path,
                mask_data=mask_data,
                initial_labels=all_labels,
                rasterio_src=src
            )

    # --- Compatibility Methods ---
    def get_perimeter(self):
        height, width = self.mask_data.shape
        return 2 * (width + height)
    
    def get_polygon(self):
        height, width = self.mask_data.shape
        return QPolygonF(QRectF(0, 0, width, height))

    def __repr__(self):
        return (f"MaskAnnotation(id={self.id}, image_path={self.image_path}, "
                f"shape={self.mask_data.shape})")