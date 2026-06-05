"""
PropagationEngine — mask propagation, multi-annotate stroke handling, and
projection logic for MVAT.

All heavy propagation computation lives here.  The engine holds its own state
(multi_annotate_enabled, buffer pool, thread executors) and delegates everything
else to the owning MVATManager via __getattr__ so method bodies can be moved
verbatim without rewriting every self.xxx reference.
"""

import os
import numpy as np
import traceback
from time import perf_counter
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPointF
from PyQt5.QtWidgets import QApplication, QMessageBox

from coralnet_toolbox.MVAT.core.Ray import CameraRay

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation


# -------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------


def resolve_class_conflicts_vectorized(element_ids: np.ndarray, class_ids: np.ndarray):
    """Resolve per-element class conflicts using vectorized vote counts."""
    try:
        element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
    except Exception:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if element_ids.size == 0 or class_ids.size == 0 or element_ids.size != class_ids.size:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    max_classes = max(100000, int(np.max(class_ids)) + 1)
    compound_ids = (element_ids * max_classes) + class_ids

    unique_compounds, vote_counts = np.unique(compound_ids, return_counts=True)
    unique_elements = unique_compounds // max_classes
    unique_classes = unique_compounds % max_classes

    # Within each element group, keep the highest vote count and prefer the
    # smaller class ID when the vote count is tied.
    sort_indices = np.lexsort((-unique_classes, vote_counts, unique_elements))
    sorted_elements = unique_elements[sort_indices]
    sorted_classes = unique_classes[sort_indices]

    _, winner_indices = np.unique(sorted_elements[::-1], return_index=True)
    winner_indices = (len(sorted_elements) - 1) - winner_indices

    return sorted_elements[winner_indices], sorted_classes[winner_indices]


def _merge_update_rects(existing_rect, new_rect):
    """Return the union of two update rects in (x1, y1, x2, y2) form."""
    if new_rect is None:
        return existing_rect
    if existing_rect is None:
        return new_rect

    return (
        min(existing_rect[0], new_rect[0]),
        min(existing_rect[1], new_rect[1]),
        max(existing_rect[2], new_rect[2]),
        max(existing_rect[3], new_rect[3]),
    )


# -------------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------------


class PropagationEngine(QObject):
    """Handles all mask propagation, multi-annotate strokes, and projection logic.

    Owns:
        multi_annotate_enabled, _propagating_annotation,
        _propagation_buffer_pool, _pending_unified_propagation_jobs,
        _propagation_executor, _unified_bg_executor

    Everything else is delegated to the owning MVATManager via __getattr__.
    """

    _universal_repaint_signal = pyqtSignal(list)

    def __init__(self, manager: 'MVATManager'):
        super().__init__()
        self.manager = manager

        # Own propagation state
        self.multi_annotate_enabled = False
        self._propagating_annotation = False
        self._pending_unified_propagation_jobs = 0
        # Busy state for the async multi-annotate semantic-prediction path.
        self._semantic_propagation_busy = False
        self._semantic_propagation_done_msg = None
        self._propagation_buffer_pool = {}

        # Thread pools for parallel propagation
        self._propagation_executor = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4),
            thread_name_prefix='mvat_propagate'
        )
        self._unified_bg_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='mvat_unified_bg'
        )

        # Tracks the class_id → label_id mapping for labels currently painted on
        # the mesh.  Updated whenever _do_universal_propagation writes 3D faces,
        # and by aggregate_camera_masks_to_mesh.  Used by
        # project_mesh_labels_to_cameras to translate mesh class IDs back to
        # label UUIDs before writing target camera masks.
        self._mesh_class_label_ids: dict = {}

        # Queued connection so repaint tasks always run on the main thread
        self._universal_repaint_signal.connect(
            self._on_universal_repaint, Qt.QueuedConnection
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to the owning MVATManager.

        This lets every propagation method use plain ``self.cameras``,
        ``self.viewer``, ``self.raster_manager`` etc. without any rewrite —
        the lookup simply falls through to the manager when the attribute is
        not defined directly on PropagationEngine.
        """
        try:
            return getattr(object.__getattribute__(self, 'manager'), name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def _on_multi_annotate_toggled(self, enabled: bool):
        """Connect or disconnect annotation propagation handlers when toggle changes."""
        self.multi_annotate_enabled = enabled
        brush_tool = self.annotation_window.tools.get('brush')
        patch_tool = self.annotation_window.tools.get('patch')
        sam_tool = self.annotation_window.tools.get('sam')
        fill_tool = self.annotation_window.tools.get('fill')
        erase_tool = self.annotation_window.tools.get('erase')

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if enabled:
            self.annotation_window.annotationCreated.connect(self._on_patch_annotation_created)
            if brush_tool is not None:
                brush_tool.post_stroke_callback = self._on_brush_stroke_applied
                brush_tool.cursor_move_callback = self._on_cursor_preview_moved
                brush_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if patch_tool is not None:
                patch_tool.cursor_move_callback = self._on_cursor_preview_moved
                patch_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if fill_tool is not None:
                fill_tool.post_stroke_callback = self._on_fill_stroke_applied
                fill_tool.cursor_move_callback = self._on_cursor_preview_moved
                fill_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if erase_tool is not None:
                erase_tool.post_stroke_callback = self._on_erase_stroke_applied
                erase_tool.cursor_move_callback = self._on_cursor_preview_moved
                erase_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if sam_tool is not None:
                # Final-mask propagation callback (no live-hover propagation for now)
                sam_tool.post_prediction_callback = self._on_sam_prediction_applied
            # Proactively compute visibility/index maps for visible context cameras
            # so True 3D mapping will be available when the user paints or applies SAM.
            try:
                visible = list(self._get_visible_context_paths())
                target_paths = set(visible)
                if self.ortho_camera is not None and not self._is_ortho_annotation_source():
                    target_paths.add(self.ortho_camera.image_path)
                if visible and self.compute_index_maps_enabled:
                    self.main_window.status_bar.showMessage("Preparing context visibility maps...", 2000)
                    # Ask the visibility system to compute index maps for these visible cameras
                    # _update_visibility_filter handles cache checks and async worker dispatch.
                    self._update_visibility_filter(visible)

                # --- Force Mask Canvas Allocation NOW ---
                # Don't wait for the first brush stroke to allocate canvases!
                project_labels = list(self.main_window.label_window.labels)
                for path in target_paths:
                    raster = self.raster_manager.get_raster(path)
                    if raster and raster.mask_annotation is None:
                        raster.get_mask_annotation(project_labels)

            except Exception:
                pass
        else:
            try:
                self.annotation_window.annotationCreated.disconnect(self._on_patch_annotation_created)
            except TypeError:
                pass

            if brush_tool is not None:
                brush_tool.post_stroke_callback = None
                brush_tool.cursor_move_callback = None
                brush_tool.cursor_clear_callback = None
            if patch_tool is not None:
                patch_tool.cursor_move_callback = None
                patch_tool.cursor_clear_callback = None
            if fill_tool is not None:
                fill_tool.post_stroke_callback = None
                fill_tool.cursor_move_callback = None
                fill_tool.cursor_clear_callback = None
            if erase_tool is not None:
                erase_tool.post_stroke_callback = None
                erase_tool.cursor_move_callback = None
                erase_tool.cursor_clear_callback = None
            if sam_tool is not None:
                sam_tool.post_prediction_callback = None
            self._on_cursor_preview_cleared()

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def _on_cursor_preview_moved(self, scene_pos, item_factory):
        """Project the cursor position into visible context cameras and show previews.

        When on an OrthoCamera (orthomosaic), projects the cursor into all visible
        perspective cameras. When on a perspective camera, projects into all visible
        context cameras.

        Uses the blazingly fast center-point projection to display brush previews
        in all visible context cameras. The tool factory already draws the correct
        brush size visually; we just need to tell it where the center is.
        """
        if self.selected_camera is None or self.context_matrix is None:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())

        # Skip if the cursor hasn't moved by at least 1 pixel since the last update —
        # avoids redundant ray casts and camera projections during tiny jitter.
        last = getattr(self, '_last_cursor_preview_px', None)
        if last is not None and last == (px, py):
            return
        self._last_cursor_preview_px = (px, py)

        # Determine which cameras should show previews
        visible_paths = self._get_annotation_target_paths()

        # Build a camera subset limited to only the visible context canvases —
        # no point projecting into cameras that won't display a preview.
        visible_cameras = {p: c for p, c in self.cameras.items() if p in visible_paths}
        if self.ortho_camera is not None and self.ortho_camera.image_path in visible_paths:
            visible_cameras[self.ortho_camera.image_path] = self.ortho_camera

        projections = self._build_projection(px, py, target_cameras=visible_cameras)
        self.context_matrix.update_cursor_previews(projections, visible_paths, item_factory)

    def _on_cursor_preview_cleared(self):
        """Clear cursor previews from all context canvases."""
        self._last_cursor_preview_px = None
        if self.context_matrix is not None:
            self.context_matrix.clear_all_cursor_previews()

    def _get_context_canvas_for_path(self, image_path: str):
        """Return the context canvas currently displaying image_path, or None."""
        if self.context_matrix is None:
            return None
        for canvas in self.context_matrix._visible_canvases:
            if canvas is not None and canvas.current_image_path == image_path:
                return canvas
        return None

    def _compute_dirty_rect_from_flat_indices(self, flat_indices, width: int, height: int, padding: int = 1):
        """Return an x/y dirty rectangle for a flat index set, or None if empty."""
        if flat_indices is None or width <= 0 or height <= 0:
            return None

        flat_indices = np.asarray(flat_indices, dtype=np.int64).ravel()
        if flat_indices.size == 0:
            return None

        y_coords, x_coords = np.divmod(flat_indices, width)
        return (
            max(0, int(x_coords.min()) - padding),
            max(0, int(y_coords.min()) - padding),
            min(width, int(x_coords.max()) + padding + 1),
            min(height, int(y_coords.max()) + padding + 1),
        )

    def _acquire_propagation_buffer(self, shape, dtype=np.uint8):
        """Return a reusable NumPy buffer for background propagation work."""
        key = (tuple(shape), np.dtype(dtype).str)
        pool = self._propagation_buffer_pool.get(key)
        if pool:
            return pool.pop()
        return np.empty(shape, dtype=dtype)

    def _release_propagation_buffer(self, buffer):
        """Return a temporary propagation buffer to the local pool."""
        if buffer is None:
            return
        key = (tuple(buffer.shape), np.dtype(buffer.dtype).str)
        self._propagation_buffer_pool.setdefault(key, []).append(buffer)

    def _apply_mask_visual_update(self, target_path: str, target_mask, label_id: Optional[str] = None, update_rect=None):
        """Apply the minimal UI refresh needed after a silent mask write."""
        if target_mask is None:
            return

        if label_id is not None and label_id not in target_mask.visible_label_ids:
            target_mask.visible_label_ids.add(label_id)

        # Wire the overlay FIRST so update_graphics_item has a scene to paint into.
        context_canvas = self._get_context_canvas_for_path(target_path)
        if context_canvas is not None and context_canvas._mask_overlay_item is None:
            try:
                context_canvas.set_mask_overlay(target_mask)
            except Exception:
                pass

        try:
            target_mask.update_graphics_item(update_rect=update_rect)
        except Exception:
            pass

    def _get_visible_context_camera_paths(self) -> list:
        """Return the ordered list of image paths currently visible in the context matrix."""
        if self.context_matrix is None:
            return []
        if hasattr(self.context_matrix, 'get_visible_camera_paths'):
            try:
                return list(self.context_matrix.get_visible_camera_paths())
            except Exception:
                pass

        visible_paths = []
        for canvas in self.context_matrix._visible_canvases:
            if canvas and canvas.active_image and canvas.current_image_path:
                visible_paths.append(canvas.current_image_path)
        return visible_paths

    def _get_visible_context_cameras(self) -> list:
        """Return the Camera objects currently visible in the context matrix."""
        return [self.cameras[path] for path in self._get_visible_context_camera_paths() if path in self.cameras]

    def _get_visible_context_target_paths(self) -> set:
        """Return visible context camera paths excluding the active annotation camera."""
        paths = set(self._get_visible_context_camera_paths())
        if self.selected_camera and self.selected_camera.image_path in paths:
            paths.discard(self.selected_camera.image_path)
        return paths

    def _get_semantic_target_paths(self, source_camera) -> set:
        """Return source-aware target paths for semantic prediction propagation."""
        if source_camera is None:
            return set()

        target_paths = set(self._get_visible_context_camera_paths())
        target_paths.discard(source_camera.image_path)

        if self.ortho_camera is not None and source_camera is not self.ortho_camera:
            target_paths.add(self.ortho_camera.image_path)

        return target_paths

    def _warn_semantic_propagation(self, message: str):
        """Show a short warning when semantic propagation cannot run."""
        print(f"⚠️ Semantic propagation skipped: {message}")

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage(message, 5000)
                return
            except Exception:
                pass

        try:
            QMessageBox.warning(self.main_window, "Semantic Propagation", message)
        except Exception:
            pass

    def propagate_current_semantic_mask(self):
        """Propagate the active AnnotationWindow semantic mask to MVAT targets."""
        annotation_window = getattr(self.main_window, 'annotation_window', None)
        if annotation_window is None:
            self._warn_semantic_propagation("AnnotationWindow is not available.")
            return

        image_path = getattr(annotation_window, 'current_image_path', None)
        if not image_path:
            self._warn_semantic_propagation("No image is currently active in the AnnotationWindow.")
            return

        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            self._warn_semantic_propagation("The active image is not loaded in MVAT.")
            return

        target_paths = self._get_semantic_target_paths(source_camera)
        if not target_paths:
            self._warn_semantic_propagation("No target cameras are currently visible for semantic propagation.")
            return

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage(
                    f"Propagating semantic mask to {len(target_paths)} target camera(s)...",
                    0,
                )
            except Exception:
                pass

        source_raster = self.raster_manager.get_raster(image_path) if self.raster_manager is not None else None
        source_mask = getattr(source_raster, 'mask_annotation', None)
        if source_mask is None:
            self._warn_semantic_propagation("The active image does not have a semantic mask to propagate.")
            return

        label_window = getattr(self.main_window, 'label_window', None)
        project_labels = list(getattr(label_window, 'labels', [])) if label_window is not None else []
        if not project_labels:
            self._warn_semantic_propagation("No project labels are available for semantic propagation.")
            return

        try:
            source_mask.sync_label_map(project_labels)
        except Exception:
            pass

        try:
            source_mask.update_graphics_item()
        except Exception:
            pass

        mask_data = getattr(source_mask, 'mask_data', None)
        if mask_data is None:
            self._warn_semantic_propagation("The active semantic mask is missing mask data.")
            return

        lock_bit = getattr(source_mask, 'LOCK_BIT', None)
        try:
            if lock_bit is not None and int(lock_bit) > 1:
                semantic_values = np.unique(mask_data % int(lock_bit))
            else:
                semantic_values = np.unique(mask_data)
        except Exception:
            semantic_values = np.unique(mask_data)

        semantic_values = semantic_values[semantic_values > 0]
        if len(semantic_values) == 0:
            self._warn_semantic_propagation("The active semantic mask does not contain any labels to propagate.")
            return

        if getattr(source_camera, '_raster', None) is None or getattr(source_camera._raster, 'index_map', None) is None:
            self._warn_semantic_propagation(
                "The active camera does not have an index map, so semantic propagation is unavailable."
            )
            return

        try:
            self._on_semantic_prediction_applied(image_path, source_mask)
        except Exception as exc:
            print(f"Error while propagating semantic mask from {image_path}: {exc}")
            traceback.print_exc()
            self._warn_semantic_propagation("Semantic propagation failed. See console for details.")
            return

        if status_bar is not None:
            try:
                status_bar.showMessage(
                    f"Semantic mask propagated to {len(target_paths)} target camera(s).",
                    3000,
                )
            except Exception:
                pass

    def _is_ortho_annotation_source(self) -> bool:
        """Return True when the active annotation source is the ortho view."""
        return self.ortho_camera is not None and self.selected_camera == self.ortho_camera

    def _get_annotation_target_paths(self) -> set:
        """Return the target camera paths for the current annotation source."""
        if self._is_ortho_annotation_source():
            return self._get_ortho_target_cameras()
        target_paths = self._get_visible_context_target_paths()
        if self.ortho_camera is not None:
            target_paths.add(self.ortho_camera.image_path)
        return target_paths

    def _extract_source_ids_from_crop_mask(self,
                                           source_camera,
                                           source_mask: np.ndarray,
                                           px: int,
                                           py: int) -> Optional[np.ndarray]:
        """Extract visible element IDs from a crop-centred binary mask."""
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        mask_h, mask_w = source_mask.shape
        scale_factor = getattr(raster, 'index_map_scale_factor', None)

        if scale_factor is not None and scale_factor != 1.0:
            map_px = int(px * scale_factor)
            map_py = int(py * scale_factor)
            map_bw = max(1, int(mask_w * scale_factor))
            map_bh = max(1, int(mask_h * scale_factor))
        else:
            map_px, map_py = px, py
            map_bw, map_bh = mask_w, mask_h

        x0 = map_px - map_bw // 2
        y0 = map_py - map_bh // 2
        x1 = x0 + map_bw
        y1 = y0 + map_bh

        img_h, img_w = source_index_map.shape
        if x0 >= img_w or y0 >= img_h or x1 <= 0 or y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x1, img_w)
        cy1 = min(y1, img_h)
        index_slice = source_index_map[cy0:cy1, cx0:cx1]

        if scale_factor is not None and scale_factor != 1.0:
            raw_ids = index_slice.ravel()
        else:
            bx0 = cx0 - x0
            by0 = cy0 - y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = source_mask[by0:by1, bx0:bx1]
            raw_ids = index_slice[mask_clip.astype(bool)]

        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_source_ids_from_full_mask(self,
                                           source_camera,
                                           source_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract visible element IDs from a full-frame binary mask."""
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        needs_resize = (
            source_mask.shape != source_index_map.shape or
            (getattr(raster, 'index_map_scale_factor', None) not in (None, 1.0))
        )

        if needs_resize:
            import cv2
            mask_bool = cv2.resize(
                source_mask.astype(np.uint8),
                (source_index_map.shape[1], source_index_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_bool = source_mask.astype(bool)

        if not np.any(mask_bool):
            return np.array([], dtype=np.int64)

        raw_ids = source_index_map[mask_bool]
        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_source_element_ids_from_full_mask(self,
                                                   source_camera,
                                                   source_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract raw visible element IDs from a full-frame binary mask.

        Unlike _extract_source_ids_from_full_mask, this preserves duplicates so
        callers can compute per-element class votes before collapsing to one
        class per element.
        """
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        needs_resize = (
            source_mask.shape != source_index_map.shape or
            (getattr(raster, 'index_map_scale_factor', None) not in (None, 1.0))
        )

        if needs_resize:
            import cv2
            mask_bool = cv2.resize(
                source_mask.astype(np.uint8),
                (source_index_map.shape[1], source_index_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_bool = source_mask.astype(bool)

        if not np.any(mask_bool):
            return np.array([], dtype=np.int64)

        raw_ids = source_index_map[mask_bool]
        return raw_ids[raw_ids > -1].astype(np.int64, copy=False)

    def _extract_source_element_ids_from_region(self,
                                                 source_camera,
                                                 source_mask: np.ndarray,
                                                 top_left) -> Optional[np.ndarray]:
        """Extract raw visible element IDs from a partial mask region.

        This mirrors the full-mask helper, but only samples the work-area tile
        (or other region payload) so semantic propagation does not leak stale
        pixels from untouched parts of the image.
        """
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        x, y = top_left
        mask_h, mask_w = source_mask.shape
        scale_factor = getattr(raster, 'index_map_scale_factor', None)

        if scale_factor is not None and scale_factor != 1.0:
            map_x0 = int(x * scale_factor)
            map_y0 = int(y * scale_factor)
            map_w = max(1, int(mask_w * scale_factor))
            map_h = max(1, int(mask_h * scale_factor))
        else:
            map_x0 = int(x)
            map_y0 = int(y)
            map_w = mask_w
            map_h = mask_h

        map_x1 = map_x0 + map_w
        map_y1 = map_y0 + map_h

        img_h, img_w = source_index_map.shape
        if map_x0 >= img_w or map_y0 >= img_h or map_x1 <= 0 or map_y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(map_x0, 0)
        cy0 = max(map_y0, 0)
        cx1 = min(map_x1, img_w)
        cy1 = min(map_y1, img_h)
        index_slice = source_index_map[cy0:cy1, cx0:cx1]

        if index_slice.size == 0:
            return np.array([], dtype=np.int64)

        if scale_factor is not None and scale_factor != 1.0:
            import cv2

            mask_resized = cv2.resize(
                source_mask.astype(np.uint8),
                (map_w, map_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            bx0 = cx0 - map_x0
            by0 = cy0 - map_y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = mask_resized[by0:by1, bx0:bx1]
        else:
            bx0 = cx0 - map_x0
            by0 = cy0 - map_y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = source_mask[by0:by1, bx0:bx1].astype(bool)

        if mask_clip.size == 0 or not np.any(mask_clip):
            return np.array([], dtype=np.int64)

        raw_ids = index_slice[mask_clip]
        return raw_ids[raw_ids > -1].astype(np.int64, copy=False)

    def _get_visible_context_paths(self) -> set:
        """Return the set of image paths currently visible in the context matrix."""
        return set(self._get_visible_context_camera_paths())

    def _get_ortho_target_cameras(self) -> set:
        """Get target camera paths when painting on OrthoCamera.

        Returns the set of visible perspective camera paths that should receive
        multi-annotations when painting on an orthomosaic. This is the same as
        the visible context target paths, since we want to paint all visible cameras
        that can see the orthomosaic's geometry.
        """
        if self.ortho_camera is None:
            return set()
        # When on orthomosaic, propagate to all visible context cameras
        # (the ContextMatrix handles filtering for which cameras are viewable)
        return self._get_visible_context_target_paths()

    def _get_camera_for_path(self, image_path: str):
        """Return the loaded camera object for a path, including the orthocamera."""
        if self.ortho_camera is not None and image_path == self.ortho_camera.image_path:
            return self.ortho_camera
        return self.cameras.get(image_path)

    def _is_ortho_path(self, image_path: str) -> bool:
        return self.ortho_camera is not None and image_path == self.ortho_camera.image_path

    def _build_projection(self, px: int, py: int, source_camera=None, target_cameras=None) -> dict:
        """Cast a ray from the selected camera at (px, py) and return projections.

        Handles both perspective cameras and OrthoCamera (orthomosaic):
        - For perspective cameras: uses existing ray-projection logic
        - For OrthoCamera: converts orthomosaic pixel → geo → world space

        Returns:
            dict mapping image_path -> (u, v, is_valid), or empty dict on failure.
        """
        camera = source_camera if source_camera is not None else self.selected_camera
        if camera is None:
            return {}

        primary_target = self.viewer.scene_context.get_primary_target()
        ray = None

        # Special handling for OrthoCamera (orthomosaic)
        if self.ortho_camera is not None and camera == self.ortho_camera:
            try:
                if not camera.is_valid:
                    return {}

                # Convert orthomosaic pixel → geo → world space
                X, Y = camera.pixel_to_geo(px, py)
                Z = camera._raster.get_z_value(px, py)
                if Z is None or np.isnan(Z):
                    Z = 0.0

                world_pt = camera.geo_to_world(X, Y, Z)

                # Get element ID if index_map is available
                element_id = None
                index_map = camera._raster.index_map
                if index_map is not None and 0 <= px < camera.width and 0 <= py < camera.height:
                    sf = getattr(camera._raster, 'index_map_scale_factor', None)
                    map_px = int(px * sf) if sf else px
                    map_py = int(py * sf) if sf else py
                    map_px = min(map_px, index_map.shape[1] - 1)
                    map_py = min(map_py, index_map.shape[0] - 1)
                    candidate_id = int(index_map[map_py, map_px])
                    if candidate_id > -1:
                        element_id = candidate_id

                # Construct a vertical ray from the orthomosaic
                # The ray origin is slightly above the world point, direction points down
                vertical_dir = camera.get_vertical_direction_world()
                ray_origin = world_pt - vertical_dir * 0.1  # Slightly above the surface
                ray_direction = vertical_dir

                ray = CameraRay(
                    origin=ray_origin,
                    direction=ray_direction,
                    terminal_point=world_pt,
                    has_accurate_depth=True,
                    pixel_coord=(px, py),
                    source_camera=camera,
                    element_id=element_id
                )
            except Exception as e:
                print(f"Error building ortho projection: {e}")
                return {}
        else:
            # Standard perspective camera logic
            # --- PLAN A: Index Map (Flawless 3D Coordinate) ---
            index_map = camera._raster.index_map
            if index_map is not None and primary_target is not None:
                try:
                    # Ensure we are inside the image bounds
                    if 0 <= px < camera.width and 0 <= py < camera.height:
                        candidate_id = int(index_map[py, px])
                        if candidate_id > -1:
                            coord = primary_target.get_element_coordinate(candidate_id)
                            if coord is not None:
                                origin = camera.position.copy()
                                direction = coord - origin
                                norm = np.linalg.norm(direction)
                                direction = direction / norm if norm > 0 else camera.R.T @ np.array([0, 0, 1])

                                ray = CameraRay(
                                    origin=origin,
                                    direction=direction,
                                    terminal_point=coord,
                                    has_accurate_depth=True,
                                    pixel_coord=(px, py),
                                    source_camera=camera,
                                    element_id=candidate_id
                                )
                except Exception:
                    pass

            # --- PLAN B: Depth Map / Scene Median Fallback ---
            if ray is None:
                depth = None
                try:
                    raster = camera._raster
                    if raster.z_channel is not None and raster.z_data_type == 'depth':
                        depth = raster.get_z_value(px, py)
                except Exception:
                    pass

                # Cache median depth per camera — only recompute when active camera changes
                cache_key = id(camera)
                if getattr(self, '_median_depth_cache_key', None) != cache_key:
                    try:
                        self._cached_median_depth = self.viewer.get_scene_median_depth(camera.position)
                    except Exception:
                        self._cached_median_depth = 10.0
                    self._median_depth_cache_key = cache_key

                default_depth = self._cached_median_depth or 10.0

                try:
                    ray = CameraRay.from_pixel_and_camera(
                        pixel_xy=(px, py),
                        camera=camera,
                        depth=depth,
                        default_depth=default_depth,
                    )
                except Exception:
                    return {}

        if ray is None:
            return {}

        if target_cameras is not None:
            cameras_for_projection = target_cameras
        else:
            cameras_for_projection = self.cameras
            if self.ortho_camera is not None and camera != self.ortho_camera:
                cameras_for_projection = dict(self.cameras)
                cameras_for_projection[self.ortho_camera.image_path] = self.ortho_camera

        return ray.project_to_cameras(cameras_for_projection)

    def _on_patch_annotation_created(self, annotation_id: str):
        """Propagate a newly created PatchAnnotation into all target cameras (perspective and ortho-aware)."""
        if self._propagating_annotation:
            return

        annotation = self.annotation_window.annotations_dict.get(annotation_id)
        if annotation is None or not isinstance(annotation, PatchAnnotation):
            return
        if self.selected_camera is None:
            return
        if annotation.image_path != self.selected_camera.image_path:
            return

        px = int(annotation.center_xy.x())
        py = int(annotation.center_xy.y())

        selected_paths = self._get_annotation_target_paths()

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        from PyQt5.QtCore import QPointF
        self._propagating_annotation = True
        try:
            # ------------------------------------------------------------------
            # Source element-ID extraction: sample a sparse grid within the
            # annotation bounding box so that get_pixels_for_elements has many
            # IDs to work with — not just the single center pixel.  More IDs
            # dramatically reduces stride false-negatives in the target cameras.
            # ------------------------------------------------------------------
            source_raster = getattr(self.selected_camera, '_raster', None)
            source_index_map = source_raster.index_map if source_raster is not None else None
            source_element_ids = None   # list[int] — passed to get_pixels_for_elements
            element_id = None           # center-pixel ID — used by _build_projection ray
            use_3d = False

            if source_index_map is not None:
                try:
                    sf = getattr(source_raster, 'index_map_scale_factor', None) or 1.0
                    img_h, img_w = source_index_map.shape
                    ann_size = annotation.annotation_size   # half-extent in image pixels

                    # Clamp the annotation bounding box to the index-map bounds
                    # (coordinates scaled by sf to match the index-map resolution).
                    x0 = max(0,       int((px - ann_size) * sf))
                    x1 = min(img_w,   int((px + ann_size) * sf) + 1)
                    y0 = max(0,       int((py - ann_size) * sf))
                    y1 = min(img_h,   int((py + ann_size) * sf) + 1)

                    if x0 < x1 and y0 < y1:
                        patch = source_index_map[y0:y1, x0:x1].ravel()
                        valid = patch[patch > -1]
                        if valid.size > 0:
                            source_element_ids = list(np.unique(valid).tolist())
                            # Prefer the exact center-pixel ID for the ray direction
                            cx = min(int(px * sf), img_w - 1)
                            cy = min(int(py * sf), img_h - 1)
                            center_eid = int(source_index_map[cy, cx])
                            element_id = center_eid if center_eid > -1 else source_element_ids[0]
                            use_3d = True
                except Exception:
                    pass

            # Lazy projection cache for fallback
            projections = None

            for target_path in selected_paths:

                target_camera = self._get_camera_for_path(target_path)
                if target_camera is None:
                    continue

                try:
                    placed = False

                    # ----------------------------------------------------------
                    # 3D centroid path: look up every sampled element ID in the
                    # target's index map and use the resulting pixel centroid.
                    # Falls through to 2D when the lookup returns empty (element
                    # too small / edge-on in target, or stride miss) rather than
                    # hard-skipping the camera.
                    # ----------------------------------------------------------
                    target_has_index = (getattr(target_camera, '_raster', None) is not None
                                        and target_camera._raster.index_map is not None)
                    if use_3d and target_has_index and source_element_ids:
                        flat = target_camera.get_pixels_for_elements(
                            np.array(source_element_ids, dtype=np.int64)
                        )
                        if flat.size > 0:
                            v_arr, u_arr = np.divmod(flat, target_camera.width)
                            u_centroid = float(np.mean(u_arr))
                            v_centroid = float(np.mean(v_arr))
                            if 0 <= u_centroid < target_camera.width and 0 <= v_centroid < target_camera.height:
                                new_annotation = PatchAnnotation(
                                    center_xy=QPointF(u_centroid, v_centroid),
                                    annotation_size=annotation.annotation_size,
                                    label=annotation.label,
                                    image_path=target_path,
                                    transparency=annotation.transparency,
                                )
                                try:
                                    self.annotation_window.add_annotation(new_annotation, record_action=True)
                                    placed = True
                                except Exception:
                                    pass

                    # ----------------------------------------------------------
                    # 2D fallback: used when no index map is available OR when
                    # the 3D lookup returned empty (element occluded / missed).
                    # ----------------------------------------------------------
                    if not placed:
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue

                        new_annotation = PatchAnnotation(
                            center_xy=QPointF(u, v),
                            annotation_size=annotation.annotation_size,
                            label=annotation.label,
                            image_path=target_path,
                            transparency=annotation.transparency,
                        )
                        try:
                            self.annotation_window.add_annotation(new_annotation, record_action=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    def _dense_mesh_hit_test(self, source_camera, pixel_mask: np.ndarray, px: int, py: int, mesh_product) -> np.ndarray:
        """Cast rays through every True pixel in pixel_mask against the mesh surface.

        Unlike the index_map approach (which captures face IDs from a downsampled
        rasterization pass), this method casts a ray through every individual painted
        pixel at full resolution, intersecting the actual triangle surface.  This
        guarantees that every triangle touched by the brush or SAM mask contributes
        its face ID to the output set, regardless of its projected pixel size.

        Uses PyVista / VTK multi_ray_trace with a per-product cached triangulated
        surface so the geometry is only prepared once per session.

        Args:
            source_camera: Perspective Camera for the selected image.
            pixel_mask: (H, W) bool/uint8 array; True pixels are ray-cast targets.
            px: X coordinate of the mask centre in source image space.
            py: Y coordinate of the mask centre in source image space.
            mesh_product: MeshProduct whose cached geometry is used.

        Returns:
            np.ndarray[int32]: Unique face IDs that were hit, or empty array on
            failure or orthographic source camera.
        """
        try:
            # 1. Obtain (or build) the cached triangulated surface for VTK ray-casting.
            #    Mesh topology never changes during annotation, so the cached surface
            #    stays valid for the entire session.
            mesh_surf = getattr(mesh_product, '_vtk_raycasting_mesh', None)
            if mesh_surf is None:
                mesh_product.prepare_geometry()
                mesh_pv = mesh_product.get_mesh()
                if mesh_pv is None or mesh_pv.n_cells == 0:
                    return np.array([], dtype=np.int32)
                mesh_surf = mesh_pv.triangulate() if not mesh_pv.is_all_triangles else mesh_pv
                mesh_product._vtk_raycasting_mesh = mesh_surf

            # 2. Map True pixels to source-image coordinates.
            mask_h, mask_w = pixel_mask.shape
            x0 = px - mask_w // 2
            y0 = py - mask_h // 2

            ys, xs = np.where(pixel_mask.astype(bool))
            if len(xs) == 0:
                return np.array([], dtype=np.int32)

            u_img = (xs + x0).astype(np.float32)
            v_img = (ys + y0).astype(np.float32)

            # Discard pixels outside the image frame.
            valid = (
                (u_img >= 0) & (u_img < source_camera.width) &
                (v_img >= 0) & (v_img < source_camera.height)
            )
            u_img = u_img[valid]
            v_img = v_img[valid]
            if len(u_img) == 0:
                return np.array([], dtype=np.int32)

            # 3. Unproject pixels to world-space ray directions.
            #    Pinhole camera model (row-vector convention):
            #      d_cam   = K_inv @ [u, v, 1]^T   →   d_cam_row = [u,v,1] @ K_inv.T
            #      d_world = R.T   @ d_cam          →   d_world_row = d_cam_row @ R
            ones        = np.ones(len(u_img), dtype=np.float32)
            pixel_homog = np.stack([u_img, v_img, ones], axis=1)     # (N, 3)
            K_inv       = source_camera.K_inv.astype(np.float32)     # (3, 3)
            R           = source_camera.R.astype(np.float32)         # (3, 3)

            dirs_cam   = pixel_homog @ K_inv.T    # (N, 3) camera-space directions
            dirs_world = dirs_cam   @ R            # (N, 3) world-space directions

            norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            dirs_world /= norms

            # 4. Build origins (all from the camera position) and cast rays.
            cam_origin = source_camera.position.astype(np.float32)   # (3,)
            origins    = np.tile(cam_origin, (len(u_img), 1))        # (N, 3)

            # PyVista multi_ray_trace returns (points, ray_indices, cell_ids).
            # We only need cell_ids (the hit triangle IDs in the triangulated mesh).
            _, _, intersection_cells = mesh_surf.multi_ray_trace(
                origins, dirs_world, first_point=True, retry=False
            )

            if len(intersection_cells) == 0:
                return np.array([], dtype=np.int32)

            hit_prim_ids = np.asarray(intersection_cells, dtype=np.int64)

            # 5. Remap triangulated-face IDs to original PyVista cell IDs when the
            #    mesh was triangulated from non-triangular faces during prepare_geometry().
            original_cell_ids = getattr(mesh_product, '_original_cell_ids', None)
            if original_cell_ids is not None:
                in_range     = hit_prim_ids < len(original_cell_ids)
                hit_prim_ids = hit_prim_ids[in_range]
                face_ids     = original_cell_ids[hit_prim_ids].astype(np.int32)
            else:
                face_ids = hit_prim_ids.astype(np.int32)

            return np.unique(face_ids)

        except Exception as e:
            print(f"⚠️ Dense mesh hit test failed: {e}")
            return np.array([], dtype=np.int32)

    def _propagate_to_camera(self, target_path, painted_ids, target_class_id_map,
                              projections, brush_w, brush_h, brush_mask, use_3d):
        """Single-camera propagation — runs in thread pool, no Qt calls."""
        target_camera = self._get_camera_for_path(target_path)
        if target_camera is None:
            return target_path, False, None

        target_raster = self.raster_manager.get_raster(target_path)
        if target_raster is None:
            return target_path, False, None

        target_mask = target_raster.mask_annotation
        if target_mask is None:
            return target_path, False, None

        target_class_id = target_class_id_map.get(target_path)
        if target_class_id is None:
            return target_path, False, None

        target_has_index = target_camera._raster.index_map is not None

        if use_3d and target_has_index and target_camera is not self.ortho_camera:
            proj = projections.get(target_path)
            bbox = None
            if proj is not None and proj[2]:
                target_u, target_v = proj[0], proj[1]
                search_radius = max(brush_w, brush_h) * 2.5
                bbox = (target_u - search_radius, target_u + search_radius,
                        target_v - search_radius, target_v + search_radius)

            flat_indices = target_camera.get_pixels_for_elements(painted_ids, bbox=bbox)
            if len(flat_indices) == 0:
                return target_path, False, None

            if hasattr(target_mask, 'mask_data'):
                current_vals = target_mask.mask_data.ravel()[flat_indices]
                flat_indices = flat_indices[(current_vals < target_mask.LOCK_BIT) &
                                            (current_vals != target_class_id)]
            if len(flat_indices) == 0:
                return target_path, False, None

            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
            update_rect = self._compute_dirty_rect_from_flat_indices(
                flat_indices,
                target_camera.width,
                target_camera.height,
            )
        else:
            proj = projections.get(target_path)
            if proj is None:
                return target_path, False, None
            u, v, is_valid = proj
            if not is_valid or not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                return target_path, False, None
            brush_location = QPointF(u - brush_w / 2.0, v - brush_h / 2.0)
            target_mask.update_mask(brush_location, brush_mask, target_class_id, silent=True)
            x_start = max(0, int(u - brush_w / 2.0))
            y_start = max(0, int(v - brush_h / 2.0))
            update_rect = (
                x_start,
                y_start,
                min(target_camera.width, x_start + brush_w),
                min(target_camera.height, y_start + brush_h),
            )

        return target_path, True, update_rect

    def _resolve_source_mask_class_context(self, source_camera, label_id: str, project_labels: list):
        """Resolve the source label, mask, and internal class ID for propagation."""
        if source_camera is None:
            return None, None

        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return None, None

        source_raster = self.raster_manager.get_raster(source_camera.image_path)
        if source_raster is None:
            return source_label, None

        source_mask = source_raster.mask_annotation
        if source_mask is None:
            source_mask = source_raster.get_mask_annotation(project_labels)
        if source_mask is None:
            return source_label, None

        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
        if source_class_id is None:
            source_mask.sync_label_map([source_label])
            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)

        return source_label, source_class_id

    def _extract_source_ids_from_sam_prediction(self,
                                                source_camera,
                                                binary_mask: np.ndarray,
                                                px: int,
                                                py: int) -> Optional[np.ndarray]:
        """Extract source element IDs for a SAM prediction on the active image."""
        if source_camera is None or binary_mask is None:
            return None

        binary_mask = np.asarray(binary_mask)
        if binary_mask.ndim != 2 or not np.any(binary_mask):
            return np.array([], dtype=np.int64)

        if self.ortho_camera is not None and source_camera is self.ortho_camera:
            return self._extract_source_ids_from_crop_mask(
                source_camera,
                binary_mask.astype(bool),
                px,
                py,
            )

        raster = getattr(source_camera, '_raster', None)
        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        mask_h, mask_w = binary_mask.shape
        x0 = px - mask_w // 2
        y0 = py - mask_h // 2
        x1 = x0 + mask_w
        y1 = y0 + mask_h

        img_h, img_w = source_index_map.shape
        if x0 >= img_w or y0 >= img_h or x1 <= 0 or y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x1, img_w)
        cy1 = min(y1, img_h)

        bx0 = cx0 - x0
        by0 = cy0 - y0
        bx1 = bx0 + (cx1 - cx0)
        by1 = by0 + (cy1 - cy0)

        index_slice = source_index_map[cy0:cy1, cx0:cx1]
        mask_clip = binary_mask[by0:by1, bx0:bx1]
        valid_mask = mask_clip.astype(bool)
        if not np.any(valid_mask):
            return np.array([], dtype=np.int64)

        source_depth_map = getattr(raster, 'z_channel', None)
        if source_depth_map is not None:
            try:
                import cv2

                depth_slice = source_depth_map[cy0:cy1, cx0:cx1]
                erosion_r = int(np.clip(min(mask_clip.shape) * 0.03, 2, 12))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * erosion_r + 1, 2 * erosion_r + 1),
                )
                interior_mask = cv2.erode(
                    valid_mask.astype(np.uint8),
                    kernel,
                    iterations=1,
                ).astype(bool)
                perimeter_mask = valid_mask & ~interior_mask

                interior_depths = depth_slice[interior_mask]
                interior_depths = interior_depths[~np.isnan(interior_depths)]

                if len(interior_depths) >= 10 and perimeter_mask.any():
                    ref_depth = np.median(interior_depths)
                    interior_spread = np.std(interior_depths)
                    abs_floor = max(0.02, ref_depth * 0.005)
                    full_tol = interior_spread * 2.0 + abs_floor
                    dist = cv2.distanceTransform(valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    norm_dist = np.clip(dist / max(erosion_r, 1), 0.0, 1.0)
                    per_pixel_tol = abs_floor + (full_tol - abs_floor) * norm_dist
                    with np.errstate(invalid='ignore'):
                        perimeter_depth_ok = np.abs(depth_slice - ref_depth) <= per_pixel_tol
                    valid_mask = interior_mask | (perimeter_mask & perimeter_depth_ok)
            except Exception:
                pass

        raw_ids = index_slice[valid_mask]
        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_semantic_element_votes(self,
                                        source_camera,
                                        source_mask_annotation,
                                        prediction_regions=None):
        """Build raw element/class vote arrays for semantic propagation."""
        if source_camera is None or source_mask_annotation is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        raster = getattr(source_camera, '_raster', None)
        if getattr(raster, 'index_map', None) is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        lock_bit = source_mask_annotation.LOCK_BIT
        element_chunks = []
        class_chunks = []
        class_label_ids = {}

        def _append_votes(real_class_id, label, raw_element_ids):
            if label is None or raw_element_ids is None:
                return

            raw_element_ids = np.asarray(raw_element_ids, dtype=np.int64).ravel()
            raw_element_ids = raw_element_ids[raw_element_ids > -1]
            if raw_element_ids.size == 0:
                return

            real_class_id = int(real_class_id)
            class_label_ids[real_class_id] = label.id
            element_chunks.append(raw_element_ids)
            class_chunks.append(
                np.full(raw_element_ids.size, real_class_id, dtype=np.int64)
            )

        if prediction_regions is not None:
            for region_mask, top_left in prediction_regions:
                if region_mask is None:
                    continue

                region_mask = np.asarray(region_mask)
                if region_mask.ndim != 2:
                    continue

                unique_real_ids = np.unique(region_mask % lock_bit)
                unique_real_ids = unique_real_ids[unique_real_ids > 0]
                for real_class_id in unique_real_ids:
                    label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                    if label is None:
                        continue

                    binary_mask = (region_mask % lock_bit == real_class_id)
                    if not np.any(binary_mask):
                        continue

                    raw_element_ids = self._extract_source_element_ids_from_region(
                        source_camera,
                        binary_mask,
                        top_left,
                    )
                    _append_votes(real_class_id, label, raw_element_ids)
        else:
            semantic_mask = np.asarray(source_mask_annotation.mask_data)
            if semantic_mask.ndim != 2:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

            unique_real_ids = np.unique(semantic_mask % lock_bit)
            unique_real_ids = unique_real_ids[unique_real_ids > 0]
            for real_class_id in unique_real_ids:
                label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                if label is None:
                    continue

                binary_mask = (semantic_mask % lock_bit == real_class_id)
                if not np.any(binary_mask):
                    continue

                raw_element_ids = self._extract_source_element_ids_from_full_mask(
                    source_camera,
                    binary_mask,
                )
                _append_votes(real_class_id, label, raw_element_ids)

        if not element_chunks or not class_chunks:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        return (
            np.concatenate(element_chunks).astype(np.int64, copy=False),
            np.concatenate(class_chunks).astype(np.int64, copy=False),
            class_label_ids,
        )

    def _on_brush_stroke_applied(self, scene_pos, label_id: str, brush_mask):
        """Propagate a brush stroke into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the brush to all visible
        perspective cameras that can see the same 3D geometry.

        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract the element IDs painted under the brush using the source
           camera's index_map.
        2. Update the 3D Scene Product directly with the new Class ID and Color.
        3. Query each target camera's inverted index for the same IDs.
        4. Paint exactly those pixels with update_mask_at_indices().

        Falls back to the legacy 2D center-stamp when the index map is absent
        (e.g., visibility not yet computed) or when no scene geometry was hit.
        """
        if self.selected_camera is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        selected_paths = self._get_annotation_target_paths()

        project_labels = list(self.main_window.label_window.labels)

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        source_label, source_class_id = self._resolve_source_mask_class_context(
            self.selected_camera,
            label_id,
            project_labels,
        )
        if source_label is None or source_class_id is None:
            return

        painted_ids = self._extract_source_ids_from_crop_mask(
            self.selected_camera,
            brush_mask,
            px,
            py,
        )
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = np.full(painted_ids.size, int(source_class_id), dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id},
            fallback_payload={
                'mode': 'brush',
                'label_id': label_id,
                'source_class_id': int(source_class_id),
                'mask': np.asarray(brush_mask, dtype=bool),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(brush_mask.shape) * 2.5),
            },
        )

    def _on_fill_stroke_applied(self, scene_pos, label_id: str, fill_mask=None):
        """Propagate a fill operation into all visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        if not selected_paths:
            return

        project_labels = list(self.main_window.label_window.labels)
        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return

        source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
        source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
        source_class_id = None
        if source_mask is not None:
            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                source_mask.sync_label_map([source_label])
                source_class_id = source_mask.label_id_to_class_id_map.get(label_id)

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        painted_ids = None
        if fill_mask is not None:
            painted_ids = self._extract_source_ids_from_full_mask(self.selected_camera, fill_mask)

        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = (
            np.full(painted_ids.size, int(source_class_id), dtype=np.int64)
            if source_class_id is not None
            else np.array([], dtype=np.int64)
        )

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id} if source_class_id is not None else {},
            fallback_payload={
                'mode': 'fill',
                'label_id': label_id,
                'source_class_id': int(source_class_id) if source_class_id is not None else None,
                'mask': np.asarray(fill_mask, dtype=bool) if fill_mask is not None else None,
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(fill_mask.shape) * 2.5) if fill_mask is not None else 0.0,
            },
        )

    def _on_erase_stroke_applied(self, scene_pos, label_id: str, brush_mask: np.ndarray):
        """Propagate an erase operation into all visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        if not selected_paths:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())
        painted_ids = self._extract_source_ids_from_crop_mask(self.selected_camera, brush_mask, px, py)
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=np.zeros(painted_ids.size, dtype=np.int64),
            target_paths=selected_paths,
            project_labels=list(self.main_window.label_window.labels),
            class_label_ids={},
            fallback_payload={
                'mode': 'erase',
                'source_class_id': 0,
                'mask': np.asarray(brush_mask, dtype=bool),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(brush_mask.shape) * 2.5),
            },
        )

    def _propagate_3d_face_ids_to_context_cameras(self, face_ids, label, erase: bool = False):
        """Propagate a 3D brush/erase stroke into visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        annotation_window = getattr(self.main_window, 'annotation_window', None)
        primary_path = getattr(annotation_window, 'current_image_path', None)
        if primary_path:
            selected_paths.add(primary_path)
        if not selected_paths:
            return

        try:
            face_ids_arr = np.asarray(face_ids, dtype=np.int64)
            face_ids_arr = np.unique(face_ids_arr[face_ids_arr >= 0])
        except Exception:
            return

        if face_ids_arr.size == 0:
            return

        project_labels = list(self.main_window.label_window.labels)
        label_id = getattr(label, 'id', None)

        if erase:
            source_class_id = 0
        else:
            if label is None or label_id is None:
                return

            source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
            source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
            if source_mask is None:
                return

            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                source_mask.sync_label_map([label])
                source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                return

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=face_ids_arr,
            class_ids=np.full(face_ids_arr.size, int(source_class_id), dtype=np.int64),
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id} if int(source_class_id) != 0 else {},
            fallback_payload=None,
            skip_3d_paint=True,
        )

    def _on_3d_brush_stroke_applied(self, face_ids, label):
        self._propagate_3d_face_ids_to_context_cameras(face_ids, label, erase=False)

    def _on_3d_erase_stroke_applied(self, face_ids, label=None):
        self._propagate_3d_face_ids_to_context_cameras(face_ids, label, erase=True)

    def _on_sam_prediction_applied(self, scene_pos, label_id: str, binary_mask: np.ndarray):
        """Propagate a final SAM mask prediction into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the SAM prediction to all visible
        perspective cameras that can see the same 3D geometry.
        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract the element IDs beneath the predicted binary_mask using the
           source camera's index_map (the mask is a crop centred at scene_pos).
        2. Update the 3D Scene Product directly with the new Class ID and Color.
        3. Query each target camera's inverted index for the same IDs.
        4. Paint exactly those pixels with update_mask_at_indices().

        Falls back to the legacy 2D stamp when the index map is absent or when
        no scene geometry was hit (e.g., sky prediction).

        Args:
            scene_pos: QPointF — centre of the prediction crop in source image pixels.
            label_id: UUID of the label used for the prediction.
            binary_mask: small (H,W) uint8 array with 1 for predicted pixels.
        """
        if self.selected_camera is None:
            return

        selected_camera = self.selected_camera
        selected_paths  = self._get_annotation_target_paths()
        if not selected_paths:
            return

        project_labels  = list(self.main_window.label_window.labels)
        source_label, source_class_id = self._resolve_source_mask_class_context(
            selected_camera,
            label_id,
            project_labels,
        )
        if source_label is None or source_class_id is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())
        painted_ids = self._extract_source_ids_from_sam_prediction(
            selected_camera,
            binary_mask,
            px,
            py,
        )
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = np.full(painted_ids.size, int(source_class_id), dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id},
            fallback_payload={
                'mode': 'sam',
                'label_id': label_id,
                'source_class_id': int(source_class_id),
                'mask': np.asarray(binary_mask, dtype=np.uint8),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(binary_mask.shape) * 2.5),
            },
        )

    def _execute_mask_propagation(self,
                                  source_camera,
                                  element_ids: np.ndarray,
                                  class_ids: np.ndarray,
                                  target_paths: set,
                                  project_labels: list,
                                  class_label_ids: dict,
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Queue a propagation job onto the single unified background worker."""
        t0 = perf_counter()
        if source_camera is None or not target_paths:
            return

        try:
            element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        except Exception:
            element_ids = np.array([], dtype=np.int64)

        try:
            class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
        except Exception:
            class_ids = np.array([], dtype=np.int64)

        has_3d_payload = (
            element_ids.size > 0 and
            class_ids.size > 0 and
            element_ids.size == class_ids.size
        )
        if not has_3d_payload:
            element_ids = np.array([], dtype=np.int64)
            class_ids = np.array([], dtype=np.int64)

        if not has_3d_payload and fallback_payload is None:
            return

        target_paths = tuple(sorted({path for path in target_paths if path}))
        if not target_paths:
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        self._pending_unified_propagation_jobs += 1
        self._propagating_annotation = True

        try:
            payload = dict(fallback_payload) if isinstance(fallback_payload, dict) else fallback_payload
            self._unified_bg_executor.submit(
                self._do_universal_propagation,
                source_camera,
                element_ids.copy(),
                class_ids.copy(),
                target_paths,
                list(project_labels),
                dict(class_label_ids or {}),
                primary_target,
                payload,
                skip_3d_paint,
            )
        except Exception:
            self._pending_unified_propagation_jobs = max(
                0,
                self._pending_unified_propagation_jobs - 1,
            )
            self._propagating_annotation = self._pending_unified_propagation_jobs > 0
            traceback.print_exc()

    def _do_universal_propagation(self,
                                  source_camera,
                                  element_ids: np.ndarray,
                                  class_ids: np.ndarray,
                                  target_paths,
                                  project_labels,
                                  class_label_ids,
                                  primary_target,
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Background worker for brush, SAM, and semantic mask propagation."""
        from time import perf_counter
        t0 = perf_counter()
        repaint_tasks = []
        mask_time = 0.0

        try:
            labels_by_id = {
                getattr(label, 'id', None): label
                for label in project_labels
                if getattr(label, 'id', None) is not None
            }

            # Build canonical class_id space to ensure mesh is always painted with
            # consistent canonical IDs, regardless of the source (semantic prediction,
            # brush stroke, SAM, etc.). This prevents class_id misalignment when
            # projecting the mesh back to cameras.
            real_labels = [lbl for lbl in project_labels if getattr(lbl, 'id', None) and lbl.id != '-1']
            canonical_id_for = {lbl.id: (idx + 1) for idx, lbl in enumerate(real_labels)}

            winning_elements = np.array([], dtype=np.int64)
            winning_classes = np.array([], dtype=np.int64)
            if element_ids is not None and class_ids is not None:
                winning_elements, winning_classes = resolve_class_conflicts_vectorized(
                    element_ids,
                    class_ids,
                )

            class_to_elements = {}
            if winning_elements.size > 0 and winning_classes.size > 0:
                for source_class_id in np.unique(winning_classes):
                    class_to_elements[int(source_class_id)] = winning_elements[
                        winning_classes == source_class_id
                    ]

            if primary_target and hasattr(primary_target, 'apply_labels') and not skip_3d_paint:
                for source_class_id, subset_elements in class_to_elements.items():
                    if subset_elements.size == 0:
                        continue

                    label_id = class_label_ids.get(int(source_class_id))
                    if int(source_class_id) == 0:
                        repaint_tasks.append({
                            'type': '3d_paint',
                            'painted_ids': subset_elements.copy(),
                            'target_color': (255, 255, 255),
                            'source_class_id': 0,
                            'primary_target': primary_target,
                            'label_id': None,
                        })
                        continue

                    label = labels_by_id.get(label_id)
                    if label is None:
                        continue

                    # CRITICAL: Convert source_class_id (local to source camera) to
                    # canonical_class_id (universal across all cameras) before painting
                    # the mesh and storing in _mesh_class_label_ids. This ensures that
                    # when the mesh is later projected to other cameras, the class IDs
                    # are consistent and correct.
                    canonical_class_id = canonical_id_for.get(label_id)
                    if canonical_class_id is None:
                        # Fallback to source_class_id if canonical mapping fails
                        canonical_class_id = int(source_class_id)

                    # Keep the mesh class-label registry up to date so
                    # project_mesh_labels_to_cameras can resolve class IDs back
                    # to label UUIDs without any additional state.
                    if label_id is not None:
                        self._mesh_class_label_ids[canonical_class_id] = label_id

                    repaint_tasks.append({
                        'type': '3d_paint',
                        'painted_ids': subset_elements.copy(),
                        'target_color': (
                            label.color.red(),
                            label.color.green(),
                            label.color.blue(),
                        ),
                        'source_class_id': canonical_class_id,
                        'primary_target': primary_target,
                        'label_id': label_id,
                    })

            centers = getattr(primary_target, '_element_centers_np', None)

            fallback_mode = None
            fallback_mask = None
            fallback_center = None
            fallback_search_radius = 0.0
            fallback_skip_ortho_index_lookup = False
            fallback_label_id = None
            fallback_projections = {}

            if isinstance(fallback_payload, dict):
                fallback_mode = str(fallback_payload.get('mode', '')).strip().lower()
                if fallback_payload.get('mask') is not None:
                    fallback_mask = np.asarray(fallback_payload.get('mask'))
                fallback_center = fallback_payload.get('center')
                fallback_search_radius = float(fallback_payload.get('search_radius', 0.0) or 0.0)
                fallback_skip_ortho_index_lookup = bool(fallback_payload.get('skip_ortho_index_lookup', False))
                fallback_label_id = fallback_payload.get('label_id')
                fallback_projections = fallback_payload.get('projections', {}) or {}

            def _project_bbox_for_subset(target_camera, subset_elements):
                if centers is None or target_camera is None or target_camera is self.ortho_camera:
                    return None
                try:
                    subset_centers = np.asarray(centers[np.asarray(subset_elements, dtype=np.int64)], dtype=np.float64)
                    if subset_centers.size == 0:
                        return None

                    min_pt = np.min(subset_centers, axis=0)
                    max_pt = np.max(subset_centers, axis=0)
                    corners_3d = np.array([
                        [min_pt[0], min_pt[1], min_pt[2]],
                        [min_pt[0], min_pt[1], max_pt[2]],
                        [min_pt[0], max_pt[1], min_pt[2]],
                        [min_pt[0], max_pt[1], max_pt[2]],
                        [max_pt[0], min_pt[1], min_pt[2]],
                        [max_pt[0], min_pt[1], max_pt[2]],
                        [max_pt[0], max_pt[1], min_pt[2]],
                        [max_pt[0], max_pt[1], max_pt[2]],
                    ], dtype=np.float64)

                    projected = []
                    for corner in corners_3d:
                        uv = target_camera.project(corner)
                        if uv is None or np.isnan(uv).any():
                            continue
                        projected.append(uv)

                    if not projected:
                        return None

                    projected = np.asarray(projected, dtype=np.float64)
                    u_min = max(0, int(np.floor(np.min(projected[:, 0]))))
                    u_max = min(int(target_camera.width), int(np.ceil(np.max(projected[:, 0]))) + 1)
                    v_min = max(0, int(np.floor(np.min(projected[:, 1]))))
                    v_max = min(int(target_camera.height), int(np.ceil(np.max(projected[:, 1]))) + 1)
                    if u_min >= u_max or v_min >= v_max:
                        return None

                    margin_u = max(1, int(round((u_max - u_min) * 0.2)))
                    margin_v = max(1, int(round((v_max - v_min) * 0.2)))
                    return (
                        max(0, u_min - margin_u),
                        min(int(target_camera.width), u_max + margin_u),
                        max(0, v_min - margin_v),
                        min(int(target_camera.height), v_max + margin_v),
                    )
                except Exception:
                    return None

            for target_path in target_paths:
                target_camera = self._get_camera_for_path(target_path)
                if target_camera is None:
                    continue

                target_raster = self.raster_manager.get_raster(target_path)
                if target_raster is None:
                    continue

                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None:
                    continue

                target_has_index = (
                    getattr(target_camera, '_raster', None) is not None and
                    target_camera._raster.index_map is not None
                )
                use_index_lookup = (
                    winning_elements.size > 0 and
                    target_has_index and
                    not (fallback_skip_ortho_index_lookup and target_camera is self.ortho_camera)
                )

                target_rect = None
                target_label_ids = set()

                if use_index_lookup:
                    t_mask_start = perf_counter()
                    target_index_map = target_camera._raster.index_map
                    target_mask_data = target_mask.mask_data
                    max_idx = int(np.max(target_index_map))

                    for source_class_id, subset_elements in class_to_elements.items():
                        if subset_elements.size == 0:
                            continue

                        if int(source_class_id) == 0:
                            target_class_id = 0
                            label_id = None
                        else:
                            label_id = class_label_ids.get(int(source_class_id))
                            label = labels_by_id.get(label_id)
                            if label is None:
                                continue

                            target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                            if target_class_id is None:
                                target_mask.sync_label_map([label])
                                target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                            if target_class_id is None:
                                continue

                        cam_bbox = _project_bbox_for_subset(target_camera, subset_elements)
                        if cam_bbox is not None:
                            u_min, u_max, v_min, v_max = cam_bbox
                            u_min = max(0, int(u_min))
                            u_max = min(target_camera.width, int(u_max))
                            v_min = max(0, int(v_min))
                            v_max = min(target_camera.height, int(v_max))
                        else:
                            u_min, u_max = 0, target_camera.width
                            v_min, v_max = 0, target_camera.height

                        if u_min >= u_max or v_min >= v_max:
                            continue

                        valid_elements = subset_elements[(subset_elements >= 0) & (subset_elements <= max_idx)]
                        if valid_elements.size == 0:
                            continue

                        lut = np.zeros(max_idx + 2, dtype=bool)
                        lut[valid_elements] = True

                        sub_mask = target_mask_data[v_min:v_max, u_min:u_max]

                        # index_map may be pixel-budget-downscaled while mask_data is
                        # full resolution — scale slice coordinates accordingly.
                        im_h, im_w = target_index_map.shape
                        mask_h, mask_w = target_mask_data.shape
                        if im_h != mask_h or im_w != mask_w:
                            sy = im_h / mask_h
                            sx = im_w / mask_w
                            im_v_min = max(0, int(v_min * sy))
                            im_v_max = min(im_h, int(np.ceil(v_max * sy)))
                            im_u_min = max(0, int(u_min * sx))
                            im_u_max = min(im_w, int(np.ceil(u_max * sx)))
                            sub_index_small = target_index_map[im_v_min:im_v_max, im_u_min:im_u_max]
                            import cv2 as _cv2
                            sub_index = _cv2.resize(
                                sub_index_small.astype(np.float32),
                                (sub_mask.shape[1], sub_mask.shape[0]),
                                interpolation=_cv2.INTER_NEAREST,
                            ).astype(np.int32)
                        else:
                            sub_index = target_index_map[v_min:v_max, u_min:u_max]

                        paintable = lut[sub_index] & (sub_mask < target_mask.LOCK_BIT) & (sub_mask != target_class_id)

                        if not np.any(paintable):
                            continue

                        y_local, x_local = np.where(paintable)
                        y_global = y_local + v_min
                        x_global = x_local + u_min
                        flat_global = (y_global * target_camera.width + x_global).astype(np.int64, copy=False)

                        target_mask.update_mask_at_indices(
                            flat_global,
                            int(target_class_id),
                            silent=True,
                        )

                        new_rect = (
                            int(np.min(x_global)),
                            int(np.min(y_global)),
                            int(np.max(x_global)) + 1,
                            int(np.max(y_global)) + 1,
                        )
                        target_rect = _merge_update_rects(target_rect, new_rect)

                        if label_id is not None:
                            target_label_ids.add(label_id)

                    mask_time += perf_counter() - t_mask_start

                if fallback_mask is not None and fallback_center is not None and (not use_index_lookup or target_rect is None):
                    t_mask_start = perf_counter()
                    if fallback_mode == 'erase':
                        target_class_id = 0
                        label = None
                    else:
                        label = labels_by_id.get(fallback_label_id)
                        if label is not None:
                            target_class_id = target_mask.label_id_to_class_id_map.get(fallback_label_id)
                            if target_class_id is None:
                                target_mask.sync_label_map([label])
                                target_class_id = target_mask.label_id_to_class_id_map.get(fallback_label_id)
                        else:
                            target_class_id = None

                    if target_class_id is not None:
                        target_center = fallback_center
                        if fallback_projections:
                            proj = fallback_projections.get(target_path)
                            if proj is not None and len(proj) >= 3 and proj[2]:
                                target_center = (proj[0], proj[1])
                            else:
                                continue

                        if target_center is None:
                            target_center = (0, 0)
                        if fallback_mode in ('brush', 'erase'):
                            brush_mask = self._acquire_propagation_buffer(fallback_mask.shape, dtype=bool)
                            try:
                                np.copyto(brush_mask, np.asarray(fallback_mask, dtype=bool))
                                brush_h, brush_w = brush_mask.shape
                                brush_location = QPointF(
                                    target_center[0] - brush_w / 2.0,
                                    target_center[1] - brush_h / 2.0,
                                )
                                target_mask.update_mask(
                                    brush_location,
                                    brush_mask,
                                    int(target_class_id),
                                    silent=True,
                                )
                                target_rect = _merge_update_rects(
                                    target_rect,
                                    (
                                        max(0, int(target_center[0] - brush_w / 2.0)),
                                        max(0, int(target_center[1] - brush_h / 2.0)),
                                        min(target_camera.width, int(target_center[0] - brush_w / 2.0) + brush_w),
                                        min(target_camera.height, int(target_center[1] - brush_h / 2.0) + brush_h),
                                    ),
                                )
                                if fallback_label_id is not None:
                                    target_label_ids.add(fallback_label_id)
                            finally:
                                self._release_propagation_buffer(brush_mask)
                        elif fallback_mode == 'fill':
                            fill_pos = QPointF(target_center[0], target_center[1])
                            fill_result = target_mask.fill_region(
                                fill_pos,
                                int(target_class_id),
                                silent=True,
                                return_update_rect=True,
                            )
                            if fill_result is not None:
                                fill_mask_result, fill_rect = fill_result
                                if fill_mask_result is not None:
                                    target_rect = _merge_update_rects(target_rect, fill_rect)
                                    if fallback_label_id is not None:
                                        target_label_ids.add(fallback_label_id)
                        elif fallback_mode == 'sam':
                            subset_mask = self._acquire_propagation_buffer(fallback_mask.shape, dtype=np.uint8)
                            try:
                                subset_mask.fill(0)
                                subset_mask[np.asarray(fallback_mask, dtype=bool)] = int(target_class_id)
                                mask_h, mask_w = subset_mask.shape
                                top_left_x = int(target_center[0] - mask_w / 2.0)
                                top_left_y = int(target_center[1] - mask_h / 2.0)
                                target_mask.update_mask_with_mask(
                                    subset_mask,
                                    (top_left_x, top_left_y),
                                    silent=True,
                                )
                                target_rect = _merge_update_rects(
                                    target_rect,
                                    (
                                        max(0, top_left_x),
                                        max(0, top_left_y),
                                        min(target_mask.mask_data.shape[1], top_left_x + mask_w),
                                        min(target_mask.mask_data.shape[0], top_left_y + mask_h),
                                    ),
                                )
                                if fallback_label_id is not None:
                                    target_label_ids.add(fallback_label_id)
                            finally:
                                self._release_propagation_buffer(subset_mask)
                    mask_time += perf_counter() - t_mask_start

                if target_rect is not None:
                    repaint_tasks.append({
                        'type': 'repaint',
                        'path': target_path,
                        'mask': target_mask,
                        'label_ids': tuple(sorted(target_label_ids)),
                        'update_rect': target_rect,
                    })

        except Exception:
            traceback.print_exc()
        finally:
            self._universal_repaint_signal.emit(repaint_tasks)
            print(
                f"DEBUG [Sync Worker]: {len(target_paths)} Cams | Total: {(perf_counter() - t0) * 1000:.2f}ms | "
                f"Mask Gen: {mask_time * 1000:.2f}ms"
            )
            return repaint_tasks

    def _on_universal_repaint(self, repaint_tasks: list):
        """Apply localized UI updates produced by the unified propagation worker."""
        t0 = perf_counter()
        needs_3d_flush = False
        try:
            for task in repaint_tasks:
                task_type = task.get('type')

                if task_type == 'status_message':
                    # Post a status-bar message from a background worker.
                    msg = task.get('message', '')
                    timeout = task.get('timeout', 5000)
                    if msg:
                        status_bar = getattr(self.main_window, 'status_bar', None)
                        if status_bar is not None:
                            try:
                                status_bar.showMessage(msg, timeout)
                            except Exception:
                                pass
                    continue

                if task_type == 'reload_annotation_window':
                    # Refresh the annotation window's mask display after a bulk write.
                    # This ensures the currently-open image shows the new labels
                    # without requiring the user to navigate away and back.
                    try:
                        aw = getattr(self.main_window, 'annotation_window', None)
                        if aw is not None and hasattr(aw, 'load_mask_annotation'):
                            aw.load_mask_annotation()
                    except Exception:
                        pass
                    continue

                if task_type == 'update_image_table':
                    # Refresh the image-table annotation counts for all paths that
                    # received new mask labels during a bulk projection.
                    paths = task.get('paths', ())
                    iw = getattr(self.main_window, 'image_window', None)
                    if iw is not None:
                        for _p in paths:
                            try:
                                iw.update_image_annotations(_p)
                            except Exception:
                                pass
                    continue

                if task_type == '3d_paint':
                    # IMPORTANT: Pass label_id to avoid relying on active UI label
                    # which could overwrite the wrong mesh_class_label_ids entry
                    label_id = task.get('label_id')
                    self.submit_3d_face_paint(
                        task['painted_ids'],
                        task['target_color'],
                        task['source_class_id'],
                        primary_target=task.get('primary_target'),
                        label_id=label_id,
                    )
                    needs_3d_flush = True
                    continue

                if task_type != 'repaint':
                    continue

                target_mask = task.get('mask')
                if target_mask is None:
                    continue

                for label_id in task.get('label_ids', ()):
                    if label_id is not None and label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)

                target_path = task.get('path')
                context_canvas = self._get_context_canvas_for_path(target_path)
                # Default to deferring: when no on-screen canvas currently
                # displays this path (canvas not materialized, or scrolled out
                # of view) the task must be QUEUED, not dropped.  Previously
                # this defaulted to True, so an absent canvas silently skipped
                # both branches below and the matrix thumbnail never refreshed
                # even though mask_data was written.
                should_update_now = (
                    context_canvas is not None
                    and self.context_matrix is not None
                    and self.context_matrix.is_canvas_on_screen(context_canvas)
                )

                if should_update_now:
                    # Wire overlay BEFORE updating so freshly-created masks
                    # (e.g. pre-allocated for cache-loaded cameras) are visible.
                    # Always (re)wire: set_mask_overlay is a no-op when the item
                    # already points at this mask, but it guarantees the matrix
                    # canvas has a live MaskGraphicsItem even for cameras that
                    # were never opened in the AnnotationWindow.
                    context_canvas.set_mask_overlay(target_mask)
                    # Recompute colored_mask + qimage from mask_data.  This must
                    # run even when the mask has no AnnotationWindow graphics_item
                    # (cache-loaded context cameras), otherwise the matrix overlay
                    # would paint a stale image and the thumbnail would not update
                    # until the camera was activated as primary.
                    target_mask.update_graphics_item(update_rect=task.get('update_rect'))
                    # Force the matrix overlay item itself to repaint from the
                    # freshly rebuilt qimage (the mask's update_graphics_item only
                    # repaints its own AnnotationWindow item, not this overlay).
                    overlay_item = getattr(context_canvas, '_mask_overlay_item', None)
                    if overlay_item is not None:
                        try:
                            overlay_item.update()
                        except Exception:
                            pass
                elif self.context_matrix is not None:
                    self.context_matrix.queue_pending_repaint(
                        target_path,
                        target_mask,
                        update_rect=task.get('update_rect'),
                        label_ids=task.get('label_ids', ()),
                    )

        except Exception as e:
            print(f"Error in _on_universal_repaint: {e}")
        finally:
            self._pending_unified_propagation_jobs = max(
                0,
                self._pending_unified_propagation_jobs - 1,
            )
            self._propagating_annotation = self._pending_unified_propagation_jobs > 0
            if needs_3d_flush:
                request_flush = getattr(self, 'request_lazy_flush', None)
                if callable(request_flush):
                    request_flush()
            # Clear the matrix busy state (cursor + propagate button) once the
            # last queued unified-repaint job has been applied.  This ties the
            # UI indicators to actual completion instead of a fixed timer, so
            # they stay active until every camera has really been updated.
            if self._pending_unified_propagation_jobs <= 0:
                if self.context_matrix is not None:
                    set_busy = getattr(self.context_matrix, 'set_propagation_busy', None)
                    if callable(set_busy):
                        try:
                            set_busy(False)
                        except Exception:
                            pass
                # Restore the cursor / post the done message for a multi-annotate
                # semantic-prediction propagation, which is async and therefore
                # cannot restore these in its own call frame.
                self._end_semantic_propagation_busy()

    def _on_semantic_prediction_applied(self, image_path: str, source_mask_annotation,
                                        prediction_regions=None, override_target_paths=None):
        """Propagate a semantic segmentation prediction to all target cameras.

        Called by Semantic.predict() after each image is processed when multi-annotate
        is enabled. When region payloads are supplied, only those tiles contribute
        to the 3D element votes, which keeps work-area predictions scoped to the
        predicted region instead of the entire source mask.

        Perspective ↔ orthomosaic propagation is supported: when the source is an
        OrthoCamera, targets are all visible perspective cameras; when the source is a
        perspective camera, targets include visible context cameras plus the ortho.

        Cameras without a pre-computed index map are skipped — full-image warping
        without geometry is too imprecise to be useful for semantic masks.

        Args:
            image_path: Path of the image whose Semantic prediction just completed.
            source_mask_annotation: The MaskAnnotation whose mask_data was just
                                    updated by the Semantic model.
            override_target_paths: When provided, use this set of paths as the
                                   propagation targets instead of the default
                                   visible-context-camera set.  Used by
                                   propagate_current_semantic_mask_to_all_cameras.
        """
        status_bar = getattr(self.main_window, 'status_bar', None)

        def _status(msg, timeout=0):
            if status_bar is not None and msg:
                try:
                    status_bar.showMessage(msg, timeout)
                except Exception:
                    pass

        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            return

        if override_target_paths is not None:
            selected_paths = set(override_target_paths)
        else:
            selected_paths = self._get_semantic_target_paths(source_camera)

        if not selected_paths:
            _status("Multi-annotate: no target cameras with index maps to propagate to.", 4000)
            return

        # This path is otherwise dead silent.  Show a WaitCursor and status-bar
        # progress so the user knows the semantic prediction is being projected
        # across the context cameras.  The actual propagation runs asynchronously
        # on the unified background executor, so the cursor is restored from the
        # completion handler (_on_universal_repaint) once the job finishes --
        # NOT here -- otherwise the indicators would clear before the work does.
        # The synchronous vote-extraction below happens first; if it yields no
        # geometry we bail and restore the cursor immediately.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._semantic_propagation_busy = True
        _status(
            f"Multi-annotate: projecting semantic prediction to {len(selected_paths)} camera(s)...",
            0,
        )
        QApplication.processEvents()

        try:
            project_labels = list(self.main_window.label_window.labels)
            element_ids, class_ids, class_label_ids = self._extract_semantic_element_votes(
                source_camera,
                source_mask_annotation,
                prediction_regions=prediction_regions,
            )
        except Exception:
            self._end_semantic_propagation_busy()
            _status("Multi-annotate: semantic prediction failed during vote extraction.", 4000)
            raise

        if element_ids.size == 0 or class_ids.size == 0 or not class_label_ids:
            self._end_semantic_propagation_busy()
            _status(
                "Multi-annotate: semantic prediction covers no scene geometry; nothing to propagate.",
                4000,
            )
            return

        _status(
            f"Multi-annotate: painting {element_ids.size:,} element(s) "
            f"across {len(selected_paths)} camera(s)...",
            0,
        )
        # Stash a completion message the async handler will post when done.
        self._semantic_propagation_done_msg = (
            f"Multi-annotate: propagated semantic prediction to {len(selected_paths)} camera(s)."
        )
        QApplication.processEvents()

        self._execute_mask_propagation(
            source_camera=source_camera,
            element_ids=element_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids=class_label_ids,
        )

    def _end_semantic_propagation_busy(self):
        """Restore the cursor / post the done message for a semantic propagation.

        Safe to call multiple times; only the first call after a busy period
        takes effect.
        """
        if not getattr(self, '_semantic_propagation_busy', False):
            return
        self._semantic_propagation_busy = False
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        done_msg = getattr(self, '_semantic_propagation_done_msg', None)
        self._semantic_propagation_done_msg = None
        if done_msg:
            status_bar = getattr(self.main_window, 'status_bar', None)
            if status_bar is not None:
                try:
                    status_bar.showMessage(done_msg, 4000)
                except Exception:
                    pass

    def _on_viewer_sam_accepted(self, binary_mask, index_map, index_map_gpu, depth_map, element_type, label):
        """Convert mask pixels to element IDs and propagate to target cameras.

        When multi_annotate_enabled is True: paints the primary annotation-window
        camera and all visible context cameras (mirrors Brush3DTool behaviour).
        When multi_annotate_enabled is False: only updates the 3D product — no 2D
        camera is painted at all, because the VTK view has no corresponding image.
        """
        # --- extract element IDs from the mask ---------------------------------
        element_ids = self._filter_index_map_ids_by_depth(
            index_map=index_map,
            binary_mask=binary_mask,
            depth_map=depth_map,
        )

        if element_ids.size == 0:
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: mask covers no scene geometry.', 4000)
            return

        n = element_ids.size
        self.main_window.status_bar.showMessage(
            f"MVAT-SAM: painting {n:,} {element_type}(s) with '{label.short_label_code}'...", 0)
        QApplication.processEvents()

        # --- resolve label color -----------------------------------------------
        color_rgb = (label.color.red(), label.color.green(), label.color.blue())

        # --- 3D-only path (multi-annotate OFF) ---------------------------------
        if not self.multi_annotate_enabled:
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target is None or not hasattr(primary_target, 'apply_labels'):
                self.main_window.status_bar.showMessage(
                    'MVAT-SAM: no 3D product to paint.', 4000)
            return

        # --- multi-annotate ON: propagate to 3D + all visible 2D cameras ------
        project_labels = list(self.main_window.label_window.labels)

        annotation_window = getattr(self.main_window, 'annotation_window', None)
        primary_path = getattr(annotation_window, 'current_image_path', None)
        target_paths = self._get_annotation_target_paths()
        if primary_path:
            target_paths.add(primary_path)

        if not target_paths:
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: no target cameras to paint.', 4000)
            return

        source_path     = next(iter(target_paths))
        source_raster   = self.raster_manager.get_raster(source_path)
        source_mask_ann = source_raster.get_mask_annotation(project_labels) if source_raster else None
        source_class_id = None
        if source_mask_ann is not None:
            source_class_id = source_mask_ann.label_id_to_class_id_map.get(label.id)
            if source_class_id is None:
                source_mask_ann.sync_label_map([label])
                source_class_id = source_mask_ann.label_id_to_class_id_map.get(label.id)

        if source_class_id is None:
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: could not resolve class ID for selected label.', 4000)
            return

        class_ids     = np.full(element_ids.size, int(source_class_id), dtype=np.int64)
        source_camera = self.cameras.get(source_path)
        self._execute_mask_propagation(
            source_camera=source_camera,
            element_ids=element_ids,
            class_ids=class_ids,
            target_paths=target_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label.id},
        )
        self.main_window.status_bar.showMessage(
            f"MVAT-SAM: applied '{label.short_label_code}' to {n:,} {element_type}(s).", 4000)

    def _filter_index_map_ids_by_depth(self,
                                       index_map: np.ndarray,
                                       binary_mask: np.ndarray,
                                       depth_map=None) -> np.ndarray:
        """Return unique element IDs under a mask, optionally rejecting depth outliers.

        The depth-aware branch mirrors the 2D SAM propagation logic: it keeps the
        interior of the mask and only accepts perimeter pixels when their depth is
        consistent with the interior band.
        """
        if index_map is None or binary_mask is None:
            return np.array([], dtype=np.int64)

        index_map = np.asarray(index_map)
        binary_mask = np.asarray(binary_mask)
        if index_map.ndim != 2 or binary_mask.ndim != 2:
            return np.array([], dtype=np.int64)

        if index_map.shape != binary_mask.shape:
            return np.array([], dtype=np.int64)

        valid_mask = binary_mask.astype(bool)
        if not np.any(valid_mask):
            return np.array([], dtype=np.int64)

        if depth_map is not None:
            try:
                import cv2

                depth_map = np.asarray(depth_map)
                if depth_map.shape != index_map.shape:
                    depth_map = cv2.resize(
                        depth_map.astype(np.float32),
                        (index_map.shape[1], index_map.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                erosion_r = int(np.clip(min(valid_mask.shape) * 0.03, 2, 12))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * erosion_r + 1, 2 * erosion_r + 1),
                )
                interior_mask = cv2.erode(
                    valid_mask.astype(np.uint8),
                    kernel,
                    iterations=1,
                ).astype(bool)
                perimeter_mask = valid_mask & ~interior_mask

                depth_slice = depth_map.astype(np.float32, copy=False)
                interior_depths = depth_slice[interior_mask]
                interior_depths = interior_depths[~np.isnan(interior_depths)]


                if len(interior_depths) >= 10 and perimeter_mask.any():
                    ref_depth = np.median(interior_depths)
                    interior_spread = np.std(interior_depths)
                    abs_floor = max(0.02, ref_depth * 0.005)
                    full_tol = interior_spread * 2.0 + abs_floor
                    dist = cv2.distanceTransform(valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    norm_dist = np.clip(dist / max(erosion_r, 1), 0.0, 1.0)
                    per_pixel_tol = abs_floor + (full_tol - abs_floor) * norm_dist
                    with np.errstate(invalid='ignore'):
                        perimeter_depth_ok = np.abs(depth_slice - ref_depth) <= per_pixel_tol
                    valid_mask = interior_mask | (perimeter_mask & perimeter_depth_ok)
            except Exception:
                pass

        raw_ids = index_map[valid_mask]
        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    # ── Flow 1b: This camera → Mesh ──────────────────────────────────────────────

    def aggregate_active_camera_mask_to_mesh(self):
        """Aggregate the active camera's semantic mask onto the 3D mesh.

        Convenience wrapper around aggregate_camera_masks_to_mesh that restricts
        the source to only the currently selected/active camera.  Useful when the
        user wants to push just one camera's annotations to the mesh without
        re-aggregating all cameras.
        """
        camera = getattr(self, 'selected_camera', None)
        if camera is None:
            self._warn_semantic_propagation(
                "No active camera selected.  Open an image first."
            )
            return

        camera_path = getattr(camera, 'image_path', None)
        if camera_path is None:
            self._warn_semantic_propagation("Active camera has no image path.")
            return

        raster = getattr(camera, '_raster', None)
        if raster is None or getattr(raster, 'index_map', None) is None:
            self._warn_semantic_propagation(
                f"'{camera.label}' has no index map.  "
                "Enable Multi-Annotate or load the index map first."
            )
            return

        self.aggregate_camera_masks_to_mesh(
            camera_paths=[camera_path],
            also_project_to_cameras=False,
        )

    # ── Flow 3b: Mesh → This camera ──────────────────────────────────────────────

    def project_mesh_labels_to_active_camera(self, skip_unlabeled: bool = True):
        """Project the 3D mesh's face labels to the active camera's semantic mask.

        Convenience wrapper around project_mesh_labels_to_cameras that writes only
        to the currently active camera.  Unlabeled mesh faces are skipped when
        ``skip_unlabeled`` is True so existing hand-painted regions are preserved.
        """
        camera = getattr(self, 'selected_camera', None)
        if camera is None:
            self._warn_semantic_propagation(
                "No active camera selected.  Open an image first."
            )
            return

        camera_path = getattr(camera, 'image_path', None)
        if camera_path is None:
            self._warn_semantic_propagation("Active camera has no image path.")
            return

        raster = getattr(camera, '_raster', None)
        if raster is None or getattr(raster, 'index_map', None) is None:
            self._warn_semantic_propagation(
                f"'{camera.label}' has no index map.  "
                "Enable Multi-Annotate or load the index map first."
            )
            return

        self.project_mesh_labels_to_cameras(
            camera_paths=[camera_path],
            skip_unlabeled=skip_unlabeled,
        )

    # ── Flow 1c: Primary camera → all cameras (with index map) ───────────────────

    def propagate_current_semantic_mask_to_all_cameras(self):
        """Propagate the primary camera's semantic mask to every camera that has an index map.

        Like propagate_current_semantic_mask but targets the entire project rather
        than only the cameras currently loaded in the context matrix.
        """
        annotation_window = getattr(self.main_window, 'annotation_window', None)
        if annotation_window is None:
            self._warn_semantic_propagation("AnnotationWindow is not available.")
            return

        image_path = getattr(annotation_window, 'current_image_path', None)
        if not image_path:
            self._warn_semantic_propagation("No image is currently active in the AnnotationWindow.")
            return

        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            self._warn_semantic_propagation("The active image is not loaded in MVAT.")
            return

        # Build target set: all cameras with an index map, excluding the source
        all_target_paths = set()
        for path, cam in self.cameras.items():
            raster = getattr(cam, '_raster', None)
            if raster is not None and getattr(raster, 'index_map', None) is not None:
                all_target_paths.add(path)
        all_target_paths.discard(image_path)

        if not all_target_paths:
            self._warn_semantic_propagation(
                "No other cameras have an index map loaded.  "
                "Load index maps first (Load Cameras → Load Index Maps)."
            )
            return

        source_raster = self.raster_manager.get_raster(image_path) if self.raster_manager is not None else None
        source_mask = getattr(source_raster, 'mask_annotation', None)
        if source_mask is None:
            self._warn_semantic_propagation("The active image does not have a semantic mask to propagate.")
            return

        if getattr(source_camera, '_raster', None) is None or \
                getattr(source_camera._raster, 'index_map', None) is None:
            self._warn_semantic_propagation(
                "The active camera does not have an index map, so semantic propagation is unavailable."
            )
            return

        try:
            self._on_semantic_prediction_applied(
                image_path,
                source_mask,
                override_target_paths=all_target_paths,
            )
        except Exception as exc:
            import traceback as _tb
            print(f"Error while propagating semantic mask from {image_path} to all cameras: {exc}")
            _tb.print_exc()
            self._warn_semantic_propagation("Semantic propagation failed. See console for details.")

    # ── Flow 3c: Mesh → visible (context-matrix) cameras ─────────────────────────

    def project_mesh_labels_to_visible_cameras(self, skip_unlabeled: bool = True):
        """Project the 3D mesh's face labels to cameras currently in the context matrix.

        Convenience wrapper around project_mesh_labels_to_cameras restricted to the
        camera paths that are currently loaded in the context matrix view.

        Args:
            skip_unlabeled: When True (default), only overwrite pixels whose face
                            carries a non-zero class so existing hand-painted
                            regions are preserved.
        """
        visible_paths = self._get_visible_context_camera_paths()
        if not visible_paths:
            self._warn_semantic_propagation(
                "No cameras are currently visible in the context matrix."
            )
            return

        self.project_mesh_labels_to_cameras(
            camera_paths=visible_paths,
            skip_unlabeled=skip_unlabeled,
        )

        # ── Flow 2: All cameras → Mesh ────────────────────────────────────────────

    def aggregate_camera_masks_to_mesh(self, camera_paths=None, also_project_to_cameras=False):
        """Aggregate semantic masks from cameras onto the 3D mesh via vote resolution.

        For every camera that has both an index map and a mask annotation, each
        labeled pixel casts a vote for the mesh face visible at that pixel.
        Conflicts (the same face seen by multiple cameras with different labels)
        are resolved by majority vote; ties break toward the smaller class ID.

        Args:
            camera_paths: Iterable of image paths to aggregate from.  Pass
                          ``None`` to use every loaded camera (including ortho).
            also_project_to_cameras: When True, immediately projects the updated
                                     mesh labels back to all cameras after the
                                     mesh has been painted (Flow 2 + Flow 3).

        TODO: Weight votes by per-face pixel coverage or model confidence once
              confidence values become meaningful.
        """
        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None or not hasattr(primary_target, 'apply_labels'):
            self._warn_semantic_propagation(
                "No primary 3D target is available for mask aggregation."
            )
            return

        project_labels = list(self.main_window.label_window.labels)
        if not project_labels:
            self._warn_semantic_propagation("No project labels are available.")
            return

        if camera_paths is not None:
            source_cameras = [
                (p, self.cameras[p]) for p in camera_paths if p in self.cameras
            ]
        else:
            source_cameras = list(self.cameras.items())
            if self.ortho_camera is not None:
                ortho_path = self.ortho_camera.image_path
                if not any(p == ortho_path for p, _ in source_cameras):
                    source_cameras.append((ortho_path, self.ortho_camera))

        if not source_cameras:
            self._warn_semantic_propagation("No cameras with index maps found.")
            return

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage(
                    f"Cameras → Mesh: scanning {len(source_cameras)} camera(s) "
                    f"(only those with index maps + labeled masks will contribute)…", 0
                )
            except Exception:
                pass

        self._pending_unified_propagation_jobs += 1
        self._propagating_annotation = True
        self._unified_bg_executor.submit(
            self._bg_aggregate_cameras_to_mesh,
            source_cameras,
            project_labels,
            primary_target,
            also_project_to_cameras,
        )

    def _bg_aggregate_cameras_to_mesh(
        self, source_cameras, project_labels, primary_target, also_project_to_cameras
    ):
        """Background worker: collect per-camera votes and paint the mesh."""
        from time import perf_counter
        t0 = perf_counter()

        # Build a canonical class_id space keyed on label UUID so that class IDs
        # from different mask annotations (which may assign different integers to
        # the same label) are normalised before votes are counted.
        # IMPORTANT: Exclude the 'Review' label (id='-1') from canonical mapping since
        # it represents unlabeled/unannotated pixels and shouldn't participate in
        # the canonical ID scheme. This prevents Review from shifting the canonical IDs
        # of actual semantic labels.
        real_labels = [lbl for lbl in project_labels if getattr(lbl, 'id', None) and lbl.id != '-1']
        canonical_id_for = {lbl.id: (idx + 1) for idx, lbl in enumerate(real_labels)}
        canonical_label_ids = {(idx + 1): lbl.id for idx, lbl in enumerate(real_labels)}
        labels_by_id = {
            getattr(lbl, 'id', None): lbl
            for lbl in project_labels
            if getattr(lbl, 'id', None) is not None
        }

        all_element_ids = []
        all_class_ids = []
        skipped_no_index = 0
        skipped_no_mask = 0
        skipped_no_votes = 0
        contributing = 0

        for path, camera in source_cameras:
            raster = getattr(camera, '_raster', None)
            if raster is None or getattr(raster, 'index_map', None) is None:
                skipped_no_index += 1
                continue
            mask_annotation = getattr(raster, 'mask_annotation', None)
            if mask_annotation is None:
                skipped_no_mask += 1
                continue

            try:
                element_ids, local_class_ids, class_label_ids = (
                    self._extract_semantic_element_votes(camera, mask_annotation)
                )
            except Exception as exc:
                print(f"aggregate_camera_masks_to_mesh: vote extraction failed for {path}: {exc}")
                skipped_no_votes += 1
                continue

            if element_ids.size == 0:
                skipped_no_votes += 1
                continue

            # Translate local class IDs → canonical IDs so votes from cameras
            # with different label orderings count toward the same label.
            canonical_ids = np.zeros_like(local_class_ids)
            for local_id, label_id in class_label_ids.items():
                canon_id = canonical_id_for.get(label_id)
                if canon_id is None:
                    continue
                canonical_ids[local_class_ids == local_id] = canon_id

            valid = canonical_ids > 0
            if not np.any(valid):
                skipped_no_votes += 1
                continue

            all_element_ids.append(element_ids[valid])
            all_class_ids.append(canonical_ids[valid])
            contributing += 1

        total = len(source_cameras)
        print(
            f"[Cameras→Mesh] {contributing}/{total} cameras contributed | "
            f"{skipped_no_index} no index map | "
            f"{skipped_no_mask} no mask | "
            f"{skipped_no_votes} no labeled votes"
        )

        if not all_element_ids:
            msg = (
                f"Cameras → Mesh: 0 of {total} camera(s) had both index maps and "
                f"labeled masks. Enable Multi-Annotate so index maps are computed, "
                f"then paint masks before aggregating."
            )
            print(f"WARNING: {msg}")
            self._universal_repaint_signal.emit([{
                'type': 'status_message', 'message': msg, 'timeout': 8000,
            }])
            return

        element_ids = np.concatenate(all_element_ids).astype(np.int64, copy=False)
        class_ids = np.concatenate(all_class_ids).astype(np.int64, copy=False)

        unique_elements, winning_classes = resolve_class_conflicts_vectorized(
            element_ids, class_ids
        )

        if unique_elements.size == 0:
            self._universal_repaint_signal.emit([])
            return

        repaint_tasks = []
        new_mesh_class_label_ids = {}

        for canon_id in np.unique(winning_classes):
            label_id = canonical_label_ids.get(int(canon_id))
            label = labels_by_id.get(label_id)
            if label is None:
                continue
            face_ids = unique_elements[winning_classes == canon_id].astype(np.int32)
            new_mesh_class_label_ids[int(canon_id)] = label_id
            repaint_tasks.append({
                'type': '3d_paint',
                'painted_ids': face_ids,
                'target_color': (
                    label.color.red(),
                    label.color.green(),
                    label.color.blue(),
                ),
                'source_class_id': int(canon_id),
                'primary_target': primary_target,
                'label_id': label_id,
            })

        # Persist the canonical mapping so project_mesh_labels_to_cameras can
        # resolve these class IDs back to label UUIDs.
        self._mesh_class_label_ids.update(new_mesh_class_label_ids)

        elapsed_ms = (perf_counter() - t0) * 1000
        label_summary = ", ".join(
            f"{labels_by_id[lid].short_label_code}:{int(np.sum(winning_classes == cid)):,}"
            for cid, lid in new_mesh_class_label_ids.items()
            if lid in labels_by_id
        )
        done_msg = (
            f"Cameras → Mesh: {unique_elements.size:,} face(s) from "
            f"{contributing}/{total} camera(s) in {elapsed_ms:.0f} ms"
            + (f"  [{label_summary}]" if label_summary else "")
        )
        print(f"[Cameras→Mesh] {done_msg}")
        repaint_tasks.append({'type': 'status_message', 'message': done_msg, 'timeout': 6000})

        if also_project_to_cameras:
            # Build the updated class_ids array locally (avoids LabelWorker race)
            # so camera repaint tasks can be emitted in the same batch.
            new_mesh_class_ids = np.asarray(
                primary_target.class_ids, dtype=np.int32
            ).copy()
            for task in repaint_tasks:
                if task['type'] == '3d_paint':
                    fids = np.asarray(task['painted_ids'], dtype=np.int64)
                    valid_fids = fids[(fids >= 0) & (fids < len(new_mesh_class_ids))]
                    new_mesh_class_ids[valid_fids] = task['source_class_id']

            camera_repaint_tasks = self._compute_mesh_to_camera_repaint_tasks(
                new_mesh_class_ids,
                new_mesh_class_label_ids,
                project_labels,
                labels_by_id,
                camera_paths=None,
                skip_unlabeled=True,
            )
            repaint_tasks.extend(camera_repaint_tasks)
            cam_count = sum(1 for t in camera_repaint_tasks if t.get('type') == 'repaint')
            repaint_tasks.append({
                'type': 'status_message',
                'message': f"Cameras → Mesh → Cameras: projected back to {cam_count} camera(s).",
                'timeout': 5000,
            })

        self._universal_repaint_signal.emit(repaint_tasks)

    # ── Flow 3: Mesh → All cameras ────────────────────────────────────────────

    def project_mesh_labels_to_cameras(self, camera_paths=None, skip_unlabeled=True):
        """Project the 3D mesh's face labels back to every camera's mask annotation.

        Each camera pixel is coloured with the class of whichever mesh face is
        visible at that pixel (via the camera's pre-computed index map).  Pixels
        whose face has no label (class_id == 0) are skipped when
        ``skip_unlabeled`` is True so existing hand-painted regions are preserved.

        Args:
            camera_paths: Iterable of image paths to write into.  Pass ``None``
                          to update every loaded camera (including ortho).
            skip_unlabeled: When True (default), only overwrite pixels whose
                            face carries a non-zero class.  When False, unlabeled
                            faces clear the pixel's existing label.
        """
        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None or not hasattr(primary_target, 'class_ids'):
            self._warn_semantic_propagation(
                "No primary 3D target with label data is available."
            )
            return

        if not self._mesh_class_label_ids:
            self._warn_semantic_propagation(
                "The mesh has no label data to project.  Paint the mesh first."
            )
            return

        project_labels = list(self.main_window.label_window.labels)
        if not project_labels:
            self._warn_semantic_propagation("No project labels are available.")
            return

        mesh_class_ids = np.asarray(primary_target.class_ids, dtype=np.int32).copy()
        mesh_class_label_ids = dict(self._mesh_class_label_ids)
        labels_by_id = {
            getattr(lbl, 'id', None): lbl
            for lbl in project_labels
            if getattr(lbl, 'id', None) is not None
        }

        # ── Pre-create mask annotations on the main thread ─────────────────
        # Cameras loaded from disk-cache have an index_map but no mask_annotation
        # yet (they were never opened in the AnnotationWindow).  We must allocate
        # the mask here — on the main thread — before the background worker tries
        # to write to it.
        if camera_paths is not None:
            _cam_items = [(p, self.cameras[p]) for p in camera_paths if p in self.cameras]
        else:
            _cam_items = list(self.cameras.items())
            if self.ortho_camera is not None:
                _ortho_path = self.ortho_camera.image_path
                if not any(p == _ortho_path for p, _ in _cam_items):
                    _cam_items.append((_ortho_path, self.ortho_camera))

        _created = 0
        for _path, _cam in _cam_items:
            _raster = getattr(_cam, '_raster', None)
            if _raster is None or getattr(_raster, 'index_map', None) is None:
                continue
            if getattr(_raster, 'mask_annotation', None) is None:
                try:
                    _raster.get_mask_annotation(project_labels)
                    _created += 1
                except Exception:
                    pass
        if _created:
            print(f"[Mesh→Cameras] Pre-created mask annotations for {_created} camera(s).")

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage("Mesh → Cameras: projecting labels…", 0)
            except Exception:
                pass

        self._pending_unified_propagation_jobs += 1
        self._propagating_annotation = True
        self._unified_bg_executor.submit(
            self._bg_project_mesh_to_cameras,
            mesh_class_ids,
            mesh_class_label_ids,
            project_labels,
            labels_by_id,
            camera_paths,
            skip_unlabeled,
        )

    def _bg_project_mesh_to_cameras(
        self,
        mesh_class_ids,
        mesh_class_label_ids,
        project_labels,
        labels_by_id,
        camera_paths,
        skip_unlabeled,
    ):
        """Background worker: project mesh class_ids to each camera's mask."""
        from time import perf_counter
        t0 = perf_counter()

        repaint_tasks = self._compute_mesh_to_camera_repaint_tasks(
            mesh_class_ids,
            mesh_class_label_ids,
            project_labels,
            labels_by_id,
            camera_paths=camera_paths,
            skip_unlabeled=skip_unlabeled,
        )

        elapsed_ms = (perf_counter() - t0) * 1000
        cam_count = sum(1 for t in repaint_tasks if t.get('type') == 'repaint')
        done_msg = f"Mesh → Cameras: {cam_count} camera(s) updated in {elapsed_ms:.0f} ms"
        print(f"[Mesh→Cameras] {done_msg}")
        # Collect updated paths for image-table refresh
        updated_paths = tuple(t['path'] for t in repaint_tasks if t.get('type') == 'repaint')
        if updated_paths:
            repaint_tasks.append({'type': 'update_image_table', 'paths': updated_paths})
        # Reload the annotation window so the currently-open image reflects the new
        # labels immediately without requiring a manual navigation round-trip.
        repaint_tasks.append({'type': 'reload_annotation_window'})
        repaint_tasks.append({'type': 'status_message', 'message': done_msg, 'timeout': 5000})
        self._universal_repaint_signal.emit(repaint_tasks)

    def _compute_mesh_to_camera_repaint_tasks(
        self,
        mesh_class_ids,
        mesh_class_label_ids,
        project_labels,
        labels_by_id,
        camera_paths,
        skip_unlabeled,
    ):
        """Compute per-camera mask writes for mesh → cameras projection.

        Returns a list of ``'repaint'`` task dicts suitable for
        ``_universal_repaint_signal``.  Runs entirely in numpy — safe to call
        from any background thread.
        """
        import cv2

        if camera_paths is not None:
            cameras = [
                (p, self.cameras[p])
                for p in camera_paths
                if p in self.cameras
            ]
        else:
            cameras = list(self.cameras.items())
            if self.ortho_camera is not None:
                ortho_path = self.ortho_camera.image_path
                if not any(p == ortho_path for p, _ in cameras):
                    cameras.append((ortho_path, self.ortho_camera))

        repaint_tasks = []

        for path, camera in cameras:
            try:
                raster = getattr(camera, '_raster', None)
                if raster is None:
                    continue
                index_map = getattr(raster, 'index_map', None)
                if index_map is None:
                    continue
                mask_annotation = getattr(raster, 'mask_annotation', None)
                if mask_annotation is None:
                    continue

                mask_data = mask_annotation.mask_data
                im_h, im_w = index_map.shape
                mask_h, mask_w = mask_data.shape

                # Upscale index_map to full mask resolution if it was downscaled
                if im_h != mask_h or im_w != mask_w:
                    index_map_full = cv2.resize(
                        index_map.astype(np.float32),
                        (mask_w, mask_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)
                else:
                    index_map_full = index_map

                # Vectorised LUT: pixel → face_id → mesh class_id
                valid = (index_map_full >= 0) & (index_map_full < len(mesh_class_ids))
                face_ids_at_pixels = index_map_full[valid].astype(np.int64)
                pixel_mesh_classes = mesh_class_ids[face_ids_at_pixels].astype(np.int32)

                new_class_layer = np.zeros(mask_data.shape, dtype=mask_data.dtype)
                new_class_layer.flat[np.flatnonzero(valid)] = pixel_mesh_classes

                lock_bit = getattr(mask_annotation, 'LOCK_BIT', 128)
                not_locked = mask_data < lock_bit

                if skip_unlabeled:
                    write_mask = (new_class_layer != 0) & not_locked
                else:
                    write_mask = valid & not_locked

                if not np.any(write_mask):
                    continue

                # Resolve canonical class IDs → per-mask class IDs, remapping
                # where they differ, and collect label IDs being written.
                # CRITICAL: Use a LUT-based approach to avoid collision issues when
                # remapping in-place (e.g., canon_id=1→target_id=2, then later
                # canon_id=2→target_id=3 would remapped pixels from step 1).
                written_classes = np.unique(new_class_layer[write_mask])
                written_classes = written_classes[written_classes > 0]
                written_label_ids = set()

                # Build complete canon_id → target_class_id mapping FIRST
                canon_to_target_lut = {}
                for canon_id in written_classes:
                    label_id = mesh_class_label_ids.get(int(canon_id))
                    if label_id is None:
                        continue
                    label = labels_by_id.get(label_id)
                    if label is None:
                        continue

                    target_class_id = mask_annotation.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        mask_annotation.sync_label_map([label])
                        target_class_id = mask_annotation.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        continue

                    canon_to_target_lut[int(canon_id)] = int(target_class_id)
                    written_label_ids.add(label_id)

                # Apply ALL remappings atomically using a clean copy
                if canon_to_target_lut:
                    final_class_layer = new_class_layer.copy()
                    for canon_id, target_id in canon_to_target_lut.items():
                        final_class_layer[new_class_layer == canon_id] = target_id
                    mask_data[write_mask] = final_class_layer[write_mask]
                else:
                    mask_data[write_mask] = new_class_layer[write_mask]

                ys, xs = np.where(write_mask)
                update_rect = (
                    int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1,
                )

                repaint_tasks.append({
                    'type': 'repaint',
                    'path': path,
                    'mask': mask_annotation,
                    'update_rect': update_rect,
                    'label_ids': written_label_ids,
                })

            except Exception as exc:
                print(f"_compute_mesh_to_camera_repaint_tasks: failed for {path}: {exc}")
                traceback.print_exc()

        return repaint_tasks

