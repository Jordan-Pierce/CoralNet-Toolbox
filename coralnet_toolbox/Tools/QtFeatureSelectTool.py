"""
FeatureSelectTool — 2D feature-similarity query tool for the annotation window.

A 2D feature-similarity selection tool. Instead of SAM, it queries a
dense feature map (DINOv2/v3, TIMM, ...) with cosine similarity and shows the
per-patch similarity as a live heatmap overlay. The interaction mirrors SAMTool:

  - The tool requires a deployed feature model (gated at the toolbar button).
  - On activation nothing is shown; the user first defines a **work area** — a
    subset of the image, or the whole image — exactly like SAMTool:
      * Space with no work area : use the current view extent as the work area.
      * Plain Left-click, Left-click : draw a custom work-area rectangle.
    Dense features are then extracted for just that crop.
  - Within the work area:
      * Ctrl + Left-click  : add a positive prototype (patch under the cursor).
      * Ctrl + Right-click : add a negative prototype.
      * Hover              : live similarity preview to the patch under the cursor.
      * Ctrl + wheel       : adjust the selection threshold (live thresholded view).
      * Space              : finalize → create a Polygon or Mask annotation.
      * Backspace          : clear prompts / cancel the work area.

The cosine / linear-head scoring is reused as-is from the shared
``QueryEngine`` (coralnet_toolbox/Features/QueryEngine.py); the tool owns the
prototype lists and treats the engine as a stateless compute kernel.
"""

import warnings

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem, QApplication

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.QtActions import MaskEditAction

from coralnet_toolbox.WorkArea import WorkArea

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import polygonize_mask_with_holes

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FeatureSelectTool(Tool):
    """Click-to-query dense-feature similarity tool (2D annotation window)."""

    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window
        self.feature_dialog = None

        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor

        # Work area + the local (crop) feature grid the active query runs over.
        self.working_area = None
        self.image_path = None
        self.original_image = None       # full-image RGB [H, W, 3]
        self.original_width = None
        self.original_height = None
        self.query_engine = None
        self.feat_h = None               # crop feature grid rows
        self.feat_w = None               # crop feature grid cols

        # Committed prototypes, as flat crop-grid element ids (gy * feat_w + gx).
        self.positive_ids = []
        self.negative_ids = []
        self.point_graphics = []

        # Selection threshold (raw similarity scale, shared with QueryEngine.select).
        self.threshold = 0.5
        self.threshold_active = False

        # Output settings — synced from the feature deploy dialog.
        self.output_type = "Polygon"
        self.allow_holes = False

        # Optional final-mask propagation callback, wired by any consumer
        # (same contract as SAMTool.post_prediction_callback).
        # Signature: callback(anchor: QPointF, label_id: str, binary_mask: np.ndarray)
        self.post_prediction_callback = None

        # Custom work-area creation state (mirroring SAMTool).
        self.creating_working_area = False
        self.working_area_start = None
        self.working_area_temp_graphics = None

        self.hover_pos = None

        # Debounce hover refresh so a flood of move events can't back up.
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._on_hover_timeout)
        self.debounce_ms = 30

        # Whether the shared colormap controls have been handed to the feature
        # overlay. Engaged only once a work area exists (not on tool activation),
        # so deactivate() only tears them down if we actually took them over.
        self._colormap_controls_engaged = False

    # The similarity overlay's colormap is now owned by the shared ColorMapOverlay
    # (driven by the colormap dropdown); the tool only emits palette indices.
    DEFAULT_COLORMAP = "Plasma"

    # ==================== Activation ====================

    def activate(self):
        """Activate the tool. Nothing is shown until a work area is created."""
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.feature_dialog = getattr(self.main_window, 'feature_deploy_model_dialog', None)
        self.sync_settings_from_dialog()
        # NOTE: the colormap controls are NOT engaged here. They are handed to the
        # feature overlay only once the user actually creates a work area (see
        # _setup_working_area), so simply selecting the tool button doesn't yet
        # flip the dropdown to Plasma or hide the Z-channel.

    def _engage_colormap_controls(self):
        """Hand the colormap dropdown + opacity slider to the feature overlay.

        Called once a work area exists (not on tool activation). Hides any depth
        (Z-channel) overlay — the user re-selects it manually on exit — points the
        shared controls at the feature overlay, defaults the colormap to Plasma,
        and enables the controls so opacity/colormap can be tuned live.
        """
        if self._colormap_controls_engaged:
            return
        aw = self.annotation_window
        try:
            aw._z_overlay.hide()
            aw.set_active_colormap_overlay('feature')
            # The dynamic-range button is depth-specific; disable it so it can't
            # drive the (hidden) Z overlay while the feature overlay is active.
            if hasattr(aw, 'z_dynamic_button'):
                aw.z_dynamic_button.setChecked(False)
                aw.z_dynamic_button.setEnabled(False)
            aw.colormap_dropdown.setEnabled(True)
            aw.colormap_opacity_slider.setEnabled(True)
            if aw.colormap_dropdown.currentText() == self.DEFAULT_COLORMAP:
                # Already Plasma: setCurrentText won't fire, so apply directly so
                # the feature overlay picks up the table.
                aw.update_overlay_colormap(self.DEFAULT_COLORMAP)
            else:
                aw.colormap_dropdown.setCurrentText(self.DEFAULT_COLORMAP)
            aw.set_overlay_opacity(aw.colormap_opacity_slider.value() / 255.0)
            self._colormap_controls_engaged = True
        except Exception:
            pass

    def _release_colormap_controls(self):
        """Return the colormap controls to the (still-hidden) depth overlay.

        Per design the Z-channel overlay is left hidden; setting the dropdown to
        'None' reflects that and disables the opacity slider. Controls are fully
        disabled when the image has no depth data. No-op if the controls were
        never engaged (the user selected the tool but never made a work area).
        """
        if not self._colormap_controls_engaged:
            return
        self._colormap_controls_engaged = False
        aw = self.annotation_window
        try:
            aw.set_active_colormap_overlay('z')
            if aw.colormap_dropdown.currentText() != 'None':
                aw.colormap_dropdown.setCurrentText('None')
            else:
                aw.update_overlay_colormap('None')
            if getattr(aw, 'z_data_raw', None) is None:
                aw.enable_z_visualization_controls(False)
        except Exception:
            pass

    def sync_settings_from_dialog(self):
        """Pull output settings from the feature deploy dialog when available."""
        if self.feature_dialog is not None:
            try:
                self.output_type = self.feature_dialog.get_output_type()
                self.allow_holes = self.feature_dialog.get_allow_holes()
            except Exception:
                pass

    def deactivate(self):
        """Deactivate, clearing the query, heatmap, prompts, and any work area."""
        self.active = False
        self.hover_timer.stop()
        self.annotation_window.setCursor(self.default_cursor)

        self.cancel_working_area_creation()
        self.cancel_working_area()
        self.annotation_window.clear_feature_overlay()

        # Return the colormap controls to the depth overlay (left hidden).
        self._release_colormap_controls()

        # If we were painting into the mask, drop the rasterization lock.
        if self.output_type == "Mask":
            self.annotation_window.unrasterize_annotations()

        super().deactivate()

    # ==================== Helpers ====================

    def _get_active_raster(self):
        """Resolve the Raster for the currently displayed image."""
        try:
            raster_manager = self.main_window.image_window.raster_manager
            return raster_manager.get_raster(self.annotation_window.current_image_path)
        except Exception:
            return None

    def _get_extractor(self):
        return getattr(self.feature_dialog, 'loaded_model', None) if self.feature_dialog else None

    def _status(self, message, msecs=4000):
        try:
            self.main_window.status_bar.showMessage(message, msecs)
        except Exception:
            pass

    # ==================== Work area ====================

    def set_working_area(self):
        """Use the current view extent as the work area (mirrors SAMTool)."""
        self.annotation_window.setCursor(Qt.WaitCursor)
        self.cancel_working_area()

        self.image_path = self.annotation_window.current_image_path
        self.original_image = pixmap_to_numpy(self.annotation_window.pixmap_image)
        self.original_width = self.annotation_window.pixmap_image.size().width()
        self.original_height = self.annotation_window.pixmap_image.size().height()

        extent = self.annotation_window.viewportToScene()
        top = max(0, round(extent.top()))
        left = max(0, round(extent.left()))
        width = round(extent.width())
        height = round(extent.height())
        bottom = min(self.original_height, top + height)
        right = min(self.original_width, left + width)

        self._setup_working_area(left, top, right, bottom)

    def set_custom_working_area(self, start_point, end_point):
        """Create a work area from two user-picked corners (mirrors SAMTool)."""
        self.annotation_window.setCursor(Qt.WaitCursor)
        self.cancel_working_area()

        self.image_path = self.annotation_window.current_image_path
        self.original_image = pixmap_to_numpy(self.annotation_window.pixmap_image)
        self.original_width = self.annotation_window.pixmap_image.size().width()
        self.original_height = self.annotation_window.pixmap_image.size().height()

        left = max(0, int(min(start_point.x(), end_point.x())))
        top = max(0, int(min(start_point.y(), end_point.y())))
        right = min(self.original_width, int(max(start_point.x(), end_point.x())))
        bottom = min(self.original_height, int(max(start_point.y(), end_point.y())))

        # Ensure a minimum size (mirrors SAMTool).
        if right - left < 10:
            right = min(left + 10, self.original_width)
        if bottom - top < 10:
            bottom = min(top + 10, self.original_height)

        self._setup_working_area(left, top, right, bottom)

    def _setup_working_area(self, left, top, right, bottom):
        """Build the work area, extract its crop features, and arm the query."""
        extractor = self._get_extractor()
        if extractor is None or not getattr(extractor, 'supports_dense', False):
            self._status("Feature Select: deploy a dense feature model first.")
            self.annotation_window.setCursor(self.cursor)
            return

        # Create the work-area graphics (identical to SAMTool).
        self.working_area = WorkArea(left, top, right - left, bottom - top, self.image_path)
        self.working_area.create_graphics(self.annotation_window.scene,
                                           include_shadow=True,
                                           image_rect=self.annotation_window.get_image_rect())
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)

        # Extract dense features for just the crop, build the local query engine.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._status("Feature Select: extracting work-area features…", 0)
        try:
            crop = self.original_image[top:bottom, left:right]
            crop_fmap = extractor.extract_dense(crop)
            if crop_fmap is None or crop_fmap.size == 0:
                self._status("Feature Select: feature extraction returned nothing.")
                self.cancel_working_area()
                return
            self._build_query_engine(crop_fmap)
            self._persist_to_full_map(crop_fmap, left, top, right, bottom, extractor)
            # The query is now armed: hand the colormap controls to the feature
            # overlay (defaults to Plasma, hides the Z-channel). Deferred to here
            # so selecting the tool button alone doesn't change the dropdown.
            self._engage_colormap_controls()
            self._status("Feature Select: Ctrl+click patches to query similarity, "
                         "Space to finalize.", 5000)
        except Exception as e:
            self._status(f"Feature Select: feature extraction failed ({e}).")
            self.cancel_working_area()
        finally:
            QApplication.restoreOverrideCursor()
            self.annotation_window.setCursor(self.cursor)
            self.annotation_window.scene.update()

    def _build_query_engine(self, crop_fmap):
        """Construct a QueryEngine over the [h, w, C] crop feature map."""
        from coralnet_toolbox.Features.QueryEngine import QueryEngine

        self.feat_h, self.feat_w = int(crop_fmap.shape[0]), int(crop_fmap.shape[1])
        features = np.asarray(crop_fmap).reshape(-1, crop_fmap.shape[2])
        valid = np.ones(features.shape[0], dtype=bool)
        self.query_engine = QueryEngine(features, valid)
        self.positive_ids = []
        self.negative_ids = []
        self.threshold_active = False

    def _persist_to_full_map(self, crop_fmap, left, top, right, bottom, extractor):
        """Best-effort: paste the crop features into the raster's full feature map.

        This keeps a persistent, disk-cached map for the image (the live query
        still runs on the local crop grid, so this never affects alignment). The
        crop's square grid is nearest-resampled onto the full map's grid region.
        """
        try:
            raster = self._get_active_raster()
            if raster is None:
                return
            from coralnet_toolbox.Features.FeatureMapCodec import save_feature_map
            import os

            stride = extractor.patch_stride or 16
            dim = extractor.out_channels

            if raster.has_feature_map() and raster.feature_map_stride:
                full = np.array(raster.feature_map, dtype=np.float16, copy=True)
                fh, fw = full.shape[0], full.shape[1]
            else:
                # No full map yet — derive a grid from the stored stride.
                fh = max(1, round(self.original_height / stride))
                fw = max(1, round(self.original_width / stride))
                full = np.zeros((fh, fw, dim), dtype=np.float16)

            gx0 = max(0, int(left / self.original_width * fw))
            gy0 = max(0, int(top / self.original_height * fh))
            gx1 = min(fw, int(np.ceil(right / self.original_width * fw)))
            gy1 = min(fh, int(np.ceil(bottom / self.original_height * fh)))
            out_h, out_w = gy1 - gy0, gx1 - gx0
            if out_h <= 0 or out_w <= 0:
                return

            resampled = self._resample_grid(np.asarray(crop_fmap), out_h, out_w)
            norms = np.linalg.norm(resampled, axis=2, keepdims=True)
            norms[norms == 0] = 1.0
            resampled = (resampled / norms).astype(np.float16)
            full[gy0:gy1, gx0:gx1, :] = resampled

            # Persist the feature map under a project-local cache beside the image.
            cache_dir = os.path.join(os.path.dirname(raster.image_path), ".cache", "features")
            basename = os.path.splitext(os.path.basename(raster.image_path))[0]
            npy_path = os.path.join(cache_dir, f"{basename}_features.npy")
            save_feature_map(npy_path, full, model_id=extractor.model_id,
                             stride=stride, dim=dim,
                             upsampler=getattr(extractor, "upsample_descriptor", None))
            raster.add_feature_map(None, model_id=extractor.model_id, stride=stride,
                                   dim=dim, path=npy_path)
        except Exception:
            pass  # Persistence is best-effort; never break the live query.

    @staticmethod
    def _resample_grid(src, out_h, out_w):
        """Nearest-neighbor resample an [sh, sw, C] grid to [out_h, out_w, C]."""
        sh, sw = src.shape[:2]
        ys = np.clip((np.arange(out_h) * sh / out_h).astype(int), 0, sh - 1)
        xs = np.clip((np.arange(out_w) * sw / out_w).astype(int), 0, sw - 1)
        return src[ys][:, xs]

    def display_working_area_preview(self, current_pos):
        """Preview rectangle while creating a custom work area (SAMTool style)."""
        if self.working_area_start is None:
            return
        if self.working_area_temp_graphics is not None:
            self.annotation_window.scene.removeItem(self.working_area_temp_graphics)
            self.working_area_temp_graphics = None
        rect = QRectF(
            min(self.working_area_start.x(), current_pos.x()),
            min(self.working_area_start.y(), current_pos.y()),
            abs(current_pos.x() - self.working_area_start.x()),
            abs(current_pos.y() - self.working_area_start.y()),
        )
        pen = QPen(QColor(0, 168, 230))
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(3)
        self.working_area_temp_graphics = QGraphicsRectItem(rect)
        self.working_area_temp_graphics.setPen(pen)
        self.working_area_temp_graphics.setBrush(QBrush(QColor(0, 168, 230, 30)))
        self.annotation_window.scene.addItem(self.working_area_temp_graphics)

    def cancel_working_area_creation(self):
        self.creating_working_area = False
        self.working_area_start = None
        if self.working_area_temp_graphics is not None:
            self.annotation_window.scene.removeItem(self.working_area_temp_graphics)
            self.working_area_temp_graphics = None
        self.annotation_window.scene.update()

    def on_working_area_removed(self, work_area):
        self.cancel_working_area()

    def cancel_working_area(self):
        """Tear down the work area, query, prompts, and heatmap."""
        self.clear_prompts()
        self.annotation_window.clear_feature_overlay()
        if self.working_area is not None:
            try:
                self.working_area.remove_from_scene()
            except Exception:
                pass
            self.working_area = None
        self.query_engine = None
        self.feat_h = self.feat_w = None
        self.annotation_window.scene.update()

    # ==================== Grid mapping ====================

    def pixel_to_cell(self, x, y):
        """Map an image pixel (x, y) to a flat crop-grid element id, or None.

        The mapping is proportional to the work-area rect (NOT a single stride):
        the extractor resizes the crop to a square before patchifying, so the
        grid's aspect ratio need not match the crop's. Using the rect keeps the
        cursor and the heatmap (rendered to the same rect) aligned.
        """
        if self.working_area is None or self.feat_w is None:
            return None
        wa = self.working_area.rect
        rx = x - wa.left()
        ry = y - wa.top()
        if rx < 0 or ry < 0 or rx >= wa.width() or ry >= wa.height():
            return None
        gx = int(rx / wa.width() * self.feat_w)
        gy = int(ry / wa.height() * self.feat_h)
        gx = max(0, min(gx, self.feat_w - 1))
        gy = max(0, min(gy, self.feat_h - 1))
        return gy * self.feat_w + gx

    # ==================== Similarity + heatmap ====================

    def _compute_similarity(self, hover_id=None):
        """Rebuild the query from committed prototypes (+ optional hover) and score."""
        if self.query_engine is None:
            return None
        qe = self.query_engine
        qe.clear()
        for pid in self.positive_ids:
            qe.add_positive(pid)
        if hover_id is not None:
            qe.add_positive(hover_id)
        for nid in self.negative_ids:
            qe.add_negative(nid)
        if not qe.positive_ids and not qe.negative_ids:
            return None
        return qe.similarity()

    def update_heatmap(self, hover_id=None):
        """Recompute similarity and refresh the work-area overlay (indexed)."""
        if self.query_engine is None or self.working_area is None:
            return
        sim = self._compute_similarity(hover_id=hover_id)
        if sim is None:
            self.annotation_window.clear_feature_overlay()
            return
        idx = self._build_index_field(sim)
        if idx is None:
            self.annotation_window.clear_feature_overlay()
            return
        # Colormap + opacity are owned by the shared overlay controls (engaged in
        # _setup_working_area); we only hand over the uint8 index field + rect.
        self.annotation_window.set_feature_overlay(idx, rect=self.working_area.rect)

    def _build_index_field(self, sim):
        """Build a uint8 palette-index field from sim [N].

        Palette: 0 = transparent (invalid), 1..254 = colormap ramp (normalized
        similarity), 255 = below-threshold scrim. The scalar field is bilinearly
        upsampled to (a capped fraction of) the work-area pixel resolution BEFORE
        indexing/thresholding, so the thresholded scrim edge follows the same
        smooth contour the commit path produces — instead of the coarse per-patch
        grid steps. The render size is capped (PREVIEW_MAX_EDGE) so the per-hover
        rebuild stays cheap. Recoloring/opacity are now free table swaps on the
        ColorMapOverlay, so this no longer bakes RGBA.
        """
        grid = np.asarray(sim, dtype=np.float32).reshape(self.feat_h, self.feat_w)
        finite = np.isfinite(grid)
        if not finite.any():
            return None

        # Normalization uses the committed grid's finite range, so the gradient is
        # stable regardless of the preview render resolution.
        vals = grid[finite]
        vmin, vmax = float(vals.min()), float(vals.max())

        # Upsample the scalar field (masked cells pinned below threshold) to match
        # the commit's bilinear-then-threshold behavior.
        safe = np.where(finite, grid, -1.0e9).astype(np.float32)
        wa = self.working_area.rect
        out_h, out_w = self._preview_dims(wa.height(), wa.width())
        up = self._upsample_similarity(safe, out_h, out_w)

        if vmax > vmin:
            norm = np.clip((up - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            norm = np.ones_like(up)

        # 1..254 colormap ramp (index 0 and 255 are reserved by the overlay).
        idx = (1 + np.clip(norm * 253.0, 0.0, 253.0)).astype(np.uint8)

        # Regions interpolated from masked cells stay transparent.
        invalid = up < -1.0e8
        idx[invalid] = 0
        if self.threshold_active:
            # Below-threshold (but valid) cells map to the scrim index so the
            # filtered-out region reads as dimmed rather than clearly exposed.
            # Thresholding the upsampled field gives a smooth edge matching the
            # committed polygon/mask.
            below = (~invalid) & (up < self.threshold)
            idx[below] = 255
        return idx

    # Cap the preview render resolution (long edge, px) so the per-hover RGBA
    # rebuild stays cheap even for large work areas. The canvas scales whatever
    # we hand it up to the work-area rect, so this only affects edge sharpness.
    PREVIEW_MAX_EDGE = 768

    def _preview_dims(self, wa_h, wa_w):
        """Work-area pixel size, scaled down so the long edge ≤ PREVIEW_MAX_EDGE."""
        wa_h = max(1, int(round(wa_h)))
        wa_w = max(1, int(round(wa_w)))
        long_edge = max(wa_h, wa_w)
        if long_edge <= self.PREVIEW_MAX_EDGE:
            return wa_h, wa_w
        scale = self.PREVIEW_MAX_EDGE / float(long_edge)
        return max(1, int(wa_h * scale)), max(1, int(wa_w * scale))

    # ==================== Mouse / keyboard ====================

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            self._status("A label must be selected before adding an annotation.")
            return

        scene_pos = self.annotation_window.mapToScene(event.pos())

        # Work-area creation (plain left clicks), mirroring SAMTool.
        if self.working_area is None and event.button() == Qt.LeftButton:
            if not self.creating_working_area:
                self.creating_working_area = True
                self.working_area_start = scene_pos
                return
            elif self.working_area_start is not None:
                self.set_custom_working_area(self.working_area_start, scene_pos)
                self.cancel_working_area_creation()
                return

        if self.working_area is None:
            return
        if not self.working_area.contains_point(scene_pos):
            return

        # Ctrl+click → positive / negative prototype.
        if event.modifiers() & Qt.ControlModifier:
            element_id = self.pixel_to_cell(scene_pos.x(), scene_pos.y())
            if element_id is None:
                return
            if event.button() == Qt.LeftButton:
                self.positive_ids.append(element_id)
                self._add_point_graphic(scene_pos, Qt.green)
            elif event.button() == Qt.RightButton:
                self.negative_ids.append(element_id)
                self._add_point_graphic(scene_pos, Qt.red)
            self.update_heatmap()
            return

        self.annotation_window.scene.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)  # crosshair
        scene_pos = self.annotation_window.mapToScene(event.pos())
        self.hover_pos = scene_pos

        if self.creating_working_area and self.working_area_start is not None:
            self.display_working_area_preview(scene_pos)
            return

        if self.working_area is None:
            return

        # Live hover preview of similarity to the patch under the cursor.
        if (self.query_engine is not None
                and self.annotation_window.cursorInWindow(event.pos())):
            self.hover_timer.start(self.debounce_ms)

    def _on_hover_timeout(self):
        if not self.active or self.query_engine is None or self.hover_pos is None:
            return
        if self.creating_working_area:
            return
        hover_id = self.pixel_to_cell(self.hover_pos.x(), self.hover_pos.y())
        self.update_heatmap(hover_id=hover_id)

    def wheelEvent(self, event: QMouseEvent):
        """Ctrl+wheel adjusts the selection threshold (live thresholded preview).

        The annotation window only forwards wheel events here when Ctrl is held.
        """
        if self.query_engine is None:
            return
        step = 0.02
        if event.angleDelta().y() > 0:
            self.threshold = min(1.0, self.threshold + step)
        else:
            self.threshold = max(0.0, self.threshold - step)
        self.threshold_active = True
        self.update_heatmap()
        self._status(f"Feature Select threshold: {self.threshold:.2f}", 2000)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            if self.creating_working_area and self.working_area_start and self.hover_pos:
                self.set_custom_working_area(self.working_area_start, self.hover_pos)
                self.cancel_working_area_creation()
            elif self.working_area is None:
                self.set_working_area()
            elif self.positive_ids or self.negative_ids:
                self.commit_selection()
            else:
                self.cancel_working_area()
            self.annotation_window.scene.update()
        elif event.key() == Qt.Key_Backspace:
            if self.creating_working_area:
                self.cancel_working_area_creation()
            elif self.positive_ids or self.negative_ids:
                self.clear_prompts()
                self.annotation_window.clear_feature_overlay()
            else:
                self.cancel_working_area()
            self.annotation_window.scene.update()

    def _add_point_graphic(self, scene_pos, color):
        """Positive/negative point dot, identical to SAMTool's points."""
        point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
        pen = QPen(color)
        pen.setCosmetic(True)
        point.setPen(pen)
        point.setBrush(QColor(color))
        self.annotation_window.scene.addItem(point)
        self.point_graphics.append(point)

    # ==================== Commit ====================

    def commit_selection(self):
        """Turn the thresholded selection into a Polygon or Mask annotation."""
        if self.query_engine is None or not self.annotation_window.selected_label:
            if not self.annotation_window.selected_label:
                self._status("A label must be selected before committing a selection.")
            return

        sim = self._compute_similarity(hover_id=None)
        if sim is None:
            return

        wa = self.working_area.rect
        wa_left, wa_top = int(wa.left()), int(wa.top())
        wa_w, wa_h = int(round(wa.width())), int(round(wa.height()))

        # Upsample the CONTINUOUS similarity field to full work-area resolution
        # with bilinear interpolation, THEN threshold. Thresholding first on the
        # coarse feature grid and nearest-upscaling the binary mask is what made
        # boundaries blocky; interpolating the scalar field first yields a smooth
        # contour that matches the (smoothly scaled) heatmap preview. The achievable
        # detail is still bounded by the feature-grid density — raise Input
        # Resolution / AnyUp in the Features dialog for genuinely finer features.
        grid = np.asarray(sim, dtype=np.float32).reshape(self.feat_h, self.feat_w)
        # Keep any masked/non-finite cells safely below threshold across the interp.
        grid = np.where(np.isfinite(grid), grid, -1.0e9).astype(np.float32)

        crop_mask = self._upsample_similarity(grid, wa_h, wa_w) >= self.threshold
        crop_mask = crop_mask.astype(np.uint8)
        if not crop_mask.any():
            self._status("Feature Select: nothing above threshold to commit.")
            return

        full_mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
        y1 = min(self.original_height, wa_top + wa_h)
        x1 = min(self.original_width, wa_left + wa_w)
        full_mask[wa_top:y1, wa_left:x1] = crop_mask[: y1 - wa_top, : x1 - wa_left]

        self.sync_settings_from_dialog()
        if self.output_type == "Mask":
            self._commit_as_mask(full_mask)
            # Multi-Annotate: propagate the selection to the context cameras, just
            # like SAMTool does for its final mask (Mask output only).
            self._propagate_to_cameras(crop_mask, wa_left, wa_top, wa_w, wa_h)
        else:
            self._commit_as_polygon(full_mask)

        # Clear prompts but keep the work area for further queries.
        self.clear_prompts()
        self.annotation_window.clear_feature_overlay()

    @staticmethod
    def _upsample_similarity(grid, out_h, out_w):
        """Bilinearly upsample a [gh, gw] float grid to [out_h, out_w].

        Used to smooth the per-patch similarity field before thresholding, so the
        committed boundary follows a smooth contour rather than the grid steps.
        Falls back to a numpy linear interpolation if OpenCV is unavailable.
        """
        out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
        try:
            import cv2
            return cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            gh, gw = grid.shape
            ys = np.linspace(0, gh - 1, out_h)
            xs = np.linspace(0, gw - 1, out_w)
            y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
            y1 = np.clip(y0 + 1, 0, gh - 1)
            x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
            x1 = np.clip(x0 + 1, 0, gw - 1)
            wy = (ys - y0)[:, None]
            wx = (xs - x0)[None, :]
            top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
            bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
            return top * (1 - wy) + bot * wy

    def _commit_as_polygon(self, full_mask):
        """Polygonize the binary mask and add Polygon annotation(s)."""
        import torch

        mask_tensor = torch.from_numpy(full_mask.astype(np.uint8))
        exterior_coords, holes_coords_list = polygonize_mask_with_holes(mask_tensor)
        if not exterior_coords or len(exterior_coords) < 3:
            self._status("Feature Select: selection too small to polygonize.")
            return

        points = [QPointF(p[0], p[1]) for p in exterior_coords]
        holes = []
        if self.allow_holes:
            for hole in holes_coords_list:
                if len(hole) >= 3:
                    holes.append([QPointF(p[0], p[1]) for p in hole])

        annotation = PolygonAnnotation(
            points=points,
            holes=holes,
            label=self.annotation_window.selected_label,
            image_path=self.annotation_window.current_image_path,
            transparency=self.main_window.get_transparency_value(),
            show_confidence=False,
        )
        if hasattr(self.annotation_window, 'rasterio_image') and self.annotation_window.rasterio_image:
            annotation.create_cropped_image(self.annotation_window.rasterio_image)
        annotation.create_graphics_item(self.annotation_window.scene)
        self.annotation_window.add_annotation_from_tool(annotation)

    def _commit_as_mask(self, full_mask):
        """Paint the selection into the raster MaskAnnotation."""
        if self.annotation_window.current_mask_annotation is None:
            self.annotation_window.rasterize_annotations()
        mask_annotation = self.annotation_window.current_mask_annotation
        if mask_annotation is None:
            self._status("Feature Select: no mask annotation available for the image.")
            return

        label = self.annotation_window.selected_label
        class_id = mask_annotation.label_id_to_class_id_map.get(label.id)
        if class_id is None:
            self._status("Feature Select: active label is not in the mask's label map.")
            return

        prediction_mask = (full_mask.astype(np.uint8) * class_id).astype(np.uint8)
        history_action = MaskEditAction(mask_annotation, description="Feature Select prediction")
        mask_annotation.update_mask_with_prediction_mask(
            prediction_mask,
            history_action=history_action,
        )
        if not history_action.is_empty():
            self.annotation_window.action_stack.push(history_action)

    def _propagate_to_cameras(self, crop_mask, wa_left, wa_top, wa_w, wa_h):
        """Multi-Annotate: hand the selection to the PropagationEngine.

        Mirrors SAMTool: send a binary crop anchored at its centre plus the active
        label id. The engine maps the crop through the source camera's index map
        and paints the matching elements on every visible context camera.
        """
        if self.post_prediction_callback is None:
            return
        label = self.annotation_window.selected_label
        if label is None:
            return
        anchor = QPointF(wa_left + wa_w / 2.0, wa_top + wa_h / 2.0)
        try:
            self.post_prediction_callback(
                anchor, label.id, np.ascontiguousarray(crop_mask.astype(np.uint8))
            )
        except Exception as e:
            print(f"[FeatureSelectTool] propagation failed: {e}")

    # ==================== Cleanup ====================

    def clear_prompts(self):
        """Clear prompts, prototype graphics, and reset the threshold view."""
        self.positive_ids = []
        self.negative_ids = []
        for item in self.point_graphics:
            try:
                self.annotation_window.scene.removeItem(item)
            except Exception:
                pass
        self.point_graphics = []
        self.threshold_active = False

        if self.query_engine is not None:
            self.query_engine.clear()

    def stop_current_drawing(self):
        """Called by the annotation window when switching tools mid-action."""
        self.cancel_working_area_creation()
