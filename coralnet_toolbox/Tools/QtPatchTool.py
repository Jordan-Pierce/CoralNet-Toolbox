import warnings

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QMouseEvent, QBrush, QPen, QColor
from PyQt5.QtWidgets import QGraphicsPathItem

from rasterio.windows import Window as RasterioWindow

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtAnnotation import RenderMode

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor  # Explicitly set, if needed

        self.live_classify_mode = False
        self._live_classify_timer = QTimer()
        self._live_classify_timer.setSingleShot(True)
        self._live_classify_timer.setInterval(75)
        self._live_classify_timer.timeout.connect(self._run_live_inference)
        self._pending_inference_pos = None
        # Last predictions ({Label: conf}) so the preview keeps showing the
        # prediction while the cursor moves, instead of flashing back to the
        # selected label between inference ticks.
        self._last_prediction = None

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        if self.live_classify_mode:
            self._exit_live_classify()
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        # Call parent deactivate to ensure crosshair and cursor preview are properly cleared
        super().deactivate()

    def mousePressEvent(self, event: QMouseEvent):

        if not self.annotation_window.selected_label:
            self.annotation_window.main_window.status_bar.showMessage(
                "A label must be selected before adding an annotation.", 4000)
            return None
        
        # Add cursor bounds check
        if not self.annotation_window.cursorInWindow(event.pos()):
            return None

        if event.button() == Qt.LeftButton:
            self.annotation_window.unselect_annotations()

            # Create a new annotation at the clicked position
            annotation = self.create_annotation(self.annotation_window.mapToScene(event.pos()), finished=True)

            # Transfer live classification predictions to the finalized annotation
            if (self.live_classify_mode and
                    self.cursor_annotation and
                    self.cursor_annotation.machine_confidence):
                annotation.update_machine_confidence(
                    dict(self.cursor_annotation.machine_confidence))

            self.annotation_window.add_annotation_from_tool(annotation)

            # After adding annotation, restore cursor annotation
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.live_classify_mode:
                self._update_live_cursor(scene_pos)
            else:
                self.update_cursor_annotation(scene_pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Call parent implementation to handle crosshair
        super().mouseMoveEvent(event)

        # Bail out (and clear) when we can't draw a cursor annotation
        if not (self.annotation_window.active_image and self.annotation_window.selected_label) \
                or not self.annotation_window.cursorInWindow(event.pos()):
            self.clear_cursor_annotation()
            if self.cursor_clear_callback:
                self.cursor_clear_callback()
            return

        scene_pos = self.annotation_window.mapToScene(event.pos())

        if self.live_classify_mode:
            # Keep a persistent cursor annotation: move it in place so the last
            # prediction stays visible, then schedule a fresh inference.
            self._update_live_cursor(scene_pos)
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
            self._schedule_inference(scene_pos)
        else:
            # Default behaviour: recreate the cursor annotation each move.
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.annotation_window.active_image and self.annotation_window.selected_label:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.annotation_window.cursorInWindow(event.pos()):
                if self.live_classify_mode:
                    self._update_live_cursor(scene_pos)
                else:
                    self.create_cursor_annotation(scene_pos)

    def wheelEvent(self, event: QMouseEvent):
        # Handle Zoom wheel for setting annotation size
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.annotation_window.set_annotation_size(delta=16)  # Zoom in
            else:
                self.annotation_window.set_annotation_size(delta=-16)  # Zoom out

            # Update the cursor annotation with the new size
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.live_classify_mode:
                # Size changed -> rebuild the live cursor and re-classify
                self._update_live_cursor(scene_pos)
                self._schedule_inference(scene_pos)
            else:
                self.update_cursor_annotation(scene_pos)
            
    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        annotation = PatchAnnotation(
            scene_pos,
            self.annotation_window.annotation_size,
            self.annotation_window.selected_label,
            self.annotation_window.current_image_path,
            transparency=self.annotation_window.main_window.get_transparency_value(),
            show_confidence=False,
        )
        return annotation

    def create_cursor_preview_item(self, u: float, v: float):
        """Return a styled patch square QGraphicsItem centred at image pixel (u, v)."""
        if not self.annotation_window.selected_label:
            return None

        label = self.annotation_window.selected_label
        size = self.annotation_window.annotation_size
        transparency = self.annotation_window.main_window.get_transparency_value()
        ann = PatchAnnotation(
            QPointF(u, v),
            size,
            label,
            "",
            transparency,
            show_confidence=False,
        )
        path = ann.get_painter_path()
        item = QGraphicsPathItem(path)
        c = QColor(label.color)
        fill = QColor(c)
        fill.setAlpha(transparency)
        item.setBrush(QBrush(fill))
        pen = QPen(c, 1)
        pen.setCosmetic(True)
        item.setPen(pen)

        return item

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a patch cursor annotation at the given position."""
        if not scene_pos or not self.annotation_window.selected_label or not self.annotation_window.active_image:
            self.clear_cursor_annotation()
            return
            
        # First ensure any existing cursor annotation is removed
        self.clear_cursor_annotation()
        
        # Create a new cursor annotation with semi-transparent appearance
        self.cursor_annotation = self.create_annotation(scene_pos)
        if self.cursor_annotation:
            # Make the cursor annotation semi-transparent to distinguish it from actual annotations
            self.cursor_annotation.update_transparency(self.annotation_window.main_window.get_transparency_value())
            # Force hydrate the cursor preview so it follows the mouse smoothly
            self.cursor_annotation.create_graphics_item(self.annotation_window.scene, force_hydrate=True)
            # Show the dimension tag while drawing
            if hasattr(self.cursor_annotation, 'dimension_tag_item') and self.cursor_annotation.dimension_tag_item:
                self.cursor_annotation.dimension_tag_item.setVisible(True)
            
            # Track current size for optimization
            self._last_annotation_size = self.annotation_window.annotation_size

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        if scene_pos is None:
            self.clear_cursor_annotation()
            return

        # If cursor annotation exists and size hasn't changed, just update position
        if (self.cursor_annotation and
            hasattr(self, '_last_annotation_size') and self._last_annotation_size == self.annotation_window.annotation_size):
            # Move the existing annotation to new position
            self.cursor_annotation.update_location(scene_pos)
        else:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)

    def toggle_live_classify(self):
        """Toggle live classification mode on/off."""
        main_window = self.annotation_window.main_window
        classify_dialog = main_window.classify_deploy_model_dialog

        if classify_dialog.loaded_model is None:
            main_window.status_bar.showMessage(
                "No classification model loaded. Load one first.", 4000)
            return

        if self.live_classify_mode:
            self._exit_live_classify()
            main_window.status_bar.showMessage(
                "Live classification mode disabled.", 4000)
        else:
            self.live_classify_mode = True
            self._last_prediction = None
            # Drop any plain cursor so the next move rebuilds a hydrated live one.
            self.clear_cursor_annotation()
            main_window.status_bar.showMessage(
                "Live classification mode enabled (Ctrl+1 to toggle).", 4000)

    def _exit_live_classify(self):
        """Clean up live classification state."""
        self.live_classify_mode = False
        self._live_classify_timer.stop()
        self._pending_inference_pos = None
        self._last_prediction = None
        self.clear_cursor_annotation()

    # ------------------------------------------------------------------
    # Live cursor rendering
    # ------------------------------------------------------------------

    def _update_live_cursor(self, scene_pos: QPointF):
        """Create or move the persistent live-classify cursor annotation.

        Unlike the default tool behaviour (which destroys and recreates the
        cursor annotation each move), this keeps a single annotation alive so
        the last prediction stays on screen while the cursor moves. The
        annotation is marked selected/FULL so every re-render hydrates its tag
        (label + confidence) instead of collapsing to an invisible phantom.
        """
        size = self.annotation_window.annotation_size

        if self.cursor_annotation and getattr(self, '_last_annotation_size', None) == size:
            # Move geometry only, preserving machine_confidence / label.
            self.cursor_annotation.set_precision(scene_pos)
            self.cursor_annotation.set_cropped_bbox()
            self.cursor_annotation.update_graphics_item()
            if getattr(self.cursor_annotation, 'dimension_tag_item', None):
                self.cursor_annotation.dimension_tag_item.setVisible(True)
            return

        # Size changed (or first hover): build a fresh hydrated cursor.
        self.clear_cursor_annotation()
        ann = self.create_annotation(scene_pos)
        if not ann:
            return
        # Force the annotation to always hydrate its UI on re-render.
        ann.is_selected = True
        ann.render_mode = RenderMode.FULL
        ann.update_transparency(self.annotation_window.main_window.get_transparency_value())
        ann.create_graphics_item(self.annotation_window.scene, force_hydrate=True)
        if getattr(ann, 'dimension_tag_item', None):
            ann.dimension_tag_item.setVisible(True)
        self.cursor_annotation = ann
        self._last_annotation_size = size

        # Re-apply the most recent prediction so the rebuilt cursor doesn't
        # flash back to the selected label before the next inference lands.
        if self._last_prediction:
            self._apply_prediction_to_cursor(self._last_prediction)

    def _apply_prediction_to_cursor(self, predictions: dict):
        """Apply a {Label: confidence} prediction to the live cursor annotation.

        Sets the prediction fields directly (avoiding update_machine_confidence's
        unconditional re-render path) then re-renders once. The cursor stays
        visible because it is flagged selected/FULL.
        """
        ann = self.cursor_annotation
        if not ann or not predictions:
            return
        ordered = dict(sorted(predictions.items(), key=lambda kv: kv[1], reverse=True))
        ann.machine_confidence = ordered
        ann.user_confidence = {}
        ann.show_confidence = True
        ann.verified = False
        ann.label = next(iter(ordered))
        ann.update_graphics_item()

    # ------------------------------------------------------------------
    # Live inference
    # ------------------------------------------------------------------

    def _schedule_inference(self, scene_pos: QPointF):
        """Queue a debounced inference at the given position."""
        self._pending_inference_pos = scene_pos
        if not self._live_classify_timer.isActive():
            self._live_classify_timer.start()

    def _extract_patch(self, src, scene_pos: QPointF):
        """Read the patch under the cursor from rasterio as an RGB uint8 array.

        Kept on the main thread (rasterio is not thread-safe). Mirrors the
        windowed-read logic used by Classify.predict so live and batch
        predictions see the same pixels.
        """
        annotation_size = self.annotation_window.annotation_size
        half = annotation_size / 2
        col_off = max(0, int(scene_pos.x() - half))
        row_off = max(0, int(scene_pos.y() - half))
        width = min(src.width - col_off, annotation_size)
        height = min(src.height - row_off, annotation_size)
        if width <= 0 or height <= 0:
            return None

        window = RasterioWindow(col_off=col_off, row_off=row_off,
                                width=width, height=height)
        if src.count >= 3:
            arr = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
        else:
            band = src.read(1, window=window)
            arr = np.stack([band, band, band], axis=-1)

        if arr.dtype != np.uint8:
            max_val = arr.max()
            if max_val > 0:
                arr = (arr.astype(np.float32) * (255.0 / max_val)).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        return np.ascontiguousarray(arr)

    def _run_live_inference(self):
        """Debounced callback: extract patch at cursor and classify it."""
        if not self.live_classify_mode or not self.cursor_annotation:
            return
        scene_pos = self._pending_inference_pos
        if scene_pos is None:
            return

        main_window = self.annotation_window.main_window
        classify_dialog = main_window.classify_deploy_model_dialog
        if classify_dialog.loaded_model is None:
            return

        src = self.annotation_window.rasterio_image
        if src is None or getattr(src, 'closed', True):
            return

        try:
            arr = self._extract_patch(src, scene_pos)
            if arr is None:
                return

            preds = classify_dialog.predict_patch(arr)
            if not preds:
                return

            # Map model class names -> project Label objects.
            label_window = main_window.label_window
            class_mapping = classify_dialog.class_mapping
            predictions = {}
            for class_name, conf in preds:
                if class_mapping and class_name in class_mapping:
                    short_code = class_mapping[class_name].get('short_label_code', class_name)
                else:
                    short_code = class_name
                label = label_window.get_label_by_short_code(short_code)
                if label:
                    predictions[label] = float(conf)

            if predictions and self.cursor_annotation:
                self._last_prediction = predictions
                self._apply_prediction_to_cursor(predictions)

        except Exception as e:
            print(f"Live classify inference error: {e}")