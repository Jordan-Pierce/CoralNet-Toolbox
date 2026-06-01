"""
SAMTool3D — frameless non-modal SAM tool covering the MVATViewer.

Interaction matches SAMTool in AnnotationWindow exactly:
  - Pulsing dotted blue border (WorkArea-style heartbeat).
  - Shadow darkens outside the image boundary.
  - Hover  -> live debounced prediction (10 ms).
  - Left-click (first)  -> start rectangle.
  - Left-click (second) -> finish rectangle (label-colored dashed rect).
  - Ctrl+Left  -> positive point (green filled circle, no outline).
  - Ctrl+Right -> negative point (red filled circle, no outline).
  - Backspace  -> clear all prompts.
  - Space/Enter with mask  -> accept.
  - Space/Enter with no mask -> cancel (matches SAMTool cancel_working_area).
  - Escape -> cancel.
"""

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import (QColor, QImage, QPainter, QPen, QBrush, QPixmap,
                         QCursor, QKeyEvent, QMouseEvent, QPainterPath)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsRectItem,
                             QGraphicsPathItem, QApplication, QSizePolicy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    h, w, _ = rgb.shape
    img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def _mask_to_qpixmap(binary_mask: np.ndarray, color: QColor) -> QPixmap:
    h, w = binary_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[binary_mask, 0] = color.red()
    rgba[binary_mask, 1] = color.green()
    rgba[binary_mask, 2] = color.blue()
    rgba[binary_mask, 3] = 128
    img = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img)


_WORK_AREA_COLOR = QColor(0, 168, 230)   # same as WorkArea.original_color
_DOT_HALF = 10                            # SAMTool uses 20x20 ellipse (r=10)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class SAMTool3D(QDialog):
    """
    Frameless non-modal dialog covering the MVATViewer for SAM interaction.

    Parameters
    ----------
    viewer       : MVATViewer
    rgb_image    : np.ndarray (H, W, 3) uint8
    index_map    : np.ndarray (H, W) int32
    element_type : str
    sam_dialog   : DeployPredictorDialog
    label        : Label  (initial; re-read live from annotation_window each prediction)
    """

    maskAccepted = pyqtSignal(object)   # np.ndarray bool (H x W)

    def __init__(self, viewer, rgb_image: np.ndarray, index_map: np.ndarray,
                 element_type: str, sam_dialog, label, parent=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(False)

        self._viewer     = viewer
        self._rgb        = rgb_image
        self._index_map  = index_map
        self._elem_type  = element_type
        self._sam        = sam_dialog
        self._label      = label
        self._img_h, self._img_w = rgb_image.shape[:2]

        self._binary_mask        = None
        self._positive_pts       = []
        self._negative_pts       = []
        self._rect_start         = None   # first click for rectangle
        self._rect_end           = None
        self._drawing_rect       = False  # waiting for second click
        self._has_active_prompts = False
        self._hover_pos          = None

        self._dot_items    = []
        self._rect_item    = None
        self._overlay_item = None
        self._border_item  = None   # pulsing dotted border
        self._shadow_item  = None   # darkens outside image

        # Hover debounce — 10 ms, same as SAMTool
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._on_hover_timeout)

        # Border pulse — heartbeat like WorkArea.tick_animation
        self._pulse_value     = 0.0
        self._pulse_direction = 1
        self._pulse_timer     = QTimer(self)
        self._pulse_timer.setInterval(50)           # 20 Hz
        self._pulse_timer.timeout.connect(self._tick_pulse)

        self._build_ui()
        self._size_to_viewer()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene = QGraphicsScene(self)
        self._view  = _PromptView(self._scene, self)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setDragMode(QGraphicsView.NoDrag)
        self._view.setStyleSheet("border: 0px; background: black;")
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view.setCursor(QCursor(Qt.CrossCursor))
        self._view.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(self._view)

        # Background screenshot
        pix = _numpy_to_qpixmap(self._rgb)
        self._bg_item = QGraphicsPixmapItem(pix)
        self._bg_item.setZValue(0)
        self._scene.addItem(self._bg_item)
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)

        self._add_work_area_graphics()

    def _add_work_area_graphics(self):
        """Shadow path + pulsing dotted border — mirrors WorkArea.create_graphics(include_shadow=True)."""
        img_rect  = QRectF(0, 0, self._img_w, self._img_h)

        # Shadow: darkens the image area (no hole needed — the whole image IS the work area)
        shadow_brush = QBrush(QColor(0, 0, 0, 150))
        shadow_path  = QPainterPath()
        shadow_path.setFillRule(Qt.OddEvenFill)
        shadow_path.addRect(img_rect)   # outer
        shadow_path.addRect(img_rect)   # inner hole (same rect = fully transparent interior)
        # Actually: work area covers the full image, so shadow should appear outside.
        # Since there is nothing outside the image in our scene, just skip the shadow.
        # Keep a very light vignette instead so the border stands out.
        vignette = QGraphicsRectItem(img_rect)
        vignette.setPen(QPen(Qt.NoPen))
        vignette.setBrush(QBrush(QColor(0, 0, 0, 30)))
        vignette.setZValue(1)
        self._shadow_item = vignette
        self._scene.addItem(self._shadow_item)

        # Pulsing dotted border
        pen = QPen(_WORK_AREA_COLOR, 3, Qt.DotLine)
        pen.setCosmetic(True)
        self._border_item = QGraphicsRectItem(img_rect)
        self._border_item.setPen(pen)
        self._border_item.setBrush(QBrush(Qt.NoBrush))
        self._border_item.setZValue(30)
        self._scene.addItem(self._border_item)
        self._pulse_timer.start()

    def _tick_pulse(self):
        """Heartbeat pulse — identical to WorkArea.tick_animation."""
        if self._pulse_direction == 1:
            self._pulse_value += 0.30
        else:
            self._pulse_value -= 0.10

        if self._pulse_value >= 1.0:
            self._pulse_value = 1.0
            self._pulse_direction = -1
        elif self._pulse_value <= 0.0:
            self._pulse_value = 0.0
            self._pulse_direction = 1

        if self._border_item is not None:
            percent = 100 + int(self._pulse_value * 80)
            color   = _WORK_AREA_COLOR.lighter(percent)
            pen     = QPen(color, 3, Qt.DotLine)
            pen.setCosmetic(True)
            self._border_item.setPen(pen)

    def _size_to_viewer(self):
        if self._viewer is None:
            return
        tl = self._viewer.mapToGlobal(self._viewer.rect().topLeft())
        self.setGeometry(tl.x(), tl.y(),
                         self._viewer.width(), self._viewer.height())

    def showEvent(self, event):
        super().showEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._view.setFocus()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Hover prediction
    # ------------------------------------------------------------------

    def on_hover_move(self, scene_pos: QPointF):
        self._hover_pos = scene_pos

        # While waiting for second rectangle click, update live preview rect
        if self._drawing_rect and self._rect_start is not None:
            self._rect_end = scene_pos
            self._draw_rect_preview()
            # Also run prediction so mask updates as rect grows
            self._hover_timer.start(10)
            return

        self._hover_timer.start(10)

    def on_hover_leave(self):
        self._hover_timer.stop()
        self._hover_pos = None
        if not self._has_active_prompts:
            self._clear_overlay()
            self._binary_mask = None

    def _on_hover_timeout(self):
        self._predict(hover_pos=self._hover_pos)

    # ------------------------------------------------------------------
    # Prompt handling
    # ------------------------------------------------------------------

    def add_positive(self, scene_pos: QPointF):
        self._positive_pts.append(scene_pos)
        self._has_active_prompts = True
        self._add_dot(scene_pos, Qt.green)
        self._predict(hover_pos=None)

    def add_negative(self, scene_pos: QPointF):
        self._negative_pts.append(scene_pos)
        self._has_active_prompts = True
        self._add_dot(scene_pos, Qt.red)
        self._predict(hover_pos=None)

    def begin_rect(self, scene_pos: QPointF):
        """First left-click — start rectangle, wait for second click."""
        self.cancel_rect()
        self._rect_start   = scene_pos
        self._rect_end     = scene_pos
        self._drawing_rect = True
        self._has_active_prompts = True

    def finish_rect(self, scene_pos: QPointF):
        """Second left-click — finalize rectangle."""
        if not self._drawing_rect:
            return
        self._rect_end     = scene_pos
        self._drawing_rect = False
        self._has_active_prompts = True
        self._draw_rect_preview()
        self._predict(hover_pos=None)

    def cancel_rect(self):
        self._rect_start   = None
        self._rect_end     = None
        self._drawing_rect = False
        self._hover_timer.stop()
        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None
        self._binary_mask = None
        self._clear_overlay()
        self._has_active_prompts = bool(self._positive_pts or self._negative_pts)

    def _clear_prompts(self):
        self._positive_pts.clear()
        self._negative_pts.clear()
        self._rect_start         = None
        self._rect_end           = None
        self._drawing_rect       = False
        self._has_active_prompts = False
        self._binary_mask        = None
        self._hover_timer.stop()
        for item in self._dot_items:
            self._scene.removeItem(item)
        self._dot_items.clear()
        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None
        self._clear_overlay()

    # ------------------------------------------------------------------
    # Current label (re-read live so user can change label while open)
    # ------------------------------------------------------------------

    def _current_label(self):
        try:
            lbl = getattr(self._viewer.mvat_manager.annotation_window,
                          'selected_label', None)
            if lbl is not None:
                return lbl
        except Exception:
            pass
        return self._label

    def _current_label_color(self) -> QColor:
        label = self._current_label()
        for candidate in (
            getattr(label, 'color', None),
            getattr(self._label, 'color', None),
        ):
            if candidate is None:
                continue
            color = QColor(candidate)
            if color.isValid():
                return color
        return QColor(255, 255, 255)

    # ------------------------------------------------------------------
    # SAM prediction
    # ------------------------------------------------------------------

    def _predict(self, hover_pos=None):
        import torch

        positive = [[p.x(), p.y()] for p in self._positive_pts]
        negative = [[p.x(), p.y()] for p in self._negative_pts]

        bbox = None
        if (self._rect_start is not None and self._rect_end is not None
                and not self._drawing_rect):
            x0 = min(self._rect_start.x(), self._rect_end.x())
            y0 = min(self._rect_start.y(), self._rect_end.y())
            x1 = max(self._rect_start.x(), self._rect_end.x())
            y1 = max(self._rect_start.y(), self._rect_end.y())
            if x1 - x0 > 2 and y1 - y0 > 2:
                bbox = [x0, y0, x1, y1]
        elif self._drawing_rect and self._rect_start and self._rect_end:
            # Live preview while dragging second click
            x0 = min(self._rect_start.x(), self._rect_end.x())
            y0 = min(self._rect_start.y(), self._rect_end.y())
            x1 = max(self._rect_start.x(), self._rect_end.x())
            y1 = max(self._rect_start.y(), self._rect_end.y())
            if x1 - x0 > 2 and y1 - y0 > 2:
                bbox = [x0, y0, x1, y1]

        if hover_pos is not None and not self._drawing_rect:
            positive = positive + [[hover_pos.x(), hover_pos.y()]]

        all_pts      = positive + negative
        bboxes_input = [bbox]    if bbox    is not None else None
        points_input = [all_pts] if all_pts              else None
        labels_input = ([[1] * len(positive) + [0] * len(negative)]
                        if all_pts else None)

        if bboxes_input is None and points_input is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = self._sam.predict_from_prompts(
                bbox=bboxes_input,
                points=points_input,
                labels=labels_input,
            )
        except Exception:
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not results:
            return
        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return

        conf = result.boxes.conf
        if isinstance(conf, torch.Tensor):
            top_idx = int(torch.argmax(conf).item())
        else:
            top_idx = int(np.argmax(np.asarray(conf)))

        mask_tensor = result.masks.data[top_idx]
        mask_np = (mask_tensor.cpu().numpy()
                   if isinstance(mask_tensor, torch.Tensor)
                   else np.asarray(mask_tensor))

        mh, mw = mask_np.shape[-2], mask_np.shape[-1]
        if (mh, mw) != (self._img_h, self._img_w):
            import cv2
            mask_np = cv2.resize(mask_np.astype(np.float32),
                                 (self._img_w, self._img_h),
                                 interpolation=cv2.INTER_LINEAR)

        self._binary_mask = (mask_np > 0.5)
        self._update_overlay()

    # ------------------------------------------------------------------
    # Graphics helpers
    # ------------------------------------------------------------------

    def _add_dot(self, pos: QPointF, color):
        """Filled circle, no outline — matches SAMTool exactly (20x20, no white border)."""
        d    = _DOT_HALF * 2
        item = QGraphicsEllipseItem(pos.x() - _DOT_HALF, pos.y() - _DOT_HALF, d, d)
        item.setPen(QPen(Qt.NoPen))
        item.setBrush(QBrush(QColor(color)))
        item.setZValue(20)
        self._scene.addItem(item)
        self._dot_items.append(item)

    def _draw_rect_preview(self):
        """Dashed rect in label color — matches SAMTool.display_rectangle."""
        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None
        if self._rect_start is None or self._rect_end is None:
            return

        color = self._current_label_color()

        rect = QRectF(self._rect_start, self._rect_end).normalized()
        pen  = QPen(color, 2, Qt.DashLine)
        pen.setCosmetic(True)
        self._rect_item = QGraphicsRectItem(rect)
        self._rect_item.setPen(pen)
        self._rect_item.setBrush(QBrush(Qt.transparent))
        self._rect_item.setZValue(10)
        self._scene.addItem(self._rect_item)

    def _update_overlay(self):
        if self._binary_mask is None:
            self._clear_overlay()
            return
        label = self._current_label()
        color = QColor(label.color) if label is not None else QColor(0, 200, 50)
        pix   = _mask_to_qpixmap(self._binary_mask, color)
        if self._overlay_item is None:
            self._overlay_item = QGraphicsPixmapItem()
            self._overlay_item.setZValue(5)
            self._scene.addItem(self._overlay_item)
        self._overlay_item.setPixmap(pix)
        self._overlay_item.setVisible(True)

    def _clear_overlay(self):
        if self._overlay_item is not None:
            self._overlay_item.setVisible(False)

    # ------------------------------------------------------------------
    # Accept / cancel
    # ------------------------------------------------------------------

    def _on_accept(self):
        # Space/Enter with no active prompts or no mask = cancel.
        # This prevents hover-only previews from committing like a real annotation.
        if not self._has_active_prompts or self._binary_mask is None:
            self._on_cancel()
            return
        self._hover_timer.stop()
        self.maskAccepted.emit(self._binary_mask.copy())
        self._clear_prompts()

    def _on_cancel(self):
        self._hover_timer.stop()
        self._pulse_timer.stop()
        self.reject()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        if k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._on_accept()
        elif k == Qt.Key_Backspace:
            if self._has_active_prompts:
                self._clear_prompts()
            else:
                self._on_cancel()
        elif k == Qt.Key_Escape:
            self._on_cancel()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Custom QGraphicsView
# ---------------------------------------------------------------------------

class _PromptView(QGraphicsView):
    def __init__(self, scene, dialog: SAMTool3D):
        super().__init__(scene)
        self._dialog      = dialog
        self._drag_origin = None   # tracks first left-click for rect
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent):
        self._dialog.keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Left = positive point
                self._dialog.add_positive(sp)
            else:
                # Plain left-click: start or finish rectangle
                if not self._dialog._drawing_rect:
                    self._dialog.begin_rect(sp)
                else:
                    self._dialog.finish_rect(sp)
        elif event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier:
                self._dialog.add_negative(sp)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())
        self._dialog.on_hover_move(sp)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        # No drag-to-finish; rectangle uses two clicks, so release does nothing
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._dialog.on_hover_leave()
        super().leaveEvent(event)
