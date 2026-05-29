"""
MVATSAMOverlay -- frameless modal SAM dialog that covers the MVATViewer.

Feels identical to using the SAMTool inside AnnotationWindow:
  - Screenshot fills the window as the background.
  - Hover  -> live debounced SAM prediction (crosshair cursor, 10 ms debounce).
  - Ctrl+Left  -> lock positive point (green dot).
  - Ctrl+Right -> lock negative point (red dot).
  - Left-drag (no Ctrl) -> bounding-box rectangle.
  - Backspace  -> clear all prompts.
  - Enter / Space -> accept current mask and close.
  - Escape     -> cancel.
"""

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import (QColor, QImage, QPainter, QPen, QBrush, QPixmap,
                         QCursor, QKeyEvent, QMouseEvent)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsRectItem,
                             QApplication, QSizePolicy)


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


_POS_COLOR  = QColor(0,   200,  50,  220)
_NEG_COLOR  = QColor(220,  30,  30,  220)
_RECT_COLOR = QColor(0,   168, 230,  200)
_DOT_R      = 6


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class MVATSAMDialog(QDialog):
    """
    Frameless modal dialog covering the MVATViewer for SAM interaction.

    Parameters
    ----------
    viewer       : MVATViewer  -- used to compute screen position/size
    rgb_image    : np.ndarray (H, W, 3) uint8
    index_map    : np.ndarray (H, W) int32
    element_type : str
    sam_dialog   : DeployPredictorDialog
    label        : Label
    """

    maskAccepted = pyqtSignal(object)   # np.ndarray bool (H x W)

    def __init__(self, viewer, rgb_image: np.ndarray, index_map: np.ndarray,
                 element_type: str, sam_dialog, label, parent=None):
        super().__init__(parent,
                         Qt.FramelessWindowHint |
                         Qt.WindowStaysOnTopHint)
        self.setModal(True)

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
        self._rect_start         = None
        self._rect_end           = None
        self._drawing_rect       = False
        self._has_active_prompts = False
        self._hover_pos          = None

        self._dot_items    = []
        self._rect_item    = None
        self._overlay_item = None

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._on_hover_timeout)

        self._build_ui()
        self._size_to_viewer()

    # ------------------------------------------------------------------
    # UI — zero-margin layout with a single QGraphicsView
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

        pix = _numpy_to_qpixmap(self._rgb)
        self._bg_item = QGraphicsPixmapItem(pix)
        self._bg_item.setZValue(0)
        self._scene.addItem(self._bg_item)
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)

    def _size_to_viewer(self):
        """Position this dialog to exactly cover the MVATViewer on screen."""
        if self._viewer is None:
            return
        tl = self._viewer.mapToGlobal(self._viewer.rect().topLeft())
        self.setGeometry(tl.x(), tl.y(),
                         self._viewer.width(), self._viewer.height())

    # ------------------------------------------------------------------
    # Show / fit
    # ------------------------------------------------------------------

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
        if self._drawing_rect and self._rect_start is not None:
            self._rect_end = scene_pos
            self._draw_rect_preview()
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
    # Fixed prompts
    # ------------------------------------------------------------------

    def add_positive(self, scene_pos: QPointF):
        self._positive_pts.append(scene_pos)
        self._has_active_prompts = True
        self._add_dot(scene_pos, _POS_COLOR)
        self._predict(hover_pos=None)

    def add_negative(self, scene_pos: QPointF):
        self._negative_pts.append(scene_pos)
        self._has_active_prompts = True
        self._add_dot(scene_pos, _NEG_COLOR)
        self._predict(hover_pos=None)

    def begin_rect(self, scene_pos: QPointF):
        self._rect_start   = scene_pos
        self._rect_end     = scene_pos
        self._drawing_rect = True

    def finish_rect(self, scene_pos: QPointF):
        if not self._drawing_rect:
            return
        self._rect_end     = scene_pos
        self._drawing_rect = False
        self._has_active_prompts = True
        self._draw_rect_preview()
        self._predict(hover_pos=None)

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

    def _add_dot(self, pos: QPointF, color: QColor):
        r    = _DOT_R
        item = QGraphicsEllipseItem(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
        pen  = QPen(Qt.white, 1.5)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(color))
        item.setZValue(20)
        self._scene.addItem(item)
        self._dot_items.append(item)

    def _draw_rect_preview(self):
        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None
        if self._rect_start is None or self._rect_end is None:
            return
        rect = QRectF(self._rect_start, self._rect_end).normalized()
        pen  = QPen(_RECT_COLOR, 2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self._rect_item = QGraphicsRectItem(rect)
        self._rect_item.setPen(pen)
        self._rect_item.setBrush(QBrush(QColor(_RECT_COLOR.red(),
                                               _RECT_COLOR.green(),
                                               _RECT_COLOR.blue(), 30)))
        self._rect_item.setZValue(10)
        self._scene.addItem(self._rect_item)

    def _update_overlay(self):
        self._clear_overlay()
        if self._binary_mask is None:
            return
        color = QColor(self._label.color) if self._label is not None else QColor(0, 200, 50)
        pix   = _mask_to_qpixmap(self._binary_mask, color)
        self._overlay_item = QGraphicsPixmapItem(pix)
        self._overlay_item.setZValue(5)
        self._scene.addItem(self._overlay_item)

    def _clear_overlay(self):
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None

    # ------------------------------------------------------------------
    # Accept / cancel
    # ------------------------------------------------------------------

    def _on_accept(self):
        if self._binary_mask is None:
            return
        self._hover_timer.stop()
        self.maskAccepted.emit(self._binary_mask.copy())
        self.accept()

    def _on_cancel(self):
        self._hover_timer.stop()
        self.reject()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        if k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._on_accept()
        elif k == Qt.Key_Backspace:
            self._clear_prompts()
        elif k == Qt.Key_Escape:
            self._on_cancel()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Custom QGraphicsView
# ---------------------------------------------------------------------------

class _PromptView(QGraphicsView):
    def __init__(self, scene, dialog: MVATSAMDialog):
        super().__init__(scene)
        self._dialog       = dialog
        self._drag_started = False
        self._drag_origin  = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent):
        self._dialog.keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self._dialog.add_positive(sp)
            else:
                self._drag_started = False
                self._drag_origin  = sp
        elif event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier:
                self._dialog.add_negative(sp)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())
        if event.buttons() & Qt.LeftButton and self._drag_origin is not None:
            diff = sp - self._drag_origin
            if not self._drag_started and (abs(diff.x()) > 4 or abs(diff.y()) > 4):
                self._drag_started = True
                self._dialog.begin_rect(self._drag_origin)
            if self._drag_started:
                self._dialog.on_hover_move(sp)
                super().mouseMoveEvent(event)
                return
        self._dialog.on_hover_move(sp)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            sp = self.mapToScene(event.pos())
            if self._drag_started:
                self._dialog.finish_rect(sp)
            self._drag_started = False
            self._drag_origin  = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._dialog.on_hover_leave()
        super().leaveEvent(event)
