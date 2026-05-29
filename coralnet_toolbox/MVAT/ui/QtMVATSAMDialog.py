"""
MVATSAMDialog -- modal SAM segmentation dialog for arbitrary 3D viewer screenshots.

Workflow
--------
1.  MVATManager calls ``MVATSAMDialog(rgb_image, index_map, element_type,
                                      sam_dialog, label, parent)``
2.  The dialog shows the screenshot in a fit-to-window GraphicsView.
3.  The user interacts exactly like the SAMTool in AnnotationWindow:
      - Move mouse -> live hover prediction (debounced, 10 ms)
      - Ctrl+Left-click  -> lock-in positive point
      - Ctrl+Right-click -> lock-in negative point
      - Left-drag        -> bounding-box rectangle
      - Backspace        -> clear prompts
      - Enter / Space    -> accept current mask
      - Escape           -> cancel
4.  Pressing Enter / clicking Accept emits ``maskAccepted(np.ndarray)`` with
    the full-resolution binary mask (same H x W as rgb_image / index_map).
5.  Pressing Escape / Cancel closes the dialog with no side-effects.
"""

import numpy as np

from PyQt5.QtCore import (Qt, QPointF, QRectF, QSize, QTimer, pyqtSignal)
from PyQt5.QtGui import (QColor, QImage, QPainter, QPen, QBrush,
                         QPixmap, QKeyEvent, QMouseEvent, QCursor, QKeySequence)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsRectItem,
                             QSizePolicy, QApplication, QShortcut)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert an H x W x 3 uint8 numpy array to a QPixmap."""
    h, w, ch = rgb.shape
    assert ch == 3, "Expected RGB image"
    img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def _mask_to_qpixmap(binary_mask: np.ndarray, color: QColor) -> QPixmap:
    """
    Convert a boolean H x W mask to a semi-transparent colour QPixmap.
    Pixels inside the mask are ``color`` at 50 % opacity; outside transparent.
    """
    h, w = binary_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[binary_mask, 0] = color.red()
    rgba[binary_mask, 1] = color.green()
    rgba[binary_mask, 2] = color.blue()
    rgba[binary_mask, 3] = 128          # 50 % alpha
    img = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(img)


# ---------------------------------------------------------------------------
# SAM prompt graphics constants
# ---------------------------------------------------------------------------

_POS_COLOR  = QColor(0,   200,  50,  220)   # green  - positive click
_NEG_COLOR  = QColor(220,  30,  30,  220)   # red    - negative click
_RECT_COLOR = QColor(0,   168, 230,  200)   # blue   - bounding box
_DOT_RADIUS = 6                              # px in scene space


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class MVATSAMDialog(QDialog):
    """
    Modal SAM segmentation dialog for MVATViewer screenshots.

    Interaction mirrors the SAMTool used in the main AnnotationWindow:
      - Hover  -> live debounced prediction (hover point as implicit positive)
      - Ctrl+Left  -> lock-in positive point
      - Ctrl+Right -> lock-in negative point
      - Left-drag  -> bounding-box rectangle
      - Backspace  -> clear all prompts
      - Enter      -> accept current mask
      - Escape     -> cancel

    Parameters
    ----------
    rgb_image   : np.ndarray, shape (H, W, 3), dtype uint8
    index_map   : np.ndarray, shape (H, W), dtype int32
    element_type : str  ('point' or 'face')
    sam_dialog  : DeployPredictorDialog  (model already loaded)
    label       : Label
    parent      : QWidget, optional
    """

    # Emitted on Accept with the full-resolution binary mask (dtype bool, H x W)
    maskAccepted = pyqtSignal(object)    # np.ndarray

    def __init__(self, rgb_image: np.ndarray, index_map: np.ndarray,
                 element_type: str, sam_dialog, label, parent=None):
        super().__init__(parent,
                         Qt.Dialog |
                         Qt.WindowTitleHint |
                         Qt.WindowCloseButtonHint)
        self.setWindowTitle("MVAT SAM  --  segment 3D view")
        self.setModal(True)

        self._rgb        = rgb_image          # H x W x 3
        self._index_map  = index_map          # H x W  int32
        self._elem_type  = element_type
        self._sam        = sam_dialog
        self._label      = label

        self._img_h, self._img_w = rgb_image.shape[:2]

        # Fixed prompt state (locked in by Ctrl+click or drag)
        self._positive_pts: list = []   # QPointF
        self._negative_pts: list = []   # QPointF
        self._rect_start = None         # QPointF | None
        self._rect_end   = None         # QPointF | None
        self._drawing_rect = False
        self._has_active_prompts = False

        # Hover state (implicit positive, cleared on next move)
        self._hover_pos = None          # QPointF | None  (scene coords)

        # Current accepted prediction
        self._binary_mask = None        # np.ndarray | None

        # Graphics items managed by us
        self._dot_items  = []
        self._rect_item  = None
        self._overlay_item = None

        # Debounce timer -- mirrors SAMTool.hover_timer
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._on_hover_timeout)
        self._debounce_ms = 10

        self._build_ui()
        self._load_image()
        self._connect_shortcuts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Instructions
        info = QLabel(
            "<b>Hover</b> preview &nbsp;|&nbsp; "
            "<b>Ctrl+Left</b> positive &nbsp;|&nbsp; "
            "<b>Ctrl+Right</b> negative &nbsp;|&nbsp; "
            "<b>Left-drag</b> bbox &nbsp;|&nbsp; "
            "<b>Backspace</b> clear"
        )
        info.setAlignment(Qt.AlignCenter)
        root.addWidget(info)

        # Graphics view
        self._scene = QGraphicsScene(self)
        self._view  = _PromptView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setDragMode(QGraphicsView.NoDrag)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root.addWidget(self._view, stretch=1)

        # Status line
        self._status = QLabel("Move mouse over the view to preview segmentation.")
        self._status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("Accept  [Enter]")
        self._btn_accept.setEnabled(False)
        self._btn_clear  = QPushButton("Clear  [Backspace]")
        self._btn_cancel = QPushButton("Cancel  [Esc]")

        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_clear.clicked.connect(self._clear_prompts)
        self._btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_accept)
        root.addLayout(btn_row)

        # Size: fill ~85 % of the screen
        screen = QApplication.primaryScreen().availableGeometry()
        dlg_w  = min(int(screen.width()  * 0.85), self._img_w + 40)
        dlg_h  = min(int(screen.height() * 0.85), self._img_h + 140)
        self.resize(dlg_w, dlg_h)

    def _load_image(self):
        """Put the screenshot into the scene and fit it."""
        pix = _numpy_to_qpixmap(self._rgb)
        self._bg_item = QGraphicsPixmapItem(pix)
        self._bg_item.setZValue(0)
        self._scene.addItem(self._bg_item)
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _connect_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Return),   self, self._on_accept)
        QShortcut(QKeySequence(Qt.Key_Enter),    self, self._on_accept)
        QShortcut(QKeySequence(Qt.Key_Space),    self, self._on_accept)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self._clear_prompts)
        QShortcut(QKeySequence(Qt.Key_Escape),   self, self.reject)

    # ------------------------------------------------------------------
    # Hover prediction (called by _PromptView on mouse move)
    # ------------------------------------------------------------------

    def on_hover_move(self, scene_pos: QPointF):
        """Debounced hover: schedule a prediction with scene_pos as implicit positive."""
        self._hover_pos = scene_pos
        # If drawing a rectangle, update it live
        if self._drawing_rect and self._rect_start is not None:
            self._rect_end = scene_pos
            self._draw_rect_preview()
            # Don't predict during mid-drag; predict on release
            return
        self._hover_timer.start(self._debounce_ms)

    def on_hover_leave(self):
        """Mouse left the view — stop hover timer, clear hover overlay if no fixed prompts."""
        self._hover_timer.stop()
        self._hover_pos = None
        if not self._has_active_prompts:
            self._clear_overlay()
            self._binary_mask = None
            self._btn_accept.setEnabled(False)
            self._status.setText("Move mouse over the view to preview segmentation.")

    def _on_hover_timeout(self):
        """Run SAM prediction using current fixed prompts + hover point as implicit positive."""
        if self._hover_pos is None and not self._has_active_prompts:
            return
        self._predict(hover_pos=self._hover_pos)

    # ------------------------------------------------------------------
    # Fixed prompt handling (called by _PromptView on Ctrl+click / drag)
    # ------------------------------------------------------------------

    def add_positive(self, scene_pos: QPointF):
        self._positive_pts.append(scene_pos)
        self._has_active_prompts = True
        self._add_dot(scene_pos, _POS_COLOR)
        self._predict(hover_pos=None)   # predict without hover ghost

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
        self._rect_start   = None
        self._rect_end     = None
        self._drawing_rect = False
        self._has_active_prompts = False
        self._binary_mask  = None
        self._hover_timer.stop()

        for item in self._dot_items:
            self._scene.removeItem(item)
        self._dot_items.clear()

        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None

        self._clear_overlay()
        self._btn_accept.setEnabled(False)
        self._status.setText("Prompts cleared.")

    # ------------------------------------------------------------------
    # SAM prediction
    # ------------------------------------------------------------------

    def _predict(self, hover_pos=None):
        """Run SAM with fixed prompts, optionally adding hover_pos as extra positive."""
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

        # Add hover point as implicit positive (like SAMTool) unless rect-dragging
        if hover_pos is not None and not self._drawing_rect:
            positive = positive + [[hover_pos.x(), hover_pos.y()]]

        all_pts = positive + negative
        bboxes_input = [bbox]   if bbox     is not None else None
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
        except Exception as e:
            self._status.setText(f"SAM prediction failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not results or len(results) == 0:
            self._status.setText("No prediction returned.")
            return

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            self._status.setText("No mask in prediction.")
            return

        conf = result.boxes.conf
        if isinstance(conf, torch.Tensor):
            top_idx = int(torch.argmax(conf).item())
        else:
            top_idx = int(np.argmax(np.asarray(conf)))

        mask_tensor = result.masks.data[top_idx]
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = np.asarray(mask_tensor)

        # Resize mask to original image size if needed
        mh, mw = mask_np.shape[-2], mask_np.shape[-1]
        if (mh, mw) != (self._img_h, self._img_w):
            import cv2
            mask_np = cv2.resize(
                mask_np.astype(np.float32),
                (self._img_w, self._img_h),
                interpolation=cv2.INTER_LINEAR,
            )

        self._binary_mask = (mask_np > 0.5)

        n_elem   = self._count_elements()
        conf_val = float(result.boxes.conf[top_idx])
        self._status.setText(
            f"Mask: {int(self._binary_mask.sum()):,} px  |  "
            f"~{n_elem:,} {self._elem_type}(s)  |  "
            f"conf {conf_val:.2f}"
        )

        self._update_overlay()
        self._btn_accept.setEnabled(True)

    def _count_elements(self) -> int:
        if self._binary_mask is None or self._index_map is None:
            return 0
        ids = self._index_map[self._binary_mask]
        return int(np.sum(ids >= 0))

    # ------------------------------------------------------------------
    # Graphics helpers
    # ------------------------------------------------------------------

    def _add_dot(self, pos: QPointF, color: QColor):
        r = _DOT_RADIUS
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
    # Accept / reject
    # ------------------------------------------------------------------

    def _on_accept(self):
        if self._binary_mask is None:
            return
        self._hover_timer.stop()
        self.maskAccepted.emit(self._binary_mask.copy())
        self.accept()

    # ------------------------------------------------------------------
    # Resize: re-fit image
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        self._hover_timer.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Custom QGraphicsView -- captures prompt mouse events
# ---------------------------------------------------------------------------

class _PromptView(QGraphicsView):
    """
    GraphicsView that forwards prompt interactions to MVATSAMDialog.

    - Mouse move           -> dialog.on_hover_move (debounced hover prediction)
    - Leave event          -> dialog.on_hover_leave
    - Ctrl+Left-click      -> dialog.add_positive
    - Ctrl+Right-click     -> dialog.add_negative
    - Left-drag (no Ctrl)  -> bbox rectangle
    """

    def __init__(self, scene, dialog: MVATSAMDialog):
        super().__init__(scene)
        self._dialog       = dialog
        self._drag_started = False
        self._drag_origin  = None   # QPointF | None
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))

    def mousePressEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self._dialog.add_positive(sp)
            else:
                # Start potential rect drag
                self._drag_started = False
                self._drag_origin  = sp
        elif event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier:
                self._dialog.add_negative(sp)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        sp = self.mapToScene(event.pos())

        # Handle left-drag for bounding box
        if event.buttons() & Qt.LeftButton and self._drag_origin is not None:
            diff = sp - self._drag_origin
            if not self._drag_started and (abs(diff.x()) > 4 or abs(diff.y()) > 4):
                self._drag_started = True
                self._dialog.begin_rect(self._drag_origin)
            if self._drag_started:
                self._dialog.on_hover_move(sp)   # updates rect preview via dialog
                super().mouseMoveEvent(event)
                return

        # Normal hover
        self._dialog.on_hover_move(sp)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            sp = self.mapToScene(event.pos())
            if self._drag_started:
                self._dialog.finish_rect(sp)
            # Single click (no Ctrl, no drag) -- no fixed-prompt action;
            # hover prediction already shows a live preview.
            self._drag_started = False
            self._drag_origin  = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._dialog.on_hover_leave()
        super().leaveEvent(event)
