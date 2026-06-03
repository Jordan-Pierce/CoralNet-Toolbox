"""
Context Matrix Widget for MVAT (Unified Flow Gallery)

Displays a dynamic, auto-flowing gallery of interactive BaseCanvas viewports.
Replaces the legacy CameraGrid.

Features include:
- Dynamic object pool of BaseCanvas instances.
- Auto-flow layout adaptation based on panel width and a user-controlled camera count.
- Image loading from RasterManager.
- Viewer-only camera navigation from clicks.
"""

import warnings
import numpy as np
from typing import List, Optional, Dict

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint, QVariantAnimation, QEasingCurve, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QToolBar, QToolButton, QSizePolicy, QFrame,
    QScrollArea, QLayout, QLayoutItem, QGraphicsOpacityEffect,
    QSpinBox, QAbstractSpinBox,
    QMenu, QAction,
)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.QtBaseCanvas import BaseCanvas
from coralnet_toolbox.MVAT.core.constants import (
    SELECT_COLOR,
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_INVALID,
)

from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Propagation mode constants used by maskPropagationRequested signal
PROPAGATE_ACTIVE_TO_CONTEXT    = "active_to_context"   # Primary → Secondary Cameras
PROPAGATE_PRIMARY_TO_MESH      = "active_camera_to_mesh"  # Primary → Mesh (alias for PROPAGATE_ACTIVE_CAMERA_TO_MESH)
PROPAGATE_ACTIVE_CAMERA_TO_MESH = "active_camera_to_mesh"  # same value; kept for back-compat
PROPAGATE_CAMERAS_TO_MESH      = "cameras_to_mesh"
PROPAGATE_MESH_TO_ACTIVE_CAMERA = "mesh_to_active_camera"
PROPAGATE_MESH_TO_CAMERAS      = "mesh_to_cameras"

# Human-readable button labels for each mode
_PROPAGATE_MODE_LABELS = {
    PROPAGATE_ACTIVE_TO_CONTEXT:    "Primary → Secondary",
    PROPAGATE_ACTIVE_CAMERA_TO_MESH: "Primary → Mesh",
    PROPAGATE_CAMERAS_TO_MESH:       "All Cams → Mesh",
    PROPAGATE_MESH_TO_ACTIVE_CAMERA: "Mesh → Primary",
    PROPAGATE_MESH_TO_CAMERAS:       "Mesh → All Cams",
}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FlowLayout(QLayout):
    """Wrap child widgets across rows."""

    def __init__(self, parent=None, margin: int = 0, h_spacing: int = -1, v_spacing: int = -1):
        super().__init__(parent)
        self._item_list: List[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index: int):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _horizontal_spacing(self) -> int:
        return self._h_spacing if self._h_spacing >= 0 else max(8, self.spacing())

    def _vertical_spacing(self) -> int:
        return self._v_spacing if self._v_spacing >= 0 else max(8, self.spacing())

    def _do_layout(self, rect: QRect, test_only: bool):
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(+margins.left(), +margins.top(), -margins.right(), -margins.bottom())
        if not self._item_list:
            return rect.height() + margins.top() + margins.bottom()

        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        space_x = self._horizontal_spacing()
        space_y = self._vertical_spacing()

        for item in self._item_list:
            item_size = item.sizeHint()
            next_x = x + item_size.width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item_size.width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))

            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + margins.bottom()


class ContextMatrixWidget(QWidget):
    """
    Interactive flow gallery of BaseCanvas viewports for multi-camera context viewing.
    Unified replacement for the legacy CameraGrid widget.

    Signals:
        contextImagePromoted: Legacy compatibility signal; not emitted by the current viewer-only interaction model.
        rankIndicatorUpdated: Emitted when rank indicator changes (start, end, total)
        multiAnnotateToggled: Emitted when multi-annotate mode is toggled (bool)
        semanticMaskPropagationRequested: Emitted when the active semantic mask should be propagated.
        loadCamerasRequested: Emitted when the Load Cameras button is clicked
        clearSelectionsRequested: Legacy compatibility signal; kept for future workflows.
        visibleCamerasChanged: Emitted when the visible canvas set changes.
    """

    contextImagePromoted = pyqtSignal(str)            # camera_path
    rankIndicatorUpdated = pyqtSignal(int, int, int)  # start, end, total
    multiAnnotateToggled = pyqtSignal(bool)           # enabled state
    semanticMaskPropagationRequested = pyqtSignal()  # kept for back-compat
    maskPropagationRequested = pyqtSignal(str)        # carries a PROPAGATE_* mode constant

    # Migrated from legacy CameraGrid
    loadCamerasRequested = pyqtSignal()
    loadIndexMapsRequested = pyqtSignal()
    clearSelectionsRequested = pyqtSignal()
    previousCameraRequested = pyqtSignal()
    nextCameraRequested = pyqtSignal()
    visibleCamerasChanged = pyqtSignal(list)

    # Click intent signals
    camera_highlighted_single = pyqtSignal(str)         # single plain-click -> jump 3D view
    new_active_camera_requested = pyqtSignal(str)       # Ctrl+Click -> jump main image

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matrix state
        self.target_camera_count = 6
        self._camera_count_cap = None
        self._last_rebuilt_count = 0
        self._canvas_count_step = 1
        self._canvas_count_min = 1
        self._canvas_tile_size = 240
        self._canvas_tile_step = 32
        self._canvas_tile_min = 32
        self._canvas_tile_max = 2048

        # Camera Data State
        self._camera_paths: List[str] = []
        self._raster_manager = None
        self._loading_flag = False

        # Marker state for conveyor belt persistence (Phase 4)
        self._last_focal_point = None
        self._cameras_ref: Optional[Dict] = None

        # Target-lock sync state (Phase 5)
        self.target_lock_enabled = True
        self._mvat_manager = None
        # Camera roll cache: (ref_path, ctx_path) -> roll_degrees
        # Camera orientations are fixed after loading, so the roll between any two
        # cameras is constant.  Caching avoids repeating the matrix+arctan2 work
        # on every navigation event.
        self._roll_cache: dict = {}

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._pending_sync = None
        self._active_camera_path = None

        self._scene_controls_enabled = False

        # Annotation visualization state (Phase 6)
        self._annotation_manager = None
        self._annotation_updates_suspended = False

        # Optional typed camera-count control in the bottom toolbar.
        self.camera_count_input = None
        self.camera_total_label = None
        self.camera_count_panel = None

        # Canvas pool
        self._canvas_pool: List[BaseCanvas] = []
        self._visible_canvases: List[BaseCanvas] = []
        self._canvas_animations: List[QVariantAnimation] = []
        self._canvas_animation_duration = 180
        self._canvas_intro_scale = 0.88

        # Scrollable flow-layout gallery
        self._canvas_host_widget = QWidget(self)
        self._canvas_host_layout = QVBoxLayout(self._canvas_host_widget)
        self._canvas_host_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_host_layout.setSpacing(0)

        self._pending_repaints = set()
        self._pending_repaint_tasks: Dict[str, dict] = {}

        self._placeholder_label = QLabel(
            "No cameras available\nLoad camera lists to populate the matrix.",
            self._canvas_host_widget,
        )
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._placeholder_label.setStyleSheet(
            f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
        )

        self._flow_widget = QWidget(self._canvas_host_widget)
        self._flow_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self._flow_layout = FlowLayout(self._flow_widget, margin=4, h_spacing=8, v_spacing=8)
        self._flow_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self._flow_widget.setLayout(self._flow_layout)

        self._canvas_host_layout.addWidget(self._placeholder_label)
        self._canvas_host_layout.addWidget(self._flow_widget)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll_area.setWidget(self._canvas_host_widget)
        self._scroll_area.verticalScrollBar().valueChanged.connect(self._flush_pending_repaints)

        # UI Setup
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._main_layout.addWidget(self._scroll_area)

        # Create initial layout
        self._evaluate_auto_layout()

    # ==================== Canvas Pool Management ====================

    def _ensure_canvas_pool_size(self, size: int):
        """Dynamically expand the BaseCanvas pool if needed."""
        while len(self._canvas_pool) < size:
            canvas = BaseCanvas(parent=self)
            canvas.hide()
            canvas.setFixedSize(self._canvas_tile_size, self._canvas_tile_size)
            canvas.mouseDoubleClickEvent = self._make_canvas_double_click_handler(canvas)
            canvas.mousePressEvent = self._make_canvas_mouse_press_handler(canvas)
            canvas.wheelEvent = self._make_canvas_wheel_handler(canvas)
            self._canvas_pool.append(canvas)

    def _update_canvas_size_bounds(self, camera_objects: Optional[List] = None):
        """Derive the maximum tile size from loaded camera dimensions."""
        max_tile_size = self._canvas_tile_max
        if camera_objects:
            observed_max = 0
            for camera in camera_objects:
                width = int(getattr(camera, 'width', 0) or 0)
                height = int(getattr(camera, 'height', 0) or 0)
                if width > 0 and height > 0:
                    observed_max = max(observed_max, width, height)

            if observed_max > 0:
                max_tile_size = max(self._canvas_tile_min, observed_max)

        self._canvas_tile_max = max(self._canvas_tile_min, max_tile_size)
        if self._canvas_tile_size > self._canvas_tile_max:
            self._set_canvas_tile_size(self._canvas_tile_max)
        else:
            self._update_canvas_size_controls()

    def _ensure_canvas_opacity_effect(self, canvas: BaseCanvas) -> QGraphicsOpacityEffect:
        effect = canvas.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(canvas)
            effect.setOpacity(1.0)
            canvas.setGraphicsEffect(effect)
        return effect

    def _release_canvas_animation(self, canvas: BaseCanvas, animation: QVariantAnimation):
        if getattr(canvas, '_context_animation', None) is animation:
            canvas._context_animation = None
        try:
            if animation in self._canvas_animations:
                self._canvas_animations.remove(animation)
        except Exception:
            pass
        try:
            animation.deleteLater()
        except Exception:
            pass

    def _stop_canvas_animation(self, canvas: BaseCanvas):
        animation = getattr(canvas, '_context_animation', None)
        if animation is None:
            return

        try:
            animation.stop()
        except Exception:
            pass
        self._release_canvas_animation(canvas, animation)

    def _animate_canvas_transition(
        self,
        canvas: BaseCanvas,
        start_size: int,
        end_size: int,
        start_opacity: float,
        end_opacity: float,
        duration: int = None,
        on_finished=None,
    ):
        self._stop_canvas_animation(canvas)
        effect = self._ensure_canvas_opacity_effect(canvas)
        animation = QVariantAnimation(self)
        animation.setDuration(duration or self._canvas_animation_duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        canvas._context_animation = animation
        self._canvas_animations.append(animation)
        canvas.show()

        def _apply(progress):
            value = float(progress)
            size = int(round(start_size + (end_size - start_size) * value))
            opacity = start_opacity + (end_opacity - start_opacity) * value
            canvas.setFixedSize(max(1, size), max(1, size))
            effect.setOpacity(max(0.0, min(1.0, opacity)))
            self._flow_widget.updateGeometry()

        def _finish():
            _apply(1.0)
            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    pass
            self._release_canvas_animation(canvas, animation)

        animation.valueChanged.connect(_apply)
        animation.finished.connect(_finish)
        _apply(0.0)
        animation.start()
        return animation

    def _animate_canvas_removal(self, canvases: List[BaseCanvas], on_finished=None):
        if not canvases:
            if on_finished is not None:
                on_finished()
            return

        pending = {"count": len(canvases)}

        def _one_finished():
            pending["count"] -= 1
            if pending["count"] > 0:
                return

            for canvas in canvases:
                try:
                    canvas.hide()
                except Exception:
                    pass
                try:
                    effect = canvas.graphicsEffect()
                    if isinstance(effect, QGraphicsOpacityEffect):
                        effect.setOpacity(1.0)
                except Exception:
                    pass

            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    pass

        for canvas in canvases:
            current_size = max(1, canvas.width() or self._canvas_tile_size)
            self._animate_canvas_transition(
                canvas,
                current_size,
                current_size,
                1.0,
                0.0,
                on_finished=_one_finished,
            )

    def _make_canvas_mouse_press_handler(self, canvas: BaseCanvas):
        """Intercept left clicks for viewer-only navigation while preserving native canvas interactions."""
        def handler(event):
            if event.button() == Qt.LeftButton:
                path = canvas.current_image_path
                if path:
                    if event.modifiers() & Qt.ControlModifier:
                        # Ctrl+Click: update main AnnotationWindow image.
                        self.new_active_camera_requested.emit(path)
                    else:
                        # Plain click: update the 3D viewer only.
                        self.camera_highlighted_single.emit(path)

            # CRITICAL: pass through so drawing/panning tools still work
            BaseCanvas.mousePressEvent(canvas, event)
        return handler

    def _make_canvas_double_click_handler(self, canvas: BaseCanvas):
        """Right double-click updates the 3D viewer only."""
        def handler(event):
            if event.button() == Qt.RightButton:
                try:
                    canvas._pan_active = False
                    canvas._pan_start = None
                    canvas._rotate_active = False
                    canvas.setCursor(Qt.ArrowCursor)
                except Exception:
                    pass
                path = canvas.current_image_path
                if path:
                    self.camera_highlighted_single.emit(path)
            BaseCanvas.mouseDoubleClickEvent(canvas, event)
        return handler

    def _make_canvas_wheel_handler(self, canvas: BaseCanvas):
        """Keep wheel events on the hovered canvas instead of letting them
        bubble up to the surrounding QScrollArea.

        Qt propagates a wheel event to the parent widget whenever the receiver
        leaves the event unaccepted. BaseCanvas.wheelEvent returns silently
        when no image is loaded, and even when zoom is applied the event isn't
        explicitly accepted — so the matrix's scroll area scrolls vertically
        on top of (or instead of) the canvas zoom. Always accept() here so the
        wheel acts purely on the hovered tile.
        """
        def handler(event):
            try:
                BaseCanvas.wheelEvent(canvas, event)
            finally:
                # Mark consumed regardless of what BaseCanvas did so the event
                # never reaches the QScrollArea above us.
                event.accept()
        return handler

    # ==================== Input / Scroll Events ====================

    def set_target_camera_count(self, count: int):
        """Update the desired number of visible context canvases."""
        try:
            count = int(count)
        except Exception:
            count = self._canvas_count_min

        max_count = self._camera_count_cap if self._camera_count_cap is not None else None
        if max_count is not None:
            count = min(count, max_count)

        self.target_camera_count = max(self._canvas_count_min, count)
        layout_ready = self._evaluate_auto_layout()
        if layout_ready:
            self._refresh_visible_canvases()

    def increase_canvas_count(self):
        self.set_target_camera_count(self.target_camera_count + self._canvas_count_step)

    def decrease_canvas_count(self):
        self.set_target_camera_count(self.target_camera_count - self._canvas_count_step)

    # ==================== Layout Rebuilding ====================

    def _evaluate_auto_layout(self):
        """Rebuild the flow layout for the current camera count target."""
        available = len(self._camera_paths)
        if available <= 0:
            effective_target = 0
        else:
            effective_target = min(available, self.target_camera_count)
            if self._camera_count_cap is not None:
                effective_target = min(effective_target, self._camera_count_cap)

        if effective_target != self._last_rebuilt_count:
            return self._rebuild_layout(effective_target)
        else:
            self._update_empty_state_visibility(effective_target > 0)
            return True

    def _apply_layout_state(self, count: int, previous_visible_count: int = 0):
        while self._flow_layout.count():
            item = self._flow_layout.takeAt(0)
            if item and item.widget():
                item.widget().hide()

        for canvas in self._canvas_pool:
            canvas.hide()

        self._visible_canvases = []

        if count <= 0:
            self._update_empty_state_visibility(False)
            self._flow_widget.updateGeometry()
            self._canvas_host_widget.adjustSize()
            self._update_canvas_count_controls()
            return

        self._ensure_canvas_pool_size(count)

        for index in range(count):
            canvas = self._canvas_pool[index]
            self._flow_layout.addWidget(canvas)
            self._visible_canvases.append(canvas)

            effect = self._ensure_canvas_opacity_effect(canvas)
            canvas.show()

            if index >= previous_visible_count:
                intro_size = max(self._canvas_tile_min, int(round(self._canvas_tile_size * self._canvas_intro_scale)))
                canvas.setFixedSize(intro_size, intro_size)
                effect.setOpacity(0.0)
                self._animate_canvas_transition(
                    canvas,
                    intro_size,
                    self._canvas_tile_size,
                    0.0,
                    1.0,
                )
            else:
                canvas.setFixedSize(self._canvas_tile_size, self._canvas_tile_size)
                effect.setOpacity(1.0)

        self._update_empty_state_visibility(True)
        self._flow_widget.updateGeometry()
        self._canvas_host_widget.adjustSize()
        self._update_canvas_count_controls()

    def _finalize_layout_refresh(self, count: int, previous_visible_count: int = 0):
        self._apply_layout_state(count, previous_visible_count)
        self._refresh_visible_canvases()

    def _rebuild_layout(self, count: int):
        """Rebuild the flow layout with the requested number of visible canvases."""
        previous_visible = list(self._visible_canvases)
        previous_count = len(previous_visible)
        self._last_rebuilt_count = count

        if count <= 0:
            if previous_visible:
                self._animate_canvas_removal(previous_visible, lambda: self._finalize_layout_refresh(0, 0))
            else:
                self._finalize_layout_refresh(0, 0)
            return False

        if count < previous_count:
            removed_canvases = previous_visible[count:]
            if removed_canvases:
                self._animate_canvas_removal(
                    removed_canvases,
                    lambda target_count=count: self._finalize_layout_refresh(target_count, target_count),
                )
                return False

        self._apply_layout_state(count, previous_count)
        return True

    def resizeEvent(self, event):
        """Auto-adjust layout on resize without changing camera count."""
        super().resizeEvent(event)
        self._flow_widget.updateGeometry()
        self._flush_pending_repaints()

    def is_canvas_on_screen(self, canvas: BaseCanvas) -> bool:
        """Return True when a canvas intersects the scroll viewport."""
        if canvas is None or not canvas.isVisible() or not canvas.active_image:
            return False

        viewport = self._scroll_area.viewport() if self._scroll_area is not None else None
        if viewport is None:
            return False

        top_left = canvas.mapTo(viewport, canvas.rect().topLeft())
        bottom_right = canvas.mapTo(viewport, canvas.rect().bottomRight())
        canvas_rect = QRect(top_left, bottom_right).normalized()
        return canvas_rect.intersects(viewport.rect())

    def queue_pending_repaint(self, path: str, mask_annotation, update_rect=None, label_ids=()):
        """Defer a repaint until the matching canvas scrolls into view."""
        if not path or mask_annotation is None:
            return

        label_ids = tuple(sorted({label_id for label_id in label_ids if label_id is not None}))
        existing = self._pending_repaint_tasks.get(path)
        if existing is None:
            self._pending_repaints.add(path)
            self._pending_repaint_tasks[path] = {
                'path': path,
                'mask': mask_annotation,
                'update_rect': update_rect,
                'label_ids': label_ids,
            }
            return

        existing['mask'] = mask_annotation
        existing['label_ids'] = tuple(sorted(set(existing.get('label_ids', ())) | set(label_ids)))
        existing_rect = existing.get('update_rect')
        if existing_rect is None:
            existing['update_rect'] = update_rect
        elif update_rect is not None:
            existing['update_rect'] = (
                min(existing_rect[0], update_rect[0]),
                min(existing_rect[1], update_rect[1]),
                max(existing_rect[2], update_rect[2]),
                max(existing_rect[3], update_rect[3]),
            )

    def _flush_pending_repaints(self, *_args):
        if not self._pending_repaints:
            return

        for path in list(self._pending_repaints):
            task = self._pending_repaint_tasks.get(path)
            if not task:
                self._pending_repaints.discard(path)
                continue

            canvas = None
            for candidate in self._visible_canvases:
                if candidate and candidate.current_image_path == path:
                    canvas = candidate
                    break

            if canvas is None or not self.is_canvas_on_screen(canvas):
                continue

            target_mask = task.get('mask')
            if target_mask is None:
                self._pending_repaints.discard(path)
                self._pending_repaint_tasks.pop(path, None)
                continue

            try:
                for label_id in task.get('label_ids', ()):
                    if label_id is not None and label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)
                target_mask.update_graphics_item(update_rect=task.get('update_rect'))
                if canvas._mask_overlay_item is None:
                    canvas.set_mask_overlay(target_mask)
                self._pending_repaints.discard(path)
                self._pending_repaint_tasks.pop(path, None)
            except Exception:
                pass

    # ==================== Data Feed ====================

    def set_raster_manager(self, raster_manager):
        """Provide a RasterManager instance for image loading."""
        self._raster_manager = raster_manager

    def set_scene_controls_enabled(self, enabled: bool):
        """Enable or disable toolbar controls that depend on loaded 3D scene data."""
        self._scene_controls_enabled = bool(enabled)
        if hasattr(self, 'load_btn'):
            self.load_btn.setEnabled(self._scene_controls_enabled)
        if hasattr(self, '_multi_annotate_btn'):
            self._multi_annotate_btn.setEnabled(self._scene_controls_enabled)
        if hasattr(self, '_propagate_mask_btn'):
            self._propagate_mask_btn.setEnabled(self._scene_controls_enabled)
        self._update_canvas_size_controls()
        self._update_canvas_count_controls()

    def set_camera_data(self, camera_objects: List, ordered_paths: List[str]):
        """Update the camera list and refresh the layout.  Camera objects are used to derive tile size bounds."""
        self._camera_paths = list(ordered_paths)
        if self._active_camera_path not in self._camera_paths:
            self._active_camera_path = None
        self._update_canvas_size_bounds(camera_objects)
        layout_ready = self._evaluate_auto_layout()
        if layout_ready:
            self._refresh_visible_canvases()

    def set_camera_order(self, ordered_paths: List[str], active_path: str = None):
        """Update the camera order and refresh the layout.  Active path is prioritized to the front if present."""
        paths = list(ordered_paths)
        if active_path and active_path in paths and paths and paths[0] != active_path:
            paths = [active_path] + [path for path in paths if path != active_path]
        self._camera_paths = paths
        self._active_camera_path = active_path if active_path in paths else None
        layout_ready = self._evaluate_auto_layout()
        if layout_ready:
            self._refresh_visible_canvases()

    def get_camera_order(self) -> List[str]:
        """Return the current ordered list of camera paths."""
        return list(self._camera_paths)

    def get_visible_camera_paths(self) -> List[str]:
        """Return the list of camera paths currently visible in the matrix."""
        return [
            canvas.current_image_path
            for canvas in self._visible_canvases
            if canvas and canvas.active_image and canvas.current_image_path
        ]

    def _emit_visible_cameras_changed(self):
        """Emit the visibleCamerasChanged signal with the current visible camera paths."""
        self.visibleCamerasChanged.emit(self.get_visible_camera_paths())

    def _update_empty_state_visibility(self, has_cameras: bool):
        """Toggle visibility of the placeholder message and the flow layout based on whether cameras are available."""
        self._placeholder_label.setVisible(not has_cameras)
        self._flow_widget.setVisible(has_cameras)

    def _refresh_visible_canvases(self):
        """Load images into visible canvases and update marker states."""
        self._clear_canvas_perimeters()

        if not self._camera_paths:
            for canvas in self._canvas_pool:
                canvas.hide()
                canvas._show_placeholder("No cameras loaded")
            self._visible_canvases = []
            self._update_empty_state_visibility(False)
            self._update_canvas_count_controls()
            self._emit_visible_cameras_changed()
            return

        self._update_empty_state_visibility(True)

        visible_paths = self._camera_paths[:self._last_rebuilt_count]
        for index, canvas in enumerate(self._visible_canvases):
            if index < len(visible_paths):
                self._load_canvas_image(canvas, visible_paths[index])
            else:
                canvas.clear_scene()
                canvas.hide()

        if self._last_focal_point is not None and self._cameras_ref is not None:
            self.update_static_markers_from_3d(self._last_focal_point, self._cameras_ref)

        if self.target_lock_enabled and self._mvat_manager:
            self._request_sync_from_main_view()

        self._apply_active_canvas_perimeter()

        self._sync_camera_status_label()
        self._update_canvas_count_controls()
        self._emit_visible_cameras_changed()
        self._flush_pending_repaints()

    def _get_visible_capacity(self) -> int:
        """Return the current number of visible canvas slots in the layout."""
        return self._last_rebuilt_count

    def _apply_canvas_tile_size_to_canvas(self, canvas: BaseCanvas):
        canvas.setFixedSize(self._canvas_tile_size, self._canvas_tile_size)

    def _update_canvas_size_controls(self):
        scene_controls_enabled = self._scene_controls_enabled
        if hasattr(self, 'size_down_btn'):
            self.size_down_btn.setEnabled(scene_controls_enabled and self._canvas_tile_size > self._canvas_tile_min)
        if hasattr(self, 'size_up_btn'):
            self.size_up_btn.setEnabled(scene_controls_enabled and self._canvas_tile_size < self._canvas_tile_max)

    def _update_canvas_count_controls(self):
        available = len(self._camera_paths)
        max_allowed = available
        if self._camera_count_cap is not None:
            max_allowed = min(max_allowed, self._camera_count_cap)
        max_allowed = max(self._canvas_count_min, max_allowed) if available > 0 else 0
        can_decrease = available > 0 and self.target_camera_count > self._canvas_count_min
        can_increase = available > 0 and self.target_camera_count < max_allowed
        scene_controls_enabled = self._scene_controls_enabled
        if hasattr(self, 'count_down_btn'):
            self.count_down_btn.setEnabled(scene_controls_enabled and can_decrease)
        if hasattr(self, 'count_up_btn'):
            self.count_up_btn.setEnabled(scene_controls_enabled and can_increase)

        camera_count_input = getattr(self, 'camera_count_input', None)
        if camera_count_input is not None:
            try:
                camera_count_input.blockSignals(True)
                if available > 0:
                    camera_count_input.setRange(self._canvas_count_min, max_allowed)
                    camera_count_input.setValue(max(self._canvas_count_min, min(self.target_camera_count, max_allowed)))
                    camera_count_input.setEnabled(scene_controls_enabled)
                else:
                    camera_count_input.setEnabled(False)
                    camera_count_input.setRange(self._canvas_count_min, self._canvas_count_min)
                    camera_count_input.setValue(self._canvas_count_min)
            finally:
                camera_count_input.blockSignals(False)

    def _sync_camera_status_label(self, visible_count: Optional[int] = None, total_count: Optional[int] = None):
        if not hasattr(self, 'stats_label'):
            return

        if visible_count is None:
            visible_count = self._last_rebuilt_count

        if total_count is None:
            total_count = len(self._camera_paths)

        try:
            visible_count = int(visible_count)
        except Exception:
            visible_count = 0

        try:
            total_count = int(total_count)
        except Exception:
            total_count = 0

        camera_count_input = getattr(self, 'camera_count_input', None)
        if camera_count_input is not None:
            try:
                camera_count_input.blockSignals(True)
                camera_count_input.setValue(max(self._canvas_count_min, visible_count))
            finally:
                camera_count_input.blockSignals(False)

        camera_total_label = getattr(self, 'camera_total_label', None)
        if camera_total_label is not None:
            camera_total_label.setText(f"/ {total_count}")

    def _on_camera_count_input_finished(self):
        camera_count_input = getattr(self, 'camera_count_input', None)
        if camera_count_input is None:
            return

        try:
            self.set_target_camera_count(int(camera_count_input.value()))
        except Exception:
            pass

    def _set_canvas_tile_size(self, size: int):
        clamped_size = max(self._canvas_tile_min, min(self._canvas_tile_max, size))
        if clamped_size == self._canvas_tile_size:
            self._update_canvas_size_controls()
            return

        previous_size = self._canvas_tile_size
        self._canvas_tile_size = clamped_size
        for canvas in self._canvas_pool:
            if canvas in self._visible_canvases and canvas.isVisible():
                self._animate_canvas_transition(
                    canvas,
                    max(1, canvas.width() or previous_size),
                    self._canvas_tile_size,
                    1.0,
                    1.0,
                )
            else:
                self._apply_canvas_tile_size_to_canvas(canvas)

        self._flow_widget.updateGeometry()
        self._canvas_host_widget.adjustSize()
        self._update_canvas_size_controls()

    def increase_canvas_size(self):
        self._set_canvas_tile_size(self._canvas_tile_size + self._canvas_tile_step)

    def decrease_canvas_size(self):
        self._set_canvas_tile_size(self._canvas_tile_size - self._canvas_tile_step)

    def _load_canvas_image(self, canvas: BaseCanvas, camera_path: str):
        if canvas.current_image_path == camera_path and canvas.active_image:
            return

        if not self._raster_manager:
            canvas._show_placeholder("No RasterManager")
            return

        try:
            raster = self._raster_manager.get_raster(camera_path)
            if raster is None:
                canvas._show_placeholder("Image not found")
                return

            q_image = raster.get_qimage()
            if q_image:
                original_width = q_image.width()
                original_height = q_image.height()
                display_q_image = q_image
                longest_edge = max(original_width, original_height)
                if longest_edge > self._canvas_tile_max:
                    display_q_image = q_image.scaled(
                        self._canvas_tile_max,
                        self._canvas_tile_max,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )

                canvas.load_visuals(
                    display_q_image,
                    camera_path,
                    raster,
                    image_dimensions=(original_width, original_height),
                )
                canvas.fit_to_image()

                # Fresh load should always restore full visibility for this canvas.
                effect = self._ensure_canvas_opacity_effect(canvas)
                effect.setOpacity(1.0)

                if self._annotation_manager:
                    annotations = self._annotation_manager.get_image_annotations(camera_path)
                    if annotations:
                        canvas._render_annotations_readonly(annotations)
                if raster.mask_annotation is not None:
                    canvas.set_mask_overlay(raster.mask_annotation)
                
                # Apply z-channel state from AnnotationWindow after loading image
                self._apply_z_channel_state_to_canvas(canvas)
            else:
                canvas._show_placeholder("Failed to load image")
        except Exception as e:
            canvas._show_placeholder(f"Error: {str(e)[:20]}")

    def _clear_canvas_perimeters(self):
        for canvas in self._canvas_pool:
            try:
                canvas.clear_perimeter_overlay()
            except Exception:
                pass

    def _apply_active_canvas_perimeter(self):
        active_path = self._active_camera_path
        if not active_path:
            return

        for canvas in self._visible_canvases:
            try:
                if canvas and canvas.active_image and canvas.current_image_path == active_path:
                    canvas.set_perimeter_overlay(SELECT_COLOR, 1)
                else:
                    canvas.clear_perimeter_overlay()
            except Exception:
                pass

    def reset_offset(self):
        scrollbar = self._scroll_area.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.minimum())

    def shift_offset(self, delta: int):
        scrollbar = self._scroll_area.verticalScrollBar()
        if scrollbar is None:
            return
        step = scrollbar.singleStep() or 20
        scrollbar.setValue(max(scrollbar.minimum(), min(scrollbar.maximum(), scrollbar.value() + delta * step)))

    def _sync_context_toolbar_scaling(self):
        icon_size = app_theme.scale_size(16)
        margin = app_theme.scale_int(5)
        spacing = app_theme.scale_int(5)

        if hasattr(self, 'toolbar'):
            self.toolbar.setIconSize(icon_size)
        if hasattr(self, 'bottom_toolbar'):
            self.bottom_toolbar.setIconSize(icon_size)

        if hasattr(self, '_top_toolbar_layout'):
            self._top_toolbar_layout.setContentsMargins(margin, margin, margin, margin)
            self._top_toolbar_layout.setSpacing(spacing)
        if hasattr(self, '_bottom_toolbar_layout'):
            self._bottom_toolbar_layout.setContentsMargins(margin, margin, margin, margin)
            self._bottom_toolbar_layout.setSpacing(spacing)

        for button_name in ('size_up_btn', 'size_down_btn', 'count_down_btn', 'count_up_btn'):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setIconSize(icon_size)

        camera_count_input = getattr(self, 'camera_count_input', None)
        if camera_count_input is not None:
            try:
                camera_count_input.setMinimumWidth(app_theme.scale_int(72))
            except Exception:
                pass

        camera_count_panel = getattr(self, 'camera_count_panel', None)
        if camera_count_panel is not None:
            try:
                camera_count_panel.setStyleSheet(
                    "QFrame#cameraCountPanel {"
                    "background-color: rgba(255, 255, 255, 0.04);"
                    "border: 1px solid rgba(255, 255, 255, 0.10);"
                    f"border-radius: {app_theme.scale_int(8)}px;"
                    "}"
                    "QFrame#cameraCountPanel QLabel {"
                    "background: transparent;"
                    "}"
                    "QFrame#cameraCountPanel QSpinBox {"
                    "background: transparent;"
                    "border: none;"
                    "padding: 0px 4px;"
                    "margin: 0px;"
                    "font-weight: bold;"
                    "}"
                )
            except Exception:
                pass

        if hasattr(self, 'stats_label'):
            self.stats_label.setStyleSheet(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; padding: 0px {app_theme.scale_int(8)}px; font-weight: bold;"
            )
        if hasattr(self, 'camera_total_label') and self.camera_total_label is not None:
            self.camera_total_label.setStyleSheet(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; padding: 0px {app_theme.scale_int(2)}px 0px 0px; font-weight: bold;"
            )

    # ==================== Toolbar (Context Matrix) ====================

    def create_top_toolbar(self) -> QToolBar:
        """Create a compact toolbar for camera loading, annotation, and size controls."""
        toolbar = QToolBar("Context Matrix Tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(app_theme.scale_size(16))
        self.toolbar = toolbar

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QHBoxLayout(container)
        self._top_toolbar_layout = layout
        layout.setContentsMargins(app_theme.scale_int(5), app_theme.scale_int(5), app_theme.scale_int(5), app_theme.scale_int(5))
        layout.setSpacing(app_theme.scale_int(5))

        self.load_btn = QToolButton()
        self.load_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.load_btn.setText("Load Cameras")
        self.load_btn.setToolTip("Load cameras or pre-load index maps for all cameras.")
        self.load_btn.setAutoRaise(True)
        def _load_btn_clicked(_checked=False):
            self.load_btn.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                self.loadCamerasRequested.emit()
            finally:
                QApplication.restoreOverrideCursor()
                QTimer.singleShot(1500, lambda: self.load_btn.setEnabled(True))

        self.load_btn.clicked.connect(_load_btn_clicked)

        load_menu = QMenu(self.load_btn)
        _act_load_cams = QAction("Load Cameras", self.load_btn)
        _act_load_cams.setToolTip("Load camera parameters from the project into the context matrix.")
        _act_load_cams.triggered.connect(lambda: self.loadCamerasRequested.emit())
        _act_load_idx = QAction("Load Index Maps", self.load_btn)
        _act_load_idx.setToolTip(
            "Attempt to load pre-computed index maps from disk cache for every loaded camera.\n"
            "Index maps are required for propagation between cameras and the mesh."
        )
        _act_load_idx.triggered.connect(lambda: self.loadIndexMapsRequested.emit())
        load_menu.addAction(_act_load_cams)
        load_menu.addAction(_act_load_idx)
        self.load_btn.setMenu(load_menu)
        layout.addWidget(self.load_btn)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.VLine)
        sep0.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep0)

        self._multi_annotate_btn = QToolButton()
        self._multi_annotate_btn.setText("Multi-Annotate")
        self._multi_annotate_btn.setCheckable(True)
        self._multi_annotate_btn.setChecked(False)
        self._multi_annotate_btn.setToolTip("Multi-Camera Annotation")
        self._multi_annotate_btn.setAutoRaise(True)
        self._multi_annotate_btn.toggled.connect(self._on_multi_annotate_toggled)
        layout.addWidget(self._multi_annotate_btn)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # Split-button: left area triggers the selected propagation mode;
        # arrow area opens a dropdown to choose the mode.
        self._propagate_mask_btn = QToolButton()
        self._propagate_mask_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._propagate_mask_btn.setAutoRaise(True)
        self._propagate_mask_btn.clicked.connect(self._on_propagate_mask_clicked)

        propagate_menu = QMenu(self._propagate_mask_btn)

        act_primary_secondary = QAction("Primary → Secondary Cameras", self._propagate_mask_btn)
        act_primary_secondary.setData(PROPAGATE_ACTIVE_TO_CONTEXT)
        act_primary_secondary.triggered.connect(
            lambda: self._on_propagate_mode_selected(PROPAGATE_ACTIVE_TO_CONTEXT)
        )

        act_primary_mesh = QAction("Primary → Mesh", self._propagate_mask_btn)
        act_primary_mesh.setData(PROPAGATE_ACTIVE_CAMERA_TO_MESH)
        act_primary_mesh.triggered.connect(
            lambda: self._on_propagate_mode_selected(PROPAGATE_ACTIVE_CAMERA_TO_MESH)
        )

        act_cameras_mesh = QAction("All Cameras → Mesh", self._propagate_mask_btn)
        act_cameras_mesh.setData(PROPAGATE_CAMERAS_TO_MESH)
        act_cameras_mesh.triggered.connect(
            lambda: self._on_propagate_mode_selected(PROPAGATE_CAMERAS_TO_MESH)
        )

        act_mesh_primary = QAction("Mesh → Primary", self._propagate_mask_btn)
        act_mesh_primary.setData(PROPAGATE_MESH_TO_ACTIVE_CAMERA)
        act_mesh_primary.triggered.connect(
            lambda: self._on_propagate_mode_selected(PROPAGATE_MESH_TO_ACTIVE_CAMERA)
        )

        act_mesh_cameras = QAction("Mesh → All Cameras", self._propagate_mask_btn)
        act_mesh_cameras.setData(PROPAGATE_MESH_TO_CAMERAS)
        act_mesh_cameras.triggered.connect(
            lambda: self._on_propagate_mode_selected(PROPAGATE_MESH_TO_CAMERAS)
        )

        propagate_menu.addAction(act_primary_secondary)
        propagate_menu.addSeparator()
        propagate_menu.addAction(act_primary_mesh)
        propagate_menu.addAction(act_cameras_mesh)
        propagate_menu.addSeparator()
        propagate_menu.addAction(act_mesh_primary)
        propagate_menu.addAction(act_mesh_cameras)
        self._propagate_mask_btn.setMenu(propagate_menu)

        # Default mode shown on the button face
        self._propagate_current_mode = PROPAGATE_ACTIVE_TO_CONTEXT
        self._propagate_mask_btn.setText(
            _PROPAGATE_MODE_LABELS.get(PROPAGATE_ACTIVE_TO_CONTEXT, "Propagate")
        )
        self._propagate_mask_btn.setToolTip(
            "Propagate the primary image\u2019s mask to secondary context cameras.\n"
            "Use the arrow to switch mode; click to run."
        )
        layout.addWidget(self._propagate_mask_btn)

        layout.addStretch(1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        self.size_up_btn = QToolButton()
        self.size_up_btn.setIcon(get_icon("up_chevron.svg"))
        self.size_up_btn.setToolTip("Larger cameras")
        self.size_up_btn.setAutoRaise(True)
        self.size_up_btn.clicked.connect(lambda _checked=False: self.increase_canvas_size())
        layout.addWidget(self.size_up_btn)

        self.size_down_btn = QToolButton()
        self.size_down_btn.setIcon(get_icon("down_chevron.svg"))
        self.size_down_btn.setToolTip("Smaller cameras")
        self.size_down_btn.setAutoRaise(True)
        self.size_down_btn.clicked.connect(lambda _checked=False: self.decrease_canvas_size())
        layout.addWidget(self.size_down_btn)

        toolbar.addWidget(container)

        self._sync_context_toolbar_scaling()
        self.set_scene_controls_enabled(self._scene_controls_enabled)
        return toolbar

    def create_bottom_toolbar(self) -> QToolBar:
        """Create a bottom toolbar for camera statistics and count controls."""
        toolbar = QToolBar("Context Matrix Stats")
        toolbar.setMovable(False)
        toolbar.setIconSize(app_theme.scale_size(16))
        self.bottom_toolbar = toolbar

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QHBoxLayout(container)
        self._bottom_toolbar_layout = layout
        layout.setContentsMargins(app_theme.scale_int(5), app_theme.scale_int(5), app_theme.scale_int(5), app_theme.scale_int(5))
        layout.setSpacing(app_theme.scale_int(5))

        self.camera_count_panel = QFrame()
        self.camera_count_panel.setObjectName("cameraCountPanel")
        self.camera_count_panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        camera_count_layout = QHBoxLayout(self.camera_count_panel)
        camera_count_layout.setContentsMargins(
            app_theme.scale_int(8),
            app_theme.scale_int(4),
            app_theme.scale_int(8),
            app_theme.scale_int(4),
        )
        camera_count_layout.setSpacing(0)

        self.stats_label = QLabel("Cameras")
        self.stats_label.setStyleSheet(
            f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; padding: 0px {app_theme.scale_int(8)}px 0px 0px; font-weight: bold;"
        )
        self.stats_label.setToolTip("Visible cameras in the matrix / total loaded cameras")

        self.camera_count_input = QSpinBox()
        self.camera_count_input.setToolTip("Type a camera count and press Enter to load that many cameras.")
        self.camera_count_input.setAlignment(Qt.AlignCenter)
        self.camera_count_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.camera_count_input.setKeyboardTracking(False)
        self.camera_count_input.setRange(self._canvas_count_min, self._canvas_count_min)
        self.camera_count_input.setValue(self._canvas_count_min)
        self.camera_count_input.editingFinished.connect(self._on_camera_count_input_finished)

        self.camera_total_label = QLabel("/ 0")
        self.camera_total_label.setToolTip("Total loaded cameras")
        self.camera_total_label.setStyleSheet(
            f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; padding: 0px 0px 0px {app_theme.scale_int(2)}px; font-weight: bold;"
        )

        camera_count_layout.addWidget(self.stats_label)
        camera_count_layout.addWidget(self.camera_count_input)
        camera_count_layout.addWidget(self.camera_total_label)

        layout.addWidget(self.camera_count_panel)

        layout.addStretch(1)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.VLine)
        sep0.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep0)

        self.count_down_btn = QToolButton()
        self.count_down_btn.setIcon(get_icon("left_chevron.svg"))
        self.count_down_btn.setToolTip("Show fewer cameras")
        self.count_down_btn.setAutoRaise(True)
        self.count_down_btn.clicked.connect(lambda _checked=False: self.decrease_canvas_count())
        layout.addWidget(self.count_down_btn)

        self.count_up_btn = QToolButton()
        self.count_up_btn.setIcon(get_icon("right_chevron.svg"))
        self.count_up_btn.setToolTip("Show more cameras")
        self.count_up_btn.setAutoRaise(True)
        self.count_up_btn.clicked.connect(lambda _checked=False: self.increase_canvas_count())
        layout.addWidget(self.count_up_btn)

        toolbar.addWidget(container)

        self._sync_context_toolbar_scaling()
        self.set_scene_controls_enabled(self._scene_controls_enabled)
        return toolbar

    def update_stats_label(self, visible_count: int, overlapping_count: int):
        """Updates the bottom toolbar text.

        The camera-count cap is intentionally NOT applied here so users can
        increase the visible count up to the total number of loaded cameras.
        The denominator shown in the toolbar always reflects the total.
        """
        total_count = len(self._camera_paths) if self._camera_paths else 0
        self._sync_camera_status_label(visible_count, total_count)
        self._update_canvas_count_controls()

    def refresh_scaling(self):
        """Refresh toolbar sizing after a UI scale change."""
        self._sync_context_toolbar_scaling()

    def _on_propagate_mode_selected(self, mode: str):
        """Called from the dropdown — switch the active mode and update tooltip."""
        self._propagate_current_mode = mode
        self._propagate_mask_btn.setText(
            _PROPAGATE_MODE_LABELS.get(mode, "Propagate")
        )
        self._update_propagate_btn_tooltip(mode)

    def _update_propagate_btn_tooltip(self, mode: str = None):
        """Rebuild the propagate-button tooltip with live camera counts."""
        if mode is None:
            mode = getattr(self, '_propagate_current_mode', PROPAGATE_ACTIVE_TO_CONTEXT)
        counts = self._get_propagation_counts() or {}
        total        = counts.get('total', 0)
        have_idx     = counts.get('have_index_map', 0)
        have_mask    = counts.get('have_mask', 0)

        if mode == PROPAGATE_ACTIVE_TO_CONTEXT:
            tip = (
                "Propagate the primary image's semantic mask to all visible\n"
                "secondary (context) cameras via the 3D mesh index maps.\n"
                "Use the arrow to switch mode; click to run."
            )
        elif mode == PROPAGATE_ACTIVE_CAMERA_TO_MESH:
            tip = (
                "Aggregate the primary camera's semantic mask onto the 3D mesh\n"
                "using majority-vote conflict resolution.\n"
                "Requires an index map for the primary camera.\n"
                "Use the arrow to switch mode; click to run."
            )
        elif mode == PROPAGATE_CAMERAS_TO_MESH:
            tip = (
                f"Aggregate semantic masks from all cameras onto the 3D mesh\n"
                f"using majority-vote conflict resolution.\n"
                f"{have_mask} of {total} camera(s) have both an index map and a labeled mask.\n"
                "Use the arrow to switch mode; click to run."
            )
        elif mode == PROPAGATE_MESH_TO_ACTIVE_CAMERA:
            tip = (
                "Project the 3D mesh's face labels to the primary camera's semantic mask.\n"
                "Requires an index map for the primary camera.\n"
                "Unlabeled mesh faces are skipped (existing labels preserved).\n"
                "Use the arrow to switch mode; click to run."
            )
        elif mode == PROPAGATE_MESH_TO_CAMERAS:
            tip = (
                f"Project the 3D mesh's face labels to every camera's semantic mask.\n"
                f"{have_idx} of {total} camera(s) have an index map and will receive labels.\n"
                "Unlabeled mesh faces are skipped (existing pixel labels preserved).\n"
                "Use the arrow to switch mode; click to run."
            )
        else:
            tip = "Use the arrow to switch propagation mode; click to run."

        self._propagate_mask_btn.setToolTip(tip)

    def _on_propagate_mask_clicked(self, _checked=False):
        """Called when the button face (not the arrow) is clicked."""
        self._run_propagate_mode(self._propagate_current_mode)

    def _get_propagation_counts(self):
        """Return camera index-map / mask counts from the MVATManager, or None."""
        mgr = getattr(self, '_mvat_manager', None)
        if mgr is None:
            return None
        try:
            return mgr.get_propagation_camera_counts()
        except Exception:
            return None

    def _run_propagate_mode(self, mode: str):
        """Emit maskPropagationRequested for the selected propagation mode.

        Dialogs have been removed.  Behaviors are hardcoded:
        - Cameras → Mesh: always run without the "also project back" option
          (user can press the Mesh → Cameras button separately afterwards).
        - Mesh → Cameras / Primary: always skip unlabeled faces so existing
          pixel labels are preserved.

        The button is disabled and a WaitCursor shown immediately so the user
        gets visual feedback.  The cursor is restored once the signal handler
        returns (background work continues asynchronously).  The button is
        re-enabled via a short QTimer delay to prevent accidental double-clicks.
        """
        btn = getattr(self, '_propagate_mask_btn', None)
        if btn is not None:
            btn.setEnabled(False)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            if mode == PROPAGATE_ACTIVE_TO_CONTEXT:
                self.semanticMaskPropagationRequested.emit()   # back-compat
                self.maskPropagationRequested.emit(mode)

            elif mode == PROPAGATE_ACTIVE_CAMERA_TO_MESH:
                self.maskPropagationRequested.emit(mode)

            elif mode == PROPAGATE_CAMERAS_TO_MESH:
                # No "also_project" — user can chain manually
                self.maskPropagationRequested.emit(mode)

            elif mode == PROPAGATE_MESH_TO_ACTIVE_CAMERA:
                # Always skip unlabeled mesh faces
                self.maskPropagationRequested.emit(mode + ":skip_unlabeled")

            elif mode == PROPAGATE_MESH_TO_CAMERAS:
                # Always skip unlabeled mesh faces
                self.maskPropagationRequested.emit(mode + ":skip_unlabeled")
        finally:
            QApplication.restoreOverrideCursor()
            # Re-enable after a short delay to absorb any accidental double-click.
            # Background propagation work may still be running — the status bar
            # will show completion.
            if btn is not None:
                QTimer.singleShot(1500, lambda: btn.setEnabled(self._scene_controls_enabled))

    def update_stats(self, perspective_count: int):
        return

    def update_selection_labels(self, active_label: str, highlighted_count: int):
        return

    def _on_multi_annotate_toggled(self, checked: bool):
        self.multi_annotate_enabled = checked
        self.multiAnnotateToggled.emit(checked)

    # ==================== Marker Routing (Phase 4) ====================

    def _get_canvas_camera_map(self) -> Dict[str, 'BaseCanvas']:
        result = {}
        for canvas in self._visible_canvases:
            if canvas and canvas.current_image_path and canvas.active_image:
                result[canvas.current_image_path] = canvas
        return result

    def reorder_canvases_by_visibility(self, visible_paths: set):
        """Move canvases whose camera is in *visible_paths* to the front of the
        flow layout, pushing dim/out-of-FOV canvases to the end.

        Only reorders when the current order differs from the desired order so
        the layout is not needlessly invalidated on every navigation event.
        """
        if not self._visible_canvases:
            return

        front = [c for c in self._visible_canvases if c.current_image_path in visible_paths]
        back  = [c for c in self._visible_canvases if c.current_image_path not in visible_paths]
        desired = front + back

        if desired == self._visible_canvases:
            return

        # Drain the layout's item list, keeping the existing QLayoutItems so we
        # don't create duplicate QWidgetItems (addWidget on an already-managed
        # widget appends a second item without removing the first).
        items = {}
        while self._flow_layout.count():
            item = self._flow_layout.takeAt(0)
            if item and item.widget():
                items[item.widget()] = item

        for canvas in desired:
            item = items.get(canvas)
            if item is not None:
                self._flow_layout.addItem(item)

        self._visible_canvases = desired

        # Force an immediate re-layout pass so positions update this frame.
        self._flow_layout.invalidate()
        self._flow_layout.activate()

    def update_dynamic_markers(self, projections: dict, accuracies: dict,
                                visibility_status: dict):
        canvas_map = self._get_canvas_camera_map()
        for path, canvas in canvas_map.items():
            proj = projections.get(path)
            if not proj:
                canvas.clear_dynamic_marker()
                continue

            px, py, is_valid = proj
            if not is_valid:
                canvas.clear_dynamic_marker()
                continue

            acc = accuracies.get(path, False)
            is_occluded = visibility_status.get(path, False)
            color = MARKER_COLOR_HIGHLIGHTED if (acc and not is_occluded) else MARKER_COLOR_INVALID
            canvas.update_dynamic_marker(px, py, color=color, is_valid=(acc and not is_occluded))

    def clear_all_dynamic_markers(self):
        for canvas in self._canvas_pool:
            canvas.clear_dynamic_marker()

    def update_static_markers_from_3d(self, point_3d, cameras: dict):
        self._last_focal_point = point_3d
        self._cameras_ref = cameras

        canvas_map = self._get_canvas_camera_map()
        for path, canvas in canvas_map.items():
            camera = cameras.get(path)
            if not camera:
                canvas.clear_static_marker()
                continue
            try:
                pixel = camera.project(point_3d)
            except Exception:
                canvas.clear_static_marker()
                continue

            if np.isnan(pixel).any():
                canvas.clear_static_marker()
                continue

            u, v = float(pixel[0]), float(pixel[1])

            # Explicit bounds check: hide if projected outside this camera's image
            cam_w = getattr(camera, 'width', 0)
            cam_h = getattr(camera, 'height', 0)
            if cam_w and cam_h and not (0 <= u < cam_w and 0 <= v < cam_h):
                canvas.clear_static_marker()
                continue

            # Static focal-point markers are always valid surface picks — always green.
            # Occlusion testing (is_point_occluded_depth_based) produces false-positives
            # for MVATViewer picks due to depth-buffer floating-point imprecision and is
            # only appropriate for dynamic hover markers, not for static focal-point marks.
            canvas.update_static_marker(u, v, color=MARKER_COLOR_HIGHLIGHTED)

    def clear_all_static_markers(self):
        self._last_focal_point = None
        for canvas in self._canvas_pool:
            canvas.clear_static_marker()

    # ==================== Target-Lock Sync (Phase 5) ====================

    def set_mvat_manager(self, manager):
        self._mvat_manager = manager
        self._roll_cache.clear()  # camera set changed; invalidate cached roll angles

    def request_sync(self, targets: dict, zoom_factor: float, reference_path: str = None, base_rotation: float = 0.0):
        self.sync_to_targets(targets, zoom_factor, reference_path, base_rotation)

    def request_zoom_only(self, canvas_indices: set, zoom_factor: float, reference_path: str = None, base_rotation: float = 0.0):
        """Apply zoom synchronisation without changing the viewport center.

        Used for canvases where the 3D world point falls outside the image FOV.
        All visible canvases will share the same relative zoom level even when
        their center pixel cannot be determined from the shared world point.
        """
        if not self.target_lock_enabled:
            return

        ref_cam = self._mvat_manager.cameras.get(reference_path) if (reference_path and self._mvat_manager) else None

        for i in canvas_indices:
            if i < len(self._canvas_pool):
                canvas = self._canvas_pool[i]
                if canvas.isVisible() and canvas.active_image:
                    absolute_zoom = canvas._min_zoom * zoom_factor

                    total_angle = base_rotation

                    if ref_cam is not None:
                        ctx_cam = self._mvat_manager.cameras.get(canvas.current_image_path)
                        if ctx_cam is not None:
                            cache_key = (reference_path, canvas.current_image_path)
                            camera_roll = self._roll_cache.get(cache_key)
                            if camera_roll is None:
                                up_ref_cam = np.array([0.0, -1.0, 0.0])
                                up_world = ref_cam.R.T @ up_ref_cam
                                up_ctx_cam = ctx_cam.R @ up_world
                                alpha = np.degrees(np.arctan2(up_ctx_cam[1], up_ctx_cam[0]))
                                camera_roll = -90.0 - alpha
                                self._roll_cache[cache_key] = camera_roll
                            total_angle = base_rotation + camera_roll

                    canvas.set_zoom_level(absolute_zoom)
                    canvas._set_absolute_rotation(total_angle)

                    # Dim to indicate the focal target is out of this camera's FOV.
                    effect = self._ensure_canvas_opacity_effect(canvas)
                    effect.setOpacity(0.25)

    def sync_to_targets(self, targets: dict, zoom_factor: float, reference_path: str = None, base_rotation: float = 0.0):
        if not self.target_lock_enabled or not self._mvat_manager:
            return

        ref_cam = self._mvat_manager.cameras.get(reference_path) if reference_path else None

        visible_paths = set()

        for i, (target_x, target_y) in targets.items():
            if i < len(self._canvas_pool):
                canvas = self._canvas_pool[i]
                if canvas.isVisible() and canvas.active_image:

                    total_angle = base_rotation

                    if ref_cam is not None:
                        ctx_cam = self._mvat_manager.cameras.get(canvas.current_image_path)
                        if ctx_cam is not None:
                            cache_key = (reference_path, canvas.current_image_path)
                            camera_roll = self._roll_cache.get(cache_key)
                            if camera_roll is None:
                                up_ref_cam = np.array([0.0, -1.0, 0.0])
                                up_world = ref_cam.R.T @ up_ref_cam
                                up_ctx_cam = ctx_cam.R @ up_world
                                alpha = np.degrees(np.arctan2(up_ctx_cam[1], up_ctx_cam[0]))
                                camera_roll = -90.0 - alpha
                                self._roll_cache[cache_key] = camera_roll
                            total_angle = base_rotation + camera_roll

                    # Snap to target with the synchronized rotation
                    canvas.snap_to_target(target_x, target_y, zoom_factor, angle_degrees=total_angle)

                    # Restore full opacity because target is visible in this canvas.
                    effect = self._ensure_canvas_opacity_effect(canvas)
                    effect.setOpacity(1.0)

                    if canvas.current_image_path:
                        visible_paths.add(canvas.current_image_path)

        # Float fully-visible canvases to the front of the layout.
        self.reorder_canvases_by_visibility(visible_paths)

    def _request_sync_from_main_view(self):
        if not self._mvat_manager:
            return
        aw = self._mvat_manager.annotation_window
        if not aw.active_image or not aw.pixmap_image:
            return

        viewport_center = aw.mapToScene(aw.viewport().rect().center())
        center_x, center_y = viewport_center.x(), viewport_center.y()
        zoom_factor = aw.zoom_factor
        self._mvat_manager._on_main_view_navigated(center_x, center_y, zoom_factor)

    # ==================== Rank Indicator ====================

    def _update_rank_label(self):
        return

    # ==================== Annotation Visualization (Phase 6) ====================

    def clear_all_annotation_overlays(self):
        """Synchronously remove every annotation overlay item from every
        visible canvas.

        Call this BEFORE starting any batch operation that removes annotation
        objects (e.g. bake + delete).  By clearing the overlays while all
        objects are still alive and consistent, we guarantee that no
        QGraphicsDropShadowEffect item remains in a canvas scene when Qt
        later coalesces window repaints — eliminating the C-level crash that
        occurs when Qt tries to render a shadow effect whose item is being
        torn down.
        """
        for canvas in self._visible_canvases:
            if canvas is not None:
                try:
                    canvas._clear_readonly_annotations()
                except Exception:
                    pass

    def suspend_annotation_updates(self):
        """Suspend canvas refreshes driven by annotation-manager signals.

        Call resume_annotation_updates() when the batch operation is complete.
        """
        self._annotation_updates_suspended = True

    def resume_annotation_updates(self):
        """Re-enable annotation change callbacks and queue one deferred refresh.

        The refresh is posted via QTimer.singleShot(0) so it runs only after
        the current call-stack fully unwinds and all pending signals settle.
        """
        self._annotation_updates_suspended = False
        QTimer.singleShot(0, self._deferred_refresh_all_canvases)

    def _deferred_refresh_all_canvases(self):
        """Refresh every visible canvas once after a suspended batch operation."""
        if not self._annotation_manager:
            return
        for canvas in self._visible_canvases:
            if canvas and canvas.active_image and canvas.current_image_path:
                path = canvas.current_image_path
                annotations = self._annotation_manager.get_image_annotations(path)
                canvas._render_annotations_readonly(annotations)
                if self._raster_manager:
                    raster = self._raster_manager.get_raster(path)
                    if raster is not None and raster.mask_annotation is not None:
                        canvas.set_mask_overlay(raster.mask_annotation)

    def set_annotation_manager(self, manager):
        self._annotation_manager = manager
        if manager is None:
            return
        manager.annotationAdded.connect(self._on_annotation_changed)
        manager.annotationRemoved.connect(self._on_annotation_changed)
        manager.annotationModified.connect(self._on_annotation_changed)
        manager.annotationLabelChanged.connect(
            lambda ann_id: self._on_annotation_changed(ann_id)
        )
        manager.annotationsAdded.connect(self._on_annotations_changed)
        manager.annotationsRemoved.connect(self._on_annotations_changed)
        manager.selectionChanged.connect(self._on_selection_changed)

    def _on_annotation_changed(self, annotation_id):
        if self._annotation_updates_suspended or not self._annotation_manager:
            return
        annotation = self._annotation_manager.annotations_dict.get(annotation_id)
        affected_path = annotation.image_path if annotation else None
        self._refresh_annotations_for_path(affected_path)

    def _on_annotations_changed(self, annotation_ids):
        if self._annotation_updates_suspended or not self._annotation_manager:
            return
        affected_paths = set()
        for ann_id in annotation_ids:
            annotation = self._annotation_manager.annotations_dict.get(ann_id)
            if annotation:
                affected_paths.add(annotation.image_path)
        if affected_paths:
            for path in affected_paths:
                self._refresh_annotations_for_path(path)
        else:
            self._refresh_annotations_for_path(None)

    def _refresh_annotations_for_path(self, image_path):
        if not self._annotation_manager:
            return
        for canvas in self._visible_canvases:
            if canvas and canvas.active_image and canvas.current_image_path:
                if image_path is None or canvas.current_image_path == image_path:
                    annotations = self._annotation_manager.get_image_annotations(
                        canvas.current_image_path
                    )
                    canvas._render_annotations_readonly(annotations)
                    if self._raster_manager:
                        raster = self._raster_manager.get_raster(canvas.current_image_path)
                        if raster is not None and raster.mask_annotation is not None:
                            canvas.set_mask_overlay(raster.mask_annotation)

    def _on_selection_changed(self, selected_ids):
        if self._annotation_updates_suspended:
            return
        selected_set = set(selected_ids) if selected_ids else set()
        for canvas in self._visible_canvases:
            if canvas and canvas.active_image:
                for item in canvas._readonly_annotation_items:
                    ann_id = getattr(item, '_source_annotation_id', None)
                    if ann_id:
                        canvas._highlight_readonly_annotation(
                            ann_id, ann_id in selected_set
                        )

    def update_cursor_previews(self, projections, visible_paths, item_factory):
        """Show a tool cursor preview on each visible context canvas."""
        canvas_map = self._get_canvas_camera_map()
        for path, canvas in canvas_map.items():
            if path not in visible_paths:
                canvas.clear_cursor_preview()
                continue
            proj = projections.get(path)
            if not proj:
                canvas.clear_cursor_preview()
                continue
            u, v, is_valid = proj
            if not is_valid:
                canvas.clear_cursor_preview()
                continue
            target_camera = None
            if self._mvat_manager is not None:
                target_camera = self._mvat_manager.cameras.get(path)
            if target_camera is not None:
                if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                    canvas.clear_cursor_preview()
                    continue
            canvas.update_cursor_preview(u, v, item_factory)

    def clear_all_cursor_previews(self):
        """Hide cursor previews on all canvases in the pool."""
        for canvas in self._canvas_pool:
            canvas.clear_cursor_preview()

    def _apply_z_channel_state_to_canvas(self, canvas):
        """
        Apply the current z-channel state from AnnotationWindow to a specific canvas.
        Called after a canvas loads a new image to ensure z-channel visualization matches.
        """
        try:
            annotation_window = (
                getattr(self._mvat_manager.main_window, 'annotation_window', None)
                if self._mvat_manager else None
            )
            if not annotation_window:
                return
            current_colormap = getattr(annotation_window.main_window, 'z_colormap_dropdown', None)
            if not current_colormap:
                return
            colormap_name = current_colormap.currentText()
            if colormap_name != 'None':
                canvas.update_z_colormap(colormap_name)
                z_transparency = getattr(annotation_window.main_window, 'z_transparency_widget', None)
                if z_transparency:
                    opacity = z_transparency.value() / 255.0
                    canvas.set_z_opacity(opacity)
                z_dynamic = getattr(annotation_window.main_window, 'z_dynamic_scaling_checkbox', None)
                if z_dynamic:
                    is_dynamic = z_dynamic.isChecked()
                    canvas.toggle_dynamic_z_scaling(is_dynamic)
                    return
        except Exception:
            pass

    def sync_z_colormap_to_all_canvases(self, colormap_name):
        """Broadcast z-channel colormap changes to all visible canvases."""
        for canvas in self._canvas_pool:
            canvas.update_z_colormap(colormap_name)

    def sync_z_opacity_to_all_canvases(self, opacity):
        """Broadcast z-channel opacity changes to all visible canvases."""
        opacity = max(0.0, min(1.0, opacity))
        for canvas in self._canvas_pool:
            canvas.set_z_opacity(opacity)

    def sync_annotations_to_all_canvases(self):
        """Re-render readonly annotation overlays on all visible canvases (e.g., after transparency change)."""
        self._refresh_annotations_for_path(None)

    def sync_z_dynamic_scaling_to_all_canvases(self, enabled):
        """Broadcast z-channel dynamic scaling toggle to all visible canvases."""
        for canvas in self._canvas_pool:
            canvas.toggle_dynamic_z_scaling(enabled)
