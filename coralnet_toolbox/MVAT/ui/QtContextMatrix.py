"""
Context Matrix Widget for MVAT (Unified Flow Gallery)

Displays a dynamic, auto-flowing gallery of interactive BaseCanvas viewports.
Replaces the legacy CameraGrid.

Features include:
- Dynamic object pool of BaseCanvas instances.
- Auto-flow layout adaptation based on panel width and a user-controlled camera count.
- Image loading from RasterManager.
- Viewer-only camera navigation from clicks and arrow buttons.
"""

import warnings
import numpy as np
from typing import List, Optional, Dict

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QToolBar, QToolButton, QSizePolicy, QFrame,
    QScrollArea, QLayout, QLayoutItem
)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.QtBaseCanvas import BaseCanvas
from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_INVALID,
)

from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        loadCamerasRequested: Emitted when the Load Cameras button is clicked
        clearSelectionsRequested: Legacy compatibility signal; kept for future workflows.
        visibleCamerasChanged: Emitted when the visible canvas set changes.
    """

    contextImagePromoted = pyqtSignal(str)            # camera_path
    rankIndicatorUpdated = pyqtSignal(int, int, int)  # start, end, total
    multiAnnotateToggled = pyqtSignal(bool)           # enabled state

    # Migrated from legacy CameraGrid
    loadCamerasRequested = pyqtSignal()
    clearSelectionsRequested = pyqtSignal()
    previousCameraRequested = pyqtSignal()
    nextCameraRequested = pyqtSignal()
    visibleCamerasChanged = pyqtSignal(list)

    # Selection intent signals (mirror CameraGrid's paradigm)
    selection_requested = pyqtSignal(list)   # request to set highlight selection (list of paths)
    toggle_requested = pyqtSignal(str)       # request to toggle a single path (Ctrl+Click)
    active_requested = pyqtSignal(str)       # legacy compatibility signal; unused by the current UI
    camera_highlighted_single = pyqtSignal(str)  # single plain-click -> jump 3D view

    def __init__(self, parent=None):
        super().__init__(parent)

        # Matrix state
        self.target_camera_count = 10
        self._last_rebuilt_count = 0
        self._canvas_count_step = 1
        self._canvas_count_min = 1
        self._canvas_tile_size = 240
        self._canvas_tile_step = 32
        self._canvas_tile_min = 160
        self._canvas_tile_max = 400

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

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._pending_sync = None

        # Annotation visualization state (Phase 6)
        self._annotation_manager = None

        # Selection tracking for Shift+Click range selection
        self._last_clicked_path: Optional[str] = None

        # Canvas pool
        self._canvas_pool: List[BaseCanvas] = []
        self._visible_canvases: List[BaseCanvas] = []

        # Scrollable flow-layout gallery
        self._canvas_host_widget = QWidget(self)
        self._canvas_host_layout = QVBoxLayout(self._canvas_host_widget)
        self._canvas_host_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_host_layout.setSpacing(0)

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
            self._canvas_pool.append(canvas)

    def _make_canvas_mouse_press_handler(self, canvas: BaseCanvas):
        """Intercept left clicks for viewer-only navigation while preserving native canvas interactions."""
        def handler(event):
            if event.button() == Qt.LeftButton:
                path = canvas.current_image_path
                if path:
                    modifiers = event.modifiers()

                    if modifiers == Qt.ControlModifier:
                        # Ctrl+Click: toggle highlight
                        self.toggle_requested.emit(path)

                    elif modifiers == Qt.ShiftModifier and self._last_clicked_path:
                        # Shift+Click: range select
                        try:
                            start_idx = self._camera_paths.index(self._last_clicked_path)
                            end_idx = self._camera_paths.index(path)
                            start, end = min(start_idx, end_idx), max(start_idx, end_idx)
                            self.selection_requested.emit(self._camera_paths[start:end + 1])
                        except ValueError:
                            pass

                    else:
                        # Plain click: update the 3D viewer only.
                        self.camera_highlighted_single.emit(path)

                    self._last_clicked_path = path

            # CRITICAL: pass through so drawing/panning tools still work
            BaseCanvas.mousePressEvent(canvas, event)
        return handler

    def _make_canvas_double_click_handler(self, canvas: BaseCanvas):
        """Double-click updates the 3D viewer only."""
        def handler(event):
            if event.button() == Qt.LeftButton:
                path = canvas.current_image_path
                if path:
                    self.camera_highlighted_single.emit(path)
            BaseCanvas.mouseDoubleClickEvent(canvas, event)
        return handler

    # ==================== Input / Scroll Events ====================

    def set_target_camera_count(self, count: int):
        """Update the desired number of visible context canvases."""
        try:
            count = int(count)
        except Exception:
            count = self._canvas_count_min

        self.target_camera_count = max(self._canvas_count_min, count)
        self._evaluate_auto_layout()
        self._refresh_visible_canvases()

    def increase_canvas_count(self):
        self.set_target_camera_count(self.target_camera_count + self._canvas_count_step)

    def decrease_canvas_count(self):
        self.set_target_camera_count(self.target_camera_count - self._canvas_count_step)

    # ==================== Layout Rebuilding ====================

    def _evaluate_auto_layout(self):
        """Rebuild the flow layout for the current camera count target."""
        available = len(self._camera_paths)
        effective_target = min(available, self.target_camera_count) if available > 0 else 0

        if effective_target != self._last_rebuilt_count:
            self._rebuild_layout(effective_target)
        else:
            self._update_empty_state_visibility(effective_target > 0)

    def _rebuild_layout(self, count: int):
        """Rebuild the flow layout with the requested number of visible canvases."""
        self._last_rebuilt_count = count

        while self._flow_layout.count():
            item = self._flow_layout.takeAt(0)
            if item and item.widget():
                item.widget().hide()

        for canvas in self._canvas_pool:
            canvas.hide()

        self._visible_canvases = []

        if count <= 0:
            self._update_empty_state_visibility(False)
            return

        self._ensure_canvas_pool_size(count)

        for index in range(count):
            canvas = self._canvas_pool[index]
            self._flow_layout.addWidget(canvas)
            canvas.show()
            self._visible_canvases.append(canvas)

        self._update_empty_state_visibility(True)
        self._flow_widget.updateGeometry()
        self._canvas_host_widget.adjustSize()

    def resizeEvent(self, event):
        """Auto-adjust layout on resize without changing camera count."""
        super().resizeEvent(event)
        self._flow_widget.updateGeometry()

    # ==================== Data Feed ====================

    def set_raster_manager(self, raster_manager):
        self._raster_manager = raster_manager

    def set_camera_data(self, camera_objects: List, ordered_paths: List[str]):
        self._camera_paths = list(ordered_paths)
        self._evaluate_auto_layout()
        self._refresh_visible_canvases()

    def set_camera_order(self, ordered_paths: List[str], active_path: str = None):
        paths = list(ordered_paths)
        if active_path and active_path in paths and paths and paths[0] != active_path:
            paths = [active_path] + [path for path in paths if path != active_path]
        self._camera_paths = paths
        self._evaluate_auto_layout()
        self._refresh_visible_canvases()

    def get_camera_order(self) -> List[str]:
        return list(self._camera_paths)

    def get_visible_camera_paths(self) -> List[str]:
        return [
            canvas.current_image_path
            for canvas in self._visible_canvases
            if canvas and canvas.active_image and canvas.current_image_path
        ]

    def _emit_visible_cameras_changed(self):
        self.visibleCamerasChanged.emit(self.get_visible_camera_paths())

    def _update_empty_state_visibility(self, has_cameras: bool):
        self._placeholder_label.setVisible(not has_cameras)
        self._flow_widget.setVisible(has_cameras)

    def _refresh_visible_canvases(self):
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

        self._update_canvas_count_controls()
        self._emit_visible_cameras_changed()

    def _get_visible_capacity(self) -> int:
        return self._last_rebuilt_count

    def _apply_canvas_tile_size_to_canvas(self, canvas: BaseCanvas):
        canvas.setFixedSize(self._canvas_tile_size, self._canvas_tile_size)

    def _update_canvas_size_controls(self):
        if hasattr(self, 'size_down_btn'):
            self.size_down_btn.setEnabled(self._canvas_tile_size > self._canvas_tile_min)
        if hasattr(self, 'size_up_btn'):
            self.size_up_btn.setEnabled(self._canvas_tile_size < self._canvas_tile_max)

    def _update_canvas_count_controls(self):
        available = len(self._camera_paths)
        can_decrease = available > 0 and self.target_camera_count > self._canvas_count_min
        can_increase = available > 0 and self.target_camera_count < available
        if hasattr(self, 'count_down_btn'):
            self.count_down_btn.setEnabled(can_decrease)
        if hasattr(self, 'count_up_btn'):
            self.count_up_btn.setEnabled(can_increase)

    def _set_canvas_tile_size(self, size: int):
        clamped_size = max(self._canvas_tile_min, min(self._canvas_tile_max, size))
        if clamped_size == self._canvas_tile_size:
            self._update_canvas_size_controls()
            return

        self._canvas_tile_size = clamped_size
        for canvas in self._canvas_pool:
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
                canvas.load_visuals(q_image, camera_path, raster)
                canvas.fit_to_image()
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

    # ==================== Toolbar (Context Matrix) ====================

    def create_top_toolbar(self) -> QToolBar:
        """Create a compact toolbar for camera loading, navigation, size, and count controls."""
        toolbar = QToolBar("Context Matrix Tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(app_theme.scale_size(16))
        self.toolbar = toolbar

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.load_btn = QToolButton()
        self.load_btn.setText("Load Cameras")
        self.load_btn.setAutoRaise(True)
        self.load_btn.clicked.connect(lambda _checked=False: self.loadCamerasRequested.emit())
        layout.addWidget(self.load_btn)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.VLine)
        sep0.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep0)

        self.previous_btn = QToolButton()
        self.previous_btn.setIcon(get_icon("left.svg"))
        self.previous_btn.setToolTip("Previous camera")
        self.previous_btn.setAutoRaise(True)
        self.previous_btn.clicked.connect(lambda _checked=False: self.previousCameraRequested.emit())
        layout.addWidget(self.previous_btn)

        self.next_btn = QToolButton()
        self.next_btn.setIcon(get_icon("right.svg"))
        self.next_btn.setToolTip("Next camera")
        self.next_btn.setAutoRaise(True)
        self.next_btn.clicked.connect(lambda _checked=False: self.nextCameraRequested.emit())
        layout.addWidget(self.next_btn)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        self.size_down_btn = QToolButton()
        self.size_down_btn.setIcon(get_icon("remove.svg"))
        self.size_down_btn.setToolTip("Smaller cameras")
        self.size_down_btn.setAutoRaise(True)
        self.size_down_btn.clicked.connect(lambda _checked=False: self.decrease_canvas_size())
        layout.addWidget(self.size_down_btn)

        self.size_up_btn = QToolButton()
        self.size_up_btn.setIcon(get_icon("add.svg"))
        self.size_up_btn.setToolTip("Larger cameras")
        self.size_up_btn.setAutoRaise(True)
        self.size_up_btn.clicked.connect(lambda _checked=False: self.increase_canvas_size())
        layout.addWidget(self.size_up_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        self.count_down_btn = QToolButton()
        self.count_down_btn.setIcon(get_icon("remove.svg"))
        self.count_down_btn.setToolTip("Show fewer cameras")
        self.count_down_btn.setAutoRaise(True)
        self.count_down_btn.clicked.connect(lambda _checked=False: self.decrease_canvas_count())
        layout.addWidget(self.count_down_btn)

        self.count_up_btn = QToolButton()
        self.count_up_btn.setIcon(get_icon("add.svg"))
        self.count_up_btn.setToolTip("Show more cameras")
        self.count_up_btn.setAutoRaise(True)
        self.count_up_btn.clicked.connect(lambda _checked=False: self.increase_canvas_count())
        layout.addWidget(self.count_up_btn)

        layout.addStretch(1)

        # Multi-Camera Annotation
        self._multi_annotate_btn = QToolButton()
        self._multi_annotate_btn.setText("Multi-Annotate")
        self._multi_annotate_btn.setCheckable(True)
        self._multi_annotate_btn.setChecked(False)
        self._multi_annotate_btn.setToolTip("Multi-Camera Annotation")
        self._multi_annotate_btn.setAutoRaise(True)
        self._multi_annotate_btn.toggled.connect(self._on_multi_annotate_toggled)
        layout.addWidget(self._multi_annotate_btn)

        toolbar.addWidget(container)

        self._update_canvas_size_controls()
        self._update_canvas_count_controls()
        return toolbar

    def refresh_scaling(self):
        """Refresh toolbar sizing after a UI scale change."""
        if hasattr(self, 'toolbar'):
            self.toolbar.setIconSize(app_theme.scale_size(16))

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
                    
                    # NEW: Calculate relative 3D camera roll using the "Up" vector
                    if ref_cam is not None:
                        ctx_cam = self._mvat_manager.cameras.get(canvas.current_image_path)
                        if ctx_cam is not None:
                            
                            # 1. Reference camera's "Up" direction (-Y axis in image space)
                            up_ref_cam = np.array([0.0, -1.0, 0.0])
                            
                            # 2. Map to World space (R.T maps camera -> world)
                            up_world = ref_cam.R.T @ up_ref_cam
                            
                            # 3. Project World "Up" into the Context camera's frame
                            up_ctx_cam = ctx_cam.R @ up_world
                            
                            # 4. Find the angle of this vector in the Context image plane
                            alpha = np.degrees(np.arctan2(up_ctx_cam[1], up_ctx_cam[0]))
                            
                            # 5. We want this vector to point "Up" on the screen (-90 degrees).
                            # The required rotation is the difference between -90 and its current angle.
                            camera_roll = -90.0 - alpha
                            
                            total_angle = base_rotation + camera_roll
                    
                    canvas.set_zoom_level(absolute_zoom)
                    canvas._set_absolute_rotation(total_angle)

    def sync_to_targets(self, targets: dict, zoom_factor: float, reference_path: str = None, base_rotation: float = 0.0):
        if not self.target_lock_enabled or not self._mvat_manager:
            return

        ref_cam = self._mvat_manager.cameras.get(reference_path) if reference_path else None

        for i, (target_x, target_y) in targets.items():
            if i < len(self._canvas_pool):
                canvas = self._canvas_pool[i]
                if canvas.isVisible() and canvas.active_image:
                    
                    total_angle = base_rotation
                    
                    # NEW: Calculate relative 3D camera roll using the "Up" vector
                    if ref_cam is not None:
                        ctx_cam = self._mvat_manager.cameras.get(canvas.current_image_path)
                        if ctx_cam is not None:
                            
                            # 1. Reference camera's "Up" direction (-Y axis in image space)
                            up_ref_cam = np.array([0.0, -1.0, 0.0])
                            
                            # 2. Map to World space (R.T maps camera -> world)
                            up_world = ref_cam.R.T @ up_ref_cam
                            
                            # 3. Project World "Up" into the Context camera's frame
                            up_ctx_cam = ctx_cam.R @ up_world
                            
                            # 4. Find the angle of this vector in the Context image plane
                            alpha = np.degrees(np.arctan2(up_ctx_cam[1], up_ctx_cam[0]))
                            
                            # 5. We want this vector to point "Up" on the screen (-90 degrees).
                            # The required rotation is the difference between -90 and its current angle.
                            camera_roll = -90.0 - alpha
                            
                            total_angle = base_rotation + camera_roll

                    # Snap to target with the synchronized rotation
                    canvas.snap_to_target(target_x, target_y, zoom_factor, angle_degrees=total_angle)

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

    def set_annotation_manager(self, manager):
        self._annotation_manager = manager
        if manager is None:
            return
        manager.annotationAdded.connect(self._on_annotation_changed)
        manager.annotationRemoved.connect(self._on_annotation_changed)
        manager.annotationModified.connect(self._on_annotation_changed)
        manager.annotationLabelChanged.connect(lambda ann_id, _: self._on_annotation_changed(ann_id))
        manager.annotationsAdded.connect(self._on_annotations_changed)
        manager.annotationsRemoved.connect(self._on_annotations_changed)
        manager.selectionChanged.connect(self._on_selection_changed)

    def _on_annotation_changed(self, annotation_id: str):
        if not self._annotation_manager:
            return
        annotation = self._annotation_manager.annotations_dict.get(annotation_id)
        affected_path = annotation.image_path if annotation else None
        self._refresh_annotations_for_path(affected_path)

    def _on_annotations_changed(self, annotation_ids: list):
        if not self._annotation_manager:
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

    def _refresh_annotations_for_path(self, image_path: str = None):
        if not self._annotation_manager:
            return
        for canvas in self._visible_canvases:
            if canvas and canvas.active_image and canvas.current_image_path:
                if image_path is None or canvas.current_image_path == image_path:
                    annotations = self._annotation_manager.get_image_annotations(canvas.current_image_path)
                    canvas._render_annotations_readonly(annotations)
                    if self._raster_manager:
                        raster = self._raster_manager.get_raster(canvas.current_image_path)
                        if raster is not None and raster.mask_annotation is not None:
                            canvas.set_mask_overlay(raster.mask_annotation)

    def _on_selection_changed(self, selected_ids):
        selected_set = set(selected_ids) if selected_ids else set()
        for canvas in self._visible_canvases:
            if canvas and canvas.active_image:
                for item in canvas._readonly_annotation_items:
                    ann_id = getattr(item, '_source_annotation_id', None)
                    if ann_id:
                        canvas._highlight_readonly_annotation(ann_id, ann_id in selected_set)

    # ==================== Cursor Preview (Tool Propagation) ====================

    def update_cursor_previews(self, projections: dict, visible_paths: set, item_factory):
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

    # --- NEW METHODS ---
    def update_live_scratchpads(self, projections, size, shape, color):
        """Draws a live vector trail on all visible context cameras."""
        canvas_map = self._get_canvas_camera_map()

        for path, canvas in canvas_map.items():
            proj = projections.get(path)
            if not proj:
                continue

            u, v, is_valid = proj
            # Only draw if the 3D point is actually visible to this camera
            if is_valid:
                try:
                    canvas.add_to_scratchpad(u, v, size, shape, color)
                except Exception:
                    pass

    def clear_all_scratchpads(self):
        """Clears the fake vector trails across all cameras."""
        for canvas in self._canvas_pool:
            try:
                canvas.clear_scratchpad()
            except Exception:
                pass

    # ==================== Z-Channel Synchronization ====================

    def _apply_z_channel_state_to_canvas(self, canvas: BaseCanvas):
        """
        Apply the current z-channel state from AnnotationWindow to a specific canvas.
        Called after a canvas loads a new image to ensure z-channel visualization matches.
        """
        try:
            # Get the main annotation window to fetch current z-channel state
            annotation_window = getattr(self._mvat_manager.main_window, 'annotation_window', None) if self._mvat_manager else None
            if not annotation_window:
                return
            
            # Only apply if z-channel is displayed in the main window
            current_colormap = getattr(annotation_window.main_window, 'z_colormap_dropdown', None)
            if not current_colormap:
                return
                
            colormap_name = current_colormap.currentText()
            if colormap_name != "None":
                # Apply colormap
                canvas.update_z_colormap(colormap_name)
                
                # Apply opacity
                z_transparency = getattr(annotation_window.main_window, 'z_transparency_widget', None)
                if z_transparency:
                    opacity = z_transparency.value() / 255.0
                    canvas.set_z_opacity(opacity)
                
                # Apply dynamic scaling state
                z_dynamic = getattr(annotation_window.main_window, 'z_dynamic_scaling_checkbox', None)
                if z_dynamic:
                    is_dynamic = z_dynamic.isChecked()
                    canvas.toggle_dynamic_z_scaling(is_dynamic)
        except Exception:
            pass  # Silently ignore any errors applying z-channel state

    def sync_z_colormap_to_all_canvases(self, colormap_name: str):
        """Broadcast z-channel colormap changes to all visible canvases."""
        for canvas in self._canvas_pool:
            canvas.update_z_colormap(colormap_name)

    def sync_z_opacity_to_all_canvases(self, opacity: float):
        """Broadcast z-channel opacity changes to all visible canvases."""
        opacity = max(0.0, min(1.0, opacity))
        for canvas in self._canvas_pool:
            canvas.set_z_opacity(opacity)

    def sync_annotations_to_all_canvases(self):
        """Re-render readonly annotation overlays on all visible canvases (e.g., after transparency change)."""
        self._refresh_annotations_for_path(None)

    def sync_z_dynamic_scaling_to_all_canvases(self, enabled: bool):
        """Broadcast z-channel dynamic scaling toggle to all visible canvases."""
        for canvas in self._canvas_pool:
            canvas.toggle_dynamic_z_scaling(enabled)
