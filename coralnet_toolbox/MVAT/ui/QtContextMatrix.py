"""
Context Matrix Widget for MVAT (Unified Grid)

Displays a dynamic, auto-flowing grid of interactive BaseCanvas viewports.
Replaces the legacy CameraGrid.

Features include:
- Dynamic object pool of BaseCanvas instances.
- Auto-flow layout adaptation based on aspect ratio and target count.
- Ctrl + Shift + Wheel to dynamically increase/decrease the number of visible cameras.
- Image loading from RasterManager.
- Double-click "Promote to Main" interaction.
"""

import warnings
import math
import numpy as np
from typing import List, Optional, Dict

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QEvent
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QToolBar, QPushButton, QToolButton, QSizePolicy, QFrame
)

from coralnet_toolbox.QtBaseCanvas import BaseCanvas
from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_INVALID,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ContextMatrixWidget(QWidget):
    """
    Interactive grid of BaseCanvas viewports for multi-camera context viewing.
    Unified replacement for the legacy CameraGrid widget.

    Signals:
        contextImagePromoted: Emitted when user double-clicks a context canvas (camera_path)
        rankIndicatorUpdated: Emitted when rank indicator changes (start, end, total)
        multiAnnotateToggled: Emitted when multi-annotate mode is toggled (bool)
        loadCamerasRequested: Emitted when the Load Cameras button is clicked
        clearSelectionsRequested: Emitted when the Clear button is clicked
    """

    contextImagePromoted = pyqtSignal(str)            # camera_path
    rankIndicatorUpdated = pyqtSignal(int, int, int)  # start, end, total
    multiAnnotateToggled = pyqtSignal(bool)           # enabled state

    # Migrated from legacy CameraGrid
    loadCamerasRequested = pyqtSignal()
    clearSelectionsRequested = pyqtSignal()

    # Selection intent signals (mirror CameraGrid's paradigm)
    selection_requested = pyqtSignal(list)   # request to set highlight selection (list of paths)
    toggle_requested = pyqtSignal(str)       # request to toggle a single path (Ctrl+Click)
    active_requested = pyqtSignal(str)       # request to set active camera (Double-Click)
    camera_highlighted_single = pyqtSignal(str)  # single plain-click -> jump 3D view

    def __init__(self, parent=None):
        super().__init__(parent)

        # Grid State
        self.target_camera_count = 1
        self.max_cameras = 36  # Upper limit to prevent memory blowout
        self.current_rows = 1
        self.current_cols = 1
        self._last_rebuilt_count = 0

        # Camera Data State
        self._camera_paths: List[str] = []
        self._current_offset = 0
        self._raster_manager = None
        self._loading_flag = False

        # Marker state for conveyor belt persistence (Phase 4)
        self._last_focal_point = None
        self._cameras_ref: Optional[Dict] = None

        # Target-lock sync state (Phase 5)
        self.target_lock_enabled = False
        self._mvat_manager = None

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._sync_timer = QTimer()
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._process_pending_sync)
        self._pending_sync = None
        self._sync_throttle_ms = 30  # ~33 fps

        # Annotation visualization state (Phase 6)
        self._annotation_manager = None

        # Selection tracking for Shift+Click range selection
        self._last_clicked_path: Optional[str] = None

        # Canvas pool
        self._canvas_pool: List[BaseCanvas] = []
        self._visible_canvases: List[List[Optional[BaseCanvas]]] = []

        # Timers
        self._resize_debounce_timer = QTimer()
        self._resize_debounce_timer.setSingleShot(True)
        self._resize_debounce_timer.timeout.connect(self._evaluate_auto_layout)

        # UI Setup
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._grid_layout: Optional[QGridLayout] = None
        self._canvas_container: Optional[QWidget] = None

        # Catch Ctrl+Shift+Wheel on the widget itself
        self.installEventFilter(self)

        # Create initial layout
        self.set_target_camera_count(1)

    # ==================== Canvas Pool Management ====================

    def _ensure_canvas_pool_size(self, size: int):
        """Dynamically expand the BaseCanvas pool if needed."""
        while len(self._canvas_pool) < size:
            canvas = BaseCanvas(parent=self)
            canvas.hide()
            canvas.setStyleSheet("border: 1px solid #444444;")
            canvas.mouseDoubleClickEvent = self._make_canvas_double_click_handler(canvas)
            canvas.mousePressEvent = self._make_canvas_mouse_press_handler(canvas)
            canvas.installEventFilter(self)  # Catch scroll events hovering over the canvas
            self._canvas_pool.append(canvas)

    def _make_canvas_mouse_press_handler(self, canvas: BaseCanvas):
        """Intercept left clicks for selection while preserving native canvas interactions."""
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
                        # Plain click: exclusive highlight + 3D view jump
                        self.selection_requested.emit([path])
                        self.camera_highlighted_single.emit(path)

                    self._last_clicked_path = path

            # CRITICAL: pass through so drawing/panning tools still work
            BaseCanvas.mousePressEvent(canvas, event)
        return handler

    def _make_canvas_double_click_handler(self, canvas: BaseCanvas):
        """Double-click promotes a context camera to the main active camera."""
        def handler(event):
            if event.button() == Qt.LeftButton:
                path = canvas.current_image_path
                if path:
                    self.active_requested.emit(path)
                    self.contextImagePromoted.emit(path)
            BaseCanvas.mouseDoubleClickEvent(canvas, event)
        return handler

    # ==================== Input / Scroll Events ====================

    def eventFilter(self, source, event):
        """Capture Ctrl+Shift+Wheel to dynamically resize the camera grid."""
        if event.type() == QEvent.Wheel:
            modifiers = event.modifiers()
            if modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
                delta = event.angleDelta().y()
                if delta > 0:
                    # Scroll Up -> Fewer cameras (Zoom in)
                    self.set_target_camera_count(self.target_camera_count - 1)
                elif delta < 0:
                    # Scroll Down -> More cameras (Zoom out)
                    self.set_target_camera_count(self.target_camera_count + 1)
                return True
        return super().eventFilter(source, event)

    def set_target_camera_count(self, count: int):
        """Update the target number of visible cameras and re-evaluate layout."""
        self.target_camera_count = max(1, min(count, self.max_cameras))
        self._evaluate_auto_layout()

    # ==================== Layout Rebuilding ====================

    def _evaluate_auto_layout(self):
        """Automatically choose grid dimensions based on aspect ratio and target count."""
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            rows = 1
            cols = 1
            N = self.target_camera_count
            if N != self._last_rebuilt_count:
                self._rebuild_layout(rows, cols, N)
            return

        aspect = self.width() / self.height()
        N = self.target_camera_count

        if N == 1:
            rows, cols = 1, 1
        else:
            # Optimal grid calculation for aspect ratio
            cols = max(1, int(round(math.sqrt(N * aspect))))
            rows = math.ceil(N / cols)

            # Squeeze empty rows/cols to minimal bounding grid
            while (rows - 1) * cols >= N:
                rows -= 1
            while rows * (cols - 1) >= N:
                cols -= 1

        if rows != self.current_rows or cols != self.current_cols or N != self._last_rebuilt_count:
            self._rebuild_layout(rows, cols, N)

    def _rebuild_layout(self, rows: int, cols: int, count: int):
        """Rebuild the grid layout with specified rows, cols, and active cells."""
        self.current_rows = rows
        self.current_cols = cols
        self._last_rebuilt_count = count

        self._ensure_canvas_pool_size(count)

        # Clamp offset
        max_offset = max(0, len(self._camera_paths) - count)
        self._current_offset = min(self._current_offset, max_offset)

        if self._grid_layout is not None:
            while self._grid_layout.count():
                item = self._grid_layout.takeAt(0)
                if item.widget():
                    item.widget().hide()

        for canvas in self._canvas_pool:
            canvas.hide()
            canvas.setParent(self)

        if self._canvas_container:
            self._main_layout.removeWidget(self._canvas_container)
            self._canvas_container.deleteLater()

        self._canvas_container = QWidget()
        self._grid_layout = QGridLayout(self._canvas_container)
        self._grid_layout.setContentsMargins(2, 2, 2, 2)
        self._grid_layout.setSpacing(2)

        self._visible_canvases = [[None for _ in range(cols)] for _ in range(rows)]

        canvas_index = 0
        for row in range(rows):
            for col in range(cols):
                if canvas_index < count:
                    canvas = self._canvas_pool[canvas_index]
                    canvas.show()
                    self._grid_layout.addWidget(canvas, row, col)
                    self._visible_canvases[row][col] = canvas
                    canvas_index += 1
                else:
                    self._visible_canvases[row][col] = None

        for row in range(rows):
            self._grid_layout.setRowStretch(row, 1)
        for col in range(cols):
            self._grid_layout.setColumnStretch(col, 1)

        self._main_layout.addWidget(self._canvas_container, 1)
        self.shift_offset(0)

    def resizeEvent(self, event):
        """Auto-adjust layout on resize without changing camera count."""
        super().resizeEvent(event)
        self._resize_debounce_timer.stop()
        self._resize_debounce_timer.start(200)

    # ==================== Data Feed ====================

    def set_raster_manager(self, raster_manager):
        self._raster_manager = raster_manager

    def set_camera_data(self, camera_objects: List, ordered_paths: List[str]):
        self._camera_paths = ordered_paths
        self._current_offset = 0
        self._refresh_visible_canvases()

    def set_camera_order(self, ordered_paths: List[str], active_path: str = None):
        if active_path:
            self._camera_paths = [p for p in ordered_paths if p != active_path]
        else:
            self._camera_paths = list(ordered_paths)
        self._current_offset = 0
        self._refresh_visible_canvases()
        self._update_rank_label()

    def _refresh_visible_canvases(self):
        if not self._raster_manager or not self._camera_paths:
            for row in self._visible_canvases:
                for canvas in row:
                    if canvas:
                        canvas._show_placeholder("No cameras loaded")
            return

        for i, canvas_row in enumerate(self._visible_canvases):
            for j, canvas in enumerate(canvas_row):
                if canvas:
                    canvas_index = i * self.current_cols + j
                    offset_index = self._current_offset + canvas_index

                    if offset_index < len(self._camera_paths):
                        camera_path = self._camera_paths[offset_index]
                        self._load_canvas_image(canvas, camera_path)
                    else:
                        canvas.clear_scene()

        if self._last_focal_point is not None and self._cameras_ref is not None:
            self.update_static_markers_from_3d(self._last_focal_point, self._cameras_ref)

        if self.target_lock_enabled and self._mvat_manager:
            self._request_sync_from_main_view()

    def _get_visible_capacity(self) -> int:
        return self._last_rebuilt_count

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
            else:
                canvas._show_placeholder("Failed to load image")
        except Exception as e:
            canvas._show_placeholder(f"Error: {str(e)[:20]}")

    def reset_offset(self):
        self._current_offset = 0
        self._refresh_visible_canvases()

    def shift_offset(self, delta: int):
        visible_capacity = self._get_visible_capacity()
        max_offset = max(0, len(self._camera_paths) - visible_capacity)
        self._current_offset = max(0, min(self._current_offset + delta, max_offset))
        self._refresh_visible_canvases()
        self._update_rank_label()

    # ==================== Toolbar (Unified Grid + Matrix) ====================

    def create_top_toolbar(self) -> QToolBar:
        """Create toolbar unifying legacy grid stats and matrix options."""
        toolbar = QToolBar("Grid Context Tools")
        toolbar.setIconSize(QSize(16, 16))

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Camera count stats (migrated from CameraGrid)
        self.stats_label = QLabel("Cameras: 0")
        self.stats_label.setStyleSheet("color: #333;")
        layout.addWidget(self.stats_label)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        self.selected_label = QLabel("None selected")
        self.selected_label.setStyleSheet("color: #666;")
        layout.addWidget(self.selected_label)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        self.selection_label = QLabel("0 highlighted")
        self.selection_label.setStyleSheet("color: #666;")
        layout.addWidget(self.selection_label)

        # Rank indicator
        self._rank_label = QLabel("\u2014")
        self._rank_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._rank_label)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep3)

        layout.addStretch(1)

        # Target-Lock Sync
        self._sync_btn = QPushButton("Sync")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setChecked(False)
        self._sync_btn.setToolTip("Target-Lock Sync (disabled)")
        self._sync_btn.toggled.connect(self._on_sync_toggled)
        layout.addWidget(self._sync_btn)

        # Multi-Camera Annotation
        self._multi_annotate_btn = QPushButton("Multi-Annotate")
        self._multi_annotate_btn.setCheckable(True)
        self._multi_annotate_btn.setChecked(False)
        self._multi_annotate_btn.setToolTip("Multi-Camera Annotation")
        self._multi_annotate_btn.toggled.connect(self._on_multi_annotate_toggled)
        layout.addWidget(self._multi_annotate_btn)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.VLine)
        sep4.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep4)

        # Load Cameras button
        self.load_btn = QToolButton()
        self.load_btn.setText("Load Cameras")
        self.load_btn.clicked.connect(self.loadCamerasRequested.emit)
        layout.addWidget(self.load_btn)

        # Clear button
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.setToolTip("Clear all selections (Escape)")
        self.clear_btn.clicked.connect(self.clearSelectionsRequested.emit)
        layout.addWidget(self.clear_btn)

        toolbar.addWidget(container)
        return toolbar

    def update_stats(self, perspective_count: int, ortho_count: int = 0):
        """Update the overall camera count labels."""
        if not hasattr(self, 'stats_label'):
            return
        ortho_str = f", {ortho_count} ortho" if ortho_count > 0 else ""
        self.stats_label.setText(f"Cameras: {perspective_count} perspective{ortho_str}")

    def update_selection_labels(self, active_label: str, highlighted_count: int):
        """Update the labels indicating selected/highlighted cameras."""
        if not hasattr(self, 'selection_label'):
            return
        self.selection_label.setText(f"{highlighted_count} highlighted")
        if active_label:
            self.selected_label.setText(f"Selected: {active_label}")
        else:
            self.selected_label.setText("None selected")

    def _on_sync_toggled(self, checked):
        self.target_lock_enabled = checked
        self._sync_btn.setToolTip("Target-Lock Sync (enabled)" if checked else "Target-Lock Sync (disabled)")
        if checked:
            self._request_sync_from_main_view()

    def _on_multi_annotate_toggled(self, checked: bool):
        self.multi_annotate_enabled = checked
        self.multiAnnotateToggled.emit(checked)

    # ==================== Selection Visuals ====================

    def sync_selection_borders(self, active_path: str, selected_paths: set):
        """Apply CSS borders to canvases based on active/highlighted status.

        Args:
            active_path: Image path of the currently active (green-bordered) camera.
            selected_paths: Set of image paths that are highlighted (cyan-bordered).
        """
        for canvas in self._canvas_pool:
            if not canvas.isVisible() or not canvas.current_image_path:
                continue
            path = canvas.current_image_path
            if path == active_path:
                canvas.setStyleSheet("border: 4px solid #32CD32;")   # Lime Green
            elif path in selected_paths:
                canvas.setStyleSheet("border: 2px solid #00FFFF;")   # Cyan
            else:
                canvas.setStyleSheet("border: 1px solid #444444;")   # Default

    # ==================== Marker Routing (Phase 4) ====================

    def _get_canvas_camera_map(self) -> Dict[str, 'BaseCanvas']:
        result = {}
        for row in self._visible_canvases:
            for canvas in row:
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
            try:
                is_occluded = camera.is_point_occluded_depth_based(point_3d, depth_threshold=0.15)
            except Exception:
                is_occluded = False

            color = MARKER_COLOR_INVALID if is_occluded else MARKER_COLOR_SELECTED
            canvas.update_static_marker(u, v, color=color)

    def clear_all_static_markers(self):
        self._last_focal_point = None
        for canvas in self._canvas_pool:
            canvas.clear_static_marker()

    # ==================== Target-Lock Sync (Phase 5) ====================

    def set_mvat_manager(self, manager):
        self._mvat_manager = manager

    def request_sync(self, targets: dict, zoom_factor: float):
        self._pending_sync = (targets, zoom_factor)
        if not self._sync_timer.isActive():
            self._sync_timer.start(self._sync_throttle_ms)

    def _process_pending_sync(self):
        if self._pending_sync:
            targets, zoom_factor = self._pending_sync
            self._pending_sync = None
            self.sync_to_targets(targets, zoom_factor)

    def sync_to_targets(self, targets: dict, zoom_factor: float):
        if not self.target_lock_enabled:
            return

        for i, (target_x, target_y) in targets.items():
            if i < len(self._canvas_pool):
                canvas = self._canvas_pool[i]
                if canvas.isVisible() and canvas.active_image:
                    canvas.snap_to_target(target_x, target_y, zoom_factor)

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
        if not hasattr(self, '_rank_label'):
            return
        total = len(self._camera_paths)
        if total == 0:
            self._rank_label.setText("\u2014")
            return
        capacity = self._get_visible_capacity()
        start = self._current_offset + 1
        end = min(self._current_offset + capacity, total)
        self._rank_label.setText(f"Neighbors {start}\u2013{end} of {total}")
        self.rankIndicatorUpdated.emit(start, end, total)

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
        for row in self._visible_canvases:
            for canvas in row:
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
        for row in self._visible_canvases:
            for canvas in row:
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
