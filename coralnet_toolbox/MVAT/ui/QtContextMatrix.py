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
    QLabel, QToolBar, QPushButton, QToolButton, QSizePolicy, QFrame, QApplication
)

from coralnet_toolbox.QtBaseCanvas import BaseCanvas
from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_INVALID,
    SELECT_COLOR,
    HIGHLIGHT_COLOR,
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
        self._pending_sync = None

        # Annotation visualization state (Phase 6)
        self._annotation_manager = None

        # Selection tracking for Shift+Click range selection
        self._last_clicked_path: Optional[str] = None

        # Canvas pool
        self._canvas_pool: List[BaseCanvas] = []
        self._visible_canvases: List[List[Optional[BaseCanvas]]] = []

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
                        # Plain click: only update the 3D view (no selection/highlight changes)
                        # User must use Ctrl or Ctrl+Shift or the Clear button to modify selections
                        self.camera_highlighted_single.emit(path)

                    self._last_clicked_path = path

            # CRITICAL: pass through so drawing/panning tools still work
            BaseCanvas.mousePressEvent(canvas, event)
        return handler

    def _make_canvas_double_click_handler(self, canvas: BaseCanvas):
        """Double-click loads a context camera image without clearing selections.
        
        Emits both active_requested (to set the active camera) and contextImagePromoted
        (to load the image). Selections/highlights are preserved—users must use
        Ctrl+Click or the Clear button to modify them.
        """
        def handler(event):
            if event.button() == Qt.LeftButton:
                path = canvas.current_image_path
                if path:
                    # Set as active and load the image without clearing selections
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
        # Cap N to the number of available camera paths so we don't show blank canvases
        available = len(self._camera_paths)
        if available > 0:
            effective_target = min(self.target_camera_count, available)
        else:
            effective_target = self.target_camera_count

        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            rows = 1
            cols = 1
            N = effective_target
            if N != self._last_rebuilt_count:
                self._rebuild_layout(rows, cols, N)
            return

        aspect = self.width() / self.height()
        N = effective_target

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
        self._evaluate_auto_layout()

    # ==================== Data Feed ====================

    def set_raster_manager(self, raster_manager):
        self._raster_manager = raster_manager

    def set_camera_data(self, camera_objects: List, ordered_paths: List[str]):
        self._camera_paths = ordered_paths
        self._current_offset = 0
        self._evaluate_auto_layout()
        self._refresh_visible_canvases()

    def set_camera_order(self, ordered_paths: List[str], active_path: str = None):
        if active_path:
            self._camera_paths = [p for p in ordered_paths if p != active_path]
        else:
            self._camera_paths = list(ordered_paths)
        self._current_offset = 0
        self._evaluate_auto_layout()
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
                
                # Apply z-channel state from AnnotationWindow after loading image
                self._apply_z_channel_state_to_canvas(canvas)
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

        # Load Cameras button (moved to far left)
        self.load_btn = QToolButton()
        self.load_btn.setText("Load Cameras")
        self.load_btn.clicked.connect(self.loadCamerasRequested.emit)
        layout.addWidget(self.load_btn)

        # Separator after Load Cameras
        sep0 = QFrame()
        sep0.setFrameShape(QFrame.VLine)
        sep0.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep0)

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

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep3)

        # Rank indicator
        self._rank_label = QLabel("\u2014")
        self._rank_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._rank_label)

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

        # Select All button
        self.select_all_btn = QToolButton()
        self.select_all_btn.setText("Select All")
        self.select_all_btn.setToolTip("Highlight all cameras (even if not visible in grid)")
        self.select_all_btn.clicked.connect(self._on_select_all)
        layout.addWidget(self.select_all_btn)

        # Clear button
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.setToolTip("Clear all selections (Escape)")
        self.clear_btn.clicked.connect(self.clearSelectionsRequested.emit)
        layout.addWidget(self.clear_btn)

        toolbar.addWidget(container)
        return toolbar

    def update_stats(self, perspective_count: int):
        """Update the overall camera count labels."""
        if not hasattr(self, 'stats_label'):
            return
        self.stats_label.setText(f"Cameras: {perspective_count} perspective")

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

    def _on_select_all(self):
        """Highlight only cameras currently visible in the grid."""
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Collect only visible camera paths from the currently displayed grid
            visible_paths = []
            for row in self._visible_canvases:
                for canvas in row:
                    if canvas and canvas.current_image_path:
                        visible_paths.append(canvas.current_image_path)
            
            if visible_paths:
                self.selection_requested.emit(visible_paths)
        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()

    def _on_multi_annotate_toggled(self, checked: bool):
        self.multi_annotate_enabled = checked
        self.multiAnnotateToggled.emit(checked)

    # ==================== Selection Visuals ====================

    def sync_selection_borders(self, active_path: str, selected_paths: set):
        """Apply CSS borders to canvases based on active/highlighted status.
        
        Resets rotation for any canvas that is not both selected and synced.
        Only synced+selected canvases maintain their synchronized rotation.

        Args:
            active_path: Image path of the currently active (green-bordered) camera.
            selected_paths: Set of image paths that are highlighted (cyan-bordered).
        """
        # Convert QColor objects to hex for CSS
        active_color = SELECT_COLOR.name()
        highlight_color = HIGHLIGHT_COLOR.name()
        
        for canvas in self._canvas_pool:
            if not canvas.isVisible() or not canvas.current_image_path:
                continue
            path = canvas.current_image_path
            if path == active_path:
                canvas.setStyleSheet(f"border: 6px dashed {active_color};")   # Active camera (SELECT_COLOR), dashed
            elif path in selected_paths:
                canvas.setStyleSheet(f"border: 5px dashed {highlight_color};")   # Highlighted cameras (HIGHLIGHT_COLOR), dashed
            else:
                canvas.setStyleSheet("border: 3px solid #000000;")   # Default, solid black
            
            # Reset rotation unless the canvas is BOTH selected AND synced
            if not self.target_lock_enabled or path not in selected_paths:
                canvas.rotation_angle = 0.0
                canvas._set_absolute_rotation(0.0)

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
            canvas.update_static_marker(u, v, color=MARKER_COLOR_SELECTED)

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
