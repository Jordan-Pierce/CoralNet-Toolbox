"""
Context Matrix Widget for MVAT

Displays a configurable grid (1x1 to 3x3) of interactive BaseCanvas viewports for
viewing nearby cameras. Features include:
- Object pool of pre-created BaseCanvas instances
- Layout presets (1x1, 1x2, 2x1, 2x2, 1x3, 3x1)
- Auto-flow layout adaptation based on dock aspect ratio
- Image loading from RasterManager
- Double-click "Promote to Main" interaction
"""

import warnings
import numpy as np
from typing import List, Optional, Tuple, Dict

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QComboBox, 
    QLabel, QToolBar, QPushButton, QSizePolicy
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
    
    Signals:
        contextImagePromoted: Emitted when user double-clicks a context canvas
                             Carries the camera image path (str)
    
    Attributes:
        current_rows: Current grid rows
        current_cols: Current grid columns
        _canvas_pool: List of 9 pre-created BaseCanvas instances
        _visible_canvases: 2D grid of currently visible canvases
        _camera_paths: Ordered list of camera paths from MVATManager
        _current_offset: Offset into camera_paths for displaying cameras
    """
    
    # Signal: emit when user double-clicks a context canvas
    contextImagePromoted = pyqtSignal(str)  # camera_path
    rankIndicatorUpdated = pyqtSignal(int, int, int)  # start, end, total
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.current_rows = 1
        self.current_cols = 1
        self._camera_paths: List[str] = []
        self._current_offset = 0
        self._raster_manager = None
        self._loading_flag = False
        self._user_layout_locked = False
        self._auto_flow_enabled = True
        
        # Marker state for conveyor belt persistence (Phase 4)
        self._last_focal_point = None
        self._cameras_ref: Optional[Dict] = None
        
        # Layout presets: index -> (rows, cols)
        self._layout_map = {
            0: (1, 1),
            1: (1, 2),
            2: (2, 1),
            3: (2, 2),
            4: (1, 3),
            5: (3, 1),
        }
        
        # Canvas pool (pre-create max 9 canvases)
        self._canvas_pool: List[BaseCanvas] = []
        self._visible_canvases: List[List[Optional[BaseCanvas]]] = []
        
        # Timers
        self._resize_debounce_timer = QTimer()
        self._resize_debounce_timer.setSingleShot(True)
        self._resize_debounce_timer.timeout.connect(self._on_resize_debounce)
        
        # UI
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        self._grid_layout: Optional[QGridLayout] = None
        self._canvas_container: Optional[QWidget] = None
        
        # Create canvas pool
        self._create_canvas_pool()
        
        # Create initial layout
        self._rebuild_layout(1, 1)
        
    # ==================== Canvas Pool Management ====================
    
    def _create_canvas_pool(self):
        """Pre-create 9 BaseCanvas instances for reuse."""
        self._canvas_pool = []
        for i in range(9):
            canvas = BaseCanvas(parent=self)
            canvas.hide()
            # Connect double-click signal
            canvas.mouseDoubleClickEvent = self._make_canvas_double_click_handler(canvas)
            self._canvas_pool.append(canvas)
    
    def _make_canvas_double_click_handler(self, canvas: BaseCanvas):
        """Create a double-click handler for a specific canvas."""
        def handler(event):
            if event.button() == Qt.LeftButton:
                # Find which canvas this is and get its camera path
                for row_idx, row in enumerate(self._visible_canvases):
                    for col_idx, c in enumerate(row):
                        if c is canvas:
                            camera_idx = row_idx * self.current_cols + col_idx
                            if self._current_offset + camera_idx < len(self._camera_paths):
                                camera_path = self._camera_paths[self._current_offset + camera_idx]
                                self.contextImagePromoted.emit(camera_path)
                            return
            # Call parent event handler
            BaseCanvas.mouseDoubleClickEvent(canvas, event)
        return handler
    
    def _get_canvas(self, index: int) -> BaseCanvas:
        """Get a canvas from the pool by index."""
        if index < len(self._canvas_pool):
            return self._canvas_pool[index]
        return None
    
    def _release_canvas(self, canvas: BaseCanvas):
        """Return a canvas to pool state (clear and hide)."""
        if canvas:
            canvas.clear_scene()
            canvas.hide()
    
    # ==================== Layout Rebuilding ====================
    
    def _rebuild_layout(self, rows: int, cols: int):
        """Rebuild the grid layout with specified rows and columns."""
        self.current_rows = rows
        self.current_cols = cols
        
        # Calculate new capacity
        new_capacity = rows * cols
        
        # Clamp offset to valid range
        max_offset = max(0, len(self._camera_paths) - new_capacity)
        self._current_offset = min(self._current_offset, max_offset)
        
        # Remove old grid layout if exists
        if self._grid_layout is not None:
            while self._grid_layout.count():
                item = self._grid_layout.takeAt(0)
                if item.widget():
                    item.widget().hide()
        
        # Reparent all pool canvases back to self so they survive container deletion
        for canvas in self._canvas_pool:
            canvas.hide()
            canvas.setParent(self)
        
        # Create new canvas container
        if self._canvas_container:
            self._main_layout.removeWidget(self._canvas_container)
            self._canvas_container.deleteLater()
        
        self._canvas_container = QWidget()
        self._grid_layout = QGridLayout(self._canvas_container)
        self._grid_layout.setContentsMargins(2, 2, 2, 2)
        self._grid_layout.setSpacing(2)
        
        # Create visible canvases grid
        self._visible_canvases = [[None for _ in range(cols)] for _ in range(rows)]
        
        # Populate grid with canvases from pool
        canvas_index = 0
        for row in range(rows):
            for col in range(cols):
                if canvas_index < len(self._canvas_pool):
                    canvas = self._canvas_pool[canvas_index]
                    canvas.show()
                    self._grid_layout.addWidget(canvas, row, col)
                    self._visible_canvases[row][col] = canvas
                    canvas_index += 1
        
        # Set row/col stretch
        for row in range(rows):
            self._grid_layout.setRowStretch(row, 1)
        for col in range(cols):
            self._grid_layout.setColumnStretch(col, 1)
        
        # Add to main layout
        self._main_layout.addWidget(self._canvas_container, 1)
        
        # Re-clamp offset, refresh canvases, and update rank label
        self.shift_offset(0)
    
    # ==================== Data Feed ====================
    
    def set_raster_manager(self, raster_manager):
        """Wire the RasterManager for fetching image data."""
        self._raster_manager = raster_manager
    
    def set_camera_data(self, camera_objects: List, ordered_paths: List[str]):
        """Receive ordered camera list from MVATManager."""
        self._camera_paths = ordered_paths
        self._current_offset = 0
        self._refresh_visible_canvases()
    
    def set_camera_order(self, ordered_paths: List[str], active_path: str = None):
        """Set the ordered list of context cameras, excluding the active camera.
        
        Args:
            ordered_paths: Proximity-sorted camera paths from MVATManager.
            active_path: The currently active camera path to exclude.
        """
        if active_path:
            self._camera_paths = [p for p in ordered_paths if p != active_path]
        else:
            self._camera_paths = list(ordered_paths)
        self._current_offset = 0
        self._refresh_visible_canvases()
        self._update_rank_label()
    
    def update_context_cameras(self, camera_paths_slice: List[str]):
        """Load images from a slice of the ordered path list."""
        for i, canvas_row in enumerate(self._visible_canvases):
            for j, canvas in enumerate(canvas_row):
                if canvas:
                    canvas_index = i * self.current_cols + j
                    if canvas_index < len(camera_paths_slice):
                        path = camera_paths_slice[canvas_index]
                        self._load_canvas_image(canvas, path)
                    else:
                        canvas.clear_scene()
    
    def _refresh_visible_canvases(self):
        """Repaint visible canvases based on current offset."""
        if not self._raster_manager or not self._camera_paths:
            # Show placeholder
            for row in self._visible_canvases:
                for canvas in row:
                    if canvas:
                        canvas._show_placeholder("No cameras loaded")
            return
        
        visible_capacity = self._get_visible_capacity()
        
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
        
        # Re-apply static focal markers to newly loaded canvases
        if self._last_focal_point is not None and self._cameras_ref is not None:
            self.update_static_markers_from_3d(self._last_focal_point, self._cameras_ref)
    
    def _get_visible_capacity(self) -> int:
        """Return the number of currently visible canvas slots."""
        return self.current_rows * self.current_cols
    
    def _load_canvas_image(self, canvas: BaseCanvas, camera_path: str):
        """Load image into a canvas via raster_manager."""
        # Skip reload if already showing this image
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
            
            # Load thumbnail initially
            q_image = raster.get_qimage()
            if q_image:
                canvas.load_visuals(q_image, camera_path, raster)
                canvas.fit_to_image()
            else:
                canvas._show_placeholder("Failed to load image")
        except Exception as e:
            canvas._show_placeholder(f"Error: {str(e)[:20]}")
    
    def reset_offset(self):
        """Reset offset to 0 (e.g., on active camera change)."""
        self._current_offset = 0
        self._refresh_visible_canvases()
    
    def shift_offset(self, delta: int):
        """
        Shift offset by delta (for Phase 3 conveyor belt).
        Clamps to valid range.
        """
        visible_capacity = self._get_visible_capacity()
        max_offset = max(0, len(self._camera_paths) - visible_capacity)
        self._current_offset = max(0, min(self._current_offset + delta, max_offset))
        self._refresh_visible_canvases()
        self._update_rank_label()
    
    # ==================== Layout UI ====================
    
    def create_top_toolbar(self) -> QToolBar:
        """Create toolbar with layout options and controls."""
        toolbar = QToolBar("Context Matrix Tools")
        toolbar.setIconSize(QSize(16, 16))
        
        # Layout label
        layout_label = QLabel("Layout:")
        toolbar.addWidget(layout_label)
        
        # Layout dropdown
        self._layout_combo = QComboBox()
        self._layout_combo.addItems([
            "1×1",
            "1×2",
            "2×1",
            "2×2",
            "1×3",
            "3×1",
        ])
        self._layout_combo.setCurrentIndex(0)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_chosen)
        toolbar.addWidget(self._layout_combo)
        
        toolbar.addSeparator()
        
        # Auto-flow toggle
        self._auto_flow_btn = QPushButton("Auto-Flow")
        self._auto_flow_btn.setCheckable(True)
        self._auto_flow_btn.setChecked(True)
        self._auto_flow_btn.setMaximumWidth(80)
        self._auto_flow_btn.toggled.connect(self._on_auto_flow_toggled)
        toolbar.addWidget(self._auto_flow_btn)
        
        # Sync view toggle (skeleton for Phase 5)
        self._sync_btn = QPushButton("Sync View")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setChecked(False)
        self._sync_btn.setMaximumWidth(80)
        self._sync_btn.setEnabled(False)  # Disabled until Phase 5
        toolbar.addWidget(self._sync_btn)
        
        # Spacer to push rank label to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Rank indicator label (Phase 3)
        self._rank_label = QLabel("—")
        self._rank_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._rank_label)
        
        return toolbar
    
    def _on_layout_chosen(self, index: int):
        """Handle layout dropdown selection."""
        if index in self._layout_map:
            rows, cols = self._layout_map[index]
            self._user_layout_locked = True
            self._rebuild_layout(rows, cols)
    
    def _on_auto_flow_toggled(self, checked: bool):
        """Handle auto-flow toggle."""
        self._auto_flow_enabled = checked
        if checked:
            self._user_layout_locked = False
            # Re-evaluate layout based on current aspect ratio
            self._evaluate_auto_layout()
    
    # ==================== Auto-Flow Layout Adaptation ====================
    
    def resizeEvent(self, event):
        """Auto-adjust layout based on aspect ratio when auto-flow is enabled."""
        super().resizeEvent(event)
        
        # Debounce resize events
        self._resize_debounce_timer.stop()
        self._resize_debounce_timer.start(200)
    
    def _on_resize_debounce(self):
        """Process resize after debounce timeout."""
        if self._auto_flow_enabled and not self._user_layout_locked:
            self._evaluate_auto_layout()
    
    def _evaluate_auto_layout(self):
        """Automatically choose layout based on aspect ratio."""
        if not self.isVisible():
            return
        
        width = self.width()
        height = self.height()
        
        if width <= 0 or height <= 0:
            return
        
        aspect = width / height
        
        # Choose layout based on aspect ratio
        if aspect > 1.5:
            # Very wide: 1x3 or 1x2
            new_layout = 4 if width > 800 else 1  # 1x3 or 1x2
        elif aspect > 1.2:
            # Landscape: 2x2 or 1x2
            new_layout = 3 if width > 600 else 1  # 2x2 or 1x2
        elif aspect < 0.67:
            # Very tall: 3x1 or 2x1
            new_layout = 5 if height > 800 else 2  # 3x1 or 2x1
        elif aspect < 0.83:
            # Portrait: 2x1
            new_layout = 2
        else:
            # Square: 2x2
            new_layout = 3
        
        if new_layout != self._layout_combo.currentIndex():
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(new_layout)
            self._layout_combo.blockSignals(False)
            
            rows, cols = self._layout_map[new_layout]
            self._rebuild_layout(rows, cols)
    
    # ==================== Marker Routing (Phase 4) ====================
    
    def _get_canvas_camera_map(self) -> Dict[str, 'BaseCanvas']:
        """Return a mapping of camera_path -> BaseCanvas for visible, loaded canvases."""
        result = {}
        for i, canvas_row in enumerate(self._visible_canvases):
            for j, canvas in enumerate(canvas_row):
                if canvas and canvas.current_image_path and canvas.active_image:
                    result[canvas.current_image_path] = canvas
        return result
    
    def update_dynamic_markers(self, projections: dict, accuracies: dict,
                                visibility_status: dict):
        """Update dynamic hover markers on all visible canvases.
        
        Args:
            projections: {image_path: (u, v, is_valid)} from ray.project_to_cameras()
            accuracies: {image_path: has_accurate_depth}
            visibility_status: {image_path: is_occluded}
        """
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
        """Clear dynamic markers from all canvases in the pool."""
        for canvas in self._canvas_pool:
            canvas.clear_dynamic_marker()
    
    def update_static_markers_from_3d(self, point_3d, cameras: dict):
        """Project a 3D focal point into all visible canvases as static markers.
        
        Args:
            point_3d: numpy array [x, y, z] in world coordinates.
            cameras: dict of {image_path: Camera} for projection.
        """
        # Store for conveyor belt persistence
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
            
            # Check occlusion for color
            try:
                is_occluded = camera.is_point_occluded_depth_based(point_3d, depth_threshold=0.15)
            except Exception:
                is_occluded = False
            
            color = MARKER_COLOR_INVALID if is_occluded else MARKER_COLOR_SELECTED
            canvas.update_static_marker(u, v, color=color)
    
    def clear_all_static_markers(self):
        """Clear static markers from all canvases in the pool."""
        self._last_focal_point = None
        for canvas in self._canvas_pool:
            canvas.clear_static_marker()
    
    # ==================== Rank Indicator ====================
    
    def _update_rank_label(self):
        """Update the rank indicator label with current position info."""
        if not hasattr(self, '_rank_label'):
            return
        
        total = len(self._camera_paths)
        if total == 0:
            self._rank_label.setText("—")
            return
        
        capacity = self._get_visible_capacity()
        start = self._current_offset + 1
        end = min(self._current_offset + capacity, total)
        self._rank_label.setText(f"Neighbors {start}–{end} of {total}")
        self.rankIndicatorUpdated.emit(start, end, total)
