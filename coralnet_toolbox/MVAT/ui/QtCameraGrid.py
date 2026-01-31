"""
Camera Grid Widget for MVAT

A virtualized, scrollable grid of camera thumbnails with highlight/select distinction:
- Highlight (Cyan, 2px): Multiple selection via Ctrl+Click, Shift+Click, Ctrl+A
- Select (Lime Green, 4px): Single active camera via Double-Click
"""

import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QRect, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QSlider, 
    QLabel, QMenu, QAction, QSizePolicy, QFrame, QToolButton
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

HIGHLIGHT_COLOR = QColor(0, 255, 255)      # Cyan for multi-highlight
SELECT_COLOR = QColor(50, 205, 50)          # Lime Green for single select
MARKER_COLOR = QColor(255, 0, 255)          # Magenta for cross-camera marker
HIGHLIGHT_WIDTH = 2
SELECT_WIDTH = 4
MARKER_SIZE = 12                            # Diameter of marker circle
MARKER_LINE_WIDTH = 2                       # Line width for marker crosshairs
DEFAULT_THUMBNAIL_SIZE = 256
MIN_THUMBNAIL_SIZE = 256
MAX_THUMBNAIL_SIZE = 1024
GRID_SPACING = 5
BUFFER_ROWS = 1  # Extra rows to load above/below viewport


# ----------------------------------------------------------------------------------------------------------------------
# CameraDataItem
# ----------------------------------------------------------------------------------------------------------------------


class CameraDataItem:
    """
    Central data model for camera UI state.
    Wraps a Camera object with highlight/select state and thumbnail caching.
    """
    
    def __init__(self, camera):
        """
        Initialize a CameraDataItem.
        
        Args:
            camera: The Camera object from core.Camera
        """
        self.camera = camera
        self._is_highlighted = False
        self._is_selected = False
        self._thumbnail_pixmap = None
        self._thumbnail_size = None
        
    @property
    def image_path(self):
        """Get the image path from the underlying camera/raster."""
        return self.camera.image_path
    
    @property
    def label(self):
        """Get the display label from the camera."""
        return self.camera.label
    
    @property
    def position(self):
        """Get the 3D position of the camera."""
        return self.camera.position
    
    @property
    def raster(self):
        """Get the underlying Raster object."""
        return self.camera._raster
    
    @property
    def is_highlighted(self):
        """Check if this camera is highlighted (multi-select)."""
        return self._is_highlighted
    
    @property
    def is_selected(self):
        """Check if this camera is selected (single active)."""
        return self._is_selected
    
    def set_highlighted(self, highlighted):
        """Set the highlight state."""
        self._is_highlighted = highlighted
        
    def set_selected(self, selected):
        """Set the selection state."""
        self._is_selected = selected
        
    def get_thumbnail(self, longest_edge=256):
        """
        Get the thumbnail pixmap, using caching.
        
        Args:
            longest_edge (int): The longest edge size for the thumbnail
            
        Returns:
            QPixmap or None: The thumbnail pixmap
        """
        # Use cached thumbnail if same size
        if self._thumbnail_pixmap is not None and self._thumbnail_size == longest_edge:
            return self._thumbnail_pixmap
            
        # Get full QImage from raster and scale to requested size
        qimage = self.raster.get_qimage()
        if qimage and not qimage.isNull():
            # Scale the image to the requested thumbnail size
            scaled = qimage.scaled(
                longest_edge, longest_edge,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._thumbnail_pixmap = QPixmap.fromImage(scaled)
            self._thumbnail_size = longest_edge
            return self._thumbnail_pixmap
            
        return None
    
    def clear_thumbnail_cache(self):
        """Clear the cached thumbnail to free memory."""
        self._thumbnail_pixmap = None
        self._thumbnail_size = None
        
    def get_tooltip_text(self):
        """
        Generate rich tooltip text with camera metadata.
        
        Returns:
            str: HTML-formatted tooltip text
        """
        raster = self.raster
        camera = self.camera
        
        lines = [
            f"<b>{self.label}</b>",
            "<hr>",
            f"<b>Dimensions:</b> {raster.width} × {raster.height}",
        ]
        
        # Add intrinsics info if available
        if raster.intrinsics is not None:
            K = raster.intrinsics
            if K.shape[0] >= 3 and K.shape[1] >= 3:
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                lines.append(f"<b>Focal:</b> ({fx:.1f}, {fy:.1f})")
                lines.append(f"<b>Principal:</b> ({cx:.1f}, {cy:.1f})")
        
        # Add position info
        pos = camera.position
        lines.append(f"<b>Position:</b> ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        return "<br>".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# CameraImageWidget
# ----------------------------------------------------------------------------------------------------------------------


class CameraImageWidget(QWidget):
    """
    Widget displaying a camera thumbnail with highlight/select borders.
    Supports lazy loading and unloading for virtualization.
    """
    
    clicked = pyqtSignal(object, object)  # (widget, QMouseEvent)
    double_clicked = pyqtSignal(object)   # (widget)
    
    def __init__(self, data_item, widget_size=256, parent=None):
        """
        Initialize a CameraImageWidget.
        
        Args:
            data_item (CameraDataItem): The data item holding camera state
            widget_size (int): The display size for the widget
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.data_item = data_item
        self.widget_size = widget_size
        self.pixmap = None
        self.is_loaded = False
        
        # Calculate aspect ratio for proper sizing
        raster = data_item.raster
        self.aspect_ratio = raster.width / raster.height if raster.height > 0 else 1.0
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._update_size()
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Set tooltip
        self.setToolTip(data_item.get_tooltip_text())
        
        # Marker overlay for cross-camera position display
        self._marker_position = None  # (x, y) in image pixel coordinates or None
        self._marker_accurate = True  # True = solid marker, False = dashed marker
        
    def _update_size(self):
        """Update widget size based on aspect ratio and target size."""
        if self.aspect_ratio >= 1.0:
            # Landscape or square
            w = self.widget_size
            h = int(self.widget_size / self.aspect_ratio)
        else:
            # Portrait
            w = int(self.widget_size * self.aspect_ratio)
            h = self.widget_size
        self.setFixedSize(w, h)
        
    def update_size(self, new_size):
        """
        Update the widget size.
        
        Args:
            new_size (int): New target size for the longest edge
        """
        self.widget_size = new_size
        self._update_size()
        
        # Reload image at new size if currently loaded
        if self.is_loaded:
            self.load_image()
            
        self.update()
        
    def load_image(self):
        """Load the thumbnail image."""
        # Always reload to ensure we have the correct size
        # Request thumbnail at the exact widget size to avoid blurriness from scaling
        self.pixmap = self.data_item.get_thumbnail(longest_edge=self.widget_size)
        self.is_loaded = True
        self.update()
            
    def unload_image(self):
        """Unload the thumbnail to free memory."""
        self.pixmap = None
        self.is_loaded = False
        # Don't clear the data_item cache - it may be reused
    
    def set_marker_position(self, x: float, y: float, accurate: bool = True):
        """
        Set a marker position to display on the thumbnail.
        
        The marker indicates where a point from another camera view
        projects onto this camera's image.
        
        Args:
            x: X pixel coordinate in the original image space.
            y: Y pixel coordinate in the original image space.
            accurate: If True, draws solid marker (from depth data).
                     If False, draws dashed marker (estimated position).
        
        # TODO: Add visual indicator when marker position may be inaccurate 
        # (no depth/potential occlusion) - currently using solid vs dashed
        """
        self._marker_position = (x, y)
        self._marker_accurate = accurate
        self.update()  # Trigger repaint
        
    def clear_marker(self):
        """Clear any displayed marker."""
        if self._marker_position is not None:
            self._marker_position = None
            self.update()  # Trigger repaint
        
    def update_selection_visuals(self):
        """Trigger a repaint to update selection/highlight visuals."""
        self.update()
        
    def paintEvent(self, event):
        """Paint the widget with image and selection borders."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Draw background
        painter.fillRect(rect, QColor(40, 40, 40))
        
        # Draw the thumbnail if loaded
        if self.pixmap and not self.pixmap.isNull():
            # Draw pixmap at actual size (already loaded at correct size)
            # Center the image in the widget
            x = (rect.width() - self.pixmap.width()) // 2
            y = (rect.height() - self.pixmap.height()) // 2
            painter.drawPixmap(x, y, self.pixmap)
        else:
            # Draw placeholder text
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(rect, Qt.AlignCenter, "Loading...")
            
        # Draw selection border (Lime Green, 4px) - takes priority
        if self.data_item.is_selected:
            pen = QPen(SELECT_COLOR, SELECT_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            # Inset the rect by half the pen width to draw inside
            inset = SELECT_WIDTH // 2
            painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
            
        # Draw highlight border (Cyan, 2px)
        elif self.data_item.is_highlighted:
            pen = QPen(HIGHLIGHT_COLOR, HIGHLIGHT_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            inset = HIGHLIGHT_WIDTH // 2
            painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
            
        # Draw white border for non-selected/non-highlighted (1px)
        else:
            pen = QPen(QColor(200, 200, 200), 1)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        # Draw marker overlay if position is set
        if self._marker_position is not None:
            self._draw_marker(painter, rect)
            
    def _draw_marker(self, painter: QPainter, rect):
        """
        Draw the cross-camera position marker.
        
        Args:
            painter: QPainter instance.
            rect: Widget rect for coordinate transformation.
        """
        # Transform image coordinates to widget coordinates
        # Account for centering of pixmap in widget
        if self.pixmap and not self.pixmap.isNull():
            # Calculate offset for centered pixmap
            pixmap_x = (rect.width() - self.pixmap.width()) // 2
            pixmap_y = (rect.height() - self.pixmap.height()) // 2
            
            # Scale from original image coords to thumbnail coords
            raster = self.data_item.raster
            scale_x = self.pixmap.width() / raster.width
            scale_y = self.pixmap.height() / raster.height
            
            # Transform marker position
            marker_x = pixmap_x + int(self._marker_position[0] * scale_x)
            marker_y = pixmap_y + int(self._marker_position[1] * scale_y)
        else:
            # Fallback: just scale to widget size
            raster = self.data_item.raster
            marker_x = int(self._marker_position[0] * rect.width() / raster.width)
            marker_y = int(self._marker_position[1] * rect.height() / raster.height)
        
        # Check if marker is within widget bounds
        if not (0 <= marker_x < rect.width() and 0 <= marker_y < rect.height()):
            return  # Don't draw marker outside widget
        
        # Configure pen based on accuracy
        pen = QPen(MARKER_COLOR, MARKER_LINE_WIDTH)
        if self._marker_accurate:
            pen.setStyle(Qt.SolidLine)
        else:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # Draw circle
        half_size = MARKER_SIZE // 2
        painter.drawEllipse(marker_x - half_size, 
                            marker_y - half_size, 
                            MARKER_SIZE, MARKER_SIZE)
        
        # Draw crosshairs extending from circle
        crosshair_extend = half_size + 4  # Extend beyond circle
        # Horizontal line
        painter.drawLine(marker_x - crosshair_extend, marker_y,
                         marker_x - half_size - 1, marker_y)
        painter.drawLine(marker_x + half_size + 1, marker_y,
                         marker_x + crosshair_extend, marker_y)
        # Vertical line
        painter.drawLine(marker_x, marker_y - crosshair_extend,
                         marker_x, marker_y - half_size - 1)
        painter.drawLine(marker_x, marker_y + half_size + 1,
                         marker_x, marker_y + crosshair_extend)
            
    def mousePressEvent(self, event):
        """Handle mouse press for selection."""
        self.clicked.emit(self, event)
        
    def mouseDoubleClickEvent(self, event):
        """Handle double-click for single selection."""
        self.double_clicked.emit(self)


# ----------------------------------------------------------------------------------------------------------------------
# CameraGrid
# ----------------------------------------------------------------------------------------------------------------------


class CameraGrid(QWidget):
    """
    Scrollable grid of camera thumbnails with virtualization.
    
    Signals:
        camera_selected: Emitted when a camera is single-selected (double-click)
        cameras_highlighted: Emitted when highlight selection changes
        selection_changed: Emitted when any selection state changes
    """
    
    camera_selected = pyqtSignal(str)       # image_path of selected camera
    cameras_highlighted = pyqtSignal(list)  # list of image_paths
    selection_changed = pyqtSignal(list)    # list of all selected/highlighted paths
    
    def __init__(self, mvat_window=None, parent=None):
        """
        Initialize the CameraGrid.
        
        Args:
            mvat_window: Reference to the parent MVATWindow
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.mvat_window = mvat_window
        
        # Data management
        self.data_items = []                    # List of CameraDataItem
        self.widgets_by_path = {}               # image_path -> CameraImageWidget
        self.widget_positions = {}              # image_path -> QRect
        
        # Selection state
        self.selected_path = None               # Single selected camera path
        self.highlighted_paths = set()          # Set of highlighted camera paths
        self.last_clicked_index = -1            # For shift-click range selection
        
        # Display settings
        self.thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        
        # Debounce timer for layout updates
        self._layout_timer = QTimer()
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._do_recalculate_layout)
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI with toolbar and scroll area."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 5, 5, 5)
        toolbar.setSpacing(5)
        
        # Size slider
        size_label = QLabel("Size:")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(MIN_THUMBNAIL_SIZE, MAX_THUMBNAIL_SIZE)
        self.size_slider.setValue(self.thumbnail_size)
        self.size_slider.setFixedWidth(100)
        self.size_slider.setToolTip("Adjust thumbnail size")
        self.size_slider.valueChanged.connect(self._on_size_changed)
        
        # Pixel value label
        self.size_value_label = QLabel(f"{self.thumbnail_size}px")
        self.size_value_label.setMinimumWidth(50)
        
        toolbar.addWidget(size_label)
        toolbar.addWidget(self.size_slider)
        toolbar.addWidget(self.size_value_label)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        toolbar.addWidget(sep)
        
        # Selected camera label
        self.selected_label = QLabel("None selected")
        self.selected_label.setStyleSheet("color: #666;")
        toolbar.addWidget(self.selected_label)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        toolbar.addWidget(sep2)
        
        # Selection info label
        self.selection_label = QLabel("0 highlighted")
        self.selection_label.setStyleSheet("color: #666;")
        toolbar.addWidget(self.selection_label)
        
        toolbar.addStretch()
        
        # Clear selection button
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.setToolTip("Clear all selections (Escape)")
        self.clear_btn.clicked.connect(self.clear_all_selections)
        toolbar.addWidget(self.clear_btn)
        
        layout.addLayout(toolbar)
        
        # --- Scroll Area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: white;")
        self.scroll_area.setWidget(self.content_widget)
        
        layout.addWidget(self.scroll_area)
        
        # Set minimum width
        self.setMinimumWidth(MIN_THUMBNAIL_SIZE + 20)
        
    def set_cameras(self, cameras):
        """
        Set the cameras to display in the grid.
        
        Args:
            cameras (dict): Dictionary of image_path -> Camera objects
        """
        # Clear existing
        self._clear_widgets()
        self.data_items.clear()
        self.widgets_by_path.clear()
        self.widget_positions.clear()
        self.selected_path = None
        self.highlighted_paths.clear()
        self.last_clicked_index = -1
        
        # Create data items and widgets
        for path, camera in cameras.items():
            data_item = CameraDataItem(camera)
            self.data_items.append(data_item)
            
            widget = CameraImageWidget(data_item, self.thumbnail_size, self.content_widget)
            widget.clicked.connect(self._on_widget_clicked)
            widget.double_clicked.connect(self._on_widget_double_clicked)
            self.widgets_by_path[path] = widget
            
        # Calculate layout
        self.recalculate_layout()
        
        # Update selection label
        self._update_selection_label()
        
    def _clear_widgets(self):
        """Remove all widgets from the content widget."""
        for widget in self.widgets_by_path.values():
            widget.setParent(None)
            widget.deleteLater()
            
    def clear_cameras(self):
        """Clear all cameras from the grid."""
        self._clear_widgets()
        self.data_items.clear()
        self.widgets_by_path.clear()
        self.widget_positions.clear()
        self.selected_path = None
        self.highlighted_paths.clear()
        
    def recalculate_layout(self):
        """Schedule a layout recalculation (debounced)."""
        self._layout_timer.start(50)  # 50ms debounce
        
    def _do_recalculate_layout(self):
        """Actually recalculate widget positions."""
        if not self.data_items:
            return
            
        # Get available width
        available_width = self.scroll_area.viewport().width() - 10  # Padding
        
        # Calculate columns based on thumbnail size
        widget_width = self.thumbnail_size + GRID_SPACING
        num_columns = max(1, available_width // widget_width)
        
        # Position widgets in grid
        x, y = GRID_SPACING, GRID_SPACING
        row_height = 0
        col = 0
        
        for data_item in self.data_items:
            path = data_item.image_path
            widget = self.widgets_by_path.get(path)
            if not widget:
                continue
                
            # Update widget size
            widget.update_size(self.thumbnail_size)
            widget_size = widget.size()
            
            # Check if we need to wrap to next row
            if col >= num_columns:
                col = 0
                x = GRID_SPACING
                y += row_height + GRID_SPACING
                row_height = 0
                
            # Store position
            pos_rect = QRect(x, y, widget_size.width(), widget_size.height())
            self.widget_positions[path] = pos_rect
            
            # Track row height
            row_height = max(row_height, widget_size.height())
            
            # Move to next column
            x += widget_size.width() + GRID_SPACING
            col += 1
            
        # Set content widget size
        total_height = y + row_height + GRID_SPACING
        self.content_widget.setFixedHeight(total_height)
        
        # Update visible widgets
        self._update_visible_widgets()
        
    def _update_visible_widgets(self):
        """Show/hide widgets based on viewport visibility."""
        if not self.data_items:
            return
            
        # Get viewport rect
        viewport = self.scroll_area.viewport()
        scroll_y = self.scroll_area.verticalScrollBar().value()
        visible_rect = QRect(0, scroll_y, viewport.width(), viewport.height())
        
        # Add buffer for smooth scrolling
        buffer_height = self.thumbnail_size * BUFFER_ROWS
        visible_rect.adjust(0, -buffer_height, 0, buffer_height)
        
        for data_item in self.data_items:
            path = data_item.image_path
            widget = self.widgets_by_path.get(path)
            pos_rect = self.widget_positions.get(path)
            
            if not widget or not pos_rect:
                continue
                
            if pos_rect.intersects(visible_rect):
                # Widget is visible - show and position it
                widget.move(pos_rect.topLeft())
                widget.show()
                widget.load_image()
            else:
                # Widget is not visible - hide and unload
                widget.hide()
                widget.unload_image()
    
    def get_visible_widgets(self) -> dict:
        """
        Get dictionary of currently visible camera widgets.
        
        Returns widgets that are currently visible in the viewport,
        used for efficient marker updates.
        
        Returns:
            Dict mapping image_path to CameraImageWidget for visible widgets.
        """
        if not self.data_items:
            return {}
            
        # Get viewport rect
        viewport = self.scroll_area.viewport()
        scroll_y = self.scroll_area.verticalScrollBar().value()
        visible_rect = QRect(0, scroll_y, viewport.width(), viewport.height())
        
        # Add small buffer for edge cases
        buffer_height = self.thumbnail_size // 2
        visible_rect.adjust(0, -buffer_height, 0, buffer_height)
        
        visible_widgets = {}
        
        for data_item in self.data_items:
            path = data_item.image_path
            widget = self.widgets_by_path.get(path)
            pos_rect = self.widget_positions.get(path)
            
            if widget and pos_rect and pos_rect.intersects(visible_rect):
                visible_widgets[path] = widget
                
        return visible_widgets
                
    def _on_scroll(self, value):
        """Handle scroll events."""
        self._update_visible_widgets()
        
    def _on_size_changed(self, value):
        """Handle thumbnail size slider change."""
        self.thumbnail_size = value
        self.size_value_label.setText(f"{value}px")
        
        # Clear all thumbnail caches to force regeneration at new size
        for data_item in self.data_items:
            data_item.clear_thumbnail_cache()
        
        self.recalculate_layout()
        
    def _on_widget_clicked(self, widget, event):
        """Handle widget click for selection/highlighting."""
        path = widget.data_item.image_path
        modifiers = event.modifiers()
        
        # Find index of clicked item
        clicked_index = -1
        for i, item in enumerate(self.data_items):
            if item.image_path == path:
                clicked_index = i
                break
                
        if modifiers & Qt.ControlModifier:
            # Ctrl+Click: Toggle highlight
            self._toggle_highlight(path)
        elif modifiers & Qt.ShiftModifier and self.last_clicked_index >= 0:
            # Shift+Click: Range selection
            self._range_highlight(self.last_clicked_index, clicked_index)
        else:
            # Plain click: Clear others, highlight this one
            self._clear_highlights()
            self._add_highlight(path)
            
        self.last_clicked_index = clicked_index
        self._emit_highlight_changed()
        
    def _on_widget_double_clicked(self, widget):
        """Handle widget double-click for single selection."""
        path = widget.data_item.image_path
        
        # Clear previous selection
        if self.selected_path and self.selected_path in self.widgets_by_path:
            old_item = self.widgets_by_path[self.selected_path].data_item
            old_item.set_selected(False)
            self.widgets_by_path[self.selected_path].update_selection_visuals()
            
        # Set new selection
        self.selected_path = path
        widget.data_item.set_selected(True)
        widget.update_selection_visuals()
        
        # Emit signal for 3D view to match perspective
        self.camera_selected.emit(path)
        
        self._update_selection_label()
        
    def _toggle_highlight(self, path):
        """Toggle highlight state of a camera."""
        data_item = self.widgets_by_path[path].data_item
        
        if path in self.highlighted_paths:
            self.highlighted_paths.discard(path)
            data_item.set_highlighted(False)
        else:
            self.highlighted_paths.add(path)
            data_item.set_highlighted(True)
            
        self.widgets_by_path[path].update_selection_visuals()
        self._update_selection_label()
        
    def _add_highlight(self, path):
        """Add highlight to a camera."""
        if path not in self.highlighted_paths:
            self.highlighted_paths.add(path)
            data_item = self.widgets_by_path[path].data_item
            data_item.set_highlighted(True)
            self.widgets_by_path[path].update_selection_visuals()
            self._update_selection_label()
            
    def _remove_highlight(self, path):
        """Remove highlight from a camera."""
        if path in self.highlighted_paths:
            self.highlighted_paths.discard(path)
            data_item = self.widgets_by_path[path].data_item
            data_item.set_highlighted(False)
            self.widgets_by_path[path].update_selection_visuals()
            self._update_selection_label()
            
    def _clear_highlights(self):
        """Clear all highlights."""
        for path in list(self.highlighted_paths):
            if path in self.widgets_by_path:
                self.widgets_by_path[path].data_item.set_highlighted(False)
                self.widgets_by_path[path].update_selection_visuals()
        self.highlighted_paths.clear()
        self._update_selection_label()
        
    def _range_highlight(self, start_index, end_index):
        """Highlight a range of cameras."""
        if start_index > end_index:
            start_index, end_index = end_index, start_index
            
        # Clear existing highlights
        self._clear_highlights()
        
        # Highlight range
        for i in range(start_index, end_index + 1):
            if i < len(self.data_items):
                path = self.data_items[i].image_path
                self._add_highlight(path)
                
    def highlight_all(self):
        """Highlight all cameras (Ctrl+A)."""
        for data_item in self.data_items:
            path = data_item.image_path
            self.highlighted_paths.add(path)
            data_item.set_highlighted(True)
            if path in self.widgets_by_path:
                self.widgets_by_path[path].update_selection_visuals()
                
        self._update_selection_label()
        self._emit_highlight_changed()
        
    def clear_all_selections(self):
        """Clear all selections and highlights."""
        # Clear selection
        if self.selected_path and self.selected_path in self.widgets_by_path:
            self.widgets_by_path[self.selected_path].data_item.set_selected(False)
            self.widgets_by_path[self.selected_path].update_selection_visuals()
        self.selected_path = None
        
        # Clear highlights
        self._clear_highlights()
        
        self._emit_highlight_changed()
        
    def _emit_highlight_changed(self):
        """Emit signal when highlight selection changes."""
        self.cameras_highlighted.emit(list(self.highlighted_paths))
        
        # Also emit combined selection
        all_selected = list(self.highlighted_paths)
        if self.selected_path and self.selected_path not in all_selected:
            all_selected.append(self.selected_path)
        self.selection_changed.emit(all_selected)
        
    def _update_selection_label(self):
        """Update the selection count label."""
        count = len(self.highlighted_paths)
        self.selection_label.setText(f"{count} highlighted")
        
        # Update selected camera label
        if self.selected_path and self.selected_path in self.widgets_by_path:
            camera_label = self.widgets_by_path[self.selected_path].data_item.label
            self.selected_label.setText(f"Selected: {camera_label}")
        else:
            self.selected_label.setText("None selected")
        
    def render_selection_from_path(self, path):
        """
        Update the grid to show selection from external source (e.g., 3D picker).
        
        Args:
            path (str): The image path to select
        """
        if path not in self.widgets_by_path:
            return
            
        # Clear previous selection
        if self.selected_path and self.selected_path in self.widgets_by_path:
            old_item = self.widgets_by_path[self.selected_path].data_item
            old_item.set_selected(False)
            self.widgets_by_path[self.selected_path].update_selection_visuals()
            
        # Set new selection
        self.selected_path = path
        widget = self.widgets_by_path[path]
        widget.data_item.set_selected(True)
        widget.update_selection_visuals()
        
        # Scroll to make visible
        pos_rect = self.widget_positions.get(path)
        if pos_rect:
            self.scroll_area.ensureVisible(
                pos_rect.center().x(),
                pos_rect.center().y(),
                pos_rect.width(),
                pos_rect.height()
            )
            
        self._update_selection_label()
        
    def render_highlight_from_paths(self, paths):
        """
        Update the grid to show highlights from external source.
        
        Args:
            paths (list): List of image paths to highlight
        """
        # Clear existing highlights
        self._clear_highlights()
        
        # Add new highlights
        for path in paths:
            if path in self.widgets_by_path:
                self._add_highlight(path)
                
    def get_highlighted_cameras(self):
        """
        Get list of highlighted Camera objects.
        
        Returns:
            list: List of Camera objects that are highlighted
        """
        cameras = []
        for path in self.highlighted_paths:
            if path in self.widgets_by_path:
                cameras.append(self.widgets_by_path[path].data_item.camera)
        return cameras
    
    def get_selected_camera(self):
        """
        Get the selected Camera object.
        
        Returns:
            Camera or None: The selected camera, or None if none selected
        """
        if self.selected_path and self.selected_path in self.widgets_by_path:
            return self.widgets_by_path[self.selected_path].data_item.camera
        return None
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            # Ctrl+A: Select all
            self.highlight_all()
        elif event.key() == Qt.Key_Escape:
            # Escape: Clear selection
            self.clear_all_selections()
        else:
            super().keyPressEvent(event)
            
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.recalculate_layout()
        
    def contextMenuEvent(self, event):
        """Show context menu for highlighted cameras."""
        count = len(self.highlighted_paths)
        if count == 0:
            return
            
        menu = QMenu(self)
        
        # Header showing count
        header = QAction(f"Actions for {count} camera(s)", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()
        
        # Go to first highlighted image
        goto_action = QAction("Go to First Image", self)
        goto_action.triggered.connect(self._goto_first_highlighted)
        menu.addAction(goto_action)
        
        # Clear selection
        clear_action = QAction("Clear Selection", self)
        clear_action.triggered.connect(self.clear_all_selections)
        menu.addAction(clear_action)
        
        menu.exec_(event.globalPos())
        
    def _goto_first_highlighted(self):
        """Navigate to the first highlighted camera's image."""
        if self.highlighted_paths and self.mvat_window:
            first_path = next(iter(self.highlighted_paths))
            self.camera_selected.emit(first_path)