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
    QLabel, QMenu, QAction, QSizePolicy, QFrame, QToolButton,
    QApplication, QToolBar, QMessageBox
)

from coralnet_toolbox.MVAT.core.constants import (
    HIGHLIGHT_COLOR,
    SELECT_COLOR,
    HOVER_COLOR,
    MARKER_COLOR_INVALID,
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_DEFAULT,
    HIGHLIGHT_WIDTH,
    SELECT_WIDTH,
    MARKER_SIZE,
    MARKER_LINE_WIDTH,
    DEFAULT_THUMBNAIL_SIZE,
    MIN_THUMBNAIL_SIZE,
    MAX_THUMBNAIL_SIZE,
    GRID_SPACING,
    BUFFER_ROWS,
)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self._is_hovered = False
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
        # Highlight state is now managed by SelectionManager (external)
        raise AttributeError("is_highlighted is managed by SelectionManager")
    
    @property
    def is_selected(self):
        """Check if this camera is selected (single active)."""
        # Selected/active state is now managed by SelectionManager (external)
        raise AttributeError("is_selected is managed by SelectionManager")
    
    def set_highlighted(self, highlighted):
        """Set the highlight state."""
        # Deprecated: state moved to SelectionManager
        return
        
    def set_selected(self, selected):
        """Set the selection state."""
        # Deprecated: state moved to SelectionManager
        return
        
    @property
    def is_hovered(self):
        """Check if this camera is hovered (with Ctrl)."""
        return self._is_hovered
    
    def set_hovered(self, hovered):
        """Set the hover state."""
        self._is_hovered = hovered
        
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
    hovered = pyqtSignal(str)             # (image_path)
    unhovered = pyqtSignal(str)           # (image_path)
    select_image_requested = pyqtSignal(str)  # (image_path) - emitted when "Select Image" chosen from context menu
    
    def __init__(self, data_item, model, widget_size=256, parent=None):
        """
        Initialize a CameraImageWidget.
        
        Args:
            data_item (CameraDataItem): The data item holding camera state
            widget_size (int): The display size for the widget
            parent: Parent widget
        """
        super().__init__(parent)
        self.data_item = data_item
        self.model = model
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
        self._marker_color = MARKER_COLOR_DEFAULT  # Color for the marker
        self._is_occluded = False  # True if the point is blocked by geometry
        
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
    
    def set_marker_position(self, x: float, y: float, accurate: bool = True, 
                            color: QColor = None, is_occluded: bool = False):
        """
        Set a marker position to display on the thumbnail.
        
        The marker indicates where a point from another camera view
        projects onto this camera's image.
        
        Args:
            x: X pixel coordinate in the original image space.
            y: Y pixel coordinate in the original image space.
            accurate: If True, draws solid marker (from depth data).
                     If False, draws dashed marker (estimated position).
            color: QColor for the marker. If None, uses MARKER_COLOR_DEFAULT.
                   Use MARKER_COLOR_SELECTED (lime) for selected camera,
                   MARKER_COLOR_HIGHLIGHTED (cyan) for highlighted cameras.
            is_occluded (bool): If True, the point is blocked by geometry (draw differently).
        
        # TODO: When depth is fully incorporated, re-evaluate solid vs dashed
        # styling based on depth accuracy at the projected point.
        """
        self._marker_position = (x, y)
        self._marker_accurate = accurate
        self._marker_color = color if color is not None else MARKER_COLOR_DEFAULT
        self._is_occluded = is_occluded  # Store the new state
        self.update()  # Trigger repaint
        
    def clear_marker(self):
        """Clear any displayed marker."""
        if self._marker_position is not None:
            self._marker_position = None
            self._marker_color = MARKER_COLOR_DEFAULT
            self._is_occluded = False
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
            
        # Draw borders with priority: hover > active (model.active_path) > selected (model.selected_paths)
        path = self.data_item.image_path
        is_active = self.model.is_active(path) if self.model else False
        is_selected = self.model.is_selected(path) if self.model else False

        if self.data_item.is_hovered:
            pen = QPen(HOVER_COLOR, SELECT_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            inset = SELECT_WIDTH // 2
            painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
        elif is_active:
            pen = QPen(SELECT_COLOR, SELECT_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            inset = SELECT_WIDTH // 2
            painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
        elif is_selected:
            pen = QPen(HIGHLIGHT_COLOR, HIGHLIGHT_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            inset = HIGHLIGHT_WIDTH // 2
            painter.drawRect(rect.adjusted(inset, inset, -inset, -inset))
        else:
            # Draw white border for non-selected/non-highlighted (1px)
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
        
        # Configure pen based on occlusion state
        pen = QPen(self._marker_color, MARKER_LINE_WIDTH)
        
        # Always draw as open circle, only color changes
        pen.setStyle(Qt.SolidLine)
        painter.setBrush(Qt.NoBrush)

        painter.setPen(pen)
        
        # Draw circle
        half_size = MARKER_SIZE // 2
        painter.drawEllipse(marker_x - half_size, 
                            marker_y - half_size, 
                            MARKER_SIZE, MARKER_SIZE)
            
    def mousePressEvent(self, event):
        """Handle mouse press for selection (left-click only)."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self, event)
        # Right-click handled by contextMenuEvent
        
    def mouseDoubleClickEvent(self, event):
        """Handle double-click for highlighting (left-click only)."""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self)
    
    def contextMenuEvent(self, event):
        """Show context menu on right-click."""
        menu = QMenu(self)
        
        # Select Image action (primary action)
        select_action = QAction("Select Image", self)
        select_action.triggered.connect(lambda: self.select_image_requested.emit(self.data_item.image_path))
        menu.addAction(select_action)
        
        menu.exec_(event.globalPos())
        
    def enterEvent(self, event):
        """Handle mouse enter for hover detection."""
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.data_item.set_hovered(True)
            self.update_selection_visuals()
            self.hovered.emit(self.data_item.camera.image_path)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave for hover detection."""
        self.data_item.set_hovered(False)
        self.update_selection_visuals()
        self.unhovered.emit(self.data_item.camera.image_path)
        super().leaveEvent(event)


# ----------------------------------------------------------------------------------------------------------------------
# CameraGrid
# ----------------------------------------------------------------------------------------------------------------------


class CameraGrid(QWidget):
    """
    Scrollable grid of camera thumbnails with virtualization.
    
    Signals:
        camera_selected: Emitted when a camera is selected (double-click) - loads image
        camera_highlighted_single: Emitted when single camera is highlighted (single-click) - changes 3D view
        cameras_highlighted: Emitted when highlight selection changes
        selection_changed: Emitted when any selection state changes
        camera_hovered: Emitted when a camera is hovered with Ctrl
        camera_unhovered: Emitted when hover ends
    """
    
    # User-intent signals (CameraGrid emits intent; SelectionManager is authoritative)
    camera_selected = pyqtSignal(str)              # image_path when double-clicked (loads image)
    camera_highlighted_single = pyqtSignal(str)    # image_path when single-clicked (changes 3D view)

    # Intent signals (preferred): emit requests; do NOT mutate model directly
    selection_requested = pyqtSignal(list)         # request to set selections (list of image_paths)
    toggle_requested = pyqtSignal(str)             # request to toggle a single path (Ctrl+Click)
    active_requested = pyqtSignal(str)             # request to set active camera (double-click)
    clear_requested = pyqtSignal()                 # request to clear selections (preserve active)

    # Backwards-compat signals (deprecated: CameraGrid should not emit authoritative selection_changed)
    cameras_highlighted = pyqtSignal(list)         # deprecated: list of image_paths
    selection_changed = pyqtSignal(list)           # deprecated: list of all selected/highlighted paths

    camera_hovered = pyqtSignal(str)               # image_path when hovered with Ctrl
    camera_unhovered = pyqtSignal(str)             # image_path when hover ends
    
    def __init__(self, model, mvat_window=None, parent=None):
        """
        Initialize the CameraGrid.
        
        Args:
            mvat_window: Reference to the parent MVATWindow
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.mvat_window = mvat_window
        self.model = model
        
        # Data management
        self.data_items = []                    # List of CameraDataItem
        self.widgets_by_path = {}               # image_path -> CameraImageWidget
        self.widget_positions = {}              # image_path -> QRect
        
        # Selection tracking (delegated to model)
        self.last_clicked_index = -1            # For shift-click range selection
        
        # Display settings
        self.thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        
        # Debounce timer for layout updates
        self._layout_timer = QTimer()
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._do_recalculate_layout)
        
        # Setup UI
        self._setup_ui()
        # Sync UI when model changes
        try:
            self.model.active_changed.connect(self._sync_ui_to_model)
            self.model.selection_changed.connect(self._sync_ui_to_model)
        except Exception:
            pass
        
    def _setup_ui(self):
        """Setup the widget UI with toolbar and scroll area."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # --- Initialize Toolbar Widgets (Do not add to layout yet) ---
        self.stats_label = QLabel("Cameras: 0")
        self.stats_label.setStyleSheet("color: #333;")

        self.size_label = QLabel("Size:")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(MIN_THUMBNAIL_SIZE, MAX_THUMBNAIL_SIZE)
        self.size_slider.setValue(self.thumbnail_size)
        self.size_slider.setFixedWidth(100)
        self.size_slider.setToolTip("Adjust thumbnail size")
        self.size_slider.valueChanged.connect(self._on_size_changed)
        
        self.size_value_label = QLabel(f"{self.thumbnail_size}px")
        self.size_value_label.setMinimumWidth(50)
        
        self.selected_label = QLabel("None selected")
        self.selected_label.setStyleSheet("color: #666;")
        
        self.selection_label = QLabel("0 highlighted")
        self.selection_label.setStyleSheet("color: #666;")

        # Load Cameras button (disabled by default until MVAT manager is wired)
        self.load_btn = QToolButton()
        self.load_btn.setText("Load Cameras")
        self.load_btn.setToolTip("Load cameras into MVAT (build frustums, markers, etc.)")
        self.load_btn.clicked.connect(self._on_load_cameras_clicked)
        self.load_btn.setEnabled(False)

        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.clear_btn.setToolTip("Clear all selections (Escape)")
        self.clear_btn.clicked.connect(self.clear_all_selections)
        
        # --- Scroll Area ---
        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #1e1e1e;")
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.scroll_area.setWidget(self.content_widget)

        # Placeholder label for empty state
        self._placeholder_label = QLabel(
            "No cameras available\nLoad camera lists to populate the grid.", 
            self.content_widget
        )
        self._placeholder_label.setStyleSheet("color: white; background-color: #1e1e1e; font-size: 14px; padding: 16px;")
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        self._show_placeholder()  # Show placeholder initially
        
        # Only add the scroll area to the main layout!
        layout.addWidget(self.scroll_area)
        
        # Set minimum width
        self.setMinimumWidth(MIN_THUMBNAIL_SIZE + 20)
        
    # --------------------------------------------------------------------------
    # DockWrapper Hooks
    # --------------------------------------------------------------------------
    
    def create_top_toolbar(self) -> QToolBar:
        """Create the top toolbar containing grid settings and stats."""
        toolbar = QToolBar("Camera Grid Tools")
        toolbar.setMovable(False)
        
        # Use a container to preserve the stretching/spacing behavior
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        layout.addWidget(self.stats_label)
        
        layout.addWidget(self.size_label)
        layout.addWidget(self.size_slider)
        layout.addWidget(self.size_value_label)
        
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)
        
        layout.addWidget(self.selected_label)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)
        
        layout.addWidget(self.selection_label)
        
        # Add stretch to push the Clear button to the far right
        layout.addStretch(1)
        # Place Load Cameras immediately before the Clear button
        try:
            layout.addWidget(self.load_btn)
        except Exception:
            pass

        layout.addWidget(self.clear_btn)
        
        toolbar.addWidget(container)
        
        return toolbar

    def _on_load_cameras_clicked(self):
        """Handler for the Load Cameras button.

        Attempts to call the MVAT manager's load_cameras method via the
        wired `mvat_window` reference. Shows an informational message if
        the manager isn't available or if an error occurs.
        """
        if not getattr(self, 'mvat_window', None):
            QMessageBox.information(self, "No MVAT", "MVAT manager is not available.")
            return

        mgr = getattr(self.mvat_window, 'mvat_manager', None)
        if mgr is None:
            QMessageBox.information(self, "No MVAT", "MVAT manager is not available.")
            return

        try:
            mgr.load_cameras()
        except Exception as e:
            QMessageBox.warning(self, "Load Cameras Failed", f"Failed to load cameras: {e}")
        
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
        # Selection state is managed by SelectionManager
        self.model.clear_selections()
        self.last_clicked_index = -1
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Start progress for loading camera / thumbnails        
        progress = ProgressBar(self, title="Setting Cameras")
        progress.show()
        progress.start_progress(len(cameras))
        
        try:
        
            # Create data items and widgets
            for path, camera in cameras.items():
                data_item = CameraDataItem(camera)
                self.data_items.append(data_item)
                
                widget = CameraImageWidget(data_item, self.model, self.thumbnail_size, self.content_widget)
                widget.clicked.connect(self._on_widget_clicked)
                widget.double_clicked.connect(self._on_widget_double_clicked)
                widget.select_image_requested.connect(self._on_widget_select_image_requested)
                widget.hovered.connect(self.camera_hovered)
                widget.unhovered.connect(self.camera_unhovered)
                self.widgets_by_path[path] = widget
                
                progress.update_progress()  # Update progress bar for each camera loaded
                
        except Exception as e:
            print(f"Error loading camera thumbnails: {e}")
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Close progress
            progress.finish_progress()
            progress.close()
            progress = None
        
        # Show or hide placeholder depending on whether we have cameras
        if not cameras:
            self._show_placeholder()
        else:
            self._hide_placeholder()

        # Calculate layout
        self.recalculate_layout()
        
        # Update selection label
        self._update_selection_label()
        
    def _clear_widgets(self):
        """Remove all widgets from the content widget."""
        for widget in self.widgets_by_path.values():
            widget.hide()  # Immediately hide to prevent ghost widgets
            widget.setParent(None)
            widget.deleteLater()

    def _show_placeholder(self, text: str = None):
        """Show placeholder in the content widget area."""
        try:
            if text:
                self._placeholder_label.setText(text)
            self._placeholder_label.setGeometry(self.content_widget.rect())
            self._placeholder_label.show()
        except Exception:
            pass

    def _hide_placeholder(self):
        """Hide the placeholder label."""
        try:
            self._placeholder_label.hide()
        except Exception:
            pass
            
    def clear_cameras(self):
        """Clear all cameras from the grid."""
        self._clear_widgets()
        self.data_items.clear()
        self.widgets_by_path.clear()
        self.widget_positions.clear()
        # Selection state is managed by SelectionManager
        self.model.clear_selections()
        # Show placeholder when empty
        self._show_placeholder()
        
    def recalculate_layout(self):
        """Schedule a layout recalculation (debounced)."""
        self._layout_timer.start(50)  # 50ms debounce
    
    # TODO figure out which method needs a progress bar
    def _do_recalculate_layout(self):
        """Actually recalculate widget positions."""
        if not self.data_items:
            # Ensure placeholder is visible when there are no items
            self._show_placeholder()
            return
        else:
            # Hide placeholder when there are items to display
            self._hide_placeholder()
            
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
        
        # Update visible widgets to refresh viewport and clear stale references
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

    def update_markers(self, projections: dict, accuracies: dict,
                       highlighted_paths: set, visibility_status: dict,
                       selected_path: str = None):
        """
        Batch update markers for visible widgets.

        Args:
            projections: dict mapping image_path -> (x, y, is_valid)
            accuracies: dict mapping image_path -> has_accurate_depth (bool)
            highlighted_paths: set of highlighted image paths
            visibility_status: dict mapping image_path -> is_occluded (bool)
            selected_path: currently selected/active image path
        """
        visible_widgets = self.get_visible_widgets()
        highlighted_paths = set(highlighted_paths or [])
        if selected_path is None:
            selected_path = self.model.active_path if self.model else None

        # Only update widgets that are visible to avoid expensive ops
        for path, widget in visible_widgets.items():
            # Only draw markers for selected or highlighted cameras
            allowed = highlighted_paths | ({selected_path} if selected_path else set())
            if path not in allowed:
                widget.clear_marker()
                continue

            proj = projections.get(path)
            if not proj:
                widget.clear_marker()
                continue

            px, py, is_valid = proj
            if not is_valid:
                widget.clear_marker()
                continue

            acc = accuracies.get(path, False)
            is_occluded = visibility_status.get(path, False)

            # Determine color
            if path == selected_path:
                color = MARKER_COLOR_SELECTED if acc else MARKER_COLOR_INVALID
            elif path in highlighted_paths:
                color = MARKER_COLOR_HIGHLIGHTED if acc else MARKER_COLOR_INVALID
            else:
                color = MARKER_COLOR_INVALID

            if is_occluded:
                color = MARKER_COLOR_INVALID

            widget.set_marker_position(px, py, accurate=acc, color=color, is_occluded=is_occluded)

    def clear_all_markers(self):
        """Clear markers from all visible camera widgets."""
        visible_widgets = self.get_visible_widgets()
        for widget in visible_widgets.values():
            widget.clear_marker()
    
    def get_highlighted_cameras(self) -> list:
        """
        Get list of Camera objects for all currently highlighted cameras.
        
        Used by MousePositionBridge to create rays from highlighted cameras
        to the 3D world point determined by the selected camera's ray.
        
        Returns:
            List of Camera objects for highlighted paths.
        """
        # Return Camera objects for all paths currently selected in the model
        cameras = []
        for path in (self.model.get_selected_list() if self.model else []):
            widget = self.widgets_by_path.get(path)
            if widget and widget.data_item:
                cameras.append(widget.data_item.camera)
        return cameras
                
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
    
    # TODO if a widget is clicked but it corresponds to a camera already selected / highlighted, make sure
    # we do not attempt to update the point cloud again unnecessarily.
    def _on_widget_clicked(self, widget, event):
        """Handle widget click for selection/highlighting using SelectionManager."""
        path = widget.data_item.image_path
        modifiers = event.modifiers()
        
        # BLOCK: If active camera is orthographic, force exit to clicked camera
        if self.model and self.model.active_path:
            active_widget = self.widgets_by_path.get(self.model.active_path)
            if active_widget:
                active_camera = active_widget.data_item.camera
                if getattr(active_camera, 'is_orthographic', False):
                    print(f"🚫 Exiting orthomosaic view → activating {path}")
                    self.active_requested.emit(path)
                    self.camera_selected.emit(path)
                    self.last_clicked_index = next((i for i, item in enumerate(self.data_items) if item.image_path == path), -1)
                    return  # Bypass all Ctrl/Shift logic

        # Find index of clicked item
        clicked_index = next((i for i, item in enumerate(self.data_items) if item.image_path == path), -1)
        if modifiers & Qt.ControlModifier:
            # Ctrl+Click: request toggle selection
            self.toggle_requested.emit(path)
        elif modifiers & Qt.ShiftModifier and self.last_clicked_index >= 0:
            # Shift+Click: request range selection
            start, end = sorted([self.last_clicked_index, clicked_index])
            paths_in_range = [item.image_path for item in self.data_items[start:end + 1]]
            self.selection_requested.emit(paths_in_range)
        else:
            # Plain click: request single selection
            self.selection_requested.emit([path])
            # Emit single highlight intent (for 3D view perspective change)
            self.camera_highlighted_single.emit(path)

        # Update last clicked index
        self.last_clicked_index = clicked_index
        
    def _on_widget_double_clicked(self, widget):
        """Handle widget double-click for highlighting only (no image load)."""
        path = widget.data_item.image_path
        
        # BLOCK: Same logic as single click - force exit if in ortho view
        if self.model and self.model.active_path:
            active_widget = self.widgets_by_path.get(self.model.active_path)
            if active_widget:
                active_camera = active_widget.data_item.camera
                if getattr(active_camera, 'is_orthographic', False):
                    print(f"🚫 Exiting orthomosaic view → activating {path}")
                    self.active_requested.emit(path)
                    self.camera_selected.emit(path)
                    return
        
        # Double-click now acts like single-click - just highlights
        # Plain click: request single selection
        self.selection_requested.emit([path])
        # Emit single highlight intent (for 3D view perspective change)
        self.camera_highlighted_single.emit(path)
    
    def _on_widget_select_image_requested(self, path):
        """Handle 'Select Image' from context menu - sets active camera and loads image."""
        # Emit request to set active camera (SelectionManager should act)
        self.active_requested.emit(path)
        # Emit grid-level signal (still used to trigger image load)
        self.camera_selected.emit(path)

    def _sync_ui_to_model(self, *args):
        """Force visible widgets to repaint their borders based on model state."""
        visible_widgets = self.get_visible_widgets()
        for widget in visible_widgets.values():
            widget.update()
        self._update_selection_label()
    
    def highlight_all(self):
        """Highlight all cameras (Ctrl+A)."""
        # Request selecting all cameras
        all_paths = [d.image_path for d in self.data_items]
        self.selection_requested.emit(all_paths)
        
    def clear_all_selections(self):
        """Clear all highlights but preserve the selected camera.
        
        The selected camera (green border) is never cleared - only highlighted
        cameras (cyan borders) are cleared. This ensures there's always a
        main camera with its point cloud subset visible.
        """
        # Request clearing secondary selections (preserve active)
        self.clear_requested.emit()
        
    def _update_selection_label(self):
        """Update the selection count label."""
        selected_set = self.model.get_selected_list() if self.model else []
        count = len(selected_set)
        self.selection_label.setText(f"{count} highlighted")

        # Update selected camera label using model.active_path
        active = self.model.active_path if self.model else None
        if active and active in self.widgets_by_path:
            camera_label = self.widgets_by_path[active].data_item.label
            self.selected_label.setText(f"Selected: {camera_label}")
        else:
            self.selected_label.setText("None selected")
        
    def update_hover_visuals(self, ctrl_pressed):
        """Update hover visuals for the currently hovered widget based on Ctrl state."""
        for widget in self.widgets_by_path.values():
            if widget.data_item.is_hovered:
                widget.data_item.set_hovered(ctrl_pressed)
                widget.update_selection_visuals()
                if ctrl_pressed:
                    self.camera_hovered.emit(widget.data_item.camera.image_path)
                else:
                    self.camera_unhovered.emit(widget.data_item.camera.image_path)
                break  # Only one can be hovered at a time
        
    def set_camera_order(self, ordered_paths):
        """
        Reorder cameras in the grid based on provided path list.
        
        Cameras will be displayed in the order specified by ordered_paths.
        Any cameras not in the list will be hidden/removed from display.
        
        Args:
            ordered_paths (list): List of image paths in desired display order
        """
        if not ordered_paths:
            return
        
        # Create new ordered data_items list
        new_data_items = []
        
        for path in ordered_paths:
            # Find the data item for this path
            for data_item in self.data_items:
                if data_item.image_path == path:
                    new_data_items.append(data_item)
                    break
        
        # Update the data_items list
        self.data_items = new_data_items
        
        # Recalculate layout with new order
        self.recalculate_layout()
        
        # Scroll to top to show the newly ordered cameras (most relevant ones first)
        self.scroll_area.verticalScrollBar().setValue(0)
    
    def render_selection_from_path(self, path):
        """
        Update the grid to show selection from external source (e.g., 3D picker).
        
        Args:
            path (str): The image path to select
        """
        if path not in self.widgets_by_path:
            return

        # Delegate selection to model (this will include active in selected_paths)
        self.model.set_active(path)

        # Scroll to make visible
        pos_rect = self.widget_positions.get(path)
        if pos_rect:
            self.scroll_area.ensureVisible(
                pos_rect.center().x(),
                pos_rect.center().y(),
                pos_rect.width(),
                pos_rect.height()
            )

        # Sync visuals
        self._sync_ui_to_model()
        
    def render_highlight_from_paths(self, paths):
        """
        Update the grid to show highlights from external source.
        
        Args:
            paths (list): List of image paths to highlight
        """
        # Delegate to model (will preserve active camera in set)
        self.model.set_selections(paths)
        self._sync_ui_to_model()
                
    def get_highlighted_cameras(self):
        """
        Get list of highlighted Camera objects.
        
        Returns:
            list: List of Camera objects that are highlighted
        """
        cameras = []
        for path in (self.model.get_selected_list() if self.model else []):
            if path in self.widgets_by_path:
                cameras.append(self.widgets_by_path[path].data_item.camera)
        return cameras
    
    def get_selected_camera(self):
        """
        Get the selected Camera object.
        
        Returns:
            Camera or None: The selected camera, or None if none selected
        """
        active = self.model.active_path if self.model else None
        if active and active in self.widgets_by_path:
            return self.widgets_by_path[active].data_item.camera
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
        selected_set = self.model.get_selected_list() if self.model else []
        count = len(selected_set)
        if count == 0:
            return
            
        menu = QMenu(self)
        
        # Header showing count
        header = QAction(f"Actions for {count} camera(s)", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()
        
        # Select Image action (sets active camera and loads)
        select_action = QAction("Select First Image", self)
        select_action.triggered.connect(self._select_first_image)
        menu.addAction(select_action)
        
        menu.addSeparator()
        
        # Go to first highlighted image (deprecated, kept for backwards compatibility)
        goto_action = QAction("Go to First Image", self)
        goto_action.triggered.connect(self._goto_first_highlighted)
        menu.addAction(goto_action)
        
        # Clear selection
        clear_action = QAction("Clear Selection", self)
        clear_action.triggered.connect(self.clear_all_selections)
        menu.addAction(clear_action)
        
        menu.exec_(event.globalPos())
        
    def _select_first_image(self):
        """Select and load the first highlighted camera's image."""
        selected_set = self.model.get_selected_list() if self.model else []
        if selected_set:
            first_path = selected_set[0]
            # Request to set active camera and load image
            self.active_requested.emit(first_path)
            self.camera_selected.emit(first_path)
        
    def _goto_first_highlighted(self):
        """Navigate to the first highlighted camera's image."""
        selected_set = self.model.get_selected_list() if self.model else []
        if selected_set and self.mvat_window:
            first_path = selected_set[0]
            self.camera_selected.emit(first_path)