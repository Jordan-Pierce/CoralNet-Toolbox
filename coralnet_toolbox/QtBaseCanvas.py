"""
BaseCanvas - Lightweight, reusable QGraphicsView subclass for image display and navigation.

This class encapsulates pure viewing responsibilities: image display, zoom/pan navigation,
Z-channel visualization, and marker slots. It is designed to be inherited by AnnotationWindow
and reused in Phase 2's context matrix for multi-viewport displays.
"""

import warnings
import traceback
import numpy as np

import pyqtgraph as pg
from PyQt5.QtGui import QMouseEvent, QPixmap, QImage, QBrush, QColor, QPen, QPainterPath, QTransform
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QSize, QObject
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsLineItem,
                             QGraphicsItem, QGraphicsPathItem, QLabel, QApplication)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BaseCanvas(QGraphicsView):
    """
    Lightweight viewport for image display with native zoom/pan navigation.
    
    Signals:
        viewNavigated: Emitted after zoom/pan, carrying (center_x, center_y, zoom_factor)
        mouseHovered: Emitted on mouse move, carrying scene coordinates (x, y)
    
    Attributes:
        scene: QGraphicsScene for rendering
        active_image: Whether an image is currently loaded
        pixmap_image: Source QPixmap for the displayed image
        current_image_path: String identifier for the current image
        zoom_factor: Current zoom level
        z_item: QGraphicsPixmapItem for Z-channel visualization layer
    """
    
    viewNavigated = pyqtSignal(float, float, float)  # center_x, center_y, zoom_factor
    mouseHovered = pyqtSignal(float, float)  # scene_x, scene_y
    
    def __init__(self, parent=None):
        """Initialize the base canvas."""
        super().__init__(parent)
        
        # Create and set the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set dark background for the annotation workspace
        try:
            self.scene.setBackgroundBrush(QBrush(QColor('#1e1e1e')))
            self.setBackgroundBrush(QBrush(QColor('#1e1e1e')))
            self.viewport().setStyleSheet("background-color: #1e1e1e;")
        except Exception:
            pass
        
        # Image state
        self.pixmap_image = None
        self.active_image = False
        self.current_image_path = None
        self._base_image_item = None  # Reference to the base image QGraphicsPixmapItem
        
        # Navigation state
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0  # Tracks absolute rotation in degrees
        self._rotate_active = False
        self._rotate_start_angle = 0.0
        self._rotate_start_mouse_angle = 0.0
        self._rotate_start_canvas_angle = 0.0
        self._min_zoom = 1.0
        self._pan_active = False
        self._pan_start = None
        
        # Z-channel visualization state
        self.z_item = None  # QGraphicsPixmapItem for Z-channel layer
        self.z_data_raw = None  # Raw Z-channel data
        self.z_data_normalized = None  # Normalized to 0-255
        self.z_data_min = None  # Min value in valid data
        self.z_data_max = None  # Max value in valid data
        self.z_data_shape = None  # Shape of Z-channel array
        self.z_nodata_mask = None  # Boolean mask of invalid pixels
        self.dynamic_z_scaling = False  # Whether to rescale based on visible area
        self._dynamic_range_timer = QTimer()  # Debounce timer for dynamic range updates
        self._dynamic_range_timer.setSingleShot(True)
        self._dynamic_range_timer.timeout.connect(self.update_dynamic_range)
        # default debounce delay (ms) — AnnotationWindow can override
        self.dynamic_range_update_delay = 500
        # Scratchpad for live vector trails from context canvases
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
        
        # Marker slots (containers; Phase 4 will populate these)
        self._static_marker = None
        self._dynamic_marker = None
        self._cursor_preview_item = None  # Preview rect for tool cursor propagation
        self._mask_overlay_item = None    # Read-only MaskAnnotation overlay for brush propagation

        # Read-only annotation overlays (Phase 6)
        self._readonly_annotation_items = []
        
        # Placeholder label for empty canvas
        self._placeholder_label = QLabel(
            "No image loaded\nImport or drag and drop an image.",
            self.viewport()
        )
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setStyleSheet(
            "color: #888888; font-size: 14px; background-color: transparent;"
        )
        
        # View transformation settings
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        
        # --- NEW OPTIMIZATION FLAGS ---
        # 1. SmartViewportUpdate analyzes the bounding rects of changes to decide 
        # whether to redraw a specific region or the whole viewport.
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        
        # 2. Prevent Qt from saving/restoring the painter state for every single item.
        # This saves massive CPU overhead when rendering thousands of items.
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState)
        
        # 3. Prevent Qt from doing sub-pixel antialiasing adjustments during fast pans.
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing)
    
    # ==================== Navigation Events ====================
    
    def wheelEvent(self, event: QMouseEvent):
        """Handle mouse wheel events for zooming."""
        if not self.active_image:
            return
        
        # Determine zoom direction
        if event.angleDelta().y() > 0:
            factor = 1.1  # Zoom in
        else:
            factor = 0.9  # Zoom out
        
        # Calculate new zoom level
        new_zoom = self.zoom_factor * factor
        
        # Prevent zooming out beyond minimum
        if new_zoom < self._min_zoom and factor < 1:
            new_zoom = self._min_zoom
            factor = new_zoom / self.zoom_factor
            
            # Apply zoom
            self.scale(factor, factor)
            self.zoom_factor = new_zoom
            
            # Center image when at minimum zoom
            self.centerOn(self.scene.sceneRect().center())
            self._emit_view_navigated()
            return
        
        # Store position before zoom for anchor-under-mouse
        old_pos = self.mapToScene(event.pos())
        
        # Apply zoom
        self.scale(factor, factor)
        self.zoom_factor = new_zoom
        
        # Correct position for natural zoom effect
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        
        # When zoomed to minimum, ensure perfect centering
        if abs(new_zoom - self._min_zoom) < 0.01:
            self.centerOn(self.scene.sceneRect().center())
        
        self._emit_view_navigated()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for rotation or panning."""
        if event.button() == Qt.RightButton and self.active_image:
            # Check for Ctrl + RightButton = rotation interaction
            if event.modifiers() == Qt.ControlModifier:
                # Initiate Rotation
                self._rotate_active = True
                self.setCursor(Qt.ClosedHandCursor)
                
                # Calculate the starting angle relative to the viewport center
                center = self.viewport().rect().center()
                dx = event.pos().x() - center.x()
                dy = event.pos().y() - center.y()
                # Store the baseline angle of the mouse and the current canvas rotation
                self._rotate_start_mouse_angle = np.degrees(np.arctan2(dy, dx))
                self._rotate_start_canvas_angle = self.rotation_angle
            else:
                # Initiate Native Pan
                self._pan_active = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for rotation, panning, and hover tracking."""
        # Handle rotation
        if self._rotate_active and self.active_image:
            center = self.viewport().rect().center()
            dx = event.pos().x() - center.x()
            dy = event.pos().y() - center.y()
            current_mouse_angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate how much the mouse has rotated since the click
            angle_delta = current_mouse_angle - self._rotate_start_mouse_angle
            new_canvas_angle = self._rotate_start_canvas_angle + angle_delta
            
            # Apply absolute rotation
            self._set_absolute_rotation(new_canvas_angle)
            
            # Emit navigation signal so context matrix syncs live
            self._emit_view_navigated()
        
        # Handle panning
        elif self._pan_active:
            if not self.active_image:
                self._pan_active = False
                return
            
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            
            # Adjust scrollbars
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
        
        # Emit hover signal with scene coordinates
        scene_pos = self.mapToScene(event.pos())
        # emit floats for higher precision (consumers may cast if needed)
        self.mouseHovered.emit(scene_pos.x(), scene_pos.y())
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for rotation or panning."""
        if event.button() == Qt.RightButton:
            if self._rotate_active:
                self._rotate_active = False
            if self._pan_active:
                self._pan_active = False
            
            self.setCursor(Qt.ArrowCursor)
            self._emit_view_navigated()
        else:
            super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize events to maintain proper view fitting."""
        super().resizeEvent(event)
        
        # Fit view to image after resize
        if self.active_image and self.pixmap_image and self.scene:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            # Sync zoom state after fitInView
            self._calculate_min_zoom()
            self.zoom_factor = self.transform().m11()
        
        # Keep placeholder geometry in sync
        try:
            if self._placeholder_label and self._placeholder_label.isVisible():
                self._placeholder_label.setGeometry(self.viewport().rect())
        except Exception:
            pass
    
    # ==================== Placeholder Management ====================
    
    def _show_placeholder(self, text: str = None):
        """Show the centered placeholder label with optional custom text."""
        try:
            if text:
                self._placeholder_label.setText(text)
            self._placeholder_label.setGeometry(self.viewport().rect())
            self._placeholder_label.show()
        except Exception:
            pass
    
    def _hide_placeholder(self):
        """Hide the placeholder label."""
        try:
            self._placeholder_label.hide()
        except Exception:
            pass
    
    # ==================== Scene Management ====================
    
    def clear_scene(self):
        """Clear the graphics scene and reset related variables."""
        # Stop any pending dynamic range update
        self._dynamic_range_timer.stop()
        
        # Clean up scene items
        if self.scene:
            for item in list(self.scene.items()):
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
                    if hasattr(item, 'deleteLater'):
                        item.deleteLater()
        
        # Clear and recreate scene
        if self.scene:
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Reset image references
        self._base_image_item = None
        self.z_item = None
        
        # Clear read-only annotation overlay references
        self._readonly_annotation_items = []
        
        # Clear Z-channel data
        self.z_data_raw = None
        self.z_data_normalized = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None
        self.z_nodata_mask = None

        # Reset scratchpad state
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
        
        # Allow subclasses to clean up their scene-dependent items
        self._on_scene_cleared()
        
        # Show placeholder
        self._show_placeholder("No image loaded")
    
    def _on_scene_cleared(self):
        """Hook for subclasses to perform cleanup after scene is cleared."""
        # Recreate marker slots on the new scene
        try:
            self._init_markers()
        except Exception:
            traceback.print_exc()
    
    # ==================== Image Loading ====================
    
    def load_visuals(self, q_image, image_path, raster=None):
        """
        Load and display a QImage with optional Z-channel visualization.
        
        This is the canonical entry point for loading images into the canvas.
        
        Args:
            q_image (QImage): The full-resolution image to display
            image_path (str): Path identifier for the image
            raster (Raster, optional): Raster object with Z-channel data
        """
        # Clear previous state
        self.clear_scene()
        
        # Hide placeholder
        self._hide_placeholder()
        
        # Create and store pixmap
        self.pixmap_image = QPixmap(q_image)
        
        # Create base image item
        self._base_image_item = QGraphicsPixmapItem(self.pixmap_image)
        self._base_image_item.setZValue(-10)
        self.scene.addItem(self._base_image_item)
        
        # Update state
        self.current_image_path = image_path
        self.active_image = True
        
        # Load Z-channel visualization if available
        if raster is not None:
            self._load_z_channel_visualization(raster)
        
        # Fit to view and calculate min zoom
        self.fit_to_image()
    
    def fit_to_image(self):
        """Fit the entire image in the view and recalculate min zoom."""
        if not self.active_image:
            return
        
        image_rect = self.get_image_rect()
        self.fitInView(image_rect, Qt.KeepAspectRatio)
        
        # Recalculate minimum zoom
        self._calculate_min_zoom()
        
        # Sync zoom_factor to the actual transform scale after fitInView
        self.zoom_factor = self.transform().m11()
        
        self._emit_view_navigated()
    
    def _calculate_min_zoom(self):
        """Calculate the minimum zoom factor needed to fit image in viewport."""
        if not self.scene or not self.pixmap_image:
            self._min_zoom = 1.0
            return
        
        view_rect = self.viewport().rect()
        scene_rect = self.scene.sceneRect()
        
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            self._min_zoom = 1.0
            return
        
        width_ratio = view_rect.width() / scene_rect.width()
        height_ratio = view_rect.height() / scene_rect.height()
        
        # ---> Allow the view to scale down as far as needed to fit huge images! <---
        self._min_zoom = max(min(width_ratio, height_ratio), 0.0001)
    
    # ==================== Viewport Control API ====================
    
    def center_on_pixel(self, x, y):
        """Center the view on the given image pixel coordinate."""
        # ---> Prevent anchor fighting during panning <---
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        
        self.centerOn(QPointF(x, y))
        
        self.setTransformationAnchor(old_anchor)
    
    def set_zoom_level(self, factor):
        """Set the absolute view transform scale, preserving current rotation."""
        if factor <= 0:
            return
        
        self._apply_transform(factor, self.rotation_angle)
    
    def snap_to_target(self, target_x, target_y, relative_zoom, angle_degrees=0.0):
        """
        Snap to a specific pixel location with a proportional zoom level and synchronized rotation.
        
        Args:
            target_x (float): Target pixel X coordinate
            target_y (float): Target pixel Y coordinate
            relative_zoom (float): Zoom ratio relative to fit-to-view
                                   (1.0 = fit whole image, 2.0 = 2x beyond fit, etc.)
            angle_degrees (float): Target rotation angle in degrees (default 0.0)
        """
        if not self.active_image:
            return
        
        absolute_zoom = self._min_zoom * relative_zoom
        
        # 1. Get the pivot point (image center in scene coordinates)
        image_rect = self.get_image_rect()
        pivot_x = image_rect.width() / 2.0
        pivot_y = image_rect.height() / 2.0
        
        # 2. Build the explicit transform: translate -> rotate -> scale -> translate
        transform = QTransform()
        transform.translate(pivot_x, pivot_y)
        transform.rotate(angle_degrees)
        transform.scale(absolute_zoom, absolute_zoom)
        transform.translate(-pivot_x, -pivot_y)
        
        # 3. Apply the transform directly
        self.setTransform(transform)
        self.zoom_factor = absolute_zoom
        self.rotation_angle = angle_degrees
        
        # 4. Snap directly to the target pixel (no pan-restoration needed here)
        self.center_on_pixel(target_x, target_y)
        self._emit_view_navigated()
    
    def _apply_transform(self, zoom, angle):
        """
        Builds and applies the transform matrix from scratch, pivoting on the image center.
        This mirrors the QtiViewWidget approach to prevent pendulum-swinging during rotation.
        
        The transform sequence is:
        1. Translate origin to image center
        2. Apply rotation around that center
        3. Apply zoom scaling
        4. Translate origin back
        
        Then restore the view so the pan position is preserved.
        """
        if not self.active_image:
            return

        # 1. Save where we are currently looking so we don't lose our pan position
        current_view_center = self.mapToScene(self.viewport().rect().center())
        
        # 2. Get the pivot point (the center of the image in scene coordinates)
        image_rect = self.get_image_rect()
        pivot_x = image_rect.width() / 2.0
        pivot_y = image_rect.height() / 2.0
        
        # 3. Build the explicit transform: translate -> rotate -> scale -> translate
        transform = QTransform()
        transform.translate(pivot_x, pivot_y)
        transform.rotate(angle)
        transform.scale(zoom, zoom)
        transform.translate(-pivot_x, -pivot_y)
        
        # 4. Apply the matrix
        self.setTransform(transform)
        
        # 5. Restore the pan position directly
        self.centerOn(current_view_center)
        
        # 6. Update state trackers
        self.zoom_factor = zoom
        self.rotation_angle = angle

    def _set_absolute_rotation(self, angle_degrees):
        """Apply absolute rotation, triggered by mouse movement."""
        self._apply_transform(self.zoom_factor, angle_degrees)
    
    # ==================== Helper Methods ====================
    
    def viewportToScene(self):
        """Convert viewport rectangle to scene coordinates."""
        top_left = self.mapToScene(self.viewport().rect().topLeft())
        bottom_right = self.mapToScene(self.viewport().rect().bottomRight())
        return QRectF(top_left, bottom_right)
    
    def get_image_dimensions(self):
        """Get the dimensions of the currently loaded image."""
        if self.pixmap_image:
            return self.pixmap_image.size().width(), self.pixmap_image.size().height()
        return 0, 0
    
    def get_image_rect(self):
        """Get the bounding rectangle of the currently loaded image in scene coordinates."""
        if self.pixmap_image:
            return QRectF(0, 0, self.pixmap_image.width(), self.pixmap_image.height())
        return QRectF()
    
    def _emit_view_navigated(self):
        """Emit viewNavigated signal with current center and zoom."""
        if self.active_image:
            center = self.mapToScene(self.viewport().rect().center())
            self.viewNavigated.emit(center.x(), center.y(), self.zoom_factor)
    
    # ==================== Read-Only Annotation Overlays (Phase 6) ====================
    
    def _render_annotations_readonly(self, annotations):
        """Render annotations as non-interactive overlays on this canvas.
        
        Args:
            annotations (list): List of Annotation objects to display.
        """
        from coralnet_toolbox.Annotations import MaskAnnotation
        
        self._clear_readonly_annotations()
        
        for annotation in annotations:
            # Skip MaskAnnotation — it's a full-image raster overlay
            if isinstance(annotation, MaskAnnotation):
                continue
            
            try:
                item = self._create_readonly_graphics_item(annotation)
                if item is not None:
                    self.scene.addItem(item)
                    self._readonly_annotation_items.append(item)
            except Exception:
                traceback.print_exc()

        self.viewport().update()
    
    def _clear_readonly_annotations(self):
        """Remove all read-only annotation items from the scene."""
        for item in self._readonly_annotation_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except Exception:
                pass
        self._readonly_annotation_items = []
    
    def _create_readonly_graphics_item(self, annotation):
        """Create a non-interactive QGraphicsPathItem from an Annotation object.
        
        Args:
            annotation: An Annotation object with get_painter_path() method.
            
        Returns:
            QGraphicsPathItem or None if creation fails.
        """
        try:
            path = annotation.get_painter_path()
        except (NotImplementedError, AttributeError):
            return None
        
        if path is None or path.isEmpty():
            return None
        
        item = QGraphicsPathItem(path)
        
        # Style: label color at annotation transparency fill, 1px pen at full color
        color = QColor(annotation.label.color)
        fill_color = QColor(color)
        fill_color.setAlpha(annotation.transparency)
        
        pen = QPen(color, 1)
        pen.setCosmetic(True)  # Constant width regardless of zoom
        
        item.setBrush(QBrush(fill_color))
        item.setPen(pen)
        
        # Non-interactive
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setFlag(QGraphicsItem.ItemIsMovable, False)
        item.setAcceptHoverEvents(False)
        
        # Z-value above base image but below markers
        item.setZValue(5)
        
        # Tag with source annotation ID for selection matching
        item.setData(0, annotation.id)
        item._source_annotation_id = annotation.id
        
        return item
    
    def _highlight_readonly_annotation(self, annotation_id, highlighted):
        """Highlight or un-highlight a read-only annotation overlay.
        
        Args:
            annotation_id (str): The annotation UUID to highlight.
            highlighted (bool): Whether to highlight (True) or revert (False).
        """
        for item in self._readonly_annotation_items:
            if getattr(item, '_source_annotation_id', None) == annotation_id:
                try:
                    if highlighted:
                        pen = QPen(QColor(255, 255, 0), 3)
                        pen.setCosmetic(True)
                        item.setPen(pen)
                        item.setZValue(6)
                    else:
                        # Restore original style from annotation data
                        original_color = item.brush().color()
                        original_color.setAlpha(255)
                        pen = QPen(original_color, 1)
                        pen.setCosmetic(True)
                        item.setPen(pen)
                        item.setZValue(5)
                except Exception:
                    pass
                break
    
    # ==================== Z-Channel Visualization ====================
    
    def _load_z_channel_visualization(self, raster):
        """
        Load and initialize the Z-channel visualization.
        
        Args:
            raster: The Raster object containing Z-channel data
        """
        # Clean up old z_item if it exists
        if self.z_item is not None:
            if self.z_item.scene() == self.scene:
                self.scene.removeItem(self.z_item)
            self.z_item = None
        
        # Check if Z-channel data is available
        if raster.z_channel_lazy is None:
            return
        
        try:
            z_data = raster.z_channel_lazy
            
            # Store raw Z-channel data
            self.z_data_raw = z_data.copy()
            self.z_data_shape = z_data.shape
            
            # Create mask for NaN and nodata values
            nodata_mask = np.isnan(z_data)
            if raster.z_nodata is not None:
                nodata_mask |= (z_data == raster.z_nodata)
            
            # Normalize Z-channel data to 0-255
            if z_data.dtype == np.float32:
                valid_data = z_data[~nodata_mask]
                if len(valid_data) > 0:
                    self.z_data_min = np.min(valid_data)
                    self.z_data_max = np.max(valid_data)
                else:
                    self.z_data_min = 0.0
                    self.z_data_max = 1.0
                
                if self.z_data_min == self.z_data_max:
                    z_norm = np.zeros_like(z_data, dtype=np.uint8)
                else:
                    z_diff = self.z_data_max - self.z_data_min
                    z_norm = ((z_data - self.z_data_min) / z_diff * 255).astype(np.uint8)
                    z_norm[nodata_mask] = 0
            else:
                valid_data = z_data[~nodata_mask]
                if len(valid_data) > 0:
                    self.z_data_min = np.min(valid_data)
                    self.z_data_max = np.max(valid_data)
                else:
                    self.z_data_min = 0.0
                    self.z_data_max = 1.0
                
                if self.z_data_min == self.z_data_max:
                    z_norm = np.zeros_like(z_data, dtype=np.uint8)
                else:
                    z_diff = self.z_data_max - self.z_data_min
                    z_norm = ((z_data - self.z_data_min) / z_diff * 255).astype(np.uint8)
                    z_norm[nodata_mask] = 0
            
            # Store normalized data and nodata mask
            self.z_nodata_mask = nodata_mask
            self.z_data_normalized = z_norm
            
            # Create QImage and QPixmap
            h, w = z_norm.shape
            z_copy = np.ascontiguousarray(z_norm)
            q_img = QImage(z_copy.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            
            # Create graphics item
            self.z_item = QGraphicsPixmapItem(pixmap)
            self.z_item.setTransformationMode(Qt.SmoothTransformation)
            self.z_item.setPos(0, 0)
            self.z_item.setZValue(-5)  # Between base image (-10) and annotations (0+)
            self.z_item.setOpacity(0.5)  # Default opacity
            self.scene.addItem(self.z_item)
            self.z_item.hide()  # Hide until colormap is selected
            
        except Exception:
            traceback.print_exc()
            self.z_item = None
    
    def update_z_colormap(self, colormap_name):
        """
        Update the Z-channel visualization colormap.
        
        Args:
            colormap_name (str): Name of the colormap (e.g., 'Viridis', 'Plasma')
        """
        if self.z_item is None or self.z_data_normalized is None:
            return
        
        try:
            if colormap_name == 'None':
                self.z_item.hide()
            else:
                # Get colormap and apply to normalized data
                colormap = pg.colormap.get(colormap_name)
                lut = colormap.getLookupTable(nPts=256, alpha=True)
                z_colored = lut[self.z_data_normalized]
                
                # Apply nodata mask (set alpha to 0)
                if self.z_nodata_mask is not None:
                    z_colored[self.z_nodata_mask, 3] = 0
                
                # Create QImage and update pixmap
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 4, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
                self.z_item.show()
                
                # Update dynamic range if enabled
                if self.dynamic_z_scaling:
                    self.update_dynamic_range()
        except Exception:
            traceback.print_exc()
    
    def set_z_opacity(self, opacity):
        """
        Set the opacity of the Z-channel visualization.
        
        Args:
            opacity (float): Opacity value from 0.0 (transparent) to 1.0 (opaque)
        """
        opacity = max(0.0, min(1.0, opacity))
        if self.z_item is not None:
            self.z_item.setOpacity(opacity)
    
    def toggle_dynamic_z_scaling(self, enabled):
        """
        Toggle dynamic Z-range scaling based on visible area.
        
        Args:
            enabled (bool): Whether to enable dynamic scaling
        """
        self.dynamic_z_scaling = enabled
        
        if enabled and self.z_item is not None:
            self.update_dynamic_range()
        elif not enabled and self.z_item is not None:
            self._reset_z_channel_to_full_range(None)
    
    def schedule_dynamic_range_update(self):
        """Schedule a debounced dynamic range update."""
        if not self.dynamic_z_scaling:
            return
        
        self._dynamic_range_timer.stop()
        self._dynamic_range_timer.start(self.dynamic_range_update_delay)
    
    def update_dynamic_range(self):
        """Update Z-channel visualization for visible viewport range."""
        if not self.dynamic_z_scaling or self.z_item is None or self.z_data_raw is None:
            return
        
        try:
            # Get visible area in scene coordinates
            visible_rect = self.viewportToScene()
            
            # Clamp to image bounds
            image_rect = self.get_image_rect()
            visible_rect = visible_rect.intersected(image_rect)
            
            if visible_rect.isEmpty():
                return
            
            # Get pixel coordinates
            x1, y1 = int(visible_rect.left()), int(visible_rect.top())
            x2, y2 = int(visible_rect.right()), int(visible_rect.bottom())
            
            # Clamp to array bounds
            h, w = self.z_data_shape
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
            
            # Calculate dynamic range from visible data
            visible_data = self.z_data_raw[y1:y2, x1:x2]
            valid_mask = ~np.isnan(visible_data)
            
            if np.any(valid_mask):
                dynamic_min = np.min(visible_data[valid_mask])
                dynamic_max = np.max(visible_data[valid_mask])
            else:
                dynamic_min = self.z_data_min
                dynamic_max = self.z_data_max
            
            # Normalize visible range to 0-255
            if dynamic_min == dynamic_max:
                z_norm = np.zeros_like(self.z_data_raw, dtype=np.uint8)
            else:
                z_diff = dynamic_max - dynamic_min
                z_norm = ((self.z_data_raw - dynamic_min) / z_diff * 255).astype(np.uint8)
                z_norm[np.isnan(self.z_data_raw)] = 0
            
            # Update pixmap with remapped visualization
            if self.z_item is not None:
                colormap_name = 'Viridis'  # Default; override in subclass if needed
                colormap = pg.colormap.get(colormap_name)
                lut = colormap.getLookupTable(nPts=256, alpha=True)
                z_colored = lut[z_norm]
                
                if self.z_nodata_mask is not None:
                    z_colored[self.z_nodata_mask, 3] = 0
                
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 4, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
        
        except Exception:
            traceback.print_exc()
    
    def _reset_z_channel_to_full_range(self, colormap_name):
        """
        Reset Z-channel visualization to full data range.
        
        Args:
            colormap_name (str, optional): Colormap to apply. If None, uses 'Viridis'.
        """
        if colormap_name is None:
            colormap_name = 'Viridis'
        
        if self.z_item is None or self.z_data_normalized is None:
            return
        
        try:
            if colormap_name != 'None':
                colormap = pg.colormap.get(colormap_name)
                lut = colormap.getLookupTable(nPts=256, alpha=True)
                z_colored = lut[self.z_data_normalized]
                
                if self.z_nodata_mask is not None:
                    z_colored[self.z_nodata_mask, 3] = 0
                
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 4, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
        except Exception:
            traceback.print_exc()
    
    def clear_z_channel_visualization(self, image_path):
        """
        Clear Z-channel visualization for a specific image.
        
        Args:
            image_path (str): Path of the image with removed Z-channel
        """
        if image_path != self.current_image_path:
            return
        
        if self.z_item is not None:
            if self.z_item.scene() == self.scene:
                self.scene.removeItem(self.z_item)
            self.z_item = None
        
        # Clear cached data
        self.z_data_raw = None
        self.z_data_normalized = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None
        self.z_nodata_mask = None
    
    # ==================== Marker Slots (Phase 4) ====================

    def update_static_marker(self, x, y, color=None):
        """Update the static focal point marker at image pixel (x, y).
        
        Args:
            x, y: Pixel coordinates in image space.
            color: QColor for the marker. Default: QColor(0, 255, 0) (green).
        """
        if self._static_marker is None:
            return
        try:
            # Bounds check
            if self.pixmap_image and not (0 <= x < self.pixmap_image.width() and
                                           0 <= y < self.pixmap_image.height()):
                self._static_marker.hide()
                return

            self._static_marker.setPos(x, y)
            color = color or QColor(0, 255, 0)
            pen = QPen(color, 2)
            for child in self._static_marker.childItems():
                try:
                    child.setPen(pen)
                except Exception:
                    pass
            self._static_marker.show()
        except Exception:
            traceback.print_exc()

    def clear_static_marker(self):
        """Clear the static focal point marker."""
        if self._static_marker is not None:
            try:
                self._static_marker.hide()
            except Exception:
                pass

    def update_dynamic_marker(self, x, y, color=None, is_valid=True):
        """Update the dynamic hover marker at image pixel (x, y).
        
        Args:
            x, y: Pixel coordinates in image space.
            color: QColor for the marker. Default: keeps current color.
            is_valid: If False, use dashed pen (occluded/estimated).
        """
        if self._dynamic_marker is None:
            return
        try:
            # Bounds check
            if self.pixmap_image and not (0 <= x < self.pixmap_image.width() and
                                           0 <= y < self.pixmap_image.height()):
                self._dynamic_marker.hide()
                return

            self._dynamic_marker.setPos(x, y)
            pen = QPen(color or QColor(0, 255, 0), 2)
            if not is_valid:
                pen.setStyle(Qt.DashLine)
            else:
                pen.setStyle(Qt.SolidLine)
            self._dynamic_marker.setPen(pen)
            self._dynamic_marker.show()
        except Exception:
            traceback.print_exc()

    def clear_dynamic_marker(self):
        """Clear the dynamic hover marker."""
        if self._dynamic_marker is not None:
            try:
                self._dynamic_marker.hide()
            except Exception:
                pass

    # ==================== Marker Initialization ====================
    def _init_markers(self):
        """Create marker graphics items (hidden by default) and add to the scene."""
        try:
            # remove existing if leftover
            if self._static_marker is not None and self._static_marker.scene() == self.scene:
                try:
                    self.scene.removeItem(self._static_marker)
                except Exception:
                    pass
            if self._dynamic_marker is not None and self._dynamic_marker.scene() == self.scene:
                try:
                    self.scene.removeItem(self._dynamic_marker)
                except Exception:
                    pass

            # Static crosshair group
            self._static_marker = QGraphicsItemGroup()
            pen = QPen(QColor(255, 64, 64))
            pen.setWidth(2)
            lh = QGraphicsLineItem(-12, 0, 12, 0)
            lv = QGraphicsLineItem(0, -12, 0, 12)
            el = QGraphicsEllipseItem(-6, -6, 12, 12)
            for it in (lh, lv, el):
                it.setPen(pen)
                it.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                self._static_marker.addToGroup(it)
            self._static_marker.setZValue(100)
            self.scene.addItem(self._static_marker)
            self._static_marker.hide()

            # Dynamic hover circle
            self._dynamic_marker = QGraphicsEllipseItem(-5, -5, 10, 10)
            self._dynamic_marker.setPen(QPen(QColor(255, 200, 0)))
            self._dynamic_marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._dynamic_marker.setZValue(100)
            self.scene.addItem(self._dynamic_marker)
            self._dynamic_marker.hide()

            # Tool cursor preview rect (created lazily; just reset the reference here)
            self._cursor_preview_item = None
            # Mask overlay item (created lazily; reset reference on scene rebuild)
            self._mask_overlay_item = None
        except Exception:
            traceback.print_exc()

    # ==================== Read-Only Annotations (Phase 6) ====================

    def render_readonly_annotations(self, annotation_data_list):
        """
        Render annotations as non-interactive visual overlays.

        Stub for Phase 6 implementation.

        Args:
            annotation_data_list: List of annotation data to render
        """
        # TODO: implement rendering of read-only annotations for context canvases
        pass

    # ==================== Cursor Preview (Tool Propagation) ====================

    def update_cursor_preview(self, u: float, v: float, item_factory):
        """Show a tool cursor preview at image pixel (u, v).

        The item is produced by item_factory(u, v) so each tool can supply its
        own style (patch square, brush circle, etc.).

        Args:
            u, v: Centre pixel coordinates in image space.
            item_factory: callable(u, v) -> QGraphicsItem
        """
        # Remove the previous preview item before creating a new one
        self.clear_cursor_preview()

        try:
            item = item_factory(u, v)
            item.setZValue(101)  # Above markers
            item.setFlag(QGraphicsItem.ItemIsSelectable, False)
            item.setFlag(QGraphicsItem.ItemIsMovable, False)
            item.setAcceptHoverEvents(False)
            self.scene.addItem(item)
            self._cursor_preview_item = item
        except Exception:
            pass

    def clear_cursor_preview(self):
        """Remove the tool cursor preview item from the scene."""
        if self._cursor_preview_item is not None:
            try:
                if self._cursor_preview_item.scene() is not None:
                    self._cursor_preview_item.scene().removeItem(self._cursor_preview_item)
            except Exception:
                pass
            self._cursor_preview_item = None

    # ==================== Mask Overlay (Brush Propagation) ====================

    def set_mask_overlay(self, mask_annotation):
        """Display or refresh a MaskAnnotation as a read-only overlay on this canvas.

        Creates a lightweight MaskGraphicsItem that paints directly from
        mask_annotation.qimage (kept up-to-date by update_mask) so the view stays
        in sync with every brush stroke without rebuilding the full pixmap.
        Safe to call repeatedly — reuses the existing item unless the annotation changes.
        """
        from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskGraphicsItem

        item = self._mask_overlay_item
        needs_new = (
            item is None
            or item.scene() is None
            or item.mask_annotation is not mask_annotation
        )
        if needs_new:
            self.clear_mask_overlay()
            item = MaskGraphicsItem(mask_annotation)
            item.setZValue(2)  # Above base image, below markers
            item.setAcceptHoverEvents(False)
            self.scene.addItem(item)
            self._mask_overlay_item = item

        item.update()

    def clear_mask_overlay(self):
        """Remove the mask overlay item from the scene."""
        if self._mask_overlay_item is not None:
            try:
                if self._mask_overlay_item.scene() is not None:
                    self._mask_overlay_item.scene().removeItem(self._mask_overlay_item)
            except Exception:
                pass
            self._mask_overlay_item = None

    # --- NEW SCRATCHPAD METHODS ---
    def add_to_scratchpad(self, u, v, size, shape, color):
        """Adds a vector shape to the canvas's live scratchpad."""
        if not self.scratchpad_item:
            self.scratchpad_item = QGraphicsPathItem()
            self.scratchpad_item.setZValue(3) # Hover above mask, below markers
            self.scratchpad_item.setPen(QPen(Qt.NoPen))
            self.scratchpad_item.setBrush(QBrush(color))
            self.scene.addItem(self.scratchpad_item)

        radius = size / 2.0
        if shape == 'circle':
            self.scratchpad_path.addEllipse(u - radius, v - radius, size, size)
        else:
            self.scratchpad_path.addRect(u - radius, v - radius, size, size)
            
        # WindingFill prevents the "striation/checkerboard" overlap bug
        self.scratchpad_path.setFillRule(Qt.WindingFill)
        self.scratchpad_item.setPath(self.scratchpad_path)

    def clear_scratchpad(self):
        """Removes the scratchpad overlay when the real NumPy mask is ready."""
        if self.scratchpad_item and self.scratchpad_item.scene():
            self.scene.removeItem(self.scratchpad_item)
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
