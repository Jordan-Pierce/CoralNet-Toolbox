import os
import warnings

import numpy as np
import torch

from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QPointF, pyqtSignal, QSignalBlocker, pyqtSlot, QEvent, QPoint
from PyQt5.QtGui import QPen, QColor, QPainter, QBrush, QPainterPath, QMouseEvent
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget, QGridLayout,
                             QMainWindow, QSplitter, QGroupBox, QGraphicsItem, QSlider,
                             QApplication, QGraphicsRectItem, QRubberBand)

from ultralytics import YOLO

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import pixmap_to_numpy

from .QtAnnotationDataItem import AnnotationDataItem
from .QtAnnotationImageWidget import AnnotationImageWidget
from .QtEmbeddingPointItem import EmbeddingPointItem
from .QtSettingsWidgets import AnnotationSettingsWidget
from .QtSettingsWidgets import EmbeddingSettingsWidget
from .QtSettingsWidgets import ModelSettingsWidget

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP
except ImportError:
    print("Warning: sklearn or umap not installed. Some features may be unavailable.")
    StandardScaler = None
    PCA = None
    TSNE = None
    UMAP = None


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_SIZE = 15
POINT_WIDTH = 3


# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingViewer(QWidget):
    """Custom QGraphicsView for interactive embedding visualization with zooming, panning, and selection."""

    # Define signals to report user actions to the ExplorerWindow controller
    selection_changed = pyqtSignal(list)
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        # Create the graphics scene first
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)

        # Initialize as a QWidget
        super(EmbeddingViewer, self).__init__(parent)
        self.explorer_window = parent

        # Create the actual graphics view
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMinimumHeight(200)

        # Custom rubber_band state variables
        self.rubber_band = None
        self.rubber_band_origin = QPointF()
        self.selection_at_press = set()

        self.points_by_id = {}  # Map annotation ID to embedding point
        self.previous_selection_ids = set()

        self.animation_offset = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)

        # Connect the scene's selection signal to our internal handler
        self.graphics_scene.selectionChanged.connect(self._on_scene_selection_changed)

        # Setup the UI with header
        self.setup_ui()
    
        # Connect mouse events to the graphics view
        self.graphics_view.mousePressEvent = self.mousePressEvent
        self.graphics_view.mouseDoubleClickEvent = self.mouseDoubleClickEvent
        self.graphics_view.mouseReleaseEvent = self.mouseReleaseEvent
        self.graphics_view.mouseMoveEvent = self.mouseMoveEvent
        self.graphics_view.wheelEvent = self.wheelEvent

    def setup_ui(self):
        """Set up the UI with header layout and graphics view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QHBoxLayout()
        self.home_button = QPushButton("Home")
        self.home_button.setToolTip("Reset view to fit all points")
        self.home_button.clicked.connect(self.fit_view_to_points)
        header_layout.addWidget(self.home_button)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Create a stacked layout for the view and placeholder
        self.view_stack = QWidget()
        stack_layout = QGridLayout(self.view_stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view.setParent(self.view_stack)
        stack_layout.addWidget(self.graphics_view, 0, 0)

        self.placeholder_label = QLabel(
            "No embedding data available.\nPress 'Apply Embedding' to generate visualization."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px; background-color: #f0f0f0;")
        self.placeholder_label.setParent(self.view_stack)
        stack_layout.addWidget(self.placeholder_label, 0, 0)

        layout.addWidget(self.view_stack)
        self.show_placeholder()

    def reset_view(self):
        """Reset the view to fit all embedding points."""
        self.fit_view_to_points()

    def show_placeholder(self):
        """Show the placeholder message and hide the graphics view."""
        self.graphics_view.setVisible(False)
        self.placeholder_label.setVisible(True)
        self.home_button.setEnabled(False)

    def show_embedding(self):
        """Show the graphics view and hide the placeholder message."""
        self.graphics_view.setVisible(True)
        self.placeholder_label.setVisible(False)
        self.home_button.setEnabled(True)

    # Delegate graphics view methods
    def setRenderHint(self, hint):
        self.graphics_view.setRenderHint(hint)
    
    def setDragMode(self, mode):
        self.graphics_view.setDragMode(mode)
    
    def setTransformationAnchor(self, anchor):
        self.graphics_view.setTransformationAnchor(anchor)
    
    def setResizeAnchor(self, anchor):
        self.graphics_view.setResizeAnchor(anchor)
    
    def mapToScene(self, point):
        return self.graphics_view.mapToScene(point)
    
    def scale(self, sx, sy):
        self.graphics_view.scale(sx, sy)
    
    def translate(self, dx, dy):
        self.graphics_view.translate(dx, dy)
    
    def fitInView(self, rect, aspect_ratio):
        self.graphics_view.fitInView(rect, aspect_ratio)

    def mousePressEvent(self, event):
        """Handle mouse press for selection (point or rubber band) and panning."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # Check if the click is on an existing point
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                # If so, toggle its selection state and do nothing else
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                item_at_pos.setSelected(not item_at_pos.isSelected())
                return  # Event handled

            # If the click was on the background, proceed with rubber band selection
            self.selection_at_press = set(self.graphics_scene.selectedItems())
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.rubber_band_origin = self.graphics_view.mapToScene(event.pos())
            self.rubber_band = QGraphicsRectItem(QRectF(self.rubber_band_origin, self.rubber_band_origin))
            self.rubber_band.setPen(QPen(QColor(0, 100, 255), 1, Qt.DotLine))
            self.rubber_band.setBrush(QBrush(QColor(0, 100, 255, 50)))
            self.graphics_scene.addItem(self.rubber_band)

        elif event.button() == Qt.RightButton:
            # Handle panning
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mousePressEvent(self.graphics_view, left_event)
        else:
            # Handle standard single-item selection
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            QGraphicsView.mousePressEvent(self.graphics_view, event)

    def mouseMoveEvent(self, event):
        """
        Handles mouse movement, specifically for drawing and updating the selection
        during a rubber-band drag.
        """
        if self.rubber_band is None or event.buttons() != Qt.LeftButton:
            QGraphicsView.mouseMoveEvent(self.graphics_view, event)
            return
    
        # Update the rubber band rectangle
        current_pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self.rubber_band_origin, current_pos).normalized()
        self.rubber_band.setRect(rect)
    
        # Find items within the rubber band
        items_in_band = self.graphics_scene.items(rect)
        embedding_points_in_band = [item for item in items_in_band if isinstance(item, EmbeddingPointItem)]
        
        # Clear current selection
        self.graphics_scene.clearSelection()
        
        # Select items that were originally selected OR are now in the band
        for item in self.selection_at_press:
            item.setSelected(True)
        
        for item in embedding_points_in_band:
            item.setSelected(True)

    def mouseReleaseEvent(self, event):
        """
        Handles mouse release to finalize and clean up the rubber-band selection.
        """
        # Check if a rubber-band operation was in progress
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            # Hide and destroy the visual rubber band
            if self.rubber_band:
                self.rubber_band.hide()
                self.rubber_band = None

            # Reset state variables for the next operation
            self.rubber_band_origin = None
            self.selection_at_press.clear()

            event.accept()
            return

        super().mouseReleaseEvent(event)
            
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and reset the main view."""
        if event.button() == Qt.LeftButton:
            # Clear selection if any items are selected
            if self.graphics_scene.selectedItems():
                self.graphics_scene.clearSelection()  # This triggers on_selection_changed
            
            # Signal the main window to revert from isolation mode
            self.reset_view_requested.emit()
            event.accept()
        else:
            # Pass other double-clicks to the base class
            super().mouseDoubleClickEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)

        old_pos = self.graphics_view.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.graphics_view.scale(zoom_factor, zoom_factor)
        new_pos = self.graphics_view.mapToScene(event.pos())
        
        delta = new_pos - old_pos
        self.graphics_view.translate(delta.x(), delta.y())

    def update_embeddings(self, data_items):
        self.clear_points()

        for item in data_items:
            point = EmbeddingPointItem(0, 0, POINT_SIZE, POINT_SIZE)
            point.setPos(item.embedding_x, item.embedding_y)
            point.setPen(QPen(QColor("black"), POINT_WIDTH))
            point.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            point.setFlag(QGraphicsItem.ItemIsSelectable)
            point.setData(0, item)

            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point

    def clear_points(self):
        self.graphics_scene.clear()
        self.points_by_id.clear()
        self.previous_selection_ids.clear()
        # Add the rubberband back if it exists, though it should be None
        if self.rubber_band:
            self.graphics_scene.addItem(self.rubber_band)

    @pyqtSlot()
    def _on_scene_selection_changed(self):
        """INTERNAL: Handles scene selection and emits a signal to the controller."""
        try:
            selected_items = self.graphics_scene.selectedItems()
            current_selection_ids = {item.data(0).annotation.id for item in selected_items}
        except (RuntimeError, AttributeError):
            return

        if current_selection_ids != self.previous_selection_ids:
            self.previous_selection_ids = current_selection_ids
            self.selection_changed.emit(list(current_selection_ids))
            self._update_animation_state()

    def _update_animation_state(self):
        """Starts or stops the selection animation timer."""
        # Reset all pens first
        for point in self.points_by_id.values():
            point.setPen(QPen(QColor("black"), POINT_WIDTH))

        if self.previous_selection_ids and self.points_by_id:
            if not self.animation_timer.isActive():
                self.animation_timer.start()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()

    def animate_selection(self):
        self.animation_offset = (self.animation_offset + 1) % 20
        for ann_id in self.previous_selection_ids:
            if ann_id in self.points_by_id:
                item = self.points_by_id[ann_id]
                original_color = item.brush().color()
                darker_color = original_color.darker(150)

                animated_pen = QPen(darker_color, POINT_WIDTH)
                animated_pen.setStyle(Qt.CustomDashLine)
                animated_pen.setDashPattern([1, 2])
                animated_pen.setDashOffset(self.animation_offset)
                item.setPen(animated_pen)

    def refresh_display(self, data_items, selected_ids):
        """
        PUBLIC: Re-renders the entire view based on the controller's data.
        This is called after deletions or major state changes.
        """
        # Block signals to prevent feedback loops
        blocker = QSignalBlocker(self.graphics_scene)

        # 1. Update the points in the scene
        self.update_embeddings(data_items)

        # 2. Re-apply the selection
        self.render_selection_from_ids(selected_ids)

        # 3. Update view state
        if not self.points_by_id:
            self.show_placeholder()
        else:
            self.show_embedding()
            # Optionally fit view if needed
            self.fit_view_to_points()

    def render_selection_from_ids(self, selected_ids):
        """PUBLIC: Updates the visual selection of points from a set of IDs."""
        blocker = QSignalBlocker(self.graphics_scene)

        self.graphics_scene.clearSelection()
        for ann_id in selected_ids:
            if ann_id in self.points_by_id:
                self.points_by_id[ann_id].setSelected(True)

        self.previous_selection_ids = set(selected_ids)
        self._update_animation_state()

    def refresh_point_colors(self):
        """PUBLIC: Forces all points to repaint, updating their colors from data_items."""
        for point in self.points_by_id.values():
            point.update()

    def fit_view_to_points(self):
        """Fit the view to show all embedding points."""
        if self.points_by_id:
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            # If no points, reset to default view
            self.graphics_view.fitInView(-2500, -2500, 5000, 5000, Qt.KeepAspectRatio)


class AnnotationViewer(QWidget):
    """
    A compound widget with a toolbar and a scrollable grid for displaying annotation crops.
    It inherits from QWidget to properly manage its child widgets.
    """

    # Signals to report user actions to the ExplorerWindow controller
    selection_changed = pyqtSignal(list)
    deletion_requested = pyqtSignal(list)
    preview_label_applied = pyqtSignal()
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        super(AnnotationViewer, self).__init__(parent)
        self.annotation_widgets_by_id = {}
        self.current_selection_ids = set()
        self.last_selected_id = None
        self.current_widget_size = 96

        self.selection_at_press = set()
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        self.mouse_pressed_on_widget = False

        self.isolated_mode = False
        self.isolated_ids = set()

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI with a toolbar and a scrollable content area."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # --- Toolbar ---
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)

        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setIcon(get_icon("focus.png"))
        self.isolate_button.setToolTip("Hide all non-selected annotations")
        self.isolate_button.clicked.connect(self.isolate_selection)
        toolbar_layout.addWidget(self.isolate_button)

        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setIcon(get_icon("show_all.png"))
        self.show_all_button.setToolTip("Show all filtered annotations")
        self.show_all_button.clicked.connect(self.show_all_annotations)
        toolbar_layout.addWidget(self.show_all_button)

        toolbar_layout.addWidget(self._create_separator())

        sort_label = QLabel("Sort By:")
        toolbar_layout.addWidget(sort_label)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Label", "Image"])
        self.sort_combo.currentTextChanged.connect(self.recalculate_widget_positions)
        toolbar_layout.addWidget(self.sort_combo)
        toolbar_layout.addStretch()

        size_label = QLabel("Size:")
        toolbar_layout.addWidget(size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(256)
        self.size_slider.setValue(96)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(32)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        toolbar_layout.addWidget(self.size_slider)

        self.size_value_label = QLabel("96")
        self.size_value_label.setMinimumWidth(30)
        toolbar_layout.addWidget(self.size_value_label)

        main_layout.addWidget(toolbar_widget)

        # --- Scroll Area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)

        # Install an event filter on the scroll area's viewport to capture mouse events
        self.scroll_area.viewport().installEventFilter(self)

        main_layout.addWidget(self.scroll_area)
        self._update_toolbar_state()
        
    def eventFilter(self, source, event):
        """
        This dispatcher catches events from the scroll area's viewport and
        forwards them to the appropriate mouse event handlers.
        """
        if source is self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True  # Mark as handled
            elif event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)
                return True  # Mark as handled
            elif event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True  # Mark as handled
            elif event.type() == QEvent.MouseButtonDblClick:
                self.mouseDoubleClickEvent(event)
                return True  # Mark as handled

        # For other events, use the default behavior
        return super().eventFilter(source, event)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self.recalculate_widget_positions)
        self._resize_timer.start(50)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and event.modifiers() == Qt.ControlModifier:
            if self.current_selection_ids:
                self.deletion_requested.emit(list(self.current_selection_ids))
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """
        Handles mouse presses on the viewport. This is the entry point for clearing
        selection or initiating a rubber-band drag.
        """
        # --- Determine what was clicked ---
        # Map the click position from viewport coordinates to the scrollable content_widget coordinates
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        content_pos = event.pos() + QPoint(h_bar.value(), v_bar.value())
        child_widget = self.content_widget.childAt(content_pos)

        # Check if the click landed on a selectable AnnotationImageWidget
        is_on_annotation_widget = False
        if child_widget:
            parent = child_widget
            while parent and parent != self.content_widget:
                if isinstance(parent, AnnotationImageWidget):
                    is_on_annotation_widget = True
                    break
                parent = parent.parent()

        if event.button() == Qt.LeftButton:
            # --- Ctrl+Click anywhere starts a rubber band selection ---
            if event.modifiers() == Qt.ControlModifier:
                # Store the selection state at the start of the drag
                self.selection_at_press = self.current_selection_ids.copy()
                # The rubber_band_origin must be in viewport coordinates
                self.rubber_band_origin = event.pos()
                event.accept()
                return

            # --- Simple click on the background clears selection ---
            if not is_on_annotation_widget and not event.modifiers():
                if self.current_selection_ids:
                    self.selection_changed.emit([])
                event.accept()
                return

        # If the event wasn't for clearing or rubber-banding, ignore it.
        # This allows the AnnotationImageWidget's own mousePressEvent to fire,
        # which will then call our `handle_annotation_selection` method.
        event.ignore()
        
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and exit isolation mode."""
        if event.button() == Qt.LeftButton:
            changed_ids = []
            
            # CORRECTED: Check for selected items using the set of IDs.
            if self.current_selection_ids:
                # CORRECTED: Get IDs directly from the set.
                changed_ids = list(self.current_selection_ids)
                self.clear_selection()
                self.selection_changed.emit(changed_ids)

            # If in isolation mode, revert to showing all annotations
            if self.isolated_mode:
                self.show_all_annotations()
            
            # Signal the main window to reset its view
            self.reset_view_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for DYNAMIC rubber band selection."""
        if self.rubber_band_origin is None or \
        event.buttons() != Qt.LeftButton or \
        event.modifiers() != Qt.ControlModifier:
            super().mouseMoveEvent(event)
            return

        # If the mouse was pressed on a widget, let that widget handle the event.
        if self.mouse_pressed_on_widget:
            super().mouseMoveEvent(event)
            return

        # Only start the rubber band after dragging a minimum distance
        distance = (event.pos() - self.rubber_band_origin).manhattanLength()
        if distance < self.drag_threshold:
            return

        # Create and show the rubber band if it doesn't exist
        if not self.rubber_band:
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.scroll_area.viewport())
        
        rect = QRect(self.rubber_band_origin, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()

        # Perform dynamic selection on every move
        selection_rect = self.rubber_band.geometry()
        content_widget = self.content_widget
        changed_ids = []

        # Get scrollbar positions
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        
        for ann_id, widget in self.annotation_widgets_by_id.items():
            # Skip processing widgets that are hidden
            if widget.isHidden():
                continue
                
            # Get widget geometry in content coordinates
            widget_rect_in_content = widget.geometry()
            
            # Map the widget position from content coordinates to viewport coordinates
            widget_pos_in_viewport = QPoint(
                widget_rect_in_content.x() - h_bar.value(),
                widget_rect_in_content.y() - v_bar.value()
            )
            
            # Create the widget rectangle in viewport coordinates
            widget_rect_in_viewport = QRect(
                widget_pos_in_viewport,
                widget_rect_in_content.size()
            )
            
            # Check if any part of the widget is visible in the viewport
            viewport_rect = self.scroll_area.viewport().rect()
            if not viewport_rect.intersects(widget_rect_in_viewport):
                continue  # Skip widgets not visible in viewport
                
            # Check if widget intersects with rubber band
            is_in_band = selection_rect.intersects(widget_rect_in_viewport)
            
            # A widget should be selected if it was selected at the start OR is in the band now
            should_be_selected = (ann_id in self.selection_at_press) or is_in_band

            if should_be_selected and not widget.is_selected():
                if self.select_widget(widget):
                    changed_ids.append(ann_id)
            elif not should_be_selected and widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(ann_id)
        
        if changed_ids:
            self.selection_changed.emit(list(self.current_selection_ids))
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete rubber band selection."""
        # Check if a rubber band drag was in progress
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None

            # Clean up the stored selection state.
            self.selection_at_press = set()
            self.rubber_band_origin = None
            self.mouse_pressed_on_widget = False
            event.accept()
            return
            
        super().mouseReleaseEvent(event)

    @pyqtSlot()
    def isolate_selection(self):
        if not self.current_selection_ids or self.isolated_mode:
            return
        self.isolated_mode = True
        self.isolated_ids = self.current_selection_ids.copy()
        self.recalculate_widget_positions()
        self._update_toolbar_state()

    @pyqtSlot()
    def show_all_annotations(self):
        if not self.isolated_mode:
            return
        self.isolated_mode = False
        self.isolated_ids.clear()
        self.recalculate_widget_positions()
        self._update_toolbar_state()

    def _update_toolbar_state(self):
        selection_exists = bool(self.current_selection_ids)
        if self.isolated_mode:
            self.isolate_button.hide()
            self.show_all_button.show()
        else:
            self.isolate_button.show()
            self.show_all_button.hide()
            self.isolate_button.setEnabled(selection_exists)

    def _create_separator(self):
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; margin: 0 5px;")
        return separator

    def _get_sorted_widgets(self):
        """Gets a sorted list of currently visible widgets."""
        sort_type = self.sort_combo.currentText()

        widgets = [w for w in self.annotation_widgets_by_id.values()if not w.isHidden()]

        if sort_type == "None":
            return widgets

        key_func = None
        if sort_type == "Label":
            def key_func(w):
                return w.data_item.effective_label.short_label_code
        elif sort_type == "Image":
            def key_func(w): return os.path.basename(
                w.data_item.annotation.image_path)

        if key_func:
            widgets.sort(key=key_func)

        return widgets

    def _group_widgets_by_sort_key(self, widgets):
        sort_type = self.sort_combo.currentText()
        if sort_type == "None":
            return [("", widgets)]

        groups = []
        current_group, current_key = [], None
        for widget in widgets:
            key = ""
            if sort_type == "Label":
                key = widget.data_item.effective_label.short_label_code
            elif sort_type == "Image":
                key = os.path.basename(widget.data_item.annotation.image_path)

            if current_key != key:
                if current_group:
                    groups.append((current_key, current_group))
                current_group, current_key = [widget], key
            else:
                current_group.append(widget)
        if current_group:
            groups.append((current_key, current_group))
        return groups

    def _clear_separator_labels(self):
        if hasattr(self, '_group_headers'):
            for header in self._group_headers:
                header.deleteLater()
        self._group_headers = []

    def _create_group_header(self, text):
        if not hasattr(self, '_group_headers'):
            self._group_headers = []
        header = QLabel(text, self.content_widget)
        header.setStyleSheet(
            "font-weight: bold; "
            "font-size: 12px; "
            "color: #555; "
            "background-color: #f0f0f0; "
            "border: 1px solid #ccc; "
            "border-radius: 3px; "
            "padding: 5px 8px; "
            "margin: 2px 0px;")
        header.setFixedHeight(30)
        header.setMinimumWidth(self.scroll_area.viewport().width() - 20)
        self._group_headers.append(header)
        return header

    def on_size_changed(self, value):
        if value % 2 != 0:
            value -= 1
        self.current_widget_size = value
        self.size_value_label.setText(str(value))

        self.content_widget.setUpdatesEnabled(False)
        for widget in self.annotation_widgets_by_id.values():
            widget.update_height(value)
        self.content_widget.setUpdatesEnabled(True)
        self.recalculate_widget_positions()

    def recalculate_widget_positions(self):
        """Manually positions widgets in a flow layout with sorting and group headers."""
        self.content_widget.setUpdatesEnabled(False)
        self._clear_separator_labels()

        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_deleted = widget.data_item.is_marked_for_deletion()
                in_isolated_view = self.isolated_mode and ann_id in self.isolated_ids
                should_be_visible = not is_deleted and (not self.isolated_mode or in_isolated_view)
                widget.setVisible(should_be_visible)

            visible_widgets = self._get_sorted_widgets()
            if not visible_widgets:
                self.content_widget.setMinimumSize(1, 1)
                return

            groups = self._group_widgets_by_sort_key(visible_widgets)
            spacing = max(5, int(self.current_widget_size * 0.08))
            available_width = self.scroll_area.viewport().width()
            x, y = spacing, spacing
            max_height_in_row = 0

            for group_name, group_widgets in groups:
                if group_name and self.sort_combo.currentText() != "None":
                    if x > spacing:
                        x, y = spacing, y + max_height_in_row + spacing
                        max_height_in_row = 0
                    header_label = self._create_group_header(group_name)
                    header_label.move(x, y)
                    header_label.show()
                    y += header_label.height() + spacing
                    x = spacing

                for widget in group_widgets:
                    widget_size = widget.size()
                    if x > spacing and x + widget_size.width() > available_width:
                        x, y = spacing, y + max_height_in_row + spacing
                        max_height_in_row = 0
                    widget.move(x, y)
                    x += widget_size.width() + spacing
                    max_height_in_row = max(
                        max_height_in_row, widget_size.height())

            total_height = y + max_height_in_row + spacing
            self.content_widget.setMinimumSize(available_width, total_height)
        finally:
            self.content_widget.setUpdatesEnabled(True)
            self.content_widget.update()
            
    def clear_selection(self):
        """
        Deselects all currently selected widgets and clears the ID set.
        """
        for ann_id in self.current_selection_ids:
            if ann_id in self.annotation_widgets_by_id:
                self.annotation_widgets_by_id[ann_id].set_selected(False)
        self.current_selection_ids.clear()
        self.last_selected_id = None

    def select_widget(self, widget):
        """
        Selects a single widget, updating its visual state and our ID set.
        Returns True if the selection state changed.
        """
        ann_id = widget.data_item.annotation.id
        if ann_id not in self.current_selection_ids:
            self.current_selection_ids.add(ann_id)
            widget.set_selected(True)
            self.last_selected_id = ann_id
            return True
        return False

    def deselect_widget(self, widget):
        """
        Deselects a single widget, updating its visual state and our ID set.
        Returns True if the selection state changed.
        """
        ann_id = widget.data_item.annotation.id
        if ann_id in self.current_selection_ids:
            self.current_selection_ids.remove(ann_id)
            widget.set_selected(False)
            return True
        return False

    def update_annotations(self, data_items):
        """PUBLIC: Rebuilds the view from a fresh list of data items."""
        self.isolated_mode = False
        self.isolated_ids.clear()

        for widget in self.annotation_widgets_by_id.values():
            widget.setParent(None)
            widget.deleteLater()

        self.annotation_widgets_by_id.clear()
        self.current_selection_ids.clear()
        self.last_selected_id = None

        for data_item in data_items:
            widget = AnnotationImageWidget(data_item, self.current_widget_size, self, self.content_widget)
            self.annotation_widgets_by_id[data_item.annotation.id] = widget

        self.recalculate_widget_positions()
        self._update_toolbar_state()

    def handle_annotation_selection(self, widget, event):
        """
        INTERNAL: Handles a click on an annotation widget and signals the controller with the new selection.
        This rewritten method correctly handles single, control-click, and shift-click selections.
        """
        target_id = widget.data_item.annotation.id
        # Use QApplication.keyboardModifiers() for the most reliable state
        modifiers = QApplication.keyboardModifiers()

        # Get a sorted list of currently visible widget IDs for accurate range selection
        visible_widget_ids = [w.data_item.annotation.id for w in self._get_sorted_widgets()]

        # Start with a copy of the current selection
        new_selection = self.current_selection_ids.copy()

        # --- Shift+Click for Range Selection ---
        if modifiers == Qt.ShiftModifier and self.last_selected_id in visible_widget_ids:
            try:
                start_idx = visible_widget_ids.index(self.last_selected_id)
                end_idx = visible_widget_ids.index(target_id)

                # Ensure the range is always forward
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

                # Add all items within the visual range to the selection
                for i in range(start_idx, end_idx + 1):
                    new_selection.add(visible_widget_ids[i])
            except ValueError:
                # Fallback to single selection if an ID is not found
                new_selection = {target_id}

        # --- Control+Click for Toggling Selection ---
        elif modifiers == Qt.ControlModifier:
            new_selection.symmetric_difference_update({target_id})

        # --- Simple Click for Single Selection ---
        else:
            new_selection = {target_id}

        # Update the anchor point for the next shift-click
        self.last_selected_id = target_id

        # Update internal state BEFORE emitting signal
        if new_selection != self.current_selection_ids:
            # Update the internal state directly
            old_selection = self.current_selection_ids.copy()
            self.current_selection_ids = new_selection
            
            # Update widget visual state
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in self.current_selection_ids
                widget.set_selected(is_selected)
            
            # Then emit the signal
            self.selection_changed.emit(list(new_selection))

    def render_selection_from_ids(self, selected_ids):
        """PUBLIC: Updates the visual selection from a set of IDs."""
        self.current_selection_ids = set(selected_ids)
        self.content_widget.setUpdatesEnabled(False)
        
        # Update all widgets' selection state
        for ann_id, widget in self.annotation_widgets_by_id.items():
            is_selected = ann_id in self.current_selection_ids
            widget.set_selected(is_selected)
            
        self.content_widget.setUpdatesEnabled(True)
        
        # Force a repaint to ensure visual changes are displayed
        self.content_widget.update()

        if self.isolated_mode and not self.current_selection_ids.issubset(self.isolated_ids):
            self.isolated_ids.update(self.current_selection_ids)
            self.recalculate_widget_positions()

        self._update_toolbar_state()

    def apply_preview_label_to_selected(self, preview_label):
        """PUBLIC: Applies a preview label and signals the controller."""
        if not self.current_selection_ids or not preview_label:
            return

        for ann_id in self.current_selection_ids:
            if ann_id in self.annotation_widgets_by_id:
                widget = self.annotation_widgets_by_id[ann_id]
                widget.data_item.set_preview_label(preview_label)
                widget.update()

        if self.sort_combo.currentText() == "Label":
            self.recalculate_widget_positions()

        self.preview_label_applied.emit()

    def refresh_display(self, data_items, selected_ids):
        """PUBLIC: Re-renders the entire view based on controller's data."""
        self.update_annotations(data_items)
        self.render_selection_from_ids(selected_ids)

    def refresh_widget_colors(self):
        """PUBLIC: Forces all visible widgets to repaint to show new colors."""
        self.content_widget.setUpdatesEnabled(False)
        for widget in self.annotation_widgets_by_id.values():
            if not widget.isHidden():
                widget.update()
        self.content_widget.setUpdatesEnabled(True)


# ----------------------------------------------------------------------------------------------------------------------
# ExplorerWindow
# ----------------------------------------------------------------------------------------------------------------------


class ExplorerWindow(QMainWindow):
    def __init__(self, main_window, label_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        # Use passed widget or fallback
        self.label_window = label_window or main_window.label_window

        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        # --- Initialize Data Members and State ---
        self.device = main_window.device
        self.loaded_model = None
        self.model_path = ""
        self.data_items_by_id = {}
        self.current_selection_ids = set()
        self.current_features = None
        self.current_feature_generating_model = ""
        
        self.is_handling_selection = False

        self.setWindowTitle("Explorer")
        self.setWindowIcon(get_icon("magic.png"))

        # --- Set up the base UI structure ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- BUILD THE UI AND CONNECTIONS ONCE ---
        # These methods will create all child widgets and set up signals.
        self.setup_ui_widgets()
        self.setup_connections()

    def setup_ui_widgets(self):
        """
        Creates all child widgets and adds them to the layout.
        This method should only ever be called ONCE from the constructor.
        """
        # --- Create all child widgets here ---
        self.annotation_settings_widget = AnnotationSettingsWidget(self.main_window, self)
        self.model_settings_widget = ModelSettingsWidget(self.main_window, self)
        self.embedding_settings_widget = EmbeddingSettingsWidget(self.main_window, self)
        self.annotation_viewer = AnnotationViewer(self)
        self.embedding_viewer = EmbeddingViewer(self)
        self.clear_preview_button = QPushButton('Clear Preview', self)
        self.exit_button = QPushButton('Exit', self)
        self.apply_button = QPushButton('Apply', self)

        # --- Assemble the layout ---
        # Top section: Settings
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.annotation_settings_widget, 2)
        top_layout.addWidget(self.model_settings_widget, 1)
        top_layout.addWidget(self.embedding_settings_widget, 1)
        self.main_layout.addLayout(top_layout)

        # Middle section: Viewers
        middle_splitter = QSplitter(Qt.Horizontal)
        annotation_group = QGroupBox("Annotation Viewer")
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.addWidget(self.annotation_viewer)
        middle_splitter.addWidget(annotation_group)

        embedding_group = QGroupBox("Embedding Viewer")
        embedding_layout = QVBoxLayout(embedding_group)
        embedding_layout.addWidget(self.embedding_viewer)
        middle_splitter.addWidget(embedding_group)
        middle_splitter.setSizes([700, 300])
        self.main_layout.addWidget(middle_splitter, 1)

        # The re-parented Label Window from MainWindow
        self.main_layout.addWidget(self.label_window)

        # Bottom section: Control Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.clear_preview_button)
        buttons_layout.addWidget(self.exit_button)
        buttons_layout.addWidget(self.apply_button)
        self.main_layout.addLayout(buttons_layout)

        # --- Set initial UI state ---
        self.annotation_settings_widget.set_default_to_current_image()

    def showEvent(self, event):
        """
        This method now only handles logic that must run *every time* the
        window becomes visible. It no longer builds or destroys the UI.
        """
        super(ExplorerWindow, self).showEvent(event)
        # Refreshing filters ensures the view is up-to-date with the project state.
        self.refresh_filters()

    def closeEvent(self, event):
        self._cleanup_resources()
        if self.main_window:
            self.main_window.setEnabled(True)
            if hasattr(self.main_window, 'explorer_closed'):
                self.main_window.explorer_closed()
                
        # This check is important for a clean shutdown
        if self.main_window:
            self.main_window.explorer_window = None
        event.accept()

    def setup_connections(self):
        """Central place to connect all signals and slots for the Controller pattern."""
        # --- Controller actions from buttons ---
        self.clear_preview_button.clicked.connect(self.clear_all_previews)
        self.exit_button.clicked.connect(self.close)
        self.apply_button.clicked.connect(self.apply_all_changes)

        # --- Viewers -> Controller signals ---
        self.annotation_viewer.selection_changed.connect(self.handle_selection_change)
        self.annotation_viewer.deletion_requested.connect(self.handle_deletion_request)
        self.annotation_viewer.preview_label_applied.connect(self.handle_preview_update)
        self.annotation_viewer.reset_view_requested.connect(self.handle_view_reset)

        self.embedding_viewer.selection_changed.connect(self.handle_selection_change)
        self.embedding_viewer.reset_view_requested.connect(self.handle_view_reset)

        # --- Other UI elements -> Controller signals ---
        self.label_window.labelSelected.connect(self.handle_label_selection_for_preview)

    # --- SLOTS: Handling signals from viewers and UI ---

    @pyqtSlot(list)
    def handle_selection_change(self, selected_ids):
        """
        Handles selection changes from ANY viewer, acting as the single point of truth.
        This version includes a re-entrancy guard to prevent instability.
        """
        # Re-entrancy Guard: If the function is already running, ignore this new signal.
        if self.is_handling_selection:
            return

        self.is_handling_selection = True
        try:
            new_selection = set(selected_ids)
            if new_selection == self.current_selection_ids:
                return  # No change

            self.current_selection_ids = new_selection

            # Update the central data model's selection state
            for item in self.data_items_by_id.values():
                item.set_selected(item.annotation.id in self.current_selection_ids)

            # Get the widget that sent the signal
            sender_widget = self.sender()

            # Command viewers to update their visual selection,
            # but DON'T update the viewer that initiated the change.
            if sender_widget is not self.annotation_viewer:
                self.annotation_viewer.render_selection_from_ids(self.current_selection_ids)
                
                # If selection came from embedding viewer and we have a non-empty selection,
                # activate isolation mode in the annotation viewer if not already active
                if (sender_widget is self.embedding_viewer and
                    new_selection and not self.annotation_viewer.isolated_mode):
                    self.annotation_viewer.isolate_selection()

            if sender_widget is not self.embedding_viewer:
                self.embedding_viewer.render_selection_from_ids(self.current_selection_ids)

            # Update other UI elements that depend on selection
            self._update_label_window_for_selection()

        finally:
            # Release the lock so the function can be called again.
            self.is_handling_selection = False

    @pyqtSlot(list)
    def handle_deletion_request(self, ids_to_delete):
        """Marks items for deletion and refreshes all views."""
        for ann_id in ids_to_delete:
            # --- ADD THIS CHECK ---
            if ann_id in self.data_items_by_id:
                self.data_items_by_id[ann_id].mark_for_deletion()

        # Since items were deleted, the selection is now invalid. Clear it.
        self.handle_selection_change([])

        # Refresh displays to hide the marked items
        self._refresh_all_views()
        self.update_button_states()

    @pyqtSlot()
    def handle_preview_update(self):
        """Handles updates after a preview label is applied to the selection."""
        # Command viewers to update visuals that depend on data item state
        self.embedding_viewer.refresh_point_colors()
        self.annotation_viewer.refresh_widget_colors()  # Use this instead of full repaint

        # Update other UI
        self._update_label_window_for_selection()
        self.update_button_states()

    @pyqtSlot(object)  # Can be Label or LabelScheme
    def handle_label_selection_for_preview(self, label):
        """Applies a selected label from the LabelWindow as a preview to the current selection."""
        if not self.current_selection_ids:
            return

        for ann_id in self.current_selection_ids:
            # --- ADD THIS CHECK ---
            if ann_id in self.data_items_by_id:
                self.data_items_by_id[ann_id].set_preview_label(label)

        # After changing the data, signal that a preview has updated
        self.handle_preview_update()

    @pyqtSlot()
    def handle_view_reset(self):
        """Resets selection and isolated view, requested from any viewer's double-click."""
        self.annotation_viewer.show_all_annotations()
        self.handle_selection_change([])

    # --- PUBLIC METHODS: High-level user actions ---

    def refresh_filters(self):
        """Filters annotations and completely refreshes all views and data."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 1. Get filtered annotations and create the central data items
            filtered_annotations = self._get_filtered_annotations()
            self._ensure_cropped_images(filtered_annotations)

            self.data_items_by_id = {ann.id: AnnotationDataItem(ann) for ann in filtered_annotations}

            # 2. Reset runtime state
            self.current_selection_ids.clear()
            self.current_features = None  # Invalidate feature cache

            # 3. Command all visual components to refresh
            self._refresh_all_views()
            self.embedding_viewer.show_placeholder()  # Embedding is now out of date
            self.update_button_states()

        finally:
            QApplication.restoreOverrideCursor()

    def run_embedding_pipeline(self):
        """Orchestrates the feature extraction and dimensionality reduction pipeline."""
        if not self.data_items_by_id:
            return

        embedding_params = self.embedding_settings_widget.get_embedding_parameters()
        selected_model = self.model_settings_widget.get_selected_model()

        progress_bar = ProgressBar(self, "Generating Embedding")
        progress_bar.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Step 1: Feature Extraction (if needed)
            if self.current_features is None or selected_model != self.current_feature_generating_model:
                visible_data_items = []
                for item in self.data_items_by_id.values():
                    if not item.is_marked_for_deletion():
                        visible_data_items.append(item)

                features, valid_items = self._extract_features(visible_data_items, progress_bar)

                if len(valid_items) != len(visible_data_items):
                    valid_ids = {item.annotation.id for item in valid_items}
                    # Remove items not in valid_ids unless marked for deletion
                    new_data_items_by_id = {}
                    for id, item in self.data_items_by_id.items():
                        if id in valid_ids or item.is_marked_for_deletion():
                            new_data_items_by_id[id] = item

                    self.data_items_by_id = new_data_items_by_id
                    self._refresh_all_views()

                self.current_features = features
                self.current_feature_generating_model = selected_model
            else:
                features = self.current_features

            if features is None or len(features) == 0:
                return

            # Step 2: Dimensionality Reduction
            progress_bar.set_busy_mode(f"Running {embedding_params['technique']}...")
            embedded_features = self._run_dimensionality_reduction(features, embedding_params)
            if embedded_features is None:
                return

            # Step 3: Update and Visualize
            progress_bar.set_busy_mode("Updating visualization...")
            visible_data_items = []
            for item in self.data_items_by_id.values():
                if not item.is_marked_for_deletion():
                    visible_data_items.append(item)

            self._update_data_items_with_embedding(visible_data_items, embedded_features)

            self.embedding_viewer.refresh_display(visible_data_items, self.current_selection_ids)
            self.embedding_viewer.fit_view_to_points()

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.close()

    def clear_all_previews(self):
        """Clears all preview labels and un-marks all deletions."""
        for item in self.data_items_by_id.values():
            item.clear_preview_label()
            item.unmark_for_deletion()

        self._refresh_all_views()
        self.update_button_states()
        self._update_label_window_for_selection()

    def apply_all_changes(self):
        """Applies all preview changes and deletions permanently."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            applied_annotations = []
            annotations_to_delete = []

            for item in self.data_items_by_id.values():
                if item.apply_preview_permanently():
                    applied_annotations.append(item.annotation)
                if item.is_marked_for_deletion():
                    annotations_to_delete.append(item.annotation)

            if applied_annotations:
                affected_images = {ann.image_path for ann in applied_annotations}
                for image_path in affected_images:
                    self.image_window.update_image_annotations(image_path)
                print(f"Applied label changes to {len(applied_annotations)} annotation(s)")

            if annotations_to_delete:
                self.annotation_window.delete_annotations(annotations_to_delete)
                print(f"Deleted {len(annotations_to_delete)} annotation(s)")

            if applied_annotations or annotations_to_delete:
                self.annotation_window.load_annotations()
                self.refresh_filters()

            self.update_button_states()
        finally:
            QApplication.restoreOverrideCursor()

    # --- HELPER / PRIVATE METHODS ---

    def _refresh_all_views(self):
        """Commands all viewers to refresh based on the current state of the data model."""
        visible_items = [item for item in self.data_items_by_id.values() if not item.is_marked_for_deletion()]

        self.annotation_viewer.refresh_display(list(self.data_items_by_id.values()), self.current_selection_ids)
        self.embedding_viewer.refresh_display(visible_items, self.current_selection_ids)

    def _update_label_window_for_selection(self):
        """Updates the LabelWindow based on the current selection's labels."""
        # Temporarily block signals from the label_window to prevent a feedback loop.
        blocker = QSignalBlocker(self.label_window)
        
        try:
            if not self.current_selection_ids:
                self.label_window.deselect_active_label()
                self.label_window.update_annotation_count()
                return

            selected_items = []
            for ann_id in self.current_selection_ids:
                if ann_id in self.data_items_by_id:
                    selected_items.append(self.data_items_by_id[ann_id])

            if not selected_items:
                self.label_window.deselect_active_label()
                self.label_window.update_annotation_count()
                return

            first_label = selected_items[0].effective_label
            all_same = all(item.effective_label.id == first_label.id for item in selected_items)

            if all_same:
                self.label_window.set_active_label(first_label)
            else:
                self.label_window.deselect_active_label()

            self.label_window.update_annotation_count()

        finally:
            # The blocker is automatically released here, or if an error occurs.
            pass

    def update_button_states(self):
        """Enables/disables the 'Apply' and 'Clear' buttons based on pending changes."""
        has_preview_changes = any(item.has_preview_changes() for item in self.data_items_by_id.values())
        has_deletions = any(item.is_marked_for_deletion() for item in self.data_items_by_id.values())
        has_changes = has_preview_changes or has_deletions
        self.clear_preview_button.setEnabled(has_changes)
        self.apply_button.setEnabled(has_changes)

    def _get_filtered_annotations(self):
        """Gets annotations from AnnotationWindow based on filter widget settings."""
        if not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return []

        selected_images = self.annotation_settings_widget.get_selected_images()
        selected_types = self.annotation_settings_widget.get_selected_annotation_types()
        selected_labels = self.annotation_settings_widget.get_selected_labels()

        if not all([selected_images, selected_types, selected_labels]):
            return []

        annotations_to_process = []
        for ann in self.main_window.annotation_window.annotations_dict.values():
            if (os.path.basename(ann.image_path) in selected_images and
                type(ann).__name__ in selected_types and
                    ann.label.short_label_code in selected_labels):
                annotations_to_process.append(ann)

        return annotations_to_process

    def _ensure_cropped_images(self, annotations):
        """Ensure all provided annotations have a cropped image available."""
        annotations_by_image = {}
        for annotation in annotations:
            if not annotation.cropped_image:
                image_path = annotation.image_path
                if image_path not in annotations_by_image:
                    annotations_by_image[image_path] = []
                annotations_by_image[image_path].append(annotation)

        if annotations_by_image:
            progress_bar = ProgressBar(self, "Cropping Image Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(annotations_by_image))

            try:
                for image_path, image_annotations in annotations_by_image.items():
                    self.annotation_window.crop_annotations(
                        image_path=image_path,
                        annotations=image_annotations,
                        return_annotations=False,
                        verbose=False
                    )
                    progress_bar.update_progress()
            finally:
                progress_bar.finish_progress()
                progress_bar.close()

    def _extract_features(self, data_items, progress_bar=None):
        """Dispatcher method to call the appropriate feature extraction function."""
        model_name = self.model_settings_widget.get_selected_model()
        if not model_name:
            return np.array([]), []

        if model_name == "Color Features":
            return self._extract_color_features(data_items, progress_bar=progress_bar)
        elif ".pt" in model_name:
            return self._extract_yolo_features(data_items, model_name, progress_bar=progress_bar)
        else:
            return np.array([]), []

    def _extract_color_features(self, data_items, progress_bar=None, bins=32):
        if progress_bar:
            progress_bar.set_title("Extracting Color Features...")
            progress_bar.start_progress(len(data_items))

        features, valid_data_items = [], []
        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                # Convert QPixmap to numpy array (H, W, 3)
                arr = pixmap_to_numpy(pixmap)
                # Flatten image to (N, 3) for RGB pixel stats
                pixels = arr.reshape(-1, 3)
                # Mean and std for each channel
                mean_color, std_color = np.mean(pixels, axis=0), np.std(pixels, axis=0)
                # Small value to avoid division by zero
                epsilon = 1e-8
                # Center pixels for skew/kurtosis
                centered = pixels - mean_color
                # Skewness for each channel
                skew = np.mean(centered**3, axis=0) / (std_color**3 + epsilon)
                # Kurtosis for each channel
                kurt = np.mean(centered**4, axis=0) / (std_color**4 + epsilon) - 3
                # Histograms for R,G,B
                hists = [np.histogram(pixels[:, i], bins=bins, range=(0, 255))[0] for i in range(3)]
                # Normalize histograms
                hists_norm = [h / (h.sum() + epsilon) for h in hists]
                # Convert to grayscale
                gray_arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                # Mean, std, range of gray
                gray_stats = np.array([np.mean(gray_arr), np.std(gray_arr), np.ptp(gray_arr)])
                # Area feature (if available), Perimeter feature (if available)
                geom_feats = np.array([getattr(item.annotation, 'area', 0.0),
                                       getattr(item.annotation, 'perimeter', 0.0)])

                current_features = np.concatenate([mean_color,
                                                   std_color,
                                                   skew,
                                                   kurt,
                                                   *hists_norm,
                                                   gray_stats,
                                                   geom_feats])
                features.append(current_features)
                valid_data_items.append(item)

            if progress_bar:
                progress_bar.update_progress()
        return np.array(features), valid_data_items

    def _extract_yolo_features(self, data_items, model_name, progress_bar=None):
        if model_name != self.model_path or self.loaded_model is None:
            try:
                self.loaded_model = YOLO(model_name)
                self.model_path = model_name
                self.imgsz = self.loaded_model.model.args.get('imgsz', 128)
                if self.imgsz > 224:
                    self.imgsz = 128
                dummy_image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                self.loaded_model.embed(dummy_image, imgsz=self.imgsz, half=True, device=self.device, verbose=False)
            except Exception as e:
                print(f"ERROR: Could not load YOLO model '{model_name}': {e}")
                return np.array([]), []

        if progress_bar:
            progress_bar.set_title("Preparing images...")
            progress_bar.start_progress(len(data_items))

        image_list, valid_data_items = [], []
        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                image_list.append(pixmap_to_numpy(pixmap))
                valid_data_items.append(item)
            if progress_bar:
                progress_bar.update_progress()
        if not valid_data_items:
            return np.array([]), []

        embeddings_list = []
        try:
            if progress_bar:
                progress_bar.set_busy_mode(f"Extracting features...")
            results_gen = self.loaded_model.embed(image_list,
                                                  stream=True,
                                                  imgsz=self.imgsz,
                                                  half=True,
                                                  device=self.device,
                                                  verbose=False)
            if progress_bar:
                progress_bar.start_progress(len(valid_data_items))

            for result in results_gen:
                embeddings_list.append(result.cpu().numpy())
                if progress_bar:
                    progress_bar.update_progress()

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.array(embeddings_list), valid_data_items

    def _run_dimensionality_reduction(self, features, params):
        technique = params.get('technique', 'UMAP')
        if len(features) <= 2:
            return None

        try:
            features_scaled = StandardScaler().fit_transform(features)

            if technique == "UMAP":
                reducer = UMAP(n_components=2,
                               random_state=42,
                               n_neighbors=min(params.get('n_neighbors', 15), len(features_scaled) - 1),
                               min_dist=params.get('min_dist', 0.1), metric='cosine')

            elif technique == "TSNE":
                reducer = TSNE(n_components=2,
                               random_state=42,
                               perplexity=min(params.get('perplexity', 30), len(features_scaled) - 1),
                               early_exaggeration=params.get('early_exaggeration', 12.0),
                               learning_rate='auto', init='pca')

            elif technique == "PCA":
                reducer = PCA(n_components=2, random_state=42)

            else:
                return None

            return reducer.fit_transform(features_scaled)

        except Exception as e:
            print(f"Error during {technique} dimensionality reduction: {e}")
            return None

    def _update_data_items_with_embedding(self, data_items, embedded_features):
        scale_factor = 4000
        min_vals = np.min(embedded_features, axis=0)
        max_vals = np.max(embedded_features, axis=0)
        # Add epsilon to avoid division by zero
        range_vals = max_vals - min_vals + 1e-8

        for i, item in enumerate(data_items):
            norm_x = (embedded_features[i, 0] - min_vals[0]) / range_vals[0]
            norm_y = (embedded_features[i, 1] - min_vals[1]) / range_vals[1]
            item.embedding_x = (norm_x * scale_factor) - (scale_factor / 2)
            item.embedding_y = (norm_y * scale_factor) - (scale_factor / 2)

    def _cleanup_resources(self):
        """Frees up heavy resources when the window is closed."""
        print("Cleaning up Explorer resources...")
        self.embedding_viewer.animation_timer.stop()
        self.loaded_model = None
        self.model_path = ""
        self.current_features = None
        self.current_feature_generating_model = ""

        if torch.cuda.is_available():
            print("Clearing CUDA cache.")
            torch.cuda.empty_cache()

        print("Cleanup complete.")
