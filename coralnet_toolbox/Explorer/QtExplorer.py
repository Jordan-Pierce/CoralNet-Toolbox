import os
import numpy as np
import torch
import warnings

from ultralytics import YOLO

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import pixmap_to_numpy

from PyQt5.QtGui import QIcon, QPen, QColor, QPainter, QBrush, QPainterPath, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QPointF, pyqtSignal, QSignalBlocker, pyqtSlot

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget, QGridLayout,
                             QMainWindow, QSplitter, QGroupBox, QFormLayout,
                             QSpinBox, QGraphicsEllipseItem, QGraphicsItem, QSlider,
                             QListWidget, QDoubleSpinBox, QApplication, QStyle,
                             QGraphicsRectItem, QRubberBand, QStyleOptionGraphicsItem,
                             QTabWidget, QLineEdit, QFileDialog)

from coralnet_toolbox.Explorer.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.QtDataItem import EmbeddingPointItem
from coralnet_toolbox.Explorer.QtDataItem import AnnotationImageWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import ModelSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import EmbeddingSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import AnnotationSettingsWidget

from coralnet_toolbox.QtProgressBar import ProgressBar

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

POINT_WIDTH = 3

# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingViewer(QWidget):
    """Custom QGraphicsView for interactive embedding visualization with zooming, panning, and selection."""
    selection_changed = pyqtSignal(list)
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the EmbeddingViewer widget."""
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)

        super(EmbeddingViewer, self).__init__(parent)
        self.explorer_window = parent

        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMinimumHeight(200)

        self.rubber_band = None
        self.rubber_band_origin = QPointF()
        self.selection_at_press = None
        self.points_by_id = {}
        self.previous_selection_ids = set()

        self.animation_offset = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)

        self.graphics_scene.selectionChanged.connect(self.on_selection_changed)
        self.setup_ui()
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
        self.home_button.clicked.connect(self.reset_view)
        header_layout.addWidget(self.home_button)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        layout.addWidget(self.graphics_view)
        self.placeholder_label = QLabel(
            "No embedding data available.\nPress 'Apply Embedding' to generate visualization."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px;")
        
        layout.addWidget(self.placeholder_label)
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
        """Set render hint for the graphics view."""
        self.graphics_view.setRenderHint(hint)
        
    def setDragMode(self, mode): 
        """Set drag mode for the graphics view."""
        self.graphics_view.setDragMode(mode)
        
    def setTransformationAnchor(self, anchor): 
        """Set transformation anchor for the graphics view."""
        self.graphics_view.setTransformationAnchor(anchor)
        
    def setResizeAnchor(self, anchor): 
        """Set resize anchor for the graphics view."""
        self.graphics_view.setResizeAnchor(anchor)
        
    def mapToScene(self, point): 
        """Map a point to the scene coordinates."""
        return self.graphics_view.mapToScene(point)
    
    def scale(self, sx, sy): 
        """Scale the graphics view."""
        self.graphics_view.scale(sx, sy)
        
    def translate(self, dx, dy): 
        """Translate the graphics view."""
        self.graphics_view.translate(dx, dy)
        
    def fitInView(self, rect, aspect_ratio): 
        """Fit the view to a rectangle with aspect ratio."""
        self.graphics_view.fitInView(rect, aspect_ratio)
        
    def keyPressEvent(self, event):
        """Handles key presses for deleting selected points."""
        # Check if the pressed key is Delete/Backspace AND the Control key is held down
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and event.modifiers() == Qt.ControlModifier:
            # Get the currently selected items from the graphics scene
            selected_items = self.graphics_scene.selectedItems()

            if not selected_items:
                super().keyPressEvent(event)
                return

            print(f"Marking {len(selected_items)} points for deletion.")

            # Mark each item for deletion and remove it from the scene
            for item in selected_items:
                if isinstance(item, EmbeddingPointItem):
                    # Mark the central data item for deletion
                    item.data_item.mark_for_deletion()

                    # Remove the point from our internal lookup
                    ann_id = item.data_item.annotation.id
                    if ann_id in self.points_by_id:
                        del self.points_by_id[ann_id]

                    # Remove the point from the visual scene
                    self.graphics_scene.removeItem(item)

            # Trigger a selection change to clear the selection state
            # and notify the ExplorerWindow.
            self.on_selection_changed()

            # Accept the event to prevent it from being processed further
            event.accept()
        else:
            # Pass any other key presses to the default handler
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for selection (point or rubber band) and panning."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                # The viewer (controller) directly changes the state on the data item.
                is_currently_selected = item_at_pos.data_item.is_selected
                item_at_pos.data_item.set_selected(not is_currently_selected)
                item_at_pos.setSelected(not is_currently_selected)  # Keep scene selection in sync
                self.on_selection_changed()  # Manually trigger update
                return

            self.selection_at_press = set(self.graphics_scene.selectedItems())
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.rubber_band_origin = self.graphics_view.mapToScene(event.pos())
            self.rubber_band = QGraphicsRectItem(QRectF(self.rubber_band_origin, self.rubber_band_origin))
            self.rubber_band.setPen(QPen(QColor(0, 100, 255), 1, Qt.DotLine))
            self.rubber_band.setBrush(QBrush(QColor(0, 100, 255, 50)))
            self.graphics_scene.addItem(self.rubber_band)

        elif event.button() == Qt.RightButton:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mousePressEvent(self.graphics_view, left_event)
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            QGraphicsView.mousePressEvent(self.graphics_view, event)
            
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and reset the main view."""
        if event.button() == Qt.LeftButton:
            if self.graphics_scene.selectedItems():
                self.graphics_scene.clearSelection()
            self.reset_view_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dynamic selection and panning."""
        if self.rubber_band:
            # Update the rubber band rectangle as the mouse moves
            current_pos = self.graphics_view.mapToScene(event.pos())
            self.rubber_band.setRect(QRectF(self.rubber_band_origin, current_pos).normalized())
            # Create a selection path from the rubber band rectangle
            path = QPainterPath()
            path.addRect(self.rubber_band.rect())
            # Block signals to avoid recursive selectionChanged events
            self.graphics_scene.blockSignals(True)
            self.graphics_scene.setSelectionArea(path)
            # Restore selection for items that were already selected at press
            if self.selection_at_press:
                for item in self.selection_at_press:
                    item.setSelected(True)
            self.graphics_scene.blockSignals(False)
            # Manually trigger selection changed logic
            self.on_selection_changed()
        elif event.buttons() == Qt.RightButton:
            # Forward right-drag as left-drag for panning
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mouseMoveEvent(self.graphics_view, left_event)
        else:
            # Default mouse move handling
            QGraphicsView.mouseMoveEvent(self.graphics_view, event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize the action and clean up."""
        if self.rubber_band:
            self.graphics_scene.removeItem(self.rubber_band)
            self.rubber_band = None
            self.selection_at_press = None
        elif event.button() == Qt.RightButton:
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mouseReleaseEvent(self.graphics_view, left_event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        else:
            QGraphicsView.mouseReleaseEvent(self.graphics_view, event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Set anchor points so zoom occurs at mouse position
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)

        # Get the scene position before zooming
        old_pos = self.graphics_view.mapToScene(event.pos())

        # Determine zoom direction
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        # Apply zoom
        self.graphics_view.scale(zoom_factor, zoom_factor)

        # Get the scene position after zooming
        new_pos = self.graphics_view.mapToScene(event.pos())

        # Translate view to keep mouse position stable
        delta = new_pos - old_pos
        self.graphics_view.translate(delta.x(), delta.y())

    def update_embeddings(self, data_items):
        """Update the embedding visualization. Creates an EmbeddingPointItem for 
        each AnnotationDataItem and links them."""
        self.clear_points()
        for item in data_items:
            # Create the point item directly from the data_item.
            # The item's constructor now handles setting position, flags, etc.
            point = EmbeddingPointItem(item)
            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point
            
    def refresh_points(self):
        """Refreshes the points in the view to match the current state of the master data list."""
        if not self.explorer_window or not self.explorer_window.current_data_items:
            return

        # Get the set of IDs for points currently in the scene
        current_point_ids = set(self.points_by_id.keys())
        
        # Get the master list of data items from the parent window
        all_data_items = self.explorer_window.current_data_items

        something_changed = False
        for item in all_data_items:
            # If a data item is NOT marked for deletion but is also NOT in the scene, add it back.
            if not item.is_marked_for_deletion() and item.annotation.id not in current_point_ids:
                point = EmbeddingPointItem(item)
                self.graphics_scene.addItem(point)
                self.points_by_id[item.annotation.id] = point
                something_changed = True
        
        if something_changed:
            print("Refreshed embedding points to show reverted items.")
            
    def clear_points(self):
        """Clear all embedding points from the scene."""
        for point in self.points_by_id.values():
            self.graphics_scene.removeItem(point)
        self.points_by_id.clear()

    def on_selection_changed(self):
        """
        Handles selection changes in the scene. Updates the central data model
        and emits a signal to notify other parts of the application.
        """
        if not self.graphics_scene: 
            return
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            return
            
        current_selection_ids = {item.data_item.annotation.id for item in selected_items}

        if current_selection_ids != self.previous_selection_ids:
            # Update the central model (AnnotationDataItem) for all points
            for point_id, point in self.points_by_id.items():
                is_selected = point_id in current_selection_ids
                # The data_item is the single source of truth
                point.data_item.set_selected(is_selected)

            self.selection_changed.emit(list(current_selection_ids))
            self.previous_selection_ids = current_selection_ids

        # Handle animation
        if hasattr(self, 'animation_timer') and self.animation_timer: 
            self.animation_timer.stop()
            
        for point in self.points_by_id.values():
            if not point.isSelected():
                point.setPen(QPen(QColor("black"), POINT_WIDTH))
        if selected_items and hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.start()

    def animate_selection(self):
        """Animate selected points with a marching ants effect."""
        if not self.graphics_scene: 
            return
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            return
            
        self.animation_offset = (self.animation_offset + 1) % 20
        for item in selected_items:
            # Get the color directly from the source of truth
            original_color = item.data_item.effective_color
            darker_color = original_color.darker(150)
            animated_pen = QPen(darker_color, POINT_WIDTH)
            animated_pen.setStyle(Qt.CustomDashLine)
            animated_pen.setDashPattern([1, 2])
            animated_pen.setDashOffset(self.animation_offset)
            item.setPen(animated_pen)
            
    def render_selection_from_ids(self, selected_ids):
        """
        Updates the visual selection of points based on a set of annotation IDs
        provided by an external controller.
        """
        blocker = QSignalBlocker(self.graphics_scene)
        
        for ann_id, point in self.points_by_id.items():
            is_selected = ann_id in selected_ids
            # 1. Update the state on the central data item
            point.data_item.set_selected(is_selected)
            # 2. Update the selection state of the graphics item itself
            point.setSelected(is_selected)
            
        blocker.unblock()
        
        # Manually trigger on_selection_changed to update animation and emit signals
        self.on_selection_changed()

    def fit_view_to_points(self):
        """Fit the view to show all embedding points."""
        if self.points_by_id:
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.graphics_view.fitInView(-2500, -2500, 5000, 5000, Qt.KeepAspectRatio)
            

class AnnotationViewer(QScrollArea):
    """Scrollable grid widget for displaying annotation image crops with selection, 
    filtering, and isolation support. Acts as a controller for the widgets."""
    selection_changed = pyqtSignal(list)
    preview_changed = pyqtSignal(list)
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the AnnotationViewer widget."""
        super(AnnotationViewer, self).__init__(parent)
        self.explorer_window = parent
        
        self.annotation_widgets_by_id = {}
        self.selected_widgets = []
        self.last_selected_index = -1
        self.current_widget_size = 96
        self.selection_at_press = set()
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        self.mouse_pressed_on_widget = False
        self.preview_label_assignments = {}
        self.original_label_assignments = {}
        self.isolated_mode = False
        self.isolated_widgets = set()
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI with a toolbar and a scrollable content area."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)

        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setToolTip("Hide all non-selected annotations")
        self.isolate_button.clicked.connect(self.isolate_selection)
        toolbar_layout.addWidget(self.isolate_button)

        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setToolTip("Show all filtered annotations")
        self.show_all_button.clicked.connect(self.show_all_annotations)
        toolbar_layout.addWidget(self.show_all_button)

        toolbar_layout.addWidget(self._create_separator())

        sort_label = QLabel("Sort By:")
        toolbar_layout.addWidget(sort_label)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Label", "Image"])
        self.sort_combo.currentTextChanged.connect(self.on_sort_changed)
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

        self.content_widget = QWidget()
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setWidget(self.content_widget)

        main_layout.addWidget(content_scroll)
        self.setWidget(main_container)
        self._update_toolbar_state()

    @pyqtSlot()
    def isolate_selection(self):
        """Hides all annotation widgets that are not currently selected."""
        if not self.selected_widgets or self.isolated_mode:
            return

        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                if widget not in self.isolated_widgets:
                    widget.hide()
            self.isolated_mode = True
            self.recalculate_widget_positions()
        finally:
            self.content_widget.setUpdatesEnabled(True)

        self._update_toolbar_state()
        self.explorer_window.main_window.label_window.update_annotation_count()

    @pyqtSlot()
    def show_all_annotations(self):
        """Shows all annotation widgets, exiting the isolated mode."""
        if not self.isolated_mode:
            return
            
        self.isolated_mode = False
        self.isolated_widgets.clear()
        
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                widget.show()
            self.recalculate_widget_positions()
        finally:
            self.content_widget.setUpdatesEnabled(True)
            
        self._update_toolbar_state()
        self.explorer_window.main_window.label_window.update_annotation_count()

    def _update_toolbar_state(self):
        """Updates the toolbar buttons based on selection and isolation mode."""
        selection_exists = bool(self.selected_widgets)
        if self.isolated_mode:
            self.isolate_button.hide()
            self.show_all_button.show()
            self.show_all_button.setEnabled(True)
        else:
            self.isolate_button.show()
            self.show_all_button.hide()
            self.isolate_button.setEnabled(selection_exists)

    def _create_separator(self):
        """Creates a vertical separator for the toolbar."""
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; margin: 0 5px;")
        return separator

    def on_sort_changed(self, sort_type):
        """Handle sort type change."""
        self.recalculate_widget_positions()

    def _get_sorted_widgets(self):
        """Get widgets sorted according to the current sort setting."""
        sort_type = self.sort_combo.currentText()
        widgets = list(self.annotation_widgets_by_id.values())
        if sort_type == "Label":
            widgets.sort(key=lambda w: w.data_item.effective_label.short_label_code)
        elif sort_type == "Image":
            widgets.sort(key=lambda w: os.path.basename(w.data_item.annotation.image_path))
        return widgets

    def _group_widgets_by_sort_key(self, widgets):
        """Group widgets by the current sort key."""
        sort_type = self.sort_combo.currentText()
        if sort_type == "None":
            return [("", widgets)]
        groups = []
        current_group = []
        current_key = None
        for widget in widgets:
            if sort_type == "Label":
                key = widget.data_item.effective_label.short_label_code
            elif sort_type == "Image":
                key = os.path.basename(widget.data_item.annotation.image_path)
            else:
                key = ""
            if current_key != key:
                if current_group:
                    groups.append((current_key, current_group))
                current_group = [widget]
                current_key = key
            else:
                current_group.append(widget)
        if current_group:
            groups.append((current_key, current_group))
        return groups

    def _clear_separator_labels(self):
        """Remove any existing group header labels."""
        if hasattr(self, '_group_headers'):
            for header in self._group_headers:
                header.setParent(None)
                header.deleteLater()
        self._group_headers = []

    def _create_group_header(self, text):
        """Create a group header label."""
        if not hasattr(self, '_group_headers'):
            self._group_headers = []
        header = QLabel(text, self.content_widget)
        header.setStyleSheet(
            "QLabel {"
            " font-weight: bold;"
            " font-size: 12px;"
            " color: #555;"
            " background-color: #f0f0f0;"
            " border: 1px solid #ccc;"
            " border-radius: 3px;"
            " padding: 5px 8px;"
            " margin: 2px 0px;"
            " }"
        )
        header.setFixedHeight(30)
        header.setMinimumWidth(self.viewport().width() - 20)
        header.show()
        self._group_headers.append(header)
        return header

    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
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
        if not self.annotation_widgets_by_id:
            self.content_widget.setMinimumSize(1, 1)
            return

        self._clear_separator_labels()
        visible_widgets = [w for w in self._get_sorted_widgets() if not w.isHidden()]
        if not visible_widgets:
            self.content_widget.setMinimumSize(1, 1)
            return

        # Create groups based on the current sort key
        groups = self._group_widgets_by_sort_key(visible_widgets)
        spacing = max(5, int(self.current_widget_size * 0.08))
        available_width = self.viewport().width()
        x, y = spacing, spacing
        max_height_in_row = 0

        # Calculate the maximum height of the widgets in each row
        for group_name, group_widgets in groups:
            if group_name and self.sort_combo.currentText() != "None":
                if x > spacing:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                header_label = self._create_group_header(group_name)
                header_label.move(x, y)
                y += header_label.height() + spacing
                x = spacing
                max_height_in_row = 0
                
            for widget in group_widgets:
                widget_size = widget.size()
                if x > spacing and x + widget_size.width() > available_width:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                widget.move(x, y)
                x += widget_size.width() + spacing
                max_height_in_row = max(max_height_in_row, widget_size.height())

        total_height = y + max_height_in_row + spacing
        self.content_widget.setMinimumSize(available_width, total_height)

    def update_annotations(self, data_items):
        """Update displayed annotations, creating new widgets for them."""
        if self.isolated_mode:
            self.show_all_annotations()
            
        for widget in self.annotation_widgets_by_id.values():
            widget.setParent(None)
            widget.deleteLater()
            
        self.annotation_widgets_by_id.clear()
        self.selected_widgets.clear()
        self.last_selected_index = -1
        
        for data_item in data_items:
            annotation_widget = AnnotationImageWidget(
                data_item, self.current_widget_size, self, self.content_widget)
            
            annotation_widget.show()
            self.annotation_widgets_by_id[data_item.annotation.id] = annotation_widget
            
        self.recalculate_widget_positions()
        self._update_toolbar_state()

    def resizeEvent(self, event):
        """On window resize, reflow the annotation widgets."""
        super().resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self.recalculate_widget_positions)
        self._resize_timer.start(100)
        
    def keyPressEvent(self, event):
        """Handles key presses for deleting selected annotations."""
        # Check if the pressed key is Delete/Backspace AND the Control key is held down
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and event.modifiers() == Qt.ControlModifier:
            # Proceed only if there are selected widgets
            if not self.selected_widgets:
                super().keyPressEvent(event)
                return

            print(f"Marking {len(self.selected_widgets)} annotations for deletion.")
            
            # Keep track of which annotations were affected
            changed_ids = []

            # Mark each selected item for deletion and hide it
            for widget in self.selected_widgets:
                widget.data_item.mark_for_deletion()
                widget.hide()
                changed_ids.append(widget.data_item.annotation.id)

            # Clear the list of selected widgets
            self.selected_widgets.clear()

            # Recalculate the layout to fill in the empty space
            self.recalculate_widget_positions()

            # Emit a signal to notify the ExplorerWindow that the selection is now empty
            # This will also clear the selection in the EmbeddingViewer
            if changed_ids:
                self.selection_changed.emit(changed_ids)

            # Accept the event to prevent it from being processed further
            event.accept()
        else:
            # Pass any other key presses to the default handler
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for starting rubber band selection OR clearing selection."""
        if event.button() == Qt.LeftButton:
            if not event.modifiers():
                # If left click with no modifiers, check if click is outside widgets
                is_on_widget = False
                child_at_pos = self.childAt(event.pos())
                
                if child_at_pos:
                    widget = child_at_pos
                    # Traverse up the parent chain to see if click is on an annotation widget
                    while widget and widget != self:
                        if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                            is_on_widget = True
                            break
                        widget = widget.parent()
                        
                # If click is outside widgets and there is a selection, clear it
                if not is_on_widget and self.selected_widgets:
                    changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                    self.clear_selection()
                    self.selection_changed.emit(changed_ids)
                    return
                
            elif event.modifiers() == Qt.ControlModifier:
                # Start rubber band selection with Ctrl+Left click
                self.selection_at_press = set(self.selected_widgets)
                self.rubber_band_origin = event.pos()
                self.mouse_pressed_on_widget = False
                child_widget = self.childAt(event.pos())
                if child_widget:
                    widget = child_widget
                    # Check if click is on a widget to avoid starting rubber band
                    while widget and widget != self:
                        if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                            self.mouse_pressed_on_widget = True
                            break
                        widget = widget.parent()
                return
            
        elif event.button() == Qt.RightButton:
            # Ignore right clicks
            event.ignore()
            return
        
        # Default handler for other cases
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and exit isolation mode."""
        if event.button() == Qt.LeftButton:
            changed_ids = []
            if self.selected_widgets:
                changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                self.clear_selection()
                self.selection_changed.emit(changed_ids)
            if self.isolated_mode:
                self.show_all_annotations()
            self.reset_view_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for DYNAMIC rubber band selection."""
        # Only proceed if Ctrl+Left mouse drag is active and not on a widget
        if (
            self.rubber_band_origin is None or
            event.buttons() != Qt.LeftButton or
            event.modifiers() != Qt.ControlModifier
        ):
            super().mouseMoveEvent(event)
            return

        if self.mouse_pressed_on_widget:
            # If drag started on a widget, do not start rubber band
            super().mouseMoveEvent(event)
            return

        # Only start selection if drag distance exceeds threshold
        distance = (event.pos() - self.rubber_band_origin).manhattanLength()
        if distance < self.drag_threshold:
            return

        # Create and show the rubber band if not already present
        if not self.rubber_band:
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())

        rect = QRect(self.rubber_band_origin, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()
        selection_rect = self.rubber_band.geometry()
        content_widget = self.content_widget
        changed_ids = []

        # Iterate over all annotation widgets to update selection state
        for widget in self.annotation_widgets_by_id.values():
            widget_rect_in_content = widget.geometry()
            # Map widget's rect to viewport coordinates
            widget_rect_in_viewport = QRect(
                content_widget.mapTo(self.viewport(), widget_rect_in_content.topLeft()),
                widget_rect_in_content.size()
            )
            is_in_band = selection_rect.intersects(widget_rect_in_viewport)
            should_be_selected = (widget in self.selection_at_press) or is_in_band

            # Select or deselect widgets as needed
            if should_be_selected and not widget.is_selected():
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)

            elif not should_be_selected and widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)

        # Emit signal if any selection state changed
        if changed_ids:
            self.selection_changed.emit(changed_ids)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete rubber band selection."""
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None
                
            self.selection_at_press = set()
            self.rubber_band_origin = None
            self.mouse_pressed_on_widget = False
            event.accept()
            return
        
        super().mouseReleaseEvent(event)

    def handle_annotation_selection(self, widget, event):
        """Handle selection of annotation widgets with different modes (single, ctrl, shift)."""
        widget_list = [w for w in self._get_sorted_widgets() if not w.isHidden()]

        try:
            widget_index = widget_list.index(widget)
        except ValueError:
            return

        modifiers = event.modifiers()
        changed_ids = []

        # Shift or Shift+Ctrl: range selection
        if modifiers == Qt.ShiftModifier or modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            if self.last_selected_index != -1:
                # Find the last selected widget in the current list
                last_selected_widget = None
                for w in self.selected_widgets:
                    if w in widget_list:
                        try:
                            last_index_in_current_list = widget_list.index(w)
                            if (
                                last_selected_widget is None
                                or last_index_in_current_list > widget_list.index(last_selected_widget)
                            ):
                                last_selected_widget = w
                        except ValueError:
                            continue
                        
                if last_selected_widget:
                    last_selected_index_in_current_list = widget_list.index(last_selected_widget)
                    start = min(last_selected_index_in_current_list, widget_index)
                    end = max(last_selected_index_in_current_list, widget_index)
                else:
                    start, end = widget_index, widget_index
                    
                # Select all widgets in the range
                for i in range(start, end + 1):
                    if self.select_widget(widget_list[i]):
                        changed_ids.append(widget_list[i].data_item.annotation.id)
            else:
                # No previous selection, just select the clicked widget
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
                self.last_selected_index = widget_index

        # Ctrl: toggle selection of the clicked widget
        elif modifiers == Qt.ControlModifier:
            if widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            self.last_selected_index = widget_index

        # No modifier: single selection
        else:
            newly_selected_id = widget.data_item.annotation.id
            
            # Deselect all others
            for w in list(self.selected_widgets):
                if w.data_item.annotation.id != newly_selected_id:
                    if self.deselect_widget(w):
                        changed_ids.append(w.data_item.annotation.id)
                        
            # Select the clicked widget
            if self.select_widget(widget):
                changed_ids.append(newly_selected_id)
            self.last_selected_index = widget_index

        # If in isolated mode, update which widgets are visible
        if self.isolated_mode:
            self._update_isolation()

        # Emit signal if any selection state changed
        if changed_ids:
            self.selection_changed.emit(changed_ids)

    def _update_isolation(self):
        """Update the isolated view to show only currently selected widgets."""
        if not self.isolated_mode: 
            return
        # If in isolated mode, only show selected widgets
        if self.selected_widgets:
            self.isolated_widgets.update(self.selected_widgets)
            self.setUpdatesEnabled(False)
            try:
                for widget in self.annotation_widgets_by_id.values():
                    if widget not in self.isolated_widgets: 
                        widget.hide()
                    else: 
                        widget.show()
                self.recalculate_widget_positions()
                
            finally:
                self.setUpdatesEnabled(True)

    def select_widget(self, widget):
        """Selects a widget, updates its data_item, and returns True if state changed."""
        if not widget.is_selected():  # is_selected() checks the data_item
            # 1. Controller modifies the state on the data item
            widget.data_item.set_selected(True)
            # 2. Controller tells the view to update its appearance
            widget.update_selection_visuals()
            self.selected_widgets.append(widget)
            self._update_toolbar_state()
            return True
        return False

    def deselect_widget(self, widget):
        """Deselects a widget, updates its data_item, and returns True if state changed."""
        if widget.is_selected():
            # 1. Controller modifies the state on the data item
            widget.data_item.set_selected(False)
            # 2. Controller tells the view to update its appearance
            widget.update_selection_visuals()
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            self._update_toolbar_state()
            return True
        return False

    def clear_selection(self):
        """Clear all selected widgets and update toolbar state."""
        for widget in list(self.selected_widgets):
            # This will internally call deselect_widget, which is fine
            self.deselect_widget(widget)
            
        self.selected_widgets.clear()
        self._update_toolbar_state()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]

    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of widgets based on a set of IDs from the controller."""
        self.setUpdatesEnabled(False)
        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in selected_ids
                # 1. Update the state on the central data item
                widget.data_item.set_selected(is_selected)
                # 2. Tell the widget to update its visuals based on the new state
                widget.update_selection_visuals()
            
            # Resync internal list of selected widgets from the source of truth
            self.selected_widgets = [w for w in self.annotation_widgets_by_id.values() if w.is_selected()]
            
            if self.isolated_mode and self.selected_widgets:
                self.isolated_widgets.update(self.selected_widgets)
                for widget in self.annotation_widgets_by_id.values():
                    widget.setHidden(widget not in self.isolated_widgets)
                self.recalculate_widget_positions()
        finally:
            self.setUpdatesEnabled(True)
        self._update_toolbar_state()

    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label and emit a signal for the embedding view to update."""
        if not self.selected_widgets or not preview_label: 
            return
        changed_ids = []
        for widget in self.selected_widgets:
            widget.data_item.set_preview_label(preview_label)
            widget.update()  # Force repaint with new color
            changed_ids.append(widget.data_item.annotation.id)
            
        if self.sort_combo.currentText() == "Label": 
            self.recalculate_widget_positions()
        if changed_ids: 
            self.preview_changed.emit(changed_ids)

    def clear_preview_states(self):
        """
        Clears all preview states, including label changes and items marked
        for deletion, reverting them to their original state.
        """
        something_changed = False
        for widget in self.annotation_widgets_by_id.values():
            # Check for and clear preview labels
            if widget.data_item.has_preview_changes():
                widget.data_item.clear_preview_label()
                widget.update()  # Repaint to show original color
                something_changed = True

            # Check for and un-mark items for deletion
            if widget.data_item.is_marked_for_deletion():
                widget.data_item.unmark_for_deletion()
                widget.show()  # Make the widget visible again
                something_changed = True

        if something_changed:
            # Recalculate positions to update sorting and re-flow the layout
            if self.sort_combo.currentText() in ("Label", "Image"):
                self.recalculate_widget_positions()

    def has_preview_changes(self):
        """Return True if there are preview changes."""
        return any(w.data_item.has_preview_changes() for w in self.annotation_widgets_by_id.values())

    def get_preview_changes_summary(self):
        """Get a summary of preview changes."""
        change_count = sum(1 for w in self.annotation_widgets_by_id.values() if w.data_item.has_preview_changes())
        return f"{change_count} annotation(s) with preview changes" if change_count else "No preview changes"

    def apply_preview_changes_permanently(self):
        """Apply preview changes permanently."""
        applied_annotations = []
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.apply_preview_permanently():
                applied_annotations.append(widget.annotation)
        return applied_annotations


# ----------------------------------------------------------------------------------------------------------------------
# ExplorerWindow
# ----------------------------------------------------------------------------------------------------------------------


class ExplorerWindow(QMainWindow):
    def __init__(self, main_window, parent=None):
        """Initialize the ExplorerWindow."""
        super(ExplorerWindow, self).__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.device = main_window.device
        self.model_path = ""
        self.loaded_model = None
        self.current_data_items = []
        self.current_features = None
        self.current_feature_generating_model = ""
        self._ui_initialized = False

        self.setWindowTitle("Explorer")
        explorer_icon_path = get_icon("magic.png")
        self.setWindowIcon(QIcon(explorer_icon_path))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        self.annotation_settings_widget = None
        self.model_settings_widget = None
        self.embedding_settings_widget = None
        self.annotation_viewer = None
        self.embedding_viewer = None

        self.clear_preview_button = QPushButton('Clear Preview', self)
        self.clear_preview_button.clicked.connect(self.clear_preview_changes)
        self.clear_preview_button.setToolTip("Clear all preview changes and revert to original labels")
        self.clear_preview_button.setEnabled(False)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setToolTip("Close the window")

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply)
        self.apply_button.setToolTip("Apply changes")
        self.apply_button.setEnabled(False)

    def showEvent(self, event):
        """Handle show event."""
        if not self._ui_initialized:
            self.setup_ui()
            self._ui_initialized = True
        super(ExplorerWindow, self).showEvent(event)

    def closeEvent(self, event):
        """Handle close event."""
        # Stop any running timers to prevent errors
        if hasattr(self, 'embedding_viewer') and self.embedding_viewer:
            if hasattr(self.embedding_viewer, 'animation_timer') and self.embedding_viewer.animation_timer:
                self.embedding_viewer.animation_timer.stop()

        # Call the main cancellation method to revert any pending changes
        self.clear_preview_changes()

        # Call the dedicated cleanup method
        self._cleanup_resources()

        # Re-enable the main window before closing
        if self.main_window:
            self.main_window.setEnabled(True)

        # Move the label_window back to the main_window
        if hasattr(self.main_window, 'explorer_closed'):
            self.main_window.explorer_closed()

        # Clear the reference in the main_window to allow garbage collection
        self.main_window.explorer_window = None

        # Set the ui_initialized flag to False so it can be re-initialized next time
        self._ui_initialized = False

        event.accept()

    def setup_ui(self):
        """Set up the UI for the ExplorerWindow."""
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)

        # Lazily initialize the settings and viewer widgets if they haven't been created yet.
        # This ensures that the widgets are only created once per ExplorerWindow instance.

        # Annotation settings panel (filters by image, type, label)
        if self.annotation_settings_widget is None: 
            self.annotation_settings_widget = AnnotationSettingsWidget(self.main_window, self)
            
        # Model selection panel (choose feature extraction model)
        if self.model_settings_widget is None: 
            self.model_settings_widget = ModelSettingsWidget(self.main_window, self)
            
        # Embedding settings panel (choose dimensionality reduction method)
        if self.embedding_settings_widget is None: 
            self.embedding_settings_widget = EmbeddingSettingsWidget(self.main_window, self)
            
        # Annotation viewer (shows annotation image crops in a grid)
        if self.annotation_viewer is None: 
            self.annotation_viewer = AnnotationViewer(self)
            
        # Embedding viewer (shows 2D embedding scatter plot)
        if self.embedding_viewer is None: 
            self.embedding_viewer = EmbeddingViewer(self)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.annotation_settings_widget, 2)
        top_layout.addWidget(self.model_settings_widget, 1)
        top_layout.addWidget(self.embedding_settings_widget, 1)
        top_container = QWidget()
        top_container.setLayout(top_layout)
        self.main_layout.addWidget(top_container)

        middle_splitter = QSplitter(Qt.Horizontal)
        annotation_group = QGroupBox("Annotation Viewer")
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.addWidget(self.annotation_viewer)
        middle_splitter.addWidget(annotation_group)

        embedding_group = QGroupBox("Embedding Viewer")
        embedding_layout = QVBoxLayout(embedding_group)
        embedding_layout.addWidget(self.embedding_viewer)
        middle_splitter.addWidget(embedding_group)
        middle_splitter.setSizes([500, 500])
        self.main_layout.addWidget(middle_splitter, 1)
        self.main_layout.addWidget(self.label_window)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.clear_preview_button)
        self.buttons_layout.addWidget(self.exit_button)
        self.buttons_layout.addWidget(self.apply_button)
        self.main_layout.addLayout(self.buttons_layout)

        self.annotation_settings_widget.set_default_to_current_image()
        self.refresh_filters()

        try:
            self.label_window.labelSelected.disconnect(self.on_label_selected_for_preview)
        except TypeError:
            pass
        
        # Connect signals to slots
        self.label_window.labelSelected.connect(self.on_label_selected_for_preview)
        self.annotation_viewer.selection_changed.connect(self.on_annotation_view_selection_changed)
        self.annotation_viewer.preview_changed.connect(self.on_preview_changed)
        self.annotation_viewer.reset_view_requested.connect(self.on_reset_view_requested)
        self.embedding_viewer.selection_changed.connect(self.on_embedding_view_selection_changed)
        self.embedding_viewer.reset_view_requested.connect(self.on_reset_view_requested)

    @pyqtSlot(list)
    def on_annotation_view_selection_changed(self, changed_ann_ids):
        """Syncs selection from AnnotationViewer to EmbeddingViewer."""
        all_selected_ids = {w.data_item.annotation.id for w in self.annotation_viewer.selected_widgets}
        if self.embedding_viewer.points_by_id:
            self.embedding_viewer.render_selection_from_ids(all_selected_ids)
        
        # Call the new centralized method
        self.update_label_window_selection()

    @pyqtSlot(list)
    def on_embedding_view_selection_changed(self, all_selected_ann_ids):
        """Syncs selection from EmbeddingViewer to AnnotationViewer."""
        # Check the state BEFORE the selection is changed
        was_empty_selection = len(self.annotation_viewer.selected_widgets) == 0

        # Now, update the selection in the annotation viewer
        self.annotation_viewer.render_selection_from_ids(set(all_selected_ann_ids))

        # The rest of the logic now works correctly
        is_new_selection = len(all_selected_ann_ids) > 0
        if (
            was_empty_selection and
            is_new_selection and
            not self.annotation_viewer.isolated_mode
        ):
            self.annotation_viewer.isolate_selection()

        self.update_label_window_selection()

    @pyqtSlot(list)
    def on_preview_changed(self, changed_ann_ids):
        """Updates embedding point colors when a preview label is applied."""
        for ann_id in changed_ann_ids:
            point = self.embedding_viewer.points_by_id.get(ann_id)
            if point:
                point.update()

    @pyqtSlot()
    def on_reset_view_requested(self):
        """Handle reset view requests from double-click in either viewer."""
        # Clear all selections in both viewers
        self.annotation_viewer.clear_selection()
        self.embedding_viewer.render_selection_from_ids(set())

        # Exit isolation mode if currently active
        if self.annotation_viewer.isolated_mode:
            self.annotation_viewer.show_all_annotations()

        self.update_label_window_selection() 
        self.update_button_states()

        print("Reset view: cleared selections and exited isolation mode")

    def update_label_window_selection(self):
        """
        Updates the label window based on the selection state of the currently
        loaded data items. This is the single, centralized point of logic.
        """
        # Get selected items directly from the master data list
        selected_data_items = [
            item for item in self.current_data_items if item.is_selected
        ]

        if not selected_data_items:
            self.label_window.deselect_active_label()
            self.label_window.update_annotation_count()
            return

        first_effective_label = selected_data_items[0].effective_label
        all_same_current_label = all(
            item.effective_label.id == first_effective_label.id 
            for item in selected_data_items
        )

        if all_same_current_label:
            self.label_window.set_active_label(first_effective_label)
            # This emit is what updates other UI elements, like the annotation list
            self.annotation_window.labelSelected.emit(first_effective_label.id)
        else:
            self.label_window.deselect_active_label()

        self.label_window.update_annotation_count()

    def get_filtered_data_items(self):
        """Gets annotations matching all conditions as AnnotationDataItem objects."""
        data_items = []
        if not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return data_items

        selected_images = self.annotation_settings_widget.get_selected_images()
        selected_types = self.annotation_settings_widget.get_selected_annotation_types()
        selected_labels = self.annotation_settings_widget.get_selected_labels()

        if not all([selected_images, selected_types, selected_labels]):
            return []

        annotations_to_process = [
            ann for ann in self.main_window.annotation_window.annotations_dict.values()
            if (os.path.basename(ann.image_path) in selected_images and
                type(ann).__name__ in selected_types and
                ann.label.short_label_code in selected_labels)
        ]

        self._ensure_cropped_images(annotations_to_process)
        return [AnnotationDataItem(ann) for ann in annotations_to_process]

    def _ensure_cropped_images(self, annotations):
        """Ensures all provided annotations have a cropped image available."""
        annotations_by_image = {}
        
        for annotation in annotations:
            if not annotation.cropped_image:
                image_path = annotation.image_path
                if image_path not in annotations_by_image:
                    annotations_by_image[image_path] = []
                annotations_by_image[image_path].append(annotation)
                
        if not annotations_by_image: 
            return

        progress_bar = ProgressBar(self, "Cropping Image Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations_by_image))
        
        try:
            for image_path, image_annotations in annotations_by_image.items():
                self.annotation_window.crop_annotations(image_path=image_path, 
                                                        annotations=image_annotations, 
                                                        return_annotations=False, 
                                                        verbose=False)
                progress_bar.update_progress()
        finally:
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def _extract_color_features(self, data_items, progress_bar=None, bins=32):
        """
        Extracts color-based features from annotation crops.

        Features extracted per annotation:
            - Mean, standard deviation, skewness, and kurtosis for each RGB channel
            - Normalized histogram for each RGB channel
            - Grayscale statistics: mean, std, range
            - Geometric features: area, perimeter (if available)
        Returns:
            features: np.ndarray of shape (N, feature_dim)
            valid_data_items: list of AnnotationDataItem with valid crops
        """
        if progress_bar:
            progress_bar.set_title("Extracting features...")
            progress_bar.start_progress(len(data_items))

        features = []
        valid_data_items = []

        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                # Convert QPixmap to numpy array (H, W, 3)
                arr = pixmap_to_numpy(pixmap)
                pixels = arr.reshape(-1, 3)

                # Basic color statistics
                mean_color = np.mean(pixels, axis=0)
                std_color = np.std(pixels, axis=0)

                # Skewness and kurtosis for each channel
                epsilon = 1e-8  # Prevent division by zero
                centered_pixels = pixels - mean_color
                skew_color = np.mean(centered_pixels ** 3, axis=0) / (std_color ** 3 + epsilon)
                kurt_color = np.mean(centered_pixels ** 4, axis=0) / (std_color ** 4 + epsilon) - 3

                # Normalized histograms for each channel
                histograms = [
                    np.histogram(pixels[:, i], bins=bins, range=(0, 255))[0]
                    for i in range(3)
                ]
                histograms = [
                    h / h.sum() if h.sum() > 0 else np.zeros(bins)
                    for h in histograms
                ]

                # Grayscale statistics
                gray_arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                grayscale_stats = np.array([
                    np.mean(gray_arr),
                    np.std(gray_arr),
                    np.ptp(gray_arr)
                ])

                # Geometric features (area, perimeter)
                area = getattr(item.annotation, 'area', 0.0)
                perimeter = getattr(item.annotation, 'perimeter', 0.0)
                geometric_features = np.array([area, perimeter])

                # Concatenate all features into a single vector
                current_features = np.concatenate([
                    mean_color,
                    std_color,
                    skew_color,
                    kurt_color,
                    *histograms,
                    grayscale_stats,
                    geometric_features
                ])

                features.append(current_features)
                valid_data_items.append(item)

            if progress_bar:
                progress_bar.update_progress()

        return np.array(features), valid_data_items

    def _extract_yolo_features(self, data_items, model_info, progress_bar=None):
        """Extracts features from annotation crops using a YOLO model."""
        # Extract model name and feature mode from the provided model_info tuple
        model_name, feature_mode = model_info
        
        if model_name != self.model_path or self.loaded_model is None:
            try:
                self.loaded_model = YOLO(model_name)
                self.model_path = model_name
                self.imgsz = getattr(self.loaded_model.model.args, 'imgsz', 128)
                dummy_image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                self.loaded_model.predict(dummy_image, imgsz=self.imgsz, half=True, device=self.device, verbose=False)
                
            except Exception as e:
                print(f"ERROR: Could not load YOLO model '{model_name}': {e}")
                return np.array([]), []

        if progress_bar: 
            progress_bar.set_title("Preparing images...")
            progress_bar.start_progress(len(data_items))
            
        image_list, valid_data_items = [], []
        for item in data_items:
            # Get the cropped image from the annotation            
            pixmap = item.annotation.get_cropped_image()
            
            if pixmap and not pixmap.isNull():
                image_list.append(pixmap_to_numpy(pixmap))
                valid_data_items.append(item)
                
            if progress_bar: 
                progress_bar.update_progress()
                
        if not valid_data_items: 
            return np.array([]), []
        
        # Specify the kwargs for YOLO model prediction
        kwargs = {'stream': True, 
                  'imgsz': self.imgsz, 
                  'half': True, 
                  'device': self.device, 
                  'verbose': False}
        
        # Run the model to extract features
        if feature_mode == "Embed Features":
            results_generator = self.loaded_model.embed(image_list, **kwargs)
        else:
            results_generator = self.loaded_model.predict(image_list, **kwargs)
            
        if progress_bar: 
            progress_bar.set_title("Extracting features...")
            progress_bar.start_progress(len(valid_data_items))
        
        # Prepare a list to hold the extracted features    
        embeddings_list = []

        try:
            # Iterate through the results and extract features based on the mode
            for i, result in enumerate(results_generator):
                if feature_mode == "Embed Features":
                    embeddings_list.append(result.cpu().numpy().flatten())
                    
                elif hasattr(result, 'probs') and result.probs is not None:
                    embeddings_list.append(result.probs.data.cpu().numpy().squeeze())
                    
                else:
                    raise TypeError("Model did not return expected output")
                
                if progress_bar: 
                    progress_bar.update_progress()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return np.array(embeddings_list), valid_data_items

    def _extract_features(self, data_items, progress_bar=None):
        """Dispatcher to call the appropriate feature extraction function."""
        # Get the selected model and feature mode from the model settings widget
        model_name, feature_mode = self.model_settings_widget.get_selected_model()
        
        if isinstance(model_name, tuple): 
            model_name = model_name[0]
            
        if not model_name: 
            return np.array([]), []
        
        if model_name == "Color Features":
            return self._extract_color_features(data_items, progress_bar=progress_bar)
        
        elif ".pt" in model_name:
            return self._extract_yolo_features(data_items, (model_name, feature_mode), progress_bar=progress_bar)
        
        return np.array([]), []

    def _run_dimensionality_reduction(self, features, params):
        """
        Runs dimensionality reduction (PCA, UMAP, or t-SNE) on the feature matrix.

        Args:
            features (np.ndarray): Feature matrix of shape (N, D).
            params (dict): Embedding parameters, including technique and its hyperparameters.

        Returns:
            np.ndarray or None: 2D embedded features of shape (N, 2), or None on failure.
        """
        technique = params.get('technique', 'UMAP')
        
        if len(features) <= 2:
            # Not enough samples for dimensionality reduction
            return None

        try:
            # Standardize features before reduction
            features_scaled = StandardScaler().fit_transform(features)

            if technique == "UMAP":
                # UMAP: n_neighbors must be < n_samples
                n_neighbors = min(params.get('n_neighbors', 15), len(features_scaled) - 1)
                
                reducer = UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=params.get('min_dist', 0.1),
                    metric=params.get('metric', 'cosine')
                )
            elif technique == "TSNE":
                # t-SNE: perplexity must be < n_samples
                perplexity = min(params.get('perplexity', 30), len(features_scaled) - 1)
                
                reducer = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    early_exaggeration=params.get('early_exaggeration', 12.0),
                    learning_rate=params.get('learning_rate', 'auto'),
                    init='pca'
                )
            elif technique == "PCA":
                reducer = PCA(n_components=2, random_state=42)
            else:
                # Unknown technique
                return None

            # Fit and transform the features
            return reducer.fit_transform(features_scaled)

        except Exception as e:
            print(f"Error during {technique} dimensionality reduction: {e}")
            return None

    def _update_data_items_with_embedding(self, data_items, embedded_features):
        """Updates AnnotationDataItem objects with embedding results."""
        scale_factor = 4000
        min_vals, max_vals = np.min(embedded_features, axis=0), np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals
        for i, item in enumerate(data_items):
            norm_x = (embedded_features[i, 0] - min_vals[0]) / range_vals[0] if range_vals[0] > 0 else 0.5
            norm_y = (embedded_features[i, 1] - min_vals[1]) / range_vals[1] if range_vals[1] > 0 else 0.5
            item.embedding_x = (norm_x * scale_factor) - (scale_factor / 2)
            item.embedding_y = (norm_y * scale_factor) - (scale_factor / 2)

    def run_embedding_pipeline(self):
        """Orchestrates the feature extraction and dimensionality reduction pipeline."""
        if not self.current_data_items: 
            return
        
        self.annotation_viewer.clear_selection()
        if self.annotation_viewer.isolated_mode: 
            self.annotation_viewer.show_all_annotations()
            
        self.embedding_viewer.render_selection_from_ids(set())
        self.update_button_states()

        # Get embedding parameters and selected model; create a cache key to avoid re-computing features
        embedding_params = self.embedding_settings_widget.get_embedding_parameters()
        model_info = self.model_settings_widget.get_selected_model()
        selected_model, selected_feature_mode = model_info if isinstance(model_info, tuple) else (model_info, "default")
        cache_key = f"{selected_model}_{selected_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Generating Embedding Visualization")
        progress_bar.show()
        try:
            if self.current_features is None or cache_key != self.current_feature_generating_model:
                features, valid_data_items = self._extract_features(self.current_data_items, progress_bar=progress_bar)
                self.current_features = features
                self.current_feature_generating_model = cache_key
                self.current_data_items = valid_data_items
                self.annotation_viewer.update_annotations(self.current_data_items)
            else:
                features = self.current_features
                
            if features is None or len(features) == 0: 
                return

            progress_bar.set_busy_mode("Running dimensionality reduction...")
            embedded_features = self._run_dimensionality_reduction(features, embedding_params)
            
            if embedded_features is None: 
                return

            progress_bar.set_busy_mode("Updating visualization...")
            self._update_data_items_with_embedding(self.current_data_items, embedded_features)
            self.embedding_viewer.update_embeddings(self.current_data_items)
            self.embedding_viewer.show_embedding()
            self.embedding_viewer.fit_view_to_points()
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def refresh_filters(self):
        """Refresh display: filter data and update annotation viewer."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.current_data_items = self.get_filtered_data_items()
            self.current_features = None 
            self.annotation_viewer.update_annotations(self.current_data_items)
            self.embedding_viewer.clear_points()
            self.embedding_viewer.show_placeholder()
        finally:
            QApplication.restoreOverrideCursor()

    def on_label_selected_for_preview(self, label):
        """Handle label selection to update preview state."""
        if hasattr(self, 'annotation_viewer') and self.annotation_viewer.selected_widgets:
            self.annotation_viewer.apply_preview_label_to_selected(label)
            self.update_button_states()

    def clear_preview_changes(self):
        """
        Clears all preview changes in both viewers, including label changes
        and items marked for deletion.
        """
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.clear_preview_states()

        if hasattr(self, 'embedding_viewer'):
            self.embedding_viewer.refresh_points()

        # After reverting all changes, update the button states
        self.update_button_states()
        print("Cleared all pending changes.")

    def update_button_states(self):
        """Update the state of Clear Preview and Apply buttons."""
        has_changes = self.annotation_viewer.has_preview_changes()
        self.clear_preview_button.setEnabled(has_changes)
        self.apply_button.setEnabled(has_changes)
        summary = self.annotation_viewer.get_preview_changes_summary()
        self.clear_preview_button.setToolTip(f"Clear all preview changes - {summary}")
        self.apply_button.setToolTip(f"Apply changes - {summary}")

    def apply(self):
        """
        Apply all pending changes, including label modifications and deletions,
        to the main application's data.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Separate items into those to be deleted and those to be kept
            items_to_delete = [item for item in self.current_data_items if item.is_marked_for_deletion()]
            items_to_keep = [item for item in self.current_data_items if not item.is_marked_for_deletion()]

            # --- 1. Process Deletions ---
            deleted_annotations = []
            if items_to_delete:
                deleted_annotations = [item.annotation for item in items_to_delete]
                print(f"Permanently deleting {len(deleted_annotations)} annotation(s).")
                self.annotation_window.delete_annotations(deleted_annotations)

            # --- 2. Process Label Changes on remaining items ---
            applied_label_changes = []
            for item in items_to_keep:
                if item.apply_preview_permanently():
                    applied_label_changes.append(item.annotation)

            # --- 3. Update UI if any changes were made ---
            if not deleted_annotations and not applied_label_changes:
                print("No pending changes to apply.")
                return

            # Update the Explorer's internal list of data items
            self.current_data_items = items_to_keep

            # Update the main application's data and UI
            all_affected_annotations = deleted_annotations + applied_label_changes
            affected_images = {ann.image_path for ann in all_affected_annotations}
            for image_path in affected_images:
                self.image_window.update_image_annotations(image_path)
            self.annotation_window.load_annotations()

            # Refresh the annotation viewer since its underlying data has changed
            self.annotation_viewer.update_annotations(self.current_data_items)

            # Reset selections and button states
            self.embedding_viewer.render_selection_from_ids(set())
            self.update_label_window_selection()
            self.update_button_states()

            print(f"Applied changes successfully.")

        except Exception as e:
            print(f"Error applying modifications: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _cleanup_resources(self):
        """Clean up resources."""
        self.loaded_model = None
        self.model_path = ""
        self.current_features = None
        self.current_feature_generating_model = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()