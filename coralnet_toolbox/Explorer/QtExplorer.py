from coralnet_toolbox.Icons import get_icon

from PyQt5.QtGui import QIcon, QBrush, QPen, QColor, QPainter, QImage
from PyQt5.QtCore import Qt, QTimer, QSize, QRect, pyqtSignal, QSignalBlocker, pyqtSlot

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget, QGridLayout,
                             QMainWindow, QSplitter, QGroupBox, QFormLayout,
                             QSpinBox, QGraphicsEllipseItem, QGraphicsItem, QSlider,
                             QListWidget, QDoubleSpinBox, QApplication, QStyle)

from coralnet_toolbox.QtProgressBar import ProgressBar

import warnings
import os
import random

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import numpy as np
    from umap import UMAP  
except ImportError:
    TSNE = None
    KMeans = None
    np = None
    UMAP = None  


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

ANNOTATION_WIDTH = 5

POINT_SIZE = 15
POINT_WIDTH = 3

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationDataItem:
    """Holds annotation information for consistent display across viewers."""

    def __init__(self, annotation, cluster_x=None, cluster_y=None, cluster_id=None):
        self.annotation = annotation
        self.cluster_x = cluster_x if cluster_x is not None else 0.0
        self.cluster_y = cluster_y if cluster_y is not None else 0.0
        self.cluster_id = cluster_id if cluster_id is not None else 0
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label

    @property
    def effective_label(self):
        """Get the current effective label (preview if exists, otherwise original)."""
        return self._preview_label if self._preview_label else self.annotation.label

    @property
    def effective_color(self):
        """Get the effective color for this annotation."""
        # Special case for Review label (id == "-1")
        if self.effective_label.id == "-1":
            return QColor("black")
        return self.effective_label.color

    @property
    def is_selected(self):
        """Check if this annotation is selected."""
        return self._is_selected

    def set_selected(self, selected):
        """Set the selection state."""
        self._is_selected = selected

    def set_preview_label(self, label):
        """Set a preview label for this annotation."""
        self._preview_label = label

    def clear_preview_label(self):
        """Clear the preview label and revert to original."""
        self._preview_label = None

    def has_preview_changes(self):
        """Check if this annotation has preview changes."""
        return self._preview_label is not None

    def apply_preview_permanently(self):
        """Apply the preview label permanently to the annotation."""
        if self._preview_label:
            self.annotation.update_label(self._preview_label)
            self.annotation.update_user_confidence(self._preview_label)
            self._original_label = self._preview_label
            self._preview_label = None
            return True
        return False

    def get_display_info(self):
        """Get display information for this annotation."""
        return {
            'id': self.annotation.id,
            'label': self.effective_label.short_label_code,
            'confidence': self.get_effective_confidence(),
            'type': type(self.annotation).__name__,
            'image': os.path.basename(self.annotation.image_path),
            'cluster_id': self.cluster_id,
            'color': self.effective_color
        }

    def get_effective_confidence(self):
        """Get the effective confidence value."""
        if self.annotation.verified and hasattr(self.annotation, 'user_confidence') and self.annotation.user_confidence:
            return list(self.annotation.user_confidence.values())[0]
        elif hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        return 0.0


class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, data_item, widget_size=128, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.data_item = data_item
        self.annotation = data_item.annotation  # For convenience
        self.annotation_viewer = annotation_viewer
        self.widget_size = widget_size
        self.animation_offset = 0

        self.setFixedSize(widget_size, widget_size)

        # Timer for marching ants animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation_frame)
        self.animation_timer.setInterval(75)

        self.setup_ui()
        self.load_annotation_image()

    def setup_ui(self):
        """Set up the basic UI with a label for the image."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: none;")

        layout.addWidget(self.image_label)

    def load_annotation_image(self):
        """Load and display the actual annotation cropped image."""
        try:
            cropped_image = self.annotation.get_cropped_image(max_size=self.widget_size - 8)
            if cropped_image and not cropped_image.isNull():
                self.image_label.setPixmap(cropped_image)
            else:
                self.image_label.setText("No Image\nAvailable")
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.image_label.setText("Error\nLoading Image")

    def set_selected(self, selected):
        """Set the selection state and update visual appearance."""
        was_selected = self.is_selected()
        
        # Update the shared data item
        self.data_item.set_selected(selected)

        # Always update animation state, regardless of whether selection changed
        if self.is_selected():
            if not self.animation_timer.isActive():
                self.animation_timer.start()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
            self.animation_offset = 0

        # Only trigger repaint if state actually changed or if we're selected
        # (to ensure animation continues)
        if was_selected != selected or selected:
            self.update()

    def is_selected(self):
        """Return whether this widget is selected via the data item."""
        return self.data_item.is_selected

    def _update_animation_frame(self):
        """Update the animation offset and schedule a repaint."""
        self.animation_offset = (self.animation_offset + 1) % 20
        self.update()

    def paintEvent(self, event):
        """Handle custom drawing for the widget, including the border."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Base the pen on the shared data_item's state
        color = self.data_item.effective_color
        
        if self.is_selected():
            pen = QPen(color, ANNOTATION_WIDTH)  
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([2, 3])
            pen.setDashOffset(self.animation_offset)
        else:
            pen = QPen(color, ANNOTATION_WIDTH)  
            pen.setStyle(Qt.SolidLine)
        
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        width = pen.width()
        half_width = (width - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)
        painter.drawRect(rect)

    def update_size(self, new_size):
        """Updates the widget's size and reloads/rescales its content."""
        self.widget_size = new_size
        self.setFixedSize(new_size, new_size)
        self.image_label.setFixedSize(new_size - 8, new_size - 8)
        self.load_annotation_image()
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press events for selection."""
        if event.button() == Qt.LeftButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                self.annotation_viewer.handle_annotation_selection(self, event)
        # Ignore right mouse button clicks - don't call super() to prevent any selection
        elif event.button() == Qt.RightButton:
            event.ignore()
            return
        super().mousePressEvent(event)
        

class ClusterPointItem(QGraphicsEllipseItem):
    """
    A custom QGraphicsEllipseItem that prevents the default selection
    rectangle from being drawn, and dynamically gets its color from the
    shared AnnotationDataItem.
    """
    def paint(self, painter, option, widget):
        # Get the shared data item, which holds the current state
        data_item = self.data(0)
        if data_item:
            # Set the brush color based on the item's effective color
            # This ensures preview colors are reflected instantly.
            self.setBrush(data_item.effective_color)

        # Remove the 'State_Selected' flag to prevent the default box
        option.state &= ~QStyle.State_Selected
        super(ClusterPointItem, self).paint(painter, option, widget)
        

# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class ClusterViewer(QGraphicsView):
    """Custom QGraphicsView for interactive cluster visualization with zooming, panning, and selection."""
    
    # Define signal to report selection changes
    selection_changed = pyqtSignal(list)  # list of all currently selected annotation IDs
    
    def __init__(self, parent=None):
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)
        
        super().__init__(self.graphics_scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.explorer_window = parent
        self.points_by_id = {}  # Map annotation ID to cluster point
        self.animation_offset = 0
        
        self.previous_selection_ids = set()  # Track previous selection to detect changes
    
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)
        
        self.graphics_scene.selectionChanged.connect(self.on_selection_changed)
        self.setMinimumHeight(200)

    def mousePressEvent(self, event):
        """Handle mouse press for selection mode with Ctrl key and right-click panning."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        elif event.button() == Qt.RightButton:
            # Right-click is for panning only - don't allow selection
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = event.__class__(event.type(), 
                                         event.localPos(), 
                                         Qt.LeftButton, 
                                         Qt.LeftButton, 
                                         event.modifiers())
            super().mousePressEvent(left_event)
            return
        elif event.button() == Qt.LeftButton:
            # Regular left-click without Ctrl - allow single selection
            super().mousePressEvent(event)
        else:
            # For any other button, ignore
            event.ignore()
            return

    def mouseReleaseEvent(self, event):
        """Handle mouse release to revert to no drag mode."""
        if event.button() == Qt.RightButton:
            left_event = event.__class__(event.type(), 
                                         event.localPos(), 
                                         Qt.LeftButton, 
                                         Qt.LeftButton, 
                                         event.modifiers())
            super().mouseReleaseEvent(left_event)
            self.setDragMode(QGraphicsView.NoDrag)
            return
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.NoDrag)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for right-click panning."""
        if event.buttons() == Qt.RightButton:
            left_event = event.__class__(event.type(), 
                                         event.localPos(), 
                                         Qt.LeftButton, 
                                         Qt.LeftButton, 
                                         event.modifiers())
            super().mouseMoveEvent(left_event)
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        old_pos = self.mapToScene(event.pos())

        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def update_clusters(self, data_items):
        """Update the cluster visualization with new data.
        
        Args:
            data_items: List of AnnotationDataItem objects.
        """
        self.clear_points()
        
        colors = [QColor("cyan"), QColor("red"), QColor("green"), QColor("blue"), 
                  QColor("orange"), QColor("purple"), QColor("brown"), QColor("pink")]
        
        for item in data_items:
            cluster_color = colors[item.cluster_id % len(colors)]
            
            # *** The only change is here: Use the new ClusterPointItem ***
            point = ClusterPointItem(0, 0, POINT_SIZE, POINT_SIZE)
            point.setPos(item.cluster_x, item.cluster_y)
            
            point.setBrush(QBrush(cluster_color))
            point.setPen(QPen(QColor("black"), POINT_WIDTH))  # Increased from 0.5 to 1.5
            
            point.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            point.setFlag(QGraphicsItem.ItemIsSelectable)
            
            # This is the crucial link: store the shared AnnotationDataItem
            point.setData(0, item)
            
            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point

    def clear_points(self):
        """Clear all cluster points from the scene."""
        for point in self.points_by_id.values():
            self.graphics_scene.removeItem(point)
        self.points_by_id.clear()

    def on_selection_changed(self):
        """Handle point selection changes and emit a signal to the controller."""
        selected_items = self.graphics_scene.selectedItems()
        current_selection_ids = {item.data(0).annotation.id for item in selected_items}

        # If the selection has actually changed, update the model and emit
        if current_selection_ids != self.previous_selection_ids:
            # Update the central model (the AnnotationDataItem) for all points
            for point_id, point in self.points_by_id.items():
                is_selected = point_id in current_selection_ids
                point.data(0).set_selected(is_selected)

            # Emit the complete list of currently selected IDs
            self.selection_changed.emit(list(current_selection_ids))
            self.previous_selection_ids = current_selection_ids

        # Handle local animation
        self.animation_timer.stop()
        for point in self.points_by_id.values():
            if not point.isSelected():
                point.setPen(QPen(QColor("black"), POINT_WIDTH))
        
        if selected_items:
            self.animation_timer.start()
            
    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of points based on a set of IDs from the controller."""
        # Block this scene's own selectionChanged signal to prevent an infinite loop
        blocker = QSignalBlocker(self.graphics_scene)
        
        for ann_id, point in self.points_by_id.items():
            point.setSelected(ann_id in selected_ids)
            
        self.previous_selection_ids = selected_ids
        
        # Trigger animation update
        self.on_selection_changed()

    def animate_selection(self):
        """Animate selected points with marching ants effect using darker versions of point colors."""
        self.animation_offset = (self.animation_offset + 1) % 20
        
        # This logic remains the same. It applies the custom pen to the selected items.
        # Because the items are ClusterPointItem, the default selection box won't be drawn.
        for item in self.graphics_scene.selectedItems():
            original_color = item.brush().color()
            darker_color = original_color.darker(150)

            animated_pen = QPen(darker_color, POINT_WIDTH)
            animated_pen.setStyle(Qt.CustomDashLine)
            animated_pen.setDashPattern([1, 2])
            animated_pen.setDashOffset(self.animation_offset)
            
            item.setPen(animated_pen)

    def fit_view_to_points(self):
        """Fit the view to show all cluster points."""
        if self.points_by_id:
            self.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)


class AnnotationViewer(QScrollArea):
    """Scrollable grid widget for displaying annotation image crops with selection support."""
    
    # Define signals to report changes to the ExplorerWindow
    selection_changed = pyqtSignal(list)  # list of changed annotation IDs
    preview_changed = pyqtSignal(list)   # list of annotation IDs with new previews
    
    def __init__(self, parent=None):
        super(AnnotationViewer, self).__init__(parent)
        self.annotation_widgets_by_id = {}
        self.selected_widgets = []
        self.last_selected_index = -1
        self.current_widget_size = 128
        
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        self.mouse_pressed_on_widget = False
        
        self.preview_label_assignments = {}
        self.original_label_assignments = {}
        
        self.setup_ui()

    def setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        header_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        header_layout.addWidget(size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(256)
        self.size_slider.setValue(128)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(32)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        header_layout.addWidget(self.size_slider)

        self.size_value_label = QLabel("128")
        self.size_value_label.setMinimumWidth(30)
        header_layout.addWidget(self.size_value_label)
        
        main_layout.addLayout(header_layout)
        
        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setSpacing(5)

        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setWidget(self.content_widget)
        
        main_layout.addWidget(content_scroll)
        self.setWidget(main_container)

    def resizeEvent(self, event):
        """Handle resize events to recalculate grid layout."""
        super().resizeEvent(event)
        if hasattr(self, 'annotation_widgets_by_id') and self.annotation_widgets_by_id:
            self.recalculate_grid_layout()

    def mousePressEvent(self, event):
        """Handle mouse press for starting rubber band selection."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.rubber_band_origin = event.pos()
            self.mouse_pressed_on_widget = False
            child_widget = self.childAt(event.pos())
            if child_widget:
                widget = child_widget
                while widget and widget != self:
                    if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                        self.mouse_pressed_on_widget = True
                        break
                    widget = widget.parent()
            super().mousePressEvent(event)
            return
        elif event.button() == Qt.RightButton:
            # Ignore right mouse button clicks entirely
            event.ignore()
            return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for rubber band selection."""
        if (self.rubber_band_origin is not None and 
            event.buttons() == Qt.LeftButton and 
            event.modifiers() == Qt.ControlModifier):
            
            distance = (event.pos() - self.rubber_band_origin).manhattanLength()
            
            if distance > self.drag_threshold and not self.mouse_pressed_on_widget:
                if not self.rubber_band:
                    from PyQt5.QtWidgets import QRubberBand
                    self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
                    self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
                    self.rubber_band.show()
                rect = QRect(self.rubber_band_origin, event.pos()).normalized()
                self.rubber_band.setGeometry(rect)
                event.accept()
                return
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete rubber band selection."""
        if (self.rubber_band_origin is not None and 
            event.button() == Qt.LeftButton and 
            event.modifiers() == Qt.ControlModifier):
            
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                selection_rect = self.rubber_band.geometry()
                content_widget = self.content_widget
                
                last_selected_in_rubber_band = -1
                # Use list() to avoid issues with dict size changes during iteration
                widget_list = list(self.annotation_widgets_by_id.values())
                for i, widget in enumerate(widget_list):
                    widget_rect_in_content = widget.geometry()
                    widget_rect_in_viewport = QRect(
                        content_widget.mapTo(self.viewport(), widget_rect_in_content.topLeft()),
                        widget_rect_in_content.size()
                    )

                    if selection_rect.intersects(widget_rect_in_viewport):
                        if not widget.is_selected():
                            self.select_widget(widget)
                        last_selected_in_rubber_band = i

                if last_selected_in_rubber_band != -1:
                    self.last_selected_index = last_selected_in_rubber_band

                self.rubber_band.deleteLater()
                self.rubber_band = None
                event.accept()
            else:
                super().mouseReleaseEvent(event)

            self.rubber_band_origin = None
            self.mouse_pressed_on_widget = False
            return
        super().mouseReleaseEvent(event)
            
    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
        if value % 2 != 0:
            value -= 1
        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        
        for widget in self.annotation_widgets_by_id.values():
            widget.update_size(value)
        self.recalculate_grid_layout()

    def recalculate_grid_layout(self):
        """Recalculate the grid layout based on current widget width."""
        if not self.annotation_widgets_by_id:
            return
            
        available_width = self.viewport().width() - 20
        widget_width = self.current_widget_size + self.grid_layout.spacing()
        cols = max(1, available_width // widget_width)
        
        for i, widget in enumerate(self.annotation_widgets_by_id.values()):
            self.grid_layout.addWidget(widget, i // cols, i % cols)

    def update_annotations(self, data_items):
        """Update the displayed annotations from a list of AnnotationDataItems."""
        for widget in self.annotation_widgets_by_id.values():
            widget.deleteLater()
        self.annotation_widgets_by_id.clear()
        self.selected_widgets.clear()
        self.last_selected_index = -1

        for data_item in data_items:
            annotation_widget = AnnotationImageWidget(
                data_item, self.current_widget_size, 
                annotation_viewer=self)
            self.annotation_widgets_by_id[data_item.annotation.id] = annotation_widget
        
        self.recalculate_grid_layout()

    def handle_annotation_selection(self, widget, event):
        """Handle selection of annotation widgets with different modes."""
        widget_list = list(self.annotation_widgets_by_id.values())
        try:
            widget_index = widget_list.index(widget)
        except ValueError:
            return

        modifiers = event.modifiers()
        changed_ids = []

        # --- The selection logic now identifies which items to change   ---
        # --- but the core state change happens in select/deselect       ---
        
        if modifiers == Qt.ShiftModifier or modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            # Range selection
            if self.last_selected_index != -1:
                start = min(self.last_selected_index, widget_index)
                end = max(self.last_selected_index, widget_index)
                for i in range(start, end + 1):
                    # select_widget will return True if a change occurred
                    if self.select_widget(widget_list[i]):
                        changed_ids.append(widget_list[i].data_item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
                self.last_selected_index = widget_index
        
        elif modifiers == Qt.ControlModifier:
            # Toggle selection
            if widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            self.last_selected_index = widget_index
                
        else:
            # Normal click: clear all others and select this one
            newly_selected_id = widget.data_item.annotation.id
            # Deselect all widgets that are not the clicked one
            for w in list(self.selected_widgets):
                if w.data_item.annotation.id != newly_selected_id:
                    if self.deselect_widget(w):
                        changed_ids.append(w.data_item.annotation.id)
            # Select the clicked widget
            if self.select_widget(widget):
                changed_ids.append(newly_selected_id)
            self.last_selected_index = widget_index
        
        # If any selections were changed, emit the signal
        if changed_ids:
            self.selection_changed.emit(changed_ids)

    def select_widget(self, widget):
        """Select a widget, update the data_item, and return True if state changed."""
        if not widget.is_selected():
            widget.set_selected(True) # This updates visuals
            widget.data_item.set_selected(True) # This updates the model
            self.selected_widgets.append(widget)
            self.update_label_window_selection()
            return True
        return False

    def deselect_widget(self, widget):
        """Deselect a widget, update the data_item, and return True if state changed."""
        if widget.is_selected():
            widget.set_selected(False)
            widget.data_item.set_selected(False)
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            self.update_label_window_selection()
            return True
        return False

    def clear_selection(self):
        """Clear all selected widgets."""
        for widget in list(self.selected_widgets):
            widget.set_selected(False)
        self.selected_widgets.clear()
        self.update_label_window_selection()

    def update_label_window_selection(self):
        """Update the label window selection based on currently selected annotations."""
        explorer_window = self.parent()
        while explorer_window and not hasattr(explorer_window, 'main_window'):
            explorer_window = explorer_window.parent()
            
        if not explorer_window or not hasattr(explorer_window, 'main_window'):
            return
            
        main_window = explorer_window.main_window
        label_window = main_window.label_window
        annotation_window = main_window.annotation_window
        
        if not self.selected_widgets:
            label_window.deselect_active_label()
            label_window.update_annotation_count()
            return
            
        selected_data_items = [widget.data_item for widget in self.selected_widgets]
        
        first_effective_label = selected_data_items[0].effective_label
        all_same_current_label = all(item.effective_label.id == first_effective_label.id for item in selected_data_items)
        
        if all_same_current_label:
            label_window.set_active_label(first_effective_label)
            if not selected_data_items[0].has_preview_changes():
                annotation_window.labelSelected.emit(first_effective_label.id)
        else:
            label_window.deselect_active_label()
        
        label_window.update_annotation_count()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]
    
    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of widgets based on a set of IDs from the controller."""
        # Block signals temporarily to prevent cascade updates
        self.setUpdatesEnabled(False)
        
        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in selected_ids
                widget.set_selected(is_selected)
            
            # Resync internal list of selected widgets
            self.selected_widgets = [w for w in self.annotation_widgets_by_id.values() if w.is_selected()]
            
        finally:
            self.setUpdatesEnabled(True)
        
        # Update label window once at the end
        self.update_label_window_selection()
    
    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label and emit a signal for the cluster view to update."""
        if not self.selected_widgets or not preview_label:
            return

        changed_ids = []
        for widget in self.selected_widgets:
            widget.data_item.set_preview_label(preview_label)
            widget.update() # Force repaint with new color
            changed_ids.append(widget.data_item.annotation.id)

        if changed_ids:
            self.preview_changed.emit(changed_ids)

    def clear_preview_states(self):
        """Clear all preview states and revert to original labels."""
        # We just need to iterate through all widgets and tell their data_items to clear
        something_cleared = False
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.has_preview_changes():
                widget.data_item.clear_preview_label()
                widget.update() # Repaint to show original color
                something_cleared = True
        
        if something_cleared:
            self.update_label_window_selection()

    def has_preview_changes(self):
        """Check if there are any pending preview changes."""
        return any(w.data_item.has_preview_changes() for w in self.annotation_widgets_by_id.values())

    def get_preview_changes_summary(self):
        """Get a summary of preview changes for user feedback."""
        change_count = sum(1 for w in self.annotation_widgets_by_id.values() if w.data_item.has_preview_changes())
        if not change_count:
            return "No preview changes"
        return f"{change_count} annotation(s) with preview changes"

    def apply_preview_changes_permanently(self):
        """Apply all preview changes permanently to the annotation data."""
        applied_annotations = []
        for widget in self.annotation_widgets_by_id.values():
            # Tell the data_item to apply its changes to the underlying annotation
            if widget.data_item.apply_preview_permanently():
                applied_annotations.append(widget.annotation)
        
        return applied_annotations


# ----------------------------------------------------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationSettingsWidget(QGroupBox):
    """Widget containing all filter annotation conditions in a multi-column layout."""

    def __init__(self, main_window, parent=None):
        super(AnnotationSettingsWidget, self).__init__("Annotation Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent  # Store reference to ExplorerWindow
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Main conditions layout - horizontal with vertical columns
        conditions_layout = QHBoxLayout()

        # Images column
        images_column = QVBoxLayout()
        images_label = QLabel("Images:")
        images_label.setStyleSheet("font-weight: bold;")
        images_column.addWidget(images_label)
        
        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.MultiSelection)
        self.images_list.setMaximumHeight(100)
        
        # Add available images (no "All" item)
        if hasattr(self.main_window, 'image_window') and hasattr(self.main_window.image_window, 'raster_manager'):
            for path in self.main_window.image_window.raster_manager.image_paths:
                self.images_list.addItem(os.path.basename(path))
        
        images_column.addWidget(self.images_list)
        
        # Images selection buttons at bottom
        images_buttons_layout = QHBoxLayout()
        self.images_select_all_btn = QPushButton("Select All")
        self.images_select_all_btn.clicked.connect(self.select_all_images)
        images_buttons_layout.addWidget(self.images_select_all_btn)
        
        self.images_deselect_all_btn = QPushButton("Deselect All")
        self.images_deselect_all_btn.clicked.connect(self.deselect_all_images)
        images_buttons_layout.addWidget(self.images_deselect_all_btn)
        images_column.addLayout(images_buttons_layout)
        
        conditions_layout.addLayout(images_column)

        # Annotation Type column
        type_column = QVBoxLayout()
        type_label = QLabel("Annotation Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_column.addWidget(type_label)
        
        self.annotation_type_list = QListWidget()
        self.annotation_type_list.setSelectionMode(QListWidget.MultiSelection)
        self.annotation_type_list.setMaximumHeight(100)
        self.annotation_type_list.addItems(["PatchAnnotation", 
                                            "RectangleAnnotation", 
                                            "PolygonAnnotation", 
                                            "MultiPolygonAnnotation"])
        
        type_column.addWidget(self.annotation_type_list)
        
        # Annotation type selection buttons at bottom
        type_buttons_layout = QHBoxLayout()
        self.type_select_all_btn = QPushButton("Select All")
        self.type_select_all_btn.clicked.connect(self.select_all_annotation_types)
        type_buttons_layout.addWidget(self.type_select_all_btn)
        
        self.type_deselect_all_btn = QPushButton("Deselect All")
        self.type_deselect_all_btn.clicked.connect(self.deselect_all_annotation_types)
        type_buttons_layout.addWidget(self.type_deselect_all_btn)
        type_column.addLayout(type_buttons_layout)
        
        conditions_layout.addLayout(type_column)

        # Label column
        label_column = QVBoxLayout()
        label_label = QLabel("Label:")
        label_label.setStyleSheet("font-weight: bold;")
        label_column.addWidget(label_label)
        
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.MultiSelection)
        self.label_list.setMaximumHeight(100)
        
        # Add available labels (no "All" item)
        if hasattr(self.main_window, 'label_window') and hasattr(self.main_window.label_window, 'labels'):
            for label in self.main_window.label_window.labels:
                self.label_list.addItem(label.short_label_code)
        
        label_column.addWidget(self.label_list)
        
        # Label selection buttons at bottom
        label_buttons_layout = QHBoxLayout()
        self.label_select_all_btn = QPushButton("Select All")
        self.label_select_all_btn.clicked.connect(self.select_all_labels)
        label_buttons_layout.addWidget(self.label_select_all_btn)
        
        self.label_deselect_all_btn = QPushButton("Deselect All")
        self.label_deselect_all_btn.clicked.connect(self.deselect_all_labels)
        label_buttons_layout.addWidget(self.label_deselect_all_btn)
        label_column.addLayout(label_buttons_layout)
        
        conditions_layout.addLayout(label_column)

        # TopK column
        topk_column = QVBoxLayout()
        topk_label = QLabel("TopK:")
        topk_label.setStyleSheet("font-weight: bold;")
        topk_column.addWidget(topk_label)
        
        self.topk_combo = QComboBox()
        self.topk_combo.addItems(["Top1", "Top2", "Top3", "Top4", "Top5"])
        self.topk_combo.setCurrentText("Top1")
        
        topk_column.addWidget(self.topk_combo)
        topk_column.addStretch()  # Add stretch to align with other columns
        conditions_layout.addLayout(topk_column)

        # Confidence column
        confidence_column = QVBoxLayout()
        confidence_label = QLabel("Confidence:")
        confidence_label.setStyleSheet("font-weight: bold;")
        confidence_column.addWidget(confidence_label)
        
        self.confidence_operator_combo = QComboBox()
        self.confidence_operator_combo.addItems([">", "<", ">=", "<=", "==", "!="])
        self.confidence_operator_combo.setCurrentText(">=")
        confidence_column.addWidget(self.confidence_operator_combo)
        
        self.confidence_value_spin = QDoubleSpinBox()
        self.confidence_value_spin.setRange(0.0, 1.0)
        self.confidence_value_spin.setSingleStep(0.1)
        self.confidence_value_spin.setDecimals(2)
        self.confidence_value_spin.setValue(0.5)
        confidence_column.addWidget(self.confidence_value_spin)
        
        confidence_column.addStretch()  # Add stretch to align with other columns
        conditions_layout.addLayout(confidence_column)

        layout.addLayout(conditions_layout)

        # Bottom buttons layout with Apply and Clear buttons on the right
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()  # Push buttons to the right
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_conditions)
        bottom_layout.addWidget(self.apply_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all_conditions)
        bottom_layout.addWidget(self.clear_button)

        layout.addLayout(bottom_layout)

        # Set defaults
        self.set_defaults()

    def select_all_images(self):
        """Select all items in the images list."""
        for i in range(self.images_list.count()):
            self.images_list.item(i).setSelected(True)

    def deselect_all_images(self):
        """Deselect all items in the images list."""
        self.images_list.clearSelection()

    def select_all_annotation_types(self):
        """Select all items in the annotation types list."""
        for i in range(self.annotation_type_list.count()):
            self.annotation_type_list.item(i).setSelected(True)

    def deselect_all_annotation_types(self):
        """Deselect all items in the annotation types list."""
        self.annotation_type_list.clearSelection()

    def select_all_labels(self):
        """Select all items in the labels list."""
        for i in range(self.label_list.count()):
            self.label_list.item(i).setSelected(True)

    def deselect_all_labels(self):
        """Deselect all items in the labels list."""
        self.label_list.clearSelection()

    def set_defaults(self):
        """Set default selections."""
        # Set current image as default (not all images)
        self.set_default_to_current_image()
        
        # Set all annotation types as default
        self.select_all_annotation_types()
        
        # Set all labels as default
        self.select_all_labels()

    def set_default_to_current_image(self):
        """Set the current image as the default selection."""
        if hasattr(self.main_window, 'annotation_window'):
            current_image_path = self.main_window.annotation_window.current_image_path
            if current_image_path:
                current_image_name = os.path.basename(current_image_path)
                # Find and select the current image
                for i in range(self.images_list.count()):
                    item = self.images_list.item(i)
                    if item.text() == current_image_name:
                        item.setSelected(True)
                        return
        
        # Fallback to selecting all images if current image not found
        self.select_all_images()

    def clear_all_conditions(self):
        """Reset all conditions to their defaults."""
        # Clear all selections
        self.images_list.clearSelection()
        self.annotation_type_list.clearSelection()
        self.label_list.clearSelection()
        
        # Reset to defaults
        self.set_defaults()
        
        # Reset TopK and Confidence to defaults
        self.topk_combo.setCurrentText("Top1")
        self.confidence_operator_combo.setCurrentText(">=")
        self.confidence_value_spin.setValue(0.5)
        
        # Auto-refresh on clear to show default results
        if self.explorer_window and hasattr(self.explorer_window, 'refresh_filters'):
            self.explorer_window.refresh_filters()

    def apply_conditions(self):
        """Apply the current filter conditions."""
        if self.explorer_window and hasattr(self.explorer_window, 'refresh_filters'):
            self.explorer_window.refresh_filters()

    def get_selected_images(self):
        """Get selected image names."""
        selected_items = self.images_list.selectedItems()
        if not selected_items:
            # If nothing selected, return empty list (no filtering)
            return []
        
        return [item.text() for item in selected_items]

    def get_single_selected_image_path(self):
        """Get the full path of the single selected image, or None if multiple/none selected."""
        selected_images = self.get_selected_images()
        if len(selected_images) == 1:
            # Find the full path for this image name
            image_name = selected_images[0]
            if hasattr(self.main_window, 'image_window') and hasattr(self.main_window.image_window, 'raster_manager'):
                for path in self.main_window.image_window.raster_manager.image_paths:
                    if os.path.basename(path) == image_name:
                        return path
        return None

    def refresh_filters(self):
        """Refresh the display based on current filter conditions."""
        # Check if only one image is selected and load it in annotation window
        single_image_path = self.get_single_selected_image_path()
        if single_image_path and hasattr(self.main_window, 'image_window'):
            # Load the single selected image in the annotation window
            self.main_window.image_window.load_image_by_path(single_image_path)
        
        # Get filtered annotations
        filtered_annotations = self.get_filtered_annotations()

        # Update annotation viewer
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.update_annotations(filtered_annotations)

    def get_selected_annotation_types(self):
        """Get selected annotation types."""
        selected_items = self.annotation_type_list.selectedItems()
        if not selected_items:
            # If nothing selected, return empty list (no filtering)
            return []
        
        return [item.text() for item in selected_items]

    def get_selected_labels(self):
        """Get selected labels."""
        selected_items = self.label_list.selectedItems()
        if not selected_items:
            # If nothing selected, return empty list (no filtering)
            return []
        
        return [item.text() for item in selected_items]

    def get_topk_selection(self):
        """Get TopK selection."""
        return self.topk_combo.currentText()

    def get_confidence_condition(self):
        """Get confidence operator and value."""
        operator = self.confidence_operator_combo.currentText()
        value = self.confidence_value_spin.value()
        return operator, value
    
    
class ClusterSettingsWidget(QGroupBox):
    """Widget containing settings with tabs for models and clustering."""

    def __init__(self, main_window, parent=None):
        super(ClusterSettingsWidget, self).__init__("Cluster Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent
        self.loaded_model = None
        self.model_path = ""
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Model selection dropdown (editable) - at the top
       # Model selection dropdown updated for feature selection
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False) # Prevent custom text
        self.model_combo.addItems([
            "Simple Color (Mean RGB)",
            "yolov8n.pt", 
            "yolov8s.pt", 
            "yolov8m.pt",
            "yolov8l.pt", 
            "yolov8x.pt"
        ])
        self.model_combo.setCurrentIndex(0)  # Default to simple color
        form_layout.addRow("Feature Model:", self.model_combo)

        # Cluster technique dropdown
        self.cluster_technique_combo = QComboBox()
        self.cluster_technique_combo.addItems(["TSNE", "UMAP"])
        form_layout.addRow("Technique:", self.cluster_technique_combo)

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(5)
        form_layout.addRow("Number of Clusters:", self.n_clusters_spin)

        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 1000)
        self.random_state_spin.setValue(42)
        form_layout.addRow("Random State:", self.random_state_spin)

        # Apply clustering button
        self.apply_cluster_button = QPushButton("Apply Clustering")
        self.apply_cluster_button.clicked.connect(self.apply_clustering)
        form_layout.addRow("", self.apply_cluster_button)

        layout.addLayout(form_layout)

    def apply_clustering(self):
        """Apply clustering with the current settings."""
        # This button now just triggers a full refresh, which includes clustering.
        # The main logic is in the ExplorerWindow.refresh_filters() method.
        if self.explorer_window:
            self.explorer_window.refresh_filters()
    

# ----------------------------------------------------------------------------------------------------------------------
# ExplorerWindow
# ----------------------------------------------------------------------------------------------------------------------


class ExplorerWindow(QMainWindow):
    def __init__(self, main_window, parent=None):
        super(ExplorerWindow, self).__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.model_path = ""
        self.loaded_model = None

        self.setWindowTitle("Explorer")
        # Set the window icon
        explorer_icon_path = get_icon("magic.png")
        self.setWindowIcon(QIcon(explorer_icon_path))

        # Create a central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)      
        # Create a left panel widget and layout for the re-parented LabelWindow
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Create widgets in __init__ so they're always available
        self.annotation_settings_widget = AnnotationSettingsWidget(self.main_window, self)
        self.cluster_settings_widget = ClusterSettingsWidget(self.main_window, self)
        self.annotation_viewer = AnnotationViewer(self)  # Pass self as parent
        self.cluster_viewer = ClusterViewer(self)
        
        # Create buttons
        self.clear_preview_button = QPushButton('Clear Preview', self)
        self.clear_preview_button.clicked.connect(self.clear_preview_changes)
        self.clear_preview_button.setToolTip("Clear all preview changes and revert to original labels")
        self.clear_preview_button.setEnabled(False)  # Initially disabled

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setToolTip("Close the window")

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply)
        self.apply_button.setToolTip("Apply changes")
        self.apply_button.setEnabled(False)  # Initially disabled

    def showEvent(self, event):
        self.setup_ui()
        super(ExplorerWindow, self).showEvent(event)

    def closeEvent(self, event):
        # Clear any preview states before closing
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.clear_preview_states()
        
        # Re-enable the main window before closing
        if self.main_window:
            self.main_window.setEnabled(True)
        
        # Move the label_window back to the main window
        if hasattr(self.main_window, 'explorer_closed'):
            self.main_window.explorer_closed()
        
        # Clear the reference in the main window
        self.main_window.explorer_window = None
        event.accept()

    def setup_ui(self):
        # Clear the main layout to remove any existing widgets
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)  # Remove from layout but don't delete

        # Top section: Conditions and Settings side by side
        top_layout = QHBoxLayout()
        
        # Add existing widgets to layout
        top_layout.addWidget(self.annotation_settings_widget, 2)  # Give more space to conditions
        top_layout.addWidget(self.cluster_settings_widget, 1)  # Less space for settings
        
        # Create container widget for top layout
        top_container = QWidget()
        top_container.setLayout(top_layout)
        self.main_layout.addWidget(top_container)

        # Middle section: Annotation Viewer (left) and Cluster Viewer (right)
        middle_splitter = QSplitter(Qt.Horizontal)
        
        # Wrap annotation viewer in a group box
        annotation_group = QGroupBox("Annotation Viewer")
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.addWidget(self.annotation_viewer)
        middle_splitter.addWidget(annotation_group)
        
        # Wrap cluster viewer in a group box
        cluster_group = QGroupBox("Cluster Viewer")
        cluster_layout = QVBoxLayout(cluster_group)
        cluster_layout.addWidget(self.cluster_viewer)
        middle_splitter.addWidget(cluster_group)

        # Set splitter proportions (annotation viewer wider)
        middle_splitter.setSizes([700, 300])
        
        # Add middle section to main layout with stretch factor
        self.main_layout.addWidget(middle_splitter, 1)
        
        # Note: LabelWindow will be re-parented here by MainWindow.open_explorer_window()
        # The LabelWindow will be added to self.left_layout at index 1 by the MainWindow
        self.main_layout.addWidget(self.label_window)
        
        # Bottom control buttons
        self.buttons_layout = QHBoxLayout()
        # Add stretch to push buttons to the right
        self.buttons_layout.addStretch(1)

        # Add existing buttons to layout
        self.buttons_layout.addWidget(self.clear_preview_button)
        self.buttons_layout.addWidget(self.exit_button)
        self.buttons_layout.addWidget(self.apply_button)

        self.main_layout.addLayout(self.buttons_layout)

        # Set default condition to current image and refresh filters
        self.annotation_settings_widget.set_default_to_current_image()
        self.refresh_filters()
        
        # Connect label selection to preview updates (only connect once)
        try:
            self.label_window.labelSelected.disconnect(self.on_label_selected_for_preview)
        except TypeError:
            pass  # Signal wasn't connected yet
        
        self.label_window.labelSelected.connect(self.on_label_selected_for_preview)
        self.annotation_viewer.selection_changed.connect(self.on_annotation_view_selection_changed)
        self.annotation_viewer.preview_changed.connect(self.on_preview_changed)
        self.cluster_viewer.selection_changed.connect(self.on_cluster_view_selection_changed)
        
    @pyqtSlot(list)
    def on_annotation_view_selection_changed(self, changed_ann_ids):
        """A selection was made in the AnnotationViewer, so update the ClusterViewer."""
        print(f"Syncing selection from Annotation View to Cluster View for {len(changed_ann_ids)} items.")
        all_selected_ids = {w.data_item.annotation.id for w in self.annotation_viewer.selected_widgets}
        self.cluster_viewer.render_selection_from_ids(all_selected_ids)
        self.update_label_window_selection() # Keep label window in sync

    @pyqtSlot(list)
    def on_cluster_view_selection_changed(self, all_selected_ann_ids):
        """A selection was made in the ClusterViewer, so update the AnnotationViewer."""
        print(f"Syncing selection from Cluster View to Annotation View for {len(all_selected_ann_ids)} items.")
        self.annotation_viewer.render_selection_from_ids(set(all_selected_ann_ids))
        self.update_label_window_selection() # Keep label window in sync

    @pyqtSlot(list)
    def on_preview_changed(self, changed_ann_ids):
        """A preview color was changed in the AnnotationViewer, so update the ClusterViewer points."""
        print(f"Syncing preview color change for {len(changed_ann_ids)} items.")
        for ann_id in changed_ann_ids:
            point = self.cluster_viewer.points_by_id.get(ann_id)
            if point:
                point.update()  # Force the point to repaint itself

    def update_label_window_selection(self):
        """Update the label window based on the selection in the annotation viewer."""
        # This logic can now be simpler as it just reads the state from the annotation_viewer
        self.annotation_viewer.update_label_window_selection()

    def get_filtered_data_items(self):
        """Get annotations that match all conditions, returned as AnnotationDataItem objects."""
        data_items = []
        if not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return data_items

        # Get current filter conditions
        selected_images = self.annotation_settings_widget.get_selected_images()
        selected_types = self.annotation_settings_widget.get_selected_annotation_types()
        selected_labels = self.annotation_settings_widget.get_selected_labels()
        topk_selection = self.annotation_settings_widget.get_topk_selection()
        confidence_operator, confidence_value = self.annotation_settings_widget.get_confidence_condition()

        annotations_to_process = []
        for annotation in self.main_window.annotation_window.annotations_dict.values():
            annotation_matches = True

            # Check image condition - if empty list, no annotations match
            if selected_images:
                annotation_image = os.path.basename(annotation.image_path)
                if annotation_image not in selected_images:
                    annotation_matches = False
            else:
                # No images selected means no annotations should match
                annotation_matches = False

            # Check annotation type condition - if empty list, no annotations match
            if annotation_matches:
                if selected_types:
                    annotation_type = type(annotation).__name__
                    if annotation_type not in selected_types:
                        annotation_matches = False
                else:
                    # No types selected means no annotations should match
                    annotation_matches = False

            # Check label condition - if empty list, no annotations match
            if annotation_matches:
                if selected_labels:
                    annotation_label = annotation.label.short_label_code
                    if annotation_label not in selected_labels:
                        annotation_matches = False
                else:
                    # No labels selected means no annotations should match
                    annotation_matches = False

            # Check TopK condition using machine_confidence
            if annotation_matches and hasattr(annotation, 'machine_confidence') and annotation.machine_confidence:
                topk_num = int(topk_selection.replace("Top", ""))
                # Get sorted confidence values (already sorted in descending order)
                confidence_list = list(annotation.machine_confidence.values())
                # Check if the current label is within the TopK predictions
                if len(confidence_list) < topk_num:
                    # If we don't have enough predictions for the requested TopK, exclude
                    annotation_matches = False
                else:
                    # Check if current label is in the top K predictions
                    sorted_labels = list(annotation.machine_confidence.keys())
                    current_label_in_topk = False
                    for i in range(min(topk_num, len(sorted_labels))):
                        if sorted_labels[i].short_label_code == annotation.label.short_label_code:
                            current_label_in_topk = True
                            break
                    if not current_label_in_topk:
                        annotation_matches = False

            # Check confidence condition
            if annotation_matches:
                conf_value = None
                
                # Get confidence value based on verification status
                if annotation.verified and hasattr(annotation, 'user_confidence') and annotation.user_confidence:
                    # For verified annotations, use user_confidence (Top1)
                    conf_value = list(annotation.user_confidence.values())[0]
                elif hasattr(annotation, 'machine_confidence') and annotation.machine_confidence:
                    # For unverified annotations, use the top machine confidence (Top1)
                    conf_value = list(annotation.machine_confidence.values())[0]
                
                # Apply confidence filter if we have a confidence value
                if conf_value is not None:
                    conf_value = float(conf_value)
                    if confidence_operator == ">":
                        if not (conf_value > confidence_value):
                            annotation_matches = False
                    elif confidence_operator == "<":
                        if not (conf_value < confidence_value):
                            annotation_matches = False
                    elif confidence_operator == "==":
                        if not (abs(conf_value - confidence_value) < 1e-6):
                            annotation_matches = False
                    elif confidence_operator == ">=":
                        if not (conf_value >= confidence_value):
                            annotation_matches = False
                    elif confidence_operator == "<=":
                        if not (conf_value <= confidence_value):
                            annotation_matches = False
                    elif confidence_operator == "!=":
                        if not (abs(conf_value - confidence_value) >= 1e-6):
                            annotation_matches = False

            if annotation_matches:
                annotations_to_process.append(annotation)

        # Ensure all filtered annotations have cropped images
        self._ensure_cropped_images(annotations_to_process)
        
        # Wrap in AnnotationDataItem
        for ann in annotations_to_process:
            data_items.append(AnnotationDataItem(ann))

        return data_items
    
    def _ensure_cropped_images(self, annotations):
        """Ensure all provided annotations have a cropped image available."""
        annotations_by_image = {}
        for annotation in annotations:
            # Only process annotations that don't have a cropped image yet
            if not annotation.cropped_image:
                image_path = annotation.image_path
                if image_path not in annotations_by_image:
                    annotations_by_image[image_path] = []
                annotations_by_image[image_path].append(annotation)
        
        # Only proceed if there are annotations that actually need cropping
        if annotations_by_image:
            progress_bar = ProgressBar(self, "Cropping Image Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(annotations_by_image))
            
            try:
                # Crop annotations for each image using the AnnotationWindow method
                # This ensures consistency with how cropped images are generated elsewhere
                for image_path, image_annotations in annotations_by_image.items():
                    self.annotation_window.crop_annotations(
                        image_path=image_path, 
                        annotations=image_annotations, 
                        return_annotations=False,  # We don't need the return value
                        verbose=False
                    )
                    # Update progress bar
                    progress_bar.update_progress()
                    
            except Exception as e:
                print(f"Error cropping annotations: {e}")
                
            finally:
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()

    def _extract_rgb_features(self, data_items):
        """Extracts mean RGB color features from annotation crops."""
        print("Extracting features (mean RGB)...")
        features = []
        valid_data_items = []
        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                width, height = qimage.width(), qimage.height()
                
                ptr = qimage.bits()
                ptr.setsize(height * width * 3)
                arr = np.array(ptr).reshape((height, width, 3))
                
                mean_color = np.mean(arr, axis=(0, 1))
                features.append(mean_color)
                valid_data_items.append(item)
            else:
                print(f"Warning: Could not get cropped image for annotation ID {item.annotation.id}. Skipping.")

        return np.array(features), valid_data_items

    def _extract_yolo_features(self, data_items, model_name):
        """Placeholder for extracting features using a YOLO model."""
        print(f"Attempting to extract features with YOLO model: {model_name}")
        print("NOTE: YOLO feature extraction is not yet implemented.")
        # In a real implementation, you would:
        # 1. Load the specified YOLO model.
        # 2. Pre-process each cropped image into a tensor.
        # 3. Pass the tensor through the model's backbone.
        # 4. Get the resulting feature vector.
        # For now, we return empty results to prevent the app from crashing.
        return np.array([]), []

    def _extract_features(self, data_items):
        """
        Dispatcher method to call the appropriate feature extraction function
        based on the user's selection in the UI.
        """
        model_name = self.cluster_settings_widget.model_combo.currentText()

        if model_name == "Simple Color (Mean RGB)":
            return self._extract_rgb_features(data_items)
        elif ".pt" in model_name:  # Simple check for a YOLO model
            return self._extract_yolo_features(data_items, model_name)
        else:
            print(f"Unknown feature model selected: {model_name}")
            return np.array([]), []

    def _run_dimensionality_reduction(self, features, technique, random_state):
        """Runs UMAP or t-SNE on the feature matrix."""
        print(f"Running {technique} on {len(features)} items...")
        if len(features) <= 1:
            print("Not enough data points for dimensionality reduction.")
            return None

        try:
            if technique == "UMAP":
                reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, len(features)-1))
                return reducer.fit_transform(features)
            else:  # Default to TSNE
                reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(features)-1), n_init='auto')
                return reducer.fit_transform(features)
        except Exception as e:
            print(f"Error during {technique} dimensionality reduction: {e}")
            return None

    def _run_clustering(self, embedded_features, n_clusters, random_state):
        """Runs KMeans clustering on the embedded features."""
        print("Running KMeans clustering...")
        
        actual_n_clusters = min(n_clusters, len(embedded_features))
        if actual_n_clusters < 2:
            print("Not enough data for multiple clusters. Assigning all to cluster 0.")
            return np.zeros(len(embedded_features), dtype=int)
            
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=random_state, n_init=10)
        return kmeans.fit_predict(embedded_features)

    def _update_data_items(self, data_items, embedded_features, cluster_labels):
        """Updates AnnotationDataItem objects with cluster results."""
        scale_factor = 4000
        min_vals = np.min(embedded_features, axis=0)
        max_vals = np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals

        for i, item in enumerate(data_items):
            norm_x = (embedded_features[i, 0] - min_vals[0]) / range_vals[0] if range_vals[0] > 0 else 0.5
            norm_y = (embedded_features[i, 1] - min_vals[1]) / range_vals[1] if range_vals[1] > 0 else 0.5
            
            item.cluster_x = (norm_x * scale_factor) - (scale_factor / 2)
            item.cluster_y = (norm_y * scale_factor) - (scale_factor / 2)
            item.cluster_id = cluster_labels[i]

    def run_clustering_on_items(self, data_items):
        """Orchestrates the new, modular clustering pipeline."""
        if not data_items:
            print("No items to cluster.")
            return

        technique = self.cluster_settings_widget.cluster_technique_combo.currentText()
        if np is None or KMeans is None or (technique == 'TSNE' and TSNE is None) or (technique == 'UMAP' and UMAP is None):
            print(f"Warning: Required library for {technique} not installed.")
            return

        # 1. Extract Features (using the new dispatcher)
        features, valid_data_items = self._extract_features(data_items)
        if not valid_data_items:
            print("No valid features could be extracted. Aborting clustering.")
            # Clear cluster view if feature extraction fails
            self.cluster_viewer.clear_points()
            return

        # 2. Dimensionality Reduction
        n_clusters = self.cluster_settings_widget.n_clusters_spin.value()
        random_state = self.cluster_settings_widget.random_state_spin.value()
        embedded_features = self._run_dimensionality_reduction(features, technique, random_state)
        if embedded_features is None:
            return

        # 3. Clustering
        cluster_labels = self._run_clustering(embedded_features, n_clusters, random_state)
        
        # 4. Update Data Items with Results
        self._update_data_items(valid_data_items, embedded_features, cluster_labels)
        
        # Also update the items for which feature extraction failed
        all_ids = {item.annotation.id for item in valid_data_items}
        for item in data_items:
            if item.annotation.id not in all_ids:
                item.cluster_x, item.cluster_y, item.cluster_id = 0, 0, -1 # Assign to a null cluster
                
    def refresh_filters(self):
        """Refresh display: filter -> cluster -> update viewers."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data_items = self.get_filtered_data_items()
            self.run_clustering_on_items(data_items)
            self.annotation_viewer.update_annotations(data_items)
            self.cluster_viewer.update_clusters(data_items)
            self.cluster_viewer.fit_view_to_points()
        finally:
            QApplication.restoreOverrideCursor()

    def on_label_selected_for_preview(self, label):
        """Handle label selection to update preview state."""
        if hasattr(self, 'annotation_viewer') and self.annotation_viewer.selected_widgets:
            self.annotation_viewer.apply_preview_label_to_selected(label)
            self.update_button_states()

    def clear_preview_changes(self):
        """Clear all preview changes and revert to original labels."""
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.clear_preview_states()
            self.update_button_states()
            print("Cleared all preview changes")

    def update_button_states(self):
        """Update the state of Clear Preview and Apply buttons."""
        has_changes = (hasattr(self, 'annotation_viewer') and self.annotation_viewer.has_preview_changes())
        
        self.clear_preview_button.setEnabled(has_changes)
        self.apply_button.setEnabled(has_changes)
        
        summary = self.annotation_viewer.get_preview_changes_summary()
        self.clear_preview_button.setToolTip(f"Clear all preview changes - {summary}")
        self.apply_button.setToolTip(f"Apply changes - {summary}")

    def apply(self):
        """Apply any modifications to the actual annotations."""
        try:
            applied_annotations = self.annotation_viewer.apply_preview_changes_permanently()
            if applied_annotations:
                affected_images = {ann.image_path for ann in applied_annotations}
                for image_path in affected_images:
                    self.image_window.update_image_annotations(image_path)
                self.annotation_window.load_annotations()
                self.refresh_filters()
                self.annotation_viewer.clear_selection()
                self.update_button_states()
                print(f"Applied changes to {len(applied_annotations)} annotation(s)")
            else:
                print("No preview changes to apply")
        except Exception as e:
            print(f"Error applying modifications: {e}")