from coralnet_toolbox.Icons import get_icon
from PyQt5.QtGui import QIcon, QBrush, QPen, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize, QRect
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget, QGridLayout,
                             QMainWindow, QSplitter, QGroupBox, QFormLayout,
                             QSpinBox, QGraphicsEllipseItem, QGraphicsItem, QSlider)
import warnings
import os
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Interactive Graphics View for Cluster Visualization
# ----------------------------------------------------------------------------------------------------------------------


class InteractiveClusterView(QGraphicsView):
    """Custom QGraphicsView for interactive cluster visualization with zooming, panning, and selection."""
    
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)  # Make the points look smooth
        
        # Set the default interaction mode to panning
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Remove scrollbars for a cleaner look
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def mousePressEvent(self, event):
        """Handle mouse press for selection mode with Ctrl key and right-click panning."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # If Ctrl is pressed, switch to RubberBandDrag mode for selection
            self.setDragMode(QGraphicsView.RubberBandDrag)
        elif event.button() == Qt.RightButton:
            # Right mouse button for panning - force ScrollHandDrag mode
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Convert right-click to left-click for proper panning behavior
            left_event = event.__class__(
                event.type(),
                event.localPos(),
                Qt.LeftButton,  # Convert to left button
                Qt.LeftButton,  # Convert to left button
                event.modifiers()
            )
            super().mousePressEvent(left_event)
            return
        # Call the base class implementation to handle the event
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to revert to no drag mode."""
        if event.button() == Qt.RightButton:
            # Convert right-click release to left-click release for proper panning
            left_event = event.__class__(
                event.type(),
                event.localPos(),
                Qt.LeftButton,  # Convert to left button
                Qt.LeftButton,  # Convert to left button
                event.modifiers()
            )
            super().mouseReleaseEvent(left_event)
            self.setDragMode(QGraphicsView.NoDrag)
            return
        # Call the base class implementation first
        super().mouseReleaseEvent(event)
        # After the event is handled, revert to no drag mode for normal selection
        self.setDragMode(QGraphicsView.NoDrag)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for right-click panning."""
        if event.buttons() == Qt.RightButton:
            # Convert right-click move to left-click move for proper panning
            left_event = event.__class__(
                event.type(),
                event.localPos(),
                Qt.LeftButton,  # Convert to left button
                Qt.LeftButton,  # Convert to left button
                event.modifiers()
            )
            super().mouseMoveEvent(left_event)
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        # Zoom Factor
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        # Save the scene pos
        old_pos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        # Get the new position
        new_pos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConditionWidget(QWidget):
    """A single condition widget with Image, Annotation Type, and Label dropdowns."""

    def __init__(self, main_window, parent=None):
        super(ConditionWidget, self).__init__(parent)
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Image dropdown
        layout.addWidget(QLabel("Image:"))
        self.image_dropdown = QComboBox()
        self.image_dropdown.addItem("All")
        if hasattr(self.main_window, 'image_window') and hasattr(self.main_window.image_window, 'raster_manager'):
            self.image_dropdown.addItems([os.path.basename(
                path) for path in self.main_window.image_window.raster_manager.image_paths])
        layout.addWidget(self.image_dropdown)

        # Annotation Type dropdown
        layout.addWidget(QLabel("Type:"))
        self.annotation_dropdown = QComboBox()
        self.annotation_dropdown.addItems(
            ["All", "PatchAnnotation", "RectangleAnnotation", "PolygonAnnotation"])
        layout.addWidget(self.annotation_dropdown)

        # Label dropdown
        layout.addWidget(QLabel("Label:"))
        self.label_dropdown = QComboBox()
        self.label_dropdown.addItem("All")
        if hasattr(self.main_window, 'label_window') and hasattr(self.main_window.label_window, 'labels'):
            self.label_dropdown.addItems(
                [label.short_label_code for label in self.main_window.label_window.labels])
        layout.addWidget(self.label_dropdown)

        # Remove button
        self.remove_button = QPushButton("×")
        self.remove_button.setFixedSize(25, 25)
        self.remove_button.setStyleSheet(
            "QPushButton { color: red; font-weight: bold; }")
        layout.addWidget(self.remove_button)

        layout.addStretch()


class ConditionsWidget(QGroupBox):
    """Widget containing all conditions with add/remove functionality."""

    def __init__(self, main_window, parent=None):
        super(ConditionsWidget, self).__init__("Conditions", parent)
        self.main_window = main_window
        self.conditions = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Header with buttons
        header_layout = QHBoxLayout()
        
        # Add Condition button on the far left
        self.add_condition_button = QPushButton("Add Condition")
        self.add_condition_button.clicked.connect(self.add_condition)
        header_layout.addWidget(self.add_condition_button)
        
        # Stretch to push other buttons to the right
        header_layout.addStretch()
        
        # Apply and Clear buttons on the far right
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_conditions)
        header_layout.addWidget(self.apply_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all_conditions)
        header_layout.addWidget(self.clear_button)

        layout.addLayout(header_layout)

        # Scroll area for conditions
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(150)

        self.conditions_widget = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_widget)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area.setWidget(self.conditions_widget)
        layout.addWidget(self.scroll_area)

        # Add initial condition
        self.add_condition()

    def add_condition(self):
        condition = ConditionWidget(self.main_window, self)
        condition.remove_button.clicked.connect(
            lambda: self.remove_condition(condition))
        self.conditions.append(condition)
        self.conditions_layout.addWidget(condition)

        # Connect condition changes to refresh
        if hasattr(self.parent(), 'refresh_filters'):
            condition.image_dropdown.currentTextChanged.connect(
                self.parent().refresh_filters)
            condition.annotation_dropdown.currentTextChanged.connect(
                self.parent().refresh_filters)
            condition.label_dropdown.currentTextChanged.connect(
                self.parent().refresh_filters)

    def remove_condition(self, condition):
        if len(self.conditions) > 1:  # Keep at least one condition
            self.conditions.remove(condition)
            condition.deleteLater()
            # Refresh after removing condition
            if hasattr(self.parent(), 'refresh_filters'):
                self.parent().refresh_filters()

    def apply_conditions(self):
        """Apply the current filter conditions."""
        if hasattr(self.parent(), 'refresh_filters'):
            self.parent().refresh_filters()

    def clear_all_conditions(self):
        """Clear all conditions and add one default condition."""
        # Remove all conditions except keep at least one
        while len(self.conditions) > 1:
            condition = self.conditions[-1]
            self.conditions.remove(condition)
            condition.deleteLater()
        
        # Reset the remaining condition to defaults
        if self.conditions:
            condition = self.conditions[0]
            condition.image_dropdown.setCurrentText("All")
            condition.annotation_dropdown.setCurrentText("All")
            condition.label_dropdown.setCurrentText("All")
        
        # Refresh after clearing
        if hasattr(self.parent(), 'refresh_filters'):
            self.parent().refresh_filters()


class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, annotation, image_path, widget_size=256, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.annotation = annotation
        self.image_path = image_path
        self.annotation_viewer = annotation_viewer  # Direct reference to AnnotationViewerWidget
        self.selected = False
        self.widget_size = widget_size
        self.animation_offset = 0  # For marching ants animation
        self.setFixedSize(widget_size, widget_size)
        
        # Timer for marching ants animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)  # Update every 100ms
        
        self.setup_ui()
        self.load_annotation_image()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Image label (use full widget size minus margins)
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.widget_size - 4, self.widget_size - 4)
        
        # No default border - only show marching ants when selected
        self.image_label.setStyleSheet("border: none;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # Scale image to fit label

        layout.addWidget(self.image_label)

    def load_annotation_image(self):
        """Load and display the actual annotation cropped image."""
        try:
            # Get the cropped image graphic from the annotation
            cropped_pixmap = self.annotation.get_cropped_image_graphic()
            
            if cropped_pixmap and not cropped_pixmap.isNull():
                # Scale the image to fit the label while maintaining aspect ratio
                scaled_pixmap = cropped_pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                # Fallback to text if no image available
                self.image_label.setText("No Image\nAvailable")
                
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.image_label.setText("Error\nLoading Image")    
    
    def set_selected(self, selected):
        """Set the selection state and update visual appearance."""
        self.selected = selected
        if selected:
            # Start marching ants animation
            self.animation_timer.start()
        else:
            # Stop animation and remove border
            self.animation_timer.stop()
            self.image_label.setStyleSheet("border: none;")    
            
    def animate_selection(self):
        """Animate selected border with marching ants effect using black dashed lines."""
        # Update animation offset for marching ants (same as QtAnnotation)
        self.animation_offset = (self.animation_offset + 1) % 20  # Reset every 20 pixels
        
        # Create animated black dashed border similar to QtAnnotation
        # Use a custom dash pattern with offset for marching ants effect
        self.image_label.setStyleSheet("""
            border: 3px dashed black;
            border-image: none;
        """)

    def is_selected(self):
        """Return whether this widget is selected."""
        return self.selected

    def mousePressEvent(self, event):
        """Handle mouse press events for selection."""
        if event.button() == Qt.LeftButton:
            # Use direct reference to annotation viewer for selection handling
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                self.annotation_viewer.handle_annotation_selection(self, event)
        super().mousePressEvent(event)


class AnnotationViewerWidget(QWidget):
    """Scrollable grid widget for displaying annotation image crops."""
    
    def __init__(self, parent=None):
        super(AnnotationViewerWidget, self).__init__(parent)
        self.annotation_widgets = []
        self.selected_widgets = []
        self.last_selected_index = -1  # Anchor for shift-selection
        self.current_widget_size = 256  # Default size
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Annotation Viewer")
        header.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Size control layout
        size_layout = QHBoxLayout()
        
        # Size label
        size_label = QLabel("Size:")
        size_layout.addWidget(size_label)
        # Size slider
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(256)
        self.size_slider.setValue(256)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(32)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_slider)
        
        # Size value label
        self.size_value_label = QLabel("256")
        self.size_value_label.setMinimumWidth(30)
        size_layout.addWidget(self.size_value_label)
        
        layout.addLayout(size_layout)
        
        # Scroll area with selection support
        self.scroll_area = SelectableAnnotationViewer(self) # Pass self to scroll area
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setSpacing(5)

        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)

    def resizeEvent(self, event):
        """Handle resize events to recalculate grid layout."""
        super().resizeEvent(event)
        if hasattr(self, 'annotation_widgets') and self.annotation_widgets:
            self.recalculate_grid_layout()    
            
    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        
        for widget in self.annotation_widgets:
            widget.widget_size = value
            widget.setFixedSize(value, value)
            widget.image_label.setFixedSize(value - 4, value - 4)
        
        self.recalculate_grid_layout()

    def recalculate_grid_layout(self):
        """Recalculate the grid layout based on current widget width."""
        if not self.annotation_widgets:
            return
            
        available_width = self.scroll_area.viewport().width() - 20
        widget_width = self.current_widget_size + self.grid_layout.spacing()
        cols = max(1, available_width // widget_width)
        
        for i, widget in enumerate(self.annotation_widgets):
            self.grid_layout.addWidget(widget, i // cols, i % cols)

    def update_annotations(self, annotations):
        """Update the displayed annotations."""
        for widget in self.annotation_widgets:
            widget.deleteLater()
        self.annotation_widgets.clear()
        self.selected_widgets.clear()
        self.last_selected_index = -1

        for annotation in annotations:
            annotation_widget = AnnotationImageWidget(
                annotation, annotation.image_path, self.current_widget_size, 
                annotation_viewer=self)  # Pass self as annotation_viewer
            self.annotation_widgets.append(annotation_widget)
        
        self.recalculate_grid_layout()

    def handle_annotation_selection(self, widget, event):
        """Handle selection of annotation widgets with different modes."""
        try:
            widget_index = self.annotation_widgets.index(widget)
        except ValueError:
            return # Widget not in list

        modifiers = event.modifiers()

        if modifiers == Qt.ShiftModifier:
            # Shift+Click: Range selection
            if self.last_selected_index != -1:
                self.clear_selection()
                start = min(self.last_selected_index, widget_index)
                end = max(self.last_selected_index, widget_index)
                for i in range(start, end + 1):
                    self.select_widget(self.annotation_widgets[i])
            else:
                # If no anchor, treat as a single selection
                self.clear_selection()
                self.select_widget(widget)
                self.last_selected_index = widget_index

        elif modifiers == Qt.ControlModifier:
            # Ctrl+Click: Toggle selection
            if widget.is_selected():
                self.deselect_widget(widget)
            else:
                self.select_widget(widget)
            self.last_selected_index = widget_index
                
        else:
            # Normal click: Single selection
            # If the clicked widget is the only one selected, do nothing
            if len(self.selected_widgets) == 1 and self.selected_widgets[0] == widget:
                return

            self.clear_selection()
            self.select_widget(widget)
            self.last_selected_index = widget_index

    def select_widget(self, widget):
        """Select a widget and add it to the selection."""
        if widget not in self.selected_widgets:
            widget.set_selected(True)
            self.selected_widgets.append(widget)

    def deselect_widget(self, widget):
        """Deselect a widget and remove it from the selection."""
        if widget in self.selected_widgets:
            widget.set_selected(False)
            self.selected_widgets.remove(widget)

    def clear_selection(self):
        """Clear all selected widgets."""
        # Create a copy of the list to iterate over, as deselect_widget modifies it
        for widget in list(self.selected_widgets):
            self.deselect_widget(widget)
        self.selected_widgets.clear()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]


class SettingsWidget(QGroupBox):
    """Widget containing settings with tabs for models and clustering."""

    def __init__(self, main_window, parent=None):
        super(SettingsWidget, self).__init__("Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent
        self.loaded_model = None
        self.model_path = ""
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Model selection dropdown (editable) - at the top
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)  # Allow users to edit
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                  "yolov8l.pt", "yolov8x.pt"])
        form_layout.addRow("Model:", self.model_combo)

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
        # Get the current settings
        cluster_technique = self.cluster_technique_combo.currentText()
        n_clusters = self.n_clusters_spin.value()
        random_state = self.random_state_spin.value()
        
        # TODO: Implement actual clustering logic with real annotation features
        # For now, generate demo cluster data
        cluster_data = self.generate_demo_cluster_data(n_clusters, random_state)
        
        # Update the cluster viewer
        if hasattr(self.explorer_window, 'cluster_widget'):
            self.explorer_window.cluster_widget.update_clusters(cluster_data)
            self.explorer_window.cluster_widget.fit_view_to_points()
            
        self.show_status_message(f"{cluster_technique} clustering applied with {n_clusters} clusters.")

    def generate_demo_cluster_data(self, n_clusters, random_state):
        """Generate demonstration cluster data.
        
        Returns:
            List of tuples (x, y, cluster_id, annotation_data)
        """
        random.seed(random_state)
        cluster_data = []
        
        # Generate cluster centers
        centers = []
        for i in range(n_clusters):
            center_x = random.uniform(-2000, 2000)
            center_y = random.uniform(-2000, 2000)
            centers.append((center_x, center_y))
        
        # Generate points around each center
        for cluster_id, (center_x, center_y) in enumerate(centers):
            n_points = random.randint(20, 60)  # Variable cluster sizes
            for _ in range(n_points):
                # Add gaussian noise around center
                x = center_x + random.gauss(0, 300)
                y = center_y + random.gauss(0, 300)
                
                # Mock annotation data
                annotation_data = {
                    'id': len(cluster_data),
                    'label': f'cluster_{cluster_id}',
                    'confidence': random.uniform(0.7, 1.0)
                }
                
                cluster_data.append((x, y, cluster_id, annotation_data))
        
        return cluster_data

    def show_status_message(self, message, error=False):
        """Show a status message in the parent window."""
        if hasattr(self.explorer_window, 'statusBar'):
            timeout = 3000 if not error else 5000
            self.explorer_window.statusBar().showMessage(message, timeout)


class ClusterWidget(QWidget):
    """Widget containing interactive cluster viewer with zoom, pan, and selection."""

    def __init__(self, parent=None):
        super(ClusterWidget, self).__init__(parent)
        self.explorer_window = parent
        self.cluster_points = []  # Store cluster point data
        self.selected_points = []  # Store currently selected points
        self.animation_offset = 0  # For marching ants animation
        self.setup_ui()
        
        # Timer for marching ants animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)  # Update every 100ms

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Cluster Viewer")
        header.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header)

        # Create scene and interactive view
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)  # Large world space
        
        self.graphics_view = InteractiveClusterView(self.graphics_scene)
        self.graphics_view.setMinimumHeight(200)

        # Connect selection change signal
        self.graphics_scene.selectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.graphics_view)
        
        # Add some demo points initially
        self.add_demo_points()

    def add_demo_points(self):
        """Add demonstration cluster points."""
        point_size = 20
        colors = [QColor("cyan"), QColor("red"), QColor("green"), QColor("blue"), QColor("orange")]
        
        # Generate some clustered demo points
        for cluster_id in range(5):
            cluster_color = colors[cluster_id % len(colors)]
            # Generate points around cluster centers
            center_x = random.uniform(-2000, 2000)
            center_y = random.uniform(-2000, 2000)
            
            for _ in range(40):  # 40 points per cluster
                # Add some randomness around cluster center
                x = center_x + random.gauss(0, 300)
                y = center_y + random.gauss(0, 300)
                
                # Create a point as a QGraphicsEllipseItem
                point = QGraphicsEllipseItem(0, 0, point_size, point_size)
                point.setPos(x, y)
                
                # Style the point with cluster color
                point.setBrush(QBrush(cluster_color))
                point.setPen(QPen(QColor("black"), 0.5))
                
                # Make point size independent of zoom level
                point.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                
                # Make the item selectable
                point.setFlag(QGraphicsItem.ItemIsSelectable)
                
                # Store cluster information
                point.setData(0, cluster_id)  # Store cluster ID as user data
                
                self.graphics_scene.addItem(point)
                self.cluster_points.append(point)

    def update_clusters(self, cluster_data):
        """Update the cluster visualization with new data.
        
        Args:
            cluster_data: List of tuples (x, y, cluster_id, annotation_data)
        """
        # Clear existing points
        self.clear_points()
        
        point_size = 10
        colors = [QColor("cyan"), QColor("red"), QColor("green"), QColor("blue"), 
                  QColor("orange"), QColor("purple"), QColor("brown"), QColor("pink")]
        
        for x, y, cluster_id, annotation_data in cluster_data:
            cluster_color = colors[cluster_id % len(colors)]
            
            # Create point
            point = QGraphicsEllipseItem(0, 0, point_size, point_size)
            point.setPos(x, y)
            
            # Style the point
            point.setBrush(QBrush(cluster_color))
            point.setPen(QPen(QColor("black"), 0.5))
            
            # Point appearance settings
            point.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            point.setFlag(QGraphicsItem.ItemIsSelectable)
            
            # Store data
            point.setData(0, cluster_id)  # Cluster ID
            point.setData(1, annotation_data)  # Annotation data
            
            self.graphics_scene.addItem(point)
            self.cluster_points.append(point)

    def clear_points(self):
        """Clear all cluster points from the scene."""
        for point in self.cluster_points:
            self.graphics_scene.removeItem(point)
        self.cluster_points.clear()    
        
    def on_selection_changed(self):
        """Handle point selection changes."""
        selected_items = self.graphics_scene.selectedItems()
        
        # Stop any running animation
        self.animation_timer.stop()
        
        if selected_items:
            print(f"{len(selected_items)} cluster points selected.")
            
            # Store selected points for animation
            self.selected_points = [item for item in selected_items if isinstance(item, QGraphicsEllipseItem)]
            
            # Start marching ants animation
            self.animation_timer.start()
                    
            # Optionally notify parent about selection
            if hasattr(self.explorer_window, 'on_cluster_points_selected'):
                selected_data = []
                for item in selected_items:
                    cluster_id = item.data(0)
                    annotation_data = item.data(1)
                    selected_data.append((cluster_id, annotation_data))
                self.explorer_window.on_cluster_points_selected(selected_data)
                
        else:
            # Clear selected points and revert to original pen
            self.selected_points = []
            for item in self.cluster_points:
                if isinstance(item, QGraphicsEllipseItem):
                    # Reset to original thin black pen
                    item.setPen(QPen(QColor("black"), 0.5))
            print("Cluster selection cleared.")

    def animate_selection(self):
        """Animate selected points with marching ants effect using darker versions of point colors."""
        # Update animation offset for marching ants
        self.animation_offset = (self.animation_offset + 1) % 20  # Reset every 20 pixels like QtAnnotation
        
        # Apply animated pen to selected points using their darker colors
        for item in self.selected_points:
            # Get the original color from the brush
            original_color = item.brush().color()
            
            # Create darker version of the color (reduce brightness by 40%)
            darker_color = original_color.darker(150)  # 150% darker
            
            # Create animated dotted pen with darker color
            animated_pen = QPen(darker_color, 2)
            animated_pen.setStyle(Qt.CustomDashLine)
            animated_pen.setDashPattern([1, 2])  # Small dots with small gaps like QtAnnotation
            animated_pen.setDashOffset(self.animation_offset)
            
            item.setPen(animated_pen)

    def fit_view_to_points(self):
        """Fit the view to show all cluster points."""
        if self.cluster_points:
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)


class SelectableAnnotationViewer(QScrollArea):
    """Scrollable area that supports rubber band selection with Ctrl+drag."""
    
    def __init__(self, annotation_viewer, parent=None):
        super().__init__(parent)
        self.annotation_viewer = annotation_viewer
        self.rubber_band = None
        self.rubber_band_origin = None
        
    def mousePressEvent(self, event):
        """Handle mouse press for starting rubber band selection."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.rubber_band_origin = event.pos()
            if not self.rubber_band:
                from PyQt5.QtWidgets import QRubberBand
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
            self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
            self.rubber_band.show()
            event.accept()
            return
            
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for rubber band selection."""
        if self.rubber_band_origin is not None:
            rect = QRect(self.rubber_band_origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)
            event.accept()
            return

        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete rubber band selection."""
        if self.rubber_band_origin is not None and self.rubber_band:
            self.rubber_band.hide()
            selection_rect = self.rubber_band.geometry()
            
            # The content_widget is where the grid layout lives
            content_widget = self.annotation_viewer.content_widget
            
            # Clear previous selection and find widgets within the rectangle
            self.annotation_viewer.clear_selection()
            
            last_selected_in_rubber_band = -1
            for i, widget in enumerate(self.annotation_viewer.annotation_widgets):
                # Map widget's position relative to the scroll area's viewport
                widget_rect_in_content = widget.geometry()
                widget_rect_in_viewport = QRect(
                    content_widget.mapTo(self.viewport(), widget_rect_in_content.topLeft()),
                    widget_rect_in_content.size()
                )

                if selection_rect.intersects(widget_rect_in_viewport):
                    self.annotation_viewer.select_widget(widget)
                    last_selected_in_rubber_band = i

            # Set the anchor for future shift-clicks to the last item in the selection
            if last_selected_in_rubber_band != -1:
                self.annotation_viewer.last_selected_index = last_selected_in_rubber_band

            self.rubber_band_origin = None
            event.accept()
            return

        super().mouseReleaseEvent(event)


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

    def showEvent(self, event):
        self.setup_ui()
        super(ExplorerWindow, self).showEvent(event)

    def closeEvent(self, event):
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
                child.widget().deleteLater()

        # Top section: Conditions and Settings side by side
        top_layout = QHBoxLayout()
        
        # Conditions on the left
        self.conditions_widget = ConditionsWidget(self.main_window, self)
        top_layout.addWidget(self.conditions_widget, 2)  # Give more space to conditions
        
        # Settings on the right
        self.settings_widget = SettingsWidget(self.main_window, self)
        top_layout.addWidget(self.settings_widget, 1)  # Less space for settings
        
        # Create container widget for top layout
        top_container = QWidget()
        top_container.setLayout(top_layout)
        self.main_layout.addWidget(top_container)

        # Middle section: Annotation Viewer (left) and Cluster Widget (right)
        middle_splitter = QSplitter(Qt.Horizontal)
        # Annotation Viewer (left side of middle section)
        self.annotation_viewer = AnnotationViewerWidget()
        middle_splitter.addWidget(self.annotation_viewer)

        # Cluster widget (right side of middle section)
        self.cluster_widget = ClusterWidget(self)
        middle_splitter.addWidget(self.cluster_widget)

        # Set splitter proportions (annotation viewer wider)
        middle_splitter.setSizes([700, 300])        # Add middle section to main layout with stretch factor
        self.main_layout.addWidget(middle_splitter, 1)
        
        # Note: LabelWindow will be re-parented here by MainWindow.open_explorer_window()
        # The LabelWindow will be added to self.left_layout at index 1 by the MainWindow
        self.main_layout.addWidget(self.label_window)
        
        # Bottom control buttons
        self.buttons_layout = QHBoxLayout()
        # Add stretch to push buttons to the right
        self.buttons_layout.addStretch(1)

        # Main action buttons
        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setToolTip("Close the window")
        self.buttons_layout.addWidget(self.exit_button)

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply)
        self.apply_button.setToolTip("Apply changes")
        self.buttons_layout.addWidget(self.apply_button)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save)
        self.save_button.setToolTip("Save changes")
        self.buttons_layout.addWidget(self.save_button)

        self.main_layout.addLayout(self.buttons_layout)

        # Initialize with sample data
        self.refresh_filters()

    def get_filtered_annotations(self):
        """Get annotations that match all conditions."""
        filtered_annotations = []

        if not hasattr(self.main_window, 'annotation_window') or \
           not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return filtered_annotations

        for annotation in self.main_window.annotation_window.annotations_dict.values():
            annotation_matches = True

            # Check each condition
            for condition in self.conditions_widget.conditions:
                # Check image condition
                image_selection = condition.image_dropdown.currentText()
                if image_selection != "All":
                    if os.path.basename(annotation.image_path) != image_selection:
                        annotation_matches = False
                        break

                # Check annotation type condition
                type_selection = condition.annotation_dropdown.currentText()
                if type_selection != "All":
                    if type(annotation).__name__ != type_selection:
                        annotation_matches = False
                        break

                # Check label condition
                label_selection = condition.label_dropdown.currentText()
                if label_selection != "All":
                    if annotation.label.short_label_code != label_selection:
                        annotation_matches = False
                        break

            if annotation_matches:
                filtered_annotations.append(annotation)

        return filtered_annotations    
    
    def refresh_filters(self):
        """Refresh the display based on current filter conditions."""
        # Get filtered annotations
        filtered_annotations = self.get_filtered_annotations()

        # Update annotation viewer
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.update_annotations(filtered_annotations)

    def filter_images(self):
        self.refresh_filters()

    def filter_labels(self):
        self.refresh_filters()

    def filter_annotations(self):
        self.refresh_filters()

    def update_table(self):
        """Legacy method - functionality moved to annotation viewer."""
        pass

    def update_graphics(self):
        """Update the cluster graphics view."""
        # Delegate to cluster widget
        if hasattr(self, 'cluster_widget'):
            pass  # TODO: Implement clustering visualization in cluster widget

    def update_scroll_area(self):
        """Legacy method - functionality moved to annotation viewer."""
        pass

    def apply(self):
        self.statusBar().showMessage("Changes applied successfully.", 3000)

    def save(self):
        self.statusBar().showMessage("Changes saved successfully.", 3000)
