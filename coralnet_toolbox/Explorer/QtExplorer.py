from coralnet_toolbox.Icons import get_icon
from PyQt5.QtGui import QIcon, QBrush, QPen, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer, QSize, QRect
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget, QGridLayout,
                             QMainWindow, QSplitter, QGroupBox, QFormLayout,
                             QSpinBox, QGraphicsEllipseItem, QGraphicsItem, QSlider,
                             QListWidget, QDoubleSpinBox, QApplication)

from coralnet_toolbox.QtProgressBar import ProgressBar

import warnings
import os
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class AnnotationDataItem:
    """Holds annotation information for consistent display across viewers."""
    
    def __init__(self, annotation, cluster_x=None, cluster_y=None, cluster_id=None):
        self.annotation = annotation
        self.cluster_x = cluster_x or 0.0
        self.cluster_y = cluster_y or 0.0
        self.cluster_id = cluster_id or 0
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

    def __init__(self, annotation, image_path, widget_size=256, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.annotation = annotation
        self.image_path = image_path
        self.annotation_viewer = annotation_viewer
        self._is_selected = False
        self.widget_size = widget_size
        self.animation_offset = 0

        self.setFixedSize(widget_size, widget_size)

        # Timer for marching ants animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation_frame)
        self.animation_timer.setInterval(75)  # Match the annotation's timer for consistency

        self.setup_ui()
        self.load_annotation_image()

    def setup_ui(self):
        """Set up the basic UI with a label for the image."""
        layout = QVBoxLayout(self)
        # Use smaller margins so the border drawn in paintEvent is clearly visible
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        # We no longer set the border here; paintEvent handles it.
        self.image_label.setStyleSheet("border: none;")

        layout.addWidget(self.image_label)

    def load_annotation_image(self):
        """Load and display the actual annotation cropped image."""
        try:
            # This now correctly uses the updated self.widget_size
            # The -8 accounts for the 4px margins on each side
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
        if self._is_selected == selected:
            return

        self._is_selected = selected
        if self._is_selected:
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            self.animation_offset = 0  # Reset offset when deselected

        # Trigger a repaint to update the border
        self.update()

    def is_selected(self):
        """Return whether this widget is selected."""
        return self._is_selected

    def _update_animation_frame(self):
        """Update the animation offset and schedule a repaint."""
        # Increment and wrap the offset, matching the Annotation class
        self.animation_offset = (self.animation_offset + 1) % 20        # self.update() schedules a call to paintEvent()
        self.update()

    def paintEvent(self, event):
        """Handle all custom drawing for the widget, including the border."""
        # First, let the widget draw its children (the QLabel)
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- This is the key part ---
        # We trick the annotation object into giving us the exact pen we need
        # by temporarily setting its state.
        original_selected = self.annotation.is_selected
        original_offset = getattr(self.annotation, '_animated_line', 0)
        
        try:
            self.annotation.is_selected = self._is_selected
            if self._is_selected:
                self.annotation._animated_line = self.animation_offset

            # Determine which color to use - preview takes priority over original
            if hasattr(self.annotation, '_preview_mode') and self.annotation._preview_mode:
                # Use preview label color (persistent until cleared or applied)
                color = self.annotation._preview_label.color
            else:
                # Use normal label color
                color = self.annotation.label.color

            # Special case: Use black for annotations with label.id == "-1", Review (easier to see)
            if (hasattr(self.annotation, '_preview_mode') and self.annotation._preview_mode and 
                self.annotation._preview_label.id == "-1") or \
               (not (hasattr(self.annotation, '_preview_mode') and self.annotation._preview_mode) and 
                self.annotation.label.id == "-1"):
                color = QColor("black")

            # Get the pen using the annotation's own logic
            pen = self.annotation._create_pen(color)
            pen.setWidth(pen.width() + 1)
            painter.setPen(pen)

        finally:
            # IMPORTANT: Restore the annotation's original state
            self.annotation.is_selected = original_selected
            if hasattr(self.annotation, '_animated_line'):
                self.annotation._animated_line = original_offset
        
        # We don't want to draw a fill, just the border
        painter.setBrush(Qt.NoBrush)

        # Draw a rectangle around the widget's edges.
        # .adjusted() moves the rectangle inwards so the border doesn't get clipped.
        width = painter.pen().width()
        # Use integer division to get an integer result
        half_width = (width - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)
        painter.drawRect(rect)
        
    def update_size(self, new_size):
        """
        Updates the widget's size and reloads/rescales its content.
        This should be called by the parent view when resizing.
        """
        self.widget_size = new_size
        self.setFixedSize(new_size, new_size)
        
        # Adjust the inner label size based on the new widget size
        # The margin (e.g., 4) should be consistent with setup_ui
        self.image_label.setFixedSize(new_size - 8, new_size - 8)
        
        # CRITICAL: Reload and rescale the image for the new size
        self.load_annotation_image()
        
        # Trigger a repaint to ensure the border is redrawn correctly
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press events for selection."""
        if event.button() == Qt.LeftButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                self.annotation_viewer.handle_annotation_selection(self, event)
        super().mousePressEvent(event)


# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class ClusterViewer(QGraphicsView):
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
        

class AnnotationViewer(QScrollArea):
    """Scrollable grid widget for displaying annotation image crops with selection support."""
    
    def __init__(self, parent=None):
        super(AnnotationViewer, self).__init__(parent)
        self.annotation_widgets = []
        self.selected_widgets = []
        self.last_selected_index = -1  # Anchor for shift-selection
        self.current_widget_size = 256  # Default size
        
        # Rubber band selection state
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5  # Minimum pixels to drag before starting rubber band
        self.mouse_pressed_on_widget = False  # Track if mouse was pressed on a widget
        
        # Track preview states
        self.preview_label_assignments = {}  # annotation_id -> preview_label
        self.original_label_assignments = {}  # annotation_id -> original_label
        
        self.setup_ui()

    def setup_ui(self):
        # Set up scroll area properties
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create main widget to contain all content
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
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
        
        # Content widget for the grid layout
        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setSpacing(5)

        layout.addWidget(self.content_widget)
        # Set the main widget as the scroll area's widget
        self.setWidget(main_widget)

    def resizeEvent(self, event):
        """Handle resize events to recalculate grid layout."""
        super().resizeEvent(event)
        if hasattr(self, 'annotation_widgets') and self.annotation_widgets:
            self.recalculate_grid_layout()

    def mousePressEvent(self, event):
        """Handle mouse press for starting rubber band selection."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # Store the origin for potential rubber band
            self.rubber_band_origin = event.pos()
            self.mouse_pressed_on_widget = False
            
            # Check if we clicked on a widget
            child_widget = self.childAt(event.pos())
            if child_widget:
                # Find the annotation widget (traverse up the hierarchy)
                widget = child_widget
                while widget and widget != self:
                    if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                        self.mouse_pressed_on_widget = True
                        break
                    widget = widget.parent()
            
            # Always let the event propagate first
            super().mousePressEvent(event)
            return
            
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for rubber band selection."""
        if (self.rubber_band_origin is not None and 
            event.buttons() == Qt.LeftButton and 
            event.modifiers() == Qt.ControlModifier):
            
            # Check if we've moved enough to start rubber band selection
            distance = (event.pos() - self.rubber_band_origin).manhattanLength()
            
            if distance > self.drag_threshold and not self.mouse_pressed_on_widget:
                # Start rubber band if not already started and didn't click on a widget
                if not self.rubber_band:
                    from PyQt5.QtWidgets import QRubberBand
                    self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
                    self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
                    self.rubber_band.show()
                
                # Update rubber band geometry
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
            
            # Only process rubber band selection if rubber band was actually shown
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                selection_rect = self.rubber_band.geometry()
                
                # The content_widget is where the grid layout lives
                content_widget = self.content_widget
                
                # Don't clear previous selection - rubber band adds to existing selection
                
                last_selected_in_rubber_band = -1
                for i, widget in enumerate(self.annotation_widgets):
                    # Map widget's position relative to the scroll area's viewport
                    widget_rect_in_content = widget.geometry()
                    widget_rect_in_viewport = QRect(
                        content_widget.mapTo(self.viewport(), widget_rect_in_content.topLeft()),
                        widget_rect_in_content.size()
                    )

                    if selection_rect.intersects(widget_rect_in_viewport):
                        # Only select if not already selected (add to selection)
                        if not widget.is_selected():
                            self.select_widget(widget)
                        last_selected_in_rubber_band = i

                # Set the anchor for future shift-clicks to the last item in the rubber band selection
                if last_selected_in_rubber_band != -1:
                    self.last_selected_index = last_selected_in_rubber_band

                # Clean up rubber band for next use
                self.rubber_band.deleteLater()
                self.rubber_band = None
                
                event.accept()
            else:
                # No rubber band was shown, let the event propagate for normal Ctrl+Click handling
                super().mouseReleaseEvent(event)

            # Reset rubber band state
            self.rubber_band_origin = None
            self.mouse_pressed_on_widget = False
            return

        super().mouseReleaseEvent(event)
            
    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
        # Ensure value is even to avoid rounding issues with borders
        if value % 2 != 0:
            value -= 1
            
        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        
        # Tell each widget to update its own size and content
        for widget in self.annotation_widgets:
            widget.update_size(value)
        # After all widgets are resized, recalculate the grid
        self.recalculate_grid_layout()

    def recalculate_grid_layout(self):
        """Recalculate the grid layout based on current widget width."""
        if not self.annotation_widgets:
            return
            
        available_width = self.viewport().width() - 20
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
            return  # Widget not in list

        modifiers = event.modifiers()

        if modifiers == Qt.ShiftModifier:
            # Shift+Click: Add range to existing selection (don't clear)
            if self.last_selected_index != -1:
                start = min(self.last_selected_index, widget_index)
                end = max(self.last_selected_index, widget_index)
                for i in range(start, end + 1):
                    if not self.annotation_widgets[i].is_selected():
                        self.select_widget(self.annotation_widgets[i])
            else:
                # If no anchor, just add this widget to selection
                self.select_widget(widget)
                self.last_selected_index = widget_index

        elif modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            # Shift+Ctrl+Click: Add range to existing selection
            if self.last_selected_index != -1:
                start = min(self.last_selected_index, widget_index)
                end = max(self.last_selected_index, widget_index)
                for i in range(start, end + 1):
                    if not self.annotation_widgets[i].is_selected():
                        self.select_widget(self.annotation_widgets[i])
            else:
                # If no anchor, just add this widget to selection
                self.select_widget(widget)
                self.last_selected_index = widget_index

        elif modifiers == Qt.ControlModifier:
            # Ctrl+Click: Toggle selection (add/remove individual items)
            if widget.is_selected():
                self.deselect_widget(widget)
            else:
                self.select_widget(widget)
            self.last_selected_index = widget_index
                
        else:
            # Normal click: Clear all and select only this widget
            self.clear_selection()
            self.select_widget(widget)
            self.last_selected_index = widget_index

    def select_widget(self, widget):
        """Select a widget and add it to the selection."""
        if widget not in self.selected_widgets:
            widget.set_selected(True)
            self.selected_widgets.append(widget)
            # Update label window selection based on selected annotations
        self.update_label_window_selection()

    def deselect_widget(self, widget):
        """Deselect a widget and remove it from the selection."""
        if widget in self.selected_widgets:
            widget.set_selected(False)
            self.selected_widgets.remove(widget)
            # Update label window selection based on remaining selected annotations
        self.update_label_window_selection()

    def clear_selection(self):
        """Clear all selected widgets."""
        # Create a copy of the list to iterate over, as deselect_widget modifies it
        for widget in list(self.selected_widgets):
            widget.set_selected(False)
        self.selected_widgets.clear()
        # Update label window selection (will deselect since no annotations selected)
        self.update_label_window_selection()

    def update_label_window_selection(self):
        """Update the label window selection based on currently selected annotations."""
        # Find the explorer window (our parent)
        explorer_window = self.parent()
        while explorer_window and not hasattr(explorer_window, 'main_window'):
            explorer_window = explorer_window.parent()
            
        if not explorer_window or not hasattr(explorer_window, 'main_window'):
            return
            
        # Get the main window and its components
        main_window = explorer_window.main_window
        label_window = main_window.label_window
        annotation_window = main_window.annotation_window
        
        if not self.selected_widgets:
            # No annotations selected - deselect active label but DON'T clear previews
            # Users should be able to see their preview changes even when nothing is selected
            label_window.deselect_active_label()
            label_window.update_annotation_count()
            return
            
        # Get all selected annotations
        selected_annotations = [widget.annotation for widget in self.selected_widgets]
        
        # Check CURRENT labels for consistency (preview labels take priority over original)
        def get_current_label(annotation):
            """Get the current effective label (preview if exists, otherwise original)."""
            if annotation.id in self.preview_label_assignments:
                return self.preview_label_assignments[annotation.id]
            else:
                return annotation.label
        
        first_current_label = get_current_label(selected_annotations[0])
        all_same_current_label = True
        
        for annotation in selected_annotations:
            current_label = get_current_label(annotation)
            if current_label.id != first_current_label.id:
                all_same_current_label = False
                break
        
        if all_same_current_label:
            # All annotations have the same current label - set it as active
            # IMPORTANT: Don't emit labelSelected signal here to avoid triggering preview override
            label_window.set_active_label(first_current_label)
            # Only emit the signal if this is NOT a preview label to avoid circular updates
            if selected_annotations[0].id not in self.preview_label_assignments:
                annotation_window.labelSelected.emit(first_current_label.id)
        else:
            # Multiple different labels - deselect active label
            label_window.deselect_active_label()
        
        # Update annotation count display to show selection
        label_window.update_annotation_count()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]

    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label to selected annotations (visual only)."""
        if not self.selected_widgets or not preview_label:
            return
            
        for widget in self.selected_widgets:
            annotation = widget.annotation
            
            # Store original label if this is the first preview change
            if annotation.id not in self.original_label_assignments:
                self.original_label_assignments[annotation.id] = annotation.label
            
            # Track the preview assignment
            self.preview_label_assignments[annotation.id] = preview_label
            
            # Update visual appearance temporarily
            self._apply_preview_visual_state(annotation, preview_label)
            
            # Force widget to repaint with new color
            widget.update()

    def _apply_preview_visual_state(self, annotation, preview_label):
        """Apply visual preview state to annotation without changing actual label."""
        # Temporarily override the annotation's visual properties
        annotation._preview_label = preview_label
        annotation._preview_mode = True
        
        # Update the graphics item if it exists
        if hasattr(annotation, 'graphics_item') and annotation.graphics_item:
            # Force a visual update
            annotation.graphics_item.update()

    def clear_preview_states(self):
        """Clear all preview states and revert to original labels."""
        if not self.preview_label_assignments:
            return  # Nothing to clear
            
        for annotation_id in list(self.preview_label_assignments.keys()):
            # Get the annotation
            annotation = self._get_annotation_by_id(annotation_id)
            if annotation:
                # Clear preview mode
                if hasattr(annotation, '_preview_mode'):
                    annotation._preview_mode = False
                if hasattr(annotation, '_preview_label'):
                    delattr(annotation, '_preview_label')
                    
                # Update graphics
                if hasattr(annotation, 'graphics_item') and annotation.graphics_item:
                    annotation.graphics_item.update()
        
        # Clear tracking dictionaries
        self.preview_label_assignments.clear()
        self.original_label_assignments.clear()
        
        # Update all widgets to show original colors
        for widget in self.annotation_widgets:
            widget.update()
        
        # Update the label window selection after clearing previews
        self.update_label_window_selection()

    def has_preview_changes(self):
        """Check if there are any pending preview changes."""
        return bool(self.preview_label_assignments)

    def get_preview_changes_summary(self):
        """Get a summary of preview changes for user feedback."""
        if not self.preview_label_assignments:
            return "No preview changes"
        
        change_count = len(self.preview_label_assignments)
        return f"{change_count} annotation(s) with preview changes"

    def apply_preview_changes_permanently(self):
        """Apply all preview changes permanently to the annotation data."""
        applied_annotations = []
        
        for annotation_id, preview_label in self.preview_label_assignments.items():
            annotation = self._get_annotation_by_id(annotation_id)
            if annotation:
                # Actually update the annotation's label
                annotation.update_label(preview_label)
                
                # MISSING: Update user confidence for the new label
                annotation.update_user_confidence(preview_label)
                
                # MISSING: Regenerate cropped image with new label context
                # Get the rasterio image for cropping
                explorer_window = self.parent()
                while explorer_window and not hasattr(explorer_window, 'main_window'):
                    explorer_window = explorer_window.parent()
                
                if explorer_window and hasattr(explorer_window, 'main_window'):
                    main_window = explorer_window.main_window
                    if hasattr(main_window, 'annotation_window') and main_window.annotation_window.rasterio_image:
                        annotation.create_cropped_image(main_window.annotation_window.rasterio_image)
                    
                    # MISSING: Update confidence window if annotation is selected
                    if (hasattr(main_window, 'confidence_window') and 
                        annotation in main_window.annotation_window.selected_annotations):
                        main_window.confidence_window.display_cropped_image(annotation)
                
                applied_annotations.append(annotation)
        
        # Clear preview state after applying
        self.clear_preview_states()
        
        return applied_annotations

    def _get_annotation_by_id(self, annotation_id):
        """Helper to get annotation by ID."""
        for widget in self.annotation_widgets:
            if widget.annotation.id == annotation_id:
                return widget.annotation
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------------------------------------------------


class ConditionsWidget(QGroupBox):
    """Widget containing all filter conditions in a multi-column layout."""

    def __init__(self, main_window, parent=None):
        super(ConditionsWidget, self).__init__("Conditions", parent)
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

        # TODO: Add model and clustering settings tabs

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
        
        self.graphics_view = ClusterViewer(self.graphics_scene)
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
        self.conditions_widget = ConditionsWidget(self.main_window, self)
        self.settings_widget = SettingsWidget(self.main_window, self)
        self.annotation_viewer = AnnotationViewer()
        self.cluster_widget = ClusterWidget(self)
        
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
        top_layout.addWidget(self.conditions_widget, 2)  # Give more space to conditions
        top_layout.addWidget(self.settings_widget, 1)  # Less space for settings
        
        # Create container widget for top layout
        top_container = QWidget()
        top_container.setLayout(top_layout)
        self.main_layout.addWidget(top_container)

        # Middle section: Annotation Viewer (left) and Cluster Widget (right)
        middle_splitter = QSplitter(Qt.Horizontal)
        middle_splitter.addWidget(self.annotation_viewer)
        middle_splitter.addWidget(self.cluster_widget)

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
        self.conditions_widget.set_default_to_current_image()
        self.refresh_filters()
        
        # Connect label selection to preview updates (only connect once)
        try:
            self.label_window.labelSelected.disconnect(self.on_label_selected_for_preview)
        except TypeError:
            pass  # Signal wasn't connected yet
        self.label_window.labelSelected.connect(self.on_label_selected_for_preview)

    def get_filtered_annotations(self):
        """Get annotations that match all conditions."""
        filtered_annotations = []

        if not hasattr(self.main_window, 'annotation_window') or \
           not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return filtered_annotations

        # Get current filter conditions
        selected_images = self.conditions_widget.get_selected_images()
        selected_types = self.conditions_widget.get_selected_annotation_types()
        selected_labels = self.conditions_widget.get_selected_labels()
        topk_selection = self.conditions_widget.get_topk_selection()
        confidence_operator, confidence_value = self.conditions_widget.get_confidence_condition()

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
                filtered_annotations.append(annotation)

        # Ensure all filtered annotations have cropped images
        if filtered_annotations:
            # Group annotations by image path to process efficiently, but only for those that need cropping
            annotations_by_image = {}
            for annotation in filtered_annotations:
                # Only process annotations that don't have cropped images
                if not annotation.cropped_image:
                    image_path = annotation.image_path
                    if image_path not in annotations_by_image:
                        annotations_by_image[image_path] = []
                    annotations_by_image[image_path].append(annotation)
            
            # Only crop annotations if there are any that need cropping
            if annotations_by_image:
                progress_bar = ProgressBar(self, "Cropping Image Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(annotations_by_image))
                
                try:
                    # Crop annotations for each image using the AnnotationWindow method
                    # This ensures consistency with how cropped images are generated elsewhere
                    for image_path, image_annotations in annotations_by_image.items():
                        # Use the existing crop_annotations method from AnnotationWindow
                        # This ensures consistency with how cropped images are generated elsewhere
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

        return filtered_annotations
    
    def refresh_filters(self):
        """Refresh the display based on current filter conditions."""
        # Set cursor to busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Check if only one image is selected and load it in annotation window
            single_image_path = self.conditions_widget.get_single_selected_image_path()
            if single_image_path and hasattr(self.main_window, 'image_window'):
                # Load the single selected image in the annotation window
                self.main_window.image_window.load_image_by_path(single_image_path)
            
            # Get filtered annotations
            filtered_annotations = self.get_filtered_annotations()

            # Update annotation viewer
            if hasattr(self, 'annotation_viewer'):
                self.annotation_viewer.update_annotations(filtered_annotations)
        finally:
            # Restore cursor to normal
            QApplication.restoreOverrideCursor()

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
            pass        # TODO: Implement clustering visualization in cluster widget

    def update_scroll_area(self):
        """Legacy method - functionality moved to annotation viewer."""
        pass

    def on_label_selected_for_preview(self, label):
        """Handle label selection to update preview state."""
        if hasattr(self, 'annotation_viewer') and self.annotation_viewer.selected_widgets:
            # Check if we're actually changing to a different label
            selected_annotations = [widget.annotation for widget in self.annotation_viewer.selected_widgets]
            
            # Check if all selected annotations already have this label (either as preview or original)
            all_already_have_label = True
            for annotation in selected_annotations:
                current_label = None
                if annotation.id in self.annotation_viewer.preview_label_assignments:
                    current_label = self.annotation_viewer.preview_label_assignments[annotation.id]
                else:
                    current_label = annotation.label
                
                if current_label.id != label.id:
                    all_already_have_label = False
                    break
            
            # Only apply preview if we're actually changing the label
            if not all_already_have_label:
                # Apply preview label to selected annotations
                self.annotation_viewer.apply_preview_label_to_selected(label)
                # Update button states
                self.update_button_states()

    def clear_preview_changes(self):
        """Clear all preview changes and revert to original labels."""
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.clear_preview_states()
            # Update button states
            self.update_button_states()
            print("Cleared all preview changes")

    def update_button_states(self):
        """Update the state of Clear Preview and Apply buttons based on preview changes."""
        has_changes = (hasattr(self, 'annotation_viewer') and self.annotation_viewer.has_preview_changes())
        
        self.clear_preview_button.setEnabled(has_changes)
        self.apply_button.setEnabled(has_changes)
        
        # Update button tooltips with summary
        if has_changes:
            summary = self.annotation_viewer.get_preview_changes_summary()
            self.clear_preview_button.setToolTip(f"Clear all preview changes - {summary}")
            self.apply_button.setToolTip(f"Apply changes - {summary}")
        else:
            self.clear_preview_button.setToolTip("Clear all preview changes and revert to original labels")
            self.apply_button.setToolTip("Apply changes")

    def apply(self):
        """Apply any modifications made in the Explorer to the actual annotations."""
        try:
            # Apply preview changes permanently first
            applied_annotations = self.annotation_viewer.apply_preview_changes_permanently()
            
            if applied_annotations:
                # Track which images need to be updated
                affected_images = set()
                for annotation in applied_annotations:
                    affected_images.add(annotation.image_path)

                # Update image annotations for all affected images
                for image_path in affected_images:
                    self.image_window.update_image_annotations(image_path)

                # Reload annotations in the annotation window
                self.annotation_window.load_annotations()

                # Refresh the filtered view
                self.refresh_filters()

                # Clear selection in the annotation viewer
                self.annotation_viewer.clear_selection()

                # Update button states
                self.update_button_states()

                # Optionally print a message
                print(f"Applied changes to {len(applied_annotations)} annotation(s)")
            else:
                print("No preview changes to apply")

        except Exception as e:
            print(f"Error applying modifications: {e}")