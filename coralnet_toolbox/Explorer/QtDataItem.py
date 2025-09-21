import warnings

import os

import numpy as np

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import QGraphicsObject, QStyle, QVBoxLayout, QLabel, QWidget, QGraphicsItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_SIZE = 15
POINT_WIDTH = 3
SPRITE_SIZE = 32
ANNOTATION_WIDTH = 5


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPointItem(QGraphicsObject):
    """
    A custom QGraphicsObject that can display as a dot or an image sprite,
    getting its state from an associated AnnotationDataItem.
    """

    def __init__(self, data_item, viewer):
        """
        Initializes the point item.
        Args:
            data_item (AnnotationDataItem): The data item that holds the state.
            viewer (EmbeddingViewer): A reference to the parent viewer.
        """
        super(EmbeddingPointItem, self).__init__()

        self.data_item = data_item
        self.viewer = viewer
        self.thumbnail_pixmap = None

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        self.default_pen = QPen(QColor("black"), POINT_WIDTH)
        self.setPos(self.data_item.embedding_x, self.data_item.embedding_y)
        self.setToolTip(self.data_item.get_tooltip_text())
        
        # Animation properties (updated for pulsing)
        self._pulse_alpha = 128  # Starting alpha for pulsing (semi-transparent)
        self._pulse_direction = 1  # 1 for increasing alpha, -1 for decreasing
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_pulse_alpha)
        self.animation_timer.setInterval(50)  # Reduced to 50ms for faster, heartbeat-like pulsing

    def boundingRect(self):
        """Returns the bounding rectangle, which depends on the display mode and depth."""
        
        scale_factor = 1.0
        if self.viewer and self.viewer.is_3d_data and self.viewer.z_range > 0:
            # Normalize z from its global range to a [0, 1] range
            z_normalized = (self.data_item.embedding_z - self.viewer.min_z) / self.viewer.z_range
            # Map normalized z to a scale factor (e.g., from 0.5x to 1.5x)
            scale_factor = 0.5 + z_normalized
    
        if self.viewer and self.viewer.display_mode == 'sprites':
            ar = self.data_item.aspect_ratio
            if ar >= 1.0:
                width = SPRITE_SIZE * scale_factor
                height = (SPRITE_SIZE / ar) * scale_factor
            else:
                height = SPRITE_SIZE * scale_factor
                width = (SPRITE_SIZE * ar) * scale_factor
            return QRectF(0, 0, width, height)
        else:
            
            size = POINT_SIZE * scale_factor
            return QRectF(0, 0, size, size)

    def update_tooltip(self):
        """Updates the tooltip by fetching the latest text from the data item."""
        self.setToolTip(self.data_item.get_tooltip_text())

    def paint(self, painter, option, widget):
        """
        Custom paint method to draw either a dot or a sprite with a border.
        """
        option.state &= ~QStyle.State_Selected
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate scale_factor for size and opacity
        scale_factor = 1.0
        if self.viewer and self.viewer.is_3d_data and self.viewer.z_range > 0:
            z_normalized = (self.data_item.embedding_z - self.viewer.min_z) / self.viewer.z_range
            scale_factor = 0.5 + z_normalized  # From 0.5x to 1.5x

        # Calculate scaled pen width for borders (clamp to avoid extremes)
        scaled_pen_width = max(1, min(POINT_WIDTH * scale_factor, 6))  # Clamp between 1 and 6 for usability
        
        # Calculate opacity
        opacity = 255
        if self.viewer and self.viewer.is_3d_data and self.viewer.z_range > 0:
            z_normalized = (self.data_item.embedding_z - self.viewer.min_z) / self.viewer.z_range
            opacity = int(128 + 127 * z_normalized)

        base_color = self.data_item.effective_color
        effective_brush_color = QColor(base_color)
        effective_brush_color.setAlpha(opacity)
        
        display_mode = self.viewer.display_mode if self.viewer else 'dots'

        if display_mode == 'sprites':
            # Ensure the pixmap is scaled to the current boundingRect size (handles dynamic scaling during rotation)
            current_size = self.boundingRect().size().toSize()
            if self.thumbnail_pixmap is None or self.thumbnail_pixmap.size() != current_size:
                source_pixmap = self.data_item.annotation.get_cropped_image_graphic()
                if source_pixmap and not source_pixmap.isNull():
                    self.thumbnail_pixmap = source_pixmap.scaled(
                        current_size,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
            
            if self.thumbnail_pixmap:
                painter.drawPixmap(self.boundingRect().topLeft(), self.thumbnail_pixmap)

            # Scaled border pen for sprites
            border_color = QColor(self.data_item.effective_color)
            border_color.setAlpha(opacity)
            border_pen = QPen(border_color, scaled_pen_width)
            if self.isSelected():
                # Use a darker version of the color for better visibility
                border_color = QColor(self.data_item.effective_color).darker(150)  
                border_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
                border_pen = QPen(border_color, scaled_pen_width)
                border_pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
                if not self.animation_timer.isActive():
                    self.animation_timer.start()  # Start pulsing on selection
            else:
                if self.animation_timer.isActive():
                    self.animation_timer.stop()  # Stop pulsing on deselection

            painter.setPen(border_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect())
        else:
            # Draw Original Dot with scaled size and pen
            if self.isSelected():
                # Use a darker version of the color for better visibility
                darker_color = QColor(self.data_item.effective_color).darker(150)  
                darker_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
                animated_pen = QPen(darker_color, scaled_pen_width)
                animated_pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
                if not self.animation_timer.isActive():
                    self.animation_timer.start()  # Start pulsing on selection
                painter.setPen(animated_pen)
            else:
                pen_color = QColor("black")
                pen_color.setAlpha(opacity)
                painter.setPen(QPen(pen_color, scaled_pen_width))
                if self.animation_timer.isActive():
                    self.animation_timer.stop()  # Stop pulsing on deselection

            painter.setBrush(effective_brush_color)
            painter.drawEllipse(self.boundingRect())
    
    def _update_pulse_alpha(self):
        """Update the pulse alpha for a heartbeat-like effect: quick rise, slow fall."""
        if self._pulse_direction == 1:
            # Quick increase (systole-like)
            self._pulse_alpha += 30
        else:
            # Slow decrease (diastole-like)
            self._pulse_alpha -= 10  # <-- Corrected from += to -=

        # Check direction before clamping to ensure smooth transition
        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255  # Clamp to max
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50   # Clamp to min
            self._pulse_direction = 1
        
        self.update()  # Trigger repaint
    
    def __del__(self):
        """Clean up the timer when the item is deleted."""
        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop()
            

class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, data_item, widget_height=96, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.data_item = data_item  # The single source of truth for state
        self.annotation = data_item.annotation
        self.annotation_viewer = annotation_viewer

        self.widget_height = widget_height
        self.aspect_ratio = 1.0
        self.pixmap = None
        self.is_loaded = False  # Flag for lazy loading

        # Animation properties (updated for pulsing)
        self._pulse_alpha = 128  # Starting alpha for pulsing (semi-transparent)
        self._pulse_direction = 1  # 1 for increasing alpha, -1 for decreasing
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_pulse_alpha)
        self.animation_timer.setInterval(50)  # Reduced to 50ms for faster, heartbeat-like pulsing

        self.setup_ui()
        self.recalculate_aspect_ratio()  # Calculate aspect ratio from geometry
        self.update_height(self.widget_height)  # Set initial size
        self.update_tooltip()

    def setup_ui(self):
        """Set up the basic UI with a label for the image."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: none;")
        layout.addWidget(self.image_label)
        
    def update_tooltip(self):
        """Updates the tooltip by fetching the latest text from the data item."""
        self.setToolTip(self.data_item.get_tooltip_text())

    def recalculate_aspect_ratio(self):
        """Calculate and store the annotation's aspect ratio."""
        annotation = self.data_item.annotation
        
        # Try to use the cropped_bbox attribute first
        if hasattr(annotation, 'cropped_bbox'):
            min_x, min_y, max_x, max_y = annotation.cropped_bbox
            width = max_x - min_x
            height = max_y - min_y
            
            if height > 0:
                self.aspect_ratio = width / height
                return
        
        # Fallback to bounding box methods
        try:
            top_left = annotation.get_bounding_box_top_left()
            bottom_right = annotation.get_bounding_box_bottom_right()
            
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                
                if height > 0:
                    self.aspect_ratio = width / height
                    return
        except (AttributeError, TypeError):
            pass
        
        # Last resort: try to get aspect ratio from the cropped image
        try:
            pixmap = annotation.get_cropped_image()
            if pixmap and not pixmap.isNull() and pixmap.height() > 0:
                self.aspect_ratio = pixmap.width() / pixmap.height()
                return
        except (AttributeError, TypeError):
            pass
        
        # Default to square if we can't determine aspect ratio
        self.aspect_ratio = 1.0

    def load_image(self):
        """Loads the image pixmap if it hasn't been loaded yet."""
        if self.is_loaded:
            return

        try:
            cropped_pixmap = self.annotation.get_cropped_image_graphic()
            if cropped_pixmap and not cropped_pixmap.isNull():
                self.pixmap = cropped_pixmap
                self.is_loaded = True
                self._display_pixmap()
            else:
                self.image_label.setText("No Image\nAvailable")
                self.pixmap = None
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.image_label.setText("Error\nLoading Image")
            self.pixmap = None

    def unload_image(self):
        """Unloads the pixmap to free memory."""
        if not self.is_loaded:
            return
        self.pixmap = None
        self.image_label.clear()
        self.is_loaded = False

    def _display_pixmap(self):
        """Scales and displays the currently loaded pixmap."""
        if self.pixmap:
            new_width = int(self.widget_height * self.aspect_ratio)
            scaled_pixmap = self.pixmap.scaled(
                new_width - 8,
                self.widget_height - 8,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def update_height(self, new_height):
        """Updates the widget's height and rescales its width and content accordingly."""
        self.widget_height = new_height
        new_width = int(self.widget_height * self.aspect_ratio)
        self.setFixedSize(new_width, new_height)
        if self.pixmap:
            self._display_pixmap()
        self.update()

    def update_selection_visuals(self):
        """
        Updates the widget's visual state based on the data_item's selection
        status. This should be called by the controlling viewer.
        """
        is_selected = self.data_item.is_selected

        if is_selected:
            if not self.animation_timer.isActive():
                self.animation_timer.start()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
            self._pulse_alpha = 128  # Reset to default

        # Trigger a repaint to show the new selection state (border, etc.)
        self.update()

    def is_selected(self):
        """Return whether this widget is selected via the data item."""
        return self.data_item.is_selected

    def _update_animation_frame(self):
        """Update the animation offset and schedule a repaint."""
        # Removed: self.animation_offset = (self.animation_offset + 1) % 20
        self.update()

    def paintEvent(self, event):
        """Handle custom drawing for the widget, including the selection border."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        effective_label = self.data_item.effective_label
        pen_color = self.data_item.effective_color
        if effective_label and effective_label.id == "-1":
            # If the label is a temporary one (e.g., "-1", Review), use black for the pen color
            pen_color = QColor("black")

        if self.is_selected():
            # Use a darker version of the color for better visibility
            pen_color = pen_color.darker(150)  # Changed to darker for brighter selected appearance
            pen_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
        else:
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setStyle(Qt.SolidLine)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        width = pen.width()
        half_width = (width - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)
        painter.drawRect(rect)
        
    def _update_pulse_alpha(self):
        """Update the pulse alpha for a heartbeat-like effect: quick rise, slow fall."""
        if self._pulse_direction == 1:
            # Quick increase (systole-like)
            self._pulse_alpha += 30
        else:
            # Slow decrease (diastole-like)
            self._pulse_alpha -= 10  # <-- Corrected from += to -=

        # Check direction before clamping to ensure smooth transition
        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255  # Clamp to max
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50   # Clamp to min
            self._pulse_direction = 1
        
        self.update()  # Trigger repaint

    def mousePressEvent(self, event):
        """Handle mouse press events for selection, delegating logic to the viewer."""
        if event.button() == Qt.LeftButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                # The viewer is the controller and will decide how to change the selection state
                self.annotation_viewer.handle_annotation_selection(self, event)
        elif event.button() == Qt.RightButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_context_menu'):
                self.annotation_viewer.handle_annotation_context_menu(self, event)
                event.accept()
                return
            else:
                event.ignore()
        super().mousePressEvent(event)
        
    def __del__(self):
        """Clean up the timer when the widget is deleted."""
        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop()


class AnnotationDataItem:
    """
    Holds all annotation state information for consistent display across viewers.
    This acts as the "ViewModel" for a single annotation, serving as the single
    source of truth for its state in the UI.
    """

    def __init__(self, annotation, embedding_x=None, embedding_y=None, embedding_id=None):
        self.annotation = annotation
        
        self.embedding_x = embedding_x if embedding_x is not None else 0.0
        self.embedding_y = embedding_y if embedding_y is not None else 0.0
        
        self.embedding_z = 0.0  # This will store the rotated Z-value (depth)
        
        # Store the original, un-rotated 3D coordinates from the embedding
        self.embedding_x_3d = 0.0
        self.embedding_y_3d = 0.0
        self.embedding_z_3d = 0.0
        
        self.embedding_id = embedding_id
        
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label

        # Calculate and store aspect ratio on initialization
        self.aspect_ratio = self._calculate_aspect_ratio()
        
        # To store pre-formatted top-k prediction details
        self.prediction_details = None
        # To store prediction probabilities for sorting
        self.prediction_probabilities = None

    def _calculate_aspect_ratio(self):
        """Calculate and return the annotation's aspect ratio."""
        annotation = self.annotation
        
        if hasattr(annotation, 'cropped_bbox'):
            min_x, min_y, max_x, max_y = annotation.cropped_bbox
            width = max_x - min_x
            height = max_y - min_y
            if height > 0:
                return width / height

        try:
            top_left = annotation.get_bounding_box_top_left()
            bottom_right = annotation.get_bounding_box_bottom_right()
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                if height > 0:
                    return width / height
        except (AttributeError, TypeError):
            pass

        try:
            pixmap = annotation.get_cropped_image()
            if pixmap and not pixmap.isNull() and pixmap.height() > 0:
                return pixmap.width() / pixmap.height()
        except (AttributeError, TypeError):
            pass
        
        return 1.0  # Default to square
        
    @property
    def effective_label(self):
        """Get the current effective label (preview if it exists, otherwise original)."""
        return self._preview_label if self._preview_label else self.annotation.label

    @property
    def effective_color(self):
        """Get the effective color for this annotation based on the effective label."""
        return self.effective_label.color

    @property
    def is_selected(self):
        """Check if this annotation is selected."""
        return self._is_selected

    def set_selected(self, selected):
        """Set the selection state. This is the single point of control."""
        self._is_selected = selected

    def set_preview_label(self, label):
        """Set a preview label for this annotation."""
        self._preview_label = label

    def clear_preview_label(self):
        """Clear the preview label and revert to the original."""
        self._preview_label = None

    def has_preview_changes(self):
        """Check if this annotation has a temporary preview label assigned."""
        return self._preview_label is not None

    def apply_preview_permanently(self):
        """Apply the preview label permanently to the underlying annotation object."""
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
            'embedding_id': self.embedding_id,
            'color': self.effective_color
        }
    
    def get_tooltip_text(self):
        """
        Generates a rich HTML-formatted tooltip with all relevant information.
        """
        info = self.get_display_info()
        
        tooltip_parts = [
            f"<b>ID:</b> {info['id']}",
            f"<b>Image:</b> {info['image']}",
            f"<b>Label:</b> {info['label']}",
            f"<b>Type:</b> {info['type']}"
        ]

        # Add prediction details if they exist
        if self.prediction_details:
            tooltip_parts.append(f"<hr>{self.prediction_details}")

        return "<br>".join(tooltip_parts)

    def get_effective_confidence(self):
        """
        Get the effective confidence value, handling scalar, array, and vector predictions.
        """
        # First check if prediction probabilities are available from model predictions
        if hasattr(self, 'prediction_probabilities') and self.prediction_probabilities is not None:
            probs = self.prediction_probabilities
            try:
                # This will succeed for lists and multi-element numpy arrays
                if len(probs) > 0:
                    return float(np.max(probs))
            except TypeError:
                # This will catch the error if `len()` is called on a scalar or 0-D array.
                # In this case, the value of `probs` itself is the confidence score.
                return float(probs)

        # Fallback to existing confidence values
        if self.annotation.verified and hasattr(self.annotation, 'user_confidence') and self.annotation.user_confidence:
            return list(self.annotation.user_confidence.values())[0]
        elif hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
            
        return 0.0