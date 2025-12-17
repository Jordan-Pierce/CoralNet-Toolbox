import warnings

import os

import numpy as np

from PyQt5.QtCore import Qt, QRectF, QRect
from PyQt5.QtGui import QPen, QColor, QPainter, QFont
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
        self.default_pen.setCosmetic(True)
        self.setPos(self.data_item.embedding_x, self.data_item.embedding_y)
        self.setToolTip(self.data_item.get_tooltip_text())
        
        # --- Animation Properties ---
        self.animation_manager = None
        self.is_animating = False
        
        # --- Animation properties ---
        self._pulse_alpha = 128  # Starting alpha for pulsing (semi-transparent)
        self._pulse_direction = 1  # 1 for increasing alpha, -1 for decreasing
        
    def set_animation_manager(self, manager):
        """
        Binds this object to the central AnimationManager.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        self.animation_manager = manager
        
    def is_graphics_item_valid(self):
        """
        Checks if the graphics item is still valid and added to a scene.
        
        Returns:
            bool: True if the item exists and has a scene, False otherwise.
        """
        try:
            return self.scene() is not None
        except RuntimeError:
            # This can happen if the C++ part of the item is deleted
            return False

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
            border_pen.setCosmetic(True)
            if self.isSelected():
                # Tell manager to start animating
                self.animate()
                
                border_color = QColor(self.data_item.effective_color).darker(150)  
                border_color.setAlpha(self._pulse_alpha)
                border_pen = QPen(border_color, scaled_pen_width)
                border_pen.setCosmetic(True)
                border_pen.setStyle(Qt.DotLine)
            else:
                # Tell manager to stop animating
                self.deanimate()

            painter.setPen(border_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect())
        else:
            # Draw Original Dot with scaled size and pen
            if self.isSelected():
                # Tell manager to start animating
                self.animate()
                
                darker_color = QColor(self.data_item.effective_color).darker(150)  
                darker_color.setAlpha(self._pulse_alpha)
                animated_pen = QPen(darker_color, scaled_pen_width)
                animated_pen.setCosmetic(True)
                animated_pen.setStyle(Qt.DotLine)
                painter.setPen(animated_pen)
            else:
                # Tell manager to stop animating
                self.deanimate()
                
                pen_color = QColor("black")
                pen_color.setAlpha(opacity)
                pen = QPen(pen_color, scaled_pen_width)
                pen.setCosmetic(True)
                painter.setPen(pen)

            painter.setBrush(effective_brush_color)
            painter.drawEllipse(self.boundingRect())
            
    def tick_animation(self):
        """
        Perform one 'tick' of the animation.
        This is the public entry point for the global manager.
        """
        # This just calls the existing private method that holds the logic
        self._update_pulse_alpha()
    
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
    
    def animate(self):
        """Start the pulsing animation by registering with the global timer."""
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)
            
    def deanimate(self):
        """Stop the pulsing animation by de-registering from the global timer."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
            
        self._pulse_alpha = 128  # Reset to default
        self.update()  # Apply the default style
    
    def __del__(self):
        """Clean up the timer when the item is deleted."""
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()
            

class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, data_item, widget_height=96, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.data_item = data_item
        self.annotation = data_item.annotation
        self.annotation_viewer = annotation_viewer

        self.widget_height = widget_height
        self.aspect_ratio = 1.0
        self.pixmap = None
        self.is_loaded = False

        # --- Animation Properties ---
        self.animation_manager = None
        self.is_animating = False
        
        # --- Animation properties (no timer) ---
        self._pulse_alpha = 128
        self._pulse_direction = 1

        self.recalculate_aspect_ratio()
        self.update_height(self.widget_height)
        self.update_tooltip()
        
    def set_animation_manager(self, manager):
        """
        Binds this object to the central AnimationManager.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        self.animation_manager = manager
        
    def is_graphics_item_valid(self):
        """
        Checks if the widget is still valid and visible.
        
        Returns:
            bool: True if the widget is visible, False otherwise.
        """
        try:
            return self.isVisible()
        except RuntimeError:
            return False
        
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
                self.update()
            else:
                self.pixmap = None
                self.update()
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.pixmap = None
            self.update()

    def unload_image(self):
        """Unloads the pixmap to free memory."""
        if not self.is_loaded:
            return
        self.pixmap = None
        self.is_loaded = False

    def update_height(self, new_height):
        """Updates the widget's height and rescales its width and content accordingly."""
        self.widget_height = new_height
        new_width = int(self.widget_height * self.aspect_ratio)
        self.setFixedSize(new_width, new_height)
        self.update()

    def update_selection_visuals(self):
        """
        Updates the widget's visual state based on the data_item's selection
        status. This should be called by the controlling viewer.
        """
        is_selected = self.data_item.is_selected

        if is_selected:
            self.animate()
        else:
            self.deanimate()
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

        # Draw the image
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                self.width() - 8,
                self.height() - 8,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.drawPixmap(4, 4, scaled_pixmap)
        else:
            painter.setPen(QColor("black"))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image\nAvailable")

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
            pen.setCosmetic(True)
            pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
        else:
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(Qt.SolidLine)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        width = pen.width()
        half_width = (width - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)
        painter.drawRect(rect)
        
        # --- Draw Machine Confidence Badge ---
        confidence_score, _ = self.data_item.get_machine_confidence()
        if confidence_score > 0:
            # Badge dimensions - minimum size with fixed font
            badge_size = 32  # Increased from 28 for better readability
            margin = 4
            badge_x = self.width() - badge_size - margin
            badge_y = margin
            badge_rect = QRect(badge_x, badge_y, badge_size, badge_size)
            
            # Get color-coded background from viewer's breaks

            try:
                if (self.annotation_viewer and
                    hasattr(self.annotation_viewer, 'confidence_breaks') and
                    self.annotation_viewer.confidence_breaks):
                    badge_color = self.data_item.get_confidence_color(self.annotation_viewer.confidence_breaks)
                else:
                    badge_color = self.data_item.get_confidence_color()
            except Exception:
                # Fallback to default color if anything goes wrong
                badge_color = QColor(0, 0, 0, 180)
            
            # Make badge semi-transparent
            badge_color.setAlpha(200)
            painter.fillRect(badge_rect, badge_color)
            
            # Draw border
            painter.setPen(QPen(QColor(0, 0, 0, 100), 1))
            painter.drawRect(badge_rect)
            
            # Draw percentage text with fixed size (doesn't scale with widget)
            confidence_percent = int(confidence_score * 100)
            painter.setFont(QFont("Arial", 10, QFont.Bold))  # Fixed size font
            painter.setPen(QColor("white"))
            
            # Add text shadow for better readability
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(badge_rect.adjusted(1, 1, 1, 1), Qt.AlignCenter, f"{confidence_percent}%")
            painter.setPen(QColor("white"))
            painter.drawText(badge_rect, Qt.AlignCenter, f"{confidence_percent}%")
        
    def tick_animation(self):
        """
        Perform one 'tick' of the animation.
        This is the public entry point for the global manager.
        """
        # This just calls the existing private method that holds the logic
        self._update_pulse_alpha()
        
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
        
    def animate(self):
        """Start the pulsing animation by registering with the global timer."""
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)
            
    def deanimate(self):
        """Stop the pulsing animation by de-registering from the global timer."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
            
        self._pulse_alpha = 128  # Reset to default
        self.update()  # Apply the default style
    
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
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()            


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
        if self.annotation.verified:
            return 1.0
        elif hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
            
        return 1.0

    def get_machine_confidence(self):
        """
        Get the top-1 machine confidence score.
        Prioritizes prediction_probabilities from current model over annotation's machine_confidence.
        Returns a tuple (confidence_value, label_name) or (0.0, None) if not available.
        """
        # First priority: current prediction probabilities (from model)
        if hasattr(self, 'prediction_probabilities') and self.prediction_probabilities is not None:
            probs = self.prediction_probabilities
            try:
                # Handle array-like probabilities
                if len(probs) > 0:
                    confidence = float(np.max(probs))
                    if confidence > 0:  # Only use if valid confidence
                        # Try to get the label if available
                        label = self.effective_label.short_label_code if self.effective_label else None
                        return confidence, label
            except TypeError:
                # Handle scalar probabilities
                confidence = float(probs)
                if confidence > 0:  # Only use if valid confidence
                    label = self.effective_label.short_label_code if self.effective_label else None
                    return confidence, label
        
        # Second priority: annotation's machine_confidence dict
        if (hasattr(self.annotation, 'machine_confidence') and 
            self.annotation.machine_confidence and 
            isinstance(self.annotation.machine_confidence, dict)
        ):
            # Get the label with highest confidence
            top_label = max(
                self.annotation.machine_confidence.keys(),
                key=lambda k: self.annotation.machine_confidence[k]
            )
            top_confidence = float(self.annotation.machine_confidence[top_label])
            return top_confidence, top_label
        
        return 0.0, None
    
    def get_confidence_color(self, breaks=None):
        """
        Get a color for the confidence score based on quantile breaks.
        
        Args:
            breaks (list): List of confidence break points (e.g., [0.5, 0.75, 0.9])
                          If None, uses simple default thresholds.
        
        Returns:
            QColor: Color representing the confidence level
        """
        confidence, _ = self.get_machine_confidence()
        
        if confidence == 0:
            return QColor(128, 128, 128)  # Gray for no confidence
        
        # Use provided breaks or default thresholds
        if breaks is None or len(breaks) < 2:
            # Default 4-tier system
            if confidence <= 0.50:
                return QColor(220, 20, 60)  # Crimson red (<=50%)
            elif confidence <= 0.75:
                return QColor(255, 215, 0)  # Gold/yellow (50-75%)
            elif confidence <= 0.90:
                return QColor(144, 238, 144)  # Light green (75-90%)
            else:
                return QColor(34, 139, 34)  # Dark green (>90%)
        else:
            # Use quantile breaks
            if len(breaks) >= 3:
                # 4 categories (3 breaks)
                if confidence <= breaks[0]:
                    return QColor(220, 20, 60)  # Crimson red
                elif confidence <= breaks[1]:
                    return QColor(255, 215, 0)  # Gold/yellow
                elif confidence <= breaks[2]:
                    return QColor(144, 238, 144)  # Light green
                else:
                    return QColor(34, 139, 34)  # Dark green
            else:
                # 2 categories (2 breaks)
                if confidence <= breaks[0]:
                    return QColor(220, 20, 60)  # Crimson red
                elif confidence <= breaks[1]:
                    return QColor(255, 215, 0)  # Gold/yellow
                else:
                    return QColor(34, 139, 34)  # Dark green
