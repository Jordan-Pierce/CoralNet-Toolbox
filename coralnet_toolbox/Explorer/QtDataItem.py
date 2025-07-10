import warnings

import os

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import QGraphicsEllipseItem, QStyle, QVBoxLayout, QLabel, QWidget, QGraphicsItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_SIZE = 15
POINT_WIDTH = 3

ANNOTATION_WIDTH = 5


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPointItem(QGraphicsEllipseItem):
    """
    A custom QGraphicsEllipseItem that gets its state and appearance
    directly from an associated AnnotationDataItem.
    """

    def __init__(self, data_item):
        """
        Initializes the point item.

        Args:
            data_item (AnnotationDataItem): The data item that holds the state
                                            for this point.
        """
        # Initialize the ellipse with a placeholder rectangle; its position will be set later.
        super(EmbeddingPointItem, self).__init__(0, 0, POINT_SIZE, POINT_SIZE)

        # Store a direct reference to the data item
        self.data_item = data_item

        # Set the item's flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        # Set initial appearance from the data_item
        self.setPen(QPen(QColor("black"), POINT_WIDTH))
        self.setBrush(self.data_item.effective_color)

        # Set the position of the point based on the data item's embedding coordinates
        self.setPos(self.data_item.embedding_x, self.data_item.embedding_y)
        # Set the tooltip with detailed information
        self.setToolTip(self.data_item.get_tooltip_text())
        
    def update_tooltip(self):
        """Updates the tooltip by fetching the latest text from the data item."""
        self.setToolTip(self.data_item.get_tooltip_text())

    def paint(self, painter, option, widget):
        """
        Custom paint method to ensure the point's color is always in sync with
        the AnnotationDataItem and to prevent the default selection box from
        being drawn.
        """
        # Dynamically get the latest color from the central data item.
        # This ensures that preview color changes are reflected instantly.
        self.setBrush(self.data_item.effective_color)

        # Remove the 'State_Selected' flag from the style options before painting.
        # This is the key to preventing Qt from drawing the default dotted
        # selection rectangle around the item.
        option.state &= ~QStyle.State_Selected

        # Call the base class's paint method to draw the ellipse
        super(EmbeddingPointItem, self).paint(painter, option, widget)
        
        
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

        self.animation_offset = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation_frame)
        self.animation_timer.setInterval(75)

        self.setup_ui()
        self.load_and_set_image()

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

    def load_and_set_image(self):
        """Load image, calculate its aspect ratio, and set the widget's initial size."""
        try:
            cropped_pixmap = self.annotation.get_cropped_image_graphic()
            if cropped_pixmap and not cropped_pixmap.isNull():
                self.pixmap = cropped_pixmap
                if self.pixmap.height() > 0:
                    self.aspect_ratio = self.pixmap.width() / self.pixmap.height()
                else:
                    self.aspect_ratio = 1.0
            else:
                self.image_label.setText("No Image\nAvailable")
                self.pixmap = None
                self.aspect_ratio = 1.0
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.image_label.setText("Error\nLoading Image")
            self.pixmap = None
            self.aspect_ratio = 1.0
            
        self.update_height(self.widget_height)
        # Set the initial tooltip
        self.update_tooltip()

    def update_height(self, new_height):
        """Updates the widget's height and rescales its width and content accordingly."""
        self.widget_height = new_height
        new_width = int(self.widget_height * self.aspect_ratio)
        self.setFixedSize(new_width, new_height)
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                new_width - 8,
                new_height - 8,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
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
            self.animation_offset = 0

        # Trigger a repaint to show the new selection state (border, etc.)
        self.update()

    def is_selected(self):
        """Return whether this widget is selected via the data item."""
        return self.data_item.is_selected

    def _update_animation_frame(self):
        """Update the animation offset and schedule a repaint."""
        self.animation_offset = (self.animation_offset + 1) % 20
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
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([2, 3])
            pen.setDashOffset(self.animation_offset)
        else:
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setStyle(Qt.SolidLine)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        width = pen.width()
        half_width = (width - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)
        painter.drawRect(rect)

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
        self.embedding_id = embedding_id if embedding_id is not None else 0
        
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label
        
        # To store pre-formatted top-k prediction details
        self.prediction_details = None
        # To store prediction probabilities for sorting
        self.prediction_probabilities = None
        
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
        """Get the effective confidence value."""
        # First check if prediction probabilities are available from model predictions
        if hasattr(self, 'prediction_probabilities') and self.prediction_probabilities is not None:
            if len(self.prediction_probabilities) > 0:
                # Use the maximum probability for confidence sorting
                return float(np.max(self.prediction_probabilities))
        
        # Fallback to existing confidence values
        if self.annotation.verified and hasattr(self.annotation, 'user_confidence') and self.annotation.user_confidence:
            return list(self.annotation.user_confidence.values())[0]
        elif hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        return 0.0
