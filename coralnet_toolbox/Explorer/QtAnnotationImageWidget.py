import warnings

from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtCore import Qt, QTimer

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QWidget)


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

ANNOTATION_WIDTH = 5

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, data_item, widget_height=96, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.data_item = data_item
        self.annotation = data_item.annotation
        self.annotation_viewer = annotation_viewer

        self.widget_height = widget_height
        self.aspect_ratio = 1.0  # Default to a square aspect ratio
        self.pixmap = None       # Cache the original, unscaled pixmap

        self.animation_offset = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation_frame)
        self.animation_timer.setInterval(75)

        self.setup_ui()
        self.load_and_set_image()  # Changed from load_annotation_image

    def setup_ui(self):
        """Set up the basic UI with a label for the image."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: none;")

        layout.addWidget(self.image_label)

    def load_and_set_image(self):
        """Load image, calculate its aspect ratio, and set the widget's initial size."""
        try:
            # Use get_cropped_image_graphic() to get the QPixmap
            cropped_pixmap = self.annotation.get_cropped_image_graphic()
            if cropped_pixmap and not cropped_pixmap.isNull():
                self.pixmap = cropped_pixmap
                # Safely calculate aspect ratio
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

        # Trigger an initial size update
        self.update_height(self.widget_height)

    def update_height(self, new_height):
        """Updates the widget's height and rescales its width and content accordingly."""
        self.widget_height = new_height

        # Calculate the new width based on the stored aspect ratio
        new_width = int(self.widget_height * self.aspect_ratio)

        # Set the new fixed size for the entire widget
        self.setFixedSize(new_width, new_height)

        if self.pixmap:
            # Scale the cached pixmap to fit the new widget size, leaving room for the border
            # Note: We use the widget's new dimensions directly
            scaled_pixmap = self.pixmap.scaled(
                new_width - 8,  # Account for horizontal margins
                new_height - 8,  # Account for vertical margins
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        self.update()  # Schedule a repaint if needed

    # Replace update_size with update_height
    def update_size(self, new_size):
        """Kept for compatibility, redirects to update_height."""
        self.update_height(new_size)

    def set_selected(self, selected):
        """Set the selection state and update visual appearance."""
        was_selected = self.is_selected()

        # Update the shared data item
        self.data_item.set_selected(selected)

        # Start or stop the animation timer based on the new selection state
        if selected:
            if not self.animation_timer.isActive():
                self.animation_timer.start()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
            # Reset offset when deselected to ensure a consistent starting look
            self.animation_offset = 0

        # A repaint is needed if the selection state changed OR if the item remains
        # selected (to keep the animation running).
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

        # Get the label that is currently active (which could be a preview)
        effective_label = self.data_item.effective_label

        # Check if the active label is the special "Review" label (id == "-1")
        if effective_label and effective_label.id == "-1":
            # If it is, ALWAYS use a black pen for visibility against the white background.
            pen_color = QColor("black")
        else:
            # Otherwise, use the effective color from the data item.
            pen_color = self.data_item.effective_color

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
        """Handle mouse press events for selection."""
        if event.button() == Qt.LeftButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                self.annotation_viewer.handle_annotation_selection(self, event)
        # Ignore right mouse button clicks - don't call super() to prevent any selection
        elif event.button() == Qt.RightButton:
            event.ignore()
            return
        super().mousePressEvent(event)
