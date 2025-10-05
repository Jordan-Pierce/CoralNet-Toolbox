import warnings

import re
import uuid
import random

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QTimer, Qt, pyqtSignal, QMimeData, QRectF, QTimer, pyqtProperty
from PyQt5.QtGui import (QColor, QPainter, QPen, QBrush, QFontMetrics, QDrag, QBrush, QColor, QDrag,
                         QFontMetrics, QLinearGradient, QPainter, QPen)
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget,
                             QVBoxLayout, QColorDialog, QLineEdit, QDialog, QHBoxLayout,
                             QPushButton, QApplication, QGroupBox, QScrollArea,
                             QApplication, QGraphicsDropShadowEffect,
                             QSizePolicy, QWidget)

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class LabelDisplay(QWidget):
    """A widget responsible for the visual rendering and interaction of a label."""
    selected = pyqtSignal()  # Signal to emit when this display part is selected

    def __init__(self, label_instance, parent=None):
        super().__init__(parent)
        self.label = label_instance  # Reference back to the main Label container
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)  # Ensure hover events are captured accurately

    def enterEvent(self, event):
        """Handle mouse entering the widget."""
        self.label.is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        self.label.is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events for selection."""
        # We only care about left-clicks for selection here
        if event.button() == Qt.LeftButton:
            self.selected.emit()
        # Right-clicks are now handled by the main Label widget for drag-and-drop
        super().mousePressEvent(event)
        
    def paintEvent(self, event):
        """Paint the label widget with a modern look and feel."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)

        # --- 1. Background Gradient ---
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        base_color = QColor(self.label.color)
        if self.label.is_hovered:
            gradient.setColorAt(0, base_color.lighter(130))
            gradient.setColorAt(1, base_color.lighter(110))
        else:
            gradient.setColorAt(0, base_color.lighter(115))
            gradient.setColorAt(1, base_color)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

        # --- 2. Animated Selection Indicator ---
        if self.label.is_selected:
            pen_color = QColor(self.label.color).darker(150)
            pen_color.setAlpha(self.label.pulse_alpha)
            pen = QPen(pen_color)
            pen.setWidthF(2.5)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

        # --- 3. Dynamic Text Color ---
        r, g, b, _ = base_color.getRgb()
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        text_color = Qt.black if luminance > 0.5 else Qt.white
        painter.setPen(QPen(text_color))

        # --- 4. Draw Text ---
        font_metrics = QFontMetrics(painter.font())
        truncated_text = font_metrics.elidedText(self.label.short_label_code, Qt.ElideRight, int(rect.width() - 10))
        painter.drawText(rect, Qt.AlignCenter, truncated_text)
        

class Label(QWidget):
    colorChanged = pyqtSignal(QColor)
    selected = pyqtSignal(object)
    label_deleted = pyqtSignal(object)

    def __init__(self, short_label_code, long_label_code, color=QColor(255, 255, 255), label_id=None, pen_width=2):
        """Initialize the Label widget."""
        super().__init__()

        # --- Basic properties ---
        self.id = str(uuid.uuid4()) if label_id is None else label_id
        self.short_label_code = short_label_code
        self.long_label_code = long_label_code
        self.color = color
        self.pen_width = pen_width
        self.transparency = 128
        self.is_selected = False
        self.is_hovered = False
        self.drag_start_position = None

        # --- Layout and Child Widgets ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)

        # 1. The visual display part of the label
        self.display_widget = LabelDisplay(self)
        self.display_widget.selected.connect(self._handle_selection)

        # 2. The checkbox
        self.link_checkbox = QCheckBox()
        self.link_checkbox.setChecked(True)
        self.link_checkbox.setToolTip("Link this label's transparency to the slider")
        
        # Add widgets to the layout
        layout.addWidget(self.display_widget)
        layout.addWidget(self.link_checkbox)

        # --- Animation properties ---
        self._pulse_alpha = 128
        self._pulse_direction = 1
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_pulse_alpha)
        self.animation_timer.setInterval(50)

        # --- Widget settings ---
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(self.long_label_code)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

    def _handle_selection(self):
        """Internal slot to handle clicks from the display widget."""
        self.is_selected = not self.is_selected
        if self.is_selected:
            self.start_animation()
        else:
            self.stop_animation()
        self.display_widget.update()  # Triggers a repaint of the child widget
        self.selected.emit(self)

    def mousePressEvent(self, event):
        """Handle mouse press events for initiating drag."""
        if event.button() == Qt.RightButton:
            if not self.is_selected:
                self._handle_selection()  # Select the label if not already selected
            self.drag_start_position = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging."""
        if event.buttons() == Qt.RightButton and self.drag_start_position:
            if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
                return
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.id)
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)
            self.drag_start_position = None  # Reset after drag
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop dragging."""
        if event.button() == Qt.RightButton:
            self.drag_start_position = None
        super().mouseReleaseEvent(event)

    @property
    def is_linked(self):
        """Returns True if the checkbox is checked, False otherwise."""
        return self.link_checkbox.isChecked()

    def select(self):
        """Programmatically select the label."""
        if not self.is_selected:
            self.is_selected = True
            self.start_animation()
            self.display_widget.update()
            self.selected.emit(self)

    def deselect(self):
        """Programmatically deselect the label."""
        if self.is_selected:
            self.is_selected = False
            self.stop_animation()
            self.display_widget.update()

    def update_color(self):
        """Trigger a repaint to reflect color changes."""
        self.display_widget.update()

    def update_selection(self):
        """Trigger a repaint to reflect selection changes."""
        self.display_widget.update()

    def update_label_color(self, new_color: QColor):
        """Update the label's color and emit the colorChanged signal."""
        if self.color != new_color:
            self.color = new_color
            self.update_color()
            self.colorChanged.emit(new_color)

    def update_transparency(self, transparency):
        """Update the label's transparency value."""
        self.transparency = max(0, min(255, transparency))

    def update_pen_width(self, pen_width):
        """Update the label's pen width value."""
        self.pen_width = pen_width
        self.update_color()

    @pyqtProperty(int)
    def pulse_alpha(self):
        """Get the current pulse alpha for animation."""
        return self._pulse_alpha
    
    @pulse_alpha.setter
    def pulse_alpha(self, value):
        """Set the pulse alpha and update the widget."""
        self._pulse_alpha = int(max(0, min(255, value)))
        self.display_widget.update()

    def _update_pulse_alpha(self):
        """Update the pulse alpha for a heartbeat-like effect."""
        if self._pulse_direction == 1:
            self._pulse_alpha += 30
        else:
            self._pulse_alpha -= 10

        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50
            self._pulse_direction = 1
        
        self.display_widget.update()

    def start_animation(self):
        """Start the pulsing animation."""
        if not self.animation_timer.isActive():
            self.animation_timer.start()
    
    def stop_animation(self):
        """Stop the pulsing animation."""
        self.animation_timer.stop()
        self._pulse_alpha = 128
        self.display_widget.update()

    def delete_label(self):
        """Emit the label_deleted signal and schedule the widget for deletion."""
        self.label_deleted.emit(self)
        self.deleteLater()

    def to_dict(self):
        """Convert the label's properties to a dictionary."""
        return {
            'id': self.id,
            'short_label_code': self.short_label_code,
            'long_label_code': self.long_label_code,
            'color': self.color.getRgb(),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Label instance from a dictionary."""
        return cls(data['short_label_code'],
                   data['long_label_code'],
                   QColor(*data['color']),
                   data['id'])

    def __eq__(self, other):
        """Two labels are considered equal if their IDs are the same."""
        if not isinstance(other, Label):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        """The hash is based on the label's unique ID."""
        return hash(self.id)

    def __repr__(self):
        """Return a string representation of the Label object."""
        return (f"Label(id={self.id}, "
                f"short_label_code='{self.short_label_code}')")
        
    def __del__(self):
        """Clean up the timer when the label is deleted."""
        try:
            if hasattr(self, 'animation_timer') and self.animation_timer:
                self.animation_timer.stop()
        except RuntimeError:
            pass


class LabelWindow(QWidget):
    labelSelected = pyqtSignal(object)
    transparencyChanged = pyqtSignal(int)

    def __init__(self, main_window):
        """Initialize the LabelWindow widget."""
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.label_locked = False
        self.locked_label = None

        self.label_height = 30
        self.label_width = 50 
        
        # Setup UI components
        self.setup_ui()
        
        # Connections
        self.add_label_button.clicked.connect(self.open_add_label_dialog)
        self.edit_label_button.clicked.connect(self.open_edit_label_dialog)
        self.delete_label_button.clicked.connect(self.delete_active_label)

        # Initialize labels
        self.labels = []
        self.active_label = None

        # Add default label
        self.add_review_label()

        # Deselect at first
        self.active_label.deselect()

        self.show_confirmation_dialog = True
        self.setAcceptDrops(True)

    def setup_ui(self):
        """Set up the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create UI sections
        self.setup_actions_section()
        self.setup_labels_section()
        self.setup_counts_section()

    def setup_actions_section(self):
        """Set up the actions section of the UI."""
        # Create a QGroupBox for Label Actions
        self.actions_group = QGroupBox("Label Actions")
        actions_layout = QVBoxLayout()
        self.actions_group.setLayout(actions_layout)

        # Top Actions Bar
        self.actions_bar = QHBoxLayout()
        self.actions_bar.setContentsMargins(0, 0, 0, 0)
        self.actions_bar.setSpacing(0)

        self.add_label_button = QPushButton()
        self.add_label_button.setIcon(self.main_window.add_icon)
        self.add_label_button.setToolTip("Add Label")
        self.add_label_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_label_button.setFixedHeight(self.label_height)
        self.actions_bar.addWidget(self.add_label_button)

        self.delete_label_button = QPushButton()
        self.delete_label_button.setIcon(self.main_window.remove_icon)
        self.delete_label_button.setToolTip("Delete Label")
        self.delete_label_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.delete_label_button.setFixedHeight(self.label_height)
        self.delete_label_button.setEnabled(False)
        self.actions_bar.addWidget(self.delete_label_button)

        self.edit_label_button = QPushButton()
        self.edit_label_button.setIcon(self.main_window.edit_icon)
        self.edit_label_button.setToolTip("Edit Label / Merge Labels")
        self.edit_label_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_label_button.setFixedHeight(self.label_height)
        self.edit_label_button.setEnabled(False)
        self.actions_bar.addWidget(self.edit_label_button)

        self.label_lock_button = QPushButton()
        self.label_lock_button.setIcon(self.main_window.unlock_icon)
        self.label_lock_button.setToolTip("Label Unlocked")
        self.label_lock_button.setCheckable(True)
        self.label_lock_button.toggled.connect(self.toggle_label_lock)
        self.label_lock_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_lock_button.setFixedHeight(self.label_height)
        self.actions_bar.addWidget(self.label_lock_button)

        self.toggle_all_button = QPushButton()
        self.toggle_all_button.setIcon(get_icon("all.png"))
        self.toggle_all_button.setToolTip("Toggle All Labels")
        self.toggle_all_button.clicked.connect(self.toggle_all_labels)
        self.toggle_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_all_button.setFixedHeight(self.label_height)
        self.actions_bar.addWidget(self.toggle_all_button)

        # Filter/Search Bar
        self.filter_bar_layout = QHBoxLayout()
        self.filter_bar = QLineEdit()
        self.filter_bar.setPlaceholderText("Filter Labels")
        self.filter_bar.textChanged.connect(self.filter_labels)
        self.filter_bar_layout.addWidget(self.filter_bar)

        # Add layouts to the group box layout
        actions_layout.addLayout(self.actions_bar)
        actions_layout.addLayout(self.filter_bar_layout)

        # Add the group box to the main layout
        self.layout.addWidget(self.actions_group)

    def setup_labels_section(self):
        """Set up the labels section of the UI."""
        # Create a QGroupBox for Label Window
        self.labels_group = QGroupBox("Label Window")
        labels_layout = QVBoxLayout()
        self.labels_group.setLayout(labels_layout)

        # --- Add scroll area and label layout ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow scroll area to expand
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.labels_layout = QVBoxLayout(self.scroll_content)
        self.labels_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_layout.setSpacing(0)
        self.scroll_content.setLayout(self.labels_layout)
        self.scroll_area.setWidget(self.scroll_content)

        # Add scroll area to the group box layout
        labels_layout.addWidget(self.scroll_area)

        # Add the group box to the main layout
        self.layout.addWidget(self.labels_group)

    def setup_counts_section(self):
        """Set up the counts section of the UI."""
        # Create a QGroupBox for Counts
        self.counts_group = QGroupBox("Counts")
        counts_layout = QVBoxLayout()
        self.counts_group.setLayout(counts_layout)

        # Bottom Status Bar
        self.status_bar = QHBoxLayout()
        self.counts_layout = QVBoxLayout()

        self.label_count_display = QLineEdit("Labels: 1")
        self.label_count_display.setReadOnly(True)
        self.label_count_display.setStyleSheet("background-color: #F0F0F0;")
        self.counts_layout.addWidget(self.label_count_display)

        self.annotation_count_display = QLineEdit("Annotations: 0")
        self.annotation_count_display.setReadOnly(True)
        self.annotation_count_display.setStyleSheet("background-color: #F0F0F0;")
        self.annotation_count_display.returnPressed.connect(self.update_annotation_count_index)
        self.counts_layout.addWidget(self.annotation_count_display)

        self.status_bar.addLayout(self.counts_layout)

        # Add layout to the group box layout
        counts_layout.addLayout(self.status_bar)

        # Add the group box to the main layout
        self.layout.addWidget(self.counts_group)
        
    def showEvent(self, event):
        """Handle the show event to force a layout update, using a timer for startup."""
        super().showEvent(event)
        # Using QTimer.singleShot with a 0ms delay schedules this call to run
        # as soon as the current event processing is finished. This is crucial
        # on initial startup to ensure the parent window has fully resized
        # before we calculate the label widths.
        QTimer.singleShot(0, lambda: self.resizeEvent(None))
        
    def resizeEvent(self, event):
        """
        Handle resize events to explicitly set the width of the scrollable content.
        This is the definitive fix for the resizing issue upon re-parenting.
        """
        # Set the minimum width of the content widget to match the viewport's width.
        # This forces the vertical layout and its children (the labels) to expand.
        self.scroll_content.setMinimumWidth(self.scroll_area.viewport().width())
        
        # Call the parent class's implementation to handle the rest of the event.
        super().resizeEvent(event)

    def dragEnterEvent(self, event):
        """Accept drag events if they contain text."""
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop events to reorder labels."""
        label_id = event.mimeData().text()
        label = self.get_label_by_id(label_id)
        if label:
            self.labels.remove(label)
            self.labels.insert(self.calculate_new_index(event.pos()), label)
            self.reorganize_labels()

    def calculate_new_index(self, pos):
        """Calculate the new index for vertical drop based on y position."""
        return max(0, min(len(self.labels) - 1, pos.y() // self.label_height))

    def update_annotation_count_state(self):
        """Update the annotation count display based on the current selection."""
        if self.annotation_window.selected_tool == "select":
            self.annotation_count_display.setReadOnly(False)  # Make it editable
            self.annotation_count_display.setStyleSheet("background-color: white;")
        else:
            self.annotation_count_display.setReadOnly(True)  # Make it uneditable
            self.annotation_count_display.setStyleSheet("background-color: #F0F0F0;")

        # Update the annotation count display after a tool is switched
        self.update_annotation_count()

    def update_annotation_count(self):
        """Update the annotation count display with current selection and total count."""
        annotations = self.annotation_window.get_image_annotations()
        # Check if we're in Explorer mode
        if self.main_window.explorer_window and \
            hasattr(self.main_window, 'explorer_window') and \
            hasattr(self.main_window.explorer_window, 'annotation_viewer'):
            annotation_viewer = self.main_window.explorer_window.annotation_viewer
           
            # Priority 1: Always check for a selection in Explorer first.
            explorer_selected_count = len(annotation_viewer.selected_widgets)
            if explorer_selected_count > 0:
                if explorer_selected_count == 1:
                    text = "Annotation: 1"
                else:
                    text = f"Annotations: {explorer_selected_count}"
                self.annotation_count_display.setText(text)
                return  # Exit early, selection count is most important.
            
            # Priority 2: If no selection, THEN check for isolation mode.
            if annotation_viewer.isolated_mode:
                count = len(annotation_viewer.isolated_widgets)
                text = f"Annotations: {count}"
                self.annotation_count_display.setText(text)
                return  # Exit early
            
        # --- ORIGINAL FALLBACK LOGIC ---
        annotation_window_selected_count = len(self.annotation_window.selected_annotations)
        if annotation_window_selected_count == 0:
            text = f"Annotations: {len(annotations)}"
        elif annotation_window_selected_count > 1:
            text = f"Annotations: {annotation_window_selected_count}"
        else:
            try:
                selected_annotation = self.annotation_window.selected_annotations[0]
                current_idx = annotations.index(selected_annotation) + 1
                text = f"Annotation: {current_idx}/{len(annotations)}"
            except ValueError:
                text = "Annotations: ???"
        self.annotation_count_display.setText(text)

    def update_annotation_count_index(self):
        """Allow the user to manually update the annotation count index
        by directly editing the annotation_count_display field."""
        user_input = self.annotation_count_display.text().strip()

        # Try to extract a number from the user input
        number_match = re.search(r"(\d+)", user_input)
        if number_match:
            try:
                new_index = int(number_match.group(1))

                # Get all annotations to check range
                annotations = self.annotation_window.get_image_annotations()
                total_count = len(annotations)

                # Validate the index is within range
                if 1 <= new_index <= total_count:
                    # Convert to zero-based index
                    zero_based_index = new_index - 1

                    # First unselect any currently selected annotations
                    self.annotation_window.unselect_annotations()

                    # Select the annotation at the specified index
                    self.annotation_window.select_annotation(annotations[zero_based_index])

                    # Center on the selected annotation
                    self.annotation_window.center_on_annotation(annotations[zero_based_index])
            except (ValueError, IndexError):
                # In case of parsing error or index out of range
                pass

        # Update the display to show the current state (after changes)
        self.update_annotation_count()
        self.annotation_count_display.clearFocus()

    def update_label_count(self):
        """Update the label count display."""
        count = len(self.labels)
        self.label_count_display.setText(f"Labels: {count}")

    def update_labels_per_row(self):
        """Calculate and update the number of labels per row based on width."""
        available_width = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.labels_per_row = max(1, available_width // self.label_width)
        self.scroll_content.setFixedWidth(self.labels_per_row * self.label_width)

    def reorganize_labels(self):
        """
        Rearrange labels in the vertical layout and ensure the container is correctly sized.
        """
        # Force the container to match the viewport width before rearranging.
        self.scroll_content.setMinimumWidth(self.scroll_area.viewport().width())

        # Clear the existing layout
        while self.labels_layout.count():
            item = self.labels_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        
        # Add labels vertically
        for label in self.labels:
            self.labels_layout.addWidget(label)
        
        # Add a stretch to push all labels to the top
        self.labels_layout.addStretch()

    def open_add_label_dialog(self):
        """Open the dialog to add a new label."""
        dialog = AddLabelDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            short_label_code, long_label_code, color = dialog.get_label_details()
            if self.label_exists(short_label_code, long_label_code):
                QMessageBox.warning(self,
                                    "Label Exists",
                                    "A label with the same short and long name already exists.")
            else:
                new_label = self.add_label(short_label_code, long_label_code, color)
                self.set_active_label(new_label)

    def open_edit_label_dialog(self):
        """Open the dialog to edit the active label."""
        if self.active_label:
            dialog = EditLabelDialog(self, self.active_label)
            if dialog.exec_() == QDialog.Accepted:
                # Update the tooltip with the new long label code
                self.active_label.setToolTip(self.active_label.long_label_code)
                self.update_labels_per_row()
                self.reorganize_labels()

    def add_review_label(self):
        """Add a review label to the window and place it at the front of the label list."""
        # Create the label
        label = Label("Review", "Review", QColor(255, 255, 255), label_id="-1")
        # Connect
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        # Insert at the beginning of the labels list instead of appending
        self.labels.insert(0, label)
        # Update in LabelWindow
        self.update_labels_per_row()
        self.reorganize_labels()
        self.set_active_label(label)

    def add_label(self, short_label_code, long_label_code, color, label_id=None):
        """Add a new label to the window."""
        # Create the label
        label = Label(short_label_code, long_label_code, color, label_id)
        # Connect
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        self.labels.append(label)
        # Update in LabelWindow
        self.update_labels_per_row()
        self.reorganize_labels()
        self.set_active_label(label)
        # Update filter bars and label count
        self.update_label_count()
        self.main_window.image_window.update_search_bars()
        self.sync_all_masks_with_labels()
        QApplication.processEvents()

        return label
    
    def get_linked_labels(self):
        """Get a list of all labels whose transparency checkbox is checked."""
        return [label for label in self.labels if label.is_linked]

    def set_active_label(self, selected_label):
        """Set the currently active label, updating UI and emitting signals."""
        if self.active_label and self.active_label != selected_label:
            # Deselect the active label
            self.deselect_active_label()

        # Make the selected label active
        self.active_label = selected_label
        self.active_label.select()
        self.labelSelected.emit(selected_label)

        # Transparency changes are now instant - emit freely!
        self.transparencyChanged.emit(self.active_label.transparency)
        
        # OPTIMIZED: Skip expensive annotation updates in mask editing mode
        # Vector annotations don't need updates when switching labels in mask mode
        if not self.annotation_window._is_in_mask_editing_mode():
            self.update_annotations_with_label(selected_label)

        # Only enable edit/delete buttons if not locked
        if not self.label_locked:
            self.delete_label_button.setEnabled(self.active_label is not None)
            self.edit_label_button.setEnabled(self.active_label is not None)

        self.scroll_area.ensureWidgetVisible(self.active_label)

    def sync_all_masks_with_labels(self):
        """Sync all existing mask annotations with the current project labels."""
        for raster in self.main_window.image_window.raster_manager.rasters.values():
            if raster.mask_annotation:
                raster.mask_annotation.sync_label_map(self.labels)

    def set_mask_transparency(self, transparency):
        """Update the mask annotation's transparency for the current image."""
        transparency = max(0, min(255, transparency))  # Clamp to valid range
        mask = self.annotation_window.current_mask_annotation
        if mask:
            # ULTRA-FAST: New render-time transparency approach - no caching needed!
            # Update transparency for all linked labels
            linked_labels = self.get_linked_labels()
            if linked_labels:
                for label in linked_labels:
                    if label.id in mask.visible_label_ids:
                        # Update the label's transparency - now instant!
                        label.update_transparency(transparency)
                # Single call to update the mask - transparency applied at render time
                mask.update_transparency(transparency)
            else:
                # Fallback to original behavior for edge cases
                mask.update_transparency(transparency)

    def deselect_active_label(self):
        """Deselect the currently active label."""
        if self.active_label:
            self.active_label.deselect()

    def delete_active_label(self):
        """Delete the currently active label."""
        if self.active_label:
            self.delete_label(self.active_label)

    def update_annotations_with_label(self, label):
        """Update selected annotations based on the properties of the given label."""
        # OPTIMIZED: Skip this expensive operation in mask editing mode
        if self.annotation_window._is_in_mask_editing_mode():
            return
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        for annotation in self.annotation_window.selected_annotations:
            if annotation.label.id == label.id:
                # Get the transparency of the label
                transparency = self.get_label_transparency(label.id)
                # Update the annotation transparency
                annotation.update_transparency(transparency)

        # Make cursor normal again
        QApplication.restoreOverrideCursor()

    def get_label_color(self, label_id):
        """Get the color of a label by its ID."""
        for label in self.labels:
            if label.id == label_id:
                return label.color
        return None

    def get_label_transparency(self, label_id):
        """Get the transparency of a label by its ID."""
        for label in self.labels:
            if label.id == label_id:
                return label.transparency
        return None

    def get_review_label(self):
        """Get the review label."""
        return self.labels[0]

    def get_label_by_id(self, label_id, return_review=True):
        """Find and return a label by its ID."""
        for label in self.labels:
            if label.id == label_id:
                return label
        print(f"Warning: Label with ID '{label_id}' not found.")
        if return_review:
            for label in self.labels:
                if label.short_label_code == "Review" and label.long_label_code == "Review":
                    return label
        return None  # Return None if not found and not returning review label

    def get_label_by_codes(self, short_label_code, long_label_code, return_review=True):
        """Find and return a label by its short and long codes (case-insensitive)."""
        s_code = short_label_code.strip().lower()
        l_code = long_label_code.strip().lower()
        for label in self.labels:
            if s_code == label.short_label_code.strip().lower() and l_code == label.long_label_code.strip().lower():
                return label
        print(f"Warning: Label with codes '{short_label_code}'/'{long_label_code}' not found.")
        if return_review:
            for label in self.labels:
                if label.short_label_code == "Review" and label.long_label_code == "Review":
                    return label
        return None  # Return None if not found and not returning review label
    
    def get_label_by_short_code(self, short_label_code, return_review=True):
        """Find and return a label by its short code (case-insensitive)."""
        s_code = short_label_code.strip().lower()
        for label in self.labels:
            if s_code == label.short_label_code.strip().lower():
                return label
        print(f"Warning: Label with short code '{short_label_code}' not found.")

        if return_review:
            for label in self.labels:
                if label.short_label_code == "Review" and label.long_label_code == "Review":
                    return label
        return None  # Return None if not found and not returning review label

    def get_label_by_long_code(self, long_label_code, return_review=True):
        """Find and return a label by its long code (case-insensitive)."""
        l_code = long_label_code.strip().lower()
        for label in self.labels:
            if l_code == label.long_label_code.strip().lower():
                return label
        print(f"Warning: Label with long code '{long_label_code}' not found.")

        if return_review:
            for label in self.labels:
                if label.short_label_code == "Review" and label.long_label_code == "Review":
                    return label
        return None  # Return None if not found and not returning review label
    
    def get_label_map(self):
        """Return a dictionary mapping class IDs (integers starting from 1) to Label objects."""
        return {i + 1: label for i, label in enumerate(self.labels)}

    def label_exists(self, short_label_code, long_label_code, label_id=None):
        """Check if a label with the given codes or ID already exists (case-insensitive for codes)."""
        s_code = short_label_code.strip().lower()
        l_code = long_label_code.strip().lower()
        for label in self.labels:
            if s_code == label.short_label_code.strip().lower() and l_code == label.long_label_code.strip().lower():
                return True
        return False

    def add_label_if_not_exists(self, short_label_code, long_label_code=None, color=None, label_id=None):
        """Add a label if it doesn't exist and return it, or return existing matching label.

        Args:
            short_label_code: Short code for the label
            long_label_code: Long description for the label (defaults to short_label_code if None)
            color: QColor object for the label (will be randomly generated if None)
            label_id: Unique ID for the label (will be generated if None)

        Returns:
            Label: Either an existing matching label or a newly created one
        """
        # If long_label_code is None, use the short_label_code
        if long_label_code is None:
            long_label_code = short_label_code

        s_code = short_label_code.strip().lower()
        l_code = long_label_code.strip().lower()

        # First check if a label with the ID exists
        if label_id is not None:
            for label in self.labels:
                if label.id == label_id:
                    return label

        # Check if a label with matching short and long codes exists (case-insensitive)
        for label in self.labels:
            if (s_code == label.short_label_code.strip().lower() and
                l_code == label.long_label_code.strip().lower()):
                return label

        # Create default values if not provided
        if color is None:
            color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if label_id is None:
            label_id = str(uuid.uuid4())

        # Create a new label and return it
        new_label = self.add_label(short_label_code, long_label_code, color, label_id)
        return new_label

    def set_selected_label(self, label_id):
        """Set the active label based on the provided label ID."""
        for lbl in self.labels:
            if lbl.id == label_id:
                self.set_active_label(lbl)
                break

    def update_label_properties(self, label_to_update, new_short, new_long, new_color):
        """
        Updates the properties of a specific label and refreshes associated annotations.
        This is for a simple edit, not a merge.
        """
        # Update the label object's properties
        label_to_update.short_label_code = new_short
        label_to_update.long_label_code = new_long
        label_to_update.setToolTip(new_long)  # Update tooltip
        label_to_update.update_label_color(new_color)  # This already updates color and emits signal

        # Update all annotations that use this label to reflect the new color/properties
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == label_to_update.id:
                # Re-apply the label to trigger a style update if needed (e.g., color change)
                annotation.update_label(label_to_update)

        # Also, force the mask annotation to re-render to show the new color (only in mask editing mode).
        if self.annotation_window.mask_annotation and self.annotation_window._is_in_mask_editing_mode():
            self.annotation_window.mask_annotation.update_graphics_item()

        # Force a repaint of the label widget itself and reorganize the grid
        label_to_update.update()
        self.reorganize_labels()
        self.sync_all_masks_with_labels()
        print(f"Note: Label '{label_to_update.id}' updated successfully.")

    def merge_labels(self, source_label, target_label):
        """
        Merges a source label into a target label. This is a global operation.
        1. Re-assigns all annotations from source to target.
        2. Scrubs the source_label from ALL machine_confidence dictionaries in the project.
        3. Deletes the source label.
        """
        print(f"Merging label '{source_label.short_label_code}' into '{target_label.short_label_code}'.")
        
        # Iterate through ALL rasters to update every existing mask.
        for raster in self.main_window.image_window.raster_manager.rasters.values():
            if raster.mask_annotation is not None:
                mask_anno = raster.mask_annotation
                
                source_cid = mask_anno.label_id_to_class_id_map.get(source_label.id)
                target_cid = mask_anno.label_id_to_class_id_map.get(target_label.id)

                if source_cid and target_cid:
                    # Find all pixels belonging to the source class
                    pixels_to_reassign = (mask_anno.mask_data % mask_anno.LOCK_BIT) == source_cid
                    # Re-assign them to the target class, preserving their locked status
                    mask_anno.mask_data[pixels_to_reassign] = target_cid + mask_anno.LOCK_BIT

                    # Clean up the old source label from the mask's maps
                    mask_anno.class_id_to_label_map.pop(source_cid, None)
                    mask_anno.label_id_to_class_id_map.pop(source_label.id, None)

        # --- GLOBAL CLEANUP OPERATION ---
        all_annotations = self.annotation_window.annotations_dict.values()
        
        for annotation in all_annotations:
            if annotation.label == source_label:
                annotation.update_label(target_label)

            if annotation.machine_confidence:
                annotation.machine_confidence.pop(source_label, None)

        # --- FINALIZING THE MERGE ---
        if self.active_label == source_label:
            self.set_active_label(target_label)
            
        if source_label in self.labels:
            self.labels.remove(source_label)
            source_label.deleteLater()

        self.update_label_count()
        self.reorganize_labels()

        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            self.annotation_window.set_image(current_image_path)
            self.update_annotation_count()
            
        self.sync_all_masks_with_labels()

        # After the merge, refresh the view of the currently displayed mask (only in mask editing mode).
        current_mask = self.annotation_window.current_mask_annotation
        if current_mask and self.annotation_window._is_in_mask_editing_mode():
            current_mask.update_graphics_item()

    def delete_label(self, label):
        """Delete the specified label and its associated annotations after confirmation."""
        if (label.short_label_code == "Review" and
                label.long_label_code == "Review" and
                label.color == QColor(255, 255, 255)):
            QMessageBox.warning(self,
                                "Cannot Delete Label",
                                "The 'Review' label cannot be deleted.")
            return

        if self.show_confirmation_dialog:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Confirm Delete")
            msg_box.setText("Are you sure you want to delete this label?\n"
                            "This will delete all associated annotations.")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            checkbox = QCheckBox("Do not show this message again")
            msg_box.setCheckBox(checkbox)

            result = msg_box.exec_()

            if checkbox.isChecked():
                self.show_confirmation_dialog = False

            if result == QMessageBox.No:
                return

        # Iterate through ALL rasters in the project to update every existing mask.
        for raster in self.main_window.image_window.raster_manager.rasters.values():
            # Only act on masks that have already been created (lazy-loading).
            if raster.mask_annotation is not None:
                mask_anno = raster.mask_annotation
                
                # Get the class ID using the fast, direct lookup.
                class_id_to_clear = mask_anno.label_id_to_class_id_map.get(label.id)
                if class_id_to_clear:
                    mask_anno.clear_pixels_for_class(class_id_to_clear)
                    # Remove the label from the mask's internal maps to keep them clean
                    mask_anno.class_id_to_label_map.pop(class_id_to_clear, None)
                    mask_anno.label_id_to_class_id_map.pop(label.id, None)

        # If the currently visible mask was affected, refresh its view (only in mask editing mode).
        current_mask = self.annotation_window.current_mask_annotation
        if current_mask and self.annotation_window._is_in_mask_editing_mode():
            current_mask.update_graphics_item()

        # Store affected image paths before deletion to update them later
        affected_images = set()
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == label.id:
                affected_images.add(annotation.image_path)

        # Remove from the LabelWindow
        self.labels.remove(label)
        label.deleteLater()

        # Delete annotations associated with the label
        self.annotation_window.delete_label_annotations(label)

        # Reset active label if it was deleted
        if self.active_label == label:
            self.active_label = None
            if self.labels:
                self.set_active_label(self.labels[0])

        # Update the LabelWindow
        self.update_labels_per_row()
        self.reorganize_labels()
        self.update_label_count()

        # Explicitly update affected images in the image window
        for image_path in affected_images:
            self.main_window.image_window.update_image_annotations(image_path)
            
        # Update the label map in the annotation window
        self.sync_all_masks_with_labels()

    def cycle_labels(self, direction):
        """Cycle through VISIBLE labels in the specified direction (1 for down/next, -1 for up/previous)."""
        # 1. Get a list of currently visible labels from the master list.
        visible_labels = [label for label in self.labels if label.isVisible()]

        # 2. If no labels are visible (e.g., filter matches nothing), do nothing.
        if not visible_labels:
            return

        try:
            # 3. Find the index of the current active label within the VISIBLE list.
            # This will raise a ValueError if the active label is not visible or doesn't exist.
            current_idx = visible_labels.index(self.active_label)
            
            # 4. Calculate the new index, wrapping around the visible list.
            new_idx = (current_idx + direction) % len(visible_labels)
            
            # 5. Set the new active label from the visible list.
            self.set_active_label(visible_labels[new_idx])
        except ValueError:
            # This block runs if the active label was not found in the visible list.
            # In this case, simply select the first available visible label.
            self.set_active_label(visible_labels[0])

    def toggle_label_lock(self, checked):
        """Toggle between lock and unlock states"""
        # Check if select tool is active, if not, revert the button state and return
        if self.main_window.annotation_window.selected_tool != "select":
            self.label_lock_button.setChecked(False)  # Revert the toggle
            return

        if checked:
            self.label_lock_button.setIcon(self.main_window.lock_icon)
            self.label_lock_button.setToolTip("Label Locked")  # Changed
            # Set add, edit, and delete label to disabled
            self.add_label_button.setEnabled(False)
            self.delete_label_button.setEnabled(False)
            self.edit_label_button.setEnabled(False)
            # Set the active label to locked
            self.locked_label = self.active_label
        else:
            self.label_lock_button.setIcon(self.main_window.unlock_icon)
            self.label_lock_button.setToolTip("Label Unlocked")  # Changed
            # Set add, edit, and delete label to enabled
            self.add_label_button.setEnabled(True)
            self.delete_label_button.setEnabled(True)
            self.edit_label_button.setEnabled(True)
            # Reset the locked label
            self.locked_label = None

        # Set the label_locked attribute
        self.label_locked = checked

    def unlock_label_lock(self):
        """Unlock the label lock by unchecking the lock button."""
        # Triggers the signal to toggle_label_lock method
        self.label_lock_button.setChecked(False)

    def toggle_all_labels(self):
        """Toggle all label checkboxes: check all if not all checked, uncheck all if all checked."""
        if not self.labels:
            return
        
        # Check if all labels are currently checked
        all_checked = all(label.is_linked for label in self.labels)
        
        # Toggle: if all checked, uncheck all; otherwise check all
        new_state = not all_checked
        
        for label in self.labels:
            label.link_checkbox.setChecked(new_state)

    def filter_labels(self, filter_text):
        """Filter labels by text."""
        filter_text = filter_text.strip().lower()
        for label in self.labels:
            label.setVisible(
                filter_text in label.short_label_code.lower() or filter_text in label.long_label_code.lower()
            )


class AddLabelDialog(QDialog):
    def __init__(self, label_window, parent=None):
        """Initialize the AddLabelDialog."""
        super().__init__(parent)
        self.label_window = label_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Add Label")
        self.setObjectName("AddLabelDialog")

        self.layout = QVBoxLayout(self)

        self.short_label_input = QLineEdit(self)
        self.short_label_input.setPlaceholderText("Short Label (max 10 characters)")
        self.short_label_input.setMaxLength(10)
        self.layout.addWidget(self.short_label_input)

        self.long_label_input = QLineEdit(self)
        self.long_label_input.setPlaceholderText("Long Label")
        self.layout.addWidget(self.long_label_input)

        self.color_button = QPushButton("Select Color", self)
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)

        self.color = self.generate_random_color()
        self.update_color_button()
        
    def showEvent(self, event):
        """Handle the show event for the dialog."""
        super().showEvent(event)
        self.label_window.annotation_window.unselect_annotations()  # Unselect any annotations when dialog opens

    def generate_random_color(self):
        """Generate a random QColor."""
        return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update_color_button(self):
        """Update the color button's background color."""
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        """Open a color dialog to select the label color."""
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def get_label_details(self):
        """Return the entered short label, long label, and selected color."""
        return self.short_label_input.text(), self.long_label_input.text(), self.color

    def validate_and_accept(self):
        """Validate the input fields and accept the dialog if valid."""
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        # Check if the label already exists
        label_exists = self.label_window.label_exists(short_label_code, long_label_code)

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
        elif label_exists:
            QMessageBox.warning(self, "Label Exists", "A label with the same short and long name already exists.")
        else:
            self.accept()


class EditLabelDialog(QDialog):
    def __init__(self, label_window, label, parent=None):
        """Initialize the EditLabelDialog."""
        super().__init__(parent)
        self.label_window = label_window
        self.label = label

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Edit Label")
        self.setObjectName("EditLabelDialog")

        self.layout = QVBoxLayout(self)

        self.short_label_input = QLineEdit(self.label.short_label_code, self)
        self.short_label_input.setPlaceholderText("Short Label (max 10 characters)")
        # Allow existing longer short codes
        if len(self.label.short_label_code) <= 10:
            self.short_label_input.setMaxLength(10)
        self.layout.addWidget(self.short_label_input)

        self.long_label_input = QLineEdit(self.label.long_label_code, self)
        self.long_label_input.setPlaceholderText("Long Label")
        self.layout.addWidget(self.long_label_input)

        self.color_button = QPushButton("Select Color", self)
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)

        self.color = self.label.color
        self.update_color_button()
        
    def showEvent(self, event):
        """Handle the show event for the dialog."""
        super().showEvent(event)
        self.label_window.annotation_window.unselect_annotations()  # Unselect any annotations when dialog opens

    def update_color_button(self):
        """Update the color button's background color."""
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        """Open a color dialog to select the label color."""
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def get_edited_details(self):
        """Return the new details entered by the user."""
        return (
            self.short_label_input.text().strip(),
            self.long_label_input.text().strip(),
            self.color
        )

    def validate_and_accept(self):
        """Validate input and signal the LabelWindow to perform the update or merge."""
        if self.label.short_label_code == 'Review' and self.label.long_label_code == 'Review':  # Simplified check
            QMessageBox.warning(self, "Cannot Edit Label", "The 'Review' label cannot be edited.")
            return

        new_short, new_long, new_color = self.get_edited_details()

        if not new_short or not new_long:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
            return

        # Use the improved, case-insensitive search and ask for None on failure
        existing_label = self.label_window.get_label_by_codes(new_short, new_long, return_review=False)

        if existing_label and existing_label.id != self.label.id:
            # --- MERGE PATH ---
            
            # Construct a more informative message for the user.
            title = "Merge Labels?"
            text = (f"A label with these codes ('{existing_label.short_label_code}') already exists.\n\n"
                    f"Do you want to merge all annotations from '{self.label.short_label_code}' "
                    f"into '{existing_label.short_label_code}'?")
            
            informative_text = (
                "<b>This action has the following consequences:</b><br>"
                "<ul>"
                "<li>All annotations will be reassigned.</li>"
                "<li>The original label ('{s_code}') will be deleted.</li>"
                "<li>For annotations with machine predictions, the confidence score for '{s_code}' "
                "will be <b>removed</b>, and the score for '{e_code}' will be kept if it exists."
                "</ul>"
            ).format(s_code=self.label.short_label_code, e_code=existing_label.short_label_code)

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle(title)
            msg_box.setText(text)
            msg_box.setInformativeText(informative_text)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            reply = msg_box.exec_()

            if reply == QMessageBox.Yes:
                # Tell LabelWindow to perform the merge.
                self.label_window.merge_labels(source_label=self.label, target_label=existing_label)
                self.accept()
            else:
                return  # User cancelled the merge.
        else:
            # --- EDIT PATH ---
            self.label_window.update_label_properties(
                label_to_update=self.label,
                new_short=new_short,
                new_long=new_long,
                new_color=new_color
            )
            self.accept()