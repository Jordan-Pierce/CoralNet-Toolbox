import warnings

import re
import uuid
import random

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFontMetrics, QDrag
from PyQt5.QtWidgets import (QGridLayout, QScrollArea, QMessageBox, QCheckBox, QWidget,
                             QVBoxLayout, QColorDialog, QLineEdit, QDialog, QHBoxLayout,
                             QPushButton, QApplication, QGroupBox, QLabel)

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Label(QWidget):
    colorChanged = pyqtSignal(QColor)
    selected = pyqtSignal(object)  # Signal to emit the selected label
    label_deleted = pyqtSignal(object)  # Signal to emit when the label is deleted

    def __init__(self, short_label_code, long_label_code, color=QColor(255, 255, 255), label_id=None):
        """Initialize the Label widget."""
        super().__init__()

        self.id = str(uuid.uuid4()) if label_id is None else label_id
        self.short_label_code = short_label_code
        self.long_label_code = long_label_code
        self.color = color
        self.transparency = 64
        self.is_selected = False

        # Set the fixed width and height
        self.fixed_width = 100  # -20 for buffer
        self.fixed_height = 30

        self.setFixedWidth(self.fixed_width)
        self.setFixedHeight(self.fixed_height)
        self.setCursor(Qt.PointingHandCursor)

        # Set tooltip for long label
        self.setToolTip(self.long_label_code)

        self.drag_start_position = None

    def mousePressEvent(self, event):
        """Handle mouse press events for selection and initiating drag."""
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update_selection()
            self.selected.emit(self)  # Emit the selected signal

        if event.button() == Qt.RightButton:
            self.is_selected = not self.is_selected
            self.update_selection()
            self.selected.emit(self)  # Emit the selected signal
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging."""
        if event.buttons() == Qt.RightButton and self.drag_start_position:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.id)
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop dragging."""
        if event.button() == Qt.RightButton:
            self.drag_start_position = None

    def select(self):
        """Select the label."""
        if not self.is_selected:
            self.is_selected = True
            self.update_selection()
            self.selected.emit(self)

    def deselect(self):
        """Deselect the label."""
        if self.is_selected:
            self.is_selected = False
            self.update_selection()

    def update_color(self):
        """Trigger a repaint to reflect color changes."""
        self.update()  # Trigger a repaint

    def update_selection(self):
        """Trigger a repaint to reflect selection changes."""
        self.update()  # Trigger a repaint

    def update_label_color(self, new_color: QColor):
        """Update the label's color and emit the colorChanged signal."""
        if self.color != new_color:
            self.color = new_color
            self.update_color()
            self.colorChanged.emit(new_color)

    def update_transparency(self, transparency):
        """Update the label's transparency value."""
        self.transparency = transparency

    def delete_label(self):
        """Emit the label_deleted signal and schedule the widget for deletion."""
        self.label_deleted.emit(self)
        self.deleteLater()

    def paintEvent(self, event):
        """Paint the label widget with its color, text, and selection state."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate the height based on the text height
        font_metrics = QFontMetrics(painter.font())
        text_height = font_metrics.height()
        self.setFixedHeight(text_height + 20)  # padding

        # Draw the outer rectangle with a light transparent fill
        outer_color = QColor(self.color)
        outer_color.setAlpha(50)
        painter.setBrush(QBrush(outer_color, Qt.SolidPattern))

        # Set the border color based on selection status
        if self.is_selected:
            painter.setPen(QPen(Qt.black, 4, Qt.DashLine))
        else:
            # Normal border with the color of the label
            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))

        # Draw the outer rectangle
        painter.drawRect(0, 0, self.width(), self.height())

        # Set the text color to black
        painter.setPen(QPen(Qt.black))

        # Truncate the text if it exceeds the width
        truncated_text = font_metrics.elidedText(self.short_label_code, Qt.ElideRight, self.width() - self.height())
        painter.drawText(12, 0, self.width() - self.height(), self.height(), Qt.AlignVCenter, truncated_text)

        super().paintEvent(event)

    def to_dict(self):
        """Convert the label's properties to a dictionary."""
        return {
            'id': self.id,
            'short_label_code': self.short_label_code,
            'long_label_code': self.long_label_code,
            'color': self.color.getRgb()
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Label instance from a dictionary."""
        return cls(data['short_label_code'], 
                   data['long_label_code'], 
                   QColor(*data['color']),
                   data['id'])

    def __repr__(self):
        """Return a string representation of the Label object."""
        return (f"Label(id={self.id}, "
                f"short_label_code={self.short_label_code}, "
                f"long_label_code={self.long_label_code}, "
                f"color={self.color.name()})")


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
        self.label_width = 100
        self.labels_per_row = 1  # Initial value, will be updated

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create a groupbox and set its title and style sheet
        self.groupBox = QGroupBox("Label Window")

        self.groupBoxLayout = QVBoxLayout()
        self.groupBox.setLayout(self.groupBoxLayout)

        # Top bar with Add Label, Edit Label, and Delete Label buttons
        self.top_bar = QHBoxLayout()
        self.add_label_button = QPushButton()
        self.add_label_button.setIcon(self.main_window.add_icon)
        self.add_label_button.setToolTip("Add Label")
        self.add_label_button.setFixedSize(self.label_width, self.label_height)
        self.top_bar.addWidget(self.add_label_button)

        self.delete_label_button = QPushButton()
        self.delete_label_button.setIcon(self.main_window.remove_icon)
        self.delete_label_button.setToolTip("Delete Label")
        self.delete_label_button.setFixedSize(self.label_width, self.label_height)
        self.delete_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.delete_label_button)

        self.edit_label_button = QPushButton()
        self.edit_label_button.setIcon(self.main_window.edit_icon)
        self.edit_label_button.setToolTip("Edit Label")
        self.edit_label_button.setFixedSize(self.label_width, self.label_height)
        self.edit_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.edit_label_button)
        
        # Lock button
        self.label_lock_button = QPushButton()
        self.label_lock_button.setIcon(self.main_window.unlock_icon)
        self.label_lock_button.setToolTip("Label Unlocked")
        self.label_lock_button.setCheckable(True)
        self.label_lock_button.toggled.connect(self.toggle_label_lock)
        self.label_lock_button.setFixedSize(self.label_height, self.label_height)
        self.top_bar.addWidget(self.label_lock_button)
        
        # Filter bar for labels
        self.filter_bar = QLineEdit()
        self.filter_bar.setPlaceholderText("Filter Labels")
        self.filter_bar.textChanged.connect(self.filter_labels)
        self.filter_bar.setFixedWidth(150)
        self.top_bar.addWidget(self.filter_bar)

        self.top_bar.addStretch()  # Add stretch to the right side
        
        # Add label count display
        self.label_count_display = QLineEdit("")
        self.label_count_display.setReadOnly(True)  # Make it uneditable
        self.label_count_display.setStyleSheet("background-color: #F0F0F0;")
        self.label_count_display.setFixedWidth(100)  # Set a reasonable fixed width
        self.top_bar.addWidget(self.label_count_display)
        
        # Add annotation count display
        self.annotation_count_display = QLineEdit("Annotations: 0")
        self.annotation_count_display.setReadOnly(True)  # Make it uneditable
        self.annotation_count_display.setStyleSheet("background-color: #F0F0F0;")
        self.annotation_count_display.setFixedWidth(150)
        self.annotation_count_display.returnPressed.connect(self.update_annotation_count_index)
        self.top_bar.addWidget(self.annotation_count_display)

        # Scroll area for labels
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_content)

        # Add layouts to the groupbox layout
        self.groupBoxLayout.addLayout(self.top_bar)
        self.groupBoxLayout.addWidget(self.scroll_area)

        # Add the groupbox to the main layout
        self.main_layout.addWidget(self.groupBox)

        # Connections
        self.add_label_button.clicked.connect(self.open_add_label_dialog)
        self.edit_label_button.clicked.connect(self.open_edit_label_dialog)
        self.delete_label_button.clicked.connect(self.delete_active_label)

        # Initialize labels
        self.labels = []
        self.active_label = None

        # Add default label
        default_short_label_code = "Review"
        default_long_label_code = "Review"
        default_color = QColor(255, 255, 255)  # White color
        self.add_label(default_short_label_code, default_long_label_code, default_color, label_id="-1")
        
        # Deselect at first
        self.active_label.deselect()

        self.show_confirmation_dialog = True
        self.setAcceptDrops(True)

    def resizeEvent(self, event):
        """Handle resize events to update the label grid layout."""
        super().resizeEvent(event)
        self.update_labels_per_row()
        self.reorganize_labels()
        
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
        """Calculate the grid index based on the drop position."""
        row = pos.y() // self.label_width
        col = pos.x() // self.label_width
        return row * self.labels_per_row + col
    
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
        selected_count = len(self.annotation_window.selected_annotations)

        if selected_count == 0:
            text = f"Annotations: {len(annotations)}"
        elif selected_count > 1:
            text = f"Annotations: {selected_count}"
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
        self.label_count_display.setText(f"# Labels: {count}")

    def update_labels_per_row(self):
        """Calculate and update the number of labels per row based on width."""
        available_width = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.labels_per_row = max(1, available_width // self.label_width)
        self.scroll_content.setFixedWidth(self.labels_per_row * self.label_width)

    def reorganize_labels(self):
        """Rearrange labels in the grid layout based on the current order and labels_per_row."""
        for i, label in enumerate(self.labels):
            row = i // self.labels_per_row
            col = i % self.labels_per_row
            self.grid_layout.addWidget(label, row, col)

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
        QApplication.processEvents()

        return label

    def set_active_label(self, selected_label):
        """Set the currently active label, updating UI and emitting signals."""
        if self.active_label and self.active_label != selected_label:
            # Deselect the active label
            self.deselect_active_label()
    
        # Make the selected label active
        self.active_label = selected_label
        self.active_label.select()
        self.labelSelected.emit(selected_label)

        # Update the transparency slider with the new label's transparency
        self.transparencyChanged.emit(self.active_label.transparency)
        # Update annotations (locked, transparency)
        self.update_annotations_with_label(selected_label)

        # Only enable edit/delete buttons if not locked
        if not self.label_locked:
            self.delete_label_button.setEnabled(self.active_label is not None)
            self.edit_label_button.setEnabled(self.active_label is not None)
 
        self.scroll_area.ensureWidgetVisible(self.active_label)

    def set_label_transparency(self, transparency):
        """Set the transparency for the active label and its associated annotations."""
        if not self.active_label:
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if self.active_label.transparency != transparency:
            # Update the active label's transparency
            self.active_label.update_transparency(transparency)
            # Update the transparency of all annotations with the active label
            for annotation in self.annotation_window.annotations_dict.values():
                if annotation.label.id == self.active_label.id:
                    annotation.update_transparency(transparency)

            self.annotation_window.scene.update()
            self.annotation_window.viewport().update()
            
        # Make cursor normal again
        QApplication.restoreOverrideCursor()

    def set_all_labels_transparency(self, transparency):
        """Set the transparency for all labels and annotations."""
        for label in self.labels:
            label.update_transparency(transparency)

        for annotation in self.annotation_window.annotations_dict.values():
            annotation.update_transparency(transparency)

        self.annotation_window.scene.update()
        self.annotation_window.viewport().update()

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

    def get_label_by_id(self, label_id):
        """Find and return a label by its ID."""
        for label in self.labels:
            if label.id == label_id:
                return label
        return None

    def get_label_by_codes(self, short_label_code, long_label_code):
        """Find and return a label by its short and long codes."""
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                return label
        return None

    def get_label_by_short_code(self, short_label_code):
        """Find and return a label by its short code."""
        for label in self.labels:
            if short_label_code == label.short_label_code:
                return label
        return None

    def get_label_by_long_code(self, long_label_code):
        """Find and return a label by its long code."""
        for label in self.labels:
            if long_label_code == label.long_label_code:
                return label
        return None

    def label_exists(self, short_label_code, long_label_code, label_id=None):
        """Check if a label with the given codes or ID already exists."""
        for label in self.labels:
            if label_id is not None and label.id == label_id:
                return True
            if label.short_label_code == short_label_code:
                return True
            if label.long_label_code == long_label_code:
                return True
        return False

    def add_label_if_not_exists(self, short_label_code, long_label_code, color, label_id=None):
        """Add a label only if it doesn't already exist."""
        if not self.label_exists(short_label_code, long_label_code, label_id):
            self.add_label(short_label_code, long_label_code, color, label_id)

    def set_selected_label(self, label_id):
        """Set the active label based on the provided label ID."""
        for lbl in self.labels:
            if lbl.id == label_id:
                self.set_active_label(lbl)
                break

    def edit_labels(self, old_label, new_label, delete_old=False):
        """Update annotations from old_label to new_label, optionally deleting the old one."""
        # Update annotations to use the new label
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == old_label.id:
                annotation.update_label(new_label)

        if delete_old:
            # Remove the old label
            self.labels.remove(old_label)
            old_label.deleteLater()
            self.update_label_count()

        # Update the active label if necessary
        if self.active_label == old_label:
            self.set_active_label(new_label)

        self.update_labels_per_row()
        self.reorganize_labels()

        # Refresh the scene with the new label (if applicable)
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            self.annotation_window.set_image(current_image_path)

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

    def handle_wasd_key(self, key):
        """Handle WASD key presses to navigate the label grid."""
        if not self.active_label or self.label_locked:
            return

        # Get all labels from grid
        grid_labels = []
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                grid_labels.append(item.widget())

        if not grid_labels:
            return

        try:
            current_index = grid_labels.index(self.active_label)
        except ValueError:
            # If active label not in grid, select first label
            if grid_labels:
                self.set_active_label(grid_labels[0])
            return
        
        if key == Qt.Key_W:
            new_index = current_index - self.labels_per_row
        elif key == Qt.Key_S:
            new_index = current_index + self.labels_per_row
        elif key == Qt.Key_A:
            new_index = current_index - 1 if current_index % self.labels_per_row != 0 else current_index
        elif key == Qt.Key_D:
            new_index = current_index + 1 if (current_index + 1) % self.labels_per_row != 0 else current_index
        else:
            return

        if 0 <= new_index < len(grid_labels):
            self.set_active_label(grid_labels[new_index])

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
        
    def filter_labels(self, filter_text):
        """Filter labels based on the filter text and rebuild the grid with matching labels"""
        # Unselect the selected annotation
        self.annotation_window.unselect_annotations()
        # Unselect the active label
        self.deselect_active_label()
        
        # Get the filter text in lowercase
        filter_text = filter_text.lower()
        
        # Clear all widgets from the grid layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Filter labels that match the filter criteria
        filtered_labels = [
            label for label in self.labels 
            if filter_text in label.short_label_code.lower() 
            or filter_text in label.long_label_code.lower()
        ]
        
        # Add matching labels to the grid
        for i, label in enumerate(filtered_labels):
            row = i // self.labels_per_row
            col = i % self.labels_per_row
            self.grid_layout.addWidget(label, row, col)
            label.show()
        
        # If we have an active label that's no longer visible, select first visible label
        if self.active_label and self.active_label not in filtered_labels:
            if filtered_labels:
                self.set_active_label(filtered_labels[0])
            else:
                self.active_label = None
                self.delete_label_button.setEnabled(False)
                self.edit_label_button.setEnabled(False)


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

    def update_color_button(self):
        """Update the color button's background color."""
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        """Open a color dialog to select the label color."""
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def validate_and_accept(self):
        """Validate the input fields, handle potential merges, and accept the dialog."""
        # Cannot edit Review
        if self.label.short_label_code == 'Review' and self.label.long_label_code == 'Review':
            QMessageBox.warning(self, 
                                "Cannot Edit Label", 
                                "The 'Review' label cannot be edited.")
            return

        # Can map other labels to Review
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, 
                                "Input Error", 
                                "Both short and long label codes are required.")
            return

        existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

        if existing_label and existing_label != self.label:
            text = (f"A label with the short code '{short_label_code}' "
                    f"and long code '{long_label_code}' already exists. "
                    f"Do you want to merge the labels?")

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setText(text)
            msg_box.setWindowTitle("Merge Labels?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            result = msg_box.exec_()

            if result == QMessageBox.Yes:
                self.label_window.edit_labels(self.label, existing_label, delete_old=True)
                self.accept()
        else:
            # Update the label itself
            self.label.short_label_code = short_label_code
            self.label.long_label_code = long_label_code
            self.label.color = self.color
            self.label.update_label_color(self.color)
            self.label.update()

            # Force a refresh of the label window grid
            self.label_window.update_labels_per_row()
            self.label_window.reorganize_labels()
            self.label_window.scroll_area.viewport().update()

            # Update annotation window
            self.label_window.edit_labels(self.label, self.label, delete_old=False)
            
            self.accept()
