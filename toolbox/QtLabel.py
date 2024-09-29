import json
import random
import uuid
import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFontMetrics, QDrag
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QColorDialog, QLineEdit, QDialog, QHBoxLayout, QPushButton, QApplication)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Label(QWidget):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)  # Signal to emit the selected label
    label_deleted = pyqtSignal(object)  # Signal to emit when the label is deleted

    def __init__(self, short_label_code, long_label_code, color=QColor(255, 255, 255), label_id=None, fixed_width=80):
        super().__init__()

        self.id = str(uuid.uuid4()) if label_id is None else label_id
        self.short_label_code = short_label_code
        self.long_label_code = long_label_code
        self.color = color
        self.is_selected = False
        self.fixed_width = fixed_width
        self.transparency = 128

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(self.fixed_width)

        # Set tooltip for long label
        self.setToolTip(self.long_label_code)

        self.drag_start_position = None

    def mousePressEvent(self, event):
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
        if event.buttons() == Qt.RightButton and self.drag_start_position:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.id)
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.drag_start_position = None

    def update_color(self):
        self.update()  # Trigger a repaint

    def update_selection(self):
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate the height based on the text height
        font_metrics = QFontMetrics(painter.font())
        text_height = font_metrics.height()
        # Add some padding
        self.setFixedHeight(text_height + 20)

        # Draw the main rectangle with a light transparent fill
        transparent_color = QColor(self.color)
        # Set higher transparency (0-255, where 255 is fully opaque)
        transparent_color.setAlpha(20)
        # Light transparent fill
        painter.setBrush(QBrush(transparent_color, Qt.SolidPattern))

        # Set the border color based on selection status
        if self.is_selected:
            # Lighter version of the label color
            selected_border_color = self.color.lighter(150)
            # Thicker border when selected
            painter.setPen(QPen(selected_border_color, 2, Qt.SolidLine))
        else:
            # Normal border with the color of the label
            painter.setPen(QPen(self.color, 1, Qt.SolidLine))

        painter.drawRect(0, 0, self.width(), self.height())

        # Draw the color rectangle only if selected
        if self.is_selected:
            # Width 5 pixels less than the main rectangle's width
            rectangle_width = self.width() - 10
            rectangle_height = 20
            inner_transparent_color = QColor(self.color)
            inner_transparent_color.setAlpha(100)
            painter.setBrush(QBrush(inner_transparent_color, Qt.SolidPattern))
            painter.drawRect(5, (self.height() - rectangle_height) // 2, rectangle_width, rectangle_height)

        # Draw the text
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))

        # Truncate the text if it exceeds the width
        elided_text = font_metrics.elidedText(self.short_label_code, Qt.ElideRight, self.width() - 30)
        painter.drawText(12, 0, self.width() - 30, self.height(), Qt.AlignVCenter, elided_text)

        super().paintEvent(event)

    def delete_label(self):
        self.label_deleted.emit(self)
        self.deleteLater()

    def to_dict(self):
        return {
            'id': self.id,
            'short_label_code': self.short_label_code,
            'long_label_code': self.long_label_code,
            'color': self.color.getRgb()
        }

    @classmethod
    def from_dict(cls, data):
        color = QColor(*data['color'])
        return cls(data['short_label_code'], data['long_label_code'], color)

    def select(self):
        if not self.is_selected:
            self.is_selected = True
            self.update_selection()
            self.selected.emit(self)

    def deselect(self):
        if self.is_selected:
            self.is_selected = False
            self.update_selection()

    def update_label_color(self, new_color: QColor):
        if self.color != new_color:
            self.color = new_color
            self.update_color()
            self.color_changed.emit(new_color)

    def update_transparency(self, transparency):
        self.transparency = transparency

    def __repr__(self):
        return (f"Label(id={self.id}, "
                f"short_label_code={self.short_label_code}, "
                f"long_label_code={self.long_label_code}, "
                f"color={self.color.name()})")


class LabelWindow(QWidget):
    labelSelected = pyqtSignal(object)  # Signal to emit the entire Label object
    transparencyChanged = pyqtSignal(int)  # Signal to emit the transparency value

    def __init__(self, main_window, label_width=100):
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.label_width = label_width
        self.labels_per_row = 1  # Initial value, will be updated

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Top bar with Add Label, Edit Label, and Delete Label buttons
        self.top_bar = QHBoxLayout()
        self.add_label_button = QPushButton("Add Label")
        self.add_label_button.setFixedSize(80, 30)
        self.top_bar.addWidget(self.add_label_button)

        self.edit_label_button = QPushButton("Edit Label")
        self.edit_label_button.setFixedSize(80, 30)
        self.edit_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.edit_label_button)

        self.delete_label_button = QPushButton("Delete Label")
        self.delete_label_button.setFixedSize(80, 30)
        self.delete_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.delete_label_button)

        self.top_bar.addStretch()  # Add stretch to the right side

        self.main_layout.addLayout(self.top_bar)

        # Scroll area for labels
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        self.add_label_button.clicked.connect(self.open_add_label_dialog)
        self.edit_label_button.clicked.connect(self.open_edit_label_dialog)
        self.delete_label_button.clicked.connect(self.delete_active_label)
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        label_id = event.mimeData().text()
        label = self.get_label_by_id(label_id)
        if label:
            self.labels.remove(label)
            self.labels.insert(self.calculate_new_index(event.pos()), label)
            self.reorganize_labels()

    def calculate_new_index(self, pos):
        row = pos.y() // self.label_width
        col = pos.x() // self.label_width
        return row * self.labels_per_row + col

    def export_labels(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                labels_data = [label.to_dict() for label in self.labels]
                with open(file_path, 'w') as file:
                    json.dump(labels_data, file, indent=4)

                QMessageBox.information(self, ""
                                              "Labels Exported",
                                        "Labels have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Importing Labels",
                                    f"An error occurred while importing labels: {str(e)}")

    def import_labels(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Import Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    labels_data = json.load(file)

                for label_data in labels_data:
                    label = Label.from_dict(label_data)
                    if not self.label_exists(label.short_label_code, label.long_label_code):
                        self.add_label(label.short_label_code, label.long_label_code, label.color, label.id)

                # Set the Review label as active
                self.set_active_label(self.get_label_by_long_code("Review"))

                QMessageBox.information(self,
                                        "Labels Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Importing Labels",
                                    f"An error occurred while importing Labels: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_labels_per_row()
        self.reorganize_labels()

    def update_labels_per_row(self):
        available_width = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.labels_per_row = max(1, available_width // self.label_width)
        self.scroll_content.setFixedWidth(self.labels_per_row * self.label_width)

    def reorganize_labels(self):
        for i, label in enumerate(self.labels):
            row = i // self.labels_per_row
            col = i % self.labels_per_row
            self.grid_layout.addWidget(label, row, col)

    def open_add_label_dialog(self):
        dialog = AddLabelDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            short_label_code, long_label_code, color = dialog.get_label_details()
            if self.label_exists(short_label_code, long_label_code):
                QMessageBox.warning(self, "Label Exists", "A label with the same short and long name already exists.")
            else:
                new_label = self.add_label(short_label_code, long_label_code, color)
                self.set_active_label(new_label)

    def open_edit_label_dialog(self):
        if self.active_label:
            dialog = EditLabelDialog(self, self.active_label)
            if dialog.exec_() == QDialog.Accepted:
                # Update the tooltip with the new long label code
                self.active_label.setToolTip(self.active_label.long_label_code)
                self.update_labels_per_row()
                self.reorganize_labels()

    def add_label(self, short_label_code, long_label_code, color, label_id=None):
        # Create the label
        label = Label(short_label_code, long_label_code, color, label_id, fixed_width=self.label_width)
        # Connect
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        self.labels.append(label)
        # Update in LabelWindow
        self.update_labels_per_row()
        self.reorganize_labels()
        self.set_active_label(label)
        QApplication.processEvents()

        return label

    def set_active_label(self, selected_label):
        if self.active_label and self.active_label != selected_label:
            self.deselect_active_label()

        self.active_label = selected_label
        self.active_label.select()
        self.labelSelected.emit(selected_label)
        self.transparencyChanged.emit(self.active_label.transparency)

        # Enable or disable the Edit Label and Delete Label buttons based on whether a label is selected
        self.edit_label_button.setEnabled(self.active_label is not None)
        self.delete_label_button.setEnabled(self.active_label is not None)

        # Update annotations with the new label
        self.update_annotations_with_label(selected_label)
        self.adjust_scrollbar_for_active_label()

    def deselect_active_label(self):
        if self.active_label:
            self.active_label.deselect()

    def delete_active_label(self):
        if self.active_label:
            self.delete_label(self.active_label)

    def update_label_transparency(self, transparency):
        if self.active_label:
            # Update the active label's transparency
            self.active_label.update_transparency(transparency)
            label = self.active_label
            # Update the transparency of all annotations with the active label
            for annotation in self.annotation_window.annotations_dict.values():
                if annotation.label.id == label.id:
                    annotation.update_transparency(transparency)

    def update_annotations_with_label(self, label):
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == label.id:
                annotation.update_label(label)

    def get_label_color(self, short_label_code, long_label_code):
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                return label.color
        return None

    def get_label_by_id(self, label_id):
        for label in self.labels:
            if label.id == label_id:
                return label
        return None

    def get_label_by_codes(self, short_label_code, long_label_code):
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                return label
        return None

    def get_label_by_short_code(self, short_label_code):
        for label in self.labels:
            if short_label_code == label.short_label_code:
                return label
        return None

    def get_label_by_long_code(self, long_label_code):
        for label in self.labels:
            if long_label_code == label.long_label_code:
                return label
        return None

    def label_exists(self, short_label_code, long_label_code, label_id=None):
        for label in self.labels:
            if label_id is not None and label.id == label_id:
                return True
            if label.short_label_code == short_label_code:
                return True
            if label.long_label_code == long_label_code:
                return True
        return False

    def add_label_if_not_exists(self, short_label_code, long_label_code, color, label_id=None):
        if not self.label_exists(short_label_code, long_label_code, label_id):
            self.add_label(short_label_code, long_label_code, color, label_id)

    def set_selected_label(self, label_id):
        for lbl in self.labels:
            if lbl.id == label_id:
                self.set_active_label(lbl)
                break

    def edit_labels(self, old_label, new_label, delete_old=False):
        # Update annotations to use the new label
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == old_label.id:
                annotation.update_label(new_label)

        if delete_old:
            # Remove the old label
            self.labels.remove(old_label)
            old_label.deleteLater()

        # Update the active label if necessary
        if self.active_label == old_label:
            self.set_active_label(new_label)

        self.update_labels_per_row()
        self.reorganize_labels()

    def delete_label(self, label):
        if (label.short_label_code == "Review" and
                label.long_label_code == "Review" and
                label.color == QColor(255, 255, 255)):
            QMessageBox.warning(self, "Cannot Delete Label", "The 'Review' label cannot be deleted.")
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

    def handle_wasd_key(self, key):
        if not self.active_label:
            return

        try:
            current_index = self.labels.index(self.active_label)
        except ValueError:
            # If the active label is not in the list, set it to None
            self.active_label = None
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

        if 0 <= new_index < len(self.labels):
            self.set_active_label(self.labels[new_index])

            # Adjust the scrollbar if necessary
            self.adjust_scrollbar_for_active_label()

    def adjust_scrollbar_for_active_label(self):
        if not self.active_label:
            return

        # Get the geometry of the active label and the scroll area's viewport
        label_geometry = self.active_label.geometry()
        viewport_geometry = self.scroll_area.viewport().geometry()

        # Calculate the vertical scrollbar position to keep the active label in view
        if label_geometry.top() < viewport_geometry.top():
            # Scroll up
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - label_geometry.height())
        elif label_geometry.bottom() > viewport_geometry.bottom():
            # Scroll down
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() + label_geometry.height())

        # Calculate the horizontal scrollbar position to keep the active label in view
        if label_geometry.left() < viewport_geometry.left():
            # Scroll left
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - label_geometry.width())
        elif label_geometry.right() > viewport_geometry.right():
            # Scroll right
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() + label_geometry.width())


class AddLabelDialog(QDialog):
    def __init__(self, label_window, parent=None):
        super().__init__(parent)
        self.label_window = label_window

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
        return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update_color_button(self):
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def get_label_details(self):
        return self.short_label_input.text(), self.long_label_input.text(), self.color

    def validate_and_accept(self):
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
        super().__init__(parent)
        self.label_window = label_window
        self.label = label

        self.setWindowTitle("Edit Label")

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
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def validate_and_accept(self):
        # Cannot edit Review
        if self.label.short_label_code == 'Review' and self.label.long_label_code == 'Review':
            QMessageBox.warning(self, "Cannot Edit Label", "The 'Review' label cannot be edited.")
            return

        # Can map other labels to Review
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
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

            return
        else:
            self.label.short_label_code = short_label_code
            self.label.long_label_code = long_label_code
            self.label.color = self.color
            self.label.update_label_color(self.color)
            self.accept()

            self.label_window.edit_labels(self.label, self.label, delete_old=False)