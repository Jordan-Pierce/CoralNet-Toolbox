import os

from PyQt5.QtWidgets import (QProgressBar, QMainWindow, QFileDialog, QApplication, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QToolBar, QAction, QScrollArea,
                             QSizePolicy, QMessageBox, QCheckBox, QDialog, QHBoxLayout, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QColorDialog, QMenu, QLineEdit, QSpinBox, QDialog, QHBoxLayout, QTextEdit,
                             QPushButton, QComboBox, QSpinBox, QGraphicsPixmapItem, QGraphicsRectItem, QSlider,
                             QFormLayout, QInputDialog, QFrame, QTabWidget, QDialogButtonBox, QDoubleSpinBox,
                             QGroupBox, QListWidget, QListWidgetItem, QPlainTextEdit, QRadioButton, QTableWidget,
                             QTableWidgetItem)

from PyQt5.QtGui import (QMouseEvent, QIcon, QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFontMetrics, QFont,
                         QCursor, QMovie)

from PyQt5.QtCore import (Qt, pyqtSignal, QSize, QObject, QThreadPool, QRunnable, QTimer, QEvent, QPointF, QRectF,
                          QThread)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ImageWindow(QWidget):
    imageSelected = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the search bar
        self.search_layout = QHBoxLayout()
        self.layout.addLayout(self.search_layout)

        # Add a search bar
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search images...")
        self.search_bar.textChanged.connect(self.filter_images)
        self.search_layout.addWidget(self.search_bar)

        # Create a horizontal layout for the labels
        self.info_layout = QHBoxLayout()
        self.layout.addLayout(self.info_layout)

        # Add a label to display the index of the currently highlighted image
        self.current_image_index_label = QLabel("Current Image: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.current_image_index_label.setFixedHeight(24)
        self.info_layout.addWidget(self.current_image_index_label)

        # Add a label to display the total number of images
        self.image_count_label = QLabel("Total Images: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.image_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.image_count_label)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(["Image Name"])
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.cellClicked.connect(self.load_image)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)

        self.layout.addWidget(self.tableWidget)

        self.image_paths = []  # Store only image paths
        self.filtered_image_paths = []  # Subset of images based on search
        self.selected_row = None
        self.show_confirmation_dialog = True
        self.images = {}  # Dictionary to store images and their paths

    def add_image(self, image_path):

        # Clear the search bar text
        self.search_bar.clear()

        if image_path not in self.main_window.imported_image_paths:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(os.path.basename(image_path)))

            self.image_paths.append(image_path)
            self.filtered_image_paths.append(image_path)

            # Select and set the first image
            if row_position == 0:
                self.tableWidget.selectRow(0)
                self.load_image(0, 0)

            # Add to main window imported path
            self.main_window.imported_image_paths.add(image_path)

            # Update the image count label
            self.update_image_count_label()

    def load_image(self, row, column):
        image_path = self.filtered_image_paths[row]
        self.load_image_by_path(image_path)

    def load_image_by_path(self, image_path):
        if self.selected_row is not None:
            self.tableWidget.item(self.selected_row, 0).setSelected(False)

        # Clear the previous image from memory
        if self.selected_row is not None and self.filtered_image_paths[self.selected_row] in self.images:
            del self.images[self.filtered_image_paths[self.selected_row]]

        image = QImage(image_path)  # Load the image only when needed
        self.images[image_path] = image  # Store the image in the dictionary
        self.selected_row = self.filtered_image_paths.index(image_path)
        self.tableWidget.selectRow(self.selected_row)
        self.annotation_window.set_image(image, image_path)
        self.imageSelected.emit(image_path)

        # Update the current image index label
        self.update_current_image_index_label()

    def delete_image(self, row):
        if self.show_confirmation_dialog:
            result = self._confirm_delete()
            if result == QMessageBox.No:
                return

        image_path = self.filtered_image_paths[row]
        self.tableWidget.removeRow(row)
        self.image_paths.remove(image_path)
        self.filtered_image_paths.remove(image_path)
        self.annotation_window.delete_image(image_path)
        self.main_window.imported_image_paths.discard(image_path)

        # Remove the image from the dictionary
        if image_path in self.images:
            del self.images[image_path]

        # Update the image count label
        self.update_image_count_label()

        # Update the selected row and load another image if possible
        if self.filtered_image_paths:
            if row < len(self.filtered_image_paths):
                self.selected_row = row
            else:
                self.selected_row = len(self.filtered_image_paths) - 1
            self.load_image(self.selected_row, 0)
        else:
            self.selected_row = None

        # Update the current image index label
        self.update_current_image_index_label()

    def _confirm_delete(self):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirm Delete")
        msg_box.setText("Are you sure you want to delete this image?\n"
                        "This will delete all associated annotations.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        checkbox = QCheckBox("Do not show this message again")
        msg_box.setCheckBox(checkbox)

        result = msg_box.exec_()

        if checkbox.isChecked():
            self.show_confirmation_dialog = False

        return result

    def show_context_menu(self, position):
        row = self.tableWidget.rowAt(position.y())
        if row >= 0:
            context_menu = QMenu(self)
            delete_action = context_menu.addAction("Delete")
            action = context_menu.exec_(self.tableWidget.mapToGlobal(position))

            if action == delete_action:
                self.delete_image(row)

    def update_image_count_label(self):
        total_images = len(self.filtered_image_paths)
        self.image_count_label.setText(f"Total Images: {total_images}")

    def update_current_image_index_label(self):
        if self.selected_row is not None:
            self.current_image_index_label.setText(f"Current Image: {self.selected_row + 1}")
        else:
            self.current_image_index_label.setText("Current Image: None")

    def cycle_previous_image(self):
        if not self.filtered_image_paths:
            return

        if self.selected_row is not None:
            new_row = (self.selected_row - 1) % len(self.filtered_image_paths)
        else:
            new_row = 0

        self.tableWidget.selectRow(new_row)
        self.load_image(new_row, 0)

    def cycle_next_image(self):
        if not self.filtered_image_paths:
            return

        if self.selected_row is not None:
            new_row = (self.selected_row + 1) % len(self.filtered_image_paths)
        else:
            new_row = 0

        self.tableWidget.selectRow(new_row)
        self.load_image(new_row, 0)

    def filter_images(self, text):
        self.filtered_image_paths = [path for path in self.image_paths if text.lower() in os.path.basename(path).lower()]
        self.tableWidget.setRowCount(0)  # Clear the table
        for path in self.filtered_image_paths:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(os.path.basename(path)))

        # Update the image count label
        self.update_image_count_label()

        # If the selected row is no longer valid, reset it
        if self.selected_row is not None and self.selected_row >= len(self.filtered_image_paths):
            self.selected_row = None
            self.update_current_image_index_label()