import warnings

import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QComboBox, QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QApplication, QMenu, QButtonGroup, QAbstractItemView,
                             QGroupBox, QPushButton, QStyle, QFormLayout, QFrame)

from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import rasterio_to_qimage

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

# TODO add a home button in the top-left corner to center the row on the current image

class ImageWindow(QWidget):
    imageSelected = pyqtSignal(str)
    imageChanged = pyqtSignal()  # New signal for image change

    def __init__(self, main_window):
        """Initialize the ImageWindow widget."""
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # -------------------------------------------
        # Create a QGroupBox for search and filters
        self.filter_group = QGroupBox("Search and Filters")
        self.filter_layout = QVBoxLayout()
        self.filter_group.setLayout(self.filter_layout)

        # Create a grid layout for the checkboxes
        self.checkbox_layout = QVBoxLayout()
        self.filter_layout.addLayout(self.checkbox_layout)

        # Create two horizontal layouts for the rows
        self.checkbox_row1 = QHBoxLayout()
        self.checkbox_row2 = QHBoxLayout()
        self.checkbox_layout.addLayout(self.checkbox_row1)
        self.checkbox_layout.addLayout(self.checkbox_row2)

        # Add a QButtonGroup for the checkboxes
        self.checkbox_group = QButtonGroup(self)
        self.checkbox_group.setExclusive(False)

        # Top row: Selected and Has Predictions
        self.selected_checkbox = QCheckBox("Selected", self) 
        self.selected_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_row1.addWidget(self.selected_checkbox)
        self.checkbox_group.addButton(self.selected_checkbox)

        self.has_predictions_checkbox = QCheckBox("Has Predictions", self)
        self.has_predictions_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_row1.addWidget(self.has_predictions_checkbox)
        self.checkbox_group.addButton(self.has_predictions_checkbox)

        # Bottom row: No Annotations and Has Annotations
        self.no_annotations_checkbox = QCheckBox("No Annotations", self)
        self.no_annotations_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_row2.addWidget(self.no_annotations_checkbox)
        self.checkbox_group.addButton(self.no_annotations_checkbox)

        self.has_annotations_checkbox = QCheckBox("Has Annotations", self)
        self.has_annotations_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_row2.addWidget(self.has_annotations_checkbox)
        self.checkbox_group.addButton(self.has_annotations_checkbox)

        # Create a form layout for the search bars
        self.search_layout = QFormLayout()
        self.filter_layout.addLayout(self.search_layout)

        fixed_width = 250

        # Create containers for search bars and buttons
        self.image_search_container = QWidget()
        self.image_search_layout = QHBoxLayout(self.image_search_container)
        self.image_search_layout.setContentsMargins(0, 0, 0, 0)

        self.label_search_container = QWidget()  
        self.label_search_layout = QHBoxLayout(self.label_search_container)
        self.label_search_layout.setContentsMargins(0, 0, 0, 0)

        # Setup image search
        self.search_bar_images = QComboBox(self)
        self.search_bar_images.setEditable(True)
        self.search_bar_images.setPlaceholderText("Type to search images")
        self.search_bar_images.setInsertPolicy(QComboBox.NoInsert)
        self.search_bar_images.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_bar_images.setFixedWidth(fixed_width)
        self.image_search_layout.addWidget(self.search_bar_images)

        self.image_search_button = QPushButton(self)
        self.image_search_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.image_search_button.clicked.connect(self.filter_images)
        self.image_search_layout.addWidget(self.image_search_button)

        # Setup label search
        self.search_bar_labels = QComboBox(self)
        self.search_bar_labels.setEditable(True)
        self.search_bar_labels.setPlaceholderText("Type to search labels")
        self.search_bar_labels.setInsertPolicy(QComboBox.NoInsert)
        self.search_bar_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_bar_labels.setFixedWidth(fixed_width)
        self.label_search_layout.addWidget(self.search_bar_labels)

        self.label_search_button = QPushButton(self)
        self.label_search_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.label_search_button.clicked.connect(self.filter_images)
        self.label_search_layout.addWidget(self.label_search_button)

        # Add rows to form layout
        self.search_layout.addRow("Search Images:", self.image_search_container)
        self.search_layout.addRow("Search Labels:", self.label_search_container)

        # Add the group box to the main layout  
        self.layout.addWidget(self.filter_group)

        # -------------------------------------------
        # Create a QGroupBox for Image Window
        self.info_table_group = QGroupBox("Image Window", self)
        info_table_layout = QVBoxLayout()
        self.info_table_group.setLayout(info_table_layout)

        # Create a horizontal layout for the labels
        self.info_layout = QHBoxLayout()
        info_table_layout.addLayout(self.info_layout)
        
        # Add Home button to the info_layout
        self.home_button = QPushButton("", self)
        self.home_button.setToolTip("Center table on current image")
        self.home_button.setIcon(get_icon("home.png"))  
        self.home_button.setFixedSize(24, 24)             
        self.home_button.setFlat(True)    
        self.home_button.clicked.connect(self.center_table_on_current_image)
        self.info_layout.addWidget(self.home_button)

        # Optionally add spacing after the button
        self.info_layout.addSpacing(10)

        # Add a label to display the index of the currently selected image
        self.current_image_index_label = QLabel("Current Image: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.current_image_index_label.setFixedHeight(24)
        self.info_layout.addWidget(self.current_image_index_label)

        # Add a label to display the total number of images
        self.image_count_label = QLabel("Total Images: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.image_count_label)

        # Create and setup table widget
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["✓", "Image Name", "Annotations"])
        
        # Change this line to False since we don't want the last column to stretch
        self.tableWidget.horizontalHeader().setStretchLastSection(False)
        
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.tableWidget.cellClicked.connect(self.load_image)
        self.tableWidget.keyPressEvent = self.tableWidget_keyPressEvent

        self.tableWidget.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
            background-color: #E0E0E0;
            padding: 4px;
            border: 1px solid #D0D0D0;
            }
        """)

        # Set width for checkboxes column (column 0)
        self.tableWidget.setColumnWidth(0, 50)
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        
        # Set width for Annotations column (column 2)
        self.tableWidget.setColumnWidth(2, 120)  # Adjust this width as needed
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        
        # Make the Image Name column (column 1) stretch to fill remaining space
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        # Add table widget to the info table group layout
        info_table_layout.addWidget(self.tableWidget)

        # Add a new horizontal layout below the table widget to hold the buttons
        self.button_layout = QHBoxLayout()
        info_table_layout.addLayout(self.button_layout)

        # Add 'Select All' button to the new layout
        self.select_all_button = QPushButton("Select All", self)
        self.select_all_button.clicked.connect(self.select_all_checkboxes)
        self.button_layout.addWidget(self.select_all_button)

        # Add 'Deselect All' button to the new layout
        self.deselect_all_button = QPushButton("Deselect All", self)
        self.deselect_all_button.clicked.connect(self.deselect_all_checkboxes)
        self.button_layout.addWidget(self.deselect_all_button)

        # Add the group box to the main layout
        self.layout.addWidget(self.info_table_group)

        self.image_paths = []  # Store all image paths
        self.image_dict = {}  # Dictionary to store all image information
        self.filtered_image_paths = []  # List to store filtered image paths
        self.selected_image_path = None
        self.right_clicked_row = None  # Attribute to store the right-clicked row

        self.q_images = {}  # Dictionary to store image paths and their QImage representation
        self.rasterio_images = {}  # Dictionary to store image paths and their Rasterio representation

        self.show_confirmation_dialog = True

        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.filter_images)

        self.checkbox_states = {}  # Store checkbox states for each image path

        # Connect annotationCreated, annotationDeleted signals to update annotation count in real time
        self.annotation_window.annotationCreated.connect(self.update_annotation_count)
        self.annotation_window.annotationDeleted.connect(self.update_annotation_count)
        
        # Preview tooltip attributes
        self.preview_tooltip = ImagePreviewTooltip()
        self.hover_row = -1
        
        # Connect table events for hover tracking
        self.tableWidget.setMouseTracking(True)
        self.tableWidget.viewport().installEventFilter(self)
        
    def dragEnterEvent(self, event):
        """Ignore drag enter events."""
        event.ignore()

    def dropEvent(self, event):
        """Ignore drop events."""
        event.ignore()

    def dragMoveEvent(self, event):
        """Ignore drag move events."""
        event.ignore()
        
    def dragLeaveEvent(self, event):
        """Ignore drag leave events."""
        event.ignore()
        
    def eventFilter(self, source, event):
        """Event filter to track mouse movement over table rows"""
        if source is self.tableWidget.viewport():
            if event.type() == event.Enter:
                # Mouse entered the table viewport
                pass
            elif event.type() == event.Leave:
                # Mouse left the table viewport
                self.hide_image_preview()
                self.hover_row = -1
            elif event.type() == event.MouseMove:
                # Mouse moved within the table viewport
                pos = event.pos()
                row = self.tableWidget.rowAt(pos.y())
                
                if row != self.hover_row:
                    # Mouse moved to a different row
                    self.hide_image_preview()
                    self.hover_row = row
                    
                    if row >= 0 and row < len(self.filtered_image_paths):
                        # Show preview immediately
                        self.show_image_preview()
                    
        return super().eventFilter(source, event)
    
    def center_table_on_current_image(self):
        """Scroll the table so the currently selected image row is in view and ensure it's highlighted."""
        if self.selected_image_path in self.filtered_image_paths:
            row = self.filtered_image_paths.index(self.selected_image_path)
            
            # Scroll to the item to center it in the view
            self.tableWidget.scrollToItem(
                self.tableWidget.item(row, 1),
                QAbstractItemView.PositionAtCenter
            )
            
            # Make sure the row is selected (highlighted)
            self.tableWidget.blockSignals(True)  # Prevent triggering load_image again
            self.tableWidget.clearSelection()
            self.tableWidget.selectRow(row)
            self.tableWidget.setFocus()  # Ensure selection is visually highlighted
            self.tableWidget.blockSignals(False)
        
    def show_image_preview(self):
        """Show image preview tooltip for the current hover row"""
        if self.hover_row < 0 or self.hover_row >= len(self.filtered_image_paths):
            return
        
        # Get the path of the image to preview
        image_path = self.filtered_image_paths[self.hover_row]
        # Use the already loaded rasterio image
        rasterio_src = self.rasterio_images[image_path]
        
        # Open with rasterio, convert to QImage; display thumbnail as a pixmap
        thumbnail = rasterio_to_qimage(rasterio_src, longest_edge=64)
        pixmap = QPixmap.fromImage(thumbnail)

        # Set image and show tooltip
        self.preview_tooltip.set_image(pixmap)
        
        # Position tooltip near the row
        pos = self.tableWidget.viewport().mapToGlobal(QPoint(
            self.tableWidget.columnWidth(0) + self.tableWidget.columnWidth(1) // 2,
            self.tableWidget.rowViewportPosition(self.hover_row) + 
            self.tableWidget.rowHeight(self.hover_row) // 2
        ))
        self.preview_tooltip.show_at(pos)
        
    def hide_image_preview(self):
        """Hide the image preview tooltip"""
        self.preview_tooltip.hide()

    def add_image(self, image_path):
        """Add an image to the image paths list and update the table widget"""
        # Check if the image path is already in the list
        if image_path not in self.image_paths:
            # Add the image path to the list
            self.image_paths.append(image_path)
            # Add the image to the image dictionary
            filename = os.path.basename(image_path)
            self.image_dict[image_path] = {
                'filename': filename,
                'has_annotations': False,
                'has_predictions': False,
                'labels': set(),  # Initialize an empty set for labels
                'annotation_count': 0  # Initialize annotation count
            }
            # Load the rasterio representation (now available for use anywhere!)
            rasterio_src = rasterio_open(image_path)
            if rasterio_src is None:
                raise ValueError("Rasterio failed to open the image")
            # Store the rasterio image in the dictionary
            self.rasterio_images[image_path] = rasterio_src
            
            # Update the table
            self.update_table_widget()
            self.update_image_count_label()
            self.update_search_bars()
            QApplication.processEvents()

    def update_table_widget(self):
        """Update the table widget with filtered image paths."""
        self.tableWidget.setRowCount(0)  # Clear the table

        # Center align the column headers
        self.tableWidget.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

        # First create all rows
        for _ in self.filtered_image_paths:
            self.tableWidget.insertRow(self.tableWidget.rowCount())

        # Then update each row using update_table_row
        for path in self.filtered_image_paths:
            self.update_table_row(path)

        self.update_table_selection()
        
    def update_table_row(self, path):
        """Update a specific row in the table widget."""
        if path in self.filtered_image_paths:
            row = self.filtered_image_paths.index(path)

            # Update checkbox
            checkbox = QCheckBox()
            checkbox.setStyleSheet("margin-left:10px;")
            if path in self.checkbox_states:
                checkbox.setChecked(self.checkbox_states[path])
            self.tableWidget.setCellWidget(row, 0, checkbox)
            checkbox.stateChanged.connect(lambda state, p=path: self.checkbox_states.update({p: bool(state)}))

            # Update filename
            item_text = f"{self.image_dict[path]['filename']}"
            item_text = item_text[:23] + "..." if len(item_text) > 25 else item_text
            item = QTableWidgetItem(item_text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip(path)
            item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, 1, item)

            # Update annotation count
            annotation_count = self.image_dict[path]['annotation_count']
            annotation_item = QTableWidgetItem(str(annotation_count))
            annotation_item.setFlags(annotation_item.flags() & ~Qt.ItemIsEditable)
            annotation_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, 2, annotation_item)

    def update_table_selection(self):
        """Update the selection in the table widget based on the selected image path."""
        if self.selected_image_path in self.filtered_image_paths:
            # Get the row index of the selected image
            row = self.filtered_image_paths.index(self.selected_image_path)
            
            # Block signals temporarily to prevent recursive calls
            self.tableWidget.blockSignals(True)
            
            # Clear previous selection
            self.tableWidget.clearSelection()
            
            # Select the entire row
            self.tableWidget.selectRow(row)
            
            # Ensure the selected row is visible in the viewport
            self.tableWidget.scrollToItem(
                self.tableWidget.item(row, 1),
                QAbstractItemView.PositionAtCenter
            )
            
            # Set focus to maintain highlight
            self.tableWidget.setFocus()
            
            # Restore signals
            self.tableWidget.blockSignals(False)
        else:
            self.tableWidget.clearSelection()

    def update_image_count_label(self):
        """Update the label displaying the total number of images."""
        total_images = len(set(self.filtered_image_paths))
        self.image_count_label.setText(f"Total Images: {total_images}")

    def update_current_image_index_label(self):
        """Update the label displaying the index of the currently selected image."""
        if self.selected_image_path and self.selected_image_path in self.filtered_image_paths:
            index = self.filtered_image_paths.index(self.selected_image_path) + 1
            self.current_image_index_label.setText(f"Current Image: {index}")
        else:
            self.current_image_index_label.setText("Current Image: None")

    def update_image_annotations(self, image_path):
        """Update annotation-related information for a specific image."""
        if image_path in self.image_dict:
            # Check for any annotations
            annotations = self.annotation_window.get_image_annotations(image_path)
            self.image_dict[image_path]['has_annotations'] = bool(annotations)
            self.image_dict[image_path]['annotation_count'] = len(annotations)
            # Check for any predictions
            predictions = [a.machine_confidence for a in annotations if a.machine_confidence != {}]
            self.image_dict[image_path]['has_predictions'] = len(predictions)
            # Check for any labels
            labels = {annotation.label for annotation in annotations}
            self.image_dict[image_path]['labels'] = labels
            # Update the table row
            self.update_table_row(image_path)
            # Update the label window annotation count
            self.main_window.label_window.update_annotation_count()
            
    def update_current_image_annotations(self):
        """Update annotations for the currently selected image."""
        if self.selected_image_path:
            self.update_image_annotations(self.selected_image_path)

    def update_annotation_count(self, annotation_id):
        """Update the annotation count for an image when an annotation is created or deleted."""
        if annotation_id in self.annotation_window.annotations_dict:
            # Get the image path associated with the annotation
            image_path = self.annotation_window.annotations_dict[annotation_id].image_path
        else:
            # It's already been deleted, so get the current image path
            image_path = self.annotation_window.current_image_path
        # Update the image annotation count in table widget
        self.update_image_annotations(image_path)

    def load_image(self, row, column):
        """Load the image associated with the clicked row in the table widget"""
        # Add safety checks
        if not self.filtered_image_paths:
            return

        if row < 0 or row >= len(self.filtered_image_paths):
            return

        # Get the image path associated with the selected row  
        image_path = self.filtered_image_paths[row]
        
        # Load the image without clearing selections
        self.load_image_by_path(image_path)

    def load_image_by_path(self, image_path, update=False):
        """Load an image by it's path, add to dictionaries, and update the table widget"""
        # Check if the image path is valid
        if image_path not in self.image_paths:
            return
        # Check if the image is already selected
        if image_path == self.selected_image_path and update is False:
            return

        try:
            # Set the cursor to the wait cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Update the selected image path
            self.selected_image_path = image_path
            self.update_table_selection()
            self.update_current_image_index_label()

            # Get the rasterio image from the dictionary
            if image_path in self.rasterio_images:
                rasterio_src = self.rasterio_images[image_path]
            else:
                # This should not happen, but just in case
                print("Warning: Rasterio image not found in dictionary.")
                rasterio_src = rasterio_open(image_path)
                self.rasterio_images[image_path] = rasterio_src
                
            # Load and display scaled-down version for immediate preview
            low_res_q_image = rasterio_to_qimage(rasterio_src, longest_edge=256)
            self.annotation_window.display_image(low_res_q_image)
            
            # Now Load the full resolution image using QImage directly (faster)
            q_image = QImage(image_path)
            if q_image.isNull():
                raise ValueError("QImage failed to load the image")
            
            # Update the image dictionaries
            self.q_images[image_path] = q_image
            
            # Update the display with the full-resolution image
            self.annotation_window.set_image(image_path)
            self.imageSelected.emit(image_path)
            
            # Emit the signal when a new image is chosen
            self.imageChanged.emit()
            # Update the search bars
            self.update_search_bars()

        except Exception as e:
            QMessageBox.warning(self, 
                                "Image Loading Error",
                                f"Error loading full resolution image {os.path.basename(image_path)}:\n{str(e)}")
            
        finally:
            QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        """Handle the window close event."""
        # Hide tooltip when window is closed
        self.hide_image_preview()
        QApplication.restoreOverrideCursor()
        super().closeEvent(event)

    def show_context_menu(self, position):
        """Show the context menu for the table widget."""
        # Get selected checkboxes
        selected_paths = self._get_selected_image_paths()

        if not selected_paths:
            return

        context_menu = QMenu(self)
        delete_all_images_action = context_menu.addAction(f"Delete {len(selected_paths)} Images")
        delete_all_images_action.triggered.connect(lambda: self.delete_selected_images())

        delete_all_annotations_action = context_menu.addAction(f"Delete Annotations for {len(selected_paths)} Images")
        delete_all_annotations_action.triggered.connect(lambda: self.delete_selected_images_annotations())

        context_menu.exec_(self.tableWidget.viewport().mapToGlobal(position))
        
    def _get_selected_image_paths(self):
        """
        Returns list of image paths for rows with checked checkboxes
        """
        selected_paths = []
        for row in range(self.tableWidget.rowCount()):
            checkbox = self.tableWidget.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_paths.append(self.filtered_image_paths[row])
        return selected_paths

    def delete_selected_images(self):
        """Delete images corresponding to the checked rows."""
        selected_paths = self._get_selected_image_paths()
        
        if not selected_paths:
            return

        reply = QMessageBox.question(self, 
                                     "Confirm Multiple Image Deletions",
                                     f"Are you sure you want to delete {len(selected_paths)} images?\n"
                                     "This will delete all associated annotations.",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Delete images and handle loading a new image if necessary
            self.delete_images(selected_paths)
                
    def delete_selected_images_annotations(self):
        """Delete annotations for images corresponding to the checked rows."""
        selected_paths = self._get_selected_image_paths()
        
        if not selected_paths:
            return

        reply = QMessageBox.question(self, 
                                     "Confirm Multiple Annotation Deletions",
                                     f"Are you sure you want to delete annotations for {len(selected_paths)} images?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            
            # Disconnect signals temporarily
            self.annotation_window.annotationCreated.disconnect(self.update_annotation_count)
            self.annotation_window.annotationDeleted.disconnect(self.update_annotation_count)

            try:
                # Make cursor busy
                QApplication.setOverrideCursor(Qt.WaitCursor)
                progress_bar = ProgressBar(self, title="Deleting Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(selected_paths))
                
                # Delete annotations for selected images
                for path in selected_paths:
                    # Delete the annotations for the image
                    self.annotation_window.delete_image_annotations(path)
                    # Update the image annotation count in the table widget
                    self.update_image_annotations(path)
                    progress_bar.update_progress()
                    
                # Close the progress bar
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()
                
            finally:
                # Reconnect signals
                self.annotation_window.annotationCreated.connect(self.update_annotation_count)
                self.annotation_window.annotationDeleted.connect(self.update_annotation_count)
                       
            # Update the table widget
            self.update_table_widget()

    def delete_images(self, image_paths):
        """
        Delete multiple images and their associated annotations.
        
        Args:
            image_paths (list): List of image paths to delete
        """
        # Validate input and create a copy to avoid mutation during iteration
        image_paths = [path for path in image_paths if path in self.image_paths]
        
        if not image_paths:
            return

        # Check if current image is being deleted
        current_image_in_deletion = self.selected_image_path in image_paths

        # Determine the next image to load if current image is deleted
        next_image_to_load = None
        if current_image_in_deletion and self.filtered_image_paths:
            # Find remaining images in the filtered list
            remaining_images = [path for path in self.filtered_image_paths if path not in image_paths]
            
            if remaining_images:
                # If possible, maintain the relative position in the list
                current_idx = self.filtered_image_paths.index(self.selected_image_path)
                
                # Find the next viable image to load (prioritize images before current)
                viable_images = [img for img in remaining_images 
                                if self.filtered_image_paths.index(img) <= current_idx]
                
                # Prioritize images at or before the current index
                next_image_to_load = viable_images[0] if viable_images else remaining_images[0]

        try:
            # Make cursor busy and show progress
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Deleting Images")
            progress_bar.show()
            progress_bar.start_progress(len(image_paths))
            
            # Delete each image
            for image_path in image_paths:
                self.cleanup_image(image_path)
                progress_bar.update_progress()
            
            # Update UI components
            self.update_table_widget()
            self.update_image_count_label()

            # Load next image or clear scene
            if next_image_to_load:
                self.load_image_by_path(next_image_to_load)
            elif not self.filtered_image_paths:
                self.selected_image_path = None
                self.annotation_window.clear_scene()

            # Update current image index label
            self.update_current_image_index_label()
        
        finally:
            # Ensure cursor is restored even if an exception occurs
            if progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()
            QApplication.restoreOverrideCursor()
            
    def cleanup_image(self, image_path):
        """
        Completely remove an image from all data structures and release all resources.
        
        Args:
            image_path (str): Path to the image to be removed
            
        Returns:
            bool: True if successful, False otherwise
        """
        if image_path not in self.image_paths:
            return False
            
        try:
            # Remove from annotation window
            self.annotation_window.delete_image(image_path)
            
            # Remove from filtered image paths
            if image_path in self.filtered_image_paths:
                self.filtered_image_paths.remove(image_path)
                
            # Close and remove rasterio resources
            if image_path in self.rasterio_images:
                self.rasterio_images[image_path] = None
                del self.rasterio_images[image_path]

            # Remove from QImage dictionary
            if image_path in self.q_images:
                self.q_images[image_path] = None
                del self.q_images[image_path]
                
            # Remove from main image collections
            self.image_paths.remove(image_path)
            
            # Remove from checkbox states
            if image_path in self.checkbox_states:
                del self.checkbox_states[image_path]
                
            # Remove from image dictionary
            if image_path in self.image_dict:
                del self.image_dict[image_path]
                
            # Force garbage collection to clean up resources
            gc.collect()
            
            return True
        except Exception as e:
            print(f"Error cleaning up image {image_path}: {str(e)}")
            return False

    def tableWidget_keyPressEvent(self, event):
        """Handle key press events in the table widget, ignoring up/down arrows."""
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            # Ignore up and down arrow keys
            return
        else:
            # Call the base class method for other keys
            super(QTableWidget, self.tableWidget).keyPressEvent(event)

    def cycle_previous_image(self):
        """Load the previous image in the filtered list."""
        if not self.filtered_image_paths:
            return

        current_index = self.filtered_image_paths.index(self.selected_image_path)
        new_index = (current_index - 1) % len(self.filtered_image_paths)
        self.load_image_by_path(self.filtered_image_paths[new_index])

    def cycle_next_image(self):
        """Load the next image in the filtered list."""
        if not self.filtered_image_paths:
            return

        current_index = self.filtered_image_paths.index(self.selected_image_path)
        new_index = (current_index + 1) % len(self.filtered_image_paths)
        self.load_image_by_path(self.filtered_image_paths[new_index])

    def filter_images(self):
        """Filter the images based on the current search and filter criteria."""
        # Store the currently selected image path before filtering
        current_selected_path = self.selected_image_path

        # Search bars
        search_text_images = self.search_bar_images.currentText()
        search_text_labels = self.search_bar_labels.currentText()
        # Filter checkboxes
        no_annotations = self.no_annotations_checkbox.isChecked()
        has_annotations = self.has_annotations_checkbox.isChecked()
        has_predictions = self.has_predictions_checkbox.isChecked()
        selected_only = self.selected_checkbox.isChecked()

        # Return early if none of the filters are active
        if (not (search_text_images or search_text_labels) and
            not (no_annotations or has_annotations or has_predictions or selected_only)):
            self.filtered_image_paths = self.image_paths.copy()
            self.update_table_widget()
            self.update_current_image_index_label()
            self.update_image_count_label()
            return
        
        # Get list of selected image paths if needed
        selected_paths = self._get_selected_image_paths() if selected_only else None

        # Initialize filtered image paths
        self.filtered_image_paths = []

        # Initialize the progress bar
        progress_dialog = ProgressBar(title="Filtering Images")
        progress_dialog.start_progress(len(self.image_paths))

        # Use a ThreadPoolExecutor to filter images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for path in self.image_paths:
                future = executor.submit(
                    self.filter_image,
                    path,
                    search_text_images,
                    search_text_labels,
                    no_annotations,
                    has_annotations,
                    has_predictions,
                    selected_paths
                )
                futures.append(future)

            for future in as_completed(futures):
                if future.result():
                    self.filtered_image_paths.append(future.result())
                progress_dialog.update_progress()

        # Sort the filtered image paths to be displaying in ImageWindow
        self.filtered_image_paths.sort()

        # Update the table widget
        self.update_table_widget()

        # After filtering, either restore the previously selected image if it's still in the filtered list,
        # or load the first image if nothing was selected or the previous selection is no longer visible
        if self.filtered_image_paths:
            if current_selected_path and current_selected_path in self.filtered_image_paths:
                self.load_image_by_path(current_selected_path)
            else:
                self.load_first_filtered_image()
        else:
            self.selected_image_path = None
            self.annotation_window.clear_scene()

        # Update the current image index label and image count label
        self.update_current_image_index_label()
        self.update_image_count_label()

        # Stop the progress bar
        progress_dialog.stop_progress()

    def filter_image(self,
                     path,
                     search_text_images,
                     search_text_labels,
                     no_annotations,
                     has_annotations,
                     has_predictions,
                     selected_paths=None):
        """
        Filter images based on search text and checkboxes

        Args:
            path (str): Path to the image
            search_text_images (str): Search text for image names
            search_text_labels (str): Search text for labels
            no_annotations (bool): Filter images with no annotations
            has_annotations (bool): Filter images with annotations
            has_predictions (bool): Filter images with predictions
            selected_paths (list): List of selected image paths

            Returns:
                str: Path to the image if it passes the filters, None otherwise
        """
        # Check selected filter first
        if selected_paths is not None and path not in selected_paths:
            return None
        
        filename = os.path.basename(path)
        # Check for annotations for the provided path
        annotations = self.annotation_window.get_image_annotations(path)
        # Check for predictions for the provided path
        predictions = self.image_dict[path]['has_predictions']
        # Check the labels for the provided path
        labels = self.image_dict[path]['labels']
        labels = [label.short_label_code for label in labels]
        
        # Filter images based on search text and checkboxes
        if search_text_images and search_text_images not in filename:
            return None
        # Filter images based on search text and checkboxes
        if search_text_labels and search_text_labels not in labels:
            return None
        # Filter images based on checkboxes, and if the image has annotations
        if no_annotations and annotations:
            return None
        # Filter images based on checkboxes, and if the image has no annotations
        if has_annotations and not annotations:
            return None
        # Filter images based on checkboxes, and if the image has predictions
        if has_predictions and not predictions:
            return None

        return path

    def load_first_filtered_image(self):
        """Load the first image in the currently filtered list."""
        if self.filtered_image_paths:
            self.annotation_window.clear_scene()
            self.load_image_by_path(self.filtered_image_paths[0])
            
    def update_search_bars(self):
        """Update the items in the image and label search bars."""
        # Store current search texts
        current_image_search = self.search_bar_images.currentText()
        current_label_search = self.search_bar_labels.currentText()

        # Clear and update items
        self.search_bar_images.clear()
        self.search_bar_labels.clear()

        try:
            image_names = set([self.image_dict[path]['filename'] for path in self.image_paths])
            label_names = set([label.short_label_code for label in self.main_window.label_window.labels])
        except Exception as e:
            return

        # Only add items if there are any to add
        if image_names:
            self.search_bar_images.addItems(sorted(image_names))
        if label_names:
            self.search_bar_labels.addItems(sorted(label_names))

        # Restore search texts only if they're not empty
        if current_image_search:
            self.search_bar_images.setEditText(current_image_search)
        else:
            self.search_bar_images.setPlaceholderText("Type to search images")
        if current_label_search:
            self.search_bar_labels.setEditText(current_label_search)
        else:
            self.search_bar_labels.setPlaceholderText("Type to search labels")

    def select_all_checkboxes(self):
        """Check all checkboxes in the table widget."""
        for row in range(self.tableWidget.rowCount()):
            checkbox = self.tableWidget.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)

    def deselect_all_checkboxes(self):
        """Uncheck all checkboxes in the table widget."""
        for row in range(self.tableWidget.rowCount()):
            checkbox = self.tableWidget.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
                
                
class ImagePreviewTooltip(QFrame):
    """
    A custom tooltip widget that displays an image preview and information text.
    """
    def __init__(self, parent=None):
        """Initialize the ImagePreviewTooltip."""
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        
        # Configure appearance
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #aaaaaa;
                border-radius: 4px;
            }
        """)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(4)  # Add spacing between elements
        
        # Create image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(64, 64)
        self.layout.addWidget(self.image_label)
        
        # Initially hidden
        self.hide()
        
    def set_image(self, pixmap):
        """Set the preview image"""
        if pixmap and not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
            
            # Adjust widget size based on the pixmap
            size = pixmap.size()
            self.image_label.setMinimumSize(size)
            self.image_label.setMaximumSize(size)
        
            # Ensure proper sizing
            self.adjustSize()
        else:
            self.hide()
            
    def show_at(self, global_pos):
        """Position and show the tooltip at the specified global position"""
        # Position slightly offset from cursor with increased x offset
        x, y = global_pos.x() + 50, global_pos.y() + 15  # Increased x offset (bottom-right)
        
        # Ensure tooltip stays within screen boundaries
        screen_rect = self.screen().geometry()
        tooltip_size = self.sizeHint()
        
        # Adjust position if needed
        if x + tooltip_size.width() > screen_rect.right():
            x = global_pos.x() - tooltip_size.width() - 5
        if y + tooltip_size.height() > screen_rect.bottom():
            y = global_pos.y() - tooltip_size.height() - 5
            
        # Set position and show
        self.move(x, y)
        self.show()