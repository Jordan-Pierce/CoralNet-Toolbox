import warnings

import os
import gc
from contextlib import contextmanager

import rasterio

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint, QThreadPool, QItemSelectionModel
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QComboBox, QHBoxLayout, QTableView, QHeaderView, QApplication, 
                             QMenu, QButtonGroup, QAbstractItemView, QGroupBox, QPushButton, 
                             QStyle, QFormLayout, QFrame)

from coralnet_toolbox.Rasters import Raster, RasterManager, ImageFilter, RasterTableModel

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class NoArrowKeyTableView(QTableView):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            event.ignore()
            return
        super().keyPressEvent(event)
        

class ImageWindow(QWidget):
    # Signals
    imageSelected = pyqtSignal(str)  # Path of selected image
    imageChanged = pyqtSignal()  # When image changes
    imageLoaded = pyqtSignal(str)  # When image is fully loaded
    filterChanged = pyqtSignal(int)  # Number of filtered images
    rasterAdded = pyqtSignal(str)  # Path of added raster
    rasterRemoved = pyqtSignal(str)  # Path of removed raster

    def __init__(self, main_window):
        """Initialize the ImageWindow widget."""
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        # Initialize managers and supporting objects
        self.raster_manager = RasterManager()
        self.image_filter = ImageFilter(self.raster_manager)
        self.selected_image_path = None
        self.hover_row = -1
        self.last_highlighted_row = -1
        
        # Connect manager signals
        self.raster_manager.rasterAdded.connect(self.on_raster_added)
        self.raster_manager.rasterRemoved.connect(self.on_raster_removed)
        self.raster_manager.rasterUpdated.connect(self.on_raster_updated)
        
        # Connect filter signals
        self.image_filter.filteringStarted.connect(self.on_filtering_started)
        self.image_filter.filteringFinished.connect(self.on_filtering_finished)
        
        # Setup UI components
        self.setup_ui()
        
        # Connect signals
        self.setup_signals()
        
        # Initialize timer and dialog flags
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.filter_images)
        self.show_confirmation_dialog = True
        
    def setup_ui(self):
        """Set up the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create UI sections
        self.setup_filter_section()
        self.setup_image_section()
        
        # Create the tooltip for previews
        self.preview_tooltip = ImagePreviewTooltip()
        
    def setup_filter_section(self):
        """Set up the filter section of the UI."""
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

        # Top row: Highlighted and Has Predictions
        self.highlighted_checkbox = QCheckBox("Highlighted", self) 
        self.highlighted_checkbox.stateChanged.connect(self.schedule_filter)
        self.checkbox_row1.addWidget(self.highlighted_checkbox)
        self.checkbox_group.addButton(self.highlighted_checkbox)

        self.has_predictions_checkbox = QCheckBox("Has Predictions", self)
        self.has_predictions_checkbox.stateChanged.connect(self.schedule_filter)
        self.checkbox_row1.addWidget(self.has_predictions_checkbox)
        self.checkbox_group.addButton(self.has_predictions_checkbox)

        # Bottom row: No Annotations and Has Annotations
        self.no_annotations_checkbox = QCheckBox("No Annotations", self)
        self.no_annotations_checkbox.stateChanged.connect(self.schedule_filter)
        self.checkbox_row2.addWidget(self.no_annotations_checkbox)
        self.checkbox_group.addButton(self.no_annotations_checkbox)

        self.has_annotations_checkbox = QCheckBox("Has Annotations", self)
        self.has_annotations_checkbox.stateChanged.connect(self.schedule_filter)
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
        self.search_bar_images.editTextChanged.connect(self.schedule_filter)
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
        self.search_bar_labels.editTextChanged.connect(self.schedule_filter)
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
        
    def setup_image_section(self):
        """Set up the image list section of the UI."""
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
        self.current_image_index_label = QLabel("Current: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.current_image_index_label.setFixedHeight(24)
        self.info_layout.addWidget(self.current_image_index_label)

        # Add a label to display the number of highlighted images
        self.highlighted_count_label = QLabel("Highlighted: 0", self)
        self.highlighted_count_label.setAlignment(Qt.AlignCenter)
        self.highlighted_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.highlighted_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.highlighted_count_label)

        # Add a label to display the total number of images
        self.image_count_label = QLabel("Total: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.image_count_label)

        # Create and setup table view
        self.tableView = NoArrowKeyTableView(self)
        self.tableView.setSelectionBehavior(QTableView.SelectRows)
        self.tableView.setSelectionMode(QTableView.ExtendedSelection)  # Support multiple selection
        self.tableView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableView.customContextMenuRequested.connect(self.show_context_menu)
        self.tableView.setMouseTracking(True)
        self.tableView.viewport().installEventFilter(self)
        
        # Install event filter for wheel events on the table view
        self.tableView.installEventFilter(self)
        
        # Set the model for the table view
        self.table_model = RasterTableModel(self.raster_manager, self)
        self.tableView.setModel(self.table_model)
        
        # Set column widths - removed checkbox column, adjust accordingly
        self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tableView.setColumnWidth(1, 120)
        self.tableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        
        # Style the header
        self.tableView.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
            background-color: #E0E0E0;
            padding: 4px;
            border: 1px solid #D0D0D0;
            }
        """)
        
        # Connect signals for clicking
        self.tableView.pressed.connect(self.on_table_pressed)
        self.tableView.doubleClicked.connect(self.on_table_double_clicked)
        
        # Add table view to the layout
        info_table_layout.addWidget(self.tableView)

        # Add a new horizontal layout below the table widget to hold the buttons
        self.button_layout = QHBoxLayout()
        info_table_layout.addLayout(self.button_layout)

        # Add 'Highlight All' button to the new layout
        self.highlight_all_button = QPushButton("Highlight All", self)
        self.highlight_all_button.clicked.connect(self.highlight_all_rows)
        self.button_layout.addWidget(self.highlight_all_button)

        # Add 'Unhighlight All' button to the new layout
        self.unhighlight_all_button = QPushButton("Unhighlight All", self)
        self.unhighlight_all_button.clicked.connect(self.unhighlight_all_rows)
        self.button_layout.addWidget(self.unhighlight_all_button)

        # Add the group box to the main layout
        self.layout.addWidget(self.info_table_group)
        
    def setup_signals(self):
        """Set up signal connections."""
        # Connect annotation signals
        self.annotation_window.annotationCreated.connect(self.update_annotation_count)
        self.annotation_window.annotationDeleted.connect(self.update_annotation_count)
        
        # Connect our own signals
        self.imageLoaded.connect(self.on_image_loaded)
        self.filterChanged.connect(self.update_image_count_label)
        
    def schedule_filter(self):
        """Schedule filtering after a short delay to avoid excessive updates."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms delay
        
    @contextmanager
    def busy_cursor(self):
        """Context manager for setting busy cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            
    def show_error(self, title, message):
        """Show an error message dialog."""
        QMessageBox.warning(self, title, message)
            
    #
    # Event handlers and overrides
    #
    
    def eventFilter(self, source, event):
        """Event filter to track mouse movement over table rows and handle scrolling."""
        # Handle wheel events on the table view to customize scrolling
        if source is self.tableView and event.type() == event.Wheel:
            # Get the direction of the scroll
            delta = event.angleDelta().y()
            
            # Get the current vertical scrollbar
            vscroll = self.tableView.verticalScrollBar()
            current_value = vscroll.value()
            
            # Calculate new position based on single row scrolling
            if delta > 0:
                # Scroll up by one row
                vscroll.setValue(current_value - self.tableView.rowHeight(0))
            else:
                # Scroll down by one row
                vscroll.setValue(current_value + self.tableView.rowHeight(0))
                
            # Event has been handled
            return True
        
        # Handle mouse movement events for showing image previews
        if source is self.tableView.viewport():
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
                index = self.tableView.indexAt(pos)
                row = index.row()
                
                if row != self.hover_row:
                    # Mouse moved to a different row
                    self.hide_image_preview()
                    self.hover_row = row
                    
                    # Only show preview if Ctrl is pressed
                    modifiers = QApplication.keyboardModifiers()
                    if (row >= 0 and row < len(self.table_model.filtered_paths)
                        and modifiers & Qt.ControlModifier):
                        # Show preview immediately
                        self.show_image_preview()
                else:
                    # If still on the same row, check if Ctrl was just pressed
                    modifiers = QApplication.keyboardModifiers()
                    if (row >= 0 and row < len(self.table_model.filtered_paths)
                        and modifiers & Qt.ControlModifier):
                        if not self.preview_tooltip.isVisible():
                            self.show_image_preview()
                    else:
                        self.hide_image_preview()
        
        return super().eventFilter(source, event)
        
    def closeEvent(self, event):
        """Handle the window close event."""
        self.hide_image_preview()
        QApplication.restoreOverrideCursor()
        super().closeEvent(event)
        
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
        
    #
    # Signal handlers
    #
    
    def on_table_pressed(self, index):
        """Handle a single click on the table view."""
        if not index.isValid():
            return
        
        # Get the path at the clicked row
        path = self.table_model.get_path_at_row(index.row())
        if not path:
            return
        
        # Get keyboard modifiers
        modifiers = QApplication.keyboardModifiers()
        
        # Handle highlighting logic
        if modifiers & Qt.ControlModifier:
            # Ctrl+Click: Toggle highlight for the clicked row
            raster = self.raster_manager.get_raster(path)
            if raster:
                self.table_model.highlight_path(path, not raster.is_highlighted)
                # Update highlighted count
                self.update_highlighted_count_label()
                
        elif modifiers & Qt.ShiftModifier:
            # Shift+Click: Highlight range from last highlighted to current
            if self.last_highlighted_row >= 0:
                # Get the current row and last highlighted row
                current_row = index.row()
                
                # Calculate range (handle both directions)
                start_row = min(self.last_highlighted_row, current_row)
                end_row = max(self.last_highlighted_row, current_row)
                
                # Highlight the range
                for row in range(start_row, end_row + 1):
                    path_to_highlight = self.table_model.get_path_at_row(row)
                    if path_to_highlight:
                        self.table_model.highlight_path(path_to_highlight, True)
            else:
                # No previous selection, just highlight the current row
                self.table_model.highlight_path(path, True)
                
            # Update the last highlighted row
            self.last_highlighted_row = index.row()
            
            # Update highlighted count
            self.update_highlighted_count_label()
        else:
            # Regular click: Clear all highlights and highlight only this row
            self.table_model.clear_highlights()
            self.table_model.highlight_path(path, True)
            self.last_highlighted_row = index.row()
            
            # Update highlighted count
            self.update_highlighted_count_label()

    def on_table_double_clicked(self, index):
        """Handle double click on table view (selects image and loads it)."""
        if not index.isValid():
            return
            
        # Get the path at the clicked row
        path = self.table_model.get_path_at_row(index.row())
        if path:
            self.load_image_by_path(path)
        
    def on_raster_added(self, path):
        """Handler for when a raster is added."""
        self.rasterAdded.emit(path)
        self.update_search_bars()
        self.filter_images()
        
    def on_raster_removed(self, path):
        """Handler for when a raster is removed."""
        self.rasterRemoved.emit(path)
        self.update_search_bars()
        self.filter_images()
        
    def on_raster_updated(self, path):
        """Handler for when a raster is updated."""
        self.update_current_image_index_label()
        
    def on_filtering_started(self):
        """Handler for when filtering starts."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
    def on_filtering_finished(self, filtered_paths):
        """Handler for when filtering finishes."""
        # Update the table model with filtered paths
        self.table_model.set_filtered_paths(filtered_paths)
        
        # Update labels
        self.update_current_image_index_label()
        self.update_highlighted_count_label()  # Update highlighted count
        self.filterChanged.emit(len(filtered_paths))
        
        # Restore selection if possible
        if self.selected_image_path in filtered_paths:
            self.table_model.set_selected_path(self.selected_image_path)
            self.select_row_for_path(self.selected_image_path)
        elif filtered_paths and not self.selected_image_path:
            # Load the first image if none is selected
            self.load_first_filtered_image()
            
        QApplication.restoreOverrideCursor()
        
    def on_image_loaded(self, path):
        """Handler for when an image is loaded."""
        self.selected_image_path = path
        
    #
    # Public methods
    #
    
    def add_image(self, image_path):
        """
        Add an image to the window.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            bool: True if the image was added successfully, False otherwise
        """
        # Check if image is already loaded
        if image_path in self.raster_manager.image_paths:
            return True
            
        try:
            # Add the raster to the manager
            result = self.raster_manager.add_raster(image_path)
            if not result:
                raise ValueError("Failed to load the image")
                
            # Immediately update filtered paths to include the new image
            # This ensures the image will be visible in the table right away
            if self.table_model.filtered_paths == self.raster_manager.image_paths[:-1]:
                # No filters are active, so just add the new path
                self.table_model.filtered_paths.append(image_path)
                self.table_model.dataChanged.emit(
                    self.table_model.index(0, 0),
                    self.table_model.index(len(self.table_model.filtered_paths) - 1, 
                                           self.table_model.columnCount() - 1))
            else:
                # Filters are active, so run filtering again
                self.filter_images()
                
            return True
                
        except Exception as e:
            self.show_error("Image Loading Error", 
                          f"Error loading image {os.path.basename(image_path)}:\n{str(e)}")
            return False
                
    @property
    def current_raster(self):
        """Get the currently selected Raster object."""
        if self.selected_image_path:
            return self.raster_manager.get_raster(self.selected_image_path)
        return None
        
    @property
    def filtered_count(self):
        """Get the count of filtered images."""
        return len(self.table_model.filtered_paths)
        
    def update_annotation_count(self, annotation_id):
        """
        Update annotation count when an annotation is created or deleted.
        
        Args:
            annotation_id: ID of the annotation
        """
        # Get the image path for the annotation
        if annotation_id in self.annotation_window.annotations_dict:
            # Get the image path from the annotation
            image_path = self.annotation_window.annotations_dict[annotation_id].image_path
        else:
            # It's already been deleted, so get the current image path
            image_path = self.annotation_window.current_image_path
            
        # Update the annotations for the raster
        self.update_image_annotations(image_path)
        
    def update_image_annotations(self, image_path):
        """
        Update annotation information for a specific image.
        
        Args:
            image_path (str): Path to the image
        """
        # Get the annotations for the image
        annotations = self.annotation_window.get_image_annotations(image_path)
        
        # Update the raster
        self.raster_manager.update_annotation_info(image_path, annotations)
        
        # Update label counts
        self.main_window.label_window.update_annotation_count()
        
    def update_current_image_annotations(self):
        """Update annotations for the currently selected image."""
        if self.selected_image_path:
            self.update_image_annotations(self.selected_image_path)
        
    def load_image_by_path(self, image_path, update=False):
        """
        Load an image by its path.
        
        Args:
            image_path (str): Path to the image
            update (bool): Whether to update the image even if it's already selected
        """
        # Validate path
        if image_path not in self.raster_manager.image_paths:
            return
            
        # Check if already selected
        if image_path == self.selected_image_path and not update:
            return
            
        with self.busy_cursor():
            try:
                # Unhighlight all rows
                self.unhighlight_all_rows()
                                
                # Get the raster
                raster = self.raster_manager.get_raster(image_path)
                
                # Update selection
                self.selected_image_path = image_path
                self.table_model.set_selected_path(image_path)
                self.select_row_for_path(image_path)
                
                # Update index label
                self.update_current_image_index_label()
                
                # Load and display a preview immediately
                low_res_image = raster.get_thumbnail(longest_edge=256)
                self.annotation_window.display_image(low_res_image)
                
                # Load the full resolution image
                q_image = raster.get_qimage()
                if q_image is None or q_image.isNull():
                    raise ValueError("Failed to load the full resolution image")
                    
                # Set the image in the annotation window
                self.annotation_window.set_image(image_path)
                
                # Emit signals
                self.imageSelected.emit(image_path)
                self.imageChanged.emit()
                self.imageLoaded.emit(image_path)
                
                # Prefetch adjacent images
                self.prefetch_adjacent_images()
                
            except Exception as e:
                self.show_error("Image Loading Error",
                              f"Error loading image {os.path.basename(image_path)}:\n{str(e)}")
                
    def select_row_for_path(self, path):
        """
        Select the row for a given path.
        
        Args:
            path (str): Path to select
        """
        row = self.table_model.get_row_for_path(path)
        if row >= 0:
            # Create model index for the row
            model_index = self.table_model.index(row, 0)
            
            # Select the row in the table view
            self.tableView.setCurrentIndex(model_index)
            
            # Do not use selectRow as it triggers clicked signal
            # Use selection model directly to avoid infinite loops
            selection_model = self.tableView.selectionModel()
            if selection_model:
                selection_flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
                selection_model.select(model_index, selection_flags)
                
    def center_table_on_current_image(self):
        """Center the table view on the current image."""
        if not self.selected_image_path:
            return
            
        # Get the row index
        row = self.table_model.get_row_for_path(self.selected_image_path)
        if row >= 0:
            # Get the model index
            index = self.table_model.index(row, 0)
            
            # Scroll to the index
            self.tableView.scrollTo(index, QTableView.PositionAtCenter)
            
    def filter_images(self):
        """Filter images based on current criteria."""
        # Get filter criteria
        search_text = self.search_bar_images.currentText()
        search_label = self.search_bar_labels.currentText()
        no_annotations = self.no_annotations_checkbox.isChecked()
        has_annotations = self.has_annotations_checkbox.isChecked()
        has_predictions = self.has_predictions_checkbox.isChecked()
        highlighted_only = self.highlighted_checkbox.isChecked()
        
        # Get highlighted paths if needed
        highlighted_paths = self.table_model.get_highlighted_paths() if highlighted_only else None
        
        # Run the filter
        self.image_filter.filter_images(
            search_text=search_text,
            search_label=search_label,
            require_annotations=has_annotations,
            require_no_annotations=no_annotations,
            require_predictions=has_predictions,
            selected_paths=highlighted_paths,
            use_threading=True
        )
        
    def update_current_image_index_label(self):
        """Update the label showing current image index."""
        if self.selected_image_path:
            # Get the index in filtered paths
            row = self.table_model.get_row_for_path(self.selected_image_path)
            if row >= 0:
                # Show 1-based index
                self.current_image_index_label.setText(f"Current: {row + 1}")
                return
                
        # No valid selection
        self.current_image_index_label.setText("Current: None")
        
    def update_image_count_label(self, count=None):
        """
        Update the label showing total image count.
        
        Args:
            count (int): Count to display (optional)
        """
        if count is None:
            count = len(self.table_model.filtered_paths)
            
        self.image_count_label.setText(f"Total: {count}")
        
    def update_highlighted_count_label(self):
        """Update the label showing highlighted image count."""
        highlighted_paths = self.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        self.highlighted_count_label.setText(f"Highlighted: {count}")
        
    def show_image_preview(self):
        """Show image preview tooltip for the current hover row."""
        if self.hover_row < 0 or self.hover_row >= len(self.table_model.filtered_paths):
            return
            
        # Get the path and raster
        path = self.table_model.get_path_at_row(self.hover_row)
        if not path:
            return
            
        # Get the thumbnail
        pixmap = self.raster_manager.get_thumbnail(path, longest_edge=256)
        if not pixmap:
            return
            
        # Set image and show tooltip
        self.preview_tooltip.set_image(pixmap)
        
        # Position tooltip near the row
        visual_rect = self.tableView.visualRect(self.table_model.index(self.hover_row, 1))
        pos = self.tableView.viewport().mapToGlobal(
            QPoint(visual_rect.right(), visual_rect.center().y())
        )
        self.preview_tooltip.show_at(pos)
        
    def hide_image_preview(self):
        """Hide the image preview tooltip."""
        self.preview_tooltip.hide()
        
    def update_search_bars(self):
        """Update items in the search bars."""
        # Store current search texts
        current_image_search = self.search_bar_images.currentText()
        current_label_search = self.search_bar_labels.currentText()

        # Clear and update items
        self.search_bar_images.clear()
        self.search_bar_labels.clear()

        try:
            # Get image names
            image_names = set()
            for path in self.raster_manager.image_paths:
                raster = self.raster_manager.get_raster(path)
                if raster:
                    image_names.add(raster.basename)
                    
            # Get label names
            label_names = set()
            # Check if label_window exists in main_window
            if hasattr(self.main_window, 'label_window') and self.main_window.label_window is not None:
                for label in self.main_window.label_window.labels:
                    if hasattr(label, 'short_label_code'):
                        label_names.add(label.short_label_code)
            # Alternative location for labels might be in annotation_window
            elif hasattr(self.annotation_window, 'labels'):
                for label in self.annotation_window.labels:
                    if hasattr(label, 'short_label_code'):
                        label_names.add(label.short_label_code)
        except Exception as e:
            print(f"Error updating search bars: {str(e)}")
            return

        # Only add items if there are any
        if image_names:
            self.search_bar_images.addItems(sorted(image_names))
        if label_names:
            self.search_bar_labels.addItems(sorted(label_names))

        # Restore search texts
        if current_image_search:
            self.search_bar_images.setEditText(current_image_search)
        else:
            self.search_bar_images.setPlaceholderText("Type to search images")
        if current_label_search:
            self.search_bar_labels.setEditText(current_label_search)
        else:
            self.search_bar_labels.setPlaceholderText("Type to search labels")
            
    def load_first_filtered_image(self):
        """Load the first image in the filtered list."""
        if not self.table_model.filtered_paths:
            return
            
        # Clear the scene first
        self.annotation_window.clear_scene()
        
        # Load the first image
        self.load_image_by_path(self.table_model.filtered_paths[0])
        
    def prefetch_adjacent_images(self):
        """
        Prefetch adjacent images for smoother navigation.
        Creates thumbnails in a background thread.
        """
        if not self.selected_image_path or not self.table_model.filtered_paths:
            return
            
        current_index = self.table_model.get_row_for_path(self.selected_image_path)
        if current_index < 0:
            return
            
        # Get next and previous indices
        next_index = (current_index + 1) % len(self.table_model.filtered_paths)
        prev_index = (current_index - 1) % len(self.table_model.filtered_paths)
        
        # Get paths
        paths = []
        if next_index != current_index:
            paths.append(self.table_model.get_path_at_row(next_index))
        if prev_index != current_index:
            paths.append(self.table_model.get_path_at_row(prev_index))
            
        # Start background thread to preload thumbnails
        if paths:
            QThreadPool.globalInstance().start(lambda: self._preload_thumbnails(paths))
            
    def _preload_thumbnails(self, paths):
        """
        Preload thumbnails for the given paths.
        
        Args:
            paths (list): List of paths to preload
        """
        for path in paths:
            if path in self.raster_manager.image_paths:
                raster = self.raster_manager.get_raster(path)
                if raster:
                    # Just access the thumbnail to trigger creation
                    raster.get_thumbnail(longest_edge=256)
                    
    def cycle_previous_image(self):
        """Load the previous image in the filtered list."""
        if not self.selected_image_path or not self.table_model.filtered_paths:
            return
            
        # Get current index
        current_index = self.table_model.get_row_for_path(self.selected_image_path)
        if current_index < 0:
            return
            
        # Get previous index
        prev_index = (current_index - 1) % len(self.table_model.filtered_paths)
        
        # Load the previous image
        self.load_image_by_path(self.table_model.get_path_at_row(prev_index))
        
    def cycle_next_image(self):
        """Load the next image in the filtered list."""
        if not self.selected_image_path or not self.table_model.filtered_paths:
            return
            
        # Get current index
        current_index = self.table_model.get_row_for_path(self.selected_image_path)
        if current_index < 0:
            return
            
        # Get next index
        next_index = (current_index + 1) % len(self.table_model.filtered_paths)
        
        # Load the next image
        self.load_image_by_path(self.table_model.get_path_at_row(next_index))
        
    def highlight_all_rows(self):
        """Highlight all rows in the filtered view."""
        for path in self.table_model.filtered_paths:
            self.table_model.highlight_path(path, True)
        
        # Update the last highlighted row
        if self.table_model.filtered_paths:
            self.last_highlighted_row = self.table_model.get_row_for_path(self.table_model.filtered_paths[-1])
            
        # Update the highlighted count label
        self.update_highlighted_count_label()
        
    def unhighlight_all_rows(self):
        """Clear all highlights."""
        self.table_model.clear_highlights()
        self.last_highlighted_row = -1
        
        # Update the highlighted count label
        self.update_highlighted_count_label()
        
    def show_context_menu(self, position):
        """
        Show the context menu for the table.
        
        Args:
            position (QPoint): Position to show the menu
        """
        highlighted_paths = self.table_model.get_highlighted_paths()
        if not highlighted_paths:
            # If no highlights, highlight the row under the cursor only
            index = self.tableView.indexAt(position)
            if index.isValid():
                path = self.table_model.get_path_at_row(index.row())
                if path:
                    self.table_model.set_highlighted_paths([path])
                    self.last_highlighted_row = index.row()
                    highlighted_paths = [path]
        else:
            # If any highlights, ensure all highlighted rows are used (no change needed)
            self.table_model.set_highlighted_paths(highlighted_paths)
        if not highlighted_paths:
            return
        context_menu = QMenu(self)
        count = len(highlighted_paths)
        delete_images_action = context_menu.addAction(f"Delete {count} Highlighted Image{'s' if count > 1 else ''}")
        delete_images_action.triggered.connect(lambda: self.delete_highlighted_images())
        delete_annotations_action = context_menu.addAction(
            f"Delete Annotations for {count} Highlighted Image{'s' if count > 1 else ''}"
        )
        delete_annotations_action.triggered.connect(
            lambda: self.delete_highlighted_images_annotations()
        )
        context_menu.exec_(self.tableView.viewport().mapToGlobal(position))
        
    def delete_highlighted_images(self):
        """Delete the highlighted images."""
        # Get all highlighted paths, the same way we do in filter_images
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Multiple Image Deletions",
            f"Are you sure you want to delete {len(highlighted_paths)} images?\n"
            "This will delete all associated annotations.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_images(highlighted_paths)
            
    def delete_highlighted_images_annotations(self):
        """Delete annotations for the highlighted images."""
        # Get all highlighted paths, the same way we do in filter_images
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Multiple Annotation Deletions",
            f"Are you sure you want to delete annotations for {len(highlighted_paths)} images?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Temporarily disconnect annotation signals
            self.annotation_window.annotationCreated.disconnect(self.update_annotation_count)
            self.annotation_window.annotationDeleted.disconnect(self.update_annotation_count)
            
            try:
                # Show progress
                progress_bar = ProgressBar(self, title="Deleting Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(highlighted_paths))
                
                # Delete annotations
                for path in highlighted_paths:
                    # Delete the annotations
                    self.annotation_window.delete_image_annotations(path)
                    
                    # Update the raster
                    self.update_image_annotations(path)
                    
                    # Update progress
                    progress_bar.update_progress()
                    
            finally:
                # Restore signals
                self.annotation_window.annotationCreated.connect(self.update_annotation_count)
                self.annotation_window.annotationDeleted.connect(self.update_annotation_count)
                
                # Close progress bar
                if 'progress_bar' in locals() and progress_bar:
                    progress_bar.stop_progress()
                    progress_bar.close()
                    
            # Reapply filters
            self.filter_images()
                
    def delete_images(self, image_paths):
        """
        Delete multiple images and their annotations.
        
        Args:
            image_paths (list): List of paths to delete
        """
        # Make sure paths are valid
        image_paths = [path for path in image_paths if path in self.raster_manager.image_paths]
        if not image_paths:
            return
            
        # Check if current image is being deleted
        is_current_deleted = self.selected_image_path in image_paths
        
        # Determine next image to select
        next_image = None
        if is_current_deleted and self.table_model.filtered_paths:
            # Find remaining images
            remaining = [p for p in self.table_model.filtered_paths if p not in image_paths]
            
            if remaining:
                # Get index of current image
                current_index = self.table_model.get_row_for_path(self.selected_image_path)
                
                # Find images before the current one
                before_current = [p for p in remaining if self.table_model.get_row_for_path(p) <= current_index]
                
                # Prefer images before current, otherwise use any remaining
                next_image = before_current[0] if before_current else remaining[0]
                
        # Show progress
        with self.busy_cursor():
            progress_bar = ProgressBar(self, title="Deleting Images")
            progress_bar.show()
            progress_bar.start_progress(len(image_paths))
            
            try:
                # Delete each image
                for path in image_paths:
                    # Remove from annotation window
                    self.annotation_window.delete_image(path)
                    
                    # Remove from raster manager
                    self.raster_manager.remove_raster(path)
                    
                    # Update progress
                    progress_bar.update_progress()
                    
                # Update UI
                if next_image:
                    self.load_image_by_path(next_image)
                elif not self.raster_manager.image_paths:
                    self.selected_image_path = None
                    self.annotation_window.clear_scene()
                    
            finally:
                # Close progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                
            # Update UI
            self.filter_images()


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
        """
        Set the preview image.
        
        Args:
            pixmap (QPixmap): Image to display
        """
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
        """
        Position and show the tooltip at the specified global position,
        always placing it to the bottom-right of the cursor.
        
        Args:
            global_pos (QPoint): Position to show the tooltip
        """
        # Always position to bottom-right of cursor with fixed offset
        x, y = global_pos.x() + 25, global_pos.y() + 25
        
        # Ensure tooltip stays within screen boundaries
        screen_rect = self.screen().geometry()
        tooltip_size = self.sizeHint()
        
        # Adjust position if needed to stay on screen
        if x + tooltip_size.width() > screen_rect.right():
            x = screen_rect.right() - tooltip_size.width() - 10
        if y + tooltip_size.height() > screen_rect.bottom():
            y = screen_rect.bottom() - tooltip_size.height() - 10
            
        # Set position and show
        self.move(x, y)
        self.show()