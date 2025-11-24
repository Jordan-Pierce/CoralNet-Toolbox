import warnings

import os
from contextlib import contextmanager

import rasterio

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint, QThreadPool, QItemSelectionModel, QModelIndex
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QComboBox, QHBoxLayout, QTableView, QHeaderView, QApplication, 
                             QMenu, QButtonGroup, QGroupBox, QPushButton, QStyle, 
                             QFormLayout, QFrame, QLineEdit, QListWidget, QListWidgetItem, QFileDialog)

from coralnet_toolbox.Rasters import RasterManager, ImageFilter, RasterTableModel

from coralnet_toolbox.Common.QtZChannelImport import ZPairingWidget

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class NoArrowKeyTableView(QTableView):
    # Custom signal to be emitted only on a left-click
    leftClicked = pyqtSignal(QModelIndex)

    def __init__(self, image_window, parent=None):
        super().__init__(parent)
        self.image_window = image_window

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            event.ignore()
            return
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            # Handle Ctrl+A to highlight all rows in addition to selecting all
            self.image_window.highlight_all_rows()
            # Fall through to let the default selection behavior happen
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # On a left mouse press, emit our custom signal
        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.leftClicked.emit(index)
        # Call the base class implementation to handle standard behavior
        # like row selection and context menu triggers.
        super().mousePressEvent(event)
        

class CheckableComboBox(QComboBox):
    """
    A custom QComboBox that displays checkable items in its dropdown list
    and stays open to allow for multiple selections.
    """
    # Signal to emit when the check state changes
    filterChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Use a non-editable QLineEdit to display selected items
        self.line_edit = QLineEdit(self)
        self.line_edit.setReadOnly(True)
        self.setLineEdit(self.line_edit)

        self.placeholder_text = "Select filters..."
        self.line_edit.setText(self.placeholder_text)
        
        # Flag to prevent recursive signal handling
        self._block_signals = False

        # Connect the itemChanged signal from the model
        self.model().itemChanged.connect(self.on_item_changed)
        
        # Install an event filter on the view's viewport
        # to prevent the popup from closing when an item is clicked.
        self.view().viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        """
        Event filter to keep the popup open and manually toggle checks.
        """
        if obj == self.view().viewport() and event.type() == event.MouseButtonRelease:
            # Check if the click was on a valid item index
            index = self.view().indexAt(event.pos())
            if index.isValid():
                
                # Get the item that was clicked
                item = self.model().itemFromIndex(index)
                if item:
                    # Find the current state and set the new, opposite state
                    state = item.checkState()
                    new_state = Qt.Checked if state == Qt.Unchecked else Qt.Unchecked
                    
                    # Manually set the new check state
                    # This will correctly trigger the on_item_changed signal
                    item.setCheckState(new_state)

                # We still return True to prevent the popup from closing
                # after the click.
                return True 
        
        # Pass on all other events
        return super().eventFilter(obj, event)

    def hidePopup(self):
        """
        Override hidePopup to update the display text when the popup closes.
        """
        self.update_display_text()
        super().hidePopup()
        
    def addItem(self, text):
        """Add a checkable item to the list."""
        super().addItem(text)
        
        # Get the QStandardItem we just added
        item = self.model().item(self.count() - 1)
        
        # Set it to be checkable
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)

    def on_item_changed(self, item):
        """Handle when an item's check state is changed."""
        if self._block_signals:
            return  # Prevent recursion

        self._block_signals = True
        
        text = item.text()
        state = item.checkState()

        # --- Enforce Mutual Exclusivity ---
        if state == Qt.Checked:
            if text == "Has Annotations":
                self.uncheck_item("No Annotations")
            elif text == "No Annotations":
                self.uncheck_item("Has Annotations")
        # ----------------------------------

        self.filterChanged.emit()
        self._block_signals = False

    def uncheck_item(self, text):
        """Find an item by its text and uncheck it."""
        for i in range(self.count()):
            item = self.model().item(i)
            if item.text() == text and item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
                break

    def update_display_text(self):
        """Update the QLineEdit to show the currently checked items."""
        checked = self.get_checked_items()
        if not checked:
            self.line_edit.setText(self.placeholder_text)
        else:
            self.line_edit.setText(", ".join(checked))

    def get_checked_items(self):
        """Return a list of strings for all checked items."""
        checked = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked
        

class ImageWindow(QWidget):
    # Signals
    imageSelected = pyqtSignal(str)  # Path of selected image
    imageChanged = pyqtSignal()  # When image changes
    imageLoaded = pyqtSignal(str)  # When image is fully loaded
    filterChanged = pyqtSignal(int)  # Number of filtered images
    rasterAdded = pyqtSignal(str)  # Path of added raster
    rasterRemoved = pyqtSignal(str)  # Path of removed raster
    filterGroupToggled = pyqtSignal(bool)  # When filter group is toggled
    zChannelRemoved = pyqtSignal(str)  # Path of raster with removed z-channel

    def __init__(self, main_window):
        """Initialize the ImageWindow widget."""
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.is_loading = False

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

        # --- Create a container widget for all contents ---
        # This one widget will hold everything inside the group box.
        self.filter_content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.filter_content_widget)
        # Remove padding so it looks seamless inside the group box
        self.content_layout.setContentsMargins(0, 0, 0, 0) 

        # Create a form layout for the search bars
        self.search_layout = QFormLayout()
        # --- Add search_layout to the new content_layout ---
        self.content_layout.addLayout(self.search_layout)

        # Set fixed width for search bars (big effect on layout width)
        fixed_width = 125

        # --- Setup Filter ComboBox ---
        self.filter_combo = CheckableComboBox(self)
        self.filter_combo.addItem("Highlighted")
        self.filter_combo.addItem("Has Predictions")
        self.filter_combo.addItem("Has Annotations")
        self.filter_combo.addItem("No Annotations")
        
        self.filter_combo.setCurrentIndex(-1)  
        
        self.filter_combo.filterChanged.connect(self.schedule_filter)

        # --- Create containers for search bars and buttons ---
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

        # Add top-k combo box
        self.top_k_combo = QComboBox(self)
        self.top_k_combo.addItems(["Top1", "Top2", "Top3", "Top4", "Top5"])
        self.top_k_combo.setCurrentText("Top1")
        self.top_k_combo.setFixedWidth(60)
        self.top_k_combo.currentTextChanged.connect(self.schedule_filter)
        self.label_search_layout.addWidget(self.top_k_combo)

        self.label_search_button = QPushButton(self)
        self.label_search_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.label_search_button.clicked.connect(self.filter_images)
        self.label_search_layout.addWidget(self.label_search_button)
        
        # --- Set horizontal policy to expand and fill the layout column ---
        self.filter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Add rows to form layout
        self.search_layout.addRow("Filters:", self.filter_combo)
        self.search_layout.addRow("Search Images:", self.image_search_container)
        self.search_layout.addRow("Search Labels:", self.label_search_container)

        # --- Add the single content widget to the group's layout ---
        self.filter_layout.addWidget(self.filter_content_widget)

        # --- Make the group box checkable and connect its signal ---
        self.filter_group.setCheckable(True)
        self.filter_group.toggled.connect(self.on_filter_group_toggled)
        
        # Set the default state to checked (expanded)
        self.filter_group.setChecked(True)

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
        
        # Set column widths
        self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Checkmark column
        self.tableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Z column
        self.tableView.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Filename column
        self.tableView.setColumnWidth(3, 120)  # Annotation column
        self.tableView.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        
        # Style the header
        self.tableView.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
            background-color: #E0E0E0;
            padding: 4px;
            border: 1px solid #D0D0D0;
            }
        """)
        
        # Connect signals for clicking
        self.tableView.leftClicked.connect(self.on_table_pressed)
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
        
    def on_filter_group_toggled(self, checked):
        """
        Shows or hides the filter content by changing its maximum height,
        which preserves the horizontal width.
        """
        if checked:
            # Set a very large max height (the default "no limit")
            self.filter_content_widget.setMaximumHeight(16777215)
        else:
            # Set max height to 0 to collapse it
            self.filter_content_widget.setMaximumHeight(0)
        
        # Emit signal to MainWindow to expand/collapse layout
        self.filterGroupToggled.emit(checked)
        
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
        """Handle a single left-click on the table view with complex modifier support."""
        if not index.isValid():
            return

        path = self.table_model.get_path_at_row(index.row())
        if not path:
            return
        
        modifiers = QApplication.keyboardModifiers()
        current_row = index.row()

        # Define conditions for modifiers
        has_ctrl = bool(modifiers & Qt.ControlModifier)
        has_shift = bool(modifiers & Qt.ShiftModifier)

        if has_shift:
            # This block handles both Shift+Click and Ctrl+Shift+Click.
            # First, determine the paths in the selection range.
            range_paths = []
            if self.last_highlighted_row >= 0:
                start = min(self.last_highlighted_row, current_row)
                end = max(self.last_highlighted_row, current_row)
                for r in range(start, end + 1):
                    p = self.table_model.get_path_at_row(r)
                    if p:
                        range_paths.append(p)
            else:
                # If there's no anchor, the range is just the clicked item.
                range_paths.append(path)

            if not has_ctrl:
                # Case 1: Simple Shift+Click. Clears previous highlights 
                # and selects only the new range.
                self.table_model.set_highlighted_paths(range_paths)
            else:
                # Case 2: Ctrl+Shift+Click. Adds the new range to the
                # existing highlighted rows without clearing them.
                for p in range_paths:
                    self.table_model.highlight_path(p, True)
        
        elif has_ctrl:
            # Case 3: Ctrl+Click. Toggles a single row's highlight state
            # and sets it as the new anchor for future shift-clicks.
            raster = self.raster_manager.get_raster(path)
            if raster:
                self.table_model.highlight_path(path, not raster.is_highlighted)
            self.last_highlighted_row = current_row
        
        else:
            # Case 4: Plain Click. Clears everything and highlights only
            # the clicked row, setting it as the new anchor.
            self.table_model.set_highlighted_paths([path])
            self.last_highlighted_row = current_row
        
        # Finally, update the count label after any changes.
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
        
        # If this raster is currently being displayed, refresh the z-channel visualization
        # (this handles the case where a z-channel is newly imported for the current image)
        if path == self.annotation_window.current_image_path:
            self.annotation_window.refresh_z_channel_visualization()
        
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
        
    def on_toggle(self, new_state: bool):
        """
        Sets the checked state for all currently highlighted rows.

        Args:
            new_state (bool): The new state to set (True for checked, False for unchecked).
        """
        highlighted_paths = self.table_model.get_highlighted_paths()
        if not highlighted_paths:
            return

        for path in highlighted_paths:
            raster = self.raster_manager.get_raster(path)
            if raster:
                raster.checkbox_state = new_state
                # Notify the model to update the view for this specific raster
                self.table_model.update_raster_data(path)
            
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
            if len(self.table_model.filtered_paths) == len(self.raster_manager.image_paths) - 1:
                # No filters are active, so efficiently add the new path to the model
                self.table_model.add_path(image_path)
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
        if self.is_loading:
            return
            
        self.is_loading = True  # Set the lock
        
        try:
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
                    
                    # Get the raster (but don't load data from it yet)
                    raster = self.raster_manager.get_raster(image_path)
                    
                    # Mark as checked when viewed
                    raster.checkbox_state = True
                    self.table_model.update_raster_data(image_path)
                    
                    # Update selection
                    self.selected_image_path = image_path
                    self.table_model.set_selected_path(image_path)
                    self.select_row_for_path(image_path)
                    
                    # Update index label
                    self.update_current_image_index_label()
                    
                    # This single call now handles the staged load (low-res -> high-res)
                    # We pass the raster object directly to avoid a duplicate lookup
                    self.annotation_window.set_image(image_path)
                    
                    # Emit signals
                    self.imageSelected.emit(image_path)
                    self.imageChanged.emit()
                    self.imageLoaded.emit(image_path)
                    
                except Exception as e:
                    # If set_image fails, it will handle its own errors, 
                    # but we catch any other unexpected errors here.
                    self.show_error("Image Loading Error",
                                    f"Error loading image {os.path.basename(image_path)}:\n{str(e)}")
        
        finally:
            self.is_loading = False  # Release the lock

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
        
        # --- Get values from the new CheckableComboBox ---
        checked_filters = self.filter_combo.get_checked_items()
        
        highlighted_only = "Highlighted" in checked_filters
        has_predictions = "Has Predictions" in checked_filters
        has_annotations = "Has Annotations" in checked_filters
        no_annotations = "No Annotations" in checked_filters
        # --- End new logic ---
        
        # Get top-k value from combo box
        top_k_text = self.top_k_combo.currentText()
        top_k = int(top_k_text.replace("Top", "")) if top_k_text.startswith("Top") else 1
        
        # Get highlighted paths if needed
        highlighted_paths = self.table_model.get_highlighted_paths() if highlighted_only else None
        
        # Run the filter
        self.image_filter.filter_images(
            search_text=search_text,
            search_label=search_label,
            top_k=top_k,
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
        Show the context menu for the table, including the toggle check state action.
        
        Args:
            position (QPoint): Position to show the menu
        """
        # Get the path corresponding to the right-clicked row
        index = self.tableView.indexAt(position)
        path_at_cursor = self.table_model.get_path_at_row(index.row()) if index.isValid() else None

        # Get the currently highlighted paths from the model
        highlighted_paths = self.table_model.get_highlighted_paths()

        # If the user right-clicked on a row that wasn't already highlighted,
        # then we assume they want to act on this row alone.
        if path_at_cursor and path_at_cursor not in highlighted_paths:
            self.table_model.set_highlighted_paths([path_at_cursor])
            self.last_highlighted_row = index.row()
            highlighted_paths = [path_at_cursor]
        
        # If no rows are highlighted, do nothing.
        if not highlighted_paths:
            return

        context_menu = QMenu(self)
        count = len(highlighted_paths)
        
        # Add the check/uncheck action
        raster_under_cursor = self.raster_manager.get_raster(path_at_cursor)
        if raster_under_cursor:
            is_checked = raster_under_cursor.checkbox_state
            if is_checked:
                action_text = f"Uncheck {count} Highlighted Image{'s' if count > 1 else ''}"
            else:
                action_text = f"Check {count} Highlighted Image{'s' if count > 1 else ''}"
            toggle_check_action = context_menu.addAction(action_text)
            toggle_check_action.triggered.connect(lambda: self.on_toggle(not is_checked))

        context_menu.addSeparator()
        
        # Add batch inference action
        batch_inference_action = context_menu.addAction(
            f"Batch Inference ({count} Highlighted Image{'s' if count > 1 else ''})"
        )
        batch_inference_action.triggered.connect(
            lambda: self.open_batch_inference_dialog(highlighted_paths)
        )
        
        context_menu.addSeparator()

        # Add import z-channel action
        import_z_channel_action = context_menu.addAction(
            f"Import Z-Channel for {count} Highlighted Image{'s' if count > 1 else ''}"
        )
        import_z_channel_action.triggered.connect(
            lambda: self.import_z_channel_highlighted_images()
        )

        # Add remove z-channel action
        remove_z_channel_action = context_menu.addAction(
            f"Remove Z-Channel from {count} Highlighted Image{'s' if count > 1 else ''}"
        )
        remove_z_channel_action.triggered.connect(
            lambda: self.remove_z_channel_highlighted_images()
        )

        context_menu.addSeparator()

        # Add delete actions
        delete_images_action = context_menu.addAction(f"Delete {count} Highlighted Image{'s' if count > 1 else ''}")
        delete_images_action.triggered.connect(lambda: self.delete_highlighted_images())
        delete_annotations_action = context_menu.addAction(
            f"Delete Annotations for {count} Highlighted Image{'s' if count > 1 else ''}"
        )
        delete_annotations_action.triggered.connect(
            lambda: self.delete_highlighted_images_annotations()
        )
        context_menu.exec_(self.tableView.viewport().mapToGlobal(position))
        
    def open_batch_inference_dialog(self, highlighted_image_paths):
        """
        Open the batch inference dialog with the highlighted images.
        
        Args:
            highlighted_image_paths (list): List of image paths to process
        """
        # Ensure images are highlighted
        if not highlighted_image_paths:
            QMessageBox.warning(
                self,
                "No Images Selected",
                "Please highlight one or more images before opening batch inference."
            )
            return
        
        # Check if any models are available
        batch_dialog = self.main_window.batch_inference_dialog
        batch_dialog.update_model_availability()
        
        if not batch_dialog.model_dialogs:
            QMessageBox.warning(
                self,
                "No Models Available",
                "Please load a model before opening batch inference."
            )
            return
        
        # Update the batch inference dialog with the highlighted images
        batch_dialog.highlighted_images = highlighted_image_paths
        # Show the dialog
        batch_dialog.exec_()
        
    def import_z_channel_highlighted_images(self):
        """Open file dialog and ZPairingWidget to import z-channel files for highlighted images."""
        # Get all highlighted paths
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
        
        # Build intelligent filters based on image basenames
        image_basenames = [os.path.splitext(os.path.basename(path))[0] for path in highlighted_paths]
        unique_basenames = sorted(set(image_basenames))
        
        filter_strings = ["Image Files (*.tif *.tiff *.png *.bmp *.jp2 *.jpg *.jpeg)"]
        
        # Add a combined filter for all selected image basenames
        if unique_basenames:
            # Create a single filter that matches all selected basenames
            basename_patterns = " ".join([f"{bn}.*" for bn in unique_basenames])
            num_images = len(unique_basenames)
            filter_label = f"Selected Images ({num_images})" if num_images > 1 else f"{unique_basenames[0]} Files"
            filter_strings.append(f"{filter_label} ({basename_patterns})")
        
        filter_strings.append("All Files (*)")
        combined_filter = ";;".join(filter_strings)
        
        # Open file dialog to select z-channel files
        z_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Z-Channel Files",
            "",
            combined_filter
        )
        
        if not z_files:
            return
        
        # Create and show ZPairingWidget with highlighted image paths and selected z-files
        # Sort both lists for consistent ordering
        image_paths = sorted(highlighted_paths)
        z_channel_files = sorted(z_files)
        
        # Create the pairing widget and keep a reference to prevent garbage collection
        self.pairing_widget = ZPairingWidget(image_paths, z_channel_files)
        
        # Connect the mapping_confirmed signal to handle the confirmed mapping
        self.pairing_widget.mapping_confirmed.connect(self.on_z_channel_mapping_confirmed)
        
        # Show the widget
        self.pairing_widget.show()
    
    def on_z_channel_mapping_confirmed(self, mapping):
        """Handle confirmed z-channel mapping from ZPairingWidget.
        
        Args:
            mapping (dict): {image_path: {"z_path": z_channel_path, "units": unit_str}}
        """
        if not mapping:
            return
        
        successful_count = 0
        failed_count = 0
        failed_images = []
        
        # Show progress bar and set busy cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing Z-Channels")
        progress_bar.show()
        progress_bar.start_progress(len(mapping))
        
        try:
            # Apply the z-channel to each raster
            for image_path, z_info in mapping.items():
                # Extract z_path and units from mapping
                if isinstance(z_info, dict):
                    z_channel_path = z_info.get("z_path")
                    z_unit = z_info.get("units")
                else:
                    # Fallback for old-style mappings (just paths)
                    z_channel_path = z_info
                    z_unit = None
                
                raster = self.raster_manager.get_raster(image_path)
                if raster:
                    try:
                        # Load z-channel from file with units
                        success = raster.load_z_channel_from_file(z_channel_path, z_unit=z_unit)
                        if success:
                            successful_count += 1
                            # Emit signal to update UI
                            self.raster_manager.rasterUpdated.emit(image_path)
                        else:
                            failed_count += 1
                            failed_images.append(os.path.basename(image_path))
                    except Exception as e:
                        failed_count += 1
                        failed_images.append(os.path.basename(image_path))
                        print(f"Exception loading z-channel for {image_path}: {str(e)}")
                
                # Update progress
                progress_bar.update_progress()
            
            # Show appropriate message based on results
            if failed_count > 0:
                # Show warning if there were any failures
                failed_list = "\n".join(f"  • {img}" for img in failed_images[:10])
                if len(failed_images) > 10:
                    failed_list += f"\n  ... and {len(failed_images) - 10} more"
                
                message = (
                    f"Z-Channel Import Results:\n\n"
                    f"✓ Successfully loaded: {successful_count}\n"
                    f"✗ Failed to load: {failed_count}\n\n"
                    f"Failed images:\n{failed_list}"
                )
                QMessageBox.warning(
                    self,
                    "Z-Channel Import - Partial Success",
                    message
                )
            elif successful_count > 0:
                # Show success message only if all loaded successfully
                QMessageBox.information(
                    self,
                    "Z-Channel Imported",
                    f"Z-channel imported for {successful_count} image{'s' if successful_count > 1 else ''}."
                )
        finally:
            # Restore cursor and close progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()
        
    def remove_z_channel_highlighted_images(self):
        """Remove z-channel from the highlighted images."""
        # Get all highlighted paths
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
        
        # Confirm removal
        count = len(highlighted_paths)
        plural = 's' if count > 1 else ''
        reply = QMessageBox.question(
            self,
            "Confirm Z-Channel Removal",
            f"Are you sure you want to remove the z-channel from {count} image{plural}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Show progress bar and set busy cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Removing Z-Channels")
            progress_bar.show()
            progress_bar.start_progress(len(highlighted_paths))
            
            try:
                # Remove z-channel from each raster
                for path in highlighted_paths:
                    raster = self.raster_manager.get_raster(path)
                    if raster:
                        raster.remove_z_channel()
                        # Emit signal to update UI
                        self.raster_manager.rasterUpdated.emit(path)
                        # Emit signal for z-channel removal
                        self.zChannelRemoved.emit(path)
                    
                    # Update progress
                    progress_bar.update_progress()
                
                # If current image is in the list, refresh the annotation window
                if self.selected_image_path in highlighted_paths:
                    self.annotation_window.update_scene()
                
                # Show success message
                QMessageBox.information(
                    self,
                    "Z-Channel Removed",
                    f"Z-channel removed from {count} image{plural}."
                )
            finally:
                # Restore cursor and close progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
        
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
            # Set busy cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                # Untoggle tools
                self.main_window.untoggle_all_tools()
                # Delete images
                self.delete_images(highlighted_paths)
            finally:
                QApplication.restoreOverrideCursor()
            
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
            # Untoggle tools
            self.main_window.untoggle_all_tools()
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
            
            # Temporarily disconnect raster manager signals to avoid triggering filters on each removal
            self.raster_manager.rasterAdded.disconnect(self.on_raster_added)
            self.raster_manager.rasterRemoved.disconnect(self.on_raster_removed)
            
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
                # Restore signals
                self.raster_manager.rasterAdded.connect(self.on_raster_added)
                self.raster_manager.rasterRemoved.connect(self.on_raster_removed)
                
                # Close progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                
            # Update search bars and reapply filters once after all deletions
            self.update_search_bars()
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
        """Position and show the tooltip at the specified global position."""
        # Position tooltip to bottom-right of cursor
        x, y = global_pos.x() + 15, global_pos.y() + 15

        # Get the screen that contains the cursor position
        screen = QApplication.screenAt(global_pos)
        if not screen:
            screen = QApplication.primaryScreen()

        # Get screen geometry and tooltip size
        screen_rect = screen.geometry()
        tooltip_size = self.sizeHint()

        # Adjust position to stay on screen
        if x + tooltip_size.width() > screen_rect.right():
            x = global_pos.x() - tooltip_size.width() - 15
        if y + tooltip_size.height() > screen_rect.bottom():
            y = global_pos.y() - tooltip_size.height() - 15

        # Set position and show
        self.move(x, y)
        self.show()
