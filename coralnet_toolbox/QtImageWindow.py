import warnings

import gc
import os
from contextlib import contextmanager

import rasterio

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint, QItemSelectionModel, QModelIndex, QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QWidget, QVBoxLayout, QLabel, 
                             QComboBox, QHBoxLayout, QTableView, QHeaderView, QApplication, 
                             QMenu, QPushButton, QStyle, QFormLayout, QFrame, 
                             QLineEdit, QFileDialog, QToolBar)

from coralnet_toolbox.Rasters import RasterManager
from coralnet_toolbox.Rasters import ImageFilter
from coralnet_toolbox.Rasters import RasterTableModel

from coralnet_toolbox.Z import ZImportDialog
from coralnet_toolbox.Z import ZExportDialog
from coralnet_toolbox.IO import ImportCameras

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox import theme as app_theme

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
            self.image_window.highlight_all_rows(select_rows=False)
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
        self.raster_manager.zChannelUpdated.connect(self.on_z_channel_updated)
        
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
        """Set up the user interface. The payload is ONLY the table view now."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Instantiate the widgets so __init__ can connect signals
        self._init_filter_widgets()
        self._init_info_widgets()
        self._init_table_widget()
        self._init_action_widgets()
        
        # Build and add the payload (the table) to the main layout
        self.layout.addWidget(self.tableView)
        
        # Create the tooltip for previews
        self.preview_tooltip = ImagePreviewTooltip()

    def _init_filter_widgets(self):
        """Set up the filter section widgets without adding to main layout."""
        self.filter_content_widget = QWidget()
        self.filter_content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.filter_layout = QVBoxLayout(self.filter_content_widget)
        self.filter_layout.setContentsMargins(4, 4, 4, 4) # Add slight padding for toolbar aesthetics
        
        self.search_layout = QFormLayout()
        self.search_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.search_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.search_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.search_layout.setHorizontalSpacing(app_theme.scale_int(8))
        self.search_layout.setVerticalSpacing(app_theme.scale_int(6))
        self.filter_layout.addLayout(self.search_layout)

        # --- Setup Filter ComboBox ---
        self.filter_combo = CheckableComboBox(self)
        self.filter_combo.addItem("Highlighted")
        self.filter_combo.addItem("Image")
        self.filter_combo.addItem("Ortho")
        self.filter_combo.addItem("Video")
        self.filter_combo.addItem("Has Z-Channel")
        self.filter_combo.addItem("has Transform")
        self.filter_combo.addItem("Has Predictions")
        self.filter_combo.addItem("Has Annotations")
        self.filter_combo.addItem("No Annotations")
        self.filter_combo.setCurrentIndex(-1)
        self.filter_combo.filterChanged.connect(self.schedule_filter)
        self.filter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.filter_combo.setToolTip("Filter images by type (Image, Ortho, Video), Z-channel presence, predictions, annotation status, and highlight state.\nSelect multiple filters to apply all criteria.")

        # Setup filter/search controls
        self.search_layout.addRow("Filters:", self.filter_combo)

        # Setup image search
        self.search_bar_images = QComboBox(self)
        self.search_bar_images.setEditable(True)
        self.search_bar_images.setPlaceholderText("Type to search images")
        self.search_bar_images.setInsertPolicy(QComboBox.NoInsert)
        self.search_bar_images.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_bar_images.editTextChanged.connect(self.schedule_filter)
        self.search_bar_images.setToolTip("Search for images by filename or file path.\nMatches any image basename containing the search text.")
        self.search_layout.addRow("Search Images:", self.search_bar_images)

        # Setup label search
        self.search_bar_labels = QComboBox(self)
        self.search_bar_labels.setEditable(True)
        self.search_bar_labels.setPlaceholderText("Type to search labels")
        self.search_bar_labels.setInsertPolicy(QComboBox.NoInsert)
        self.search_bar_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_bar_labels.editTextChanged.connect(self.schedule_filter)
        self.search_bar_labels.setToolTip("Search for images by annotation label name.\nShows only images containing annotations with labels matching the search text.")
        self.search_layout.addRow("Search Labels:", self.search_bar_labels)

    def _init_info_widgets(self):
        """Instantiate info labels and home button."""
        home_button_size = app_theme.scale_int(26)
        self.home_button = QPushButton("", self)
        self.home_button.setToolTip("Center table on current image")
        self.home_button.setIcon(get_icon("home.svg"))  
        self.home_button.setFixedSize(home_button_size, home_button_size)
        self.home_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.home_button.setIconSize(app_theme.scale_size(16))
        self.home_button.setFlat(True)    
        self.home_button.clicked.connect(self.center_table_on_current_image)

        self.current_image_index_label = QLabel("Current: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.highlighted_count_label = QLabel("Highlighted: 0", self)
        self.highlighted_count_label.setAlignment(Qt.AlignCenter)
        self.highlighted_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.image_count_label = QLabel("Total: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _init_table_widget(self):
        """Instantiate and configure the central table view."""
        self.tableView = NoArrowKeyTableView(self)
        self.tableView.setSelectionBehavior(QTableView.SelectRows)
        self.tableView.setSelectionMode(QTableView.ExtendedSelection)
        self.tableView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableView.customContextMenuRequested.connect(self.show_context_menu)
        self.tableView.setMouseTracking(True)
        self.tableView.setWordWrap(False)
        self.tableView.setTextElideMode(Qt.ElideRight)
        self.tableView.viewport().installEventFilter(self)
        self.tableView.installEventFilter(self)
        
        self.table_model = RasterTableModel(self.raster_manager, self)
        self.tableView.setModel(self.table_model)
        
        self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents) 
        self.tableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents) 
        self.tableView.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) 
        self.tableView.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        
        self.tableView.horizontalHeader().setStyleSheet(
            app_theme.scale_qss(
                """
            QHeaderView::section {
            background-color: %s;
            padding: 4px;
            border: 1px solid %s;
            color: %s;
            }
        """ % (app_theme.SURFACE_COLOR.name(), app_theme.SURFACE_BORDER_COLOR.name(), app_theme.TEXT_PRIMARY_COLOR.name()))
        )
        
        self.tableView.leftClicked.connect(self.on_table_pressed)
        self.tableView.doubleClicked.connect(self.on_table_double_clicked)

    def _init_action_widgets(self):
        """Instantiate bulk action buttons. Replaced two buttons with a single toggle button."""
        self.toggle_highlight_button = QPushButton("Toggle Highlighted", self)
        self.toggle_highlight_button.setToolTip("Highlight all filtered images or unhighlight if all highlighted")
        self.toggle_highlight_button.clicked.connect(self.toggle_highlighted)
        # Disabled by default until there are filtered rows
        self.toggle_highlight_button.setEnabled(False)

    def trigger_highlight_all_shortcut(self):
        """Invoke the same Ctrl+A shortcut path used by the table view."""
        if not self.table_model.filtered_paths:
            return

        self.tableView.setFocus(Qt.ShortcutFocusReason)
        key_event = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.ControlModifier, "a")
        QApplication.sendEvent(self.tableView, key_event)

    def toggle_highlighted(self):
        """Toggle highlight state: highlight all if any unhighlighted, unhighlight all if all highlighted."""
        # No-op if there are no filtered paths
        filtered = getattr(self.table_model, 'filtered_paths', [])
        if not filtered:
            return

        # Use the model API to get currently highlighted paths
        try:
            highlighted = self.table_model.get_highlighted_paths()
        except Exception:
            highlighted = []

        # If all filtered rows are highlighted, unhighlight all; otherwise highlight all
        if highlighted and len(highlighted) == len(filtered):
            self.unhighlight_all_rows()
        else:
            self.highlight_all_rows(select_rows=True)

    def refresh_scaling(self):
        """Refresh scale-sensitive controls after the global UI scale changes."""
        home_button_size = app_theme.scale_int(26)
        self.home_button.setFixedSize(home_button_size, home_button_size)
        self.home_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.home_button.setIconSize(app_theme.scale_size(16))
        self.tableView.horizontalHeader().setStyleSheet(
            app_theme.scale_qss(
                """
            QHeaderView::section {
            background-color: %s;
            padding: 4px;
            border: 1px solid %s;
            color: %s;
            }
        """ % (app_theme.SURFACE_COLOR.name(), app_theme.SURFACE_BORDER_COLOR.name(), app_theme.TEXT_PRIMARY_COLOR.name()))
        )
        
    # --- DOCK WRAPPER HOOKS ---

    def create_filter_toolbar(self) -> QToolBar:
        """Create the top toolbar containing the search and filter form."""
        toolbar = QToolBar("Image Filters")
        toolbar.setMovable(False)
        
        # Simply mount the entire filter widget into the toolbar
        self.filter_content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.filter_content_widget)
        
        return toolbar

    def create_info_toolbar(self) -> QToolBar:
        """Create a toolbar for the table statistics and home button."""
        toolbar = QToolBar("Image Info")
        toolbar.setMovable(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(app_theme.scale_int(6))

        layout.addWidget(self.home_button, 0)
        layout.addWidget(self.current_image_index_label, 1)
        layout.addWidget(self.highlighted_count_label, 1)
        layout.addWidget(self.image_count_label, 1)

        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(container)

        return toolbar

    def create_action_toolbar(self) -> QToolBar:
        """Create the bottom toolbar for bulk table actions."""
        toolbar = QToolBar("Image Actions")
        toolbar.setMovable(False)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Use the new toggle button in place of separate highlight/unhighlight buttons
        if hasattr(self, 'toggle_highlight_button'):
            self.toggle_highlight_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            layout.addWidget(self.toggle_highlight_button)

        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(container)

        return toolbar
        
    def setup_signals(self):
        """Set up signal connections."""
        # Connect annotation signals
        self.annotation_window.annotationCreated.connect(self.update_annotation_count)
        self.annotation_window.annotationDeleted.connect(self.update_annotation_count)
        # Update raster table and counts when annotation labels change
        self.annotation_window.annotationLabelChanged.connect(self.on_annotation_label_changed)
        self.annotation_window.annotationsLabelsChanged.connect(self.on_annotations_labels_changed)
        
        # Connect our own signals
        self.imageLoaded.connect(self.on_image_loaded)
        self.filterChanged.connect(self.update_image_count_label)

    def on_annotation_label_changed(self, ann_id, new_label=None):
        """Handler called when a single annotation's label changes.

        Looks up the annotation's image path and refreshes annotation info
        for that raster so the table and tooltips update immediately.
        """
        try:
            # ann_id may be None or invalid; guard defensively
            if not ann_id:
                return
            annotation = self.annotation_window.annotations_dict.get(ann_id)
            if annotation and getattr(annotation, 'image_path', None):
                self.update_image_annotations(annotation.image_path)
        except Exception:
            pass

    def on_annotations_labels_changed(self, changes):
        """Handler for multiple label changes.

        `changes` is expected to be a list of tuples: (annotation_id, old_label, new_label)
        We aggregate affected image paths and update them once each.

        Optimized: for large batches, avoid O(N) dict lookups to find affected image paths.
        Instead use the annotation_window's current_image_path directly (which covers the
        common case) and fall back to full iteration only for small batches.
        """
        try:
            if not changes:
                return

            _LARGE_BATCH = 2000
            images_to_update = set()

            if len(changes) > _LARGE_BATCH:
                # For large batches the changed annotations almost always all belong to
                # the current image (user selected all on canvas then relabeled).
                # Skip the O(N) dict-lookup loop and just update the current image.
                current_path = getattr(self.annotation_window, 'current_image_path', None)
                if current_path:
                    images_to_update.add(current_path)
                # Also check the first annotation in the list to catch cross-image cases
                # without iterating the whole list.
                try:
                    ann_id = changes[0][0]
                    annotation = self.annotation_window.annotations_dict.get(ann_id)
                    if annotation and getattr(annotation, 'image_path', None):
                        images_to_update.add(annotation.image_path)
                except Exception:
                    pass
            else:
                annotations_dict = self.annotation_window.annotations_dict
                for item in changes:
                    try:
                        ann_id = item[0]
                        annotation = annotations_dict.get(ann_id)
                        if annotation and getattr(annotation, 'image_path', None):
                            images_to_update.add(annotation.image_path)
                    except Exception:
                        continue

            # update_counts=False: skip per-image update_annotation_count call;
            # call it once below instead of once per image.
            for image_path in images_to_update:
                self.update_image_annotations(image_path, update_counts=False)

            # One consolidated annotation count refresh
            try:
                self.main_window.label_window.update_annotation_count()
            except Exception:
                pass
        except Exception:
            pass
        
    def schedule_filter(self):
        """Schedule filtering after a short delay to avoid excessive updates."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms delay
        
    # Filter toggling removed; filters are always visible now.
        
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
        # If the table view hasn't been created yet, skip custom handling
        if not hasattr(self, 'tableView'):
            return super().eventFilter(source, event)

        # Handle wheel events on the table view to customize scrolling
        if source is self.tableView and event.type() == QEvent.Wheel:
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
            if event.type() == QEvent.Enter:
                # Mouse entered the table viewport
                pass
            elif event.type() == QEvent.Leave:
                # Mouse left the table viewport
                self.hide_image_preview()
                self.hover_row = -1
            elif event.type() == QEvent.MouseMove:
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

        # If BatchInferenceDialog is open, push updated highlighted paths so
        # it can adjust video-related controls dynamically.
        try:
            if hasattr(self.main_window, 'batch_inference_dialog') and \
               self.main_window.batch_inference_dialog and \
               self.main_window.batch_inference_dialog.isVisible():
                highlighted = self.table_model.get_highlighted_paths()
                # Apply same VideoRaster filtering as open_batch_inference_dialog:
                # if mix of Rasters+VideoRasters, strip the VideoRasters
                video_hl = [p for p in highlighted
                            if (lambda r: r is not None and
                                getattr(r, 'raster_type', '') == 'VideoRaster')(
                                    self.raster_manager.get_raster(p))]
                raster_hl = [p for p in highlighted if p not in video_hl]
                if video_hl and raster_hl:
                    highlighted = raster_hl
                self.main_window.batch_inference_dialog.update_highlighted_images(highlighted)
        except Exception:
            # Swallow any errors coming from cross-widget signaling
            pass
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
        """Handler for when a raster is updated (scale, annotations, etc.)."""
        self.update_current_image_index_label()
    
    def on_z_channel_updated(self, path):
        """Handler for when a raster's z-channel is updated."""
        # If this raster is currently being displayed, refresh the z-channel visualization
        if path == self.annotation_window.current_image_path:
            self.annotation_window.refresh_z_channel_visualization()
            
            # Force status bar Z-value refresh at current mouse position
            # This ensures changes to z_nodata are immediately reflected
            if hasattr(self.annotation_window, 'update_z_value_at_mouse_position'):
                raster = self.raster_manager.get_raster(path)
                self.annotation_window.update_z_value_at_mouse_position(raster)
        
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

        # Enable/disable the toggle highlight button based on whether there are filtered rows
        try:
            if hasattr(self, 'toggle_highlight_button'):
                self.toggle_highlight_button.setEnabled(bool(filtered_paths))
        except Exception:
            pass
        
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
            
    def update_image_annotations(self, image_path, update_counts=True):
        """
        Update annotation information for a specific image.
        
        Args:
            image_path (str): Path to the image
            update_counts (bool): Whether to update the annotation counts in the label window
        """
        # For video virtual frame paths (e.g. video.mp4::frame_5), resolve to the
        # video path and aggregate annotations across all frames so the VideoRaster
        # row in the table shows the correct total count.
        if '::frame_' in image_path:
            video_path = image_path.rsplit('::frame_', 1)[0]
            prefix = video_path + '::frame_'
            all_annotations = []
            for key, anns in self.annotation_window.image_annotations_dict.items():
                if key.startswith(prefix):
                    all_annotations.extend(anns)
            self.raster_manager.update_annotation_info(video_path, all_annotations)
            self._add_video_mask_counts(video_path, prefix)
        else:
            # Check if this is a bare VideoRaster path (all frame annotations are stored
            # under virtual ::frame_ keys, so get_image_annotations() would return [] for
            # the bare path, which would reset the count to 0).
            raster = self.raster_manager.get_raster(image_path)
            if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                prefix = image_path + '::frame_'
                all_annotations = []
                for key, anns in self.annotation_window.image_annotations_dict.items():
                    if key.startswith(prefix):
                        all_annotations.extend(anns)
                self.raster_manager.update_annotation_info(image_path, all_annotations)
                self._add_video_mask_counts(image_path, prefix)
            else:
                annotations = self.annotation_window.get_image_annotations(image_path)
                self.raster_manager.update_annotation_info(image_path, annotations)
        
        if update_counts:
            self.main_window.label_window.update_annotation_count()
        
    def _add_video_mask_counts(self, video_path: str, prefix: str):
        """Add per-frame mask-annotation counts to a VideoRaster's annotation_count.

        update_annotation_info only knows about vector annotations and the shared
        raster-level mask.  Per-frame semantic overlays are stored in
        annotation_window.batch_results_cache keyed by virtual frame paths.
        Each frame that has a non-empty mask overlay counts as +1.
        """
        try:
            import numpy as np
            raster = self.raster_manager.get_raster(video_path)
            if raster is None:
                return
            cache = getattr(self.annotation_window, 'batch_results_cache', None) or {}
            mask_frame_count = 0
            for key, cached in cache.items():
                if not (isinstance(key, str) and key.startswith(prefix) and cached):
                    continue
                mask_arr = cached.get('mask_arr')
                if mask_arr is not None:
                    try:
                        if np.any(mask_arr):
                            mask_frame_count += 1
                        continue
                    except Exception:
                        pass
                if cached.get('mask_qimage') is not None:
                    mask_frame_count += 1
            if mask_frame_count > 0:
                # update_annotation_info already counted the shared raster mask as 1
                # if has_mask_content is True.  Replace that with the per-frame count
                # so the number reflects actual annotated frames.
                shared_mask_counted = 1 if raster.has_mask_content else 0
                raster.annotation_count = (
                    raster.annotation_count - shared_mask_counted + mask_frame_count
                )
                raster.has_annotations = True
        except Exception:
            pass

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
            
    def filter_images(self, use_threading: bool = True):
        """Filter images based on current criteria."""
        # Get filter criteria
        search_text = self.search_bar_images.currentText()
        search_label = self.search_bar_labels.currentText()
        
        # --- Get values from the new CheckableComboBox ---
        checked_filters = self.filter_combo.get_checked_items()
        
        highlighted_only = "Highlighted" in checked_filters
        raster_type_map = {
            "Image": "ImageRaster",
            "Ortho": "OrthoRaster",
            "Video": "VideoRaster",
        }
        allowed_raster_types = {
            raster_type_map[item] for item in checked_filters if item in raster_type_map
        }
        if not allowed_raster_types:
            allowed_raster_types = None

        require_z_channel = "Has Z-Channel" in checked_filters
        require_transform = "has Transform" in checked_filters
        has_predictions = "Has Predictions" in checked_filters
        has_annotations = "Has Annotations" in checked_filters
        no_annotations = "No Annotations" in checked_filters
        # --- End new logic ---
        
        
        # Get highlighted paths if needed
        highlighted_paths = self.table_model.get_highlighted_paths() if highlighted_only else None
        
        # Run the filter (skip threading for initial import/load for speed)
        self.image_filter.filter_images(
            search_text=search_text,
            search_label=search_label,
            require_annotations=has_annotations,
            require_no_annotations=no_annotations,
            require_predictions=has_predictions,
            allowed_raster_types=allowed_raster_types,
            require_z_channel=require_z_channel,
            require_transform=require_transform,
            selected_paths=highlighted_paths,
            use_threading=use_threading
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
        
        # Update Z-Deploy dialog with current highlighted images
        if hasattr(self.main_window, 'z_deploy_model_dialog') and self.main_window.z_deploy_model_dialog:
            self.main_window.z_deploy_model_dialog.update_highlighted_images(highlighted_paths)
        
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
        
    def highlight_all_rows(self, select_rows: bool = True):
        """Highlight all rows in the filtered view and optionally select them."""
        # Batch highlight all filtered paths for better performance
        self.table_model.set_highlighted_paths(self.table_model.filtered_paths)
        
        # Update the last highlighted row
        if self.table_model.filtered_paths:
            self.last_highlighted_row = self.table_model.get_row_for_path(self.table_model.filtered_paths[-1])
            
        # Update the highlighted count label
        self.update_highlighted_count_label()

        if select_rows and self.table_model.filtered_paths:
            self.tableView.setFocus(Qt.OtherFocusReason)
            self.tableView.selectAll()
        
    def unhighlight_all_rows(self):
        """Clear all highlights."""
        selection_model = self.tableView.selectionModel()
        if selection_model:
            selection_model.clearSelection()

        self.table_model.clear_highlights()
        self.last_highlighted_row = -1
        
        if self.selected_image_path:
            self.table_model.set_selected_path(self.selected_image_path)
            self.select_row_for_path(self.selected_image_path)

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
        target_path = path_at_cursor if path_at_cursor is not None else (highlighted_paths[0] if highlighted_paths else None)
        raster_under_cursor = self.raster_manager.get_raster(target_path) if target_path is not None else None
        if raster_under_cursor:
            is_checked = raster_under_cursor.checkbox_state
            if is_checked:
                action_text = f"Uncheck {count} Highlighted Raster{'s' if count > 1 else ''}"
            else:
                action_text = f"Check {count} Highlighted Raster{'s' if count > 1 else ''}"
            toggle_check_action = context_menu.addAction(action_text)
            toggle_check_action.triggered.connect(lambda checked=False, state=not is_checked: self.on_toggle(state))

        context_menu.addSeparator()
        
        # Add batch inference action
        batch_inference_action = context_menu.addAction(
            f"Batch Inference ({count} Highlighted Raster{'s' if count > 1 else ''})"
        )
        batch_inference_action.triggered.connect(
            lambda: self.open_batch_inference_dialog(highlighted_paths)
        )
        
        context_menu.addSeparator()

        highlighted_rasters = [self.raster_manager.get_raster(path) for path in highlighted_paths]
        has_ortho = any(getattr(raster, 'raster_type', '') == 'OrthoRaster' for raster in highlighted_rasters if raster is not None)
        _single_ortho = count == 1 and has_ortho

        # Single VideoRaster: offer frame extraction (and annotation re-creation).
        _single_video = (count == 1 and highlighted_rasters[0] is not None and
                         getattr(highlighted_rasters[0], 'raster_type', '') == 'VideoRaster')
        if _single_video:
            extract_frames_action = context_menu.addAction("Extract Frames...")
            extract_frames_action.triggered.connect(
                lambda checked=False, p=highlighted_paths[0]: self._open_extract_frames_dialog(p)
            )
            context_menu.addSeparator()

        if _single_ortho:
            transforms_menu = context_menu.addMenu("Transforms...")
            _ortho_path = highlighted_paths[0]
            set_proj_action = transforms_menu.addAction("Set Projection Matrix")
            set_proj_action.triggered.connect(
                lambda checked=False, p=_ortho_path: self._set_ortho_projection_matrix(p)
            )

            set_chunk_action = transforms_menu.addAction("Set Chunk Tranform")
            set_chunk_action.triggered.connect(
                lambda checked=False, p=_ortho_path: self._set_ortho_chunk_transform(p)
            )
        elif not has_ortho:
            cameras_menu = context_menu.addMenu("Cameras...")
            # Normal perspective images: import / remove camera calibration
            import_cameras_action = cameras_menu.addAction(
                f"Import Cameras for {count} Highlighted Raster{'s' if count > 1 else ''}"
            )
            import_cameras_action.triggered.connect(
                lambda: self._open_import_cameras_for_highlighted(highlighted_paths)
            )

            cameras_menu.addSeparator()

            remove_cameras_action = cameras_menu.addAction(
                f"Remove Cameras from {count} Highlighted Raster{'s' if count > 1 else ''}"
            )
            remove_cameras_action.triggered.connect(lambda: self.remove_cameras_highlighted_images())

        # Create Z-Channel sub-menu
        z_channel_menu = context_menu.addMenu("Z-Channel...")

        # Add import z-channel action
        import_z_channel_action = z_channel_menu.addAction(
            f"Import for {count} Highlighted Raster{'s' if count > 1 else ''}"
        )
        import_z_channel_action.triggered.connect(
            lambda: self.import_z_channel_highlighted_images()
        )

        # Add export z-channel action
        export_z_channel_action = z_channel_menu.addAction(
            f"Export for {count} Highlighted Raster{'s' if count > 1 else ''}"
        )
        export_z_channel_action.triggered.connect(
            lambda: self.export_z_channel_highlighted_images()
        )

        z_channel_menu.addSeparator()

        # Add remove z-channel action
        remove_z_channel_action = z_channel_menu.addAction(
            f"Remove from {count} Highlighted Raster{'s' if count > 1 else ''}"
        )
        remove_z_channel_action.triggered.connect(
            lambda: self.remove_z_channel_highlighted_images()
        )

        # Create Features sub-menu (Tier-1 dense feature maps)
        feature_menu = context_menu.addMenu("Features...")
        remove_feature_action = feature_menu.addAction(
            f"Remove from {count} Highlighted Raster{'s' if count > 1 else ''}"
        )
        remove_feature_action.triggered.connect(
            lambda: self.remove_feature_map_highlighted_images()
        )

        context_menu.addSeparator()

        # Add delete actions
        delete_images_action = context_menu.addAction(f"Delete {count} Highlighted Raster{'s' if count > 1 else ''}")
        delete_images_action.triggered.connect(lambda: self.delete_highlighted_images())
        delete_annotations_action = context_menu.addAction(
            f"Delete Annotations for {count} Highlighted Raster{'s' if count > 1 else ''}"
        )
        delete_annotations_action.triggered.connect(
            lambda: self.delete_highlighted_images_annotations()
        )

        context_menu.exec_(self.tableView.viewport().mapToGlobal(position))

    def _open_extract_frames_dialog(self, video_path: str):
        """Open the Extract Frames dialog pre-loaded with a VideoRaster.

        The dialog runs in video-raster mode: the video file is locked, the
        Keyframes selection mode is available, and the user may opt to re-create
        the video frames' annotations on the imported still images.
        """
        raster = self.raster_manager.get_raster(video_path)
        if raster is None or getattr(raster, 'raster_type', '') != 'VideoRaster':
            return
        from coralnet_toolbox.IO.QtImportFrames import ImportFrames
        dialog = ImportFrames(self.main_window, parent=self, video_raster=raster)
        dialog.exec_()

    def _set_ortho_projection_matrix(self, path: str):
        """
        Show a 4×4 matrix dialog so the user can supply the Metashape orthomosaic
        projection matrix for a local-coordinate-system project.

        The matrix is stored on the OrthoRaster and, if an OrthoCamera has already
        been built by MVATManager, its inverse is updated immediately so that
        subsequent mouse-move events use the new projection.
        """
        from PyQt5.QtWidgets import QDialog
        from coralnet_toolbox.Common.QtTransformInput import TransformInputDialog

        raster = self.raster_manager.get_raster(path)
        if raster is None or getattr(raster, 'raster_type', '') != 'OrthoRaster':
            return

        dialog = TransformInputDialog(
            self,
            title="Ortho Projection Matrix",
            prompt_text=(
                "Enter the 4x4 orthomosaic projection matrix:\n"
                "(Leave as identity for local coordinate systems without a separate projection)"
            ),
        )

        existing = getattr(raster, 'ortho_projection_matrix', None)
        if existing is not None:
            dialog.set_matrix(existing)

        if dialog.exec_() == QDialog.Accepted:
            mat = dialog.get_matrix()
            raster.ortho_projection_matrix = mat
            # Propagate to live OrthoCamera if available
            mvat_manager = getattr(self.main_window, 'mvat_manager', None)
            if mvat_manager is not None and mvat_manager.ortho_camera is not None:
                if mvat_manager.ortho_camera.image_path == path:
                    mvat_manager.ortho_camera.update_ortho_projection_matrix(mat)
            try:
                self.raster_manager.rasterUpdated.emit(path)
            except Exception:
                pass

    def _set_ortho_chunk_transform(self, path: str):
        """Show a 4x4 matrix dialog for the OrthoRaster chunk transform."""
        from PyQt5.QtWidgets import QDialog
        from coralnet_toolbox.Common.QtTransformInput import TransformInputDialog

        raster = self.raster_manager.get_raster(path)
        if raster is None or getattr(raster, 'raster_type', '') != 'OrthoRaster':
            return

        dialog = TransformInputDialog(
            self,
            title="Metashape Chunk Transform",
            prompt_text=(
                "Enter the 4x4 chunk transform matrix:\n"
                "(Leave as identity for local coordinate systems without a chunk transform)"
            ),
        )

        existing = getattr(raster, 'chunk_transform_matrix', None)
        if existing is not None:
            dialog.set_matrix(existing)

        if dialog.exec_() == QDialog.Accepted:
            mat = dialog.get_matrix()
            raster.chunk_transform_matrix = mat

            mvat_manager = getattr(self.main_window, 'mvat_manager', None)
            if mvat_manager is not None:
                mvat_manager._chunk_transform = mat
                if mvat_manager.ortho_camera is not None and mvat_manager.ortho_camera.image_path == path:
                    mvat_manager.ortho_camera.update_chunk_transform(mat)

            try:
                self.raster_manager.rasterUpdated.emit(path)
            except Exception:
                pass

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

        # Classify highlighted paths into VideoRasters vs regular Rasters
        video_paths = []
        raster_paths = []
        for path in highlighted_image_paths:
            raster = self.raster_manager.get_raster(path)
            if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                video_paths.append(path)
            else:
                raster_paths.append(path)

        if video_paths and raster_paths:
            # Mixed selection: silently drop VideoRasters and proceed with Rasters only
            highlighted_image_paths = raster_paths
        elif video_paths and not raster_paths:
            # Only VideoRasters selected
            if len(video_paths) > 1:
                QMessageBox.warning(
                    self,
                    "Multiple Videos Selected",
                    "Only one video can be used with batch inference at a time."
                )
                return
            # Single VideoRaster — allow normally (highlighted_image_paths unchanged)
        
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
        # Show the dialog modelessly so users can change highlighted rows while it is open
        try:
            batch_dialog.setModal(False)
            batch_dialog.setWindowModality(Qt.NonModal)
            # Ensure the dialog stays on top of the main window while modeless
            batch_dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        except Exception:
            pass

        # Refresh the dialog display to reflect the initial highlighted selection
        try:
            if hasattr(batch_dialog, 'inference_type_combo') and batch_dialog.inference_type_combo.currentText() == 'Tiled':
                batch_dialog.update_status_label_for_tiled()
            else:
                batch_dialog.update_status_label()
        except Exception:
            pass

        batch_dialog.show()
        batch_dialog.raise_()
        batch_dialog.activateWindow()

    def _open_import_cameras_for_highlighted(self, highlighted_paths: list):
        """
        Open the Import Cameras dialog and set it to operate on the highlighted images.
        """
        if not highlighted_paths:
            QMessageBox.warning(self, "No Images Selected", "Please highlight one or more images before importing cameras.")
            return

        # Use the main window's ImportCameras dialog instance if available
        try:
            dialog = self.main_window.import_cameras_dialog
        except Exception:
            dialog = None

        if dialog is None:
            # Fallback - instantiate a temporary dialog
            dialog = ImportCameras(self.main_window)

        # Provide the highlighted images to the dialog so it restricts matching
        dialog.highlighted_images = highlighted_paths
        dialog.exec_()
    
    def import_z_channel_highlighted_images(self):
        """Open file dialog and ZImportDialog to import z-channel files for highlighted images."""
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
        
        # Create and show ZImportDialog with highlighted image paths and selected z-files
        # Sort both lists for consistent ordering
        image_paths = sorted(highlighted_paths)
        z_channel_files = sorted(z_files)
        
        # Create the pairing widget and keep a reference to prevent garbage collection
        self.pairing_widget = ZImportDialog(image_paths, z_channel_files)
        
        # Connect the mapping_confirmed signal to handle the confirmed mapping
        self.pairing_widget.mapping_confirmed.connect(self.on_z_channel_mapping_confirmed)
        
        # Show the widget
        self.pairing_widget.exec_()
    
    def on_z_channel_mapping_confirmed(self, mapping):
        """Handle confirmed z-channel mapping from ZImportDialog.
        
        Args:
            mapping (dict): {image_path: {"z_path": z_channel_path, "units": unit_str, "z_data_type": type_str}}
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
                # Extract z_path, units, and z_data_type from mapping
                if isinstance(z_info, dict):
                    z_channel_path = z_info.get("z_path")
                    z_unit = z_info.get("units")
                    z_data_type = z_info.get("z_data_type")
                else:
                    # Fallback for old-style mappings (just paths)
                    z_channel_path = z_info
                    z_unit = None
                    z_data_type = None
                
                raster = self.raster_manager.get_raster(image_path)
                if raster:
                    try:
                        # Load z-channel from file with units and data type
                        success = raster.load_z_channel_from_file(
                            z_channel_path, 
                            z_unit=z_unit,
                            z_data_type=z_data_type
                        )
                        if success:
                            # Data type is now set during load_z_channel_from_file
                            # No automatic transformations - user specified the data type,
                            # so we trust the values are already in the correct format.
                            # Transformations and calibrations are done in ScaleTool.
                                    
                            successful_count += 1
                            # Note: load_z_channel_from_file -> add_z_channel automatically emits zChannelUpdated
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
            f"Are you sure you want to remove the z-channel from {count} raster{plural}?",
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

    def remove_feature_map_highlighted_images(self):
        """Remove dense feature maps from the highlighted images."""
        highlighted_paths = self.table_model.get_highlighted_paths()

        if not highlighted_paths:
            return

        # Only act on rasters that actually have a feature map.
        targets = []
        for path in highlighted_paths:
            raster = self.raster_manager.get_raster(path)
            if raster is not None and getattr(raster, "has_feature_map", lambda: False)():
                targets.append((path, raster))

        if not targets:
            QMessageBox.information(
                self,
                "No Feature Maps",
                "None of the highlighted rasters have a feature map to remove.",
            )
            return

        count = len(targets)
        plural = 's' if count > 1 else ''
        reply = QMessageBox.question(
            self,
            "Confirm Feature Map Removal",
            f"Are you sure you want to remove the feature map from {count} raster{plural}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Removing Feature Maps")
        progress_bar.show()
        progress_bar.start_progress(count)

        try:
            for path, raster in targets:
                try:
                    raster.clear_feature_map()
                    self.raster_manager.rasterUpdated.emit(path)
                except Exception as e:
                    print(f"Failed to remove feature map for {path}: {e}")
                progress_bar.update_progress()

            QMessageBox.information(
                self,
                "Feature Maps Removed",
                f"Feature map removed from {count} raster{plural}."
            )
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

    def remove_cameras_highlighted_images(self):
        """Remove camera intrinsics/extrinsics from the highlighted images."""
        highlighted_paths = self.table_model.get_highlighted_paths()
        if not highlighted_paths:
            return

        count = len(highlighted_paths)
        plural = 's' if count > 1 else ''
        reply = QMessageBox.question(
            self,
            "Confirm Camera Parameter Removal",
            f"Are you sure you want to remove cameras from {count} raster{plural}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Removing Cameras")
        progress_bar.show()
        progress_bar.start_progress(len(highlighted_paths))

        try:
            removed_count = 0
            for path in highlighted_paths:
                raster = self.raster_manager.get_raster(path)
                if raster:
                    raster.remove_intrinsics()
                    raster.remove_extrinsics()
                    # Notify raster manager/UI of update
                    self.raster_manager.rasterUpdated.emit(path)
                    removed_count += 1
                progress_bar.update_progress()

            # If current image is affected, refresh viewer tools
            if self.selected_image_path in highlighted_paths:
                self.annotation_window.update_scene()

            QMessageBox.information(
                self,
                "Cameras Removed",
                f"Cameras removed from {removed_count} image{plural}."
            )
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()
    
    def export_z_channel_highlighted_images(self):
        """Export z-channels from the highlighted images."""
        # Get all highlighted paths
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
        
        # Get rasters for highlighted images
        highlighted_rasters = []
        for path in highlighted_paths:
            raster = self.raster_manager.get_raster(path)
            if raster:
                highlighted_rasters.append(raster)
        
        if not highlighted_rasters:
            return
        
        # Create and show ZExportDialog
        self.export_dialog = ZExportDialog(highlighted_rasters, parent=self)
        self.export_dialog.exec_()
        
    def delete_highlighted_images(self):
        """Delete the highlighted images."""
        # Get all highlighted paths, the same way we do in filter_images
        highlighted_paths = self.table_model.get_highlighted_paths()
        
        if not highlighted_paths:
            return
            
        # Confirm deletion
        plural = 's' if len(highlighted_paths) > 1 else ''
        reply = QMessageBox.question(
            self,
            "Confirm Multiple Image Deletions",
            f"Are you sure you want to delete {len(highlighted_paths)} raster{plural}?\n"
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
        plural = 's' if len(highlighted_paths) > 1 else ''
        reply = QMessageBox.question(
            self,
            "Confirm Multiple Annotation Deletions",
            f"Are you sure you want to delete annotations for {len(highlighted_paths)} raster{plural}?",
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

                # Refresh video tick marks in case any deleted path was a video
                try:
                    self.annotation_window._update_video_annotation_marks()
                except Exception:
                    pass

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
                    self.raster_manager.remove_raster(path, collect_garbage=False)
                    
                    # Update progress
                    progress_bar.update_progress()

                gc.collect()
                    
                # Update UI
                if next_image:
                    self.load_image_by_path(next_image)
                elif not self.raster_manager.image_paths:
                    self.selected_image_path = None
                    self.annotation_window.clear_scene()

                self.update_search_bars()
                self.update_image_count_label(len(self.table_model.filtered_paths))
                self.update_current_image_index_label()
                self.update_highlighted_count_label()
                    
            finally:
                # Restore signals
                self.raster_manager.rasterAdded.connect(self.on_raster_added)
                self.raster_manager.rasterRemoved.connect(self.on_raster_removed)
                
                # Close progress bar
                progress_bar.stop_progress()
                progress_bar.close()


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
        self.setStyleSheet(
            app_theme.scale_qss(
                """
            QFrame {
                background-color: %s;
                border: 1px solid %s;
                border-radius: 6px;
                color: %s;
            }
        """ % (app_theme.SURFACE_COLOR.name(), app_theme.SURFACE_BORDER_COLOR.name(), app_theme.TEXT_PRIMARY_COLOR.name()))
        )
        
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
