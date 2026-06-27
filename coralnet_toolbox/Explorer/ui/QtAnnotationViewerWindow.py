# coralnet_toolbox/Explorer/QtAnnotationViewerWindow.py
"""
Standalone Annotation Gallery Window.

This module provides a fully self-contained gallery viewer for annotations
that integrates directly with MainWindow as a dockable widget. It combines
the gallery display functionality with built-in filtering capabilities.
"""

import os

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QEvent, QSignalBlocker
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QComboBox,
    QLabel, QPushButton, QApplication, QListView,
    QHBoxLayout
)

from coralnet_toolbox import theme as app_theme

from coralnet_toolbox.Explorer.core.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.ui.QtExplorerWidgets import MultiSelectCombo
from coralnet_toolbox.Explorer.models.AnnotationListModel import AnnotationListModel 
from coralnet_toolbox.Explorer.models.AnnotationListModel import AnnotationItemDelegate
from coralnet_toolbox.Explorer.workers.QtCroppingWorker import CroppingWorker

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationViewerWindow(QWidget):
    """
    Standalone gallery window for displaying and filtering annotation crops.
    
    This widget is designed to be wrapped by DockWrapper and integrated into
    MainWindow as a persistent dock. It owns its own filter controls and
    data model, communicating with other components via signals.
    
    Signals:
        selection_changed (list): Emitted when annotations are selected/deselected.
        annotations_filtered (list): Emitted when filter is applied with list of annotation IDs.
        preview_changed (list): Emitted when preview labels are applied.
    """
    
    selection_changed = pyqtSignal(list)  # List of annotation IDs
    annotations_filtered = pyqtSignal(list)  # List of annotation IDs after filtering
    preview_changed = pyqtSignal(list)  # List of annotation IDs with preview changes
    reset_view_requested = pyqtSignal()  # Request to reset view state
    cleared = pyqtSignal()
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the AnnotationViewerWindow.
        
        Args:
            main_window: Reference to the MainWindow instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        
        # Animation manager reference
        # Cropping state
        self._cropping_in_progress = False
        self._cropping_worker = None
        
        # Data model
        self.data_item_cache = {}  # annotation_id -> AnnotationDataItem
        self.all_data_items = []  # Currently filtered data items
        
        # Selection state (real selection via QListView)
        self.last_selected_item_id = None
        self._syncing_selection = False  # Flag to prevent selection sync loops
        
        # Isolation state
        self.isolated_mode = False
        self.isolated_ids = None
        
        # (legacy rubber-band removed; QListView provides native selection)
        
        # Sorting state
        self.active_ordered_ids = []
        self._group_headers = []
        
        # Display options
        self.current_widget_size = app_theme.scale_int(96)
        # Widget size limits and step for Ctrl+Scroll resizing
        self._widget_size_min = app_theme.scale_int(32)
        self._widget_size_max = app_theme.scale_int(256)
        self._widget_size_step = max(1, app_theme.scale_int(8))
        
        # Selection blocking (for external wizards)
        self.selection_blocked = False
        # Filter applied flag: when True the gallery is allowed to populate widgets
        # (the user must explicitly press "Apply Filter" to set this).
        self._filter_applied = False

        # Label-change coalescing: collect changed IDs during the current event-loop
        # tick, then flush once via _flush_label_change_update().
        self._pending_label_changes: set = set()
        self._label_change_timer = QTimer(self)
        self._label_change_timer.setSingleShot(True)
        self._label_change_timer.setInterval(0)  # fire on next event-loop iteration
        self._label_change_timer.timeout.connect(self._flush_label_change_update)

        # annotationUpdated coalescing: same pattern as label changes.
        # During batch operations (e.g. classification of 500 annotations)
        # annotationUpdated fires once per annotation.  Without coalescing,
        # _on_annotation_updated → on_annotation_modified queues 500 individual
        # refresh_annotations() calls, each iterating the full annotation dict
        # (O(N²) work).  We collect dirty IDs and flush a single refresh.
        self._pending_annotation_updates: set = set()
        self._annotation_update_timer = QTimer(self)
        self._annotation_update_timer.setSingleShot(True)
        self._annotation_update_timer.setInterval(0)
        self._annotation_update_timer.timeout.connect(self._flush_annotation_updates)

        # Virtualization disabled: using model/view

        # Build the UI
        self._setup_ui()
        
    def showEvent(self, event):
        """Handle show event to refresh filters when dock becomes visible."""
        super().showEvent(event)
        # Refresh filter options when window becomes visible
        QTimer.singleShot(0, self.refresh_filter_options)
        
        # Force a layout recalculation right after the Qt layout engine has settled
        QTimer.singleShot(100, self._recalculate_layout)
        

    def create_top_toolbar(self) -> QToolBar:
        """
        Create the top toolbar with viewing controls.
        
        Returns:
            QToolBar: Configured toolbar for dock integration.
        """
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        
        # Isolate Selection button
        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setToolTip("Show only selected annotations (double-click to exit)")
        self.isolate_button.clicked.connect(self._isolate_selection)
        self.isolate_button.setEnabled(False)
        # toolbar.addWidget(self.isolate_button)
        
        toolbar.addSeparator()
        
        # Sort controls
        sort_label = QLabel("Sort:")
        toolbar.addWidget(sort_label)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Label", "Image", "Confidence", "Area", "Cluster"])
        self.sort_combo.insertSeparator(self.sort_combo.findText("Cluster"))
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.sort_combo.setMinimumWidth(100)
        toolbar.addWidget(self.sort_combo)

        # "Cluster" is disabled until cluster data arrives from the EmbeddingViewer.
        # The separator keeps it visually grouped away from the standard sorts.
        self._set_cluster_sort_item_enabled(False)

        toolbar.addSeparator()

        # Image filter - multi-select combo
        image_label = QLabel("Images:")
        toolbar.addWidget(image_label)

        self.image_filter_combo = MultiSelectCombo()
        self.image_filter_combo.setFixedWidth(80)
        self.image_filter_combo.setToolTip("Filter by image (multi-select)")
        self.image_filter_combo.selection_changed.connect(lambda v: None)
        toolbar.addWidget(self.image_filter_combo)
        
        # Label filter - searchable combo box
        label_label = QLabel("Labels:")
        toolbar.addWidget(label_label)

        self.label_filter_combo = MultiSelectCombo()
        self.label_filter_combo.setFixedWidth(80)
        self.label_filter_combo.setToolTip("Filter by label (multi-select)")
        toolbar.addWidget(self.label_filter_combo)

        # Type filter - searchable combo box
        type_label = QLabel("Annotations:")
        toolbar.addWidget(type_label)

        self.type_filter_combo = MultiSelectCombo()
        self.type_filter_combo.setFixedWidth(80)
        # Populate type filter with fixed options
        type_opts = [
            ("Patch", "PatchAnnotation"),
            ("Rectangle", "RectangleAnnotation"),
            ("Polygon", "PolygonAnnotation"),
            ("MultiPolygon", "MultiPolygonAnnotation"),
        ]
        self.type_filter_combo.set_options(type_opts)
        toolbar.addWidget(self.type_filter_combo)
        
        toolbar.addSeparator()

        # Initialize filter options now that the controls exist
        self._populate_filter_combos()
        
        return toolbar
    
    def create_bottom_toolbar(self) -> QToolBar:
        """
        Create the bottom toolbar with filter controls using searchable combo boxes.
        
        Returns:
            QToolBar: Configured filter toolbar for dock integration.
        """
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        # Clear button - clears gallery and notifies other viewers
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear gallery and reset view")
        self.clear_button.clicked.connect(self.clear)
        toolbar.addWidget(self.clear_button)
        
        # Apply button
        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.setToolTip("Apply current filter settings")
        self.apply_filter_button.clicked.connect(self.apply_filters)
        toolbar.addWidget(self.apply_filter_button)
        
        return toolbar
    
    def _populate_filter_combos(self):
        """Populate all filter combo boxes with current options."""
        self._populate_image_filter()
        self._populate_label_filter()
    
    def _populate_image_filter(self):
        """Populate image filter combo with current images."""
        opts = []
        image_window = getattr(self.main_window, 'image_window', None)
        if image_window:
            raster_manager = getattr(image_window, 'raster_manager', None)
            if raster_manager:
                for path in raster_manager.image_paths:
                    image_name = os.path.basename(path)
                    opts.append((image_name, image_name))
        
        try:
            # Set options (MultiSelectCombo expects list of tuples)
            self.image_filter_combo.set_options(opts)
            current_image_path = getattr(self.annotation_window, 'current_image_path', None)
            current_image_name = os.path.basename(current_image_path) if current_image_path else None
            self.image_filter_combo.set_highlighted_value(current_image_name)
        except Exception:
            pass
    
    def _populate_label_filter(self):
        """Populate label filter combo with current labels."""
        opts = []
        label_window = getattr(self.main_window, 'label_window', None)
        if label_window:
            labels = getattr(label_window, 'labels', [])
            for label in labels:
                if hasattr(label, 'short_label_code'):
                    opts.append((label.short_label_code, label.short_label_code))
        try:
            self.label_filter_combo.set_options(opts)
        except Exception:
            pass
    
    def refresh_filter_options(self):
        """Refresh filter options based on current state."""
        self._populate_filter_combos()

    def apply_filters(self):
        """Apply the current UI filters and populate the gallery.

        The user must explicitly invoke this; until then the gallery remains
        in the placeholder state and no widgets/crops are created.
        """
        try:
            self._filter_applied = True
            self.refresh_annotations()
        except Exception:
            pass
    
    @pyqtSlot(str)
    def on_image_loaded(self, image_path):
        """
        Handle when a new image is loaded in ImageWindow.
        
        Args:
            image_path: Path to the loaded image.
        """
        # If filters are not currently applied (viewer in placeholder state),
        # do not auto-populate when the image changes. If the gallery was
        # previously populated, clear it so it reverts to placeholder state.
        if not getattr(self, '_filter_applied', False):
            # Still refresh filter combos so the dropdowns reflect the new image
            self._populate_filter_combos()
            return

        # Remember current filter selection before refreshing
        current_selection = None
        if hasattr(self, 'image_filter_combo'):
            try:
                current_selection = self.image_filter_combo.selected_values()
            except Exception:
                try:
                    current_selection = self.image_filter_combo.currentData()
                except Exception:
                    current_selection = None

        # Refresh filters to include any new images
        self._populate_filter_combos()

        # If a specific image was selected previously, preserve selection if possible
        if current_selection and image_path:
            # For MultiSelectCombo we won't try to programmatically reselect by name
            # Just clear the gallery so the user can re-apply if needed
            if getattr(self, '_filter_applied', False):
                self.clear()
    
    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create list view for content
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setSelectionMode(QListView.ExtendedSelection)
        self.list_view.setSpacing(app_theme.scale_int(5))
        # Override key press events to support Ctrl+A selection
        self.list_view.keyPressEvent = self._list_view_key_press_event
        # Set background and rubber-band styling (cyan rubber band)
        self.list_view.setStyleSheet(
            "background-color: %s;"
            "QRubberBand { border: 1px solid %s; background: rgba(61,122,237,40); }"
            % (app_theme.BACKGROUND_COLOR.name(), app_theme.ACCENT_COLOR.name())
        )
        layout.addWidget(self.list_view)

        # Backwards-compatibility aliases for code paths that still reference the
        # old scroll-area/content_widget API. These point to the list view so
        # older methods won't crash while the refactor is completed.
        self.content_widget = self.list_view
        self.scroll_area = self.list_view

        # Placeholder label shown when no annotations are available
        self.placeholder_label = QLabel(
            "No annotations available\nLoad annotations or adjust the gallery filters to display results."
        )
        self.placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; "
                "background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setAutoFillBackground(True)
        self._show_placeholder()
        layout.addWidget(self.placeholder_label)

        # Install event filter for Ctrl+Wheel on list viewport
        self.list_view.viewport().installEventFilter(self)

        # Setup model and delegate
        self.list_model = AnnotationListModel(self)
        self.list_delegate = AnnotationItemDelegate(item_size=self.current_widget_size)

        self.list_view.setModel(self.list_model)
        self.list_view.setItemDelegate(self.list_delegate)
        self.list_view.selectionModel().selectionChanged.connect(self._on_list_selection_changed)

        # Sticky header overlay (hidden until needed)
        class StickyHeaderWidget(QWidget):
            def __init__(self, parent=None, height=32):
                super().__init__(parent)
                self.setFixedHeight(height)
                self.setAutoFillBackground(False)
                self._group_key = None
                self._callback = None
                self._select_callback = None
                hl = QHBoxLayout(self)
                hl.setContentsMargins(8, 0, 8, 0)
                self.title = QLabel('', self)
                self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self.chev = QLabel('', self)
                self.chev.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
                hl.addWidget(self.title)
                hl.addWidget(self.chev)

            def set_callback(self, cb):
                self._callback = cb

            def set_select_callback(self, cb):
                self._select_callback = cb

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    if event.modifiers() & Qt.ControlModifier and self._select_callback:
                        try:
                            self._select_callback(self._group_key)
                        except Exception:
                            pass
                    elif self._callback:
                        try:
                            self._callback(self._group_key)
                        except Exception:
                            pass
                return super().mousePressEvent(event)

            def set_content(self, text, bg_color, text_color, expanded, group_key):
                self._group_key = group_key
                self.title.setText(text)
                self.chev.setText('▾' if expanded else '▸')
                # apply styles
                try:
                    r, g, b = bg_color.red(), bg_color.green(), bg_color.blue()
                except Exception:
                    r, g, b = (51, 51, 51)
                self.setStyleSheet(f'background-color: rgba({r},{g},{b},220);')
                col = text_color.name() if hasattr(text_color, 'name') else '#fff'
                self.title.setStyleSheet(f'color: {col}; font-weight: bold;')
                self.chev.setStyleSheet(f'color: {col}; font-weight: bold;')

        self._sticky_header = StickyHeaderWidget(self.list_view.viewport(), height=self.list_delegate.header_height)
        self._sticky_header.hide()
        # clicking header will toggle group expansion; Ctrl+click selects all in group
        self._sticky_header.set_callback(lambda key: self._toggle_group_from_header(key))
        self._sticky_header.set_select_callback(lambda key: self._select_group_annotations(key))

        # update sticky header when view scrolls or model changes
        self.list_view.verticalScrollBar().valueChanged.connect(self._update_sticky_header)
        try:
            self.list_model.layoutChanged.connect(self._update_sticky_header)
        except Exception:
            pass
        try:
            self.list_model.modelReset.connect(self._update_sticky_header)
        except Exception:
            pass

    def refresh_scaling(self):
        """Refresh gallery sizing after a UI scale change."""
        self.current_widget_size = app_theme.scale_int(96)
        self._widget_size_min = app_theme.scale_int(32)
        self._widget_size_max = app_theme.scale_int(256)
        self._widget_size_step = max(1, app_theme.scale_int(8))
        self.list_view.setSpacing(app_theme.scale_int(5))
        self.placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; "
                "background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        if hasattr(self, 'list_delegate'):
            self.list_delegate.item_size = self.current_widget_size
        self.list_view.doItemsLayout()
        self.list_view.update()
        
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_currently_displayed_annotations(self):
        """
        Get the list of currently displayed annotation IDs.
        
        Returns:
            list: List of annotation IDs currently shown in the gallery.
        """
        if not getattr(self, '_filter_applied', False):
            return []
        return [item.annotation.id for item in self.all_data_items]
    
    def get_selected_annotation_ids(self):
        """
        Get the list of currently selected annotation IDs.

        Returns:
            list: List of selected annotation IDs.
        """
        if hasattr(self, 'list_view') and self.list_view is not None:
            sel = self.list_view.selectionModel().selectedIndexes()
            ids = []
            for idx in sel:
                data = idx.data(self.list_model.DataItemRole)
                if data and data.get('type') == 'annotation':
                    ids.append(data['item'].annotation.id)
            return ids
        return []

    def get_selected_annotation_count(self):
        """
        Get the count of currently selected annotations — fast path that avoids
        building the full ID list (used by update_annotation_count in LabelWindow).

        Returns:
            int: Number of selected annotation items (excludes header rows).
        """
        if hasattr(self, 'list_view') and self.list_view is not None:
            try:
                sel = self.list_view.selectionModel().selectedIndexes()
                # Count only rows that are annotation items (not group headers)
                return sum(
                    1 for idx in sel
                    if (idx.data(self.list_model.DataItemRole) or {}).get('type') == 'annotation'
                )
            except Exception:
                pass
        return 0
    
    def highlight_annotations(self, ids):
        """
        Highlight specific annotations in the gallery.
        
        Args:
            ids: List or set of annotation IDs to highlight.
        """
        ids_set = set(ids)
        self.render_selection_from_ids(ids_set)

    def _show_placeholder(self):
        """
        Show the placeholder label and hide the gallery scroll area.
        Safe to call even if UI not fully initialized.
        """
        try:
            if hasattr(self, 'placeholder_label'):
                self.placeholder_label.show()
            if hasattr(self, 'list_view') and self.list_view is not None:
                self.list_view.hide()
        except Exception:
            pass
            
    def _show_annotation_gallery(self):
        """
        Show the gallery scroll area and hide the placeholder label.
        """
        try:
            # If the user has not applied the filter, remain in placeholder state
            if not getattr(self, '_filter_applied', False):
                self._show_placeholder()
                return
            if hasattr(self, 'placeholder_label'):
                self.placeholder_label.hide()
            if hasattr(self, 'list_view') and self.list_view is not None:
                self.list_view.show()
        except Exception:
            pass
        
    def refresh_annotations(self):
        """
        Refresh the gallery based on current filter settings.
        
        This method:
        1. Gets annotations from AnnotationWindow that match the filter
        2. Creates/updates data items
        3. Rebuilds the gallery layout
        4. Emits annotations_filtered signal
        """
        # Only refresh (and create widgets) when the user has explicitly
        # applied the filter via the Apply Filter button.
        if not getattr(self, '_filter_applied', False):
            return

        # Get filter selections
        selected_images = self._get_selected_images()
        selected_types = self._get_selected_types()
        selected_labels = self._get_selected_labels()
        
        # Get annotations from AnnotationWindow
        if not hasattr(self.annotation_window, 'annotations_dict'):
            self.all_data_items = []
            self._update_annotations_display([])
            return
            
        # Filter annotations
        filtered_annotations = []
        for ann in self.annotation_window.annotations_dict.values():
            image_name = os.path.basename(ann.image_path)
            type_name = type(ann).__name__
            label_code = ann.label.short_label_code
            
            # Check image filter
            if selected_images and image_name not in selected_images:
                continue
            
            # Check type filter
            if selected_types and type_name not in selected_types:
                continue
            
            # Check label filter
            if selected_labels and label_code not in selected_labels:
                continue
            
            filtered_annotations.append(ann)
        
        # Ensure cropped images are available. If any are missing, run
        # cropping in a background worker and show a modal ProgressBar.
        anns_needing_crops = [ann for ann in filtered_annotations
                              if not hasattr(ann, 'cropped_image') or ann.cropped_image is None]

        if anns_needing_crops:
            # If already cropping, wait for the worker to finish
            if getattr(self, '_cropping_in_progress', False):
                return

            image_window = getattr(self.main_window, 'image_window', None)
            raster_manager = getattr(image_window, 'raster_manager', None) if image_window else None

            # If no raster manager available, fall back to synchronous cropping
            if raster_manager is None:
                try:
                    self._ensure_cropped_images(filtered_annotations)
                except Exception:
                    pass
            else:
                # Start background worker
                progress_bar = ProgressBar(parent=self.main_window, title="Cropping images...")
                progress_bar.set_title("Cropping images...")
                progress_bar.start_progress(len(anns_needing_crops))
                progress_bar.cancel_button.setEnabled(True)
                progress_bar.show()
                QApplication.processEvents()

                worker = CroppingWorker(anns_needing_crops, raster_manager)
                self._cropping_in_progress = True
                self._cropping_worker = worker

                # Track processed count locally so we can update the modal
                # ProgressBar via the `update_progress()` incrementing API.
                processed_counter = {'count': 0}

                def _on_progress(step):
                    try:
                        # Increment internal counter and update dialog by one step.
                        processed_counter['count'] += int(step)
                        progress_bar.update_progress()
                        # Update status message with percentage
                        pct = int((processed_counter['count'] / max(1, len(anns_needing_crops))) * 100)
                        self.main_window.statusBar().showMessage(f"Cropping images... {pct}%")
                    except Exception:
                        pass

                def _on_finished():
                    try:
                        progress_bar.finish_progress()
                    except Exception:
                        pass
                    try:
                        progress_bar.close()
                    except Exception:
                        pass
                    try:
                        self.main_window.statusBar().clearMessage()
                    except Exception:
                        pass
                    self._cropping_in_progress = False
                    self._cropping_worker = None
                    # Re-run the refresh now that crops should exist
                    QTimer.singleShot(50, self.refresh_annotations)

                def _on_error(msg):
                    try:
                        progress_bar.close()
                    except Exception:
                        pass
                    try:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, "Cropping Error", str(msg))
                    except Exception:
                        pass
                    self._cropping_in_progress = False
                    self._cropping_worker = None
                    try:
                        self.main_window.statusBar().clearMessage()
                    except Exception:
                        pass

                worker.progress.connect(_on_progress)
                worker.finished.connect(_on_finished)
                worker.error.connect(_on_error)
                progress_bar.cancel_button.clicked.connect(worker.cancel)
                worker.start()
                return
        
        # Get or create data items
        data_items = []
        for ann in filtered_annotations:
            if ann.id not in self.data_item_cache:
                self.data_item_cache[ann.id] = AnnotationDataItem(ann)
            self._connect_annotation_updates(ann)
            data_items.append(self.data_item_cache[ann.id])
        
        self.all_data_items = data_items
        self._update_annotations_display(data_items)
        # Toggle placeholder visibility depending on results
        if not data_items:
            self._show_placeholder()
        else:
            self._show_annotation_gallery()

        # Emit signal with filtered IDs
        filtered_ids = [item.annotation.id for item in data_items]
        self.annotations_filtered.emit(filtered_ids)
        
    def _get_selected_images(self):
        """Get list of selected image names from filter combo."""
        if not hasattr(self, 'image_filter_combo'):
            return None  # No filter = show all
        # Support MultiSelectCombo's API
        try:
            vals = self.image_filter_combo.selected_values()
            return vals
        except Exception:
            # Fallback to legacy QComboBox behavior if present
            try:
                current_data = self.image_filter_combo.currentData()
                if current_data == "all":
                    return None
                current_text = self.image_filter_combo.currentText()
                if current_text == "All Images":
                    return None
                return [current_text] if current_text else None
            except Exception:
                return None
    
    def _get_selected_types(self):
        """Get list of selected annotation types from filter combo."""
        if not hasattr(self, 'type_filter_combo'):
            return None
        try:
            vals = self.type_filter_combo.selected_values()
            return vals
        except Exception:
            try:
                current_data = self.type_filter_combo.currentData()
                if current_data == "all":
                    return None
                current_text = self.type_filter_combo.currentText()
                if current_text == "All Types":
                    return None
                type_map = {
                    "Patch": "PatchAnnotation",
                    "Rectangle": "RectangleAnnotation",
                    "Polygon": "PolygonAnnotation",
                    "MultiPolygon": "MultiPolygonAnnotation"
                }
                if current_data:
                    return [current_data]
                return [type_map.get(current_text, current_text)] if current_text else None
            except Exception:
                return None
    
    def _get_selected_labels(self):
        """Get list of selected labels from filter combo."""
        if not hasattr(self, 'label_filter_combo'):
            return None
        try:
            vals = self.label_filter_combo.selected_values()
            return vals
        except Exception:
            try:
                current_data = self.label_filter_combo.currentData()
                if current_data == "all":
                    return None
                current_text = self.label_filter_combo.currentText()
                if current_text == "All Labels":
                    return None
                return [current_text] if current_text else None
            except Exception:
                return None
    
    def _ensure_cropped_images(self, annotations):
        """Ensure cropped images are available for annotations."""
        # If the user has not applied the filter, avoid creating crops to
        # prevent unnecessary background work while the gallery is in the
        # placeholder state.
        if not getattr(self, '_filter_applied', False):
            return

        # Group annotations by image path to minimize raster loading
        image_window = getattr(self.main_window, 'image_window', None)
        if not image_window:
            return
        
        raster_manager = getattr(image_window, 'raster_manager', None)
        if not raster_manager:
            return
        
        # Group by image path
        anns_by_image = {}
        for ann in annotations:
            if not hasattr(ann, 'cropped_image') or ann.cropped_image is None:
                if ann.image_path not in anns_by_image:
                    anns_by_image[ann.image_path] = []
                anns_by_image[ann.image_path].append(ann)
        
        # Process each image group
        for image_path, anns in anns_by_image.items():
            try:
                # Get the raster for this image
                raster = raster_manager.get_raster(image_path)
                if not raster:
                    continue
                
                # Ensure the rasterio source is loaded
                if not hasattr(raster, '_rasterio_src') or raster._rasterio_src is None:
                    raster.load_rasterio()
                
                rasterio_src = getattr(raster, '_rasterio_src', None)
                if rasterio_src is None:
                    continue
                
                # Create cropped images for all annotations from this image
                for ann in anns:
                    try:
                        ann.create_cropped_image(rasterio_src)
                    except Exception as e:
                        print(f"Warning: Failed to create cropped image for annotation {ann.id}: {e}")
            except Exception as e:
                print(f"Warning: Failed to load raster for {image_path}: {e}")
    
    # -------------------------------------------------------------------------
    # Reactive Slots (for MainWindow signal connections)
    # -------------------------------------------------------------------------
    
    @pyqtSlot(str)
    def on_annotation_created(self, annotation_id):
        """
        Handle a new annotation being created.
        
        Args:
            annotation_id: ID of the newly created annotation.
        """
        # If the gallery is not currently populated (filters not applied), ignore
        # incoming annotation events to avoid creating widgets unexpectedly.
        if not getattr(self, '_filter_applied', False):
            return

        # Refresh filter options in case new images/labels were added
        self.refresh_filter_options()
        
        # Add to cache if it matches current filter
        if hasattr(self.annotation_window, 'annotations_dict'):
            ann = self.annotation_window.annotations_dict.get(annotation_id)
            if ann:
                image_name = os.path.basename(ann.image_path)
                type_name = type(ann).__name__
                label_code = ann.label.short_label_code
                
                # Get filters (None means all selected)
                selected_images = self._get_selected_images()
                selected_types = self._get_selected_types()
                selected_labels = self._get_selected_labels()
                
                # Check if annotation matches filter (None = no filter = include)
                matches_image = selected_images is None or image_name in selected_images
                matches_type = selected_types is None or type_name in selected_types
                matches_label = selected_labels is None or label_code in selected_labels
                
                if matches_image and matches_type and matches_label:
                    if annotation_id not in self.data_item_cache:
                        self._ensure_cropped_images([ann])
                        self.data_item_cache[annotation_id] = AnnotationDataItem(ann)
                        self._connect_annotation_updates(ann)
                    
                    data_item = self.data_item_cache[annotation_id]
                    if data_item not in self.all_data_items:
                        self.all_data_items.append(data_item)
                        self._recalculate_layout()
    
    @pyqtSlot(str)
    def on_annotation_deleted(self, annotation_id):
        """Handle an annotation being deleted."""
        # If the deleted annotation was isolated, exit isolation mode so
        # _recalculate_layout can rebuild from the full remaining set rather
        # than filtering to an empty list and showing a stale placeholder.
        if self.isolated_mode and self.isolated_ids and annotation_id in self.isolated_ids:
            self.isolated_mode = False
            self.isolated_ids = None

        self.data_item_cache.pop(annotation_id, None)
        self.all_data_items = [item for item in self.all_data_items
                               if item.annotation.id != annotation_id]
        QTimer.singleShot(0, self.refresh_annotations)
        
    @pyqtSlot(list)
    def on_annotations_deleted(self, annotation_ids):
        """Handle bulk deletion of annotations."""
        if not annotation_ids:
            return

        ids_set = set(annotation_ids)

        # Exit isolation mode if any isolated annotation was deleted, for the
        # same reason as on_annotation_deleted: the isolation filter would
        # otherwise produce an empty list and show a stale placeholder.
        if self.isolated_mode and self.isolated_ids and ids_set & self.isolated_ids:
            self.isolated_mode = False
            self.isolated_ids = None

        for ann_id in annotation_ids:
            self.data_item_cache.pop(ann_id, None)
        self.all_data_items = [item for item in self.all_data_items
                               if item.annotation.id not in ids_set]
        QTimer.singleShot(0, self.refresh_annotations)

    @pyqtSlot(list)
    def on_annotations_created(self, annotation_ids):
        """
        Handle a bulk creation of annotations (e.g., from an Undo action).
        Processes the incoming IDs once, generates crops, updates cache,
        and recalculates layout a single time.
        """
        if not getattr(self, '_filter_applied', False):
            return

        if not annotation_ids or not hasattr(self.annotation_window, 'annotations_dict'):
            return

        # Refresh filter options (in case new images/labels were added)
        self.refresh_filter_options()

        selected_images = self._get_selected_images()
        selected_types = self._get_selected_types()
        selected_labels = self._get_selected_labels()

        annotations_to_add = []

        for ann_id in annotation_ids:
            ann = self.annotation_window.annotations_dict.get(ann_id)
            if not ann:
                continue

            image_name = os.path.basename(ann.image_path)
            type_name = type(ann).__name__
            label_code = ann.label.short_label_code

            matches_image = selected_images is None or image_name in selected_images
            matches_type = selected_types is None or type_name in selected_types
            matches_label = selected_labels is None or label_code in selected_labels

            if matches_image and matches_type and matches_label:
                annotations_to_add.append(ann)

        if not annotations_to_add:
            return

        # Create crops for all annotations in one pass
        self._ensure_cropped_images(annotations_to_add)

        # Add to cache and main list
        for ann in annotations_to_add:
            if ann.id not in self.data_item_cache:
                self.data_item_cache[ann.id] = AnnotationDataItem(ann)
            self._connect_annotation_updates(ann)

            data_item = self.data_item_cache[ann.id]
            if data_item not in self.all_data_items:
                self.all_data_items.append(data_item)

        # Refresh the list view
        QTimer.singleShot(0, self.refresh_annotations)
    
    @pyqtSlot(str, str)
    def on_annotation_label_changed(self, annotation_id, new_label):
        """
        Handle an annotation's label being changed.

        We don't recreate AnnotationDataItem here because it references the
        mutated annotation in-place. Recreating it would orphan the model's reference.

        Changes are coalesced: if multiple annotations change in the same event-loop
        tick (e.g. bulk-relabel), _flush_label_change_update() runs only once after
        all signals have fired.
        """
        self._pending_label_changes.add(annotation_id)
        if not self._label_change_timer.isActive():
            self._label_change_timer.start()
    
    @pyqtSlot(str)
    def on_annotation_modified(self, annotation_id):
        """
        Handle an annotation being modified (moved/resized).

        Coalesced via _pending_annotation_updates so that N simultaneous
        annotationModified emissions (e.g. during classification inference)
        produce exactly one refresh_annotations() call rather than N.
        """
        self._pending_annotation_updates.add(annotation_id)
        if not self._annotation_update_timer.isActive():
            self._annotation_update_timer.start()
    
    @pyqtSlot(object)
    def on_annotation_selection_changed(self, selected_ids):
        """
        Handle selection changes from AnnotationWindow.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.render_selection_from_ids(set(selected_ids) if selected_ids else set())
        finally:
            self._syncing_selection = False
    
    @pyqtSlot(object)
    def on_annotations_labels_changed(self, changes):
        """
        Handle batch label changes.

        Args:
            changes: List of tuples (annotation_id, old_label, new_label)

        All changed IDs are queued and flushed together in one deferred update,
        so N simultaneous label changes produce exactly one repaint/refresh.
        """
        try:
            if changes:
                # Fast path: avoid O(N) tuple unpacking for large batches.
                # _pending_label_changes only needs IDs, not old/new labels.
                _LARGE_BATCH = 2000
                if len(changes) > _LARGE_BATCH:
                    # For very large batches signal a full refresh by using a sentinel
                    # rather than iterating all N items just to build a set of IDs.
                    self._pending_label_changes.add('__all__')
                else:
                    for ann_id, _old, _new in changes:
                        self._pending_label_changes.add(ann_id)
        except Exception:
            pass
        if not self._label_change_timer.isActive():
            self._label_change_timer.start()

    def _flush_label_change_update(self):
        """
        Process all coalesced label changes in one go.

        Called once per event-loop iteration after one or more label-change signals
        fired.  Drains _pending_label_changes, updates the label filter combo once,
        then either triggers a full refresh (when sorting/filtering requires it) or
        does a lightweight viewport repaint.
        """
        if not self._pending_label_changes:
            return

        # Drain the pending set atomically so re-entrant signals don't double-fire
        _changed = self._pending_label_changes
        self._pending_label_changes = set()

        # Refresh the label filter combo once for the whole batch
        self._populate_label_filter()

        # Check if structural changes are needed
        active_label_filters = self._get_selected_labels()
        current_sort = self.sort_combo.currentText()
        is_sorting_by_label = (current_sort == "Label")
        is_sorting_by_confidence = (current_sort == "Confidence")

        if active_label_filters or is_sorting_by_label or is_sorting_by_confidence:
            # Full refresh required — annotations may need reordering/regrouping
            self.refresh_annotations()
        else:
            # Pure color/text update — just repaint the visible cells
            if hasattr(self, 'list_view') and self.list_view is not None:
                self.list_view.viewport().update()

    @pyqtSlot(str, object)
    def on_annotation_moved(self, annotation_id, move_data):
        """
        Handle annotation being moved.
        
        Args:
            annotation_id: ID of the moved annotation
            move_data: Dict with 'old_center' and 'new_center' QPointF
        """
        # Update cache and refresh list view for moved annotation
        if hasattr(self.annotation_window, 'annotations_dict'):
            ann = self.annotation_window.annotations_dict.get(annotation_id)
            if ann:
                self._ensure_cropped_images([ann])
                self.data_item_cache[annotation_id] = AnnotationDataItem(ann)
                QTimer.singleShot(0, self.refresh_annotations)
    
    @pyqtSlot(str, object)
    def on_annotation_geometry_edited(self, annotation_id, geometry_data):
        """
        Handle annotation geometry being edited.
        
        Args:
            annotation_id: ID of the annotation
            geometry_data: Dict with 'old_geom' and 'new_geom'
        """
        # Geometry changed - regenerate crop and refresh
        if hasattr(self.annotation_window, 'annotations_dict'):
            ann = self.annotation_window.annotations_dict.get(annotation_id)
            if ann:
                self._ensure_cropped_images([ann])
                self.data_item_cache[annotation_id] = AnnotationDataItem(ann)
                QTimer.singleShot(0, self.refresh_annotations)
    
    @pyqtSlot(str, object)
    def on_annotation_cut(self, original_annotation_id, new_annotations):
        """
        Handle annotation being cut into multiple pieces.
        
        Args:
            original_annotation_id: ID of the original annotation
            new_annotations: List of new annotation objects
        """
        # Remove original from cache and add new ones, then refresh
        self.data_item_cache.pop(original_annotation_id, None)
        self._ensure_cropped_images(new_annotations)
        for ann in new_annotations:
            self.data_item_cache[ann.id] = AnnotationDataItem(ann)
        QTimer.singleShot(0, self.refresh_annotations)
    
    @pyqtSlot(object)
    def on_annotations_merged(self, merge_data):
        """
        Handle multiple annotations being merged into one.
        
        Args:
            merge_data: Dict with 'original_ids' list and 'merged' annotation object
        """
        original_ids = merge_data['original_ids']
        merged_annotation = merge_data['merged']
        
        # Remove originals from cache and add merged annotation
        for ann_id in original_ids:
            self.data_item_cache.pop(ann_id, None)
        self._ensure_cropped_images([merged_annotation])
        self.data_item_cache[merged_annotation.id] = AnnotationDataItem(merged_annotation)
        QTimer.singleShot(0, self.refresh_annotations)
    
    @pyqtSlot(str, object)
    def on_annotation_split(self, original_annotation_id, new_annotations):
        """
        Handle annotation being split into multiple pieces.
        
        Args:
            original_annotation_id: ID of the original annotation
            new_annotations: List of new annotation objects
        """
        # Same as cut - remove original and add new ones
        self.on_annotation_cut(original_annotation_id, new_annotations)
    
    # -------------------------------------------------------------------------
    # Gallery Display Logic
    # -------------------------------------------------------------------------
    
    def _update_annotations_display(self, data_items):
        """Update the gallery display with new data items."""
        # If the user has not applied the filter, do not update or create widgets
        if not getattr(self, '_filter_applied', False):
            return
        # Migrate to model/view: replace widget-centric management with model updates
        self.all_data_items = data_items
        # let the view manage selection
        self.last_selected_item_id = None

        # Rebuild model layout (groups + headers) and show gallery
        self._recalculate_layout()
        self._update_toolbar_state()
        
    def _recalculate_layout(self):
        """Rebuild list model layout by grouping items and updating model/delegate."""
        # If the user has not applied the filter, show placeholder and skip
        if not getattr(self, '_filter_applied', False):
            self._show_placeholder()
            return

        if not self.all_data_items:
            self._show_placeholder()
            return

        # Build sorted list
        sorted_data_items = self._get_sorted_data_items()

        # If isolated, only consider isolated ids
        if self.isolated_mode:
            isolated_ids = getattr(self, 'isolated_ids', None)
            if isolated_ids:
                sorted_data_items = [item for item in sorted_data_items if item.annotation.id in isolated_ids]

        # --- Use the central SelectionManager as the absolute source of truth ---
        # This prevents highlights from being lost during delayed layout rebuilds 
        # or cache wipes triggered by image cropping.
        if hasattr(self.main_window, 'selection_manager'):
            current_selection = self.main_window.selection_manager.get_selected_ids()
        else:
            current_selection = self.get_selected_annotation_ids()
        # -----------------------------------------------------------------------------

        # If the effective set to display is empty (e.g. isolation mode with all
        # items deleted, or a filter that matches nothing after isolation pruning),
        # switch to the placeholder rather than showing an empty scroll area.
        if not sorted_data_items:
            self._show_placeholder()
            return

        # Group and set into model (supports headers and collapsed groups)
        groups = self._group_data_items_by_sort_key(sorted_data_items)

        # Synchronous update wrapped in lock to prevent rogue signals
        self._syncing_selection = True
        self.list_model.set_grouped_items(groups)

        # Restore the persistent selection
        if current_selection:
            self.render_selection_from_ids(set(current_selection))

        self._syncing_selection = False

        # Inform delegate of size change
        if self.list_delegate:
            self.list_delegate.item_size = self.current_widget_size
        try:
            self._show_annotation_gallery()
        except Exception:
            pass
    
    def _schedule_update(self):
        """Schedule a delayed update for virtualization."""
        self.update_timer.start(50)
    
    def _update_visible_widgets(self):
        """Show widgets in viewport, hide others for performance."""
        # Virtualization removed in model/view refactor; keep stub to avoid
        # accidental calls to legacy code paths.
        return
    
    # -------------------------------------------------------------------------
    # Sorting Logic
    # -------------------------------------------------------------------------
    
    def _get_sorted_data_items(self):
        """Get data items sorted according to current settings."""
        if self.active_ordered_ids:
            item_map = {i.annotation.id: i for i in self.all_data_items}
            return [item_map[aid] for aid in self.active_ordered_ids if aid in item_map]
        
        sort_type = self.sort_combo.currentText()
        items = list(self.all_data_items)

        if sort_type == "Label":
            items.sort(key=lambda i: (i.effective_label.short_label_code, i.get_effective_confidence()))
        elif sort_type == "Image":
            items.sort(key=lambda i: (os.path.basename(i.annotation.image_path), i.get_effective_confidence()))
        elif sort_type == "Confidence":
            items.sort(key=self._confidence_sort_key)
        elif sort_type == "Area":
            items.sort(key=self._area_sort_key)
        elif sort_type == "Cluster":
            # Build {ann_id -> (cluster_id, dist_from_centroid)} from the EmbeddingViewer
            cluster_map = {}
            try:
                ev = getattr(self.main_window, 'embedding_viewer_window', None)
                if ev is not None and ev._cluster_labels.size > 0:
                    import numpy as _np
                    coords = ev._point_coords_2d
                    labels = ev._cluster_labels
                    ids = ev._point_ids
                    # Compute per-cluster centroids
                    k = int(labels.max()) + 1
                    centroids = _np.zeros((k, 2), dtype=_np.float32)
                    for c in range(k):
                        mask = labels == c
                        if mask.any():
                            centroids[c] = coords[mask].mean(axis=0)
                    # Compute distance from centroid for each point
                    dists = _np.linalg.norm(coords - centroids[labels], axis=1)
                    for ann_id, cluster_id, dist in zip(ids.tolist(), labels.tolist(), dists.tolist()):
                        cluster_map[ann_id] = (int(cluster_id), float(dist))
            except Exception:
                pass
            _NO_CLUSTER = (2 ** 31, 0.0)
            items.sort(key=lambda i: cluster_map.get(i.annotation.id, _NO_CLUSTER))

        return items
    
    def _group_data_items_by_sort_key(self, data_items):
        """Group data items by current sort key for headers."""
        sort_type = self.sort_combo.currentText()
        
        if (not self.active_ordered_ids and sort_type == "None") or self.active_ordered_ids:
            return [("", None, data_items)]

        # Build cluster map once (needed for "Cluster" sort)
        _cluster_map = {}
        _cluster_colors = {}
        if sort_type == "Cluster":
            try:
                ev = getattr(self.main_window, 'embedding_viewer_window', None)
                if ev is not None and ev._cluster_labels.size > 0:
                    for ann_id, cluster_id in zip(ev._point_ids.tolist(), ev._cluster_labels.tolist()):
                        _cluster_map[ann_id] = int(cluster_id)
                        
                    # Fetch cluster colors from the embedding viewer to style the headers
                    if hasattr(ev, '_cluster_colors_rgba'):
                        from PyQt5.QtGui import QColor
                        for i, qc in enumerate(ev._cluster_colors_rgba):
                            _cluster_colors[i] = QColor(int(qc[0]), int(qc[1]), int(qc[2]))
            except Exception:
                pass

        # Group by Label or Image — collect into OrderedDict so identical keys
        # are merged even if items appear out-of-order in the incoming list.
        from collections import OrderedDict
        groups_map = OrderedDict()

        for item in data_items:
            if sort_type == "Label":
                key = item.effective_label.short_label_code
                color = item.effective_color
            elif sort_type == "Image":
                key = os.path.basename(item.annotation.image_path)
                color = None
            elif sort_type == "Confidence":
                key = self._confidence_group_key(item)
                color = None
            elif sort_type == "Cluster":
                cid = _cluster_map.get(item.annotation.id)
                key = f"Cluster {cid}" if cid is not None else "No Cluster"
                color = _cluster_colors.get(cid)  # Apply the centroid color here
            else:
                key = ""
                color = None

            if not key:
                # treat empty key as a single unnamed group
                key = ""

            if key not in groups_map:
                groups_map[key] = {"color": color, "items": []}
            # prefer the first-seen color for the group
            if groups_map[key]["color"] is None and color is not None:
                groups_map[key]["color"] = color
            groups_map[key]["items"].append(item)

        groups = []
        for k, v in groups_map.items():
            groups.append((k, v.get("color"), v.get("items", [])))

        return groups

    @staticmethod
    def _confidence_value(item):
        """Return the confidence value used for sorting and binning."""
        try:
            return float(item.get_effective_confidence())
        except Exception:
            return 1.0

    @staticmethod
    def _confidence_bucket_start(confidence):
        """Map confidence to a 10% bucket start (0, 10, ..., 90)."""
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))
        bucket_start = int(confidence * 10) * 10
        return min(bucket_start, 90)

    @classmethod
    def _confidence_bucket_label(cls, confidence):
        """Return the display label for a confidence bucket."""
        bucket_start = cls._confidence_bucket_start(confidence)
        if bucket_start >= 90:
            return "90-100%"
        return f"{bucket_start}-{bucket_start + 9}%"

    def _confidence_group_key(self, item):
        """Return the confidence group key for a data item."""
        if getattr(item.annotation, 'verified', False):
            return "Verified"
        return self._confidence_bucket_label(self._confidence_value(item))

    def _confidence_sort_key(self, item):
        """Sort unverified annotations by bucket, then by confidence; verified last."""
        confidence = self._confidence_value(item)
        verified = bool(getattr(item.annotation, 'verified', False))
        if verified:
            return (1, 
                    10, 
                    confidence, 
                    item.effective_label.short_label_code, 
                    os.path.basename(item.annotation.image_path), 
                    item.annotation.id)

        bucket_start = self._confidence_bucket_start(confidence)
        return (0, 
                bucket_start, 
                confidence, 
                item.effective_label.short_label_code, 
                os.path.basename(item.annotation.image_path), 
                item.annotation.id)

    def _area_sort_key(self, item):
        """Sort annotations from smallest area to largest area."""
        try:
            area = float(item.annotation.get_area())
        except Exception:
            area = float('inf')

        return (
            area,
            item.effective_label.short_label_code,
            os.path.basename(item.annotation.image_path),
            item.annotation.id,
        )

    @pyqtSlot(object)
    def _on_annotation_updated(self, updated_annotation):
        """Coalesce annotationUpdated signals before refreshing the gallery.

        During batch operations (e.g. classification inference on 500
        annotations) this slot fires once per annotation.  Calling
        on_annotation_modified directly would queue 500 separate
        refresh_annotations() invocations — O(N²) work.  Instead we
        accumulate IDs and let the timer fire a single flush on the next
        event-loop iteration.
        """
        try:
            if updated_annotation is not None:
                self._pending_annotation_updates.add(updated_annotation.id)
                if not self._annotation_update_timer.isActive():
                    self._annotation_update_timer.start()
        except Exception:
            pass

    def _flush_annotation_updates(self):
        """Process all coalesced annotationUpdated notifications in one pass."""
        ids = self._pending_annotation_updates
        self._pending_annotation_updates = set()
        if not ids:
            return
        if not hasattr(self.annotation_window, 'annotations_dict'):
            return
        changed = False
        for annotation_id in ids:
            ann = self.annotation_window.annotations_dict.get(annotation_id)
            if ann:
                # Crops already exist after embedding; this is a no-op for
                # classification but still correct for interactive edits.
                self._ensure_cropped_images([ann])
                self.data_item_cache[annotation_id] = AnnotationDataItem(ann)
                changed = True
        if changed:
            QTimer.singleShot(0, self.refresh_annotations)

    def _connect_annotation_updates(self, annotation):
        """Connect annotation update signals so confidence changes refresh the gallery."""
        try:
            annotation.annotationUpdated.disconnect(self._on_annotation_updated)
        except Exception:
            pass

        try:
            annotation.annotationUpdated.connect(self._on_annotation_updated)
        except Exception:
            pass

        try:
            annotation.verifiedChanged.disconnect(self._on_annotation_updated)
        except Exception:
            pass

        try:
            annotation.verifiedChanged.connect(self._on_annotation_updated)
        except Exception:
            pass
    
    def _clear_separator_labels(self):
        """Remove existing group headers."""
        for header in self._group_headers:
            header.setParent(None)
            header.deleteLater()
        self._group_headers = []
    
    def _create_group_header(self, text, color=None):
        """Create a group header label."""
        header = QLabel(text, self.content_widget)
        
        bg_color = color.name() if color else app_theme.SURFACE_COLOR.name()
        if color:
            luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
            text_color = "#ffffff" if luminance < 0.5 else "#000000"
        else:
            text_color = app_theme.TEXT_PRIMARY_COLOR.name()
        
        header.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
                font-size: 12px;
                color: {text_color};
                background-color: {bg_color};
                border: 2px solid {bg_color if color else '#ccc'};
                border-radius: 4px;
                padding: 6px 10px;
                margin: 2px 0px;
            }}
        """)
        header.setFixedHeight(32)
        if hasattr(self, 'list_view') and self.list_view is not None:
            header.setMinimumWidth(self.list_view.viewport().width() - 20)
        else:
            header.setMinimumWidth(400)
        header.show()
        self._group_headers.append(header)
        return header
    
    # -------------------------------------------------------------------------
    # Toolbar Event Handlers
    # -------------------------------------------------------------------------
    
    def _set_cluster_sort_item_enabled(self, enabled: bool):
        """Enable or disable the 'Cluster' entry in the sort combo."""
        try:
            model = self.sort_combo.model()
            # Locate by text so the helper still works if the sort menu order changes.
            idx = self.sort_combo.findText("Cluster")
            if idx < 0:
                return
            item = model.item(idx)
            if item is None:
                return
            from PyQt5.QtCore import Qt as _Qt
            if enabled:
                item.setFlags(item.flags() | _Qt.ItemIsEnabled | _Qt.ItemIsSelectable)
            else:
                item.setFlags(item.flags() & ~(_Qt.ItemIsEnabled | _Qt.ItemIsSelectable))
                # If "Cluster" is currently selected, fall back to "None"
                if self.sort_combo.currentText() == "Cluster":
                    self.sort_combo.setCurrentText("None")
        except Exception:
            pass

    def update_cluster_sort_state(self, has_clusters: bool):
        """Called by EmbeddingViewerWindow when cluster data is created or cleared."""
        self._set_cluster_sort_item_enabled(has_clusters)

    def _on_sort_changed(self, sort_type):
        """Handle sort type change."""
        self.active_ordered_ids = []
        self._recalculate_layout()

    
    def _isolate_selection(self):
        """Hide non-selected annotations."""
        # Determine current selected ids from the view
        sel_ids = set(self.get_selected_annotation_ids())
        if not sel_ids:
            return

        self.isolated_mode = True
        self.isolated_ids = sel_ids
        # Rebuild model to show only isolated items
        self._recalculate_layout()
        self._update_toolbar_state()
    
    def _show_all_annotations(self):
        """Show all annotations, exit isolation mode."""
        if not self.isolated_mode:
            return

        self.isolated_mode = False
        self.isolated_ids = None
        self.active_ordered_ids = []
        self._recalculate_layout()
        self._update_toolbar_state()
    
    def _update_toolbar_state(self):
        """Update toolbar button states."""
        if hasattr(self, 'list_view') and self.list_view is not None:
            try:
                selection_exists = self.list_view.selectionModel().hasSelection()
            except Exception:
                selection_exists = False
        else:
            selection_exists = False
        
        # Isolate button: enabled only when NOT in isolation mode AND has selection
        # When isolated, button is disabled (user exits via double-click)
        if hasattr(self, 'isolate_button') and self.isolate_button is not None:
            try:
                self.isolate_button.setEnabled(not self.isolated_mode and selection_exists)
            except Exception:
                pass
    
    # -------------------------------------------------------------------------
    # Selection Management
    # -------------------------------------------------------------------------
    
    def select_widget(self, widget):
        """Select a widget."""
        # Accept either a widget-like object, data_item, or annotation id
        aid = None
        try:
            if isinstance(widget, (str, int)):
                aid = widget
            elif hasattr(widget, 'data_item'):
                aid = widget.data_item.annotation.id
            elif hasattr(widget, 'annotation'):
                aid = widget.annotation.id
        except Exception:
            return False

        if aid is None:
            return False

        # Map to model row and select in the view
        row = self.list_model._id_to_row.get(aid)
        if row is None:
            return False
        idx = self.list_model.index(row)
        sel_model = self.list_view.selectionModel()
        sel_model.select(idx, sel_model.Select | sel_model.Rows)
        # update compatibility list
        # No compatibility list maintained anymore
        self._update_toolbar_state()
        return True
    
    def deselect_widget(self, widget):
        """Deselect a widget."""
        aid = None
        try:
            if isinstance(widget, (str, int)):
                aid = widget
            elif hasattr(widget, 'data_item'):
                aid = widget.data_item.annotation.id
            elif hasattr(widget, 'annotation'):
                aid = widget.annotation.id
        except Exception:
            return False

        if aid is None:
            return False

        row = self.list_model._id_to_row.get(aid)
        if row is None:
            return False
        idx = self.list_model.index(row)
        sel_model = self.list_view.selectionModel()
        sel_model.select(idx, sel_model.Deselect | sel_model.Rows)
        # No compatibility list maintained anymore
        self._update_toolbar_state()
        return True
    
    def toggle_widget_selection(self, widget):
        """Toggle widget selection state."""
        aid = None
        try:
            if isinstance(widget, (str, int)):
                aid = widget
            elif hasattr(widget, 'data_item'):
                aid = widget.data_item.annotation.id
            elif hasattr(widget, 'annotation'):
                aid = widget.annotation.id
        except Exception:
            return False
        if aid is None:
            return False
        # determine current selection
        sel_model = self.list_view.selectionModel()
        row = self.list_model._id_to_row.get(aid)
        if row is None:
            return False
        idx = self.list_model.index(row)
        if sel_model.isSelected(idx):
            return self.deselect_widget(aid)
        else:
            return self.select_widget(aid)
    
    def clear_selection(self):
        """Clear all selections."""
        if hasattr(self, 'list_view') and self.list_view is not None:
            blocker = QSignalBlocker(self.list_view.selectionModel())
            try:
                self.list_view.clearSelection()

                # --- Ensure underlying data items are unmarked ---
                for item in self.all_data_items:
                    item.set_selected(False)
                # ------------------------------------------------------
            finally:
                del blocker
            # Force an immediate repaint so the custom delegate removes the
            # dashed selection border from all previously-selected cells.
            # clearSelection() inside a QSignalBlocker can leave the viewport
            # without a scheduled repaint, causing stale dashed borders to persist
            # until the next unrelated redraw event.
            self.list_view.viewport().update()
        else:
            # Nothing to clear when no list view is present
            pass
        self._update_toolbar_state()
    
    def render_selection_from_ids(self, selected_ids):
        """Update visual selection using fast set-diffing."""
        if not selected_ids:
            # clear selection
            self.clear_selection()
            return

        selected_ids_set = set(selected_ids)
        if not hasattr(self, 'list_view') or self.list_view is None:
            return

        sel_model = self.list_view.selectionModel()
        blocker = QSignalBlocker(sel_model)
        try:
            sel_model.clearSelection()
            
            # --- Sync the underlying data items so custom widgets draw correctly ---
            for item in self.all_data_items:
                item.set_selected(item.annotation.id in selected_ids_set)
            # ----------------------------------------------------------------------------
                
            first_idx = None
            for aid in selected_ids_set:
                row = self.list_model._id_to_row.get(aid)
                if row is None:
                    continue
                idx = self.list_model.index(row)
                sel_model.select(idx, sel_model.Select | sel_model.Rows)
                if first_idx is None:
                    first_idx = idx
                    
            # Scroll once to the first selected index to avoid repeated repaints
            if first_idx is not None:
                try:
                    self.list_view.scrollTo(first_idx)
                except Exception:
                    pass
        finally:
            del blocker

        # Force a repaint so the delegate redraws dashed borders for the new
        # selection state — same reason as in clear_selection().
        self.list_view.viewport().update()

        self._update_toolbar_state()

    def _on_list_selection_changed(self, selected, deselected):
        """Slot for list view selection changes -> emit selection_changed(ids)."""
        if getattr(self, '_syncing_selection', False):
            return
        
        # build list of selected ids
        try:
            sel_model = self.list_view.selectionModel()
            indexes = sel_model.selectedIndexes()
            ids = []
            for idx in indexes:
                data = idx.data(self.list_model.DataItemRole)
                if data and data.get('type') == 'annotation':
                    aid = data['item'].annotation.id
                    ids.append(aid)
            if ids:
                self._syncing_selection = True
                try:
                    self.selection_changed.emit(ids)
                finally:
                    self._syncing_selection = False

            else:
                # emit empty selection
                self._syncing_selection = True
                try:
                    self.selection_changed.emit([])
                finally:
                    self._syncing_selection = False

        except Exception:
            pass
    
    def handle_annotation_selection(self, widget, event):
        """Handle selection with keyboard modifiers."""
        # Legacy widget selection handler no longer used with QListView.
        # Selection is handled by the view and `_on_list_selection_changed`.
        return
    
    def handle_annotation_context_menu(self, widget, event):
        """Handle right-click context menu on annotation."""
        if event.modifiers() == Qt.ControlModifier:
            # Ctrl+right-click: locate in annotation window
            ann_id = widget.data_item.annotation.id
            
            # Use SelectionManager if available for context menu navigation
            if hasattr(self.main_window, 'selection_manager'):
                self.main_window.selection_manager.handle_context_menu_selection(
                    ann_id, navigate_to=True
                )
            else:
                # Fallback: manual handling
                ann = widget.data_item.annotation
                
                self.clear_selection()
                self.select_widget(widget)
                # Emit selection change - SelectionManager will handle tool switching
                self.selection_changed.emit([ann_id])
                
                # Change image if needed
                if self.annotation_window.current_image_path != ann.image_path:
                    self.annotation_window.set_image(ann.image_path)
                
                # Select and center on annotation
                self.annotation_window.select_annotation(ann, quiet_mode=True)
                if hasattr(self.annotation_window, 'center_on_annotation'):
                    self.annotation_window.center_on_annotation(ann)
            
            event.accept()
    
    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------
    
    def resizeEvent(self, event):
        """Handle resize to reflow widgets."""
        super().resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._recalculate_layout)
        self._resize_timer.start(100)
    
    def eventFilter(self, source, event):
        """Filter events for list viewport (Ctrl+Wheel zoom and double-click reset)."""
        if hasattr(self, 'list_view') and source is self.list_view.viewport():
            # Handle Ctrl+Wheel for zooming
            if event.type() == QEvent.Wheel:
                try:
                    if event.modifiers() & Qt.ControlModifier:
                        delta = event.angleDelta().y()
                        if delta == 0:
                            return True
                        step = self._widget_size_step if delta > 0 else -self._widget_size_step
                        new_size = self.current_widget_size + step
                        new_size = max(self._widget_size_min, min(self._widget_size_max, new_size))
                        # Prefer even sizes
                        if new_size % 2 != 0:
                            new_size = new_size + 1 if step > 0 else new_size - 1
                            new_size = max(self._widget_size_min, min(self._widget_size_max, new_size))
                        if new_size != self.current_widget_size:
                            self.current_widget_size = new_size
                            if self.list_delegate:
                                self.list_delegate.item_size = self.current_widget_size
                            try:
                                self.list_model.layoutChanged.emit()
                            except Exception:
                                pass
                            try:
                                self.list_view.doItemsLayout()
                            except Exception:
                                pass
                        return True
                except Exception:
                    return True
                
            # Handle double-click on empty space to reset view
            if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                # Check if double-click was on empty space
                index = self.list_view.indexAt(event.pos())
                if not index.isValid():
                    self.list_view.clearSelection()
                    self.reset_view_requested.emit()
                    return True
                
        return super().eventFilter(source, event)

    def _list_view_key_press_event(self, event):
        """Handle key press events for the list view."""
        try:
            if event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
                # Prevent selecting if we are in the placeholder state
                if getattr(self, '_filter_applied', False) and self.all_data_items:
                    try:
                        self.list_view.selectAll()
                    except Exception:
                        pass
                event.accept()
                return
            if event.key() == Qt.Key_Space and (event.modifiers() & Qt.ControlModifier):
                self._confirm_selected_annotations()
                event.accept()
                return
        except Exception:
            pass

        # Fallback to native behavior
        QListView.keyPressEvent(self.list_view, event)

    def _confirm_selected_annotations(self):
        """Confirm selected annotations from the gallery with Ctrl+Space."""
        if not hasattr(self.annotation_window, 'annotations_dict'):
            return

        selected_ids = self.get_selected_annotation_ids()
        if not selected_ids:
            return

        for annotation_id in selected_ids:
            ann = self.annotation_window.annotations_dict.get(annotation_id)
            if not ann:
                continue
            if ann.machine_confidence:
                ann.update_user_confidence(next(iter(ann.machine_confidence)))

        try:
            self.refresh_annotations()
        except Exception:
            pass

        if len(selected_ids) == 1 and hasattr(self.main_window, 'confidence_window'):
            try:
                self.main_window.confidence_window.refresh_display()
            except Exception:
                pass

    def _toggle_group_from_header(self, group_key):
        """Toggle group expansion when sticky header clicked."""
        if not group_key:
            return
        try:
            self.list_model.toggle_group(group_key)
        except Exception:
            try:
                grouped = getattr(self.list_model, '_grouped_items', [])
                self.list_model.set_grouped_items(grouped)
            except Exception:
                pass

    def _select_group_annotations(self, group_key):
        """Select all annotations belonging to the given group (Ctrl+click header)."""
        if not group_key:
            return
        ids = []
        for gk, _gc, items in getattr(self.list_model, '_grouped_items', []):
            if gk == group_key:
                ids.extend(it.annotation.id for it in items)
                break
        if ids:
            self.render_selection_from_ids(set(ids))
            self.selection_changed.emit(ids)

    def _update_sticky_header(self):
        """Update the header overlay to show the group for the top visible items."""
        try:
            if not hasattr(self, 'list_view') or self.list_view is None:
                return
            top_pt = QtCore.QPoint(2, 2)
            idx = self.list_view.indexAt(top_pt)
            if not idx.isValid():
                self._sticky_header.hide()
                return
            row = idx.row()
            model = self.list_model
            header_found = None
            while row >= 0:
                data = model.data(model.index(row), model.DataItemRole)
                if data and data.get('type') == 'header':
                    header_found = data
                    break
                row -= 1

            if header_found is None:
                self._sticky_header.hide()
                return

            text = header_found.get('text', '')
            color = header_found.get('color') or QtGui.QColor('#333333')
            try:
                bg = QtGui.QColor(color) if not isinstance(color, QtGui.QColor) else color
            except Exception:
                bg = QtGui.QColor('#333333')
            r, g, b = bg.red(), bg.green(), bg.blue()
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            text_color = QtGui.QColor('#000000') if luminance > 0.5 else QtGui.QColor('#ffffff')
            expanded = header_found.get('expanded', True)
            self._sticky_header.set_content(text, bg, text_color, expanded, header_found.get('key'))
            vh = self.list_delegate.header_height if self.list_delegate else 32
            self._sticky_header.setGeometry(0, 0, self.list_view.viewport().width(), vh)
            self._sticky_header.show()
            self._sticky_header.raise_()
        except Exception:
            pass
    
    def _viewport_mouse_press(self, event):
        """Handle mouse press for selection."""
        # Let QListView handle mouse presses and selection; disable legacy rubber-band.
        return False
    
    def _viewport_mouse_double_click(self, event):
        """Handle double-click to reset view."""
        # Let QListView handle double-clicks; forward reset-view when double-clicking empty space.
        if event.button() == Qt.LeftButton:
            self.reset_view_requested.emit()
            return True
        return False
    
    def _viewport_mouse_move(self, event):
        """Handle mouse move for rubber band selection."""
        # Rubber-band selection removed; QListView provides native selection.
        return False
    
    def _viewport_mouse_release(self, event):
        """Handle mouse release to finalize selection."""
        # No-op for legacy rubber-band release
        return False
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def isolate_and_select_from_ids(self, ids_to_isolate):
        """Isolate and select specific annotations by ID.

        When the gallery is already isolated to exactly this set of IDs, skips
        the full beginResetModel() rebuild and only updates the selection highlight.
        This is the common case when the user is rotating the embedding or clicking
        through annotations — same selection, no structural change needed.
        """
        ids_set = set(ids_to_isolate)

        # Fast path: already isolated to the same set — just refresh selection highlight
        if self.isolated_mode and self.isolated_ids == ids_set:
            self._syncing_selection = True
            self.render_selection_from_ids(ids_to_isolate)
            self._syncing_selection = False
            self._update_toolbar_state()
            return

        # Build grouped list containing only the requested IDs
        groups = self._group_data_items_by_sort_key(self.all_data_items)
        new_groups = []
        for group_key, group_color, items in groups:
            filtered = [it for it in items if it.annotation.id in ids_set]
            if filtered:
                new_groups.append((group_key, group_color, filtered))

        if not new_groups:
            return

        self.isolated_mode = True
        self.isolated_ids = ids_set

        # --- Synchronous update wrapped in lock to prevent rogue signals ---
        self._syncing_selection = True
        self.list_model.set_grouped_items(new_groups)
        self.render_selection_from_ids(ids_to_isolate)
        self._syncing_selection = False
        # ------------------------------------------------------------------------

        self._update_toolbar_state()
    
    def display_and_isolate_ordered_results(self, ordered_ids):
        """Display annotations in a specific order (e.g., similarity results)."""
        self.active_ordered_ids = ordered_ids
        # Build ordered data items
        item_map = {i.annotation.id: i for i in self.all_data_items}
        ordered_items = [item_map[aid] for aid in ordered_ids if aid in item_map]
        
        # --- Synchronous update wrapped in lock to prevent rogue signals ---
        self._syncing_selection = True
        self.list_model.set_grouped_items([("", None, ordered_items)])
        self._syncing_selection = False
        # ------------------------------------------------------------------------
        
        self.isolated_mode = True
        self.isolated_ids = set(ordered_ids)
        self.render_selection_from_ids(set(ordered_ids))
        self._update_toolbar_state()

    def clear(self):
        """Clear the entire gallery UI and internal state, then emit cleared signal.

        This clears widgets, cached data items, selections, and returns the
        gallery to the placeholder state. It also emits selection/filtered
        signals and the `cleared` signal for other viewers to respond.
        """
        try:
            # Clear caches and lists (model/view)
            self.data_item_cache.clear()
            self.all_data_items = []
            self.last_selected_item_id = None
            self.active_ordered_ids = []
            self.isolated_mode = False
            self.isolated_ids = None

            # Clear model and headers
            self._syncing_selection = True

            try:
                self.list_model.set_grouped_items([])
            except Exception:
                pass
            
            self._syncing_selection = False

            try:
                self._clear_separator_labels()
            except Exception:
                pass

            # Recalculate layout and show placeholder
            try:
                self._recalculate_layout()
            except Exception:
                pass

            # Mark that filters are not applied when cleared so the gallery
            # will remain in placeholder state until the user presses
            # "Apply Filter" again.
            self._filter_applied = False

            self._show_placeholder()
            self._update_toolbar_state()

            # Emit signals indicating no annotations are displayed/selected
            try:
                self.annotations_filtered.emit([])
                self.selection_changed.emit([])
            except Exception:
                pass

        finally:
            try:
                self.cleared.emit()
            except Exception:
                pass

