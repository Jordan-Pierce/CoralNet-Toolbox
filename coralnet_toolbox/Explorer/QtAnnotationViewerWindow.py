# coralnet_toolbox/Explorer/QtAnnotationViewerWindow.py
"""
Standalone Annotation Gallery Window.

This module provides a fully self-contained gallery viewer for annotations
that integrates directly with MainWindow as a dockable widget. It combines
the gallery display functionality with built-in filtering capabilities.
"""

import os
import warnings

import numpy as np

try:
    import jenkspy
    JENKSPY_AVAILABLE = True
except ImportError:
    JENKSPY_AVAILABLE = False

from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, pyqtSlot, QEvent
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QToolButton, QComboBox,
    QLabel, QSlider, QPushButton, QScrollArea, QRubberBand,
    QSizePolicy
)

from coralnet_toolbox.Explorer.QtDataItem import AnnotationImageWidget, AnnotationDataItem
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.animation_manager = None
        
        # Data model
        self.data_item_cache = {}  # annotation_id -> AnnotationDataItem
        self.all_data_items = []  # Currently filtered data items
        self.annotation_widgets_by_id = {}  # annotation_id -> AnnotationImageWidget
        
        # Selection state
        self.selected_widgets = []
        self.last_selected_item_id = None
        self.selection_at_press = set()
        self.mouse_pressed_on_widget = False
        self._syncing_selection = False  # Flag to prevent selection sync loops
        
        # Isolation state
        self.isolated_mode = False
        self.isolated_widgets = set()
        
        # Rubber band selection
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        
        # Sorting state
        self.active_ordered_ids = []
        self.is_confidence_sort_available = True
        self.confidence_breaks = None
        self._group_headers = []
        
        # Display options
        self.current_widget_size = 96
        self.show_confidence = False
        
        # Selection blocking (for external wizards)
        self.selection_blocked = False
        
        # Virtualization
        self.widget_positions = {}  # annotation_id -> QRect
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_visible_widgets)
        
        # Build the UI
        self._setup_ui()
        
    def showEvent(self, event):
        """Handle show event to refresh filters when dock becomes visible."""
        super().showEvent(event)
        # Refresh filter options when window becomes visible
        # This ensures filters are populated even if dock was created before images/labels loaded
        QTimer.singleShot(0, self.refresh_filter_options)
        
    def set_animation_manager(self, manager):
        """
        Set the animation manager for visual effects.
        
        Args:
            manager: AnimationManager instance from MainWindow.
        """
        self.animation_manager = manager
        
    # -------------------------------------------------------------------------
    # Toolbar Creation (for DockWrapper integration)
    # -------------------------------------------------------------------------
    
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
        toolbar.addWidget(self.isolate_button)
        
        toolbar.addSeparator()
        
        # Sort controls
        sort_label = QLabel(" Sort: ")
        toolbar.addWidget(sort_label)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Label", "Image", "Quality", "Anomaly", "Confidence"])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.sort_combo.setMinimumWidth(100)
        toolbar.addWidget(self.sort_combo)
        
        toolbar.addSeparator()
        
        # Size slider
        size_label = QLabel(" Size: ")
        toolbar.addWidget(size_label)
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(256)
        self.size_slider.setValue(96)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(32)
        self.size_slider.setFixedWidth(120)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        toolbar.addWidget(self.size_slider)
        
        self.size_value_label = QLabel("96")
        self.size_value_label.setMinimumWidth(30)
        toolbar.addWidget(self.size_value_label)
        
        toolbar.addSeparator()
        
        # Confidence toggle
        self.confidence_toggle_button = QToolButton()
        self.confidence_toggle_button.setIcon(get_icon("percentage.svg"))
        self.confidence_toggle_button.setToolTip("Toggle confidence badge visibility")
        self.confidence_toggle_button.setCheckable(True)
        self.confidence_toggle_button.setChecked(False)
        self.confidence_toggle_button.clicked.connect(self._on_confidence_toggle_changed)
        toolbar.addWidget(self.confidence_toggle_button)
        
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
        
        # Image filter - searchable combo box
        image_label = QLabel(" Image: ")
        toolbar.addWidget(image_label)
        
        self.image_filter_combo = QComboBox()
        self.image_filter_combo.setEditable(True)
        self.image_filter_combo.setInsertPolicy(QComboBox.NoInsert)
        self.image_filter_combo.setMinimumWidth(150)
        self.image_filter_combo.setToolTip("Filter by image (searchable)")
        self.image_filter_combo.lineEdit().setPlaceholderText("Search images...")
        toolbar.addWidget(self.image_filter_combo)
        
        toolbar.addSeparator()
        
        # Type filter - searchable combo box
        type_label = QLabel(" Type: ")
        toolbar.addWidget(type_label)
        
        self.type_filter_combo = QComboBox()
        self.type_filter_combo.setEditable(True)
        self.type_filter_combo.setInsertPolicy(QComboBox.NoInsert)
        self.type_filter_combo.setMinimumWidth(120)
        self.type_filter_combo.setToolTip("Filter by annotation type")
        self.type_filter_combo.lineEdit().setPlaceholderText("Search types...")
        # Populate type filter with fixed options
        self.type_filter_combo.addItem("All Types", "all")
        self.type_filter_combo.addItem("Patch", "PatchAnnotation")
        self.type_filter_combo.addItem("Rectangle", "RectangleAnnotation")
        self.type_filter_combo.addItem("Polygon", "PolygonAnnotation")
        self.type_filter_combo.addItem("MultiPolygon", "MultiPolygonAnnotation")
        toolbar.addWidget(self.type_filter_combo)
        
        toolbar.addSeparator()
        
        # Label filter - searchable combo box
        label_label = QLabel(" Label: ")
        toolbar.addWidget(label_label)
        
        self.label_filter_combo = QComboBox()
        self.label_filter_combo.setEditable(True)
        self.label_filter_combo.setInsertPolicy(QComboBox.NoInsert)
        self.label_filter_combo.setMinimumWidth(120)
        self.label_filter_combo.setToolTip("Filter by label (searchable)")
        self.label_filter_combo.lineEdit().setPlaceholderText("Search labels...")
        toolbar.addWidget(self.label_filter_combo)
        
        # Spacer to push apply button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Apply button
        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.setToolTip("Apply current filter settings")
        self.apply_filter_button.clicked.connect(self.refresh_annotations)
        toolbar.addWidget(self.apply_filter_button)
        
        # Initialize filter options
        self._populate_filter_combos()
        
        return toolbar
    
    def _populate_filter_combos(self):
        """Populate all filter combo boxes with current options."""
        self._populate_image_filter()
        self._populate_label_filter()
    
    def _populate_image_filter(self):
        """Populate image filter combo with current images."""
        self.image_filter_combo.clear()
        self.image_filter_combo.addItem("All Images", "all")
        
        current_image = None
        if hasattr(self.annotation_window, 'current_image_path') and self.annotation_window.current_image_path:
            current_image = os.path.basename(self.annotation_window.current_image_path)
        
        # Access raster_manager through image_window
        image_window = getattr(self.main_window, 'image_window', None)
        if image_window:
            raster_manager = getattr(image_window, 'raster_manager', None)
            if raster_manager:
                for path in raster_manager.image_paths:
                    image_name = os.path.basename(path)
                    self.image_filter_combo.addItem(image_name, image_name)
        
        # Set default to current image if available
        if current_image:
            index = self.image_filter_combo.findData(current_image)
            if index >= 0:
                self.image_filter_combo.setCurrentIndex(index)
    
    def _populate_label_filter(self):
        """Populate label filter combo with current labels."""
        self.label_filter_combo.clear()
        self.label_filter_combo.addItem("All Labels", "all")
        
        # Access labels through label_window
        label_window = getattr(self.main_window, 'label_window', None)
        if label_window:
            labels = getattr(label_window, 'labels', [])
            for label in labels:
                if hasattr(label, 'short_label_code'):
                    self.label_filter_combo.addItem(label.short_label_code, label.short_label_code)
    
    def refresh_filter_options(self):
        """Refresh filter options based on current state."""
        self._populate_filter_combos()
    
    @pyqtSlot(str)
    def on_image_loaded(self, image_path):
        """
        Handle when a new image is loaded in ImageWindow.
        
        Args:
            image_path: Path to the loaded image.
        """
        # Remember current filter selection before refreshing
        current_data = "all"
        if hasattr(self, 'image_filter_combo'):
            current_data = self.image_filter_combo.currentData()
        
        # Refresh filters to include any new images
        self._populate_filter_combos()
        
        # Only update filter if a specific image was selected (not "All Images")
        # This allows filtering by specific image to follow the user as they switch images
        if current_data != "all" and image_path:
            image_name = os.path.basename(image_path)
            index = self.image_filter_combo.findData(image_name)
            if index >= 0:
                self.image_filter_combo.setCurrentIndex(index)
            # Clear stale widgets and refresh with new image's annotations
            self.refresh_annotations()
    
    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create scroll area for content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        
        layout.addWidget(self.scroll_area)
        
        # Connect scrollbar for virtualization
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._schedule_update)
        
        # Install event filter for rubber band selection
        self.scroll_area.viewport().installEventFilter(self)
        
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_currently_displayed_annotations(self):
        """
        Get the list of currently displayed annotation IDs.
        
        Returns:
            list: List of annotation IDs currently shown in the gallery.
        """
        return [item.annotation.id for item in self.all_data_items]
    
    def get_selected_annotation_ids(self):
        """
        Get the list of currently selected annotation IDs.
        
        Returns:
            list: List of selected annotation IDs.
        """
        return [w.data_item.annotation.id for w in self.selected_widgets]
    
    def highlight_annotations(self, ids):
        """
        Highlight specific annotations in the gallery.
        
        Args:
            ids: List or set of annotation IDs to highlight.
        """
        ids_set = set(ids)
        self.render_selection_from_ids(ids_set)
        
    def refresh_annotations(self):
        """
        Refresh the gallery based on current filter settings.
        
        This method:
        1. Gets annotations from AnnotationWindow that match the filter
        2. Creates/updates data items
        3. Rebuilds the gallery layout
        4. Emits annotations_filtered signal
        """
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
        
        # Ensure cropped images are available
        self._ensure_cropped_images(filtered_annotations)
        
        # Get or create data items
        data_items = []
        for ann in filtered_annotations:
            if ann.id not in self.data_item_cache:
                self.data_item_cache[ann.id] = AnnotationDataItem(ann)
            data_items.append(self.data_item_cache[ann.id])
        
        self.all_data_items = data_items
        self._update_annotations_display(data_items)
        
        # Emit signal with filtered IDs
        filtered_ids = [item.annotation.id for item in data_items]
        self.annotations_filtered.emit(filtered_ids)
        
    def _get_selected_images(self):
        """Get list of selected image names from filter combo."""
        if not hasattr(self, 'image_filter_combo'):
            return None  # No filter = show all
        
        current_data = self.image_filter_combo.currentData()
        if current_data == "all":
            return None  # Show all images
        
        current_text = self.image_filter_combo.currentText()
        if current_text == "All Images":
            return None
        
        # Return list with single selected image
        return [current_text] if current_text else None
    
    def _get_selected_types(self):
        """Get list of selected annotation types from filter combo."""
        if not hasattr(self, 'type_filter_combo'):
            return None  # No filter = show all
        
        current_data = self.type_filter_combo.currentData()
        if current_data == "all":
            return None  # Show all types
        
        current_text = self.type_filter_combo.currentText()
        if current_text == "All Types":
            return None
        
        # Map display name back to class name if needed
        type_map = {
            "Patch": "PatchAnnotation",
            "Rectangle": "RectangleAnnotation", 
            "Polygon": "PolygonAnnotation",
            "MultiPolygon": "MultiPolygonAnnotation"
        }
        
        if current_data:
            return [current_data]
        return [type_map.get(current_text, current_text)] if current_text else None
    
    def _get_selected_labels(self):
        """Get list of selected labels from filter combo."""
        if not hasattr(self, 'label_filter_combo'):
            return None  # No filter = show all
        
        current_data = self.label_filter_combo.currentData()
        if current_data == "all":
            return None  # Show all labels
        
        current_text = self.label_filter_combo.currentText()
        if current_text == "All Labels":
            return None
        
        # Return list with single selected label
        return [current_text] if current_text else None
    
    def _ensure_cropped_images(self, annotations):
        """Ensure cropped images are available for annotations."""
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
                    
                    data_item = self.data_item_cache[annotation_id]
                    if data_item not in self.all_data_items:
                        self.all_data_items.append(data_item)
                        self._recalculate_layout()
    
    @pyqtSlot(str)
    def on_annotation_deleted(self, annotation_id):
        """
        Handle an annotation being deleted.
        
        Args:
            annotation_id: ID of the deleted annotation.
        """
        # Remove from cache
        if annotation_id in self.data_item_cache:
            del self.data_item_cache[annotation_id]
        
        # Remove from current data items
        self.all_data_items = [item for item in self.all_data_items 
                               if item.annotation.id != annotation_id]
        
        # Remove widget if exists
        if annotation_id in self.annotation_widgets_by_id:
            widget = self.annotation_widgets_by_id[annotation_id]
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            widget.setParent(None)
            widget.deleteLater()
            del self.annotation_widgets_by_id[annotation_id]
        
        # Refresh layout
        self._recalculate_layout()
    
    @pyqtSlot(str, str)
    def on_annotation_label_changed(self, annotation_id, new_label):
        """
        Handle an annotation's label being changed.
        
        Args:
            annotation_id: ID of the annotation.
            new_label: New label ID.
        """
        # Update widget visuals
        if annotation_id in self.annotation_widgets_by_id:
            widget = self.annotation_widgets_by_id[annotation_id]
            widget.update()
            widget.update_tooltip()
        
        # Recalculate layout if sorting by label
        if self.sort_combo.currentText() == "Label":
            self._recalculate_layout()
            
        # Refresh label filter options
        self._populate_label_filter()
    
    @pyqtSlot(str)
    def on_annotation_modified(self, annotation_id):
        """
        Handle an annotation being modified (moved/resized).
        
        Args:
            annotation_id: ID of the modified annotation.
        """
        if annotation_id in self.annotation_widgets_by_id:
            widget = self.annotation_widgets_by_id[annotation_id]
            widget.recalculate_aspect_ratio()
            widget.unload_image()
            self._recalculate_layout()
    
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
    
    # -------------------------------------------------------------------------
    # Gallery Display Logic
    # -------------------------------------------------------------------------
    
    def _update_annotations_display(self, data_items):
        """Update the gallery display with new data items."""
        if self.isolated_mode:
            self._show_all_annotations()
        
        # Remove widgets for items no longer in the set
        current_ids = {item.annotation.id for item in data_items}
        for ann_id, widget in list(self.annotation_widgets_by_id.items()):
            if ann_id not in current_ids:
                if widget in self.selected_widgets:
                    self.selected_widgets.remove(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.annotation_widgets_by_id[ann_id]
        
        self.all_data_items = data_items
        self.selected_widgets.clear()
        self.last_selected_item_id = None
        
        self._recalculate_layout()
        self._update_toolbar_state()
        
    def _recalculate_layout(self):
        """Calculate positions for all widgets and resize content area."""
        if not self.all_data_items:
            self.content_widget.setMinimumSize(1, 1)
            return
        
        self._clear_separator_labels()
        sorted_data_items = self._get_sorted_data_items()
        
        # If isolated, only consider isolated widgets
        if self.isolated_mode:
            isolated_ids = {w.data_item.annotation.id for w in self.isolated_widgets}
            sorted_data_items = [item for item in sorted_data_items 
                                if item.annotation.id in isolated_ids]
        
        if not sorted_data_items:
            self.content_widget.setMinimumSize(1, 1)
            return
        
        # Group items by sort key
        groups = self._group_data_items_by_sort_key(sorted_data_items)
        spacing = max(5, int(self.current_widget_size * 0.08))
        available_width = self.scroll_area.viewport().width()
        x, y = spacing, spacing
        max_height_in_row = 0
        
        self.widget_positions.clear()
        
        # Calculate positions
        for group_data in groups:
            if len(group_data) == 3:
                group_name, group_color, group_items = group_data
            else:
                group_name, group_items = group_data
                group_color = None
            
            # Add header if grouped
            if group_name and self.sort_combo.currentText() != "None":
                if x > spacing:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                header = self._create_group_header(group_name, group_color)
                header.move(x, y)
                y += header.height() + spacing
                x = spacing
                max_height_in_row = 0
            
            for data_item in group_items:
                ann_id = data_item.annotation.id
                
                if ann_id in self.annotation_widgets_by_id:
                    widget = self.annotation_widgets_by_id[ann_id]
                    widget.update_height(self.current_widget_size)
                else:
                    widget = AnnotationImageWidget(
                        data_item, self.current_widget_size, self, self.content_widget
                    )
                    widget.set_animation_manager(self.animation_manager)
                    widget.recalculate_aspect_ratio()
                    self.annotation_widgets_by_id[ann_id] = widget
                
                widget_size = widget.size()
                if x > spacing and x + widget_size.width() > available_width:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                
                self.widget_positions[ann_id] = QRect(x, y, widget_size.width(), widget_size.height())
                
                x += widget_size.width() + spacing
                max_height_in_row = max(max_height_in_row, widget_size.height())
        
        total_height = y + max_height_in_row + spacing
        self.content_widget.setMinimumSize(available_width, total_height)
        
        self._update_visible_widgets()
    
    def _schedule_update(self):
        """Schedule a delayed update for virtualization."""
        self.update_timer.start(50)
    
    def _update_visible_widgets(self):
        """Show widgets in viewport, hide others for performance."""
        if not self.widget_positions:
            return
        
        self.content_widget.setUpdatesEnabled(False)
        
        scroll_y = self.scroll_area.verticalScrollBar().value()
        visible_rect = QRect(
            0, scroll_y,
            self.scroll_area.viewport().width(),
            self.scroll_area.viewport().height()
        )
        
        # Add buffer for smoother scrolling
        buffer = self.scroll_area.viewport().height() // 2
        visible_rect.adjust(0, -buffer, 0, buffer)
        
        visible_ids = set()
        for ann_id, rect in self.widget_positions.items():
            if rect.intersects(visible_rect):
                visible_ids.add(ann_id)
        
        for ann_id, widget in self.annotation_widgets_by_id.items():
            if ann_id in visible_ids:
                widget.setGeometry(self.widget_positions[ann_id])
                widget.load_image()
                widget.show()
            else:
                if widget.isVisible():
                    widget.hide()
                    widget.unload_image()
        
        self.content_widget.setUpdatesEnabled(True)
    
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
            items.sort(key=lambda i: i.get_effective_confidence(), reverse=False)
        elif sort_type == "Quality":
            items.sort(key=lambda i: i.quality_score if i.quality_score is not None else 0.5, reverse=False)
        elif sort_type == "Anomaly":
            items.sort(key=lambda i: i.anomaly_score if i.anomaly_score is not None else 0.0, reverse=True)
        
        return items
    
    def _group_data_items_by_sort_key(self, data_items):
        """Group data items by current sort key for headers."""
        sort_type = self.sort_combo.currentText()
        
        if (not self.active_ordered_ids and sort_type == "None") or self.active_ordered_ids:
            return [("", None, data_items)]
        
        if sort_type in ("Quality", "Anomaly", "Confidence"):
            return self._group_by_score_ranges(data_items, sort_type)
        
        # Group by Label or Image
        groups = []
        current_group = []
        current_key = None
        current_color = None
        
        for item in data_items:
            if sort_type == "Label":
                key = item.effective_label.short_label_code
                color = item.effective_color
            elif sort_type == "Image":
                key = os.path.basename(item.annotation.image_path)
                color = None
            else:
                key = ""
                color = None
            
            if key and current_key != key:
                if current_group:
                    groups.append((current_key, current_color, current_group))
                current_group = [item]
                current_key = key
                current_color = color
            else:
                current_group.append(item)
        
        if current_group:
            groups.append((current_key, current_color, current_group))
        
        return groups
    
    def _group_by_score_ranges(self, data_items, score_type):
        """Group items by score ranges for Quality, Anomaly, or Confidence."""
        if score_type == "Quality":
            ranges = [
                ("Poor Quality (<40%)", QColor(220, 20, 60), lambda s: s is not None and s < 0.4),
                ("Fair Quality (40-60%)", QColor(255, 215, 0), lambda s: s is not None and 0.4 <= s < 0.6),
                ("Good Quality (60-80%)", QColor(144, 238, 144), lambda s: s is not None and 0.6 <= s < 0.8),
                ("Excellent Quality (≥80%)", QColor(34, 139, 34), lambda s: s is not None and s >= 0.8),
                ("Unknown Quality", None, lambda s: s is None),
            ]
            get_score = lambda i: i.quality_score
        elif score_type == "Anomaly":
            ranges = [
                ("Very Anomalous (≥80%)", QColor(220, 20, 60), lambda s: s is not None and s >= 0.8),
                ("Anomalous (60-80%)", QColor(255, 140, 0), lambda s: s is not None and 0.6 <= s < 0.8),
                ("Slightly Anomalous (40-60%)", QColor(255, 215, 0), lambda s: s is not None and 0.4 <= s < 0.6),
                ("Normal (<40%)", QColor(144, 238, 144), lambda s: s is not None and s < 0.4),
                ("Unknown Anomaly", None, lambda s: s is None),
            ]
            get_score = lambda i: i.anomaly_score
        else:  # Confidence
            return self._group_by_confidence(data_items)
        
        groups = []
        for label, color, condition in ranges:
            items = [i for i in data_items if condition(get_score(i))]
            if items:
                groups.append((label, color, items))
        return groups
    
    def _group_by_confidence(self, data_items):
        """Group items by confidence score using Jenks breaks."""
        confidences = [i.get_effective_confidence() for i in data_items if i.get_effective_confidence() > 0]
        
        if not confidences:
            return [("", None, data_items)]
        
        confidences = np.array(confidences)
        
        # Calculate breaks
        try:
            if JENKSPY_AVAILABLE and len(confidences) >= 5:
                breaks = jenkspy.jenks_breaks(confidences, n_classes=6)[1:-1]
            else:
                breaks = [float(np.quantile(confidences, q)) for q in [0.17, 0.33, 0.50, 0.67, 0.83]]
            breaks = sorted(list(set(breaks)))
        except Exception:
            breaks = None
        
        self.confidence_breaks = breaks
        
        if not breaks:
            return [("", None, data_items)]
        
        # Create bins
        bins = [-0.001] + breaks + [1.001]
        bin_groups = [[] for _ in range(len(bins) - 1)]
        
        for item in data_items:
            conf = item.get_effective_confidence()
            for i in range(len(bins) - 1):
                if bins[i] < conf <= bins[i + 1]:
                    bin_groups[i].append(item)
                    break
        
        groups = []
        color_map = [
            QColor(220, 20, 60), QColor(255, 99, 71), QColor(255, 165, 0),
            QColor(255, 215, 0), QColor(144, 238, 144), QColor(34, 139, 34)
        ]
        
        for i, items in enumerate(bin_groups):
            if items:
                min_pct = int(max(0, bins[i]) * 100)
                max_pct = int(min(1, bins[i + 1]) * 100)
                label = f"Confidence: {min_pct}-{max_pct}%"
                color = color_map[min(i, len(color_map) - 1)]
                groups.append((label, color, items))
        
        return groups
    
    def _clear_separator_labels(self):
        """Remove existing group headers."""
        for header in self._group_headers:
            header.setParent(None)
            header.deleteLater()
        self._group_headers = []
    
    def _create_group_header(self, text, color=None):
        """Create a group header label."""
        header = QLabel(text, self.content_widget)
        
        bg_color = color.name() if color else "#f0f0f0"
        if color:
            luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
            text_color = "#ffffff" if luminance < 0.5 else "#000000"
        else:
            text_color = "#555"
        
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
        header.setMinimumWidth(self.scroll_area.viewport().width() - 20)
        header.show()
        self._group_headers.append(header)
        return header
    
    # -------------------------------------------------------------------------
    # Toolbar Event Handlers
    # -------------------------------------------------------------------------
    
    def _on_sort_changed(self, sort_type):
        """Handle sort type change."""
        self.active_ordered_ids = []
        if sort_type != "Confidence":
            self.confidence_breaks = None
        self._recalculate_layout()
    
    def _on_size_changed(self, value):
        """Handle size slider change."""
        if value % 2 != 0:
            value -= 1
        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        self._recalculate_layout()
    
    def _on_confidence_toggle_changed(self):
        """Handle confidence badge toggle."""
        self.show_confidence = self.confidence_toggle_button.isChecked()
        for widget in self.annotation_widgets_by_id.values():
            widget.update()
    
    def _isolate_selection(self):
        """Hide non-selected annotations."""
        if not self.selected_widgets:
            return
        
        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                if widget not in self.isolated_widgets:
                    widget.hide()
            self.isolated_mode = True
            self._recalculate_layout()
        finally:
            self.content_widget.setUpdatesEnabled(True)
        
        self._update_toolbar_state()
    
    def _show_all_annotations(self):
        """Show all annotations, exit isolation mode."""
        if not self.isolated_mode:
            return
        
        self.isolated_mode = False
        self.isolated_widgets.clear()
        self.active_ordered_ids = []
        
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                widget.show()
            self._recalculate_layout()
        finally:
            self.content_widget.setUpdatesEnabled(True)
        
        self._update_toolbar_state()
    
    def _update_toolbar_state(self):
        """Update toolbar button states."""
        selection_exists = bool(self.selected_widgets)
        
        # Isolate button: enabled only when NOT in isolation mode AND has selection
        # When isolated, button is disabled (user exits via double-click)
        self.isolate_button.setEnabled(not self.isolated_mode and selection_exists)
    
    # -------------------------------------------------------------------------
    # Selection Management
    # -------------------------------------------------------------------------
    
    def select_widget(self, widget):
        """Select a widget."""
        if not widget.is_selected():
            widget.data_item.set_selected(True)
            widget.update_selection_visuals()
            self.selected_widgets.append(widget)
            self._update_toolbar_state()
            return True
        return False
    
    def deselect_widget(self, widget):
        """Deselect a widget."""
        if widget.is_selected():
            widget.data_item.set_selected(False)
            widget.update_selection_visuals()
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            self._update_toolbar_state()
            return True
        return False
    
    def toggle_widget_selection(self, widget):
        """Toggle widget selection state."""
        if widget.is_selected():
            return self.deselect_widget(widget)
        else:
            return self.select_widget(widget)
    
    def clear_selection(self):
        """Clear all selections."""
        for widget in list(self.selected_widgets):
            self.deselect_widget(widget)
        self.selected_widgets.clear()
        self._update_toolbar_state()
    
    def render_selection_from_ids(self, selected_ids):
        """Update visual selection based on ID set."""
        self.setUpdatesEnabled(False)
        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in selected_ids
                widget.data_item.set_selected(is_selected)
                widget.update_selection_visuals()
            
            self.selected_widgets = [w for w in self.annotation_widgets_by_id.values() if w.is_selected()]
        finally:
            self.setUpdatesEnabled(True)
        self._update_toolbar_state()
    
    def handle_annotation_selection(self, widget, event):
        """Handle selection with keyboard modifiers."""
        if self.selection_blocked:
            return
        
        sorted_items = self._get_sorted_data_items()
        if self.isolated_mode:
            isolated_ids = {w.data_item.annotation.id for w in self.isolated_widgets}
            sorted_items = [i for i in sorted_items if i.annotation.id in isolated_ids]
        
        try:
            current_index = sorted_items.index(widget.data_item)
        except ValueError:
            return
        
        modifiers = event.modifiers()
        changed_ids = []
        
        # Shift: range selection
        if modifiers in (Qt.ShiftModifier, Qt.ShiftModifier | Qt.ControlModifier):
            last_index = -1
            if self.last_selected_item_id:
                try:
                    last_item = self.data_item_cache[self.last_selected_item_id]
                    last_index = sorted_items.index(last_item)
                except (KeyError, ValueError):
                    last_index = -1
            
            if last_index != -1:
                start = min(last_index, current_index)
                end = max(last_index, current_index)
                for i in range(start, end + 1):
                    item = sorted_items[i]
                    w = self.annotation_widgets_by_id.get(item.annotation.id)
                    if w and self.select_widget(w):
                        changed_ids.append(item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            
            self.last_selected_item_id = widget.data_item.annotation.id
        
        # Ctrl: toggle
        elif modifiers == Qt.ControlModifier:
            if self.toggle_widget_selection(widget):
                changed_ids.append(widget.data_item.annotation.id)
            self.last_selected_item_id = widget.data_item.annotation.id
        
        # No modifier: single selection
        else:
            newly_selected_id = widget.data_item.annotation.id
            for w in list(self.selected_widgets):
                if w.data_item.annotation.id != newly_selected_id:
                    if self.deselect_widget(w):
                        changed_ids.append(w.data_item.annotation.id)
            if self.select_widget(widget):
                changed_ids.append(newly_selected_id)
            self.last_selected_item_id = widget.data_item.annotation.id
        
        if changed_ids:
            # Switch to Select tool when selecting annotations
            if hasattr(self.main_window, 'select_tool_action'):
                self.main_window.select_tool_action.setChecked(True)
            self.selection_changed.emit(changed_ids)
    
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
                # Switch to Select tool when selecting annotations
                if hasattr(self.main_window, 'select_tool_action'):
                    self.main_window.select_tool_action.setChecked(True)
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
        """Filter events for rubber band selection."""
        if source is self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                return self._viewport_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self._viewport_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self._viewport_mouse_release(event)
            elif event.type() == QEvent.MouseButtonDblClick:
                return self._viewport_mouse_double_click(event)
        return super().eventFilter(source, event)
    
    def _viewport_mouse_press(self, event):
        """Handle mouse press for selection."""
        if self.selection_blocked:
            return False
        
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            self.selection_at_press = set(self.selected_widgets)
            self.rubber_band_origin = event.pos()
            
            content_pos = self.content_widget.mapFrom(self.scroll_area.viewport(), event.pos())
            child = self.content_widget.childAt(content_pos)
            self.mouse_pressed_on_widget = isinstance(child, AnnotationImageWidget)
            return True
        
        elif event.button() == Qt.LeftButton and not event.modifiers():
            content_pos = self.content_widget.mapFrom(self.scroll_area.viewport(), event.pos())
            if self.content_widget.childAt(content_pos) is None:
                if self.selected_widgets:
                    changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                    self.clear_selection()
                    # No need to switch tool on deselection
                    self.selection_changed.emit(changed_ids)
                return True
        
        return False
    
    def _viewport_mouse_double_click(self, event):
        """Handle double-click to reset view."""
        if self.selection_blocked:
            return False
        
        if event.button() == Qt.LeftButton:
            if self.selected_widgets:
                changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                self.clear_selection()
                # No need to switch tool on deselection/double-click
                self.selection_changed.emit(changed_ids)
            if self.isolated_mode:
                self._show_all_annotations()
            self.reset_view_requested.emit()
            return True
        return False
    
    def _viewport_mouse_move(self, event):
        """Handle mouse move for rubber band selection."""
        if self.selection_blocked:
            return False
        
        if (self.rubber_band_origin is None or
            event.buttons() != Qt.LeftButton or
            event.modifiers() != Qt.ControlModifier or
            self.mouse_pressed_on_widget):
            return False
        
        distance = (event.pos() - self.rubber_band_origin).manhattanLength()
        if distance < self.drag_threshold:
            return True
        
        if not self.rubber_band:
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.scroll_area.viewport())
            self.rubber_band.setStyleSheet(
                "QRubberBand { border: 3px rgb(0, 168, 230); border-style: dashed; "
                "background-color: rgba(0, 168, 230, 30); }"
            )
        
        rect = QRect(self.rubber_band_origin, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()
        
        selection_rect = self.rubber_band.geometry()
        changed_ids = []
        
        for widget in self.annotation_widgets_by_id.values():
            mapped_pos = self.content_widget.mapTo(self.scroll_area.viewport(), widget.geometry().topLeft())
            widget_rect = QRect(mapped_pos, widget.geometry().size())
            
            is_in_band = selection_rect.intersects(widget_rect)
            should_select = (widget in self.selection_at_press) or is_in_band
            
            if should_select and not widget.is_selected():
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            elif not should_select and widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
        
        if changed_ids:
            # Switch to Select tool when selecting annotations via rubber band
            if hasattr(self.main_window, 'select_tool_action'):
                self.main_window.select_tool_action.setChecked(True)
            self.selection_changed.emit(changed_ids)
        
        return True
    
    def _viewport_mouse_release(self, event):
        """Handle mouse release to finalize selection."""
        if self.selection_blocked:
            if self.rubber_band:
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None
            self.rubber_band_origin = None
            return False
        
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None
            self.rubber_band_origin = None
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Preview Label Support
    # -------------------------------------------------------------------------
    
    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label to selected annotations."""
        if not self.selected_widgets or not preview_label:
            return
        
        changed_ids = []
        for widget in self.selected_widgets:
            widget.data_item.set_preview_label(preview_label)
            widget.update()
            changed_ids.append(widget.data_item.annotation.id)
        
        if self.sort_combo.currentText() == "Label":
            self._recalculate_layout()
        
        if changed_ids:
            self.preview_changed.emit(changed_ids)
    
    def clear_preview_states(self):
        """Clear all preview label states."""
        changed = False
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.has_preview_changes():
                widget.data_item.clear_preview_label()
                widget.update()
                changed = True
        
        if changed and self.sort_combo.currentText() == "Label":
            self._recalculate_layout()
    
    def has_preview_changes(self):
        """Check if there are any preview changes."""
        return any(w.data_item.has_preview_changes() for w in self.annotation_widgets_by_id.values())
    
    def apply_preview_changes_permanently(self):
        """Apply all preview changes permanently."""
        applied = []
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.apply_preview_permanently():
                applied.append(widget.annotation)
        return applied
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def isolate_and_select_from_ids(self, ids_to_isolate):
        """Isolate and select specific annotations by ID."""
        widgets = {
            self.annotation_widgets_by_id[aid]
            for aid in ids_to_isolate
            if aid in self.annotation_widgets_by_id
        }
        
        if not widgets:
            return
        
        self.isolated_widgets = widgets
        self.isolated_mode = True
        
        self.render_selection_from_ids(ids_to_isolate)
        self._recalculate_layout()
        self._update_toolbar_state()
    
    def display_and_isolate_ordered_results(self, ordered_ids):
        """Display annotations in a specific order (e.g., similarity results)."""
        self.active_ordered_ids = ordered_ids
        self.render_selection_from_ids(set(ordered_ids))
        
        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                if widget in self.isolated_widgets:
                    widget.show()
                else:
                    widget.hide()
            self.isolated_mode = True
            self._recalculate_layout()
        finally:
            self.content_widget.setUpdatesEnabled(True)
        
        self._update_toolbar_state()
