import warnings

import os
import traceback
import time
from typing import Optional

import numpy as np

import pyqtgraph as pg
from PyQt5.QtGui import QMouseEvent, QPixmap, QImage, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QSize, QObject, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QGraphicsPixmapItem, 
                             QSlider, QLabel, QHBoxLayout, QWidget, QComboBox, QToolButton, QToolBar, QSizePolicy)

from coralnet_toolbox.QtBaseCanvas import BaseCanvas

from coralnet_toolbox.MVAT.core.Ray import CameraRay

from coralnet_toolbox.Annotations import (
    PatchAnnotation,
    PolygonAnnotation,
    RectangleAnnotation,
    MaskAnnotation,
)

from coralnet_toolbox.Tools import (
    PatchTool,
    RectangleTool,
    PolygonTool,
    BrushTool,
    EraseTool,
    FillTool,
    DropperTool,
    SAMTool,
    SeeAnythingTool,
    SelectTool,
    WorkAreaTool,
    ScaleTool,
    RugosityTool,
    PatchSamplingTool,
)

from coralnet_toolbox.QtActions import (
    AddAnnotationAction,
    DeleteAnnotationAction,
    AddAnnotationsAction,
    DeleteAnnotationsAction,
    ActionStack,
    ChangeLabelAction,
    ChangeLabelsAction,
    ResizeAnnotationAction,
    AnnotationGeometryEditAction,
)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.Icons import ColorComboBox
from coralnet_toolbox.Icons import ColormapDelegate

from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import convert_scale_units

from coralnet_toolbox.QtVideoPlayer import VideoPlayerWidget


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationWindow(BaseCanvas):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    viewChanged = pyqtSignal(int, int)  # Signal to emit when view is changed
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes
    annotationSelected = pyqtSignal(int)  # Signal to emit when annotation is selected
    annotationDeleted = pyqtSignal(str)  # Signal to emit when annotation is deleted
    annotationsDeleted = pyqtSignal(list)  # Signal to emit when multiple annotations are deleted
    annotationCreated = pyqtSignal(str)  # Signal to emit when annotation is created
    annotationsCreated = pyqtSignal(list)  # Signal to emit when multiple annotations are created
    annotationModified = pyqtSignal(str)  # Signal to emit when annotation is modified
    annotationMoved = pyqtSignal(str, object)  # annotation_id, {'old_center': QPointF, 'new_center': QPointF}
    annotationLabelChanged = pyqtSignal(str, str)  # annotation_id, new_label
    annotationsLabelsChanged = pyqtSignal(object)  # list of (annotation_id, old_label, new_label)
    annotationCut = pyqtSignal(str, object)  # original_annotation_id, [new_annotations]
    annotationsMerged = pyqtSignal(object)  # {'original_ids':[...], 'merged': merged_annotation}
    annotationSplit = pyqtSignal(str, object)  # original_annotation_id, [new_annotations]
    annotationGeometryEdited = pyqtSignal(str, object)  # annotation_id, {'old_geom':..., 'new_geom':...}
    annotationSelectionChanged = pyqtSignal(object)  # list of annotation IDs when selection changes

    def __init__(self, main_window, parent=None):
        """Initialize the annotation window with the main window and parent widget."""
        super().__init__(parent)  # BaseCanvas initializes scene, pixmap_image, zoom_factor, etc.
        self.main_window = main_window

        # Reference to the global animation manager
        self.animation_manager = None
        self.set_animation_manager(main_window.animation_manager)
        
        # Central annotation data store (owned by MainWindow's AnnotationManager)
        self.annotation_manager = self.main_window.annotation_manager

        self.annotation_size = 224
        self.transparency = 128

        self.drag_start_pos = None
        self.cursor_annotation = None
        self.rasterized_annotations_cache = []  # Caches vector annotations during mask mode
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state
        self._syncing_selection = False  # Flag to prevent selection sync loops
        # Streaming inference mode: when True, new annotations are saved to the
        # data model but heavy Qt graphics are skipped to keep playback smooth.
        self.is_streaming_inference = False
        
        # Image state (BaseCanvas has pixmap_image, active_image, current_image_path)
        self.rasterio_image = None

        # Update placeholder label text for AnnotationWindow's context
        self._placeholder_label.setText(
            "No image loaded\nImport or drag and drop an image or Project file."
        )
        self._placeholder_label.setStyleSheet("color: white; background-color: #1e1e1e; font-size: 14px; padding: 16px;")
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setAutoFillBackground(True)
        
        # Z-channel visualization (BaseCanvas has z_item, z_data_raw, z_data_normalized, etc.)
        # Just set up AnnotationWindow-specific debounce timer (BaseCanvas has generic one)
        self.dynamic_range_timer = self._dynamic_range_timer  # Reference BaseCanvas timer
        self.dynamic_range_update_delay = 100  # milliseconds

        # Video playback state
        self._active_video_raster = None   # VideoRaster when a video is loaded
        self._current_frame_idx: int = 0
        # Pass the annotation window instance to the player so it can access
        # the active VideoRaster even when reparented into toolbar widgets.
        self._video_player = VideoPlayerWidget(self, annotation_window=self)
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._playback_tick)
        # Video toolbar is created lazily via create_video_toolbar()
        self._video_toolbar = None

        # Connect signals to slots
        self.toolChanged.connect(self.set_selected_tool)
        
        self.tools = {}
        self.mask_tools = {}
        
        # Bridge AnnotationWindow lifecycle signals to the central AnnotationManager
        self.annotationCreated.connect(self.annotation_manager.annotationAdded)
        self.annotationsCreated.connect(self.annotation_manager.annotationsAdded)
        self.annotationDeleted.connect(self.annotation_manager.annotationRemoved)
        self.annotationsDeleted.connect(self.annotation_manager.annotationsRemoved)
        self.annotationModified.connect(self.annotation_manager.annotationModified)
        self.annotationLabelChanged.connect(self.annotation_manager.annotationLabelChanged)
        self.annotationSelectionChanged.connect(self.annotation_manager.selectionChanged)

        # Keep video scrub-bar tick marks in sync with annotation changes
        # Connect both singular (for individual operations) and plural (for batch operations)
        self.annotationCreated.connect(self._on_annotation_change_for_video)
        self.annotationDeleted.connect(self._on_annotation_change_for_video)
        self.annotationsCreated.connect(self._on_annotation_change_for_video)  # batch inference
        self.annotationsDeleted.connect(self._on_annotation_change_for_video)  # bulk delete
        
        # Initialize toolbar and status bar widgets
        self._init_toolbar_widgets()  # Likely causes an error

    # --- Property aliases delegating data to central AnnotationManager ---

    @property
    def annotations_dict(self):
        return self.annotation_manager.annotations_dict

    @property
    def image_annotations_dict(self):
        return self.annotation_manager.image_annotations_dict

    @property
    def selected_annotations(self):
        return self.annotation_manager.selected_annotations

    @selected_annotations.setter
    def selected_annotations(self, value):
        self.annotation_manager.selected_annotations = value

    @property
    def action_stack(self):
        return self.annotation_manager.action_stack
        
    def _init_toolbar_widgets(self):
        """Instantiate all status and toolbar widgets previously held by MainWindow."""
        # --- State Properties ---
        self.scaled_view_width_m = 0.0
        self.scaled_view_height_m = 0.0
        self.current_unit_scale = 'm'
        self.current_unit_z = 'm'
        self.current_z_value = None
        self.current_mouse_x = 0
        self.current_mouse_y = 0

        # --- Transparency ---
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)
        # Let the annotation transparency slider naturally expand
        self.transparency_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)

        # --- Positional/Dimensional Labels ---
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.mouse_position_label.setFixedWidth(150)
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.image_dimensions_label.setFixedWidth(150)
        self.view_dimensions_label = QLabel("View: 0 x 0")
        self.view_dimensions_label.setFixedWidth(150)

        # --- Scale ---
        self.scaled_dimensions_label = QLabel("Scale: 0 x 0")
        self.scaled_dimensions_label.setFixedWidth(220)
        self.scaled_dimensions_label.setEnabled(False)
        self.scale_unit_dropdown = QComboBox()
        self.scale_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.scale_unit_dropdown.setCurrentIndex(2)
        self.scale_unit_dropdown.setFixedWidth(72)
        self.scale_unit_dropdown.setEnabled(False)
        self.scale_unit_dropdown.currentTextChanged.connect(self.on_scale_unit_changed)

        # --- Z-Channel Controls ---
        self.z_unit_dropdown = QComboBox()
        self.z_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.z_unit_dropdown.insertSeparator(self.z_unit_dropdown.count())
        self.z_unit_dropdown.addItem('px')
        self.z_unit_dropdown.setCurrentIndex(2)
        self.z_unit_dropdown.setFixedWidth(72)
        self.z_unit_dropdown.setEnabled(False)
        self.z_unit_dropdown.currentTextChanged.connect(self.on_z_unit_changed)

        self.z_label = QLabel("Z: -----")
        self.z_label.setFixedWidth(140)
        self.z_label.setEnabled(False)

        self.z_colormap_dropdown = ColorComboBox()
        delegate = ColormapDelegate(self.z_colormap_dropdown)
        self.z_colormap_dropdown.setItemDelegate(delegate)
        self.z_colormap_dropdown.addItems(['None', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo'])
        self.z_colormap_dropdown.setCurrentIndex(0)
        self.z_colormap_dropdown.setFixedWidth(100)
        self.z_colormap_dropdown.setEnabled(False)
        self.z_colormap_dropdown.currentTextChanged.connect(self.on_z_colormap_changed)

        self.z_transparency_widget = QSlider(Qt.Horizontal)
        self.z_transparency_widget.setRange(0, 255)
        self.z_transparency_widget.setValue(128)
        # Allow the Z transparency slider to naturally expand like the main transparency slider
        self.z_transparency_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.z_transparency_widget.setEnabled(False)
        self.z_transparency_widget.valueChanged.connect(self.update_z_transparency)

        self.z_dynamic_button = QToolButton()
        self.z_dynamic_button.setCheckable(True)
        self.z_dynamic_button.setIcon(get_icon("dynamic.svg"))
        self.z_dynamic_button.setEnabled(False)
        self.z_dynamic_button.toggled.connect(self.on_z_dynamic_toggled)

        # --- Internal Signal Connections ---
        self.mouseMoved.connect(self.update_mouse_position)
        self.imageLoaded.connect(self.update_image_dimensions)
        self.viewChanged.connect(self.update_view_dimensions)
        
    # --- UI LOGIC METHODS (Moved from MainWindow) ---
    def update_mouse_position(self, x, y):
        """Update the mouse position label in the status bar"""
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")
        
        # Store current mouse position for z-channel lookup
        self.current_mouse_x = x
        self.current_mouse_y = y
        
        # Update z-channel value at new mouse position
        raster = None
        if self.current_image_path:
            raster = self.main_window.image_window.raster_manager.get_raster(
                self.current_image_path
            )
        self.update_z_value_at_mouse_position(raster)

    def update_image_dimensions(self, width, height):
        """Update the image dimensions label in the status bar"""
        self.image_dimensions_label.setText(f"Image: {height} x {width}")

    def update_view_dimensions(self, original_width, original_height):
        """Update the view dimensions label in the status bar"""
        # Current extent (view)
        extent = self.viewportToScene()

        top = round(extent.top())
        left = round(extent.left())
        width = round(extent.width())
        height = round(extent.height())

        bottom = top + height
        right = left + width

        # If the current extent includes areas outside the
        # original image, reduce it to be only the original image
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > original_height:
            bottom = original_height
        if right > original_width:
            right = original_width

        width = right - left
        height = bottom - top

        # Update the pixel-based view dimensions
        self.view_dimensions_label.setText(f"View: {height} x {width}")
        
        raster = None
        if self.current_image_path:
            raster = self.main_window.image_window.raster_manager.get_raster(
                self.current_image_path
            )

        if raster and raster.scale_units:
            # Scale exists and is always in meters (standardized internally)
            # Calculate dimensions in meters
            self.scaled_view_width_m = width * raster.scale_x
            self.scaled_view_height_m = height * raster.scale_y
            
            # Check if the scale unit dropdown was previously disabled
            was_disabled = not self.scale_unit_dropdown.isEnabled()

            # Enable the scale widgets
            self.scaled_dimensions_label.setEnabled(True)
            self.scale_unit_dropdown.setEnabled(True)
            
            # If it was disabled before, set to the last selected unit by default
            if was_disabled:
                self.scale_unit_dropdown.blockSignals(True)
                self.scale_unit_dropdown.setCurrentText(self.current_unit_scale)
                self.scale_unit_dropdown.blockSignals(False)

            # Manually call the update function to display the new values
            self.on_scale_unit_changed(self.scale_unit_dropdown.currentText())

        else:
            # No scale, disable and reset
            self.scaled_view_width_m = 0.0
            self.scaled_view_height_m = 0.0
            
            self.scaled_dimensions_label.setText("Scale: 0 x 0")
            self.scaled_dimensions_label.setEnabled(False)
            self.scale_unit_dropdown.setEnabled(False)
            
        # Update z_label with z-channel value at current mouse position
        self.update_z_value_at_mouse_position(raster)

    def update_z_value_at_mouse_position(self, raster):  
        """Update the z_label with z-channel value at current mouse position."""
        if raster and raster.z_channel_lazy is not None:
            # Check if mouse coordinates are within image bounds
            if (0 <= self.current_mouse_x < raster.width and 
                0 <= self.current_mouse_y < raster.height):
                
                try:
                    # Get raw z-value
                    z_value = raster.get_z_value(self.current_mouse_x, self.current_mouse_y)
                    
                    if z_value is None:
                        # Value is NaN or nodata
                        self.z_label.setText("Z: ----")
                        self.z_label.setToolTip("No valid Z-value at this location")
                    else:
                        # Cache the z-value for unit conversion
                        self.current_z_value = z_value
                        
                        # Get the original unit from the raster
                        original_unit = raster.z_unit if raster.z_unit else 'm'
                        
                        # Convert to selected unit if different from original
                        display_value = z_value
                        if self.current_unit_z != original_unit:
                            display_value = convert_scale_units(z_value, original_unit, self.current_unit_z)
                        
                        # Format the display based on data type
                        if raster.z_channel.dtype == np.float32:
                            self.z_label.setText(f"Z: {display_value:.3f}")
                        else:
                            self.z_label.setText(f"Z: {int(display_value)}")
                        
                        # Set simple tooltip with data type and unit
                        z_type = raster.z_data_type if raster.z_data_type else 'Z-channel'
                        tooltip_text = f"{z_type.capitalize()} data in {original_unit}"
                        self.z_label.setToolTip(tooltip_text)
                    
                    # Enable the z_label and dropdown since we have valid data
                    self.z_label.setEnabled(True)
                    self.z_unit_dropdown.setEnabled(True)
                    self.z_colormap_dropdown.setEnabled(True)
                    # Only enable dynamic button if colormap is not set to "None"
                    if self.z_colormap_dropdown.currentText() != "None":
                        self.z_dynamic_button.setEnabled(True)
                    
                except (IndexError, ValueError):
                    pass

    def enable_z_visualization_controls(self, enabled):
        """
        Centralized method to enable or disable all Z-channel visualization controls.
        
        Args:
            enabled (bool): True to enable controls, False to disable them
        """
        self.z_label.setEnabled(enabled)
        self.z_unit_dropdown.setEnabled(enabled)
        self.z_colormap_dropdown.setEnabled(enabled)
        self.z_transparency_widget.setEnabled(enabled)
        
        # Dynamic button is only enabled when a colormap is active (not "None")
        if enabled and self.z_colormap_dropdown.currentText() != "None":
            self.z_dynamic_button.setEnabled(True)
        else:
            self.z_dynamic_button.setEnabled(False)

    def on_image_loaded_check_z_channel(self, image_path):
        """
        Check if the newly loaded image has a z-channel.
        If it doesn't, disable all z-channel UI elements.
        
        Args:
            image_path (str): Path of the loaded image
        """
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if raster and raster.z_channel is None:
            # Image has no z-channel, disable UI elements
            self.z_label.setText("Z: -----")
            self.z_colormap_dropdown.setCurrentText("None")
            self.enable_z_visualization_controls(False)
        elif raster and raster.z_channel is not None:
            # Image has z-channel, enable UI elements
            self.enable_z_visualization_controls(True)
            
            # Force status bar Z-value refresh at current mouse position
            # This ensures z_nodata is properly reflected when switching images
            self.update_z_value_at_mouse_position(raster)

    def on_z_channel_removed(self, image_path):
        """
        Handle z-channel removal for a raster.
        
        Args:
            image_path (str): Path of the raster with removed z-channel
        """
        # If the removed z-channel belongs to the currently displayed image,
        # clear the z-label in the status bar and disable the dropdown
        if image_path == self.current_image_path:
            self.z_label.setText("Z: -----")
            self.z_colormap_dropdown.setCurrentText("None")
            self.enable_z_visualization_controls(False)

    def on_scale_unit_changed(self, to_unit):
        """
        Converts stored meter values to the selected unit and updates the label.
        """
        if not self.scale_unit_dropdown.isEnabled():
            self.scaled_dimensions_label.setText("Scale: 0 x 0")
            return

        # Convert the stored meter values
        converted_height = convert_scale_units(self.scaled_view_height_m, 'm', to_unit)
        converted_width = convert_scale_units(self.scaled_view_width_m, 'm', to_unit)

        # Update the dimensions label
        self.scaled_dimensions_label.setText(f"Scale: {converted_height:.2f} x {converted_width:.2f}")

        # Remember the selected unit
        self.current_unit_scale = to_unit
        
        # Refresh the confidence window if an annotation is selected
        # This is the only refresh needed, as it's the only
        # change that can happen *while* an annotation is displayed.
        if self.main_window.confidence_window.annotation:
            self.main_window.confidence_window.refresh_display()

    def on_z_unit_changed(self, selected_unit):
        """Handle z-unit dropdown changes by re-displaying cached z-value in new unit."""
        # Update the selected unit
        self.current_unit_z = selected_unit
        
        # Re-convert and display the cached z-value in the new unit
        if self.current_z_value is not None:
            # Use the current image path to get the correct raster for unit info
            image_path = self.main_window.image_window.selected_image_path
            try:
                # Get the current raster to fetch original unit and data type info
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if raster and raster.z_channel_lazy is not None:
                    original_unit = raster.z_unit if raster.z_unit else 'm'
                    z_channel = raster.z_channel_lazy
                    
                    # Convert from original unit to selected unit
                    converted_value = convert_scale_units(
                        self.current_z_value, 
                        original_unit, 
                        selected_unit
                    )
                    
                    # Format the display based on data type
                    if z_channel.dtype == np.float32:
                        self.z_label.setText(f"Z: {converted_value:.3f}")
                    else:
                        self.z_label.setText(f"Z: {int(converted_value)}")
            except Exception:
                pass  # If conversion fails, keep last value displayed
        
        # Refresh the confidence window if an annotation is selected
        if self.main_window.confidence_window.annotation:
            self.main_window.confidence_window.refresh_display()

    def on_z_colormap_changed(self, colormap_name):
        """Handle z-colormap dropdown changes by updating the annotation window."""
        self.update_z_colormap(colormap_name)
        
        # Sync to all context matrix canvases
        if hasattr(self.main_window, 'context_matrix') and self.main_window.context_matrix:
            self.main_window.context_matrix.sync_z_colormap_to_all_canvases(colormap_name)
        
        # Enable/disable z_transparency_widget based on colormap selection
        if colormap_name == "None":
            self.z_transparency_widget.setEnabled(False)
            self.z_dynamic_button.setEnabled(False)
            self.z_dynamic_button.setChecked(False)
        else:
            # Enable the transparency slider and dynamic range button if Z data is available
            if self.z_data_raw is not None:
                self.z_transparency_widget.setEnabled(True)
                self.z_dynamic_button.setEnabled(True)

    def update_z_transparency(self, value):
        """
        Update the Z-channel visualization opacity.
        
        Args:
            value (int): Slider value from 0-255
        """
        # Convert slider value (0-255) to opacity (0.0-1.0)
        opacity = value / 255.0
        
        # Update the annotation window's z-channel opacity
        self.set_z_opacity(opacity)
        
        # Sync to all context matrix canvases
        if hasattr(self.main_window, 'context_matrix') and self.main_window.context_matrix:
            self.main_window.context_matrix.sync_z_opacity_to_all_canvases(opacity)

    def on_z_dynamic_toggled(self, checked):
        """Handle z-dynamic scaling button toggle."""
        self.toggle_dynamic_z_scaling(checked)
        
        # Sync to all context matrix canvases
        if hasattr(self.main_window, 'context_matrix') and self.main_window.context_matrix:
            self.main_window.context_matrix.sync_z_dynamic_scaling_to_all_canvases(checked)

    def update_label_transparency(self, value):
        """Update the transparency for all annotations in the current image."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Clamp the transparency value to valid range
        transparency = max(0, min(255, value))
        
        # Update transparency slider position
        if self.transparency_slider.value() != transparency:
            # Temporarily block signals to prevent infinite recursion
            self.transparency_slider.blockSignals(True)
            self.transparency_slider.setValue(transparency)
            self.transparency_slider.blockSignals(False)

        # Update transparency for ALL vector annotations in the current image
        # (regardless of visibility - this ensures hidden annotations have correct transparency when shown)
        for annotation in self.get_image_annotations():
            annotation.update_transparency(transparency)

        try:
            # Handle mask annotation updates
            mask = self.current_mask_annotation
            if mask:
                self.main_window.label_window.set_mask_transparency(transparency)
        except Exception as e:
            pass

        # Sync transparency to all context matrix canvases
        if hasattr(self.main_window, 'context_matrix') and self.main_window.context_matrix:
            self.main_window.context_matrix.sync_annotations_to_all_canvases()

        # PHANTOM ARCHITECTURE: Re-render phantom layer with new transparency
        self.refresh_phantom_annotations()

        # Restore cursor
        QApplication.restoreOverrideCursor()
        
    # --- VIDEO TOOLBAR HOOK ---
    def create_video_toolbar(self) -> QToolBar:
        """Create the video player toolbar (hidden until a VideoRaster is loaded)."""
        toolbar = QToolBar("Video Player")
        toolbar.setMovable(False)
        toolbar.addWidget(self._video_player)
        toolbar.setVisible(False)
        self._video_toolbar = toolbar
        return toolbar

    # --- DOCK WRAPPER HOOKS ---
    def create_top_toolbar(self) -> QToolBar:
        """Create the top toolbar with annotation tools and transparency slider.
        """
        toolbar = QToolBar("Annotation Tools")
        toolbar.setMovable(False)
        
        toolbar.addSeparator()

        # Transparency widget (annotation transparency)
        trans_widget = QWidget()
        trans_layout = QHBoxLayout(trans_widget)
        trans_layout.setContentsMargins(4, 0, 4, 0)
        t_icon = QLabel()
        t_icon.setPixmap(get_icon("transparent.svg").pixmap(QSize(16, 16)))
        o_icon = QLabel()
        o_icon.setPixmap(get_icon("opaque.svg").pixmap(QSize(16, 16)))
        trans_layout.addWidget(t_icon)
        trans_layout.addWidget(self.transparency_slider)
        trans_layout.addWidget(o_icon)
        toolbar.addWidget(trans_widget)
        
        toolbar.addSeparator()

        # Z-channel controls moved to top toolbar (to the right of annotation transparency)
        z_widget = QWidget()
        z_layout = QHBoxLayout(z_widget)
        z_layout.setContentsMargins(4, 0, 4, 0)
        # Order: dynamic range button, z transparency slider, then colormap combo (swapped)
        z_layout.addWidget(self.z_dynamic_button)
        z_layout.addWidget(self.z_transparency_widget)
        z_layout.addWidget(self.z_colormap_dropdown)
        toolbar.addWidget(z_widget)

        toolbar.addSeparator()

        return toolbar

    def create_bottom_toolbar(self) -> QToolBar:
        """Create the bottom toolbar with mouse position, image/view dimensions, 
        scale, and z-channel info.
        """
        toolbar = QToolBar("Annotation Status")
        toolbar.setMovable(False)
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(12)
        
        def make_group(*widgets):
            g = QWidget()
            l = QHBoxLayout(g)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(6)
            for w in widgets: l.addWidget(w)
            return g
            
        group_mouse = make_group(self.mouse_position_label)
        group_image = make_group(self.image_dimensions_label)
        group_view = make_group(self.view_dimensions_label)
        group_scale = make_group(self.scale_unit_dropdown, self.scaled_dimensions_label)
        # Keep only the unit dropdown and the z value label in the bottom status bar.
        group_z = make_group(self.z_unit_dropdown, self.z_label)
        
        layout.addWidget(group_mouse)
        layout.addStretch(1)
        layout.addWidget(group_image)
        layout.addStretch(1)
        layout.addWidget(group_view)
        layout.addStretch(1)
        layout.addWidget(group_scale)
        layout.addStretch(1)
        layout.addWidget(group_z)
        
        toolbar.addWidget(container)
        return toolbar
        
    def initialize_tools(self):
        """Initialize tools"""
        self.tools = {
            # Selectable annotation tools
            "select": SelectTool(self),
            "patch": PatchTool(self),
            "rectangle": RectangleTool(self),
            "polygon": PolygonTool(self),
            "sam": SAMTool(self),
            "see_anything": SeeAnythingTool(self),
            "work_area": WorkAreaTool(self),
            # Selectable mask tools
            "brush": BrushTool(self),
            "fill": FillTool(self),
            "erase": EraseTool(self),
            "dropper": DropperTool(self),
            # Dialog tools
            "scale": ScaleTool(self),
            "rugosity": RugosityTool(self),
            "patch_sampling": PatchSamplingTool(self),
        }
        
        # Defines which tools trigger mask mode
        self.mask_tools = {"brush", "fill", "erase", "dropper"}
        
    def set_animation_manager(self, manager):
        """
        Receives the central AnimationManager from the MainWindow.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        self.animation_manager = manager

    def _is_in_mask_editing_mode(self):
        """Check if the annotation window is currently in mask editing mode."""
        return self.selected_tool and self.selected_tool in self.mask_tools
    
    def on_annotation_updated(self, updated_annotation):
        """
        Handle annotation update signal - refresh graphics if annotation is currently displayed.
        This is called when an annotation's label or other properties change.
        """
        # Only update graphics if the annotation belongs to the currently displayed image
        # and has a valid graphics item in the scene
        if (updated_annotation.image_path == self.current_image_path and 
            updated_annotation.is_graphics_item_valid()):
            updated_annotation.update_graphics_item()
            
    def set_incoming_marker(self, u, v, color):
        """Set the incoming marker (focal point) position and color from MVAT.
        
        Routes to the unified BaseCanvas static marker system so AnnotationWindow
        displays the focal point using the same marker as all other viewports.
        """
        # Use BaseCanvas's unified static marker (crosshair) instead of a separate marker
        self.update_static_marker(int(u), int(v), color=color)
            
    def showEvent(self, event):
        """Handle show events to fit the view to the image."""
        super().showEvent(event)
        
        # Connect to ImageWindow signals
        self.main_window.image_window.imageLoaded.connect(self.on_image_loaded_check_z_channel)
        self.main_window.image_window.zChannelRemoved.connect(self.on_z_channel_removed)
    
    def resizeEvent(self, event):
        """Handle resize events to maintain proper view fitting."""
        super().resizeEvent(event)
        
        # Only fit view if we have an active image
        if self.active_image and self.pixmap_image and self.scene:
            # No zoom tool or hasn't been used, safe to fit
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Keep placeholder geometry in sync with viewport size
        try:
            if hasattr(self, '_placeholder_label') and self._placeholder_label.isVisible():
                self._placeholder_label.setGeometry(self.viewport().rect())
        except Exception:
            pass

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

    def wheelEvent(self, event: QMouseEvent):
        """Handle mouse wheel events for zooming."""
        # Handle zooming with the mouse wheel (pass to active tool if Ctrl+wheel)
        if self.selected_tool and event.modifiers() & Qt.ControlModifier:
            self.tools[self.selected_tool].wheelEvent(event)
        else:
            # Let BaseCanvas handle native zoom via super()
            super().wheelEvent(event)

        self.viewChanged.emit(*self.get_image_dimensions())
        
        # Debounce dynamic Z-range update during zoom (prevents stuttering)
        self.schedule_dynamic_range_update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for the active tool."""        
        # Check if a tool is selected before proceeding
        if self.selected_tool:
            # If the selected tool is a mask tool, delegate the event to it
            if self.selected_tool in self.mask_tools:
                self.tools[self.selected_tool].mousePressEvent(event)
            # Otherwise, use the original logic for vector annotation tools
            else:
                self.tools[self.selected_tool].mousePressEvent(event)
        
        # Let BaseCanvas handle native pan/zoom via super()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement events for the active tool."""
        # Check if a tool is selected before proceeding
        if self.selected_tool:
            # If the selected tool is a mask tool, delegate the event to it
            if self.selected_tool in self.mask_tools:
                self.tools[self.selected_tool].mouseMoveEvent(event)
            # Otherwise, use the original logic for vector annotation tools
            else:
                self.tools[self.selected_tool].mouseMoveEvent(event)
        
        scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))

        if not self.cursorInWindow(event.pos()):
            self.toggle_cursor_annotation()

        # Let BaseCanvas handle native pan/zoom via super()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for the active tool."""
        # Check if a tool is selected before proceeding
        if self.selected_tool:
            # If the selected tool is a mask tool, delegate the event to it
            if self.selected_tool in self.mask_tools:
                self.tools[self.selected_tool].mouseReleaseEvent(event)
            # Otherwise, use the original logic for vector annotation tools
            else:
                self.tools[self.selected_tool].mouseReleaseEvent(event)
        
        self.toggle_cursor_annotation()
        self.drag_start_pos = None
        
        # Update dynamic Z-range after panning completes (debounced)
        self.schedule_dynamic_range_update()
        
        # Let BaseCanvas handle native pan/zoom via super()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle mouse double-click events to set focal point in MVATViewer."""
        # Only process left double-clicks
        if event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        
        # Check if MVAT manager exists and is accessible
        mvat_manager = getattr(self.main_window, 'mvat_manager', None)
        if mvat_manager is None:
            super().mouseDoubleClickEvent(event)
            return
        
        # Check if current image has camera data
        if not self.current_image_path or self.current_image_path not in mvat_manager.cameras:
            super().mouseDoubleClickEvent(event)
            return
        
        # Get scene position
        scene_pos = self.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        # Check if position is within image bounds
        camera = mvat_manager.cameras[self.current_image_path]
        if not (0 <= x < camera.width and 0 <= y < camera.height):
            super().mouseDoubleClickEvent(event)
            return
        
        terminal_point = None

        # --- PLAN A: Index Map (Flawless 3D Coordinate) ---
        try:
            primary_target = mvat_manager.viewer.scene_context.get_primary_target()
            if primary_target is not None:
                candidate_id = camera.get_index_at_pixel(x, y)
                if candidate_id is not None and int(candidate_id) > -1:
                    raw_coord = primary_target.get_element_coordinate(int(candidate_id))
                    if raw_coord is not None:
                        # ---> Safely cast PyTorch Tensor to NumPy! <---
                        if hasattr(raw_coord, 'cpu'):
                            terminal_point = raw_coord.cpu().numpy().astype(np.float64)
                        else:
                            terminal_point = np.asarray(raw_coord, dtype=np.float64)
        except Exception:
            pass

        # --- PLAN B: Depth Map / Scene Median Fallback ---
        if terminal_point is None:
            raster = camera._raster
            depth = None
            
            if raster.z_channel is not None and raster.z_data_type == 'depth':
                try:
                    depth = raster.get_z_value(x, y)
                except Exception:
                    pass
            
            # Get default depth from scene if no depth available
            if depth is None or depth <= 0 or np.isnan(depth):
                try:
                    default_depth = mvat_manager.viewer.get_scene_median_depth(camera.position)
                except Exception:
                    default_depth = 10.0
            else:
                default_depth = depth
            
            # Create ray from pixel position to get 3D world point
            try:
                ray = CameraRay.from_pixel_and_camera(
                    pixel_xy=(x, y),
                    camera=camera,
                    depth=depth,
                    default_depth=default_depth
                )
                terminal_point = ray.terminal_point
            except Exception as e:
                print(f"Warning: Could not set focal point from double-click: {e}")
        
        # Trigger projection to all context cameras
        if terminal_point is not None:
            mvat_manager.viewer.set_focal_point(terminal_point)
        
        super().mouseDoubleClickEvent(event)
        
    def keyPressEvent(self, event):
        """Handle keyboard press events including undo/redo and deletion of selected annotations."""
        # Handle Ctrl+H to reset the scene view
        if event.key() == Qt.Key_H and event.modifiers() == Qt.ControlModifier:
            self.reset_scene_view()
            return
        
        # Handle Ctrl+A for select/unselect all annotations
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            current_annotations = self.get_image_annotations()
            if len(self.selected_annotations) == len(current_annotations):
                self.unselect_annotations()
            else:
                if not self.main_window.select_tool_action.isChecked():
                    self.main_window.choose_specific_tool("select")
                self.select_annotations()
            return
        
        if self.active_image and self.selected_tool:
            self.tools[self.selected_tool].keyPressEvent(event)

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle keyboard release events for the active tool."""
        if self.active_image and self.selected_tool:
            self.tools[self.selected_tool].keyReleaseEvent(event)
        super().keyReleaseEvent(event)

    # =========================================================================
    # VIDEO MODE
    # =========================================================================

    def _clear_annotation_graphics_single(self, annotation):
        """Strip all graphics references from a single annotation without crashing."""
        try:
            if (hasattr(annotation, 'graphics_item_group') and
                    annotation.graphics_item_group is not None):
                try:
                    if annotation.graphics_item_group.scene():
                        annotation.graphics_item_group.scene().removeItem(
                            annotation.graphics_item_group
                        )
                except RuntimeError:
                    pass
            annotation.graphics_item_group = None
            annotation.graphics_item = None
            annotation.center_graphics_item = None
            annotation.bounding_box_graphics_item = None
            if hasattr(annotation, 'tag_item'):
                annotation.tag_item = None
            if hasattr(annotation, 'dimension_tag_item'):
                annotation.dimension_tag_item = None
            annotation.is_selected = False
        except Exception:
            pass

    def _on_annotation_change_for_video(self, *args):
        """Slot called when annotations are created or deleted; refreshes scrub-bar ticks."""
        if self._active_video_raster is not None:
            self._update_video_annotation_marks()

    def _get_annotated_frame_indices(self) -> set:
        """Return the set of frame indices that have at least one annotation for the active video."""
        if self._active_video_raster is None:
            return set()
        prefix = self._active_video_raster.image_path + '::frame_'
        frame_indices = set()
        for key, annotations in self.image_annotations_dict.items():
            if key.startswith(prefix) and annotations:
                try:
                    frame_indices.add(int(key.split('::frame_', 1)[1]))
                except (ValueError, IndexError):
                    pass
        return frame_indices

    def _update_video_annotation_marks(self):
        """Compute which frame indices have annotations and push them to the player slider."""
        self._video_player.update_annotation_marks(self._get_annotated_frame_indices())

    def _activate_video_mode(self, video_raster):
        """Switch the annotation window into video mode for the given VideoRaster."""
        # If already active for the same raster, just ensure player is visible
        if self._active_video_raster is video_raster:
            if self._video_toolbar is not None:
                self._video_toolbar.setVisible(True)
            return

        # Stop any existing playback
        self._playback_timer.stop()

        self._active_video_raster = video_raster
        self._current_frame_idx = 0

        # Start the background decode worker for this raster (paused until play is clicked)
        video_raster.start_decode_worker(start_frame=0)
        video_raster.frameReady.connect(self._on_worker_frame_ready)

        # Show the video player toolbar
        if self._video_toolbar is not None:
            self._video_toolbar.setVisible(True)

        # Connect player signals (disconnect first to avoid duplicates)
        try:
            self._video_player.seekChanged.disconnect()
        except Exception:
            pass
        try:
            self._video_player.playClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.pauseClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.nextFrameClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevFrameClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.nextAnnotatedClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevAnnotatedClicked.disconnect()
        except Exception:
            pass

        self._video_player.seekChanged.connect(self._on_video_seek)
        self._video_player.playClicked.connect(self._on_video_play)
        self._video_player.pauseClicked.connect(self._on_video_pause)
        self._video_player.nextFrameClicked.connect(self._on_video_next)
        self._video_player.prevFrameClicked.connect(self._on_video_prev)
        self._video_player.nextAnnotatedClicked.connect(self._on_video_next_annotated)
        self._video_player.prevAnnotatedClicked.connect(self._on_video_prev_annotated)

        # Reset player state
        self._video_player.reset()

        # Display frame 0
        self._display_video_frame(0)

        # Populate tick marks for any pre-existing annotations
        self._update_video_annotation_marks()

    def _deactivate_video_mode(self):
        """Leave video mode (called when switching to a regular image)."""
        if self._active_video_raster is None:
            return

        self._playback_timer.stop()
        self._video_player.set_paused()

        # Stop the decode worker for the outgoing raster
        if self._active_video_raster is not None:
            try:
                self._active_video_raster.frameReady.disconnect(self._on_worker_frame_ready)
            except Exception:
                pass
            self._active_video_raster.stop_decode_worker()

        # Disconnect player signals
        try:
            self._video_player.seekChanged.disconnect()
        except Exception:
            pass
        try:
            self._video_player.playClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.pauseClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.nextFrameClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevFrameClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.nextAnnotatedClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevAnnotatedClicked.disconnect()
        except Exception:
            pass

        self._video_player.reset()
        self.main_window.set_video_playback_tools_enabled(True)

        if self._video_toolbar is not None:
            self._video_toolbar.setVisible(False)

        self._active_video_raster = None
        self._current_frame_idx = 0

    def _display_video_frame(self, frame_idx: int):
        """
        Full display path: load frame, run load_visuals, load annotations.
        Only called on seek/pause (not during playback).
        """
        video_raster = self._active_video_raster
        if video_raster is None:
            return

        frame_idx = max(0, min(frame_idx, video_raster.frame_count - 1))
        self._current_frame_idx = frame_idx

        # Build the virtual path that serves as this frame's image_path key
        virtual_path = video_raster.make_frame_path(video_raster.image_path, frame_idx)

        # Get the frame QImage
        q_image = video_raster.get_frame(frame_idx)
        if q_image is None or q_image.isNull():
            return

        # Make sure rasterio_image is set for annotation cropping
        self.rasterio_image = video_raster.rasterio_src

        # Use BaseCanvas canonical loader (clears scene, sets pixmap, fits view)
        self.load_visuals(q_image, virtual_path, None)

        # Restore cursor (load_visuals may not set it)
        self.active_image = True

        # Update status bar dimensions
        self.imageLoaded.emit(video_raster.width, video_raster.height)
        self.viewChanged.emit(video_raster.width, video_raster.height)

        # Load annotations for this virtual frame path
        self.load_annotations()

        # Update the player widget state first so the slider range is valid
        self._video_player.update_state(frame_idx, video_raster.frame_count)

        # Then refresh scrub-bar tick marks (cheap; only runs on seek/pause, not playback)
        # AnnotatedSlider.paintEvent checks slider.maximum(), so ensure range set first.
        self._update_video_annotation_marks()

    def _on_video_seek(self, frame_idx: int):
        """Handle slider seek: stop playback, display frame."""
        self._playback_timer.stop()
        if self._active_video_raster is not None:
            self._active_video_raster.pause_decode_worker()
        if self._video_player.is_playing:
            self._video_player.set_paused()
            self.main_window.set_video_playback_tools_enabled(True)
        self._display_video_frame(frame_idx)

    def _on_worker_frame_ready(self, frame_idx: int, q_img):
        """
        Fast paint path: called by the decode worker for every decoded frame.

        During playback this is the *only* thing the main thread does per frame:
        swap the scene pixmap and update the slider label.  No scene rebuild,
        no annotation load.  Those happen only when playback pauses.

        Frames that arrive while the player is paused (e.g. stale events queued
        between a pause signal and the main-thread slot running) are discarded.
        """
        vr = self._active_video_raster
        if vr is None or not self._video_player.is_playing:
            # Release the drop-frame gate even on discard so the worker
            # can emit the next frame if playback resumes.
            if vr is not None and vr._decode_worker is not None:
                vr._decode_worker._pending_emit = False
            return

        self._current_frame_idx = frame_idx
        self.current_image_path = vr.make_frame_path(vr.image_path, frame_idx)

        # --- PHASE 4: FAST PLAYBACK RENDERING ---
        if self._base_image_item is not None:
            # 1. Update the background image instantly
            self._base_image_item.set_image(q_img)
            
            # 2. Compile paths without creating Qt Items
            try:
                frame_annotations = self.image_annotations_dict.get(self.current_image_path, [])
                paths_data = []
                
                for a in frame_annotations:
                    if getattr(a.label, 'is_visible', True) and not hasattr(a, 'mask_data'):
                        try:
                            paths_data.append((a.get_painter_path(), a.label.color, a.transparency))
                        except Exception:
                            pass
                
                # Send the raw paths to the fast item
                self._base_image_item.set_readonly_annotations(paths_data)
                # Also check for per-frame cached mask overlay and set/clear it
                try:
                    cached = getattr(self, 'batch_results_cache', {}).get(self.current_image_path)
                    if cached and cached.get('mask_qimage') is not None:
                        try:
                            self._base_image_item.set_mask_image(cached.get('mask_qimage'), cached.get('opacity', 128 / 255.0))
                        except Exception:
                            pass
                    else:
                        try:
                            self._base_image_item.set_mask_image(None)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Also check for a per-frame cached mask overlay and set it (or clear)
                try:
                    cached = getattr(self, 'batch_results_cache', {}).get(self.current_image_path)
                    if cached and cached.get('mask_qimage') is not None:
                        try:
                            self._base_image_item.set_mask_image(cached.get('mask_qimage'), cached.get('opacity', 128 / 255.0))
                        except Exception:
                            pass
                    else:
                        try:
                            # No per-frame mask available: ensure overlay cleared
                            self._base_image_item.set_mask_image(None)
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        # ----------------------------------------

        # Update slider and counter silently (no seekChanged feedback loop)
        self._video_player.slider.blockSignals(True)
        self._video_player.slider.setValue(frame_idx)
        self._video_player.slider.blockSignals(False)
        self._video_player.lbl_frame.setText(f"{frame_idx} / {vr.frame_count}")

        # Clear the drop-frame gate so the worker sends the next frame
        if vr._decode_worker is not None:
            vr._decode_worker._pending_emit = False

    def _on_video_play(self):
        """Start the playback timer, clearing annotation graphics first."""
        if self._active_video_raster is None:
            return
        # Remove annotation graphics for the current frame from the scene.
        # The annotation data is preserved; graphics are rebuilt on pause.
        self._clear_current_frame_annotation_graphics()
        self.main_window.set_video_playback_tools_enabled(False)
        # Sync the worker to the current display position then start streaming frames
        self._active_video_raster.seek_decode_worker(self._current_frame_idx)
        self._active_video_raster.resume_decode_worker()

    def _clear_current_frame_annotation_graphics(self):
        """Remove annotation graphics items for the current frame from the scene.
        Annotation data is kept intact so they reload correctly on pause."""
        path = self.current_image_path
        if not path:
            return
        for annotation in list(self.image_annotations_dict.get(path, [])):
            try:
                # Remove the group from the scene first
                if (hasattr(annotation, 'graphics_item_group') and
                        annotation.graphics_item_group is not None):
                    try:
                        if annotation.graphics_item_group.scene():
                            annotation.graphics_item_group.scene().removeItem(
                                annotation.graphics_item_group
                            )
                    except RuntimeError:
                        pass  # C++ object already deleted

                # Null out ALL graphics item references so deselect() / delete()
                # don't try to operate on dangling C++ objects
                annotation.graphics_item_group = None
                annotation.graphics_item = None
                annotation.center_graphics_item = None
                annotation.bounding_box_graphics_item = None
                if hasattr(annotation, 'tag_item'):
                    annotation.tag_item = None
                if hasattr(annotation, 'dimension_tag_item'):
                    annotation.dimension_tag_item = None

                # Mark as deselected so unselect_annotations() skips the deselect path
                annotation.is_selected = False
            except Exception:
                pass

    def _on_video_pause(self):
        """Stop the decode worker and do a full frame redisplay with annotations."""
        self._playback_timer.stop()
        if self._active_video_raster is not None:
            self._active_video_raster.pause_decode_worker()
        self.main_window.set_video_playback_tools_enabled(True)
        self._display_video_frame(self._current_frame_idx)

    def _on_video_next(self):
        """Advance one frame."""
        if self._active_video_raster is None:
            return
        next_idx = min(self._current_frame_idx + 1, self._active_video_raster.frame_count - 1)
        self._display_video_frame(next_idx)

    def _on_video_prev(self):
        """Step back one frame."""
        if self._active_video_raster is None:
            return
        prev_idx = max(self._current_frame_idx - 1, 0)
        self._display_video_frame(prev_idx)

    def _on_video_next_annotated(self):
        """Jump to the nearest annotated frame after the current position."""
        if self._active_video_raster is None:
            return
        candidates = [f for f in self._get_annotated_frame_indices() if f > self._current_frame_idx]
        if candidates:
            self._display_video_frame(min(candidates))

    def _on_video_prev_annotated(self):
        """Jump to the nearest annotated frame before the current position."""
        if self._active_video_raster is None:
            return
        candidates = [f for f in self._get_annotated_frame_indices() if f < self._current_frame_idx]
        if candidates:
            self._display_video_frame(max(candidates))

    def _playback_tick(self):
        """
        Fast playback path: update the scene pixmap only — no annotation load.
        Annotations are only loaded when playback pauses.
        """
        if self._active_video_raster is None:
            self._playback_timer.stop()
            return

        next_idx = (self._current_frame_idx + 1) % self._active_video_raster.frame_count
        q_image = self._active_video_raster.get_frame(next_idx)
        if q_image is None:
            return

        self._current_frame_idx = next_idx
        self.current_image_path = self._active_video_raster.make_frame_path(
            self._active_video_raster.image_path, next_idx
        )

        # --- PHASE 4: FAST PLAYBACK RENDERING ---
        if self._base_image_item is not None:
            self._base_image_item.set_image(q_image)
            try:
                frame_annotations = self.image_annotations_dict.get(self.current_image_path, [])
                paths_data = []
                for a in frame_annotations:
                    if getattr(a.label, 'is_visible', True) and not hasattr(a, 'mask_data'):
                        try:
                            paths_data.append((a.get_painter_path(), a.label.color, a.transparency))
                        except Exception:
                            pass
                self._base_image_item.set_readonly_annotations(paths_data)
            except Exception:
                pass
        else:
            self.load_visuals(q_image, self.current_image_path, None)
        # ----------------------------------------

        # Update slider silently
        self._video_player.slider.blockSignals(True)
        self._video_player.slider.setValue(next_idx)
        self._video_player.slider.blockSignals(False)
        self._video_player.lbl_frame.setText(f"{next_idx} / {self._active_video_raster.frame_count}")

    def cursorInWindow(self, pos, mapped=False):
        """Check if the cursor position is within the image bounds."""
        if not pos or not self.pixmap_image:
            return False

        image_rect = QGraphicsPixmapItem(self.pixmap_image).boundingRect()
        if not mapped:
            pos = self.mapToScene(pos)

        return image_rect.contains(pos)

    def cursorInViewport(self, pos):
        """Check if the cursor position is within the viewport bounds."""
        if not pos:
            return False

        return self.viewport().rect().contains(pos)
    
    def get_selected_tool(self):
        """Get the currently selected tool."""
        return self.selected_tool

    def set_selected_tool(self, tool, preserve_selection=False):
        """Set the currently active tool and update the UI layers for the correct editing mode.
        
        Args:
            tool: The tool name to activate.
            preserve_selection: If True, existing selections will be preserved during tool switch.
                               Use this when switching to select tool with existing selections from viewers.
        """
        
        previous_tool = self.selected_tool
        
        if self.selected_tool:
            self.tools[self.selected_tool].stop_current_drawing()
            self.tools[self.selected_tool].deactivate()
            
        if tool is None or tool not in self.tools:
            self.selected_tool = None
            if not preserve_selection:
                self.unselect_annotations()
            return
        
        self.selected_tool = tool

        # --- OPTIMIZED LOGIC FOR MASK/VECTOR MODE SWITCHING (DO NOT CHANGE) ---
        # Determine if we are entering or leaving mask editing mode
        is_entering_mask_mode = self.selected_tool in self.mask_tools
        is_leaving_mask_mode = previous_tool in self.mask_tools

        # Transitioning from a vector tool to a mask tool: LOCK the vector annotations
        if is_entering_mask_mode and not is_leaving_mask_mode:
            self.rasterize_annotations()
        
        # Transitioning from a mask tool to a vector tool: UNLOCK the vector annotations
        elif is_leaving_mask_mode and not is_entering_mask_mode:
            self.unrasterize_annotations()

        # If we are transitioning between either mode, unselect annotations
        # (Mode switching always clears selection, even if preserve_selection=True)
        if is_entering_mask_mode or is_leaving_mask_mode:
            self.unselect_annotations()
        # --------------------------------------------------------
        
        if self.selected_tool:
            self.tools[self.selected_tool].activate()
        
        # Unselect annotations unless we are in select mode or preserve_selection is True
        if self.selected_tool != "select" and not preserve_selection:
            self.unselect_annotations()

        self.toggle_cursor_annotation()
        
    def set_selected_label(self, label):
        """Set the currently selected label and update selected annotations if needed."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        self.selected_label = label

        # Collect changes for action stack
        changes = []  # list of (annotation_id, old_label, new_label)

        # Handle both valid labels and None (no label selected)
        if label is not None:

            for annotation in self.selected_annotations:
                if annotation.label.id != label.id:
                    old_label = annotation.label
                    annotation.update_user_confidence(self.selected_label)
                    annotation.create_cropped_image(self.rasterio_image)
                    self.main_window.confidence_window.display_cropped_image(annotation)
                    changes.append((annotation.id, old_label, self.selected_label))

            if self.cursor_annotation:
                if self.cursor_annotation.label.id != label.id:
                    self.toggle_cursor_annotation()
        else:
            # Clear cursor annotation when no label is selected
            if self.cursor_annotation:
                self.toggle_cursor_annotation()

        # Record action(s)
        try:
            if changes:
                if len(changes) == 1:
                    ann_id, old_label, new_label = changes[0]
                    action = ChangeLabelAction(self, ann_id, old_label, new_label)
                    self.action_stack.push(action)
                    try:
                        self.annotationLabelChanged.emit(ann_id, new_label.id if hasattr(new_label, 'id') else str(new_label))
                    except Exception:
                        pass
                else:
                    action = ChangeLabelsAction(self, changes)
                    self.action_stack.push(action)
                    try:
                        self.annotationsLabelsChanged.emit(changes)
                    except Exception:
                        pass
        except Exception:
            pass
                
        # Make cursor normal again
        QApplication.restoreOverrideCursor()
        
    def set_annotation_scale(self, annotation, image_path=None):
        """
        Updates a single annotation's scale properties to match its raster.
        Uses the provided image_path if available, otherwise defaults to the
        path stored on the annotation object itself.
        """
        if not annotation:
            return
            
        # Determine the correct image path to use
        path_to_use = image_path if image_path is not None else annotation.image_path
            
        raster = self.main_window.image_window.raster_manager.get_raster(path_to_use)
        if raster:
            annotation.scale_x = raster.scale_x
            annotation.scale_y = raster.scale_y
            annotation.scale_units = raster.scale_units
        else:
            # Ensure scale is None if raster isn't found
            annotation.scale_x = None
            annotation.scale_y = None
            annotation.scale_units = None

    def set_annotations_scale(self, image_path):
        """
        Updates the scale properties of all annotations associated with a specific
        image path by calling set_annotation_scale on each one.
        """
        annotations = self.get_image_annotations(image_path)
        if not annotations:
            return

        # Loop through all annotations for this image and sync their scale
        for annotation in annotations:
            # Pass the image_path for efficiency
            self.set_annotation_scale(annotation, image_path=image_path)
            
    def set_annotation_location(self, annotation_id, new_center_xy: QPointF):
        """Update the location of an annotation to a new center point."""
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            try:
                # Disconnect the confidence window from the annotation, so it won't update while moving
                annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                annotation.annotationUpdated.disconnect(self.on_annotation_updated)
            except Exception:
                pass  # Ignore if not connected
            
            annotation.update_location(new_center_xy)
            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
            annotation.annotationUpdated.connect(self.on_annotation_updated)
            # Display the cropped image in the confidence window
            self.main_window.confidence_window.display_cropped_image(annotation)

    def set_annotation_size(self, size=None, delta=0):
        """Set or adjust the size of the current annotation(s)."""
        if size is not None:
            self.annotation_size = size
        else:
            self.annotation_size += delta
            self.annotation_size = max(1, self.annotation_size)

        # Cursor or 1 annotation selected
        if len(self.selected_annotations) == 1:
            annotation = self.selected_annotations[0]
            if not self.is_annotation_moveable(annotation):
                return
            
            # Disconnect the confidence window from the annotation, so it won't update while resizing
            try:
                annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                annotation.annotationUpdated.disconnect(self.on_annotation_updated)
            except Exception:
                pass

            # Record previous state for undo/redo
            if isinstance(annotation, PatchAnnotation):
                old_size = getattr(annotation, 'annotation_size', None)
                annotation.update_annotation_size(self.annotation_size)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(self.annotation_size)
                new_size = getattr(annotation, 'annotation_size', None)
                # Push a resize action; ActionStack will coalesce consecutive resizes
                if old_size is not None and new_size is not None and old_size != new_size:
                    try:
                        action = ResizeAnnotationAction(self, annotation.id, old_size, new_size)
                        self.action_stack.push(action)
                    except Exception:
                        pass
            elif isinstance(annotation, RectangleAnnotation):
                scale_factor = 1 + delta / 100.0
                # Capture old geometry (top_left, bottom_right)
                try:
                    old_tl = QPointF(annotation.top_left.x(), annotation.top_left.y())
                    old_br = QPointF(annotation.bottom_right.x(), annotation.bottom_right.y())
                    old_geom = (old_tl, old_br)
                except Exception:
                    old_geom = None

                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)

                # Capture new geometry and push geometry-edit action
                try:
                    new_tl = QPointF(annotation.top_left.x(), annotation.top_left.y())
                    new_br = QPointF(annotation.bottom_right.x(), annotation.bottom_right.y())
                    new_geom = (new_tl, new_br)
                except Exception:
                    new_geom = None

                if old_geom is not None and new_geom is not None and old_geom != new_geom:
                    try:
                        action = AnnotationGeometryEditAction(self, annotation.id, old_geom, new_geom)
                        self.action_stack.push(action)
                    except Exception:
                        pass
            elif isinstance(annotation, PolygonAnnotation):
                scale_factor = 1 + delta / 100.0
                # Capture old polygon points and holes
                try:
                    pts = [QPointF(p.x(), p.y()) for p in annotation.points]
                    holes = []
                    if hasattr(annotation, 'holes') and annotation.holes:
                        for hole in annotation.holes:
                            holes.append([QPointF(p.x(), p.y()) for p in hole])
                    old_geom = (pts, holes)
                except Exception:
                    old_geom = None

                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)

                # Capture new polygon geometry
                try:
                    pts = [QPointF(p.x(), p.y()) for p in annotation.points]
                    holes = []
                    if hasattr(annotation, 'holes') and annotation.holes:
                        for hole in annotation.holes:
                            holes.append([QPointF(p.x(), p.y()) for p in hole])
                    new_geom = (pts, holes)
                except Exception:
                    new_geom = None

                if old_geom is not None and new_geom is not None and old_geom != new_geom:
                    try:
                        action = AnnotationGeometryEditAction(self, annotation.id, old_geom, new_geom)
                        self.action_stack.push(action)
                    except Exception:
                        pass

            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
            annotation.annotationUpdated.connect(self.on_annotation_updated)
            # Display the cropped image in the confidence window
            self.main_window.confidence_window.display_cropped_image(annotation)

        # Only emit if 1 or no annotations are selected
        if len(self.selected_annotations) <= 1:
            # Emit that the annotation size has changed
            self.annotationSizeChanged.emit(self.annotation_size)
            
    def set_annotation_visibility(self, annotation, force_visibility=None):
        """Set the visibility of an annotation and update its graphics item based on its label's visibility.
        
        Args:
            annotation: The annotation to update
            force_visibility: If provided, force this visibility state regardless of label checkbox.
                            If None, use the label's visibility checkbox state.
        """
        # Determine visibility based on force_visibility or the label's visibility checkbox state
        if force_visibility is not None:
            visible = force_visibility
        else:
            visible = annotation.label.is_visible
        
        # Always update transparency for vector annotations (regardless of visibility)
        if not hasattr(annotation, 'mask_data'):  # Vector annotations only
            slider_value = self.main_window.get_transparency_value()
            annotation.update_transparency(slider_value)
        
        # Set visibility state
        if visible:
            # Show the annotation
            annotation.set_visibility(True)
            # Note: Mask annotations handle visibility through update_visible_labels() method
        else:
            # Hide the annotation (but transparency is already updated above)
            annotation.set_visibility(False)
                
    def set_label_visibility(self, visible):
        """Set the visibility for all labels."""
        # Block signals for batch update
        self.blockSignals(True)
        try:
            # Handle vector annotations
            for annotation in self.annotations_dict.values():
                self.set_annotation_visibility(annotation, force_visibility=visible)
            
            # Handle mask annotation visibility - synchronize with vector annotations
            mask = self.current_mask_annotation
            if mask:
                if visible:
                    # Show mask by making all visible labels visible
                    visible_labels = self.main_window.label_window.get_visible_labels()
                    visible_label_ids = {label.id for label in visible_labels}
                    mask.update_visible_labels(visible_label_ids) 
                else:
                    # Hide mask by clearing all visible labels
                    mask.update_visible_labels(set())
        finally:
            self.blockSignals(False)
    
        # PHANTOM ARCHITECTURE: Re-render phantom layer with visibility changes
        self.refresh_phantom_annotations()
        
        self.scene.update()
        self.viewport().update()
        
    def is_annotation_moveable(self, annotation):
        """Check if an annotation can be moved and show a warning if not."""
        if annotation.show_message:
            self.unselect_annotations()
            annotation.show_warning_message()
            return False
        return True

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Toggle cursor annotation visibility by delegating to the active tool.
        
        This method serves as a bridge between annotation window events and tool-specific
        cursor annotation handling.
        
        Args:
            scene_pos: Position in scene coordinates. If provided, creates/updates
                      cursor annotation at this position. If None, clears the annotation.
        """
        if self.selected_tool and self.active_image and self.selected_label:
            if scene_pos:
                self.tools[self.selected_tool].update_cursor_annotation(scene_pos)
            else:
                self.tools[self.selected_tool].clear_cursor_annotation()
        
        # Clear our reference to any cursor annotation
        self.cursor_annotation = None
        
    def update_scene(self):
        """Update the graphics scene and its items."""
        self.scene.update()
        self.viewport().update()
        QApplication.processEvents()

    def _show_placeholder(self, text: str = None):
        """Show the centered placeholder label with optional custom text."""
        try:
            if text:
                self._placeholder_label.setText(text)
            self._placeholder_label.setGeometry(self.viewport().rect())
            self._placeholder_label.show()
        except Exception:
            pass

    def _hide_placeholder(self):
        """Hide the placeholder label."""
        try:
            self._placeholder_label.hide()
        except Exception:
            pass
            
    def clear_scene(self):
        """
        Clear the scene with AnnotationWindow-specific cleanup.
        Delegates to BaseCanvas.clear_scene() which will call _on_scene_cleared() at the end.
        """
        # AnnotationWindow-specific cleanup before base clear
        self.unselect_annotations()
        
        # Nullify graphics_item references for all annotations
        for annotation in self.annotations_dict.values():
            if hasattr(annotation, 'graphics_item'):
                annotation.graphics_item = None
        
        # Disconnect z-channel signal from previously displayed raster
        try:
            if self.current_image_path:
                prev_raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
                if prev_raster is not None:
                    try:
                        prev_raster.zChannelChanged.disconnect(self.refresh_z_channel_visualization)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Call BaseCanvas clear_scene which will handle scene cleanup and call _on_scene_cleared hook
        super().clear_scene()
    
    def _on_scene_cleared(self):
        """Hook called by BaseCanvas after scene is cleared. Handles AnnotationWindow-specific cleanup."""
        # Allow BaseCanvas to re-create its markers and other scene-level items
        try:
            super()._on_scene_cleared()
        except Exception:
            pass
    
    def reset_scene_view(self):
        """Resets the scene view, rotation, and 3D perspective (if MVAT is active)"""
        # Fit the image to the view and recalculate zoom constraints
        self.fit_to_image()
        # Reset rotation to default
        self.rotation_angle = 0.0
        self._set_absolute_rotation(self.rotation_angle )  # Apply the rotation transform reset
        self.viewChanged.emit(*self.get_image_dimensions())
        
        # If MVAT viewer is active, sync 3D view to current image's perspective
        if hasattr(self.main_window, 'mvat_manager') and self.main_window.mvat_manager:
            mvat_manager = self.main_window.mvat_manager
            # Find and select the camera corresponding to the current image
            if self.current_image_path:
                for camera in mvat_manager.cameras.values():
                    if camera.image_path == self.current_image_path:
                        mvat_manager.viewer.match_camera_perspective(camera, animate=True)
                        break
        
        # Process events
        QApplication.processEvents()

    def display_image(self, q_image):
        """Display a QImage in the annotation window without setting it."""
        # Clean up
        self.clear_scene()

        # Hide placeholder since we will display an image
        self._hide_placeholder()

        # Display NaN values the image dimensions in status bar
        self.imageLoaded.emit(0, 0)
        self.viewChanged.emit(0, 0)

        # Set the image representations
        self.pixmap_image = QPixmap(q_image)
        self.scene.addItem(QGraphicsPixmapItem(self.pixmap_image))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        QApplication.processEvents()

    def set_image(self, image_path):
        """Set and display an image at the given path using a staged load for instant feedback."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Stop any current drawing operation before switching images
        if self.selected_tool and self.selected_tool in self.tools:
            self.tools[self.selected_tool].stop_current_drawing()
            if self.selected_tool in ["scale"]:
                self.main_window.untoggle_all_tools()

        # ---- VIDEO BRANCH ----
        # Resolve virtual frame paths to the underlying video path for the raster lookup
        lookup_path = image_path
        if '::frame_' in image_path:
            lookup_path = image_path.rsplit('::frame_', 1)[0]

        raster_check = self.main_window.image_window.raster_manager.get_raster(lookup_path)

        # Import here to avoid circular imports at module level
        try:
            from coralnet_toolbox.Rasters.VideoRaster import VideoRaster as _VideoRaster
            _video_raster_cls = _VideoRaster
        except ImportError:
            _video_raster_cls = None

        if _video_raster_cls is not None and isinstance(raster_check, _video_raster_cls):
            QApplication.restoreOverrideCursor()
            self._activate_video_mode(raster_check)
            return
        else:
            # Deactivate video mode if we're switching to a regular image
            self._deactivate_video_mode()
        # ---- END VIDEO BRANCH ----
                    
        # Clean up (This is the ONLY scene clear)
        self.clear_scene()

        # Clear the action stack
        self.action_stack.undo_stack.clear()
        self.action_stack.redo_stack.clear()

        # Check that the image path is valid
        if image_path not in self.main_window.image_window.raster_manager.image_paths:
            QApplication.restoreOverrideCursor()
            return

        # Get the raster
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if not raster:
            QApplication.restoreOverrideCursor()
            return
        
        # Load z_channel data if available (deferred loading)
        if raster.z_channel_path and raster.z_channel is None:
            try:
                raster.load_z_channel_from_file(raster.z_channel_path)
            except Exception:
                # Z-channel loading failure is non-critical; proceed without it
                pass

        # Connect raster's zChannelChanged to refresh visualization for live updates
        try:
            # Disconnect previous if exists to avoid duplicate connections
            if hasattr(self, 'current_image_path') and self.current_image_path:
                prev_raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
                if prev_raster is not None:
                    try:
                        prev_raster.zChannelChanged.disconnect(self.refresh_z_channel_visualization)
                    except Exception:
                        pass
            raster.zChannelChanged.connect(self.refresh_z_channel_visualization)
        except Exception:
            pass

        # Get low-res thumbnail first for a preview
        low_res_qimage = raster.get_thumbnail(longest_edge=256)
        if low_res_qimage is None or low_res_qimage.isNull():
            # If thumbnail fails, just exit
            self.main_window.image_window.show_error(
                "Image Loading Error",
                f"Image {os.path.basename(image_path)} thumbnail could not be loaded."
            )
            QApplication.restoreOverrideCursor()
            return
        
        # Display low-res thumbnail for quick preview
        # (This step is before load_visuals to show preview immediately)
        self._show_placeholder()  # Start with placeholder
        low_res_pixmap = QPixmap.fromImage(low_res_qimage)
        base_image_item = QGraphicsPixmapItem(low_res_pixmap)
        base_image_item.setZValue(-10)
        self.scene.addItem(base_image_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._hide_placeholder()
        QApplication.processEvents()
        
        # Update the rasterio image source for cropping annotations
        self.rasterio_image = raster.rasterio_src
        
        # Get full-resolution QImage
        q_image = raster.get_qimage()
        if q_image is None or q_image.isNull():
            self.main_window.image_window.show_error(
                "Image Loading Error",
                f"Image {os.path.basename(image_path)} full resolution could not be loaded."
            )
            QApplication.restoreOverrideCursor()
            return  # Failed to load full res, but preview is still visible
        
        # Use BaseCanvas canonical loader for the full-resolution image (preserves base logic)
        # Update the rasterio image reference used by annotation scaling/cropping
        self.rasterio_image = raster.rasterio_src
        # Load visuals into the BaseCanvas (this clears the preview and installs full-res image)
        self.load_visuals(q_image, image_path, raster)

        # Apply the current colormap selection and preserve UI opacity
        current_colormap = self.main_window.z_colormap_dropdown.currentText()
        if current_colormap != "None":
            self.update_z_colormap(current_colormap)
        if self.z_item is not None:
            try:
                current_opacity = self.main_window.z_transparency_widget.value() / 255.0
                self.z_item.setOpacity(current_opacity)
            except Exception:
                pass
        
        # Sync z-channel state to all ContextMatrix BaseCanvases when image changes
        try:
            context_matrix = getattr(self.main_window, 'context_matrix', None)
            if context_matrix:
                if current_colormap != "None":
                    context_matrix.sync_z_colormap_to_all_canvases(current_colormap)
                    current_opacity = self.main_window.z_transparency_widget.value() / 255.0
                    context_matrix.sync_z_opacity_to_all_canvases(current_opacity)
                # Also sync dynamic scaling state
                is_dynamic_enabled = self.main_window.z_dynamic_scaling_checkbox.isChecked()
                context_matrix.sync_z_dynamic_scaling_to_all_canvases(is_dynamic_enabled)
        except Exception:
            pass

        # Automatically mark this image as checked when viewed
        raster.checkbox_state = True
        self.main_window.image_window.table_model.update_raster_data(image_path)

        # Toggle the cursor annotation
        self.toggle_cursor_annotation()

        # Re-fit the view to the new, full-res pixmap
        self.reset_scene_view()

        # Load all associated annotations
        self.load_annotations()
        # Update the image window's image annotations
        self.main_window.image_window.update_image_annotations(image_path)
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

        QApplication.processEvents()

        # Set the image dimensions, and current view in status bar
        self.imageLoaded.emit(self.pixmap_image.width(), self.pixmap_image.height())
        self.viewChanged.emit(self.pixmap_image.width(), self.pixmap_image.height())
        
        # Show loaded message in status bar (centralized here per MVAT convention)
        try:
            self.main_window.status_bar.showMessage(f"Loaded image: {os.path.basename(image_path)}", 2000)
        except Exception:
            pass

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def _load_z_channel_visualization(self, raster):
        """Override to set opacity from main_window widget after BaseCanvas loads."""
        super()._load_z_channel_visualization(raster)
        
        # Set opacity from current slider value (preserves user's transparency preference)
        if self.z_item is not None:
            try:
                current_opacity = self.main_window.z_transparency_widget.value() / 255.0
                self.z_item.setOpacity(current_opacity)
            except Exception:
                pass

    def update_z_colormap(self, colormap_name):
        # Delegate to BaseCanvas and keep opacity in sync with the UI widget
        super().update_z_colormap(colormap_name)
        if self.z_item is not None:
            try:
                self.z_item.setOpacity(self.main_window.z_transparency_widget.value() / 255.0)
            except Exception:
                pass

    def _reset_z_channel_to_full_range(self, colormap_name=None):
        """Override to fetch colormap from main_window and pass to BaseCanvas."""
        if colormap_name is None:
            colormap_name = self.main_window.z_colormap_dropdown.currentText()
        super()._reset_z_channel_to_full_range(colormap_name)
    
    def update_current_image_path(self, image_path):
        """Update the current image path being displayed.

        For video rasters, do not override the virtual per-frame `current_image_path`
        already set by `_display_video_frame`. The `ImageWindow` emits a raw
        video path after calling `set_image`, so ignore that emission when
        we're in active video mode to avoid losing the `::frame_N` suffix.
        """
        # If we're currently in video mode and the annotation window already has
        # a per-frame virtual path, don't override it with the raw video path.
        if getattr(self, '_active_video_raster', None) is not None:
            if hasattr(self, 'current_image_path') and self.current_image_path and '::frame_' in str(self.current_image_path):
                return

        self.current_image_path = image_path
        
    def update_mask_label_map(self):
        """Update the label_map in the current MaskAnnotation to reflect changes in LabelWindow."""
        if self.current_mask_annotation:
            # Call the new sync method instead of just overwriting the map.
            all_current_labels = self.main_window.label_window.labels
            self.current_mask_annotation.sync_label_map(all_current_labels)
    
    def refresh_z_channel_visualization(self):
        """
        Refresh the Z-channel visualization if it's available for the current image.
        This is called when a z-channel is newly imported for the currently displayed image.
        """
        if self.current_image_path:
            raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
            if raster and raster.z_channel is not None:
                # Reload the z-channel visualization
                self._load_z_channel_visualization(raster)
                
                # Apply the current colormap selection to the newly loaded z_item
                current_colormap = self.main_window.z_colormap_dropdown.currentText()
                # Always call update_z_colormap to handle visibility correctly
                self.update_z_colormap(current_colormap)
                
                # Force scene update to ensure visual changes are immediately reflected
                self.scene.update()
                self.viewport().update()
    
    @property
    def current_mask_annotation(self) -> Optional[MaskAnnotation]:
        """A helper property to get the MaskAnnotation for the currently active image."""
        if not self.current_image_path:
            return None
        raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
        if not raster:
            return None
        
        # This will get the existing mask or create it on the first call
        is_new = raster.mask_annotation is None
        project_labels = self.main_window.label_window.labels
        mask_annotation = raster.get_mask_annotation(project_labels)
        if is_new:
            self.main_window.status_bar.showMessage(
                f"Creating mask annotation for {os.path.basename(self.current_image_path)}…", 3000
            )

        return mask_annotation

    def rasterize_annotations(self):
        """
        Mark vector annotation pixels as protected (locked) to prevent painting over them.
        Vector annotations remain visible, but their pixel locations become off-limits for mask editing.
        This provides pixel-level protection without expensive visual operations.
        """
        if not self.current_mask_annotation:
            return

        annotations = self.get_image_annotations()
        if not annotations:
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
            
        # The MaskAnnotation handles the efficient protection marking internally
        self.current_mask_annotation.rasterize_annotations(annotations)

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def unrasterize_annotations(self):
        """
        Remove protection from vector annotation pixels, allowing mask editing over those areas again.
        This clears the locked status from pixels that were protected during mask editing mode.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        if self.current_mask_annotation:
            self.current_mask_annotation.unrasterize_annotations()
            
        # Restore cursor
        QApplication.restoreOverrideCursor()

    def viewportToScene(self):
        """Convert viewport coordinates to scene coordinates."""
        # Map the top-left and bottom-right corners of the viewport to the scene coordinates
        top_left = self.mapToScene(self.viewport().rect().topLeft())
        bottom_right = self.mapToScene(self.viewport().rect().bottomRight())
        # Create and return a QRectF object from these points
        return QRectF(top_left, bottom_right)

    def animate_to_rect(self, target_rect: QRectF, duration: int = 500, max_zoom: float = 4.0):
        """Smoothly animate the view center and zoom to fit `target_rect`.

        This avoids instant jumps and provides a brief inertia-like transition.
        """
        if target_rect is None or target_rect.isNull():
            return

        # View geometry
        view_rect = self.viewport().rect()
        view_w = max(1.0, float(view_rect.width()))
        view_h = max(1.0, float(view_rect.height()))

        # Compute desired zoom to fit the target rect (KeepAspectRatio behaviour)
        tw = max(1.0, float(target_rect.width()))
        th = max(1.0, float(target_rect.height()))
        desired_zoom = min(view_w / tw, view_h / th)
        desired_zoom = min(desired_zoom, max_zoom)

        # Current center and zoom
        start_center = self.mapToScene(self.viewport().rect().center())
        end_center = target_rect.center()
        try:
            start_zoom = float(self.transform().m11())
        except Exception:
            start_zoom = 1.0

        # Prepare animator object and animations
        animator = ViewAnimator(self)
        animator._center_x = start_center.x()
        animator._center_y = start_center.y()
        animator._zoom = start_zoom

        cx_anim = QPropertyAnimation(animator, b'center_x')
        cx_anim.setStartValue(start_center.x())
        cx_anim.setEndValue(end_center.x())
        cx_anim.setDuration(duration)
        cx_anim.setEasingCurve(QEasingCurve.OutCubic)

        cy_anim = QPropertyAnimation(animator, b'center_y')
        cy_anim.setStartValue(start_center.y())
        cy_anim.setEndValue(end_center.y())
        cy_anim.setDuration(duration)
        cy_anim.setEasingCurve(QEasingCurve.OutCubic)

        z_anim = QPropertyAnimation(animator, b'zoom')
        z_anim.setStartValue(start_zoom)
        z_anim.setEndValue(desired_zoom)
        z_anim.setDuration(duration)
        z_anim.setEasingCurve(QEasingCurve.OutCubic)

        # Keep references to animations so they are not garbage collected
        self._active_view_animations = [cx_anim, cy_anim, z_anim]

        # Restore anchor back to previous behavior when animations finish
        def _on_finished():
            try:
                self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            except Exception:
                pass
            # Clean up references
            self._active_view_animations = []
            # Emit a viewChanged signal for status updates
            try:
                self.viewChanged.emit(*self.get_image_dimensions())
            except Exception:
                pass

        # Connect last animation finished to cleanup
        z_anim.finished.connect(_on_finished)

        # Start animations
        cx_anim.start()
        cy_anim.start()
        z_anim.start()

    def get_image_dimensions(self):
        """Get the dimensions of the currently loaded image."""
        if self.pixmap_image:
            return self.pixmap_image.size().width(), self.pixmap_image.size().height()
        return 0, 0
    
    def get_image_rect(self):
        """Get the bounding rectangle of the currently loaded image in scene coordinates."""
        if self.pixmap_image:
            return QRectF(0, 0, self.pixmap_image.width(), self.pixmap_image.height())
        return QRectF()
    
    def center_on_work_area(self, work_area):
        """Center the view on the specified work area."""
        # Create graphics item if it doesn't exist
        if not work_area.graphics_item:
            work_area.create_graphics_item(self.scene)

        # Get the bounding rect of the work area in scene coordinates
        work_area_rect = work_area.graphics_item.boundingRect()
        work_area_center = work_area_rect.center()

        # Smoothly animate the view to the work area
        self.animate_to_rect(work_area_rect)

    def center_on_annotation(self, annotation):
        """Center the view on the specified annotation."""
        # Create graphics item if it doesn't exist
        if not annotation.graphics_item:
            annotation.create_graphics_item(self.scene)

        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()
        annotation_center = annotation_rect.center()

        # Smoothly animate the view to the annotation
        self.animate_to_rect(annotation_rect)
    
    def center_and_zoom_on_annotation(self, annotation):
        """Center and zoom in to focus on the specified annotation with relaxed zoom and dynamic padding."""
        # Create graphics item if it doesn't exist
        if not annotation.graphics_item:
            annotation.create_graphics_item(self.scene)

        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()

        # Step 1: Calculate annotation and image area
        annotation_area = annotation_rect.width() * annotation_rect.height()
        if self.pixmap_image:
            image_width = self.pixmap_image.width()
            image_height = self.pixmap_image.height()
        else:
            # Fallback to scene rect if image not loaded
            image_width = self.scene.sceneRect().width()
            image_height = self.scene.sceneRect().height()
        image_area = image_width * image_height

        # Step 2: Compute the relative area ratio (avoid division by zero)
        if image_area > 0:
            relative_area = annotation_area / image_area
        else:
            relative_area = 1.0  # fallback, treat as full image

        # Step 3: Map ratio to padding factor (smaller annotation = more padding)
        import math
        min_padding = 0.15  # 15% (relaxed from 10%)
        max_padding = 0.35  # 35% (relaxed from 50%)
        if relative_area > 0:
            padding_factor = max(min(0.35 * (1 / math.sqrt(relative_area)), max_padding), min_padding)
        else:
            padding_factor = min_padding

        # Step 4: Apply dynamic padding with minimum values to prevent zero width/height
        min_padding_absolute = 2.0  # Minimum padding in pixels (relaxed from 1.0)
        padding_x = max(annotation_rect.width() * padding_factor, min_padding_absolute)
        padding_y = max(annotation_rect.height() * padding_factor, min_padding_absolute)
        padded_rect = annotation_rect.adjusted(-padding_x, -padding_y, padding_x, padding_y)

        # Animate the view to the padded annotation rect instead of jumping
        self.animate_to_rect(padded_rect, duration=600)
    
    def cycle_annotations(self, direction, _bypass_debounce: bool = False):
        """Cycle through annotations in the specified direction."""
        # Debounce: coalesce rapid repeated calls into one action
        debounce_ms = 150
        if not _bypass_debounce:
            # Lazy-create a single-shot timer to coalesce repeated calls
            if not hasattr(self, '_cycle_debounce_timer'):
                self._cycle_debounce_timer = QTimer(self)
                self._cycle_debounce_timer.setSingleShot(True)
                # When timer fires, call the cycle with bypass flag to actually run
                self._cycle_debounce_timer.timeout.connect(lambda: self.cycle_annotations(getattr(self, '_pending_cycle', 0), True))
            # Store the most recent requested direction
            self._pending_cycle = direction
            # Restart debounce timer
            self._cycle_debounce_timer.stop()
            self._cycle_debounce_timer.start(debounce_ms)
            return

        # When bypassing debounce, clear pending marker
        if hasattr(self, '_pending_cycle'):
            try:
                del self._pending_cycle
            except Exception:
                self._pending_cycle = None

        # Get the annotations for the current image
        annotations = self.get_image_annotations()
        if not annotations:
            return

        if self.selected_tool == "select" and self.active_image:
            # If label is locked, only cycle through annotations with that label
            if self.main_window.label_window.label_locked:
                locked_label = self.main_window.label_window.locked_label
                indices = [i for i, a in enumerate(annotations) if a.label.id == locked_label.id]

                if not indices:
                    return

                if self.selected_annotations:
                    current_index = annotations.index(self.selected_annotations[0])
                else:
                    current_index = indices[0] if indices else 0

                if current_index in indices:
                    # Find position in indices list and cycle within that
                    current_pos = indices.index(current_index)
                    new_pos = (current_pos + direction) % len(indices)
                    new_index = indices[new_pos]  # Get the actual annotation index
                else:
                    # Find next valid index based on direction
                    if direction > 0:
                        next_indices = [i for i in indices if i > current_index]
                        new_index = next_indices[0] if next_indices else indices[0]
                    else:
                        prev_indices = [i for i in indices if i < current_index]
                        new_index = prev_indices[-1] if prev_indices else indices[-1]

            elif self.selected_annotations:
                # Cycle through all the annotations
                current_index = annotations.index(self.selected_annotations[0])
                new_index = (current_index + direction) % len(annotations)
            else:
                # Select the first annotation if direction is positive, last if negative
                new_index = 0 if direction > 0 else len(annotations) - 1

            if 0 <= new_index < len(annotations):
                # Select the new annotation
                self.select_annotation(annotations[new_index])
                # Center the view on the new annotation
                self.center_on_annotation(annotations[new_index])
                
    def get_selected_annotation_type(self):
        """Get the type of the currently selected annotation."""
        if len(self.selected_annotations) == 1:
            return type(self.selected_annotations[0])
        return None

    def select_annotation(self, annotation, multi_select=False, quiet_mode=False, bulk_mode=False):
        """Select an annotation and update the UI accordingly."""
        if annotation in self.selected_annotations and multi_select:
            self.unselect_annotation(annotation, bulk_mode=bulk_mode)
            return
        
        if not multi_select:
            self.unselect_annotations()
            
        if annotation not in self.selected_annotations:
            self.selected_annotations.append(annotation)

            # First, mark the annotation selected so create_graphics_item
            # will not early-return due to the Phantom Gatekeeper.
            annotation.select()

            # PHANTOM ARCHITECTURE: Build the Qt objects if they don't exist
            if not annotation.is_graphics_item_valid():
                annotation.create_graphics_item(self.scene)
            self.selected_label = annotation.label
            self.annotationSelected.emit(annotation.id)
            
            if len(self.selected_annotations) == 1 and not quiet_mode:
                self.labelSelected.emit(annotation.label.id)
                if not annotation.cropped_image:
                    annotation.create_cropped_image(self.rasterio_image)
                annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
                annotation.annotationUpdated.connect(self.on_annotation_updated)
                self.main_window.confidence_window.display_cropped_image(annotation)
        
        self.set_annotation_visibility(annotation)
        
        # --- BULK MODE CHECK ---
        # Skip these heavy UI operations if we are looping through hundreds of items
        if not bulk_mode:
            if len(self.selected_annotations) > 1 and not quiet_mode:
                self.main_window.label_window.deselect_active_label()
                self.main_window.confidence_window.clear_display()
            self.viewport().update()
            # PHANTOM ARCHITECTURE: Re-render phantom layer to remove this annotation from it
            self.refresh_phantom_annotations()
            self._emit_selection_changed()

    def select_annotations(self):
        """Select all annotations in the current image (Optimized for Bulk)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.unselect_annotations()
        
        annotations = self.get_image_annotations()
        label_locked = self.main_window.label_window.label_locked
        locked_label_id = self.main_window.label_window.locked_label.id if label_locked else None
        
        # Turn on signal blocking for selection
        self._syncing_selection = True
        
        # --- Disable BSP Indexing ---
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        
        for annotation in annotations:
            if label_locked and annotation.label.id != locked_label_id:
                continue
            # Pass bulk_mode=True to prevent viewport repaints on every item
            self.select_annotation(annotation, multi_select=True, bulk_mode=True)

        # --- Restore BSP Indexing ---
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        self._syncing_selection = False
        
        # Perform the UI updates EXACTLY ONCE at the end
        if len(self.selected_annotations) > 1:
            self.main_window.label_window.deselect_active_label()
            self.main_window.confidence_window.clear_display()
            
        self.viewport().update()
        self._emit_selection_changed()
        QApplication.restoreOverrideCursor()

    def select_annotations_by_ids(self, annotation_ids, scroll_to_first=True, quiet_mode=True):
        """Select a batch of annotations by their IDs."""
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Prevent selection feedback loops BEFORE clearing the existing selection
        self._syncing_selection = True

        # Clear existing selection first
        self.unselect_annotations()

        # --- Correctly handle empty selections by clearing the canvas ---
        if not annotation_ids:
            self._syncing_selection = False
            self.viewport().update()
            self._emit_selection_changed()
            QApplication.restoreOverrideCursor()
            return
        # -----------------------------------------------------------------------

        annotations_dict = getattr(self, 'annotations_dict', {})

        # Disable BSP indexing and block signals for speed
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.blockSignals(True)

        first_selected = None
        for ann_id in annotation_ids:
            ann = annotations_dict.get(ann_id)
            if not ann:
                continue
            
            # --- Only select annotations belonging to the current image! ---
            if ann.image_path != self.current_image_path:
                continue
            # ----------------------------------------------------------------------

            if first_selected is None:
                first_selected = ann
            # Use bulk_mode to avoid per-item heavy UI updates
            self.select_annotation(ann, multi_select=True, quiet_mode=quiet_mode, bulk_mode=True)

        # Restore indexing and signals
        self.blockSignals(False)
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        self._syncing_selection = False

        # One consolidated UI update
        if len(self.selected_annotations) > 1:
            self.main_window.label_window.deselect_active_label()
            self.main_window.confidence_window.clear_display()

        # Optionally center/scroll to the first selected item
        try:
            if first_selected and scroll_to_first:
                self.center_on_annotation(first_selected)
        except Exception:
            pass

        self.viewport().update()
        self._emit_selection_changed()
        QApplication.restoreOverrideCursor()

    def unselect_annotation(self, annotation, bulk_mode=False):
        """Unselect a specific annotation."""
        if annotation in self.selected_annotations:
            self.selected_annotations.remove(annotation)
            
            if hasattr(annotation, 'annotationUpdated') and self.main_window.confidence_window.isVisible():
                try: 
                    annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                except TypeError: 
                    pass
                try: 
                    annotation.annotationUpdated.disconnect(self.on_annotation_updated)
                except TypeError: 
                    pass
            
            # PHANTOM ARCHITECTURE: Destroy Qt objects BEFORE deselect() so the group's
            # children (center, bbox, tag) are still attached and removed cleanly together,
            # preventing orphaned items being left in the scene.
            self._clear_annotation_graphics_single(annotation)
            annotation.deselect()
            
            # --- BULK MODE CHECK ---
            if not bulk_mode:
                if not self.selected_annotations:
                    self.main_window.confidence_window.clear_display()
                self.viewport().update()
                # PHANTOM ARCHITECTURE: Re-render phantom layer to draw this annotation in it
                self.refresh_phantom_annotations()
                self._emit_selection_changed()

    def unselect_annotations(self):
        """Unselect all currently selected annotations."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # --- Disable BSP indexing ---
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        
        # Create a copy to safely iterate through
        annotations_to_unselect = self.selected_annotations.copy()
        
        # Clear the list first to avoid modification during iteration
        self.selected_annotations = []
        
        for annotation in annotations_to_unselect:
            # Disconnect from confidence window if needed
            if hasattr(annotation, 'annotationUpdated') and self.main_window.confidence_window.isVisible():
                try:
                    annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                except TypeError:
                    pass
                try:
                    annotation.annotationUpdated.disconnect(self.on_annotation_updated)
                except TypeError:
                    pass
            
            # PHANTOM ARCHITECTURE: Destroy Qt objects BEFORE deselect() so the group's
            # children (center, bbox, tag) are still attached and removed cleanly together,
            # preventing orphaned items being left in the scene.
            self._clear_annotation_graphics_single(annotation)
            # Update annotation's internal state
            annotation.deselect()
            
        # --- NEW: Restore BSP indexing ---
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        
        # PHANTOM ARCHITECTURE: Re-render phantom layer with all now-deselected annotations
        self.refresh_phantom_annotations()
        
        # Update the viewport once for all changes
        self.viewport().update()
        
        # Make cursor normal again
        QApplication.restoreOverrideCursor()
        
        # Emit selection changed signal
        self._emit_selection_changed()
    
    def _emit_selection_changed(self):
        """Emit the annotationSelectionChanged signal with current selection IDs."""
        if self._syncing_selection:
            return  # Prevent infinite loops
        selected_ids = [ann.id for ann in self.selected_annotations]
        self.annotationSelectionChanged.emit(selected_ids)

    def load_annotation(self, annotation):
        """Load a single annotation into the scene."""
        # Set the animation manager
        annotation.set_animation_manager(self.animation_manager)
        
        # Inject / update scale
        self.set_annotation_scale(annotation)
        
        # Remove the graphics item from its current scene if it exists
        if annotation.graphics_item and annotation.graphics_item.scene():
            annotation.graphics_item.scene().removeItem(annotation.graphics_item)

        # Update transparency to match the global slider value
        current_slider_value = self.main_window.get_transparency_value()
        annotation.update_transparency(current_slider_value)

        # PHANTOM ARCHITECTURE: Only create graphics items for selected annotations
        # Unselected annotations remain as phantom data (just paths and boundaries)
        if annotation.is_selected:
            annotation.create_graphics_item(self.scene)
            # Set the visibility based on the label's visibility checkbox
            self.set_annotation_visibility(annotation)
        
        # Connect essential update signals (guard prevents duplicate connections)
        if not annotation._signals_connected:
            annotation.selected.connect(self.select_annotation)
            annotation.annotationDeleted.connect(self.delete_annotation)
            annotation.annotationUpdated.connect(self.on_annotation_updated)
            annotation._signals_connected = True

    def load_annotations(self, image_path=None, annotations=None):
        """Load annotations for the specified image path or current image."""
        # First load the mask annotation if it exists
        self.load_mask_annotation()
        
        # Determine if we were given an explicit list of annotations to load
        explicit_annotations_provided = annotations is not None
    
        # Get raw annotations (if not explicitly provided)
        if annotations is None:
            annotations = self.get_image_annotations(image_path or self.current_image_path)
        
        if not len(annotations):
            return
        
        # Only filter by visibility if we're loading all annotations for an image
        # (not when a specific list of annotations was provided by the caller)
        if not explicit_annotations_provided:
            # Get visible labels to filter annotations (lazy-loading approach)
            visible_labels = self.main_window.label_window.get_visible_labels()
            visible_label_ids = {label.id for label in visible_labels}
            
            # Filter annotations to only load those with visible labels BEFORE cropping
            annotations_to_load = [ann for ann in annotations if ann.label.id in visible_label_ids]
        else:
            # Explicit annotations list provided - trust the caller's filtering
            annotations_to_load = annotations
    
        if not len(annotations_to_load):
            return
    
        # NOTE: Removed upfront cropping - annotations will be cropped on-demand when needed
        # (e.g., when selected, during classification, or when displayed in confidence window)
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Loading Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations_to_load))
        
        # Suspend spatial indexing before the loop
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        try:
            # Load each annotation and update progress
            for idx, annotation in enumerate(annotations_to_load):
                if progress_bar.wasCanceled():
                    break

                # Load the annotation
                self.load_annotation(annotation)

                # Update every 10% of the annotations (or for each item if total is small)
                if len(annotations_to_load) > 10:
                    if idx % (len(annotations_to_load) // 10) == 0:
                        progress_bar.update_progress_percentage((idx / len(annotations_to_load)) * 100)
                else:
                    progress_bar.update_progress_percentage((idx / len(annotations_to_load)) * 100)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

        finally:
            # Restore spatial indexing after all items are added
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
            
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        # Update the label window tool tips (this might need to be optimized later)
        self.main_window.label_window.update_tooltips()
        
        # PHANTOM ARCHITECTURE: Render all unselected annotations to phantom layer
        self.refresh_phantom_annotations()
        
        QApplication.processEvents()
        self.viewport().update()

    def load_mask_annotation(self):
        """Load the mask annotation for the current image, if it exists."""
        if not self.current_image_path:
            return
        # If this is a virtual video frame and we have a per-frame overlay cached,
        # show that overlay directly instead of creating or mutating a per-raster
        # MaskAnnotation. This avoids creating a single MaskAnnotation shared
        # across all frames which leads to ghosting.
        try:
            if '::frame_' in str(self.current_image_path) and hasattr(self, 'batch_results_cache'):
                cached = self.batch_results_cache.get(self.current_image_path)
                if cached:
                    qimg = cached.get('mask_qimage')
                    opacity = cached.get('opacity', 128 / 255.0)
                    try:
                        if getattr(self, '_base_image_item', None) is not None:
                            self._base_image_item.set_mask_image(qimg, opacity)
                    except Exception:
                        pass
                    # We displayed the per-frame overlay — do not create raster-level mask
                    return
        except Exception:
            pass

        # Fallback: existing behavior for non-virtual frames (may create a raster-level mask)
        mask_annotation = self.current_mask_annotation
        if not mask_annotation:
            return

        # Remove the graphics item from its current scene if it exists
        if mask_annotation.graphics_item and mask_annotation.graphics_item.scene():
            mask_annotation.graphics_item.scene().removeItem(mask_annotation.graphics_item)

        # Create the graphics item (scene previously cleared)
        mask_annotation.create_graphics_item(self.scene)
        # Set the Z-value to be above the base image but below annotations
        if mask_annotation.graphics_item:
            mask_annotation.graphics_item.setZValue(-5)
            
        # Update the mask graphic item
        mask_annotation.update_graphics_item()

        # Update the view
        self.viewport().update()

    def refresh_phantom_annotations(self):
        """
        Draws all unselected vector annotations using the ultra-fast readonly pass.
        This is called when selections change to update the phantom layer.
        """
        # If we don't have the FastImageItem active, bail out
        if getattr(self, '_base_image_item', None) is None:
            return
            
        annotations = self.get_image_annotations()
        paths_data = []
        
        for a in annotations:
            # Only draw it as a Phantom if it's visible, NOT selected, and NOT a mask
            if getattr(a.label, 'is_visible', True) and not a.is_selected and not hasattr(a, 'mask_data'):
                try:
                    paths_data.append((a.get_painter_path(), a.label.color, a.transparency))
                except Exception:
                    pass
                    
        # Hand off to the fast C++ painter
        self._base_image_item.set_readonly_annotations(paths_data)

    def get_image_annotations(self, image_path=None):
        """Get all annotations for the specified image path or current image."""
        if not image_path:
            image_path = self.current_image_path

        return self.image_annotations_dict.get(image_path, [])

    def get_image_review_annotations(self, image_path=None):
        """Get all annotations marked for review for the specified image path or current image."""
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path and annotation.label.id == '-1':
                annotations.append(annotation)

        return annotations

    def crop_annotations(self, image_path=None, annotations=None, return_annotations=True, verbose=True):
        """Crop the image around each annotation for the specified image path."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_path:
            image_path = self.current_image_path

        if annotations is None:
            annotations = self.get_image_annotations(image_path)

        if not annotations:
            QApplication.restoreOverrideCursor()
            return []
        
        progress_bar = None
        if verbose:
            progress_bar = ProgressBar(self, title="Cropping Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(annotations))

        rasterio_image = rasterio_open(image_path)
        for annotation in annotations:
            try:
                # Only crop if not already cropped
                if not annotation.cropped_image:
                    annotation.create_cropped_image(rasterio_image)
                if verbose:
                    progress_bar.update_progress()

            except Exception:
                import traceback
                traceback.print_exc()

        QApplication.restoreOverrideCursor()
        if verbose:
            progress_bar.stop_progress()
            progress_bar.close()

        if return_annotations:
            return annotations
    
    def add_annotation_from_tool(self, annotation, record_action=True):
        """
        Adds a new annotation created by a user tool.
        
        This method provides immediate user feedback by cropping the annotation
        and displaying it in the confidence window when the annotation is created
        on the current image.
        """       
        # First, add the annotation using the primary method
        self.add_annotation(annotation, record_action=record_action)
        
        # Then provide user feedback for tool-created annotations
        if annotation.image_path == self.current_image_path and annotation.label.is_visible:
            
            # Crop the annotation for immediate display in confidence window
            if not annotation.cropped_image and self.rasterio_image:
                annotation.create_cropped_image(self.rasterio_image)
            
            # Display in confidence window to give user immediate feedback
            if annotation.cropped_image:
                annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
                annotation.annotationUpdated.connect(self.on_annotation_updated)
                self.main_window.confidence_window.display_cropped_image(annotation)
                
    def add_annotation(self, annotation, record_action=True):
        """
        The single, primary method for adding an annotation.

        It adds the annotation to data structures and connects signals. It will only create
        graphics and cropped images if the annotation's image is currently displayed AND its label is visible.
        """
        if annotation is None:
            return
        
        # Set the animation manager
        annotation.set_animation_manager(self.animation_manager)

        # --- Core Logic (runs for every annotation) ---
        # Add to the main annotation dictionary
        self.annotations_dict[annotation.id] = annotation

        # Add to the dictionary that groups annotations by image path
        if annotation.image_path not in self.image_annotations_dict:
            self.image_annotations_dict[annotation.image_path] = []
        if annotation not in self.image_annotations_dict[annotation.image_path]:
            self.image_annotations_dict[annotation.image_path].append(annotation)
            
        # Inject / update scale
        self.set_annotation_scale(annotation)

        # Connect signals for future interaction
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        annotation.annotationUpdated.connect(self.on_annotation_updated)
        
        # If this is a MaskAnnotation, update the raster's reference to it
        if isinstance(annotation, MaskAnnotation):
            raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
            if raster:
                raster.mask_annotation = annotation

        # --- Conditional UI Logic (runs only if the image is visible AND label is visible) ---
        if annotation.image_path == self.current_image_path and annotation.label.is_visible:
            # ---> Skip heavy graphics if streaming inference <---
            if getattr(self, 'is_streaming_inference', False):
                pass
            else:
                # Create graphics item for display in the scene
                if not annotation.graphics_item:
                    annotation.create_graphics_item(self.scene)
                    
                # Set the visibility based on the current UI state (will respect label checkbox)
                self.set_annotation_visibility(annotation)
                
                # If video is currently playing, immediately strip the graphics we just created
                # so the annotation doesn't ghost over the advancing frames.
                if self._playback_timer.isActive():
                    self._clear_annotation_graphics_single(annotation)
                else:
                    # PHANTOM ARCHITECTURE: Push the new annotation into the phantom layer
                    # so it is immediately visible without needing to be selected first.
                    self.refresh_phantom_annotations()
                    # Force the screen to instantly show the newly drawn item
                    self.viewport().update()

        # --- Finalization ---
        # Update the annotation count in the ImageWindow table (always, regardless of visibility)
        self.main_window.image_window.update_image_annotations(annotation.image_path)

        # If requested, record this single addition as an undo-able action
        if record_action:
            self.action_stack.push(AddAnnotationAction(self, annotation))
        
        # Emit the signal that an annotation was created
        self.annotationCreated.emit(annotation.id)
        
    def add_annotations(self, annotations_list: list, record_action: bool = True):
        """
        Efficiently adds a list of annotations to the data models and then
        updates the relevant UI components in a single batch.
        """
        if not annotations_list:
            return

        images_to_update = set()
        
        # Suspend spatial indexing
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        for annotation in annotations_list:
            if annotation is None or annotation.id in self.annotations_dict:
                continue

            annotation.set_animation_manager(self.animation_manager)
            self.set_annotation_scale(annotation)

            # Update data dictionaries
            self.annotations_dict[annotation.id] = annotation
            if annotation.image_path not in self.image_annotations_dict:
                self.image_annotations_dict[annotation.image_path] = []
            self.image_annotations_dict[annotation.image_path].append(annotation)

            images_to_update.add(annotation.image_path)

            # Connect signals (guard prevents duplicates if load_annotation is also called)
            if not annotation._signals_connected:
                annotation.selected.connect(self.select_annotation)
                annotation.annotationDeleted.connect(self.delete_annotation)
                annotation.annotationUpdated.connect(self.on_annotation_updated)
                annotation._signals_connected = True

            if isinstance(annotation, MaskAnnotation):
                raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
                if raster:
                    raster.mask_annotation = annotation

            # If the annotation belongs to the current image, we MUST 
            # create its visual item in the scene immediately.
            if annotation.image_path == self.current_image_path:
                # ---> Skip heavy graphics if streaming inference <---
                if getattr(self, 'is_streaming_inference', False):
                    pass
                else:
                    self.load_annotation(annotation)
                
        # Restore spatial indexing
        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        if images_to_update:
            # ---> Respect streaming flag to avoid O(N²) UI freezes <---
            if not getattr(self, 'is_streaming_inference', False):
                for path in images_to_update:
                    # Pass False so it only updates the raster, not the whole UI
                    self.main_window.image_window.update_image_annotations(path, update_counts=False)
                # The final UI update handles the counts ONCE
                self.main_window.label_window.update_annotation_count()
            
            # Repaint exactly ONCE, but only if the active image was affected by the import
            if self.current_image_path in images_to_update:
                # PHANTOM ARCHITECTURE: Push all new annotations into the phantom layer
                self.refresh_phantom_annotations()
                self.viewport().update()

        if record_action:
            self.action_stack.push(AddAnnotationsAction(self, list(annotations_list)))

        added_ids = [ann.id for ann in annotations_list if ann and ann.id in self.annotations_dict]
        if added_ids:
            self.annotationsCreated.emit(added_ids)

    def delete_annotation(self, annotation_id, record_action=True, bulk_mode=False):
        """Delete an annotation by its ID from dicts."""
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            
            # Always suppress the phantom refresh inside unselect_annotation; we must
            # remove the annotation from image_annotations_dict FIRST before refreshing,
            # otherwise refresh_phantom_annotations() would still find it in the dict and
            # paint it back into the phantom layer as a ghost.
            self.unselect_annotation(annotation, bulk_mode=True)

            if annotation.image_path in self.image_annotations_dict:
                if annotation in self.image_annotations_dict[annotation.image_path]:
                    self.image_annotations_dict[annotation.image_path].remove(annotation)

            annotation.delete()
            del self.annotations_dict[annotation_id]
            self.annotationDeleted.emit(annotation_id)

            if record_action:
                self.action_stack.push(DeleteAnnotationAction(self, annotation))

            # --- BULK MODE CHECK ---
            if not bulk_mode:
                try: 
                    self.main_window.image_window.update_image_annotations(annotation.image_path)
                except Exception: 
                    pass
                try: 
                    self.main_window.label_window.update_annotation_count()
                except Exception: 
                    pass
                self.main_window.confidence_window.clear_display()
                # Refresh the phantom layer NOW that the annotation is fully removed from
                # all dicts, so the ghost is immediately erased.
                self.refresh_phantom_annotations()
                # Ensure scene and viewport are updated and events are processed
                try:
                    self.scene.update()
                except Exception:
                    pass
                self.viewport().update()
                QApplication.processEvents()

    def delete_annotations(self, annotations, record_action=True):
        """Delete a list of annotations (Ultimate Bulk Optimization)."""
        if not annotations:
            return
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # 1. Record the action stack once
        if record_action:
            try:
                self.action_stack.push(DeleteAnnotationsAction(self, list(annotations)))
            except Exception:
                pass

        # 2. Fast tracking of IDs and affected images
        ann_ids_to_delete = {ann.id for ann in annotations}
        affected_images = {ann.image_path for ann in annotations if ann.image_path}
        
        # 3. INSTANT LIST REBUILD
        for image_path in affected_images:
            if image_path in self.image_annotations_dict:
                self.image_annotations_dict[image_path] = [
                    ann for ann in self.image_annotations_dict[image_path] 
                    if ann.id not in ann_ids_to_delete
                ]
                # Clean up empty lists to prevent memory leaks
                if not self.image_annotations_dict[image_path]:
                    del self.image_annotations_dict[image_path]

        # 4. Remove from main dict, scene, and emit signals (Optimized)
        
        # Suspend the Scene Index and Block Signals
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.blockSignals(True)

        for ann in annotations:
            if ann.id in self.annotations_dict:
                del self.annotations_dict[ann.id]
                
            # Block the annotation's own internal signals as well
            ann.blockSignals(True)
            ann.delete()
            ann.blockSignals(False)

        # Turn signals and spatial indexing back on
        self.blockSignals(False)
        self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        
        # --- Emit the bulk deletion to update the galleries instantly ---
        self.annotationsDeleted.emit(list(ann_ids_to_delete))
        # ----------------------------------------------------------------
            
        # 5. UI Updates EXACTLY ONCE at the very end
        for image_path in affected_images:
            try: 
                self.main_window.image_window.update_image_annotations(image_path)
            except Exception: 
                pass
            
        try: 
            self.main_window.label_window.update_annotation_count()
        except Exception: 
            pass
            
        self.main_window.confidence_window.clear_display()
        
        # Rebuild the phantom layer from the now-pruned annotation dicts so that
        # deleted annotations don't linger as ghost outlines. This MUST happen after
        # image_annotations_dict is updated (step 3 above) and before the viewport
        # repaint, otherwise the stale path data from the previous unselect_annotations()
        # call is used and the phantoms stay visible until the next click.
        if self.current_image_path in affected_images:
            self.refresh_phantom_annotations()
        
        # A single viewport update after the scene is completely modified
        self.viewport().update()
        
        QApplication.restoreOverrideCursor()

    def delete_selected_annotations(self):
        """Delete all currently selected annotations in a single batch."""
        # Get the selected annotations
        selected_annotations = self.selected_annotations.copy()
        # Unselect them first to clean up confidence window connections
        self.unselect_annotations()
        # Call the bulk delete method to trigger the optimized viewer slots
        self.delete_annotations(selected_annotations)

    def delete_label_annotations(self, label):
        """Delete all annotations with the specified label (Bulk Optimized)."""
        # 1. Use list comprehension for significantly faster filtering across the master dict
        labeled_annotations = [
            ann for ann in self.annotations_dict.values() 
            if ann.label.id == label.id
        ]
        
        # 2. Only trigger the deletion process if work is required
        if labeled_annotations:
            # Delegate to the optimized bulk method which handles cursors, 
            # signal blocking, and a single consolidated UI refresh.
            self.delete_annotations(labeled_annotations)

    def delete_image_annotations(self, image_path):
        """Delete all annotations associated with a specific image path (Bulk Optimized)."""
        # For VideoRaster base paths, annotations live under ::frame_ virtual keys.
        # Recurse over every frame key so the caller doesn't need to know about them.
        if image_path not in self.image_annotations_dict:
            prefix = image_path + '::frame_'
            frame_keys = [k for k in list(self.image_annotations_dict.keys()) if k.startswith(prefix)]
            for frame_key in frame_keys:
                self.delete_image_annotations(frame_key)
            # If the canvas is currently displaying a frame of this video, force a full
            # reload to guarantee stale graphics items are cleared from the scene.
            if (self._active_video_raster is not None and
                    self.current_image_path and
                    self.current_image_path.startswith(prefix)):
                self._display_video_frame(self._current_frame_idx)
            return

        # 1. Access label lock state once
        label_window = self.main_window.label_window
        label_locked = label_window.label_locked
        locked_label_id = label_window.locked_label.id if label_locked else None
        
        # 2. Efficiently filter the image-specific list using comprehension
        annotations_to_delete = [
            ann for ann in self.image_annotations_dict[image_path]
            if not (label_locked and ann.label.id == locked_label_id)
        ]
        
        if annotations_to_delete:
            # 3. Use bulk delete to handle internal dictionaries and viewer updates
            self.delete_annotations(annotations_to_delete)
            
        # 4. Handle Mask/Semantic Reset
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if raster:
            raster.delete_mask_annotation()
        
        # --- THE FIX ---
        # Removed redundant self.scene.update() and self.viewport().update() calls.
        # Since delete_annotations() already calls viewport().update() at the end, 
        # removing these prevents a second expensive repaint pass.

    def delete_image(self, image_path):
        """Delete an image and all its associated annotations."""
        # Delete all annotations associated with image path
        self.delete_image_annotations(image_path)
        # Delete the image
        if self.current_image_path == image_path:
            self.scene.clear()
            self.main_window.confidence_window.clear_display()
            self.current_image_path = None
            self.pixmap_image = None
            self.rasterio_image = None
            self.active_image = False


class ViewAnimator(QObject):
    """Top-level helper QObject with animatable properties to smoothly update view center and zoom.

    The animator exposes `center_x`, `center_y`, and `zoom` properties so
    `QPropertyAnimation` can interpolate them. On each setter call the
    corresponding view transform/centering is applied immediately.
    """
    def __init__(self, view):
        super().__init__()
        self.view = view
        self._center_x = 0.0
        self._center_y = 0.0
        # Use current horizontal scale as zoom (assumes uniform scaling)
        try:
            self._zoom = float(self.view.transform().m11())
        except Exception:
            self._zoom = 1.0

    def _get_center_x(self):
        return self._center_x

    def _set_center_x(self, v):
        self._center_x = float(v)
        # Keep center_y in sync when centering
        self.view.centerOn(QPointF(self._center_x, self._center_y))

    def _get_center_y(self):
        return self._center_y

    def _set_center_y(self, v):
        self._center_y = float(v)
        self.view.centerOn(QPointF(self._center_x, self._center_y))

    def _get_zoom(self):
        return self._zoom

    def _set_zoom(self, v):
        # Apply absolute zoom by resetting transform and scaling
        try:
            self._zoom = float(v)
            self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self.view.resetTransform()
            # Clamp zoom to a sensible positive range
            z = max(0.0001, self._zoom)
            self.view.scale(z, z)
            # Remember zoom on view object as well
            self.view.zoom_factor = z
        except Exception:
            pass

    center_x = pyqtProperty(float, _get_center_x, _set_center_x)
    center_y = pyqtProperty(float, _get_center_y, _set_center_y)
    zoom = pyqtProperty(float, _get_zoom, _set_zoom)