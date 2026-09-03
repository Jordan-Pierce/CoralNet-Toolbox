import warnings

import os
import traceback
import time
from pathlib import Path
from typing import Optional

import numpy as np

import pyqtgraph as pg
from PyQt5.QtGui import QMouseEvent, QPixmap, QImage, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QSize, QObject, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QGraphicsPixmapItem,
                             QSlider, QSpinBox, QLabel, QHBoxLayout, QVBoxLayout, QFormLayout,
                             QDialog, QDialogButtonBox, QGroupBox,
                             QWidget, QComboBox, QToolButton, QToolBar, QSizePolicy)

from coralnet_toolbox.QtBaseCanvas import BaseCanvas, phantom_group_key

from coralnet_toolbox.Annotations import (
    PatchAnnotation,
    PolygonAnnotation,
    RectangleAnnotation,
    MaskAnnotation,
)
from coralnet_toolbox.Annotations.QtAnnotation import RenderMode

from coralnet_toolbox.Tools import (
    PatchTool,
    RectangleTool,
    PolygonTool,
    BrushTool,
    EraseTool,
    FillTool,
    DropperTool,
    SAMTool,
    FeatureSelectTool,
    SeeAnythingTool,
    SelectTool,
    WorkAreaTool,
    ScaleTool,
    PatchSamplingTool,
)

from coralnet_toolbox.QtActions import (
    AddAnnotationAction,
    DeleteAnnotationAction,
    AddAnnotationsAction,
    DeleteAnnotationsAction,
    CompoundAction,
    ChangeLabelAction,
    ChangeLabelsAction,
    ResizeAnnotationAction,
    AnnotationGeometryEditAction,
    MaskEditAction,
)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.Icons import ColorComboBox
from coralnet_toolbox.Icons import ColormapDelegate

from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import convert_scale_units
from coralnet_toolbox.utilities import is_length_unit
from coralnet_toolbox.utilities import get_view_scale

from coralnet_toolbox.QtVideoPlayer import VideoPlayerWidget
from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.MachineLearning.ExportDataset.export_dataset_utils import parse_frame_path

warnings.filterwarnings("ignore", category=DeprecationWarning)

_PERF_LOG = bool(os.environ.get("CNT_PERF_LOG"))

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

# Longest edge of the interim image shown while a large raster's full-resolution
# decode runs on a worker thread (see AnnotationWindow.set_image). Read off an
# overview pyramid when the raster has one, so it is nearly free. Rasters
# smaller than this are decoded synchronously — there is nothing to hide.
PROGRESSIVE_INTERIM_EDGE = 2048


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
    annotationSelectionChanged = pyqtSignal(object)
    unitScaleChanged = pyqtSignal(str)  # display unit for derived measurements  # list of annotation IDs when selection changes

    def __init__(self, main_window, parent=None):
        """Initialize the annotation window with the main window and parent widget."""
        super().__init__(parent)  # BaseCanvas initializes scene, pixmap_image, zoom_factor, etc.
        self.main_window = main_window

        # Reference to the global animation manager
        
        # Central annotation data store (owned by MainWindow's AnnotationManager)
        self.annotation_manager = self.main_window.annotation_manager

        self.annotation_size = 224
        self.transparency = 128

        self.drag_start_pos = None
        self.rasterized_annotations_cache = []  # Caches vector annotations during mask mode
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state
        self._syncing_selection = False  # Flag to prevent selection sync loops
        self._skip_phantom_refresh = False  # Flag to coalesce phantom rebuilds
        # Streaming inference mode: when True, new annotations are saved to the
        # data model but heavy Qt graphics are skipped to keep playback smooth.
        self.is_streaming_inference = False
        
        # Image state (BaseCanvas has pixmap_image, active_image, current_image_path)
        self.rasterio_image = None
        # Background full-resolution decode (progressive load). The timer
        # debounces navigation so paging through images does not start a decode
        # per image; see _start_full_res_decode.
        self._full_res_worker = None
        self._pending_full_res_path = None
        self._full_res_timer = QTimer(self)
        self._full_res_timer.setSingleShot(True)
        self._full_res_timer.setInterval(250)
        self._full_res_timer.timeout.connect(self._launch_full_res_decode)
        # Workers kept alive until their thread ends; see _start_full_res_decode.
        self._live_workers = set()
        _app = QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._shutdown_full_res_decode)

        # Update placeholder label text for AnnotationWindow's context
        self._placeholder_label.setText(
            "No image loaded\nImport or drag and drop an image or Project file."
        )
        self._placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setAutoFillBackground(True)
        
        # Z-channel visualization (BaseCanvas has z_item, z_data_raw, z_index, etc.)
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

        # Guards against duplicate connections from repeated showEvent calls
        self._scale_signals_connected = False

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
        self.transparency_slider.setToolTip(
            "Annotation transparency\n"
            "\n"
            "Sets the fill opacity of all annotations on\n"
            "the current image.\n"
            "Left = transparent, right = opaque."
        )
        self.transparency_slider.valueChanged.connect(self._on_transparency_slider_changed)
        # Debounce: applying transparency is O(N annotations) + full phantom rebuild,
        # so coalesce rapid slider ticks into one apply ~75 ms after the last tick.
        self._pending_transparency = self.transparency_slider.value()
        self._transparency_debounce = QTimer(self)
        self._transparency_debounce.setSingleShot(True)
        self._transparency_debounce.setInterval(75)
        self._transparency_debounce.timeout.connect(self._apply_pending_transparency)

        # Phantom-layer rebuild coalescing. Unlike the transparency debounce
        # there is no interval: the flush is posted for the end of the current
        # event-loop turn, so it still runs before the next paint and no
        # user-visible latency is introduced.
        self._phantom_pending_full = False
        self._phantom_pending_annotations = []
        self._phantom_flush_scheduled = False

        # Lazily-selected annotations gain their Qt items when panning or
        # zooming brings them into view.
        self.viewNavigated.connect(self._on_view_navigated_promote)

        # --- Positional/Dimensional Labels ---
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.mouse_position_label.setMinimumWidth(app_theme.scale_int(150))
        self.mouse_position_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.mouse_position_label.setToolTip("Cursor position within the image, in pixels.\nMeasured from the top-left corner of the image.")
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.image_dimensions_label.setMinimumWidth(app_theme.scale_int(150))
        self.image_dimensions_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.image_dimensions_label.setToolTip("Dimensions of the loaded image, in pixels.\nShown as height x width.")
        self.view_dimensions_label = QLabel("View: 0 x 0")
        self.view_dimensions_label.setMinimumWidth(app_theme.scale_int(150))
        self.view_dimensions_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.view_dimensions_label.setToolTip("Dimensions of the visible area, in pixels.\nShown as height x width, clipped to the image bounds, so it shrinks as you zoom in.")

        # --- Scale ---
        self.scaled_dimensions_label = QLabel("Scale: 0 x 0")
        self.scaled_dimensions_label.setMinimumWidth(app_theme.scale_int(240))
        self.scaled_dimensions_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.scaled_dimensions_label.setEnabled(False)
        self.scale_unit_dropdown = QComboBox()
        self.scale_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.scale_unit_dropdown.setCurrentIndex(2)
        self.scale_unit_dropdown.setFixedWidth(app_theme.scale_int(72))
        self.scale_unit_dropdown.setEnabled(False)
        self.scale_unit_dropdown.setToolTip("Unit of measurement for scaled annotation dimensions.\nChange to convert the display scale between metric and imperial units.")
        self.scale_unit_dropdown.currentTextChanged.connect(self.on_scale_unit_changed)

        # --- Z-Channel Controls ---
        self.z_unit_dropdown = QComboBox()
        self.z_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.z_unit_dropdown.insertSeparator(self.z_unit_dropdown.count())
        self.z_unit_dropdown.addItem('px')
        self.z_unit_dropdown.setCurrentIndex(2)
        self.z_unit_dropdown.setFixedWidth(app_theme.scale_int(72))
        self.z_unit_dropdown.setEnabled(False)
        self.z_unit_dropdown.setToolTip("Unit for Z-channel depth values.\nMetric/imperial units or 'px' for pixel values.")
        self.z_unit_dropdown.currentTextChanged.connect(self.on_z_unit_changed)

        self.z_label = QLabel("Z: -----")
        self.z_label.setMinimumWidth(app_theme.scale_int(140))
        self.z_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.z_label.setEnabled(False)

        self.colormap_dropdown = ColorComboBox()
        delegate = ColormapDelegate(self.colormap_dropdown)
        self.colormap_dropdown.setItemDelegate(delegate)
        self.colormap_dropdown.addItems(['None', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo'])
        self.colormap_dropdown.setCurrentIndex(0)
        self.colormap_dropdown.setFixedWidth(app_theme.scale_int(100))
        self.colormap_dropdown.setEnabled(False)
        self.colormap_dropdown.setToolTip("Colormap for overlay visualization.\n'None' shows no colorization; other options apply scientific colormaps to the active overlay (Z-channel depth, or feature similarity when the Feature Select tool is active).")
        self.colormap_dropdown.currentTextChanged.connect(self.on_colormap_changed)

        self.colormap_opacity_slider = QSlider(Qt.Horizontal)
        self.colormap_opacity_slider.setRange(0, 255)
        self.colormap_opacity_slider.setValue(128)
        # Allow the overlay opacity slider to naturally expand like the main transparency slider
        self.colormap_opacity_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.colormap_opacity_slider.setEnabled(False)
        self.colormap_opacity_slider.setToolTip(
            "Overlay opacity\n"
            "\n"
            "Sets the opacity of the active colormap overlay:\n"
            "the Z-channel depth map, or the feature-similarity\n"
            "heatmap while the Feature Select tool is active.\n"
            "Left = transparent, right = opaque."
        )
        self.colormap_opacity_slider.valueChanged.connect(self.update_colormap_opacity)

        self.z_dynamic_button = QToolButton()
        self.z_dynamic_button.setCheckable(True)
        self.z_dynamic_button.setIcon(get_icon("dynamic.svg"))
        self.z_dynamic_button.setEnabled(False)
        self.z_dynamic_button.setToolTip(
            "Dynamic Z-range scaling\n"
            "\n"
            "Rescales the depth colormap to the min/max of the\n"
            "currently visible area instead of the whole image,\n"
            "so depth contrast adapts as you pan and zoom.\n"
            "Applies to the Z-channel overlay only."
        )
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

        # is_length_unit guards the metre assumption below: a raster may carry
        # a non-convertible unit (e.g. 'degree' from a lon/lat world file), and
        # treating that as metres would mislabel the whole readout.
        if raster and is_length_unit(raster.scale_units):
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
            self.on_scale_unit_changed(self.scale_unit_dropdown.currentText(), refresh_confidence=False)

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
                        
                        # Convert to selected unit if different from original.
                        # A relative z-channel carries 'px', which cannot be
                        # converted - leave the value alone rather than showing
                        # the same number under a different unit.
                        display_value = z_value
                        if (self.current_unit_z != original_unit
                                and is_length_unit(original_unit)
                                and is_length_unit(self.current_unit_z)):
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
                    self.colormap_dropdown.setEnabled(True)
                    # Only enable dynamic button if colormap is not set to "None"
                    if self.colormap_dropdown.currentText() != "None":
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
        self.colormap_dropdown.setEnabled(enabled)
        self.colormap_opacity_slider.setEnabled(enabled)
        
        # Dynamic button is only enabled when a colormap is active (not "None")
        if enabled and self.colormap_dropdown.currentText() != "None":
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
            self.colormap_dropdown.setCurrentText("None")
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
            self.colormap_dropdown.setCurrentText("None")
            self.enable_z_visualization_controls(False)

    def on_scale_unit_changed(self, to_unit, refresh_confidence=True):
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
        
        # Every derived measurement on screen -- the confidence tooltip, the
        # Metadata panel's built-in rows -- is now expressed in the wrong unit.
        # Announce it rather than refreshing one known consumer, so a panel
        # added later stays correct without editing this method.
        self.unitScaleChanged.emit(to_unit)

        # Refresh the confidence window if an annotation is selected
        if refresh_confidence and self.main_window.confidence_window.annotation:
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
                    
                    # Convert from original unit to selected unit, unless the
                    # z-channel is relative ('px') and has nothing to convert
                    converted_value = self.current_z_value
                    if is_length_unit(original_unit) and is_length_unit(selected_unit):
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

    def on_colormap_changed(self, colormap_name):
        """Handle colormap dropdown changes for the active overlay.

        Drives whichever overlay is active (Z-channel depth by default, or the
        feature-similarity overlay while the Feature Select tool is active). The
        dynamic-range button only applies to the Z overlay.
        """
        self.update_overlay_colormap(colormap_name)

        feature_active = self._active_colormap_overlay is self._feature_overlay

        # Enable/disable the opacity slider based on colormap selection.
        if colormap_name == "None":
            self.colormap_opacity_slider.setEnabled(False)
            if not feature_active:
                self.z_dynamic_button.setEnabled(False)
                self.z_dynamic_button.setChecked(False)
        else:
            if feature_active:
                # Feature overlay has no depth data / dynamic range; just the slider.
                self.colormap_opacity_slider.setEnabled(True)
            elif self.z_data_raw is not None:
                self.colormap_opacity_slider.setEnabled(True)
                self.z_dynamic_button.setEnabled(True)

    def update_colormap_opacity(self, value):
        """
        Update the active overlay's opacity from the slider.

        Args:
            value (int): Slider value from 0-255
        """
        # Convert slider value (0-255) to opacity (0.0-1.0)
        opacity = value / 255.0

        # Update the active overlay (Z-channel depth or feature similarity).
        self.set_overlay_opacity(opacity)

    def on_z_dynamic_toggled(self, checked):
        """Handle z-dynamic scaling button toggle."""
        self.toggle_dynamic_z_scaling(checked)

    def _on_transparency_slider_changed(self, value):
        """Debounced slider handler; the heavy apply runs after the drag pauses."""
        self._pending_transparency = value
        self._transparency_debounce.start()

    def _apply_pending_transparency(self):
        self.update_label_transparency(self._pending_transparency)

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

        # Video frames render the mask overlay via FastImageItem._mask_opacity
        # (a baked-in value), not via MaskGraphicsItem's render-time transparency.
        # Push the new opacity to the live item and the per-frame cache so the
        # slider actually affects the displayed mask (including down to 0).
        if getattr(self, '_active_video_raster', None) is not None:
            opacity = transparency / 255.0
            cached = getattr(self, 'batch_results_cache', {}).get(self.current_image_path)
            if cached is not None:
                cached['opacity'] = opacity
            bii = getattr(self, '_base_image_item', None)
            if bii is not None and getattr(bii, '_mask_image', None) is not None:
                try:
                    bii.set_mask_image(bii._mask_image, opacity)
                except Exception:
                    pass

        # Rebuild phantom layer with updated transparency
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
        self.transparent_icon_label = QLabel()
        self.transparent_icon_label.setPixmap(get_icon("transparent.svg").pixmap(app_theme.scale_size(16)))
        self.opaque_icon_label = QLabel()
        self.opaque_icon_label.setPixmap(get_icon("opaque.svg").pixmap(app_theme.scale_size(16)))
        trans_layout.addWidget(self.transparent_icon_label)
        trans_layout.addWidget(self.transparency_slider)
        trans_layout.addWidget(self.opaque_icon_label)
        toolbar.addWidget(trans_widget)

        toolbar.addSeparator()

        # Z-channel controls moved to top toolbar (to the right of annotation transparency)
        z_widget = QWidget()
        z_layout = QHBoxLayout(z_widget)
        z_layout.setContentsMargins(4, 0, 4, 0)
        # Order: dynamic range button, z transparency slider, then colormap combo (swapped)
        z_layout.addWidget(self.z_dynamic_button)
        z_layout.addWidget(self.colormap_opacity_slider)
        z_layout.addWidget(self.colormap_dropdown)
        toolbar.addWidget(z_widget)

        toolbar.addSeparator()

        return toolbar

    def refresh_scaling(self):
        """Refresh annotation-window elements that depend on the selected UI scale."""
        self._placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        if hasattr(self, 'transparent_icon_label'):
            self.transparent_icon_label.setPixmap(get_icon("transparent.svg").pixmap(app_theme.scale_size(16)))
        if hasattr(self, 'opaque_icon_label'):
            self.opaque_icon_label.setPixmap(get_icon("opaque.svg").pixmap(app_theme.scale_size(16)))
        if hasattr(self, '_video_player'):
            self._video_player.refresh_scaling()

    def create_bottom_toolbar(self) -> QToolBar:
        """Create the bottom toolbar with mouse position, image/view dimensions, 
        scale, and z-channel info.
        """
        toolbar = QToolBar("Annotation Status")
        toolbar.setMovable(False)
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(app_theme.scale_int(4), app_theme.scale_int(2), app_theme.scale_int(4), app_theme.scale_int(2))
        layout.setSpacing(app_theme.scale_int(8))
        
        def make_group(*widgets):
            g = QWidget()
            l = QHBoxLayout(g)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(app_theme.scale_int(4))
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
            "feature_select": FeatureSelectTool(self),
            "see_anything": SeeAnythingTool(self),
            "work_area": WorkAreaTool(self),
            # Selectable mask tools
            "brush": BrushTool(self),
            "fill": FillTool(self),
            "erase": EraseTool(self),
            "dropper": DropperTool(self),
            # Dialog tools
            "scale": ScaleTool(self),
            "patch_sampling": PatchSamplingTool(self),
        }
        
        # Defines which tools trigger mask mode
        self.mask_tools = {"brush", "fill", "erase", "dropper"}
        

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

        if getattr(updated_annotation, 'is_mask_annotation', False):
            # MaskAnnotation stores its item in graphics_item, not graphics_item_group,
            # so is_graphics_item_valid() always returns False for it.  Call
            # refresh_graphics() directly: this recreates the QImage (busting Qt's
            # OpenGL texture cache) and marks the item dirty so paint() is invoked.
            # For video frames, current_image_path is a virtual path like
            # "video.mp4::frame_0" while mask_annotation.image_path is "video.mp4",
            # so check both the exact match and the prefix match.
            cur = self.current_image_path or ''
            ann_path = updated_annotation.image_path or ''
            path_matches = (
                ann_path == cur
                or ('::frame_' in cur and cur.startswith(ann_path))
            )
            if path_matches:
                updated_annotation.refresh_graphics()
                # For video frames, also sync the updated mask into the per-frame cache
                # so load_mask_annotation shows the right overlay on navigation.
                # Skip when deferred (e.g. during batch predict tile loop — the caller
                # will do a single sync at the end instead).
                if '::frame_' in cur and not getattr(self, '_deferring_video_cache_sync', False):
                    self._sync_video_mask_to_cache()
            self.refresh_mask_annotation_view(updated_annotation)

        try:
            self.annotationModified.emit(updated_annotation.id)
        except Exception:
            pass

    def refresh_mask_annotation_view(self, mask_annotation):
        """Recompute mask statistics and refresh the raster metadata for one image."""
        if mask_annotation is None:
            return

        cur = self.current_image_path or ''
        ann_path = mask_annotation.image_path or ''
        path_matches = (
            ann_path == cur
            or ('::frame_' in cur and cur.startswith(ann_path))
        )
        if path_matches:
            self.viewport().update()
            # For regular (non-video) images, also refresh the RasterTable count so
            # the annotation count column updates immediately when the mask is painted
            # or erased.  Video frames are handled by _sync_video_mask_to_cache which
            # already calls update_image_annotations, so skip them here to avoid a
            # double update.
            if '::frame_' not in cur:
                try:
                    self.main_window.image_window.update_image_annotations(
                        cur, update_counts=False
                    )
                except Exception:
                    pass

    def showEvent(self, event):
        """Handle show events to fit the view to the image."""
        super().showEvent(event)
        
        # Connect to ImageWindow signals
        self.main_window.image_window.imageLoaded.connect(self.on_image_loaded_check_z_channel)
        self.main_window.image_window.zChannelRemoved.connect(self.on_z_channel_removed)

        # Keep annotation scale fields in sync whenever a raster's scale changes
        if not self._scale_signals_connected:
            self.main_window.image_window.raster_manager.scaleUpdated.connect(
                self.on_raster_scale_updated
            )
            self._scale_signals_connected = True
    
    def resizeEvent(self, event):
        """Handle resize events to maintain proper view fitting."""
        super().resizeEvent(event)
        
        # Only fit view if we have an active image
        if self.active_image and self.scene:
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
            self.tools[self.selected_tool].mouseMoveEvent(event)

        scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))

        if not self.cursorInWindow(event.pos()):
            self.toggle_cursor_annotation()

        # Let BaseCanvas handle native pan/zoom via super()
        super().mouseMoveEvent(event)

    def on_pointer_left(self):
        """Tear down hover graphics when the pointer leaves the window.

        cursorInWindow() only ever gets consulted from mouseMoveEvent, and Qt
        delivers no move event once the pointer is outside the widget — so a
        fast exit at the widget edge (or an alt-tab, or moving onto a dock) left
        the crosshair and cursor annotation frozen on screen. Tool.leave() 
        clears hover state only; an in-progress stroke or polygon survives.
        """
        if self.selected_tool and self.selected_tool in self.tools:
            try:
                self.tools[self.selected_tool].leave()
            except Exception:
                pass

        # BaseCanvas drops this window's own cursor preview + dynamic marker.
        super().on_pointer_left()

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

    def keyPressEvent(self, event):
        """Handle keyboard press events including undo/redo and deletion of selected annotations."""
        # Handle Ctrl+H or Home to reset the scene view and clear static markers
        if (event.key() == Qt.Key_H and event.modifiers() == Qt.ControlModifier) or \
                event.key() == Qt.Key_Home:
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
            if annotation.graphics_item_group is not None:
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
            annotation.tag_item = None
            annotation.dimension_tag_item = None
            annotation.is_selected = False
            annotation.render_mode = RenderMode.PHANTOM
        except Exception:
            pass

    def _on_annotation_change_for_video(self, *args):
        """Slot called when annotations are created or deleted; refreshes scrub-bar ticks."""
        if self._active_video_raster is not None:
            self._update_video_annotation_marks()

    def _get_annotated_frame_indices(self) -> set:
        """Return the set of frame indices that have at least one annotation for the active video.

        Includes both vector annotations (from image_annotations_dict) and
        per-frame semantic mask overlays (from batch_results_cache).
        """
        if self._active_video_raster is None:
            return set()
        prefix = self._active_video_raster.image_path + '::frame_'
        frame_indices = set()

        # Vector annotations
        for key, annotations in self.image_annotations_dict.items():
            if key.startswith(prefix) and annotations:
                try:
                    frame_indices.add(int(key.split('::frame_', 1)[1]))
                except (ValueError, IndexError):
                    pass

        # Per-frame masks owned by the raster (authoritative, and already
        # populated for a reopened project whose frames have not been shown yet)
        try:
            frame_indices |= self._active_video_raster.get_frame_mask_indices()
        except AttributeError:
            pass

        # Per-frame semantic mask overlays and detect/segment video results
        cache = getattr(self, 'batch_results_cache', None) or {}
        for key, cached in cache.items():
            if not (isinstance(key, str) and key.startswith(prefix) and cached):
                continue
            # Confirm the entry has actual content
            has_content = False
            if isinstance(cached, dict):
                # Semantic overlay stored as a legacy dict with mask_arr / mask_qimage
                mask_arr = cached.get('mask_arr')
                if mask_arr is not None:
                    try:
                        has_content = bool(np.any(mask_arr))
                    except Exception:
                        has_content = cached.get('mask_qimage') is not None
                else:
                    has_content = cached.get('mask_qimage') is not None
            else:
                # Raw Ultralytics Results object (detect / segment video frames)
                try:
                    boxes = getattr(cached, 'boxes', None)
                    masks = getattr(cached, 'masks', None)
                    has_content = bool(
                        (boxes is not None and len(boxes) > 0) or
                        (masks is not None and len(masks) > 0)
                    )
                except Exception:
                    has_content = True  # assume content if we can't check
            if has_content:
                try:
                    frame_indices.add(int(key.split('::frame_', 1)[1]))
                except (ValueError, IndexError):
                    pass

        return frame_indices

    def _update_video_annotation_marks(self):
        """Compute which frame indices have annotations and push them to the player slider."""
        self._video_player.update_annotation_marks(self._get_annotated_frame_indices())

    def _get_keyframe_indices(self) -> set:
        """Return the set of keyframe indices for the active video raster."""
        if self._active_video_raster is None:
            return set()
        try:
            return self._active_video_raster.get_keyframes()
        except Exception:
            return set()

    def _update_video_keyframe_marks(self):
        """Push the active video's keyframe indices to the player slider."""
        self._video_player.update_keyframe_marks(self._get_keyframe_indices())

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
        try:
            self._video_player.nextKeyframeClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevKeyframeClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.keyframeToggled.disconnect()
        except Exception:
            pass

        self._video_player.seekChanged.connect(self._on_video_seek)
        self._video_player.playClicked.connect(self._on_video_play)
        self._video_player.pauseClicked.connect(self._on_video_pause)
        self._video_player.nextFrameClicked.connect(self._on_video_next)
        self._video_player.prevFrameClicked.connect(self._on_video_prev)
        self._video_player.nextAnnotatedClicked.connect(self._on_video_next_annotated)
        self._video_player.prevAnnotatedClicked.connect(self._on_video_prev_annotated)
        self._video_player.nextKeyframeClicked.connect(self._on_video_next_keyframe)
        self._video_player.prevKeyframeClicked.connect(self._on_video_prev_keyframe)
        self._video_player.keyframeToggled.connect(self._on_keyframe_toggled)

        # Reset player state
        self._video_player.reset()

        # Display frame 0
        self._display_video_frame(0)

        # Populate tick marks for any pre-existing annotations and keyframes
        self._update_video_annotation_marks()
        self._update_video_keyframe_marks()

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
        try:
            self._video_player.nextKeyframeClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.prevKeyframeClicked.disconnect()
        except Exception:
            pass
        try:
            self._video_player.keyframeToggled.disconnect()
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

        # Reset the shared editing buffer for THIS displayed frame. If the frame
        # has no cached mask, clear it to a blank slate (load_mask_annotation's
        # _restore is load-only and intentionally won't zero on a cache miss).
        # When a cache exists, _restore (inside load_annotations) loads it.
        cache = getattr(self, 'batch_results_cache', {}) or {}
        if (cache.get(virtual_path) is None
                and video_raster.get_frame_mask(frame_idx) is None):
            self._clear_video_frame_mask_data()

        # Load annotations for this virtual frame path
        self.load_annotations()

        # Update the player widget state first so the slider range is valid
        self._video_player.update_state(frame_idx, video_raster.frame_count)

        # Then refresh scrub-bar tick marks (cheap; only runs on seek/pause, not playback)
        # AnnotatedSlider.paintEvent checks slider.maximum(), so ensure range set first.
        self._update_video_annotation_marks()
        self._update_video_keyframe_marks()

        # Reflect whether this frame is a keyframe on the star toggle button
        self._video_player.set_keyframe_state(video_raster.is_keyframe(frame_idx))

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
                            paths_data.append((a.get_cached_painter_path(), a.label.color, a.transparency))
                        except Exception:
                            pass
                
                # Send the raw paths to the fast item
                self._base_image_item.set_readonly_annotations(paths_data)
                # Per-frame mask overlay, rendered on demand for frames restored
                # from a project file that have never been displayed.
                self._apply_video_frame_mask_overlay(self.current_image_path)
            except Exception:
                pass

        # Update slider and counter silently (no seekChanged feedback loop)
        self._video_player.slider.blockSignals(True)
        self._video_player.slider.setValue(frame_idx)
        self._video_player.slider.blockSignals(False)
        self._video_player.lbl_frame.setText(f"{frame_idx} / {vr.frame_count}")

        # Live-update the star button so it lights up only on keyframes
        self._video_player.set_keyframe_state(vr.is_keyframe(frame_idx))

        # Clear the drop-frame gate so the worker sends the next frame
        if vr._decode_worker is not None:
            vr._decode_worker._pending_emit = False

    def _on_video_play(self):
        """Start the playback timer, clearing annotation graphics first."""
        if self._active_video_raster is None:
            return
        # Prepare the scene for fast streaming: reset the canvas to a
        # clean base-image-only state (no annotation QGraphicsItems).
        # This mirrors the full-frame redisplay that happens on pause
        # but avoids loading annotations so playback won't show stale
        # graphics artifacts.
        try:
            if hasattr(self, '_prepare_scene_for_streaming'):
                self._prepare_scene_for_streaming()
            else:
                # Fallback to old, cheaper clear if helper not present
                self._clear_current_frame_annotation_graphics()
        except Exception:
            try:
                self._clear_current_frame_annotation_graphics()
            except Exception:
                pass
        self.main_window.set_video_playback_tools_enabled(False)
        # Start the worker from the NEXT frame so the current frame (already
        # cleanly displayed with no annotation QGraphicsItems) is not re-emitted
        # by the worker, which would cause a brief annotation flash before
        # the video advances to subsequent frames.
        vr = self._active_video_raster
        next_frame = (self._current_frame_idx + 1) % vr.frame_count
        vr.seek_decode_worker(next_frame)
        vr.resume_decode_worker()

    def _prepare_scene_for_streaming(self):
        """Reset the canvas to a base-image-only state for fast streaming.

        This method uses the BaseCanvas loader to clear the scene and install
        a fresh FastImageItem for the currently displayed frame. It intentionally
        does NOT call `load_annotations()` so that no per-frame annotation
        QGraphicsItems remain visible during playback or streaming inference.
        """
        vr = self._active_video_raster
        if vr is None:
            return
        frame_idx = max(0, min(self._current_frame_idx, vr.frame_count - 1))
        virtual_path = vr.make_frame_path(vr.image_path, frame_idx)
        q_image = vr.get_frame(frame_idx)
        if q_image is None:
            return

        # Ensure rasterio ref exists for downstream crop operations
        try:
            self.rasterio_image = vr.rasterio_src
        except Exception:
            pass

        # Use canonical loader to clear scene and install fresh base image item
        try:
            self.load_visuals(q_image, virtual_path, None)
        except Exception:
            try:
                # As a fallback, clear existing graphics for the frame
                self._clear_current_frame_annotation_graphics()
            except Exception:
                pass

        # Ensure the fast-image item has no annotation overlays lingering
        try:
            if self._base_image_item is not None:
                try:
                    self._base_image_item.set_readonly_annotations([])
                except Exception:
                    pass
                try:
                    self._base_image_item.set_mask_image(None)
                except Exception:
                    pass
        except Exception:
            pass

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
                annotation.tag_item = None
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

    def _on_keyframe_toggled(self):
        """Toggle the keyframe state of the currently displayed frame."""
        vr = self._active_video_raster
        if vr is None:
            return
        new_state = vr.toggle_keyframe(self._current_frame_idx)
        # Reflect the new state on the button and refresh the slider ticks
        self._video_player.set_keyframe_state(new_state)
        self._update_video_keyframe_marks()

    def _on_video_next_keyframe(self):
        """Jump to the nearest keyframe after the current position."""
        if self._active_video_raster is None:
            return
        candidates = [f for f in self._get_keyframe_indices() if f > self._current_frame_idx]
        if candidates:
            self._display_video_frame(min(candidates))

    def _on_video_prev_keyframe(self):
        """Jump to the nearest keyframe before the current position."""
        if self._active_video_raster is None:
            return
        candidates = [f for f in self._get_keyframe_indices() if f < self._current_frame_idx]
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

        if self._base_image_item is not None:
            self._base_image_item.set_image(q_image)
            try:
                frame_annotations = self.image_annotations_dict.get(self.current_image_path, [])
                paths_data = []
                for a in frame_annotations:
                    if getattr(a.label, 'is_visible', True) and not hasattr(a, 'mask_data'):
                        try:
                            paths_data.append((a.get_cached_painter_path(), a.label.color, a.transparency))
                        except Exception:
                            pass
                self._base_image_item.set_readonly_annotations(paths_data)
                # Masks were missing from this path entirely: _prepare_scene_for_streaming
                # clears the overlay when playback starts and nothing here put one
                # back, so timer-driven playback showed no mask on any frame.
                self._apply_video_frame_mask_overlay(self.current_image_path)
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

        # Live-update the star button so it lights up only on keyframes
        self._video_player.set_keyframe_state(self._active_video_raster.is_keyframe(next_idx))

    def cursorInWindow(self, pos, mapped=False):
        """Check if the cursor position is within the image bounds."""
        if not pos or not self.active_image:
            return False

        image_rect = self.get_image_rect()
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

    def refresh_tool_label_preview(self):
        """Tell the active tool to repaint its previews for the current label.

        Called when the selected label changes and when the active label's own
        properties (color, codes) are edited in the LabelWindow.
        """
        if not self.selected_tool or self.selected_tool not in self.tools:
            return
        try:
            self.tools[self.selected_tool].refresh_label_preview()
        except Exception:
            pass

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

        # Repaint the active tool's live previews with the new label. This runs
        # before the no-selection early return below, because those previews are
        # driven by hovering and exist whether or not anything is selected.
        self.refresh_tool_label_preview()

        # Collect changes for action stack
        changes = []  # list of (annotation_id, old_label, new_label)

        # Prefer the global selection manager when available so label changes
        # can apply to gallery-selected annotations even if they are not in the
        # current canvas image.
        target_annotations = []
        try:
            selection_manager = getattr(self.main_window, 'selection_manager', None)
            selected_ids = []
            if selection_manager and hasattr(selection_manager, 'get_selected_ids'):
                selected_ids = list(selection_manager.get_selected_ids() or [])

            if selected_ids:
                annotations_dict = getattr(self, 'annotations_dict', {})
                for ann_id in selected_ids:
                    ann = annotations_dict.get(ann_id)
                    if ann:
                        target_annotations.append(ann)
            else:
                target_annotations = list(self.selected_annotations)
        except Exception:
            target_annotations = list(self.selected_annotations)

        if not target_annotations:
            QApplication.restoreOverrideCursor()
            return

        def _get_raster_source_for_annotation(annotation):
            try:
                image_window = getattr(self.main_window, 'image_window', None)
                raster_manager = getattr(image_window, 'raster_manager', None) if image_window else None
                if raster_manager and hasattr(raster_manager, 'get_raster'):
                    raster = raster_manager.get_raster(annotation.image_path)
                    if raster:
                        if getattr(raster, '_rasterio_src', None) is None and hasattr(raster, 'load_rasterio'):
                            raster.load_rasterio()
                        return getattr(raster, '_rasterio_src', None)
            except Exception:
                pass
            return getattr(self, 'rasterio_image', None)

        # Handle both valid labels and None (no label selected)
        if label is not None:
            # Track the last annotation that actually changed so we can update the
            # confidence window exactly once at the end instead of N times.
            _last_changed_annotation = None

            # Raster-source cache: avoid repeated raster lookups for annotations on the
            # same image (common when all selected annotations share one image path).
            _raster_cache: dict = {}

            def _get_raster_cached(annotation):
                ip = annotation.image_path
                if ip in _raster_cache:
                    return _raster_cache[ip]
                src = _get_raster_source_for_annotation(annotation)
                _raster_cache[ip] = src
                return src

            _n_on_canvas = 0
            _is_bulk = len(target_annotations) > 1
            for annotation in target_annotations:
                if annotation.label.id != label.id:
                    old_label = annotation.label
                    _is_on_canvas = (annotation.image_path == self.current_image_path)
                    if _is_on_canvas:
                        _n_on_canvas += 1

                    if _is_bulk:
                        # FAST PATH for bulk relabel: apply only the data changes inline,
                        # skipping update_graphics_item() entirely. For on-canvas selected
                        # annotations we're about to deselect them and rebuild the phantom
                        # layer anyway — the full QGraphicsItemGroup teardown+rebuild done by
                        # update_user_confidence() is wasted work in this case.
                        # Off-canvas annotations have no graphics items to update.
                        annotation.blockSignals(True)
                        try:
                            # Pure data update — skip update_graphics_item() entirely.
                            #
                            # On-canvas selected annotations: their QGraphicsItemGroup will be
                            # torn down by _clear_annotation_graphics_single() moments later
                            # during deselect, so any color/label rebuild here is thrown away.
                            # The phantom layer rebuilt after deselect uses annotation.label
                            # directly (the new value), so visual correctness is preserved.
                            #
                            # Off-canvas annotations: graphics_item_group is None — the call
                            # would already be a no-op, but we skip even the None-check cost.
                            annotation.verified = True
                            annotation.user_confidence = {label: 1.0}
                            if annotation.machine_confidence:
                                annotation.machine_confidence.pop(old_label, None)
                            annotation.label = label
                        finally:
                            annotation.blockSignals(False)
                    else:
                        # Single annotation: use normal path (signal + full update)
                        annotation.blockSignals(True)
                        annotation.update_user_confidence(self.selected_label)
                        annotation.blockSignals(False)

                    _last_changed_annotation = annotation
                    changes.append((annotation.id, old_label, self.selected_label))

            # For on-canvas selected annotations that went through the bulk fast-path,
            # update_graphics_item() was skipped to avoid wasted teardown/rebuild.
            # But if the user doesn't immediately deselect, their selection color and
            # label tag still show the OLD label. Fix: call update_graphics_item() for
            # every on-canvas annotation that has a live QGraphicsItemGroup right now.
            # This is much cheaper than the full update_user_confidence() call because
            # the data is already correct; we only need to repaint the item group.
            if _is_bulk and _n_on_canvas > 0:
                for annotation in target_annotations:
                    if (annotation.image_path == self.current_image_path and
                            annotation.is_graphics_item_valid()):
                        try:
                            annotation.update_graphics_item()
                        except Exception:
                            pass

            # Crop and display ONLY for the last changed annotation (confidence window only
            # shows one at a time anyway, and create_cropped_image is expensive per annotation).
            if _last_changed_annotation is not None:
                raster_source = _get_raster_cached(_last_changed_annotation)
                if raster_source is not None:
                    try:
                        _last_changed_annotation.create_cropped_image(raster_source)
                    except Exception:
                        pass
                try:
                    self.main_window.confidence_window.display_cropped_image(_last_changed_annotation)
                except Exception:
                    pass

        # Record action(s)
        try:
            if changes:
                if len(changes) == 1:
                    ann_id, old_label, new_label = changes[0]
                    action = ChangeLabelAction(self, ann_id, old_label, new_label)
                    self.action_stack.push(action)
                    try:
                        self.annotationLabelChanged.emit(
                            ann_id,
                            new_label.id if hasattr(new_label, 'id') else str(new_label)
                        )
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
            
    def on_raster_scale_updated(self, image_path):
        """
        Re-sync annotation scale properties when a raster's scale is set or removed.

        Annotations cache scale_x/scale_y/scale_units from their raster when they
        are loaded, so without this they keep reporting pixel units (or a stale
        scale) until the image is navigated away from and back.
        """
        # Video rasters emit the base video path, but their annotations are keyed
        # by virtual frame paths, so those need syncing as well.
        paths = [image_path]
        frame_prefix = f"{image_path}::frame_"
        paths.extend(p for p in self.image_annotations_dict if p.startswith(frame_prefix))

        for path in paths:
            self.set_annotations_scale(path)

        # Rebuild the tooltip for the annotation currently on display
        confidence_window = self.main_window.confidence_window
        if confidence_window.annotation and confidence_window.annotation.image_path in paths:
            confidence_window.refresh_display()

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
    
        self.refresh_phantom_annotations()
        
        self.scene.update()
        self.viewport().update()
        
    def is_annotation_moveable(self, annotation, use_status_bar=False):
        """Check if an annotation can be moved and show a warning if not verified."""
        if not annotation.verified:
            if use_status_bar:
                try:
                    self.main_window.status_bar.showMessage(
                        "Verify by selecting and pressing Ctrl+Space, "
                        "clicking a label in the ConfidenceWindow, "
                        "or updating the label manually.",
                        4000,
                    )
                except Exception:
                    pass
            else:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Warning")
                msg_box.setText(
                    "Altering an annotation that still has machine learning predictions and is not verified "
                    "cannot be done because it would overwrite the machine-generated label before you confirm it. "
                    "To verify an annotation, select it and press Ctrl+Space, click a label in the ConfidenceWindow, "
                    "or update the label manually."
                )
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()
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

    def update_scene(self):
        """Update the graphics scene and its items."""
        self.scene.update()
        self.viewport().update()
        self.viewport().repaint()

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
        # A background full-resolution decode targets the item this is about to
        # destroy; its result is stale the moment the scene is cleared.
        self._cancel_full_res_decode()

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
        """Resets the scene view and rotation."""
        # Fit the image to the view and recalculate zoom constraints
        self.fit_to_image()
        # Reset rotation to default
        self.rotation_angle = 0.0
        self._set_absolute_rotation(self.rotation_angle)  # Apply the rotation transform reset
        self.viewChanged.emit(*self.get_image_dimensions())

        # Clear own static marker (focal-point crosshair)
        self.clear_static_marker()

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
        # Force the image onto the screen now without letting unrelated queued
        # work re-enter this call.
        self.viewport().repaint()

    def _show_loading_status(self, image_path):
        """Show a persistent status bar message while an image is loading."""
        try:
            status_bar = self.main_window.status_bar
            status_bar.showMessage(f"Loading image: {os.path.basename(image_path)}...")
            # Paint it now; the full-res decode below blocks the event loop
            status_bar.repaint()
        except Exception:
            pass

    def _clear_loading_status(self):
        """Clear the persistent loading message (used on early-exit paths)."""
        try:
            self.main_window.status_bar.clearMessage()
        except Exception:
            pass

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
            # _activate_video_mode always lands on frame 0, and returns early when
            # the raster is already active, so a virtual frame path has to be
            # honoured explicitly. Without this, set_image("clip.mp4::frame_5")
            # silently shows frame 0 and no frame can be selected programmatically.
            requested_frame = self._video_frame_index(image_path)
            if requested_frame is not None:
                requested_frame = max(0, min(requested_frame, raster_check.frame_count - 1))
                if requested_frame != self._current_frame_idx:
                    self._display_video_frame(requested_frame)
            return
        else:
            # Deactivate video mode if we're switching to a regular image
            self._deactivate_video_mode()
        # ---- END VIDEO BRANCH ----

        # Persistent loading message: stays up through the staged load
        # (low-res preview -> full-res) until the loaded message replaces it
        self._show_loading_status(image_path)

        # Clean up (This is the ONLY scene clear)
        self.clear_scene()

        # Clear the action stack
        self.action_stack.undo_stack.clear()
        self.action_stack.redo_stack.clear()

        # Check that the image path is valid
        if image_path not in self.main_window.image_window.raster_manager.image_paths:
            self._clear_loading_status()
            QApplication.restoreOverrideCursor()
            return

        # Get the raster
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if not raster:
            self._clear_loading_status()
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

        # Update the rasterio image source for cropping annotations
        self.rasterio_image = raster.rasterio_src

        # Decide what to put on screen now, and whether the full-resolution
        # image follows on a worker thread.
        #
        # Progressive loading exists because a full decode of a 24k ortho blocks
        # the GUI thread for ~2 s even with GDAL threading, and there is no way
        # to make decoding 576 megapixels much faster than that. So the wait is
        # moved off the critical path instead: show a low-resolution image
        # immediately, decode the real one in the background, swap it in. Unlike
        # the display-proxy cap, nothing is permanently lost -- full resolution
        # still arrives, just a moment later.
        defer_full_res = max(raster.width, raster.height) > PROGRESSIVE_INTERIM_EDGE

        if defer_full_res:
            q_image = raster.get_thumbnail(longest_edge=PROGRESSIVE_INTERIM_EDGE)
        else:
            q_image = raster.get_qimage()

        if q_image is None or q_image.isNull():
            self.main_window.image_window.show_error(
                "Image Loading Error",
                f"Image {os.path.basename(image_path)} could not be loaded."
            )
            self._clear_loading_status()
            QApplication.restoreOverrideCursor()
            return

        # Use BaseCanvas canonical loader for the full-resolution image (preserves base logic)
        # Update the rasterio image reference used by annotation scaling/cropping
        self.rasterio_image = raster.rasterio_src
        # Load visuals into the BaseCanvas. The scene keeps the raster's true
        # dimensions even when q_image is a downscaled proxy, so every scene
        # coordinate -- annotations, work areas, markers -- stays in full-
        # resolution image space.
        self.load_visuals(q_image, image_path, raster,
                          image_dimensions=(raster.width, raster.height))

        if defer_full_res:
            self._start_full_res_decode(image_path)

        # Apply the current colormap selection and preserve UI opacity
        current_colormap = self.main_window.colormap_dropdown.currentText()
        if current_colormap != "None":
            self.update_overlay_colormap(current_colormap)
        if self._z_overlay.item is not None:
            try:
                current_opacity = self.main_window.colormap_opacity_slider.value() / 255.0
                self._z_overlay.set_opacity(current_opacity)
            except Exception:
                pass
        
        # Automatically mark this image as checked when viewed
        self.main_window.image_window.table_model.set_checkbox_state(image_path, True)

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

        # Set the image dimensions, and current view in status bar
        self.imageLoaded.emit(*self.get_image_dimensions())
        self.viewChanged.emit(*self.get_image_dimensions())
        
        # Show loaded message in status bar
        try:
            self.main_window.status_bar.showMessage(f"Loaded image: {os.path.basename(image_path)}", 2000)
        except Exception:
            pass

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def _start_full_res_decode(self, image_path):
        """Schedule the background full-resolution decode for `image_path`.

        Deliberately debounced rather than started immediately. A decode cannot
        be interrupted once it has begun, and each one holds a full-resolution
        buffer (~1.7 GB on a 24k ortho) until it finishes. Arrow-keying through
        a folder would otherwise spawn one per image and hold all of them at
        once. Waiting for the view to settle means a user paging through images
        pays for one decode -- the one they stop on.
        """
        self._cancel_full_res_decode()
        self._pending_full_res_path = image_path
        try:
            self.main_window.status_bar.showMessage(
                f"Loading full resolution: {os.path.basename(image_path)}…"
            )
        except Exception:
            pass
        self._full_res_timer.start()

    def _launch_full_res_decode(self):
        """Timer callback: the view has settled, so decode for real."""
        from coralnet_toolbox.Rasters.QtRaster import FullResDecodeWorker

        image_path = self._pending_full_res_path
        self._pending_full_res_path = None
        if not image_path or image_path != self.current_image_path:
            return

        # Deliberately unparented: a QThread destroyed with its parent while
        # still running takes the process down with it. We hold the only
        # reference in _live_workers and drop it when the thread reports
        # finished, so the object always outlives its own run().
        worker = FullResDecodeWorker(image_path)
        worker.decoded.connect(self._on_full_res_decoded)
        worker.finished.connect(lambda w=worker: self._retire_worker(w))
        self._live_workers.add(worker)
        self._full_res_worker = worker
        worker.start()

    def _cancel_full_res_decode(self, wait_ms: int = 0):
        """Drop any scheduled or in-flight background decode.

        `wait_ms` blocks for the running decode to finish. Navigation passes 0
        — the result is simply discarded and the thread is left to wind down on
        its own. Shutdown passes a real timeout, because letting a QThread be
        destroyed with its parent while still running is how Qt hangs or
        crashes on quit.
        """
        self._pending_full_res_path = None
        try:
            self._full_res_timer.stop()
        except Exception:
            pass
        worker = getattr(self, '_full_res_worker', None)
        if worker is None:
            return
        try:
            worker.cancel()
            worker.decoded.disconnect(self._on_full_res_decoded)
        except Exception:
            pass
        if wait_ms:
            try:
                worker.wait(wait_ms)
            except Exception:
                pass
        self._full_res_worker = None

    def _retire_worker(self, worker):
        """Drop our last reference once a worker's thread has actually ended."""
        self._live_workers.discard(worker)

    def _shutdown_full_res_decode(self):
        """Wait for any background decode before the process tears down.

        Qt hangs (or crashes) if a QThread is destroyed while still running,
        and a decode cannot be interrupted mid-read. This is wired to
        QApplication.aboutToQuit rather than a closeEvent: AnnotationWindow
        lives inside a dock, so closing the main window never delivers a close
        event here.
        """
        self._cancel_full_res_decode(wait_ms=10000)
        for worker in list(self._live_workers):
            try:
                worker.cancel()
                worker.wait(10000)
            except Exception:
                pass
        self._live_workers.clear()

    def _on_full_res_decoded(self, image_path, q_image):
        """Swap the full-resolution image in, if it is still the one on screen.

        Only the base image item is replaced. Scene coordinates were already at
        full resolution while the interim image was showing, so nothing about
        the annotations, the view transform or the zoom needs to move -- the
        picture simply gets sharper in place.
        """
        self._full_res_worker = None

        if q_image is None or q_image.isNull():
            # Interim image stays on screen; it is a worse picture, not a
            # broken one, so this is a status message rather than a dialog.
            try:
                self.main_window.status_bar.showMessage(
                    f"Could not load full resolution for {os.path.basename(image_path)}", 5000)
            except Exception:
                pass
            return

        # The user may have navigated away while this was decoding.
        if image_path != self.current_image_path:
            return

        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if raster is not None:
            # Adopt it as the raster's cached image so a revisit is instant.
            raster.set_full_qimage(q_image)

        if self._base_image_item is not None:
            self._base_image_item.set_image(q_image, target_size=self._image_dimensions)
            self.viewport().update()

        try:
            self.main_window.status_bar.showMessage(
                f"Loaded image: {os.path.basename(image_path)}", 2000)
        except Exception:
            pass

    def _load_z_channel_visualization(self, raster):
        """Override to set opacity from main_window widget after BaseCanvas loads."""
        super()._load_z_channel_visualization(raster)

        # Set opacity from current slider value (preserves user's transparency preference)
        if self._z_overlay.item is not None:
            try:
                current_opacity = self.main_window.colormap_opacity_slider.value() / 255.0
                self._z_overlay.set_opacity(current_opacity)
            except Exception:
                pass

    def update_overlay_colormap(self, colormap_name):
        # Delegate to BaseCanvas and keep opacity in sync with the UI widget
        super().update_overlay_colormap(colormap_name)
        try:
            self._active_colormap_overlay.set_opacity(
                self.main_window.colormap_opacity_slider.value() / 255.0
            )
        except Exception:
            pass

    def _reset_z_channel_to_full_range(self, colormap_name=None):
        """Override to fetch colormap from main_window and pass to BaseCanvas."""
        if colormap_name is None:
            colormap_name = self.main_window.colormap_dropdown.currentText()
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
                current_colormap = self.main_window.colormap_dropdown.currentText()
                # Always call update_overlay_colormap to handle visibility correctly
                self.update_overlay_colormap(current_colormap)
                
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

        # Video frames share ONE mask_annotation across every frame, so its
        # mask_data is only valid for whichever frame was last loaded into it.
        # Re-seed it from THIS frame's cached prediction before handing it to an
        # editing tool; otherwise the first brush stroke starts from the wrong
        # pixels and the post-stroke sync overwrites the frame's cached
        # prediction (the "painting wipes the batch result" bug). The navigation
        # restore can leave the buffer wrong here, e.g. when the cached mask was
        # a different resolution than the native buffer and got dropped by the
        # shape guard — _restore_video_frame_mask_data now resizes instead, and
        # this re-seed guarantees the edit target matches the displayed frame.
        # Skipped while a predict pass is deferring syncs — that path manages the
        # shared buffer itself.
        #
        # Called even on a cache miss: _restore_video_frame_mask_data falls back
        # to the VideoRaster's durable store, which is the only place a mask
        # restored from a project file lives until its frame has been displayed.
        # (A frame with neither is a no-op there, by design.)
        if ('::frame_' in str(self.current_image_path)
                and not getattr(self, '_deferring_video_cache_sync', False)):
            _cache = getattr(self, 'batch_results_cache', {}) or {}
            self._restore_video_frame_mask_data(_cache.get(self.current_image_path))

        try:
            self.annotation_manager.register_mask_annotation(mask_annotation)
        except Exception:
            pass
        try:
            if not getattr(mask_annotation, '_signals_connected', False):
                mask_annotation.annotationUpdated.connect(self.on_annotation_updated)
                mask_annotation._signals_connected = True
        except Exception:
            pass
        if is_new:
            # Newly created annotation has no graphics item — add it to the scene now
            # so paint/fill/create operations render immediately without a load cycle.
            #
            # Building the colour canvas blocks the UI thread for seconds on a
            # large raster (a 24k ortho needs a 2.3 GB RGBA buffer), so say what
            # is happening *before* starting rather than after finishing.
            # showMessage on its own only queues a repaint that the blocked
            # event loop never reaches; repaint() puts the text on screen now.
            basename = os.path.basename(self.current_image_path)
            status_bar = None
            try:
                status_bar = self.main_window.status_bar
                status_bar.showMessage(f"Creating mask annotation for {basename}…")
                status_bar.repaint()
            except Exception:
                status_bar = None

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                if mask_annotation.graphics_item and mask_annotation.graphics_item.scene():
                    mask_annotation.graphics_item.scene().removeItem(mask_annotation.graphics_item)

                # create_graphics_item builds the colour canvas through
                # _ensure_canvas. update_graphics_item() with no rect would then
                # run _update_full_canvas and recompute that identical buffer a
                # second time — measured at 5 s of pure duplication on a 24k
                # ortho — so ask Qt to paint what was just built instead.
                mask_annotation.create_graphics_item(self.scene)
                if mask_annotation.graphics_item:
                    mask_annotation.graphics_item.setZValue(-5)
                    mask_annotation.graphics_item.update()
            finally:
                QApplication.restoreOverrideCursor()

            if status_bar is not None:
                try:
                    status_bar.showMessage(f"Mask annotation ready for {basename}", 3000)
                except Exception:
                    pass

        return mask_annotation

    def prompt_bake_or_unbake_annotations(self):
        """Offer a choice to bake vectors into the mask or unbake the mask into vectors."""
        if not self.current_image_path:
            return False

        vector_annotations = []
        for annotation in self.get_image_annotations():
            if getattr(annotation, 'is_mask_annotation', False):
                continue

            geometry_getter = getattr(annotation, 'get_rasterization_geometry', None)
            geometry = None
            if callable(geometry_getter):
                try:
                    geometry = geometry_getter()
                except Exception:
                    geometry = None

            if geometry is not None and not getattr(geometry, 'is_empty', False):
                vector_annotations.append(annotation)

        raster = None
        try:
            raster_manager = getattr(self.main_window.image_window, 'raster_manager', None)
            if raster_manager is not None:
                raster = raster_manager.get_raster(self.current_image_path)
        except Exception:
            raster = None

        mask_annotation = getattr(raster, 'mask_annotation', None) if raster is not None else None
        has_mask_regions = False
        if mask_annotation is not None:
            try:
                has_mask_regions = bool(np.any(mask_annotation.mask_data % mask_annotation.LOCK_BIT))
            except Exception:
                has_mask_regions = True

        if not vector_annotations and not has_mask_regions:
            try:
                self.main_window.status_bar.showMessage(
                    "No vector or mask annotations are available on the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

        # ------------------------------------------------------------------ #
        # Build a custom dialog so we can embed the Min Hole Area spinbox
        # alongside the Bake / Unbake choice.
        # ------------------------------------------------------------------ #
        dialog = QDialog(self)
        dialog.setWindowTitle("Convert Current Image Annotations")
        dialog.setModal(True)

        root_layout = QVBoxLayout(dialog)
        root_layout.setSpacing(12)

        # Description
        desc_label = QLabel(
            "<b>Choose how to convert annotations for the current image.</b><br>"
            "Bake rasterizes vector annotations into the mask.<br>"
            "Unbake vectorizes the current mask regions into vector annotations."
        )
        desc_label.setWordWrap(True)
        root_layout.addWidget(desc_label)

        # Unbake options group (only meaningful for Unbake)
        unbake_group = QGroupBox("Unbake Options")
        unbake_group.setEnabled(has_mask_regions)
        form_layout = QFormLayout(unbake_group)
        form_layout.setContentsMargins(8, 8, 8, 8)

        min_hole_spinbox = QSpinBox()
        min_hole_spinbox.setRange(0, 1_000_000)
        min_hole_spinbox.setValue(500)
        min_hole_spinbox.setSingleStep(100)
        min_hole_spinbox.setSuffix(" px²")
        min_hole_spinbox.setToolTip(
            "When vectorizing (unbaking) a mask, interior voids — holes — inside\n"
            "each region are traced as interior rings in the resulting polygon.\n\n"
            "Holes smaller than this area are silently filled, preventing the\n"
            "vertex explosion that comes from tracing every small gap or\n"
            "noise-level void in the mask.\n\n"
            "Holes at or above this threshold are preserved as true polygon\n"
            "holes, keeping significant voids (e.g. a sand patch inside a coral\n"
            "colony) accurately represented.\n\n"
            "0 = preserve all holes (maximum detail, most vertices).\n"
            "Higher values = fewer, larger holes kept (smoother polygons)."
        )
        min_hole_label = QLabel("Min hole area to preserve:")
        min_hole_label.setToolTip(min_hole_spinbox.toolTip())
        form_layout.addRow(min_hole_label, min_hole_spinbox)

        root_layout.addWidget(unbake_group)

        # Buttons
        button_box = QDialogButtonBox()
        bake_button = button_box.addButton("Bake", QDialogButtonBox.AcceptRole)
        unbake_button = button_box.addButton("Unbake", QDialogButtonBox.AcceptRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)

        bake_button.setEnabled(bool(vector_annotations))
        unbake_button.setEnabled(has_mask_regions)

        # Track which action button was clicked
        chosen = [None]

        def _on_bake():
            chosen[0] = "bake"
            dialog.accept()

        def _on_unbake():
            chosen[0] = "unbake"
            dialog.accept()

        bake_button.clicked.connect(_on_bake)
        unbake_button.clicked.connect(_on_unbake)
        cancel_button.clicked.connect(dialog.reject)

        root_layout.addWidget(button_box)
        dialog.setMinimumWidth(380)

        if dialog.exec_() != QDialog.Accepted or chosen[0] is None:
            return False
        if chosen[0] == "bake":
            return self.bake_vector_annotations(prompt_user=False)
        if chosen[0] == "unbake":
            return self.vectorize_mask_annotations(min_hole_area=min_hole_spinbox.value())
        return False

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
        try:
            history_action = MaskEditAction(self.current_mask_annotation, description="Protect vector annotations")

            # The MaskAnnotation handles the efficient protection marking internally
            self.current_mask_annotation.rasterize_annotations(annotations, history_action=history_action)

            if not history_action.is_empty():
                self.action_stack.push(history_action)
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def bake_vector_annotations(self, prompt_user=True):
        """Bake current-image vector annotations into the mask and delete them.

        This is the destructive counterpart to rasterize_annotations(): it
        permanently writes vector labels into the semantic mask and then removes
        the vector annotations from the current image.
        """
        if not self.current_image_path:
            return False

        annotations = []
        for annotation in self.get_image_annotations():
            if getattr(annotation, 'is_mask_annotation', False):
                continue

            geometry_getter = getattr(annotation, 'get_rasterization_geometry', None)
            geometry = None
            if callable(geometry_getter):
                try:
                    geometry = geometry_getter()
                except Exception:
                    geometry = None

            if geometry is not None and not getattr(geometry, 'is_empty', False):
                annotations.append(annotation)

        if not annotations:
            try:
                self.main_window.status_bar.showMessage(
                    "No vector annotations on the current image can be baked into the mask.",
                    3000,
                )
            except Exception:
                pass
            return False

        if prompt_user:
            reply = QMessageBox.question(
                self,
                "Bake Vector Annotations",
                "Bake all vector annotations in the current image into the mask and remove the vectors?\n\n"
                "Undo will restore both the mask pixels and the vector annotations.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                return False

        mask_annotation = self.current_mask_annotation
        if mask_annotation is None:
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Block signals at the source for the entire bake + delete operation.
            #
            # Signal blocking prevents every mid-bake canvas refresh from firing.
            # We intentionally do NOT call clear_all_annotation_overlays() here:
            # calling scene.removeItem() on QGraphicsPathItem objects that carry a
            # QGraphicsDropShadowEffect schedules a deferred paint pass for the
            # effect's source region.  When Qt services that repaint (after this
            # function returns but before the QTimer.singleShot(0) fires), it
            # accesses rendering state that has already been partially torn down →
            # C-level crash.  Leaving the overlay items in their scenes is safe
            # because no signal can rebuild or destroy them during the blocked
            # window, and _deferred_refresh_all_canvases will cleanly replace them
            # once the event loop is fully settled.
            _annotation_manager = getattr(self, 'annotation_manager', None)

            mask_annotation.blockSignals(True)
            if _annotation_manager is not None:
                _annotation_manager.blockSignals(True)

            baked_annotations = []
            skipped_annotations = []
            history_action = None
            delete_action = None
            try:
                history_action = MaskEditAction(mask_annotation, description="Bake vector annotations")
                bake_summary = mask_annotation.bake_annotations(annotations, history_action=history_action)

                baked_annotations = bake_summary.get("baked_annotations", []) if bake_summary else []
                skipped_annotations = bake_summary.get("skipped_annotations", []) if bake_summary else []

                if not baked_annotations:
                    try:
                        self.main_window.status_bar.showMessage(
                            "No vector annotations could be baked into the current mask.",
                            3000,
                        )
                    except Exception:
                        pass
                    return False

                self.unselect_annotations()

                delete_action = DeleteAnnotationsAction(self, baked_annotations)
                self.delete_annotations(baked_annotations, record_action=False)
            finally:
                if _annotation_manager is not None:
                    _annotation_manager.blockSignals(False)
                mask_annotation.blockSignals(False)

                try:
                    mask_annotation.refresh_graphics()
                    self.refresh_mask_annotation_view(mask_annotation)
                except Exception:
                    pass

            compound_action = CompoundAction(
                [history_action, delete_action],
                description="Bake vector annotations",
            )
            if history_action is not None and delete_action is not None:
                self.action_stack.push(compound_action)

            try:
                if skipped_annotations:
                    self.main_window.status_bar.showMessage(
                        f"Baked {len(baked_annotations)} vector annotations; skipped {len(skipped_annotations)} that could not be rasterized.",
                        4000,
                    )
                else:
                    self.main_window.status_bar.showMessage(
                        f"Baked {len(baked_annotations)} vector annotations into the mask.",
                        3000,
                    )
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()

        return True

    def vectorize_mask_annotations(self, min_hole_area: int = 500):
        """Convert the current image's mask regions into vector annotations.

        Args:
            min_hole_area: Minimum hole area in pixels to preserve as an
                interior ring. Holes smaller than this threshold are filled.
        """
        mask_annotation = self.current_mask_annotation
        if mask_annotation is None:
            try:
                self.main_window.status_bar.showMessage(
                    "No mask annotation is available for the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

        # Regions too small to become polygons are noise; their pixels are
        # dropped alongside the vectorized ones so nothing is left behind.
        rejected_indices = []
        try:
            vector_annotations = mask_annotation.to_vector_annotations(
                transparency=self.main_window.get_transparency_value(),
                show_confidence=False,
                min_hole_area=min_hole_area,
                rejected_indices_out=rejected_indices,
                # File them under the displayed frame, not the video: a
                # VideoRaster's mask is shared across frames and carries only
                # the video's path.
                image_path=self.current_image_path,
            )
        except Exception:
            vector_annotations = []
            rejected_indices = []

        if not vector_annotations and not rejected_indices:
            try:
                self.main_window.status_bar.showMessage(
                    "No mask regions could be vectorized from the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            _annotation_manager = getattr(self, 'annotation_manager', None)

            mask_annotation.blockSignals(True)
            if _annotation_manager is not None:
                _annotation_manager.blockSignals(True)

            add_action = None
            clear_action = None
            try:
                self.unselect_annotations()

                if vector_annotations:
                    add_action = AddAnnotationsAction(self, vector_annotations)
                    add_action.do()

                clear_action = MaskEditAction(mask_annotation, description="Vectorize mask annotations")
                mask_annotation.clear_pixels_for_annotations(
                    vector_annotations,
                    history_action=clear_action,
                    extra_flat_indices=rejected_indices,
                )
            finally:
                if _annotation_manager is not None:
                    _annotation_manager.blockSignals(False)
                mask_annotation.blockSignals(False)

                try:
                    mask_annotation.refresh_graphics()
                    self.refresh_mask_annotation_view(mask_annotation)
                except Exception:
                    pass

                # The clear ran with signals blocked, so on_annotation_updated
                # never fired and the video frame's stored/cached mask still
                # holds the pixels that were just vectorized away — they would
                # come back on the next navigation. Sync explicitly.
                try:
                    if '::frame_' in str(self.current_image_path):
                        self._sync_video_mask_to_cache()
                except Exception:
                    pass

            if clear_action is None or clear_action.is_empty():
                try:
                    if vector_annotations:
                        self.delete_annotations(vector_annotations, record_action=False)
                    self.main_window.status_bar.showMessage(
                        "No editable mask pixels were changed during vectorization.",
                        3000,
                    )
                except Exception:
                    pass
                return False

            actions = [action for action in (add_action, clear_action) if action is not None]
            if len(actions) > 1:
                self.action_stack.push(CompoundAction(
                    actions,
                    description="Vectorize mask annotations",
                ))
            else:
                self.action_stack.push(actions[0])

            try:
                message = f"Vectorized {len(vector_annotations)} mask regions into annotations."
                if rejected_indices:
                    message += f" Discarded {len(rejected_indices)} sub-threshold regions."
                self.main_window.status_bar.showMessage(message, 3000)
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()

        return True

    def unrasterize_annotations(self):
        """
        Remove protection from vector annotation pixels, allowing mask editing over those areas again.
        This clears the locked status from pixels that were protected during mask editing mode.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.current_mask_annotation:
                history_action = MaskEditAction(self.current_mask_annotation, description="Unprotect vector annotations")
                self.current_mask_annotation.unrasterize_annotations(history_action=history_action)

                if not history_action.is_empty():
                    self.action_stack.push(history_action)
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def viewportToScene(self):
        """Convert viewport coordinates to scene coordinates."""
        # Use the QRect overload (returns a QPolygonF of all 4 mapped corners)
        # instead of mapping just topLeft/bottomRight: under a rotated view
        # transform those two screen corners no longer correspond to the
        # scene's min/max x/y, which can yield a QRectF with negative
        # width/height. boundingRect() gives the correct axis-aligned bounding
        # box of the visible region for any rotation angle.
        return self.mapToScene(self.viewport().rect()).boundingRect()

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
            start_zoom = float(get_view_scale(self.transform()))
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
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            # Clean up references
            self._active_view_animations = []
            # Emit a viewChanged signal for status updates
            self.viewChanged.emit(*self.get_image_dimensions())
            # Emit the standard navigation signal after the final animated view is in place.
            self._emit_view_navigated()

        # Connect last animation finished to cleanup
        z_anim.finished.connect(_on_finished)

        # Start animations
        cx_anim.start()
        cy_anim.start()
        z_anim.start()

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
        image_width, image_height = self.get_image_dimensions()
        if not image_width:
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
        min_padding = 0.30  # More surrounding context than before
        max_padding = 0.70  # Allow up to ~2x the previous framing context
        if relative_area > 0:
            padding_factor = max(min(0.70 * (1 / math.sqrt(relative_area)), max_padding), min_padding)
        else:
            padding_factor = min_padding

        # Step 4: Apply dynamic padding with minimum values to prevent zero width/height
        min_padding_absolute = 2.0  # Minimum padding in pixels (relaxed from 1.0)
        padding_x = max(annotation_rect.width() * padding_factor, min_padding_absolute)
        padding_y = max(annotation_rect.height() * padding_factor, min_padding_absolute)
        padded_rect = annotation_rect.adjusted(-padding_x, -padding_y, padding_x, padding_y)

        # Animate the view to the padded annotation rect instead of jumping
        self.animate_to_rect(padded_rect, duration=600)
    
    def cycle_annotations(self, direction):
        """Cycle through annotations in the given direction.

        Navigates immediately on every keypress.  The animation is only shown
        when the user presses slowly (>= 500 ms since the last cycle press);
        rapid presses snap the view instantly so the UI stays responsive.
        """
        now = time.monotonic()
        use_animation = (now - getattr(self, '_last_cycle_time', 0.0)) >= 0.5
        self._last_cycle_time = now

        # Cancel any in-flight view animation so it cannot override the
        # position we are about to set (fast presses arrive before the
        # previous 500ms animation has finished).
        for _anim in getattr(self, '_active_view_animations', []):
            try:
                _anim.stop()
            except Exception:
                pass
        self._active_view_animations = []

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
                self.select_annotation(annotations[new_index])
                ann = annotations[new_index]
                if use_animation:
                    self.center_on_annotation(ann)
                else:
                    # Fast cycling: snap instantly without animation
                    if ann.center_xy:
                        self.center_on_pixel(ann.center_xy.x(), ann.center_xy.y())
                
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
            # Let unselect_annotations() request its own rebuild rather than
            # suppressing it. This used to set _skip_phantom_refresh on the
            # promise of doing it "once at the end", but the end only rebuilds
            # `annotation`'s own colour group — so annotations of every *other*
            # label that had just been deselected were left out of the phantom
            # layer, invisible until some later full rebuild happened to
            # restore them. The two requests coalesce into one flush anyway.
            self.unselect_annotations()
            
        if annotation not in self.selected_annotations:
            self.selected_annotations.append(annotation)

            annotation.select()

            # Build Qt objects if they don't exist yet
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
        
        # Skip these heavy UI operations if we are looping through hundreds of items
        if not bulk_mode:
            if len(self.selected_annotations) > 1 and not quiet_mode:
                self.main_window.label_window.deselect_active_label()
                self.main_window.confidence_window.clear_display()
            self.viewport().update()
            # Rebuild only this annotation's color group (it left the phantom layer)
            self.refresh_phantom_annotations(only_annotation=annotation)
            self._emit_selection_changed()

    def _selection_materialization_split(self, annotations):
        """Split a bulk selection into (hydrate now, leave phantom).

        Building a QGraphicsItemGroup per annotation costs more than it can
        possibly show when most of them are off screen; the far half is drawn
        by the phantom layer's selected-state group instead, and hydrated by
        _promote_visible_selected once it scrolls into view.
        """
        view = self.mapToScene(self.viewport().rect()).boundingRect()
        near, far = [], []
        for annotation in annotations:
            bbox = getattr(annotation, 'cropped_bbox', None)
            if bbox and view.intersects(QRectF(bbox[0], bbox[1],
                                               bbox[2] - bbox[0],
                                               bbox[3] - bbox[1])):
                near.append(annotation)
            else:
                far.append(annotation)
        return near, far

    def _on_view_navigated_promote(self, *_):
        """viewNavigated adapter. The signal carries centre and zoom; the
        promotion recomputes the viewport itself, so the payload is ignored."""
        self._promote_visible_selected()

    def _promote_visible_selected(self):
        """Hydrate selected annotations that have scrolled into view.

        Restores the invariant select_phantom defers: anything both selected
        and visible owns its Qt items, so resize handles, moves and cuts all
        find what they expect.
        """
        pending = [a for a in self.selected_annotations
                   if a.render_mode is RenderMode.PHANTOM]
        if not pending:
            return

        near, _ = self._selection_materialization_split(pending)
        if not near:
            return

        for annotation in near:
            annotation.select()
            if not annotation.is_graphics_item_valid():
                annotation.create_graphics_item(self.scene)
            self.set_annotation_visibility(annotation)
        self.refresh_phantom_annotations()

    def select_annotations(self):
        """Select all annotations in the current image.

        Annotations inside the viewport take the same per-item path as
        rubber-band multi-select, so what the user can see and touch is
        identical: a QGraphicsItemGroup with name tag and dimension tag. The
        rest are marked selected in the phantom layer and hydrated later by
        _promote_visible_selected when they scroll into view -- building 75,000
        Qt items to represent a selection nobody is looking at is most of the
        cost of this operation.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self._skip_phantom_refresh = True
        self.unselect_annotations()
        self._skip_phantom_refresh = False

        annotations = self.get_image_annotations()
        if not annotations:
            QApplication.restoreOverrideCursor()
            return

        label_locked = self.main_window.label_window.label_locked
        locked_label_id = self.main_window.label_window.locked_label.id if label_locked else None

        self._syncing_selection = True
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.blockSignals(True)

        eligible = [a for a in annotations
                    if not (label_locked and a.label.id != locked_label_id)]
        near, far = self._selection_materialization_split(eligible)

        for annotation in near:
            # bulk_mode=True and quiet_mode=True suppress per-item UI updates.
            self.select_annotation(annotation, multi_select=True,
                                   quiet_mode=True, bulk_mode=True)

        for annotation in far:
            # Selected, drawn selected, but no Qt items until it comes into
            # view. Appended without a membership test: unselect_annotations()
            # emptied the list above and `near` is disjoint from `far`, so a
            # duplicate is impossible -- and `x not in list` here would make
            # selecting 15k annotations quadratic.
            self.selected_annotations.append(annotation)
            annotation.select_phantom()

        self.blockSignals(False)
        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self._syncing_selection = False

        if len(self.selected_annotations) > 1:
            self.main_window.label_window.deselect_active_label()
            self.main_window.confidence_window.clear_display()

        self.refresh_phantom_annotations()
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

        # Only center/scroll when exactly ONE annotation was selected.
        # Multi-annotation selections span arbitrary locations; animating toward
        # the first item in an arbitrary list is confusing and uninformative.
        try:
            if first_selected and scroll_to_first and len(annotation_ids) == 1:
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
            
            # Destroy Qt objects before deselect() so children are removed cleanly
            self._clear_annotation_graphics_single(annotation)
            annotation.deselect()
            
            if not bulk_mode:
                if not self.selected_annotations:
                    self.main_window.confidence_window.clear_display()
                self.viewport().update()
                if not self._skip_phantom_refresh:
                    self.refresh_phantom_annotations(only_annotation=annotation)
                self._emit_selection_changed()

    def unselect_annotations(self):
        """Unselect all currently selected annotations."""
        QApplication.setOverrideCursor(Qt.WaitCursor)

        annotations_to_unselect = self.selected_annotations.copy()
        if not annotations_to_unselect:
            QApplication.restoreOverrideCursor()
            self._emit_selection_changed()
            return

        # A lazily-selected annotation (see select_phantom) never materialised
        # its own items — it is drawn by the phantom layer's selected-state
        # group. Removing it therefore needs that group rebuilt as well, which
        # the per-group incremental path does not do: it only ever rebuilds the
        # unselected-state key. So the presence of even one forces the full
        # rebuild, while the ordinary case of deselecting a handful of
        # materialised annotations stays on the cheap path (~3 ms a group
        # against ~40 ms for the whole layer at 13k annotations).
        needs_full_rebuild = any(a.render_mode is RenderMode.PHANTOM
                                 for a in annotations_to_unselect)

        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        self.selected_annotations = []
        for annotation in annotations_to_unselect:
            if self.main_window.confidence_window.isVisible():
                try:
                    annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                except TypeError:
                    pass
                try:
                    annotation.annotationUpdated.disconnect(self.on_annotation_updated)
                except TypeError:
                    pass

            self._clear_annotation_graphics_single(annotation)
            annotation.deselect()

        if self.scene:
            self.scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        self.main_window.confidence_window.clear_display()
        if not self._skip_phantom_refresh:
            if needs_full_rebuild:
                self.refresh_phantom_annotations()
            else:
                # Coalesced by _flush_phantom_refresh, which codes these down to
                # one rebuild per distinct colour group and falls back to a full
                # rebuild past a handful of them.
                for annotation in annotations_to_unselect:
                    self.refresh_phantom_annotations(only_annotation=annotation)
        self.viewport().update()
        QApplication.restoreOverrideCursor()
        self._emit_selection_changed()
    
    def _emit_selection_changed(self):
        """Emit the annotationSelectionChanged signal with current selection IDs."""
        if self._syncing_selection:
            return  # Prevent infinite loops
        selected_ids = [ann.id for ann in self.selected_annotations]
        self.annotationSelectionChanged.emit(selected_ids)

    def load_annotation(self, annotation):
        """Load a single annotation into the scene."""
        # Inject / update scale
        self.set_annotation_scale(annotation)
        
        # Remove the graphics item from its current scene if it exists
        if annotation.graphics_item and annotation.graphics_item.scene():
            annotation.graphics_item.scene().removeItem(annotation.graphics_item)

        # Update transparency to match the global slider value
        current_slider_value = self.main_window.get_transparency_value()
        annotation.update_transparency(current_slider_value)

        # Only create Qt items for already-selected annotations; others are phantom
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
        
        # Render all unselected annotations to phantom layer
        self.refresh_phantom_annotations()
        
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
            if '::frame_' in str(self.current_image_path):
                # Video frame — always use the per-frame cache.  The VideoRaster
                # holds ONE mask_annotation shared across ALL frames, so falling
                # through to the raster-level path would display the mask from
                # whichever frame was painted last on every other frame too.
                cache = getattr(self, 'batch_results_cache', {}) or {}
                cached = cache.get(self.current_image_path)
                bii = getattr(self, '_base_image_item', None)

                # Reset the shared editing target (vr.mask_annotation.mask_data)
                # to THIS frame's pixels. The displayed overlay is per-frame, but
                # the brush/fill tools edit vr.mask_annotation directly via
                # current_mask_annotation. Without this reset, painting on a new
                # frame mutates the previous frame's mask_data, so its pixels leak
                # onto every subsequent frame.
                self._restore_video_frame_mask_data(cached)

                if not cached:
                    # A frame whose mask came from the project file has pixels in
                    # the VideoRaster store but no display overlay yet. Render and
                    # cache one now: this is what makes a reopened project show
                    # its saved masks.
                    cached = self._ensure_video_frame_mask_overlay(self.current_image_path)

                if cached:
                    qimg = cached.get('mask_qimage')
                    opacity = cached.get('opacity', 128 / 255.0)
                    try:
                        if bii is not None:
                            bii.set_mask_image(qimg, opacity)
                    except Exception:
                        pass
                else:
                    # No per-frame overlay — make sure nothing from a previous
                    # frame lingers on the FastImageItem.
                    try:
                        if bii is not None:
                            bii.set_mask_image(None)
                    except Exception:
                        pass
                return  # Never fall through to raster-level mask for video frames
        except Exception:
            pass

        # Display an existing mask; never create one here.
        #
        # `current_mask_annotation` creates a MaskAnnotation on first access,
        # which is the right behaviour for a brush stroke and the wrong
        # behaviour for merely opening an image: it allocates a full-size zero
        # mask plus an RGBA colour canvas at 4 bytes per pixel, then renders it.
        # On a 24k ortho that is a 0.6 GB array and a 2.3 GB canvas, ~12 s, to
        # display nothing at all -- and it ran for every image, whether or not
        # that image had ever been painted.
        #
        # Raster.get_mask_annotation is still the lazy constructor; the editing
        # tools reach it through current_mask_annotation when a stroke actually
        # needs a buffer.
        raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
        if raster is None or raster.mask_annotation is None:
            return

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

    def _restore_video_frame_mask_data(self, cached):
        """Reset the shared VideoRaster mask annotation to a frame's cached pixels.

        The VideoRaster holds a single MaskAnnotation reused for every frame; its
        ``mask_data`` is what the brush/fill tools edit. When navigating to a
        frame we must reload that buffer from the frame's cached ``mask_arr``
        (or clear it to zeros when the frame has no mask) so edits never leak
        across frames.

        ``cached`` is the batch_results_cache entry for the current frame, or
        None when the frame has no stored mask.
        """
        try:
            vr = getattr(self, '_active_video_raster', None)
            if vr is None:
                return

            stored = cached.get('mask_arr') if cached else None
            if stored is None:
                # No display-cache entry: fall back to the VideoRaster's durable
                # per-frame store, which is what a freshly reopened project has
                # before any frame has been displayed and cached.
                frame_idx = self._video_frame_index(self.current_image_path)
                if frame_idx is not None:
                    stored = vr.get_frame_mask(frame_idx)
            else:
                # Mirror a cache-only entry into the durable store so it survives
                # a save. Writers that only know about batch_results_cache
                # therefore still persist, without each having to know about the
                # raster-level store.
                frame_idx = self._video_frame_index(self.current_image_path)
                if frame_idx is not None and vr.get_frame_mask(frame_idx) is None:
                    vr.set_frame_mask(frame_idx, stored)

            ma = getattr(vr, 'mask_annotation', None)
            if ma is None:
                # No shared mask exists yet. If this frame has cached pixels we
                # must create the buffer NOW and seed it with them. Batch
                # inference writes video predictions ONLY to batch_results_cache
                # (it never touches vr.mask_annotation), so without this the
                # buffer is created empty by the first brush stroke and the next
                # sync overwrites the frame's cached prediction — the "painting
                # wipes the batch result" bug. With no cached pixels there is
                # nothing to load, so defer creation to the edit path (a blank
                # frame correctly starts from a clean zero buffer).
                if stored is None:
                    return
                try:
                    project_labels = self.main_window.label_window.labels
                    ma = vr.get_mask_annotation(project_labels)
                except Exception:
                    return
                if ma is None:
                    return

            # Defensive: a cached prediction whose resolution differs from the
            # native edit buffer (e.g. a mask reconstructed at model size on the
            # live-preview tensor path) must NOT be silently dropped — resize it
            # to the buffer shape so it actually loads. Label maps require
            # nearest-neighbour so class IDs are preserved.
            if (stored is not None
                    and stored.shape != ma.mask_data.shape):
                try:
                    import cv2 as _cv2
                    stored = _cv2.resize(
                        stored,
                        (ma.mask_data.shape[1], ma.mask_data.shape[0]),
                        interpolation=_cv2.INTER_NEAREST,
                    )
                except Exception:
                    pass

            if stored is not None and stored.shape == ma.mask_data.shape:
                np.copyto(ma.mask_data, stored)
            else:
                # No per-frame mask for this frame.
                #
                # CRITICAL: do NOT zero mask_data here. _restore can be invoked
                # transiently for frames that are not actually being displayed
                # (e.g. mid-batch reloads) while
                # the shared mask_data legitimately holds an in-progress result
                # for another frame. Zeroing on a cache-miss destroys that work.
                #
                # Zeroing for a genuinely-blank frame is the responsibility of the
                # explicit display path (_clear_video_frame_mask_data, called from
                # _display_video_frame) and of the editing tools, which start a
                # fresh frame from a clean buffer. Here we only ever *load* known
                # cached pixels; a miss is a no-op.
                return

            # The in-memory color canvas / qimage now describe the wrong frame.
            # Invalidate them so the next display or edit rebuilds from mask_data.
            ma.canvas = None
            ma.qimage = None
            ma._invalidate_stats_cache()
        except Exception:
            pass

    def _clear_video_frame_mask_data(self):
        """Zero the shared VideoRaster mask_data for a genuinely-blank displayed frame.

        Called only from the real display path (_display_video_frame) when the
        frame being shown has no cached mask, so editing starts from a clean
        buffer. Unlike _restore_video_frame_mask_data this is allowed to zero,
        because it runs only when we are actually committing to display this
        frame — never during transient/background reloads.
        """
        try:
            vr = getattr(self, '_active_video_raster', None)
            if vr is None:
                return
            ma = getattr(vr, 'mask_annotation', None)
            if ma is None:
                return
            ma.mask_data[...] = 0
            ma.canvas = None
            ma.qimage = None
            ma._invalidate_stats_cache()
        except Exception:
            pass

    @staticmethod
    def _video_frame_index(frame_path):
        """Return the frame index encoded in a virtual frame path, or None."""
        if not frame_path or '::frame_' not in str(frame_path):
            return None
        try:
            return int(str(frame_path).rsplit('::frame_', 1)[1])
        except (ValueError, IndexError):
            return None

    def _render_video_frame_overlay(self, raster, mask_arr):
        """Colour one frame's class-ID array into a QImage the fast item can paint.

        Renders through a throwaway MaskAnnotation rather than the raster's
        shared edit buffer, because this runs for frames that are merely being
        streamed past -- touching the shared buffer would make the edit target
        follow playback.
        """
        project_labels = self.main_window.label_window.labels
        if not project_labels:
            return None, None
        try:
            temp_mask = MaskAnnotation(
                image_path=raster.image_path,
                mask_data=mask_arr,
                initial_labels=project_labels,
                rasterio_src=None,
            )
            # A fresh MaskAnnotation assigns class IDs from the label list; the
            # stored pixels were written against the shared buffer's map. Adopt
            # that map so the colours match what editing the frame would show.
            shared = getattr(raster, 'mask_annotation', None)
            if shared is not None and shared.class_id_to_label_map:
                temp_mask.class_id_to_label_map = dict(shared.class_id_to_label_map)
                temp_mask.label_id_to_class_id_map = dict(shared.label_id_to_class_id_map)
                temp_mask.visible_label_ids = set(shared.visible_label_ids)
                temp_mask.invalidate_color_map()
            temp_mask._ensure_canvas()
            if temp_mask.qimage is None:
                return None, None
            # Copy: temp_mask owns the buffer and is about to be dropped.
            return temp_mask.qimage.copy(), temp_mask.get_current_transparency() / 255.0
        except Exception as e:
            print(f"Video frame overlay render error: {e}")
            return None, None

    def _ensure_video_frame_mask_overlay(self, frame_path):
        """Return this frame's display-cache entry, rendering it if it is missing.

        The streaming and playback paths read the overlay straight out of
        batch_results_cache, which only ever gets filled by a frame passing
        through the full display path. A project that was just reopened has its
        pixels in VideoRaster._frame_masks and no cache entry at all, so those
        paths drew nothing until the user happened to seek to the frame -- which
        is why the scrub bar showed ticks for masks that would not appear until
        prev/next-annotated was pressed. Build the overlay on first request and
        cache it, so each frame pays for it once.
        """
        cache = getattr(self, 'batch_results_cache', None) or {}
        cached = cache.get(frame_path)
        if isinstance(cached, dict) and cached.get('mask_qimage') is not None:
            return cached
        if cached is not None and not isinstance(cached, dict):
            return None  # raw Ultralytics Results (detect/segment), not a mask

        frame_idx = self._video_frame_index(frame_path)
        if frame_idx is None:
            return cached
        video_path = str(frame_path).rsplit('::frame_', 1)[0]
        raster = self.main_window.image_window.raster_manager.get_raster(video_path)
        if raster is None:
            return cached

        mask_arr = cached.get('mask_arr') if isinstance(cached, dict) else None
        if mask_arr is None:
            try:
                mask_arr = raster.get_frame_mask(frame_idx)
            except AttributeError:
                return cached  # not a VideoRaster
        if mask_arr is None:
            return cached

        qimg, opacity = self._render_video_frame_overlay(raster, mask_arr)
        if qimg is None:
            return cached

        if not hasattr(self, 'batch_results_cache') or self.batch_results_cache is None:
            self.batch_results_cache = {}
        entry = {'mask_qimage': qimg, 'mask_arr': mask_arr, 'opacity': opacity}
        self.batch_results_cache[frame_path] = entry
        return entry

    def _apply_video_frame_mask_overlay(self, frame_path):
        """Push this frame's mask overlay onto the fast image item, or clear it."""
        base_image_item = getattr(self, '_base_image_item', None)
        if base_image_item is None:
            return
        try:
            cached = self._ensure_video_frame_mask_overlay(frame_path)
            if isinstance(cached, dict) and cached.get('mask_qimage') is not None:
                base_image_item.set_mask_image(cached.get('mask_qimage'),
                                               cached.get('opacity', 128 / 255.0))
            else:
                # Nothing on this frame: clear, or the previous frame's mask
                # stays painted over every frame that follows it.
                base_image_item.set_mask_image(None)
        except Exception:
            pass

    def _store_video_frame_mask(self, frame_path, mask_arr, mask_qimage, opacity):
        """Record one frame's mask in both the durable store and the display cache.

        The VideoRaster's ``_frame_masks`` is the authoritative copy — it is what
        the project file is written from — while ``batch_results_cache`` holds the
        derived overlay (a coloured QImage plus its baked-in opacity) that
        FastImageItem paints. Every writer goes through here so the two cannot
        drift apart, which is how painted frames used to be lost on save.
        """
        if not hasattr(self, 'batch_results_cache') or self.batch_results_cache is None:
            self.batch_results_cache = {}
        self.batch_results_cache[frame_path] = {
            'mask_qimage': mask_qimage,
            'mask_arr': None if mask_arr is None else mask_arr.copy(),
            'opacity': opacity,
        }

        frame_idx = self._video_frame_index(frame_path)
        if frame_idx is None:
            return
        video_path = str(frame_path).rsplit('::frame_', 1)[0]
        raster = self.main_window.image_window.raster_manager.get_raster(video_path)
        if raster is None:
            return
        try:
            raster.set_frame_mask(frame_idx, mask_arr)
        except AttributeError:
            pass  # not a VideoRaster

    def _delete_video_frame_masks(self, video_path, frame_idx=None):
        """Drop stored per-frame masks and their overlays for a video.

        Per-frame masks live in VideoRaster._frame_masks and never appear in
        image_annotations_dict — only vector annotations create keys there. So
        the frame-key recursion in delete_image_annotations cannot see them, and
        a video whose annotations are all masks (the usual result of semantic
        batch inference) has no frame keys at all: the loop iterates nothing and
        "Delete Annotations" appears to do nothing.

        Pass ``frame_idx`` to drop a single frame, or leave it None for the
        whole video.
        """
        try:
            raster = self.main_window.image_window.raster_manager.get_raster(video_path)
            if raster is None:
                return
            try:
                indices = (raster.get_frame_mask_indices() if frame_idx is None
                           else {int(frame_idx)})
            except AttributeError:
                return  # not a VideoRaster

            for idx in indices:
                raster.clear_frame_mask(idx)

            # Drop the derived display overlays too, or the next navigation
            # rebuilds the mask straight back onto the frame.
            cache = getattr(self, 'batch_results_cache', None) or {}
            if frame_idx is None:
                prefix = str(video_path) + '::frame_'
                stale = [k for k in list(cache.keys())
                         if isinstance(k, str) and k.startswith(prefix)]
            else:
                stale = [raster.make_frame_path(video_path, int(frame_idx))]
            for key in stale:
                cache.pop(key, None)

            # Clear the live overlay and the shared edit buffer when the frame
            # on screen was one of the deleted ones.
            current = str(self.current_image_path or '')
            if current.startswith(str(video_path) + '::frame_'):
                current_idx = self._video_frame_index(current)
                if frame_idx is None or current_idx == int(frame_idx):
                    base_image_item = getattr(self, '_base_image_item', None)
                    if base_image_item is not None:
                        try:
                            base_image_item.set_mask_image(None)
                        except Exception:
                            pass
                    self._clear_video_frame_mask_data()

            try:
                self._update_video_annotation_marks()
            except Exception:
                pass
        except Exception:
            pass

    def _sync_video_mask_to_cache(self, frame_path=None):
        """Store the current VideoRaster mask annotation state in batch_results_cache.

        This bridges the direct-paint / single-image-predict path (which writes to
        VideoRaster.mask_annotation) with load_mask_annotation's per-frame cache
        lookup so the painted mask is displayed when navigating back to this frame
        and is NOT shown on other frames.

        Args:
            frame_path: The virtual frame path the current ``mask_data`` belongs to.
                Defaults to ``current_image_path`` (the displayed frame), which is
                correct for interactive painting. Batch inference processes frames
                other than the displayed one and MUST pass the frame it just wrote
                so the result is cached under the right key rather than overwriting
                the displayed frame's entry.
        """
        try:
            if frame_path is None:
                frame_path = self.current_image_path
            if '::frame_' not in str(frame_path):
                return
            vr = getattr(self, '_active_video_raster', None)
            if vr is None or vr.mask_annotation is None:
                return
            ma = vr.mask_annotation
            # ma.graphics_item is often None on video frames (the scene is cleared on
            # every navigation), which causes update_graphics_item() to early-return
            # without refreshing the canvas.  Force-rebuild from mask_data so the
            # snapshot reflects the *current* frame's pixels, not a stale copy left
            # over from whichever frame last had a live graphics_item.
            #
            # _update_full_canvas both resyncs the pixels and reapplies the colour
            # table on ma.qimage, which owns its buffer -- constructing a
            # replacement QImage here would discard the canvas it just filled.
            try:
                ma._ensure_canvas()
                ma._update_full_canvas()
            except Exception:
                pass
            if ma.qimage is None:
                return
            # Deep copy so mutations to the canvas don't corrupt the cached image
            qimg_copy = ma.qimage.copy()
            opacity = ma.get_current_transparency() / 255.0
            # Ensure cache dict exists
            if not hasattr(self, 'batch_results_cache') or self.batch_results_cache is None:
                self.batch_results_cache = {}
            self._store_video_frame_mask(frame_path, ma.mask_data, qimg_copy, opacity)
            # Only push the overlay to the live fast image item when this is the
            # frame currently on screen. During batch inference the synced frame is
            # usually NOT the displayed one, and pushing it would show the wrong
            # mask over the visible frame.
            if frame_path == self.current_image_path:
                bii = getattr(self, '_base_image_item', None)
                if bii is not None:
                    try:
                        bii.set_mask_image(qimg_copy, opacity)
                    except Exception:
                        pass
            # Refresh slider tick marks and image-window annotation count so the
            # new mask frame is immediately reflected in the UI.
            try:
                self._update_video_annotation_marks()
            except Exception:
                pass
            try:
                self.main_window.image_window.update_image_annotations(frame_path)
            except Exception:
                pass
        except Exception:
            pass

    def paintEvent(self, event):
        """Settle any owed phantom rebuild before the frame is drawn.

        refresh_phantom_annotations() defers, so a caller that tears an
        annotation's own graphics out of the scene and then asks for a rebuild
        leaves it drawn by neither representation until the flush runs. That
        window is not theoretical: unselect_annotations() does exactly this and
        then calls viewport().update(), and when the sequence runs inside a
        mouse-press handler — which is the only way a user reaches it — Qt
        delivers the posted UpdateRequest *before* the zero-interval timer. The
        result was one frame with every annotation missing, held on screen for
        as long as the rebuild took (~40 ms at 13k annotations, because a bulk
        deselect always takes the full-rebuild branch).

        Flushing here fixes it for every caller at once instead of asking each
        of the twenty-odd call sites to remember the ordering, and it keeps the
        coalescing: several refresh requests in one turn still collapse into
        one rebuild, it just happens a moment earlier than the timer would have
        fired. The queued timer then finds nothing pending and does nothing.
        """
        if self._phantom_flush_scheduled:
            self._flush_phantom_refresh()
        super().paintEvent(event)

    def refresh_phantom_annotations(self, only_annotation=None):
        """Rebuild the read-only phantom layer from unselected annotations.

        Only PHANTOM-mode annotations are included; FULL-mode (selected)
        annotations own their own QGraphicsItemGroup and are excluded.
        Mask annotations and invisible-label annotations are skipped.

        When ``only_annotation`` is given and the layer already exists, only the
        single color group that annotation belongs to is rebuilt — O(group size)
        instead of O(all annotations).

        This only *records* that a rebuild is owed and schedules one for the
        end of the current event-loop turn. Twenty-odd call sites reach this
        method and several subsystems routinely react to the same user action;
        batching in the caller was a convention that had to be remembered every
        time, and forgetting it cost a full O(N) rebuild rather than anything
        visible. The deferred rebuild still lands before the next paint, so
        nothing observable changes.
        """
        if self._skip_phantom_refresh or not self.active_image:
            return

        if only_annotation is None:
            self._phantom_pending_full = True
        elif not self._phantom_pending_full:
            self._phantom_pending_annotations.append(only_annotation)
        if not self._phantom_flush_scheduled:
            self._phantom_flush_scheduled = True
            QTimer.singleShot(0, self._flush_phantom_refresh)

    def _flush_phantom_refresh(self):
        """Apply whatever phantom rebuilds accumulated this event-loop turn."""
        # The state that suppressed the refresh may have arrived after it was
        # queued — an image swap, or a bulk operation that set the skip guard.
        # Checked before the pending work is consumed: draining it first and
        # then returning would discard a rebuild that nothing else is going to
        # ask for again, leaving the layer stale indefinitely rather than for
        # one frame.
        if self._skip_phantom_refresh or not self.active_image:
            return

        self._phantom_flush_scheduled = False
        full = self._phantom_pending_full
        pending = self._phantom_pending_annotations
        self._phantom_pending_full = False
        self._phantom_pending_annotations = []

        if full:
            self._refresh_phantom_annotations_now(None)
            return

        # Collapse to distinct groups: ten annotations sharing a colour need
        # one rebuild between them, not ten identical ones.
        distinct = {}
        for annotation in pending:
            try:
                distinct.setdefault(phantom_group_key(annotation, is_selected=False),
                                    annotation)
            except (AttributeError, TypeError):
                distinct = None
                break

        # Past a handful of groups the incremental path stops paying: each one
        # walks every annotation in the image to collect its group.
        if distinct is None or len(distinct) > 3:
            self._refresh_phantom_annotations_now(None)
            return

        for annotation in distinct.values():
            self._refresh_phantom_annotations_now(annotation)

    def _refresh_phantom_annotations_now(self, only_annotation=None):
        """Do the rebuild immediately. See refresh_phantom_annotations."""
        if self._skip_phantom_refresh or not self.active_image:
            return

        if (only_annotation is not None
                and self._readonly_annotation_items
                and not hasattr(only_annotation, 'mask_data')):
            key = phantom_group_key(only_annotation, is_selected=False)
            group = [
                a for a in self.get_image_annotations()
                if not hasattr(a, 'mask_data')
                and getattr(a.label, 'is_visible', True)
                and a.render_mode is RenderMode.PHANTOM
                and phantom_group_key(a, is_selected=False) == key
            ]
            self.update_readonly_group(key, group)
            return

        phantom = [
            a for a in self.get_image_annotations()
            if not hasattr(a, 'mask_data')
            and getattr(a.label, 'is_visible', True)
            and a.render_mode is RenderMode.PHANTOM
        ]
        self.render_readonly_annotations(phantom)

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

        source_path, frame_idx = parse_frame_path(image_path)

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

        rasterio_image = None
        if frame_idx is not None:
            raster = self.main_window.image_window.raster_manager.get_raster(source_path)
            if raster is not None and hasattr(raster, 'update_shim_for_frame'):
                raster.update_shim_for_frame(frame_idx)
                rasterio_image = raster.rasterio_src

        if rasterio_image is None:
            # Normalize path for cross-drive compatibility (convert backslashes to forward slashes)
            normalized_path = str(Path(source_path).as_posix())
            rasterio_image = rasterio_open(normalized_path)

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
            try:
                self.annotation_manager.register_mask_annotation(annotation)
            except Exception:
                pass
            try:
                annotation._signals_connected = True
            except Exception:
                pass
            raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
            if raster:
                raster.mask_annotation = annotation

        # --- Conditional UI Logic (runs only if the image is visible AND label is visible) ---
        if annotation.image_path == self.current_image_path and annotation.label.is_visible:
            # ---> Skip heavy graphics if streaming inference <---
            if not getattr(self, 'is_streaming_inference', False):
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

            if isinstance(annotation, MaskAnnotation):
                try:
                    self.annotation_manager.unregister_mask_annotation(annotation)
                except Exception:
                    pass
                # Clear the raster's reference so has_mask_content returns False
                # immediately — without this, update_image_annotations (called below)
                # still sees the orphaned mask_data and keeps annotation_count at 1.
                try:
                    raster = self.main_window.image_window.raster_manager.get_raster(
                        annotation.image_path
                    )
                    if raster is not None and raster.mask_annotation is annotation:
                        raster.mask_annotation = None
                        # update_image_annotations refreshes has_mask, but it only
                        # runs when not bulk_mode — clear it here so the "Has Mask"
                        # filter cannot keep matching a raster whose mask is gone.
                        raster.has_mask = False
                except Exception:
                    pass

            annotation.delete()
            del self.annotations_dict[annotation_id]
            self.annotationDeleted.emit(annotation_id)

            if record_action:
                self.action_stack.push(DeleteAnnotationAction(self, annotation))

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

            if isinstance(ann, MaskAnnotation):
                try:
                    self.annotation_manager.unregister_mask_annotation(ann)
                except Exception:
                    pass
                
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
        # Start with canvas-visible selected annotations (current image)
        selected_set = {ann.id: ann for ann in self.selected_annotations}

        # Include cross-image annotations tracked by SelectionManager so that
        # annotations selected via the Explorer (embedding/gallery) across
        # multiple images are also deleted.
        selection_manager = getattr(self.main_window, 'selection_manager', None)
        if selection_manager and hasattr(selection_manager, 'get_selected_ids'):
            all_ids = selection_manager.get_selected_ids() or []
            annotations_dict = getattr(self, 'annotations_dict', {})
            for ann_id in all_ids:
                if ann_id not in selected_set:
                    ann = annotations_dict.get(ann_id)
                    if ann:
                        selected_set[ann_id] = ann

        selected_annotations = list(selected_set.values())
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
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)

        # For VideoRaster base paths, annotations live under ::frame_ virtual keys.
        # Recurse over every frame key so the caller doesn't need to know about them.
        if image_path not in self.image_annotations_dict:
            prefix = image_path + '::frame_'
            frame_keys = [k for k in list(self.image_annotations_dict.keys()) if k.startswith(prefix)]
            for frame_key in frame_keys:
                self.delete_image_annotations(frame_key)
            # Per-frame masks are keyed by frame index on the raster, not by a
            # path in image_annotations_dict, so the loop above never reaches
            # them — and for a mask-only video there are no frame keys to loop
            # over at all.
            self._delete_video_frame_masks(image_path)
            # If the canvas is currently displaying a frame of this video, force a full
            # reload to guarantee stale graphics items are cleared from the scene.
            if (self._active_video_raster is not None and
                    self.current_image_path and
                    self.current_image_path.startswith(prefix)):
                self._display_video_frame(self._current_frame_idx)
            if raster:
                raster.delete_mask_annotation()
                try:
                    self.main_window.image_window.update_image_annotations(image_path)
                except Exception:
                    pass
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
        frame_idx = self._video_frame_index(image_path)
        if frame_idx is not None:
            # A virtual frame path: its mask lives on the raster, not in the
            # shared buffer that delete_mask_annotation clears.
            self._delete_video_frame_masks(
                str(image_path).rsplit('::frame_', 1)[0], frame_idx=frame_idx)
        if raster:
            raster.delete_mask_annotation()
            try:
                self.main_window.image_window.update_image_annotations(image_path)
            except Exception:
                pass
        
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
            # Geometry now comes from _image_dimensions, which this path has to
            # clear itself: it calls scene.clear() rather than clear_scene(),
            # so nothing else resets it and the view would keep reporting the
            # deleted image's size.
            self._image_dimensions = None
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
        # Use the view's true uniform scale as zoom (rotation-safe)
        try:
            self._zoom = float(get_view_scale(self.view.transform()))
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
        self._zoom = max(0.0001, float(v))
        self.view.set_zoom_level(self._zoom)

    center_x = pyqtProperty(float, _get_center_x, _set_center_x)
    center_y = pyqtProperty(float, _get_center_y, _set_center_y)
    zoom = pyqtProperty(float, _get_zoom, _set_zoom)
