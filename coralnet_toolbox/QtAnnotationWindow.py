import warnings

import os
import traceback
from typing import Optional

import numpy as np

import pyqtgraph as pg
from PyQt5.QtGui import QMouseEvent, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QMessageBox, QGraphicsPixmapItem)

from coralnet_toolbox.Annotations import (
    PatchAnnotation,
    PolygonAnnotation,
    RectangleAnnotation,
    MaskAnnotation,
)

from coralnet_toolbox.Tools import (
    PanTool,
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
    ZoomTool,
    WorkAreaTool,
    ScaleTool
)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import rasterio_open

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Action:
    """Base class for undo/redo actions."""
    def do(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class AddAnnotationAction(Action):
    def __init__(self, annotation_window, annotation):
        self.annotation_window = annotation_window
        self.annotation = annotation

    def do(self):
        # Only add if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.add_annotation_from_tool(self.annotation, record_action=False)

    def undo(self):
        # Only delete if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.delete_annotation(self.annotation.id, record_action=False)


class DeleteAnnotationAction(Action):
    def __init__(self, annotation_window, annotation):
        self.annotation_window = annotation_window
        self.annotation = annotation

    def do(self):
        # Only delete if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.delete_annotation(self.annotation.id, record_action=False)

    def undo(self):
        # Only add if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.add_annotation_from_tool(self.annotation, record_action=False)


class ActionStack:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            action.undo()
            self.redo_stack.append(action)

    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            action.do()
            self.undo_stack.append(action)


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    viewChanged = pyqtSignal(int, int)  # Signal to emit when view is changed
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes
    annotationSelected = pyqtSignal(int)  # Signal to emit when annotation is selected
    annotationDeleted = pyqtSignal(str)  # Signal to emit when annotation is deleted
    annotationCreated = pyqtSignal(str)  # Signal to emit when annotation is created
    annotationModified = pyqtSignal(str)  # Signal to emit when annotation is modified

    def __init__(self, main_window, parent=None):
        """Initialize the annotation window with the main window and parent widget."""
        super().__init__(parent)
        self.main_window = main_window

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Reference to the global animation manager
        self.animation_manager = None
        self.set_animation_manager(main_window.animation_manager)

        self.annotation_size = 224
        self.transparency = 128

        self.zoom_factor = 1.0
        self.pan_active = False
        self.pan_start = None
        self.drag_start_pos = None
        self.cursor_annotation = None

        self.annotations_dict = {}  # Dictionary to store annotations by UUID
        self.image_annotations_dict = {}  # Dictionary to store annotations by image path

        self.selected_annotations = []  # Stores the selected annotations
        self.rasterized_annotations_cache = []  # Caches vector annotations during mask mode
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state
                
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.pixmap_image = None
        self.rasterio_image = None
        self.active_image = False
        self.current_image_path = None
        
        # Z-channel visualization attributes
        self.z_item = None  # QGraphicsPixmapItem for Z-channel visualization
        self.dynamic_z_scaling = False  # State flag for dynamic range scaling
        self.z_data_raw = None  # Raw Z-channel data
        self.z_data_normalized = None  # Normalized (0-255) Z-channel data
        self.z_data_min = None  # Minimum value of raw Z-data
        self.z_data_max = None  # Maximum value of raw Z-data
        self.z_data_shape = None  # Shape of Z-data array
        
        # Debounce timer for dynamic range updates (prevents lag during zoom)
        self.dynamic_range_timer = QTimer()
        self.dynamic_range_timer.setSingleShot(True)
        self.dynamic_range_timer.timeout.connect(self.update_dynamic_range)
        self.dynamic_range_update_delay = 100  # milliseconds

        # Connect signals to slots
        self.toolChanged.connect(self.set_selected_tool)
        
        self.tools = {
            "pan": PanTool(self),
            "zoom": ZoomTool(self),
            "select": SelectTool(self),
            "patch": PatchTool(self),
            "rectangle": RectangleTool(self),
            "polygon": PolygonTool(self),
            "sam": SAMTool(self),
            "see_anything": SeeAnythingTool(self),
            "work_area": WorkAreaTool(self),
            "scale": ScaleTool(self),
            "brush": BrushTool(self),
            "fill": FillTool(self),
            "erase": EraseTool(self),
            "dropper": DropperTool(self)
        }

        # Defines which tools trigger mask mode
        self.mask_tools = {"brush", "fill", "erase", "dropper"}

        # Initialize the action stack for undo/redo
        self.action_stack = ActionStack()
        
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
    
    def resizeEvent(self, event):
        """Handle resize events to maintain proper view fitting."""
        super().resizeEvent(event)
        
        # Only fit view if we have an active image
        if self.active_image and self.pixmap_image and self.scene:
            # No zoom tool or hasn't been used, safe to fit
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

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
        # Handle zooming with the mouse wheel
        if self.selected_tool and event.modifiers() & Qt.ControlModifier:
            self.tools[self.selected_tool].wheelEvent(event)
        elif self.active_image:
            self.tools["zoom"].wheelEvent(event)

        self.viewChanged.emit(*self.get_image_dimensions())
        
        # Debounce dynamic Z-range update during zoom (prevents stuttering)
        self.schedule_dynamic_range_update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for the active tool."""        
        # Panning should be active in both modes, so we call it first.
        if self.active_image:
            self.tools["pan"].mousePressEvent(event)

        # Check if a tool is selected before proceeding
        if self.selected_tool:
            # If the selected tool is a mask tool, delegate the event to it
            if self.selected_tool in self.mask_tools:
                self.tools[self.selected_tool].mousePressEvent(event)
            # Otherwise, use the original logic for vector annotation tools
            else:
                self.tools[self.selected_tool].mousePressEvent(event)
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement events for the active tool."""
        # Panning should be active in both modes
        if self.active_image:
            self.tools["pan"].mouseMoveEvent(event)

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

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for the active tool."""
        # Panning should be active in both modes
        if self.active_image:
            self.tools["pan"].mouseReleaseEvent(event)

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
        
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard press events including undo/redo and deletion of selected annotations."""
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

    def set_selected_tool(self, tool):
        """Set the currently active tool and update the UI layers for the correct editing mode."""
        
        previous_tool = self.selected_tool
        
        if self.selected_tool:
            self.tools[self.selected_tool].stop_current_drawing()
            self.tools[self.selected_tool].deactivate()
            
        if tool is None or tool not in self.tools:
            self.selected_tool = None
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
        if is_entering_mask_mode or is_leaving_mask_mode:
            self.unselect_annotations()
        # --------------------------------------------------------
        
        if self.selected_tool:
            self.tools[self.selected_tool].activate()
        
        # Unselect annotations unless we are in select mode.
        if self.selected_tool != "select":
            self.unselect_annotations()

        self.toggle_cursor_annotation()
        
    def set_selected_label(self, label):
        """Set the currently selected label and update selected annotations if needed."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        self.selected_label = label
        
        # Handle both valid labels and None (no label selected)
        if label is not None:

            for annotation in self.selected_annotations:
                if annotation.label.id != label.id:
                    annotation.update_user_confidence(self.selected_label)
                    annotation.create_cropped_image(self.rasterio_image)
                    self.main_window.confidence_window.display_cropped_image(annotation)

            if self.cursor_annotation:
                if self.cursor_annotation.label.id != label.id:
                    self.toggle_cursor_annotation()
        else:
            # Clear cursor annotation when no label is selected
            if self.cursor_annotation:
                self.toggle_cursor_annotation()
                
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
            # Disconnect the confidence window from the annotation, so it won't update while moving
            annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
            annotation.update_location(new_center_xy)
            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
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
            annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)

            if isinstance(annotation, PatchAnnotation):
                annotation.update_annotation_size(self.annotation_size)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(self.annotation_size)
            elif isinstance(annotation, RectangleAnnotation):
                scale_factor = 1 + delta / 100.0
                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)
            elif isinstance(annotation, PolygonAnnotation):
                scale_factor = 1 + delta / 100.0
                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)

            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
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
            
    def clear_scene(self):
        """Clear the graphics scene and reset related variables."""
        # Clean up
        self.unselect_annotations()

        # Nullify graphics_item references for all annotations to prevent stale references
        for annotation in self.annotations_dict.values():
            if hasattr(annotation, 'graphics_item'):
                annotation.graphics_item = None

        # Clear the previous scene and delete its items
        if self.scene:
            for item in self.scene.items():
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
                    del item
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

    def display_image(self, q_image):
        """Display a QImage in the annotation window without setting it."""
        # Clean up
        self.clear_scene()

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
        
        # Calculate GDIs for Windows if needed
        self.main_window.check_windows_gdi_count()
        
        # Stop any current drawing operation before switching images
        if self.selected_tool and self.selected_tool in self.tools:
            self.tools[self.selected_tool].stop_current_drawing()
            if self.selected_tool == "scale":
                self.main_window.untoggle_all_tools()
                    
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
            
        low_res_pixmap = QPixmap.fromImage(low_res_qimage)
        
        # Add the base image pixmap and set its Z-value
        base_image_item = QGraphicsPixmapItem(low_res_pixmap)
        base_image_item.setZValue(-10)
        self.scene.addItem(base_image_item)

        # Fit the low-res image in view
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # Force Qt to process events and redraw the screen with the low-res image
        QApplication.processEvents()
        
        # Update the rasterio image source for cropping annotations
        self.rasterio_image = raster.rasterio_src
        # Get QImage and convert to QPixmap for display
        q_image = raster.get_qimage()
        if q_image is None or q_image.isNull():
            self.main_window.image_window.show_error(
                "Image Loading Error",
                f"Image {os.path.basename(image_path)} full resolution could not be loaded."
            )
            QApplication.restoreOverrideCursor()
            return  # Failed to load full res, but preview is still visible

        # Convert and set the QPixmap
        self.pixmap_image = QPixmap.fromImage(q_image)
        self.current_image_path = image_path
        self.active_image = True

        # --- SWAP IN FULL-RES PIXMAP (NO SCENE CLEAR) ---
        base_image_item.setPixmap(self.pixmap_image)

        # Load and display Z-channel if available
        self._load_z_channel_visualization(raster)
        
        # Apply the current colormap selection to the newly loaded z_item
        current_colormap = self.main_window.z_colormap_dropdown.currentText()
        if current_colormap != "None":
            self.update_z_colormap(current_colormap)
        
        # Automatically mark this image as checked when viewed
        raster.checkbox_state = True
        self.main_window.image_window.table_model.update_raster_data(image_path)
        
        # Update the zoom tool's state
        self.tools["zoom"].reset_zoom()
        # Re-fit the view to the new, full-res pixmap
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.tools["zoom"].calculate_min_zoom()

        # Toggle the cursor annotation
        self.toggle_cursor_annotation()

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
        
        # Restore cursor
        QApplication.restoreOverrideCursor()

    def _load_z_channel_visualization(self, raster):
        """
        Load and initialize the Z-channel visualization using QGraphicsPixmapItem.
        Uses native Qt rendering instead of PyQtGraph for compatibility.
        
        Args:
            raster: The Raster object containing Z-channel data
        """
        # Clean up old z_item if it exists (it should already be removed by clear_scene, but be safe)
        if self.z_item is not None:
            # Only try to remove if it's actually in this scene
            if self.z_item.scene() == self.scene:
                self.scene.removeItem(self.z_item)
            self.z_item = None
        
        # Check if Z-channel data is available
        if raster.z_channel_lazy is None:
            return
        
        try:
            z_data = raster.z_channel_lazy
            
            # Store raw Z-channel data for dynamic range calculations
            self.z_data_raw = z_data.copy()
            self.z_data_shape = z_data.shape
            
            # Normalize the Z-channel data to 0-255 range for colormap
            # This handles both float32 and uint8 data
            if z_data.dtype == np.float32:
                # For float32, normalize to 0-255 range
                self.z_data_min = np.nanmin(z_data)
                self.z_data_max = np.nanmax(z_data)
                if self.z_data_min == self.z_data_max:
                    z_norm = np.zeros_like(z_data, dtype=np.uint8)
                else:
                    z_diff = self.z_data_max - self.z_data_min
                    z_norm = (
                        (z_data - self.z_data_min) / z_diff * 255
                    ).astype(np.uint8)
            else:
                # For uint8, use as-is
                self.z_data_min = np.nanmin(z_data)
                self.z_data_max = np.nanmax(z_data)
                z_norm = z_data.astype(np.uint8)
            
            # Store normalized data for colormap application
            self.z_data_normalized = z_norm
            
            # Create QImage from uint8 grayscale data
            h, w = z_norm.shape
            z_copy = np.ascontiguousarray(z_norm)
            q_img = QImage(z_copy.data, w, h, w,
                           QImage.Format_Grayscale8)
            
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_img)
            
            # Get base image dimensions for proper scaling
            pixmap_width = self.pixmap_image.width()
            pixmap_height = self.pixmap_image.height()
            
            # Scale pixmap to match base image if needed
            width_mismatch = pixmap.width() != pixmap_width
            height_mismatch = pixmap.height() != pixmap_height
            if width_mismatch or height_mismatch:
                pixmap = pixmap.scaled(
                    pixmap_width, pixmap_height,
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            
            # Create graphics item (native Qt, compatible)
            self.z_item = QGraphicsPixmapItem(pixmap)
            
            # Position at origin to align with base image
            self.z_item.setPos(0, 0)
            
            # Set Z-value between base image (-10) and annotations (0+)
            self.z_item.setZValue(2)
            
            # Set initial opacity (half transparent)
            self.z_item.setOpacity(0.5)
            
            # Add to scene
            self.scene.addItem(self.z_item)
            
            # Initially hide until a colormap is selected
            self.z_item.hide()
            
        except Exception:
            # Z-channel visualization failure is non-critical
            traceback.print_exc()
            self.z_item = None

    def update_z_colormap(self, colormap_name):
        """
        Update the Z-channel visualization colormap.
        
        Args:
            colormap_name (str): Name of the colormap
                (e.g., 'Viridis', 'Plasma', or 'None')
        """
        if self.z_item is None or self.z_data_normalized is None:
            return
        
        try:
            if colormap_name == 'None':
                # Hide the Z-channel visualization
                self.z_item.hide()
            else:
                # Get the colormap from pyqtgraph
                colormap = pg.colormap.get(colormap_name)
                
                # Get the lookup table (0-255 -> RGB)
                lut = colormap.getLookupTable(nPts=256)
                
                # Apply LUT to normalized data
                z_colored = lut[self.z_data_normalized]
                
                # Create QImage from RGB data
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 3, QImage.Format_RGB888)
                
                # Convert to QPixmap and set in scene item
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
                
                # Show the visualization
                self.z_item.show()
                
                # Update dynamic range if enabled
                if self.dynamic_z_scaling:
                    self.update_dynamic_range()
        except Exception:
            # Colormap application failure is non-critical
            traceback.print_exc()

    def toggle_dynamic_z_scaling(self, enabled):
        """
        Toggle dynamic Z-range scaling based on visible area.
        
        Args:
            enabled (bool): Whether to enable dynamic scaling
        """
        self.dynamic_z_scaling = enabled
        
        if enabled and self.z_item is not None:
            # Immediately update to current view range
            self.update_dynamic_range()
        elif not enabled and self.z_item is not None:
            # Reset to full range visualization when disabled
            self._reset_z_channel_to_full_range()

    def _reset_z_channel_to_full_range(self):
        """
        Reset the Z-channel visualization to show the full data range (0-255).
        Used when dynamic scaling is disabled to restore the full-range view.
        """
        z_item_valid = self.z_item is not None
        z_data_valid = self.z_data_normalized is not None
        raw_data_valid = self.z_data_raw is not None
        if not (z_item_valid and z_data_valid and raw_data_valid):
            return
        
        try:
            # Get current colormap name from main window
            colormap_name = (
                self.main_window.z_colormap_dropdown.currentText())
            
            if colormap_name != 'None':
                # Apply the colormap to the normalized data (full range)
                colormap = pg.colormap.get(colormap_name)
                lut = colormap.getLookupTable(nPts=256)
                
                # Apply LUT directly to normalized data without rescaling
                z_colored = lut[self.z_data_normalized]
                
                # Create QImage from RGB data
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 3,
                               QImage.Format_RGB888)
                
                # Convert to QPixmap and update scene item
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
        except Exception:
            # Reset failure is non-critical
            import traceback
            traceback.print_exc()

    def clear_z_channel_visualization(self, image_path):
        """
        Clear the Z-channel visualization from the scene.
        Only clears if the removed z-channel belongs to the currently displayed image.
        
        Args:
            image_path (str): Path of the raster with removed z-channel
        """
        # Only clear if the removed z-channel belongs to the currently displayed image
        if image_path != self.current_image_path:
            return
        
        if self.z_item is not None:
            # Only try to remove if it's actually in this scene
            if self.z_item.scene() == self.scene:
                self.scene.removeItem(self.z_item)
            self.z_item = None
        
        # Clear all cached z-channel data
        self.z_data_raw = None
        self.z_data_normalized = None
        self.z_data_min = None
        self.z_data_max = None
        self.z_data_shape = None

    def schedule_dynamic_range_update(self):
        """
        Schedule a dynamic range update with debouncing.
        This prevents rapid updates during zoom/pan operations that would cause stuttering.
        Multiple calls within the debounce window are consolidated into a single update.
        """
        if not self.dynamic_z_scaling:
            return
        
        # Restart the timer (cancels any pending update and schedules a new one)
        self.dynamic_range_timer.stop()
        self.dynamic_range_timer.start(self.dynamic_range_update_delay)

    def update_dynamic_range(self):
        """
        Calculate and apply min/max Z values based on visible pixels.
        This reveals hidden detail by adjusting contrast dynamically.
        """
        z_item_valid = self.z_item is not None
        z_data_valid = self.z_data_normalized is not None
        if not (z_item_valid and self.dynamic_z_scaling and z_data_valid):
            return
        
        try:
            # Get normalized Z-channel data
            z_data = self.z_data_normalized
            if z_data is None:
                return
            
            # Get visible viewport area in scene coordinates
            visible_rect = (
                self.mapToScene(self.viewport().rect()).boundingRect())
            
            # Convert scene rect to image coordinates
            x1 = max(0, int(visible_rect.left()))
            y1 = max(0, int(visible_rect.top()))
            x2 = min(z_data.shape[1], int(visible_rect.right()))
            y2 = min(z_data.shape[0], int(visible_rect.bottom()))
            
            # Ensure we have a valid region
            if x1 >= x2 or y1 >= y2:
                return
            
            # Extract visible region and calculate min/max
            visible_region = z_data[y1:y2, x1:x2]
            z_vis_min = np.nanmin(visible_region)
            z_vis_max = np.nanmax(visible_region)
            
            # Avoid division by zero if all values are the same
            if z_vis_min == z_vis_max:
                z_vis_max = z_vis_min + 1
            
            # Get current colormap name from main window
            colormap_name = (
                self.main_window.z_colormap_dropdown.currentText())
            
            if colormap_name != 'None':
                # Get the colormap and apply with adjusted range
                colormap = pg.colormap.get(colormap_name)
                lut = colormap.getLookupTable(nPts=256)
                
                # Create a rescaled version of the normalized data
                # to span the full 0-255 range based on visible min/max
                z_rescaled = (
                    (z_data - z_vis_min) / (z_vis_max - z_vis_min) * 255
                ).astype(np.uint8)
                
                # Apply LUT to rescaled data
                z_colored = lut[z_rescaled]
                
                # Create QImage from RGB data
                h, w = z_colored.shape[:2]
                z_copy = np.ascontiguousarray(z_colored)
                q_img = QImage(z_copy.data, w, h, w * 3,
                               QImage.Format_RGB888)
                
                # Convert to QPixmap and update scene item
                pixmap = QPixmap.fromImage(q_img)
                self.z_item.setPixmap(pixmap)
            
        except Exception:
            # Dynamic range update failure is non-critical
            import traceback
            traceback.print_exc()

    def update_current_image_path(self, image_path):
        """Update the current image path being displayed."""
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
                if current_colormap != "None":
                    self.update_z_colormap(current_colormap)
    
    @property
    def current_mask_annotation(self) -> Optional[MaskAnnotation]:
        """A helper property to get the MaskAnnotation for the currently active image."""
        if not self.current_image_path:
            return None
        raster = self.main_window.image_window.raster_manager.get_raster(self.current_image_path)
        if not raster:
            return None
        
        # This will get the existing mask or create it on the first call
        project_labels = self.main_window.label_window.labels
        mask_annotation = raster.get_mask_annotation(project_labels)
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

    def get_image_dimensions(self):
        """Get the dimensions of the currently loaded image."""
        if self.pixmap_image:
            return self.pixmap_image.size().width(), self.pixmap_image.size().height()
        return 0, 0
    
    def center_on_work_area(self, work_area):
        """Center the view on the specified work area."""
        # Create graphics item if it doesn't exist
        if not work_area.graphics_item:
            work_area.create_graphics_item(self.scene)

        # Get the bounding rect of the work area in scene coordinates
        work_area_rect = work_area.graphics_item.boundingRect()
        work_area_center = work_area_rect.center()

        # Center the view on the work area's center
        self.centerOn(work_area_center)

    def center_on_annotation(self, annotation):
        """Center the view on the specified annotation."""
        # Create graphics item if it doesn't exist
        if not annotation.graphics_item:
            annotation.create_graphics_item(self.scene)

        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()
        annotation_center = annotation_rect.center()

        # Center the view on the annotation's center
        self.centerOn(annotation_center)
    
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

        # Fit the padded annotation rect in the view
        self.fitInView(padded_rect, Qt.KeepAspectRatio)

        # Update the zoom factor based on the new view transformation with safety checks
        view_rect = self.viewport().rect()
        if padded_rect.width() > 0:
            zoom_x = view_rect.width() / padded_rect.width()
        else:
            zoom_x = 1.0  # Default zoom if width is zero

        if padded_rect.height() > 0:
            zoom_y = view_rect.height() / padded_rect.height()
        else:
            zoom_y = 1.0  # Default zoom if height is zero

        # Relax the zoom by capping the maximum zoom factor
        max_zoom = 4.0  # Do not zoom in more than 4x
        self.zoom_factor = min(min(zoom_x, zoom_y), max_zoom)

        # Signal that the view has changed
        self.viewChanged.emit(*self.get_image_dimensions())
    
    def cycle_annotations(self, direction):
        """Cycle through annotations in the specified direction."""
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

    def select_annotation(self, annotation, multi_select=False, quiet_mode=False):
        """Select an annotation and update the UI accordingly."""
        # If the annotation is already selected and Ctrl is pressed, unselect it
        if annotation in self.selected_annotations and multi_select:
            self.unselect_annotation(annotation)
            return
        
        # If not adding to selection (Ctrl not pressed), deselect all others first
        if not multi_select:
            self.unselect_annotations()
            
        # Only add if not already selected (shouldn't happen after the checks above, but just to be safe)
        if annotation not in self.selected_annotations:
            # Add to selection
            self.selected_annotations.append(annotation)
            annotation.select()
            
            # Update UI state
            self.selected_label = annotation.label
            
            # Emit signal for annotation selection
            self.annotationSelected.emit(annotation.id)
            
            # If this is the only selected annotation, update label window and confidence window
            if len(self.selected_annotations) == 1:
                if not quiet_mode:
                    # Emit the label selected signal, unless in quiet mode.
                    # This is in Explorer to avoid overwriting preview label.
                    self.labelSelected.emit(annotation.label.id)
                
                # Make sure we have a cropped image
                if not annotation.cropped_image:
                    annotation.create_cropped_image(self.rasterio_image)
                
                # Display in confidence window
                annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
                self.main_window.confidence_window.display_cropped_image(annotation)
        
        # Special handling for multiple selected annotations
        if len(self.selected_annotations) > 1:
            self.main_window.label_window.deselect_active_label()
            self.main_window.confidence_window.clear_display()
        
        # Set the current visibility of the annotation
        self.set_annotation_visibility(annotation)
        # Always update the viewport
        self.viewport().update()
        
    def select_annotations(self):
        """Select all annotations in the current image."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # First unselect any currently selected annotations
        self.unselect_annotations()
        
        # Get all annotations in the current image
        annotations = self.get_image_annotations()
        
        # Check if label is locked
        label_locked = self.main_window.label_window.label_locked
        locked_label_id = self.main_window.label_window.locked_label.id if label_locked else None
        
        # Select all appropriate annotations
        for annotation in annotations:
            # Skip annotations that don't match the locked label
            if label_locked and annotation.label.id != locked_label_id:
                continue
                
            # Use multi_select=True to add to selection without clearing
            self.select_annotation(annotation, multi_select=True)

        # Make cursor normal again
        QApplication.restoreOverrideCursor()

    def unselect_annotation(self, annotation):
        """Unselect a specific annotation."""
        if annotation in self.selected_annotations:
            # Remove from selected list
            self.selected_annotations.remove(annotation)
            
            # Disconnect from confidence window if needed
            if hasattr(annotation, 'annotationUpdated') and self.main_window.confidence_window.isVisible():
                try:
                    annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
                except TypeError:
                    # Already disconnected
                    pass
            
            # Update annotation's internal state
            annotation.deselect()
            # Set the current visibility of the annotation
            self.set_annotation_visibility(annotation)
            
            # Clear confidence window if no annotations remain selected
            if not self.selected_annotations:
                self.main_window.confidence_window.clear_display()
            
            # Update the viewport
            self.viewport().update()

    def unselect_annotations(self):
        """Unselect all currently selected annotations."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
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
                    # Already disconnected
                    pass
            
            # Update annotation's internal state
            annotation.deselect()
            # Set the visibility of the annotation
            self.set_annotation_visibility(annotation)
        
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        
        # Update the viewport once for all changes
        self.viewport().update()
        
        # Make cursor normal again
        QApplication.restoreOverrideCursor()

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

        # Create the graphics item (scene previously cleared)
        annotation.create_graphics_item(self.scene)
        # Set the visibility based on the label's visibility checkbox
        self.set_annotation_visibility(annotation)
        
        # Connect essential update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        
        # Update the view
        self.viewport().update()

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
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        # Update the label window tool tips (this might need to be optimized later)
        self.main_window.label_window.update_tooltips()
        
        QApplication.processEvents()
        self.viewport().update()

    def load_mask_annotation(self):
        """Load the mask annotation for the current image, if it exists."""
        if not self.current_image_path:
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
        
        # If this is a MaskAnnotation, update the raster's reference to it
        if isinstance(annotation, MaskAnnotation):
            raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
            if raster:
                raster.mask_annotation = annotation

        # --- Conditional UI Logic (runs only if the image is visible AND label is visible) ---
        if annotation.image_path == self.current_image_path and annotation.label.is_visible:
            # Create graphics item for display in the scene
            if not annotation.graphics_item:
                annotation.create_graphics_item(self.scene)
                
            # Set the visibility based on the current UI state (will respect label checkbox)
            self.set_annotation_visibility(annotation)

        # --- Finalization ---
        # Update the annotation count in the ImageWindow table (always, regardless of visibility)
        self.main_window.image_window.update_image_annotations(annotation.image_path)

        # If requested, record this single addition as an undo-able action
        if record_action:
            self.action_stack.push(AddAnnotationAction(self, annotation))
        
        # Emit the signal that an annotation was created
        self.annotationCreated.emit(annotation.id)
        
    def add_annotations(self, annotations_list: list):
        """
        Efficiently adds a list of annotations to the data models and then
        updates the relevant UI components in a single batch.
        """
        if not annotations_list:
            return

        # Use a set to efficiently track unique image paths that need updating
        images_to_update = set()

        for annotation in annotations_list:
            if annotation is None or annotation.id in self.annotations_dict:
                continue

            # --- Core Logic: Only update data dictionaries ---
            self.annotations_dict[annotation.id] = annotation
            if annotation.image_path not in self.image_annotations_dict:
                self.image_annotations_dict[annotation.image_path] = []
            self.image_annotations_dict[annotation.image_path].append(annotation)

            # Track the image path for a final UI update
            images_to_update.add(annotation.image_path)

            # --- Connect signals for future interaction ---
            annotation.selected.connect(self.select_annotation)
            annotation.annotationDeleted.connect(self.delete_annotation)

        # --- Final UI Updates (after all annotations are processed) ---
        if images_to_update:
            # Update the annotation count in the ImageWindow table for each affected image
            for path in images_to_update:
                self.main_window.image_window.update_image_annotations(path)
            
            # Update the global annotation counts in the LabelWindow once
            self.main_window.label_window.update_annotation_count()

    def delete_annotation(self, annotation_id, record_action=True):
        """Delete an annotation by its ID from dicts."""
        # Check if the annotation ID exists
        if annotation_id in self.annotations_dict:
            # Get the annotation from dict
            annotation = self.annotations_dict[annotation_id]
            # Unselect the annotation (if selected)
            self.unselect_annotation(annotation)

            # Check if the annotation image is still in the image annotations dict (key)
            if annotation.image_path in self.image_annotations_dict:
                # Check if the annotation itself is in the image annotations dict (value)
                if annotation in self.image_annotations_dict[annotation.image_path]:
                    # Remove it from the image annotations dict
                    self.image_annotations_dict[annotation.image_path].remove(annotation)

            # Delete the annotation
            annotation.delete()
            # Remove the annotation from the annotations dict
            del self.annotations_dict[annotation_id]
            # Emit the annotation deleted signal
            self.annotationDeleted.emit(annotation_id)
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

            # Record action for undo/redo if requested
            if record_action:
                self.action_stack.push(DeleteAnnotationAction(self, annotation))

    def delete_annotations(self, annotations):
        """Delete a list of annotations."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        for annotation in annotations:
            self.delete_annotation(annotation.id)
            
        # Make cursor normal again
        QApplication.restoreOverrideCursor()

    def delete_selected_annotations(self):
        """Delete all currently selected annotations."""
        # Get the selected annotations
        selected_annotations = self.selected_annotations.copy()
        # Unselect them first
        self.unselect_annotations()
        # Delete each selected annotation
        self.delete_annotations(selected_annotations)

    def delete_label_annotations(self, label):
        """Delete all annotations with the specified label."""
        labeled_annotations = []
        for annotation in self.annotations_dict.values():
            if annotation.label.id == label.id:
                labeled_annotations.append(annotation)
                
        # Delete the labeled annotations
        self.delete_annotations(labeled_annotations)

    def delete_image_annotations(self, image_path):
        """Delete all annotations associated with a specific image path."""
        if image_path in self.image_annotations_dict:
            # Check if a label is locked
            label_locked = self.main_window.label_window.label_locked
            locked_label_id = self.main_window.label_window.locked_label.id if label_locked else None
            
            # Create a copy of annotations to safely iterate
            annotations = list(self.image_annotations_dict[image_path].copy())
            annotations_to_delete = []
            
            # Filter annotations based on locked label
            for annotation in annotations:
                # Skip annotations with locked label
                if label_locked and annotation.label.id == locked_label_id:
                    continue
                
                # Add to delete list
                annotations_to_delete.append(annotation)
            
            # Delete filtered annotations
            self.delete_annotations(annotations_to_delete)
            
            # If all annotations were deleted, remove the image path from the dictionary
            if not self.image_annotations_dict.get(image_path, []):
                del self.image_annotations_dict[image_path]
            
        # Clear the mask_annotation to ensure semantic segmentation data is reset
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if raster:
            raster.delete_mask_annotation()
        
        # Always update the viewport
        self.scene.update()
        self.viewport().update()

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
