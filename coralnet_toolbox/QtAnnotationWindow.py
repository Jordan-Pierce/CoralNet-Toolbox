import warnings

from typing import Optional

from rtree import index

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QGraphicsPixmapItem)

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
    SAMTool,
    SeeAnythingTool,
    SelectTool,
    ZoomTool,
    WorkAreaTool
)

from coralnet_toolbox.Common.QtGraphicsUtility import GraphicsUtility

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import rasterio_open

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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

        self.annotation_size = 224
        self.annotation_color = None
        self.transparency = 128

        self.zoom_factor = 1.0
        self.pan_active = False
        self.pan_start = None
        self.drag_start_pos = None
        self.cursor_annotation = None
        
        # Create an R-tree index property for 2D spatial data.
        p = index.Property()
        p.dimension = 2
        # The index will store annotation objects directly.
        p.object_capacity = 10 
        # Create the index instance. It will be populated when annotations are loaded.
        self.spatial_index = index.Index(properties=p)

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

        # Initialize the graphics utility class for standardized visual elements
        self.graphics_utility = GraphicsUtility()

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
            "brush": BrushTool(self),
            "fill": FillTool(self),
            "erase": EraseTool(self)
        }
        # Defines which tools trigger mask mode
        self.mask_tools = {"brush", 
                           "fill", 
                           "erase"}  

    def _is_in_mask_editing_mode(self):
        """Check if the annotation window is currently in mask editing mode."""
        return self.selected_tool and self.selected_tool in self.mask_tools

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
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard press events including deletion of selected annotations."""
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

        # Switch between mask editing mode and vector annotation mode
        if self.selected_tool in self.mask_tools and previous_tool not in self.mask_tools:
            mask_anno = self.current_mask_annotation
            vector_annos = self.get_image_annotations()
            
            if mask_anno and vector_annos:
                # This ensures new vector annotations "erase" any underlying mask data.
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    mask_anno.clear_pixels_for_annotations(vector_annos)
                finally:
                    QApplication.restoreOverrideCursor()

            self.unselect_annotations()
        
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
        self.annotation_color = label.color

        for annotation in self.selected_annotations:
            if annotation.label.id != label.id:
                annotation.update_user_confidence(self.selected_label)
                annotation.create_cropped_image(self.rasterio_image)
                self.main_window.confidence_window.display_cropped_image(annotation)

        if self.cursor_annotation:
            if self.cursor_annotation.label.id != label.id:
                self.toggle_cursor_annotation()
                
        # Make cursor normal again
        QApplication.restoreOverrideCursor()

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
        """Set the visibility of an annotation and update its graphics item.
        
        Args:
            annotation: The annotation to update
            force_visibility: If provided, force this visibility state regardless of hide button.
                            If None, use hide button state.
        """
        # Determine visibility based on force_visibility or hide button state
        if force_visibility is not None:
            visible = force_visibility
        else:
            visible = not self.main_window.hide_action.isChecked()
        
        if visible:
            # Show the annotation
            annotation.set_visibility(True)
            # Update transparency to match the annotation's own label transparency (not active label)
            if not hasattr(annotation, 'mask_data'):  # Vector annotations only
                annotation.update_transparency(annotation.label.transparency)
            # Note: Mask annotations handle visibility through update_visible_labels() method
        else:
            # Hide the annotation
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
                    # Show mask by making all linked labels visible
                    linked_labels = self.main_window.label_window.get_linked_labels()
                    visible_label_ids = {label.id for label in linked_labels}
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
        """Set and display an image at the given path."""
        # Calculate GDIs for Windows if needed
        self.main_window.check_windows_gdi_count()
        
        # Stop any current drawing operation before switching images
        if self.selected_tool and self.selected_tool in self.tools:
            self.tools[self.selected_tool].stop_current_drawing()
        
        # Clean up
        self.clear_scene()

        # Check that the image path is valid
        if image_path not in self.main_window.image_window.raster_manager.image_paths:
            return

        # Set the image representations in the annotation window
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        if not raster:
            return
        
        # Update the rasterio image source for cropping annotations
        self.rasterio_image = raster.rasterio_src
        # Get QImage and convert to QPixmap for display
        q_image = raster.get_qimage()
        if q_image is None or q_image.isNull():
            return
        
        # Convert and set the QPixmap
        self.pixmap_image = QPixmap.fromImage(q_image)
        self.current_image_path = image_path
        self.active_image = True

        # Automatically mark this image as checked when viewed
        raster.checkbox_state = True
        self.main_window.image_window.table_model.update_raster_data(image_path)
        
        # Add the base image pixmap and set its Z-value to be at the bottom
        base_image_item = QGraphicsPixmapItem(self.pixmap_image)
        base_image_item.setZValue(-10)
        self.scene.addItem(base_image_item)

        # Update the zoom tool's state
        self.tools["zoom"].reset_zoom()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.tools["zoom"].calculate_min_zoom()

        # Toggle the cursor annotation to reflect the new image and label state
        self.toggle_cursor_annotation()

        # Load all associated annotations
        self.load_annotations()
        # Now that all annotations are loaded and drawn, we build the index for them.
        self._rebuild_spatial_index()
        # Update the image window's image annotations
        self.main_window.image_window.update_image_annotations(image_path)
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

        QApplication.processEvents()

        # Set the image dimensions, and current view in status bar
        self.imageLoaded.emit(self.pixmap_image.width(), self.pixmap_image.height())
        self.viewChanged.emit(self.pixmap_image.width(), self.pixmap_image.height())

    def update_current_image_path(self, image_path):
        """Update the current image path being displayed."""
        self.current_image_path = image_path
        
    def update_mask_label_map(self):
        """Update the label_map in the current MaskAnnotation to reflect changes in LabelWindow."""
        if self.current_mask_annotation:
            # Call the new sync method instead of just overwriting the map.
            all_current_labels = self.main_window.label_window.labels
            self.current_mask_annotation.sync_label_map(all_current_labels)
            
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
    
    def get_intersecting_annotations(self, target_rect: QRectF):
        """
        Queries the R-tree spatial index for annotation IDs that intersect the
        target_rect, then returns the full annotation objects from a dictionary lookup.
        """
        # Return immediately if the index hasn't been created yet.
        if not hasattr(self, 'spatial_index') or self.spatial_index is None:
            return []

        # Get the bounding box coordinates in (left, bottom, right, top) format.
        query_bounds = target_rect.getCoords()

        # The query now returns the annotation IDs (strings) we stored.
        hits = self.spatial_index.intersection(query_bounds, objects=True)
        intersecting_ids = [hit.object for hit in hits]

        # --- THE FIX: Look up the full annotation objects using their IDs. ---
        results = []
        for anno_id in intersecting_ids:
            annotation = self.annotations_dict.get(anno_id)
            if annotation and not isinstance(annotation, MaskAnnotation):
                results.append(annotation)
        
        return results

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
            
        # The MaskAnnotation handles the efficient protection marking internally
        self.current_mask_annotation.rasterize_annotations(annotations)
        
    def unrasterize_annotations(self):
        """
        Remove protection from vector annotation pixels, allowing mask editing over those areas again.
        This clears the locked status from pixels that were protected during mask editing mode.
        """
        if self.current_mask_annotation:
            self.current_mask_annotation.unrasterize_annotations()

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
            self.annotation_color = annotation.label.color
            
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
        # Remove the graphics item from its current scene if it exists
        if annotation.graphics_item and annotation.graphics_item.scene():
            annotation.graphics_item.scene().removeItem(annotation.graphics_item)

        # Create the graphics item (scene previously cleared)
        annotation.create_graphics_item(self.scene)
        # Set the visibility based on the hide button state
        self.set_annotation_visibility(annotation)
        
        # Connect essential update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
        
        # Update the view
        self.viewport().update()

    def load_annotations(self, image_path=None, annotations=None):
        """Load annotations for the specified image path or current image."""
        # First load the mask annotation if it exists
        self.load_mask_annotation()
        
        # Then crop annotations (if image_path and annotations are provided, they are used)
        annotations = self.crop_annotations(image_path, annotations)

        if not len(annotations):
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Loading Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        try:
            # Load each annotation and update progress
            for idx, annotation in enumerate(annotations):
                if progress_bar.wasCanceled():
                    break

                # Load the annotation
                self.load_annotation(annotation)

                # Update every 10% of the annotations (or for each item if total is small)
                if len(annotations) > 10:
                    if idx % (len(annotations) // 10) == 0:
                        progress_bar.update_progress_percentage((idx / len(annotations)) * 100)
                else:
                    progress_bar.update_progress_percentage((idx / len(annotations)) * 100)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

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
                if not annotation.cropped_image:
                    annotation.create_cropped_image(rasterio_image)
                if verbose:
                    progress_bar.update_progress()

            except Exception as e:
                print(f"Error cropping annotation {annotation.id}: {e}")

        QApplication.restoreOverrideCursor()
        if verbose:
            progress_bar.stop_progress()
            progress_bar.close()

        if return_annotations:
            return annotations

    def add_annotation_from_tool(self, annotation):
        """Add a new annotation at the specified position using the current tool."""

        if annotation is None:
            self.toggle_cursor_annotation()
            return

        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)

        # Create the graphics item and cropped image
        if not annotation.graphics_item:
            annotation.create_graphics_item(self.scene)
        if not annotation.cropped_image:
            annotation.create_cropped_image(self.rasterio_image)

        # Display the cropped image in the confidence window
        self.main_window.confidence_window.display_cropped_image(annotation)

        # Add to annotation dict
        self.add_annotation_to_dict(annotation)

        # Update the table in ImageWindow
        self.annotationCreated.emit(annotation.id)

    def add_annotation_to_dict(self, annotation):
        """Add an annotation to the internal dictionaries and the spatial index."""
        # Add to annotation dict
        self.annotations_dict[annotation.id] = annotation
        # Add to image annotations dict (if not already present)
        if annotation.image_path not in self.image_annotations_dict:
            self.image_annotations_dict[annotation.image_path] = []
        if annotation not in self.image_annotations_dict[annotation.image_path]:
            self.image_annotations_dict[annotation.image_path].append(annotation)

        # Add the new annotation to our spatial index.
        self.add_to_spatial_index(annotation)

        # Set the visibility based on the hide button state
        self.set_annotation_visibility(annotation)

        # Update the ImageWindow Table
        self.main_window.image_window.update_annotation_count(annotation.id)
        
    def add_to_spatial_index(self, annotation):
        """
        Adds a single vector annotation to the R-tree spatial index.

        This method checks if the index exists and if the annotation is a valid
        geometric type with a graphical representation before attempting to insert it.
        """
        if self.spatial_index is None or not hasattr(annotation, 'get_polygon'):
            return

        # Ensure the graphics item exists to prevent errors.
        if annotation.graphics_item:
            bounds = annotation.graphics_item.boundingRect().getCoords()
            # Store annotation ID string instead of annotation object
            self.spatial_index.insert(hash(annotation.id), bounds, obj=annotation.id)

    def remove_from_spatial_index(self, annotation):
        """
        Removes a single vector annotation from the R-tree spatial index.

        This method checks if the index exists and if the annotation is a valid
        geometric type with a graphical representation before attempting to remove it.
        """
        if self.spatial_index is None or not hasattr(annotation, 'get_polygon'):
            return

        # Ensure the graphics item exists to prevent errors.
        if annotation.graphics_item:
            bounds = annotation.graphics_item.boundingRect().getCoords()
            # Use annotation ID string for removal
            self.spatial_index.delete(hash(annotation.id), bounds)
            
    def _rebuild_spatial_index(self):
        """
        Clears and rebuilds the R-tree spatial index from scratch for all
        vector annotations associated with the current image.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Create a new, empty 2D index
        p = index.Property()
        p.dimension = 2
        p.object_capacity = 10
        self.spatial_index = index.Index(properties=p)

        # Get all annotations for the currently displayed image
        annotations_for_image = self.get_image_annotations()

        for anno in annotations_for_image:
            self.add_to_spatial_index(anno)
            
        # Restore the cursor
        QApplication.restoreOverrideCursor()

    def delete_annotation(self, annotation_id):
        """Delete an annotation by its ID from dicts and the spatial index."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        if annotation_id in self.annotations_dict:
            # Get the annotation from dict
            annotation = self.annotations_dict[annotation_id]
            # Unselect the annotation (if selected)
            self.unselect_annotation(annotation)

            # Remove the annotation from our spatial index.
            self.remove_from_spatial_index(annotation)

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
            
        # Restore the cursor
        QApplication.restoreOverrideCursor()

    def delete_annotations(self, annotations):
        """Delete a list of annotations."""
        for annotation in annotations:
            self.delete_annotation(annotation.id)

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