import warnings

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem, QApplication

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation

from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import simplify_polygon
from coralnet_toolbox.utilities import polygonize_mask_with_holes

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window
        self.sam_dialog = None
        
        # Set the animation manager for pulse animations
        self.animation_manager = self.annotation_window.animation_manager

        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor 

        # Store polygon points from SAM prediction
        self.points = []

        # Tracking for point prompts and their graphics
        self.positive_points = []
        self.negative_points = []
        self.point_graphics = []

        # Working area and related attributes
        self.working_area = None
        self.image_path = None
        self.original_image = None
        self.original_width = None
        self.original_height = None
        self.image = None

        # Temporary annotation display
        self.temp_annotation = None

        # Last hover position for continuous annotation updates
        self.hover_pos = None

        # Rectangle drawing attributes
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False
        self.rectangle_graphics = None

        # Flag to track if we have active prompts
        self.has_active_prompts = False
        
        # Add state variables for custom working area creation
        self.creating_working_area = False
        self.working_area_start = None
        self.working_area_temp_graphics = None
        
        # Output settings - synced from dialog
        self.output_type = "Polygon"  # Default value, will be synced from dialog
        self.allow_holes = False  # Default value, will be synced from dialog

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog
        # Sync settings from dialog when tool is activated
        self.sync_settings_from_dialog()

    def sync_settings_from_dialog(self):
        """
        Synchronize output_type and allow_holes from the dialog to local attributes.
        """
        if self.sam_dialog:
            self.output_type = self.sam_dialog.get_output_type()
            self.allow_holes = self.sam_dialog.get_allow_holes()

    def deactivate(self):
        """
        Deactivates the tool.
        """
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.sam_dialog = None
        self.cancel_working_area()
        self.has_active_prompts = False
        self.cancel_working_area_creation()
        
        # If output type was Mask, unrasterize annotations to remove lock protection
        if self.output_type == "Mask":
            self.annotation_window.unrasterize_annotations()

        # Call parent deactivate to ensure crosshair is properly cleared
        super().deactivate()

    def set_working_area(self):
        """
        Set the working area for the tool using the WorkArea class.
        """
        self.annotation_window.setCursor(Qt.WaitCursor)

        # Cancel the current working area if it exists
        self.cancel_working_area()

        # Original image (grab current from the annotation window)
        self.image_path = self.annotation_window.current_image_path
        self.original_image = pixmap_to_numpy(self.annotation_window.pixmap_image)
        self.original_width = self.annotation_window.pixmap_image.size().width()
        self.original_height = self.annotation_window.pixmap_image.size().height()

        # Current extent (view)
        extent = self.annotation_window.viewportToScene()

        top = max(0, round(extent.top()))
        left = max(0, round(extent.left()))
        width = round(extent.width())
        height = round(extent.height())
        bottom = min(self.original_height, top + height)
        right = min(self.original_width, left + width)

        # Create the WorkArea instance
        self.working_area = WorkArea(left, top, right - left, bottom - top, self.image_path)
        self.working_area.set_animation_manager(self.animation_manager)
        
        # Create and add the working area graphics
        self.working_area.create_graphics(self.annotation_window.scene, include_shadow=True)
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)

        # Update the working area image, set in model
        self.image = self.original_image[top:bottom, left:right]
        self.sam_dialog.set_image(self.image, self.image_path)

        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.scene.update()
        
    def set_custom_working_area(self, start_point, end_point):
        """
        Create a working area from custom points selected by the user.
        
        Args:
            start_point (QPointF): First corner of the working area
            end_point (QPointF): Opposite corner of the working area
        """
        self.annotation_window.setCursor(Qt.WaitCursor)
        
        # Cancel any existing working area
        self.cancel_working_area()
        
        # Calculate the rectangle bounds
        left = max(0, int(min(start_point.x(), end_point.x())))
        top = max(0, int(min(start_point.y(), end_point.y())))
        right = min(int(self.annotation_window.pixmap_image.size().width()), 
                    int(max(start_point.x(), end_point.x())))
        bottom = min(int(self.annotation_window.pixmap_image.size().height()),
                     int(max(start_point.y(), end_point.y())))
        
        # Ensure minimum size (at least 10x10 pixels)
        if right - left < 10:
            right = min(left + 10, int(self.annotation_window.pixmap_image.size().width()))
        if bottom - top < 10:
            bottom = min(top + 10, int(self.annotation_window.pixmap_image.size().height()))
            
        # Original image information
        self.image_path = self.annotation_window.current_image_path
        self.original_image = pixmap_to_numpy(self.annotation_window.pixmap_image)
        self.original_width = self.annotation_window.pixmap_image.size().width()
        self.original_height = self.annotation_window.pixmap_image.size().height()
            
        # Create the WorkArea instance
        self.working_area = WorkArea(left, top, right - left, bottom - top, self.image_path)
        self.working_area.set_animation_manager(self.animation_manager)
        
        # Create and add the working area graphics
        self.working_area.create_graphics(self.annotation_window.scene, include_shadow=True)
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)
        
        # Update the working area image in the SAM model
        self.image = self.original_image[top:bottom, left:right]
        self.sam_dialog.set_image(self.image, self.image_path)
        
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.scene.update()
        
    def display_working_area_preview(self, current_pos):
        """
        Display a preview rectangle for the working area being created.
        
        Args:
            current_pos (QPointF): Current mouse position
        """
        if not self.working_area_start:
            return
            
        # Remove previous preview if it exists
        if self.working_area_temp_graphics:
            self.annotation_window.scene.removeItem(self.working_area_temp_graphics)
            self.working_area_temp_graphics = None
            
        # Create preview rectangle
        rect = QRectF(
            min(self.working_area_start.x(), current_pos.x()),
            min(self.working_area_start.y(), current_pos.y()),
            abs(current_pos.x() - self.working_area_start.x()),
            abs(current_pos.y() - self.working_area_start.y())
        )
        
        # Create a dashed blue pen for the working area preview
        pen = QPen(QColor(0, 168, 230))
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(3)
        
        self.working_area_temp_graphics = QGraphicsRectItem(rect)
        self.working_area_temp_graphics.setPen(pen)
        self.working_area_temp_graphics.setBrush(QBrush(QColor(0, 168, 230, 30)))  # Light blue transparent fill
        self.annotation_window.scene.addItem(self.working_area_temp_graphics)

    def cancel_working_area_creation(self):
        """
        Cancel the process of creating a working area.
        """
        self.creating_working_area = False
        self.working_area_start = None
        
        if self.working_area_temp_graphics:
            self.annotation_window.scene.removeItem(self.working_area_temp_graphics)
            self.working_area_temp_graphics = None
            
        self.annotation_window.scene.update()
        
    def on_working_area_removed(self, work_area):
        """
        Handle when the work area is removed via its internal mechanism.
        """
        self.cancel_working_area()

    def clear_temp_annotation(self):
        """
        Clear the temporary annotation and its graphics.
        """
        if self.temp_annotation:
            # Stop animation first
            self.temp_annotation.deanimate()
            # Then delete the annotation
            self.temp_annotation.delete()
            self.temp_annotation = None
            
        # Force scene update to ensure graphics are removed
        self.annotation_window.scene.update()

    def clear_prompt_graphics(self):
        """
        Clear all prompt graphics (points, rectangles) but keep the working area.
        """
        # Clear temporary annotation
        self.clear_temp_annotation()

        # Remove point graphics
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)
        self.point_graphics = []

        # Clear points lists
        self.points = []
        self.positive_points = []
        self.negative_points = []

        # Clear rectangle
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None

        # Remove rectangle graphics if any
        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        # Reset active prompts flag
        self.has_active_prompts = False

        self.annotation_window.scene.update()

    def create_temp_annotation(self, scene_pos=None):
        """
        Create and display a temporary annotation based on current prompts.
        
        Note: Temporary annotations use Polygon or Rectangle (not Mask) for performance.
        If user selected Mask output type, temp annotations show as Polygon.
        The final output type is applied when the user accepts the annotation.

        Args:
            scene_pos (QPointF): Current scene position for hover point
        """
        # Clear any existing temporary annotation first
        self.clear_temp_annotation()

        if not self.working_area:
            return

        if not self.annotation_window.active_image or not self.annotation_window.selected_label:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Prepare points and bounding box for SAM
            positive = [[point.x(), point.y()] for point in self.positive_points]
            negative = [[point.x(), point.y()] for point in self.negative_points]
            bbox = np.array([])

            # Add hover point as a positive point if available and we're not drawing a rectangle
            if scene_pos and not self.drawing_rectangle:
                # Adjust hover point relative to working area
                working_area_top_left = self.working_area.rect.topLeft()
                adjusted_pos = QPointF(scene_pos.x() - working_area_top_left.x(),
                                       scene_pos.y() - working_area_top_left.y())
                # Add to positive points for prediction
                positive.append([adjusted_pos.x(), adjusted_pos.y()])

            # If we have a rectangle, use it
            if self.start_point and self.end_point and self.top_left is not None and self.bottom_right is not None:
                bbox = np.array([self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y()])

            # Convert to numpy arrays for SAM
            labels = np.array([1] * len(positive) + [0] * len(negative))
            points = np.array(positive + negative)

            # If no prompts, return without creating annotation
            if len(points) == 0 and bbox.size == 0:
                QApplication.restoreOverrideCursor()
                return

            # Predict the mask from prompts
            results = self.sam_dialog.predict_from_prompts(bbox, points, labels)

            if not results or not results.boxes.conf.numel():
                QApplication.restoreOverrideCursor()
                return

            # Skip low confidence predictions for temporary annotations
            if results.boxes.conf[0] < self.main_window.get_uncertainty_thresh():
                QApplication.restoreOverrideCursor()
                return

            # Get the top confidence prediction's mask tensor
            top1_index = np.argmax(results.boxes.conf)
            mask_tensor = results[top1_index].masks.data

            # For temporary annotations, don't use Mask output type (convert to Polygon for speed)
            # Save the desired output type and temporarily override it if needed
            saved_output_type = self.output_type
            if self.output_type == "Mask":
                self.output_type = "Polygon"
            
            # Create temporary annotation (Rectangle or Polygon, never Mask)
            self.temp_annotation = self.create_annotation_from_mask(mask_tensor)
            
            # Restore the desired output type for final annotation
            self.output_type = saved_output_type
            
            if not self.temp_annotation:
                QApplication.restoreOverrideCursor()
                return
            
            self.temp_annotation.set_animation_manager(self.animation_manager)
            # Create the graphics item for the temporary annotation
            self.temp_annotation.create_graphics_item(self.annotation_window.scene)
            # Make the annotation animated immediately
            self.temp_annotation.animate(force=True)
            
        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()

    def display_rectangle(self):
        """
        Display the rectangle during drawing.
        """
        if not self.working_area or not self.start_point or not self.end_point:
            return

        working_area_top_left = self.working_area.rect.topLeft()

        # Calculate rectangle coordinates
        top_left = QPointF(min(self.start_point.x(), self.end_point.x()),
                           min(self.start_point.y(), self.end_point.y()))

        bottom_right = QPointF(max(self.start_point.x(), self.end_point.x()),
                               max(self.start_point.y(), self.end_point.y()))

        # Adjust points relative to working area for SAM
        self.top_left = QPointF(top_left.x() - working_area_top_left.x(),
                                top_left.y() - working_area_top_left.y())

        self.bottom_right = QPointF(bottom_right.x() - working_area_top_left.x(),
                                    bottom_right.y() - working_area_top_left.y())

        # Remove previous rectangle graphic
        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        # Create rectangle graphic
        rect = QRectF(top_left, bottom_right)

        # Create a dashed pen for the rectangle
        pen = QPen(self.annotation_window.selected_label.color)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(4)

        self.rectangle_graphics = QGraphicsRectItem(rect)
        self.rectangle_graphics.setPen(pen)
        self.rectangle_graphics.setBrush(QBrush(Qt.transparent))
        self.annotation_window.scene.addItem(self.rectangle_graphics)

        # Update the temporary annotation to show segmentation preview
        self.create_temp_annotation(self.end_point)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press events.
        """
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return

        # Get position in scene coordinates
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # Handle working area creation mode
        if not self.working_area and event.button() == Qt.LeftButton:
            if not self.creating_working_area:
                # Start working area creation
                self.creating_working_area = True
                self.working_area_start = scene_pos
                return
            elif self.creating_working_area and self.working_area_start:
                # Finish working area creation
                self.set_custom_working_area(self.working_area_start, scene_pos)
                self.cancel_working_area_creation()
                return

        if not self.working_area:
            return

        # Check if position is within working area
        if not self.working_area.contains_point(scene_pos):
            return

        # Get position relative to working area
        working_area_top_left = self.working_area.rect.topLeft()
        adjusted_pos = QPointF(scene_pos.x() - working_area_top_left.x(),
                               scene_pos.y() - working_area_top_left.y())

        # Handle ctrl+click for positive/negative points
        if event.modifiers() == Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                # Add positive point
                self.positive_points.append(adjusted_pos)

                # Create point graphic
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                pen = QPen(Qt.green)
                pen.setCosmetic(True)
                point.setPen(pen)
                point.setBrush(QColor(Qt.green))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)

                # Update active prompts flag
                self.has_active_prompts = True

                # Update temporary annotation
                self.create_temp_annotation(scene_pos)

            elif event.button() == Qt.RightButton:
                # Add negative point
                self.negative_points.append(adjusted_pos)

                # Create point graphic
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                pen = QPen(Qt.red)
                pen.setCosmetic(True)
                point.setPen(pen)
                point.setBrush(QColor(Qt.red))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)

                # Update active prompts flag
                self.has_active_prompts = True

                # Update temporary annotation
                self.create_temp_annotation(scene_pos)

        # Handle regular clicks for rectangle drawing
        elif event.button() == Qt.LeftButton:
            if not self.drawing_rectangle:
                # Clear temporary annotation
                self.clear_temp_annotation()

                # Start rectangle drawing
                self.start_point = scene_pos
                self.drawing_rectangle = True
                self.has_active_prompts = True
            else:
                # Finish rectangle drawing
                self.end_point = scene_pos
                self.display_rectangle()
                self.drawing_rectangle = False

        self.annotation_window.scene.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle mouse move events.
        """
        # Call parent implementation to handle crosshair
        super().mouseMoveEvent(event)
        
        # Continue with tool-specific behavior
        scene_pos = self.annotation_window.mapToScene(event.pos())
        self.hover_pos = scene_pos

        # Update working area preview during creation
        if self.creating_working_area and self.working_area_start:
            self.display_working_area_preview(scene_pos)
            return
            
        if not self.working_area:
            return

        # Update rectangle during drawing
        if self.drawing_rectangle and self.start_point:
            self.end_point = scene_pos
            self.display_rectangle()
        # Create hover annotation when not drawing rectangle
        elif not self.drawing_rectangle and self.annotation_window.cursorInWindow(event.pos()):
            # Only create new temp annotation if we don't have active prompts
            if not self.has_active_prompts:
                self.create_temp_annotation(scene_pos)
            # If we have points but no rectangle, update the temp annotation with new hover point
            elif len(self.positive_points) > 0 or len(self.negative_points) > 0:
                self.create_temp_annotation(scene_pos)
        # Remove hover annotation when cursor leaves window
        elif not self.annotation_window.cursorInWindow(event.pos()):
            # Only clear if we don't have active points or rectangle
            if not self.has_active_prompts:
                self.clear_temp_annotation()

        self.annotation_window.scene.update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handle key press events.
        """
        if event.key() == Qt.Key_Space:
            # If creating working area, confirm it
            if self.creating_working_area and self.working_area_start and self.hover_pos:
                self.set_custom_working_area(self.working_area_start, self.hover_pos)
                self.cancel_working_area_creation()
                return

            # If no working area, set it up
            if not self.working_area:
                self.set_working_area()

            # If we have active prompts, create a permanent annotation
            elif self.has_active_prompts:
                
                # Create the final annotation
                if self.temp_annotation:
                    # Get the mask tensor that was used for the temp annotation
                    # We need to re-predict with the final output type
                    final_annotation = self.create_annotation(True)
                    
                    # For Mask output type, create_annotation returns None after updating the raster mask
                    if final_annotation is None:
                        # Mask was updated successfully, just clear prompts
                        self.clear_prompt_graphics()
                    
                    elif final_annotation:
                        # Copy confidence data from temp annotation if available
                        if hasattr(self.temp_annotation, 'machine_confidence'):
                            final_annotation.update_machine_confidence(
                                self.temp_annotation.machine_confidence
                            )
                        
                        # Create the graphics item for the final annotation
                        final_annotation.create_graphics_item(self.annotation_window.scene)
                        # Add the annotation to the scene
                        self.annotation_window.add_annotation_from_tool(final_annotation)
                        
                        # Clear all temporary graphics and prompts
                        self.clear_prompt_graphics()
                else:
                    # If no temp annotation, create one from current prompts without hover point
                    final_annotation = self.create_annotation(True)
                    
                    # For Mask output type, create_annotation returns None after updating the raster mask
                    if final_annotation is None:
                        # Mask was updated successfully, just clear prompts
                        self.clear_prompt_graphics()

                    elif final_annotation:
                        self.annotation_window.add_annotation_from_tool(final_annotation)
                        self.clear_prompt_graphics() 
            # If no active prompts, cancel the working area
            else:
                self.cancel_working_area()

            self.annotation_window.scene.update()

        elif event.key() == Qt.Key_Backspace:
            # If creating working area, cancel it
            if self.creating_working_area:
                self.cancel_working_area_creation()
                return
                
            # If drawing rectangle, cancel it
            if self.drawing_rectangle:
                self.cancel_rectangle_drawing()
            # If we have active prompts, clear them
            elif self.has_active_prompts:
                self.clear_prompt_graphics()
            # Otherwise cancel working area
            else:
                self.cancel_working_area()

            self.annotation_window.scene.update()

    def cancel_rectangle_drawing(self):
        """
        Cancel the current rectangle drawing operation.
        """
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False

        # Remove rectangle graphic
        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        # Clear temporary annotation
        self.clear_temp_annotation()

        # Reset active prompts flag if no points exist
        if len(self.positive_points) == 0 and len(self.negative_points) == 0:
            self.has_active_prompts = False

        self.annotation_window.scene.update()

    def create_annotation(self, final=True):
        """
        Create a final annotation based on current prompts.

        Args:
            final (bool): Whether this is a final annotation

        Returns:
            PolygonAnnotation or None: The created annotation or None if creation fails
        """
        if not self.working_area or not self.annotation_window.active_image:
            return None

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Get positive and negative points
        positive = [[point.x(), point.y()] for point in self.positive_points]
        negative = [[point.x(), point.y()] for point in self.negative_points]
        bbox = np.array([])

        # Use rectangle if available
        if self.top_left is not None and self.bottom_right is not None:
            bbox = np.array([self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y()])

        # Create labels and points arrays
        labels = np.array([1] * len(positive) + [0] * len(negative))
        points = np.array(positive + negative)

        # If no prompts, return None
        if len(points) == 0 and bbox.size == 0:
            QApplication.restoreOverrideCursor()
            return None

        # Predict mask from prompts
        results = self.sam_dialog.predict_from_prompts(bbox, points, labels)

        if not results or not results.boxes.conf.numel():
            QApplication.restoreOverrideCursor()
            return None

        # Get the top confidence prediction's mask tensor
        top1_index = np.argmax(results.boxes.conf)
        mask_tensor = results[top1_index].masks.data

        # Sync latest settings from dialog before creating annotation
        self.sync_settings_from_dialog()
        
        # Create annotation using the helper method with local attributes
        annotation = self.create_annotation_from_mask(mask_tensor)
        
        # If annotation is None, it might be a Mask update (not a new annotation)
        if not annotation:
            QApplication.restoreOverrideCursor()
            return None

        # Set the animation manager for the annotation
        annotation.set_animation_manager(self.animation_manager)

        # Update confidence - make sure to extract confidence from results
        confidence = float(results.boxes.conf[top1_index])
        annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})

        # Create cropped image only for vector annotations (Polygon/Rectangle), not Mask
        if not isinstance(annotation, MaskAnnotation):
            if hasattr(self.annotation_window, 'rasterio_image'):
                annotation.create_cropped_image(self.annotation_window.rasterio_image)

        # Restore cursor
        QApplication.restoreOverrideCursor()

        return annotation

    def create_annotation_from_mask(self, mask_tensor):
        """
        Create annotation (Rectangle, Polygon, or Mask) from a mask tensor.
        
        Uses self.output_type and self.allow_holes attributes to determine
        the annotation type and whether to include holes.
        
        For Mask output: Updates the existing raster mask annotation instead of creating a new one.
        For Rectangle/Polygon: Creates and returns a new annotation object.
        
        Args:
            mask_tensor: The tensor containing the mask data
        
        Returns:
            Annotation object or None if creation fails (except for Mask which returns None after updating)
        """
        if not self.working_area:
            return None
            
        if self.output_type == "Mask":         
            # Convert mask tensor to numpy array
            mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            
            # Get the working area information
            working_area_top_left = self.working_area.rect.topLeft()
            wa_x, wa_y = int(working_area_top_left.x()), int(working_area_top_left.y())
            wa_height, wa_width = mask_np.shape
            
            # Get the existing mask annotation from the raster (lazy-loads if needed)
            mask_annotation = self.annotation_window.current_mask_annotation
            if not mask_annotation:
                return None
            
            # Get the label and its class ID
            label = self.annotation_window.selected_label
            class_id = mask_annotation.label_id_to_class_id_map.get(label.id)
            if class_id is None:
                return None
            
            # Create a prediction mask (full-size, initialized with zeros)
            prediction_mask = np.zeros_like(mask_annotation.mask_data)
            
            # Place the SAM prediction into the full mask at the working area position
            prediction_mask[wa_y:wa_y + wa_height, wa_x:wa_x + wa_width] = np.where(
                mask_np > 0,
                class_id,
                0
            )
            
            # Update the existing mask annotation with the prediction
            mask_annotation.update_mask_with_prediction_mask(prediction_mask)
            
            # Return None to indicate this is not a new annotation object to add
            return None
            
        elif self.output_type == "Rectangle":
            # For rectangle output, just get the bounding box of the mask
            # Find the bounding rectangle of the mask
            y_indices, x_indices = np.where(mask_tensor.cpu().numpy()[0] > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
                
            # Get the min/max coordinates
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            
            # Apply the offset from working area
            working_area_top_left = self.working_area.rect.topLeft()
            offset_x, offset_y = working_area_top_left.x(), working_area_top_left.y()
            
            top_left = QPointF(min_x + offset_x, min_y + offset_y)
            bottom_right = QPointF(max_x + offset_x, max_y + offset_y)
            
            # Create a rectangle annotation
            annotation = RectangleAnnotation(
                top_left,
                bottom_right,
                self.annotation_window.selected_label.short_label_code,
                self.annotation_window.selected_label.long_label_code,
                self.annotation_window.selected_label.color,
                self.annotation_window.current_image_path,
                self.annotation_window.selected_label.id,
                self.main_window.get_transparency_value()
            )
            return annotation
        else:
            # Original polygon code
            # Polygonize the mask using the new method to get the exterior and holes
            exterior_coords, holes_coords_list = polygonize_mask_with_holes(mask_tensor)
            
            # Safety check for an empty result
            if not exterior_coords:
                return None
                
            # --- Process and Clean the Polygon Points ---
            working_area_top_left = self.working_area.rect.topLeft()
            offset_x, offset_y = working_area_top_left.x(), working_area_top_left.y()
            
            # Simplify, offset, and convert the exterior points
            simplified_exterior = simplify_polygon(exterior_coords, 0.1)
            self.points = [QPointF(p[0] + offset_x, p[1] + offset_y) for p in simplified_exterior]
            
            # Simplify, offset, and convert each hole only if allowed
            final_holes = []
            if self.allow_holes:
                for hole_coords in holes_coords_list:
                    simplified_hole = simplify_polygon(hole_coords, 0.1)
                    if len(simplified_hole) >= 3:
                        hole_points = [QPointF(p[0] + offset_x, p[1] + offset_y) for p in simplified_hole]
                        final_holes.append(hole_points)
            
            # Require at least 3 points for valid polygon
            if len(self.points) < 3:
                return None
                
            # Create final annotation, now passing the holes argument
            annotation = PolygonAnnotation(
                points=self.points,
                holes=final_holes,
                short_label_code=self.annotation_window.selected_label.short_label_code,
                long_label_code=self.annotation_window.selected_label.long_label_code,
                color=self.annotation_window.selected_label.color,
                image_path=self.annotation_window.current_image_path,
                label_id=self.annotation_window.selected_label.id,
                transparency=self.main_window.get_transparency_value()
            )
            return annotation

    def cancel_working_area(self):
        """
        Cancel the working area and clean up all associated resources.
        """
        # Clear temporary annotation
        self.clear_temp_annotation()

        # Remove all point graphics
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)
        self.point_graphics = []

        # Clear rectangle graphics
        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        # Remove working area graphic
        if self.working_area:
            self.working_area.remove_from_scene()
            self.working_area = None

        # Reset all state variables
        self.points = []
        self.positive_points = []
        self.negative_points = []
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False
        self.has_active_prompts = False
        self.image_path = None
        self.original_image = None
        self.image = None

        self.annotation_window.scene.update()
