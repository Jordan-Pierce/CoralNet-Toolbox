import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem, QApplication

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import clean_polygon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window
        self.sam_dialog = None
        
        self.top_left = None

        self.cursor = Qt.CrossCursor
        self.points = []
        self.point_graphics = []
        self.annotation_graphics = None

        self.image = None
        self.positive_points = []
        self.negative_points = []

        self.working_area = None
        self.shadow_area = None

        self.image_path = None
        self.original_image = None
        self.original_width = None
        self.original_height = None

        self.hover_pos = None
        self.hover_point = None
        self.hover_graphics = None
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self.display_hover_annotation)

        self.hover_active = True  # Flag to control hover timer

        # Rectangle drawing attributes
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False
        self.rectangle_graphics = None
        
        self.results = None
        
        # Flag to track if we have active prompts
        self.has_active_prompts = False

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog
        self.hover_active = True  # Ensure hover is active when SAMTool is activated

    def deactivate(self):
        """
        Deactivates the tool.
        """
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.sam_dialog = None
        self.cancel_working_area()
        self.hover_active = False  # Ensure hover is inactive when SAMTool is deactivated
        self.has_active_prompts = False

    def start_hover_timer(self, pos):
        """
        Start the hover timer to display the annotation.

        Args:
            pos (QPointF): The position of the cursor.
        """
        if self.hover_active:
            self.hover_timer.start(3000)
            self.hover_pos = pos

    def stop_hover_timer(self):
        """
        Stop the hover timer
        """
        self.hover_timer.stop()
        self.display_hover_annotation()

    def display_hover_annotation(self):
        """
        Display the hover annotation.
        """
        if self.working_area and self.hover_active and not self.drawing_rectangle:
            if self.annotation_window.cursorInWindow(self.hover_pos, mapped=True):
                # Adjust points relative to the working area's top-left corner
                working_area_top_left = self.working_area.rect().topLeft()
                adjusted_pos = QPointF(self.hover_pos.x() - working_area_top_left.x(),
                                       self.hover_pos.y() - working_area_top_left.y())

                # Add the adjusted point to the list
                self.hover_point = adjusted_pos
                # Create cursor annotation at the adjusted position
                self.create_cursor_annotation(self.hover_pos)

    def display_rectangle_annotation(self):
        """
        Display the rectangle annotation.
        """
        if not self.working_area:
            return

        if self.working_area and self.end_point and self.drawing_rectangle:
            working_area_top_left = self.working_area.rect().topLeft()

            # Ensure top_left and bottom_right are correctly calculated
            top_left = QPointF(min(self.start_point.x(), self.end_point.x()),
                               min(self.start_point.y(), self.end_point.y()))

            bottom_right = QPointF(max(self.start_point.x(), self.end_point.x()),
                                   max(self.start_point.y(), self.end_point.y()))

            # Adjust the points relative to the working area's top-left corner
            self.top_left = QPointF(top_left.x() - working_area_top_left.x(),
                                    top_left.y() - working_area_top_left.y())

            self.bottom_right = QPointF(bottom_right.x() - working_area_top_left.x(),
                                        bottom_right.y() - working_area_top_left.y())

            # Update cursor annotation to show preview of segmentation
            self.update_cursor_annotation(self.end_point)

    def get_workarea_thickness(self):
        """
        Calculate line thickness so it appears visually consistent regardless of zoom or image size.
        """
        view = self.annotation_window
        if not view.pixmap_image:
            return 5  # fallback

        # Get the current zoom scale from the view's transformation matrix
        # m11() is the horizontal scale factor (scene to view)
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero

        desired_px = 5  # Desired thickness in screen pixels

        # To keep the line visually consistent, divide by the scale
        thickness = max(1, int(round(desired_px / scale)))
        return thickness
    
    def set_working_area(self):
        """
        Set the working area for the tool.
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
        if bottom > self.original_height:
            bottom = self.original_height
        if right > self.original_width:
            right = self.original_width

        # Set the working area
        working_rect = QRectF(left, top, right - left, bottom - top)

        # Create the graphic for the working area
        pen = QPen(Qt.green)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(self.get_workarea_thickness())
        self.working_area = QGraphicsRectItem(working_rect)
        self.working_area.setPen(pen)

        # Add the working area to the scene
        self.annotation_window.scene.addItem(self.working_area)

        # Create a semi-transparent overlay for the shadow
        shadow_brush = QBrush(QColor(0, 0, 0, 150))  # Semi-transparent black
        shadow_path = QPainterPath()
        shadow_path.addRect(self.annotation_window.scene.sceneRect())  # Cover the entire scene
        shadow_path.addRect(working_rect)  # Add the work area rect
        # Subtract the work area from the overlay
        shadow_path = shadow_path.simplified()

        # Create the shadow item
        self.shadow_area = QGraphicsPathItem(shadow_path)
        self.shadow_area.setBrush(shadow_brush)
        self.shadow_area.setPen(QPen(Qt.NoPen))  # No outline for the shadow

        # Add the shadow item to the scene
        self.annotation_window.scene.addItem(self.shadow_area)

        # Crop the image based on the working_rect
        self.image = self.original_image[top:bottom, left:right]

        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.scene.update()
        
    def get_rectangle_graphic_thickness(self):
        """
        Calculate line thickness so it appears visually consistent regardless of zoom or image size.
        """
        view = self.annotation_window
        if not view.pixmap_image:
            return 2  # fallback

        # Get the current zoom scale from the view's transformation matrix
        # m11() is the horizontal scale factor (scene to view)
        scale = view.transform().m11()
        if scale == 0:
            scale = 1  # avoid division by zero

        desired_px = 2  # Desired thickness in screen pixels

        # To keep the line visually consistent, divide by the scale
        thickness = max(1, int(round(desired_px / scale)))
        return thickness 

    def clear_prompt_graphics(self):
        """
        Clear all prompt graphics (points, rectangles) but keep the working area.
        """
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
            
        # Clear any hover annotation
        self.cancel_hover_annotation()
        
        self.annotation_window.scene.update()

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handles the mouse press event.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if not self.working_area:
            return

        # Position in the scene
        scene_pos = self.annotation_window.mapToScene(event.pos())

        # Check if the position is within the working area
        if not self.working_area.rect().contains(scene_pos):
            return

        # Get the adjusted position relative to the working area's top-left corner
        working_area_top_left = self.working_area.rect().topLeft()
        adjusted_pos = QPointF(scene_pos.x() - working_area_top_left.x(),
                               scene_pos.y() - working_area_top_left.y())

        # Clear any existing annotation preview
        self.clear_cursor_annotation()

        if event.modifiers() == Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                self.positive_points.append(adjusted_pos)
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                point.setPen(QPen(Qt.green))
                point.setBrush(QColor(Qt.green))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)
                # Set flag indicating we have active prompts
                self.has_active_prompts = True
                
                # Create and display the annotation immediately
                temp_annotation = self.create_annotation(scene_pos, False)
                if temp_annotation:
                    if self.hover_graphics:
                        self.annotation_window.scene.removeItem(self.hover_graphics)
                    self.hover_graphics = temp_annotation.create_graphics_item(self.annotation_window.scene)
                    self.cursor_annotation = temp_annotation

            elif event.button() == Qt.RightButton:
                self.negative_points.append(adjusted_pos)
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                point.setPen(QPen(Qt.red))
                point.setBrush(QColor(Qt.red))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)
                # Set flag indicating we have active prompts
                self.has_active_prompts = True
                
                # Create and display the annotation immediately
                temp_annotation = self.create_annotation(scene_pos, False)
                if temp_annotation:
                    if self.hover_graphics:
                        self.annotation_window.scene.removeItem(self.hover_graphics)
                    self.hover_graphics = temp_annotation.create_graphics_item(self.annotation_window.scene)
                    self.cursor_annotation = temp_annotation

        elif event.modifiers() != Qt.ControlModifier:
            if event.button() == Qt.LeftButton and not self.drawing_rectangle:
                # Remove the hover annotation
                self.cancel_hover_annotation()
                # Get the start point
                self.start_point = self.annotation_window.mapToScene(event.pos())
                # Start drawing the rectangle
                self.drawing_rectangle = True
                # Set flag indicating we have active prompts
                self.has_active_prompts = True

            elif event.button() == Qt.LeftButton and self.drawing_rectangle:
                # Get the end point
                self.end_point = self.annotation_window.mapToScene(event.pos())
                # Calculate rectangle coordinates
                self.display_rectangle_annotation()
                # Finish drawing the rectangle
                self.drawing_rectangle = False
                # Set flag indicating we have active prompts (rectangle is complete)
                self.has_active_prompts = True
                
                # Create and display the annotation immediately
                temp_annotation = self.create_annotation(scene_pos, False)
                if temp_annotation:
                    if self.hover_graphics:
                        self.annotation_window.scene.removeItem(self.hover_graphics)
                    self.hover_graphics = temp_annotation.create_graphics_item(self.annotation_window.scene)
                    self.cursor_annotation = temp_annotation

        elif event.button() == Qt.RightButton and self.drawing_rectangle:
            # Panning the image while drawing
            pass

        self.annotation_window.scene.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handles the mouse move event.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        if self.working_area and self.drawing_rectangle:
            # Update the end point while drawing the rectangle
            self.end_point = self.annotation_window.mapToScene(event.pos())
            self.display_rectangle_annotation()

            # Update the annotation graphics
            active_image = self.annotation_window.active_image
            pixmap_image = self.annotation_window.pixmap_image
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and pixmap_image and cursor_in_window and self.start_point:
                # Always update cursor annotation during rectangle drawing
                self.update_cursor_annotation(self.end_point)
        else:
            if self.annotation_window.cursorInWindow(event.pos()):
                self.start_hover_timer(event.pos())
            else:
                self.cancel_hover_annotation()
                self.stop_hover_timer()

        self.annotation_window.scene.update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles the key press event.

        Args:
            event (QKeyEvent): The key press event
        """
        if event.key() == Qt.Key_Space:
            # If there is no working area, set it
            if not self.working_area:
                self.set_working_area()
                self.sam_dialog.set_image(self.image, self.image_path)
            # If there are active prompts, process them into annotations
            elif self.has_active_prompts:
                annotation = self.create_annotation(None, True)
                if annotation:
                    # Use annotation_window's method to add the annotation
                    self.annotation_window.add_annotation_from_tool(annotation)
                    self.has_active_prompts = False
                    
                    # Clear the points and graphics after creating an annotation
                    self.clear_prompt_graphics()
            # If no more active prompts, cancel the working area
            else:
                self.cancel_working_area()

            self.annotation_window.scene.update()
            
        elif event.key() == Qt.Key_Backspace:
            # Cancel current rectangle being drawn
            if self.drawing_rectangle:
                self.cancel_rectangle_annotation()
                self.has_active_prompts = False
            # If we have active prompts (points or completed rectangle), clear them first
            elif self.has_active_prompts:
                self.cancel_annotation()
                self.has_active_prompts = False
            # If no prompts, cancel working area
            else:
                self.cancel_working_area()
                
            self.annotation_window.scene.update()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        """
        Create an annotation based on the given scene position.
        This is used for hover annotation and also for creating final annotations.

        Args:
            scene_pos (QPointF): The scene position
            finished (bool): Flag to indicate if the annotation is finished
        """
        if not self.annotation_window.active_image:
            return None

        if not self.annotation_window.pixmap_image:
            return None

        if not self.working_area:
            return None

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Get the current transparency
        transparency = self.main_window.label_window.active_label.transparency

        # Get the positive and negative points, bbox
        positive = [[point.x(), point.y()] for point in self.positive_points]
        negative = [[point.x(), point.y()] for point in self.negative_points]
        bbox = np.array([])

        if self.hover_point and not finished and not self.drawing_rectangle:
            positive.append([self.hover_point.x(), self.hover_point.y()])
            transparency //= 2

        if self.start_point and self.end_point:
            # If we have rectangle coordinates, use them for prediction
            if self.top_left is not None and self.bottom_right is not None:
                bbox = np.array([self.top_left.x(),
                                self.top_left.y(),
                                self.bottom_right.x(),
                                self.bottom_right.y()])

        # Create the labels, points, and bbox as numpy arrays
        labels = np.array([1] * len(positive) + [0] * len(negative))
        points = np.array(positive + negative)

        # If we have no prompts (no positive points, no negative points, no bbox), return None
        if len(points) == 0 and bbox.size == 0:
            QApplication.restoreOverrideCursor()
            return None

        # Predict the mask provided prompts
        results = self.sam_dialog.predict_from_prompts(bbox, points, labels)

        if not results:
            QApplication.restoreOverrideCursor()
            return None

        if not finished and results.boxes.conf[0] < self.main_window.get_uncertainty_thresh():
            QApplication.restoreOverrideCursor()
            return None

        # Get the points of the top1 mask
        top1_index = np.argmax(results.boxes.conf)
        predictions = results[top1_index].masks.xy[0]

        # Safety check: make sure we have predicted points
        if len(predictions) == 0:
            QApplication.restoreOverrideCursor()
            return None

        # Clean the polygon using the Ramer-Douglas-Peucker algorithm
        predictions = clean_polygon(predictions)

        # Move the points back to the original image space
        working_area_top_left = self.working_area.rect().topLeft()
        points = [(point[0] + working_area_top_left.x(), point[1] + working_area_top_left.y()) for point in predictions]
        self.points = [QPointF(*point) for point in points]
        
        # Safety check: make sure we have at least 3 points to create a valid polygon
        if len(self.points) < 3:
            QApplication.restoreOverrideCursor()
            return None

        # Get confidence for the annotation
        confidence = results.boxes.conf[top1_index].item()

        # Create the annotation
        annotation = PolygonAnnotation(self.points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       transparency)
                                    
        # Update the confidence score of annotation
        annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})

        # Create cropped image
        if hasattr(self.annotation_window, 'rasterio_image'):
            annotation.create_cropped_image(self.annotation_window.rasterio_image)

        # Restore cursor
        QApplication.restoreOverrideCursor()

        return annotation

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Override parent method: Create a cursor annotation at the given position.
        
        Args:
            scene_pos (QPointF): The scene position for the cursor annotation
        """
        # Clear any existing cursor annotation
        self.clear_cursor_annotation()
        
        if not self.annotation_window.selected_label or not self.annotation_window.active_image:
            return
        
        # Always use the scene position if provided, otherwise use the last hover position
        position = scene_pos if scene_pos else self.hover_pos
        
        if not position:
            return
        
        if self.working_area and self.hover_active:
            # Show annotation for points (positive or negative) regardless of hover state
            if len(self.positive_points) > 0 or len(self.negative_points) > 0:
                self.cursor_annotation = self.create_annotation(position, False)
                if self.cursor_annotation:
                    # Store the hover graphics item for easy removal later
                    self.hover_graphics = self.cursor_annotation.create_graphics_item(self.annotation_window.scene)
            # Allow cursor annotation during rectangle drawing if we have both start and end points
            elif self.drawing_rectangle and self.start_point and self.end_point:
                if self.top_left is not None and self.bottom_right is not None:
                    self.cursor_annotation = self.create_annotation(position, False)
                    if self.cursor_annotation:
                        # Store the hover graphics item for easy removal later
                        self.hover_graphics = self.cursor_annotation.create_graphics_item(self.annotation_window.scene)
            # Normal hover annotation (just hover point)
            elif not self.drawing_rectangle and self.hover_point:
                self.cursor_annotation = self.create_annotation(position, False)
                if self.cursor_annotation:
                    # Store the hover graphics item for easy removal later
                    self.hover_graphics = self.cursor_annotation.create_graphics_item(self.annotation_window.scene)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Override parent method: Update the cursor annotation to a new position.
        
        Args:
            scene_pos (QPointF): The new scene position for the cursor annotation
        """
        # Remove the current cursor annotation and create a new one
        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)
        self.annotation_window.scene.update()

    def clear_cursor_annotation(self):
        """
        Override parent method: Clear the current cursor annotation.
        """
        if self.cursor_annotation:
            if self.hover_graphics:
                self.annotation_window.scene.removeItem(self.hover_graphics)
                self.hover_graphics = None
            self.cursor_annotation.delete()
            self.cursor_annotation = None

    def cancel_hover_annotation(self):
        """
        Cancel the hover annotation.
        """
        self.clear_cursor_annotation()
        self.hover_point = None
        self.annotation_window.scene.update()

    def cancel_rectangle_annotation(self):
        """
        Cancel the rectangle annotation.
        """
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False
        self.clear_cursor_annotation()

        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        self.annotation_window.scene.update()

    def cancel_annotation(self):
        """
        Cancel the annotation.
        """
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)

        if self.annotation_graphics:
            self.annotation_window.scene.removeItem(self.annotation_graphics)

        self.points = []
        self.positive_points = []
        self.negative_points = []
        self.point_graphics = []
        self.rectangle_graphics = None

        self.cancel_hover_annotation()
        self.cancel_rectangle_annotation()
        self.annotation_window.scene.update()

    def cancel_working_area(self):
        """
        Cancel the working area.
        """
        if self.working_area:
            self.annotation_window.scene.removeItem(self.working_area)
            self.working_area = None

        if self.shadow_area:
            self.annotation_window.scene.removeItem(self.shadow_area)
            self.shadow_area = None

        self.image_path = None
        self.original_image = None
        self.image = None
        
        # Reset the active prompts flag
        self.has_active_prompts = False
        
        # Clean up any remaining graphics
        self.cancel_annotation()
