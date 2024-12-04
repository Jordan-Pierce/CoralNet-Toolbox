import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.top_left = None
        self.sam_dialog = None

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

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.sam_dialog = self.annotation_window.main_window.sam_deploy_model_dialog
        self.hover_active = True  # Ensure hover is active when SAMTool is activated

    def deactivate(self):
        """
        Deactivates the tool.
        """
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.sam_dialog = None
        self.cancel_annotation()
        self.cancel_working_area()
        self.hover_active = False  # Ensure hover is inactive when SAMTool is deactivated

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
                # Toggle the cursor annotation
                self.annotation_window.toggle_cursor_annotation(adjusted_pos)

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

            self.annotation_window.toggle_cursor_annotation()

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

        if event.modifiers() == Qt.ControlModifier:

            if event.button() == Qt.LeftButton:
                self.positive_points.append(adjusted_pos)
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                point.setPen(QPen(Qt.green))
                point.setBrush(QColor(Qt.green))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)

            elif event.button() == Qt.RightButton:
                self.negative_points.append(adjusted_pos)
                point = QGraphicsEllipseItem(scene_pos.x() - 10, scene_pos.y() - 10, 20, 20)
                point.setPen(QPen(Qt.red))
                point.setBrush(QColor(Qt.red))
                self.annotation_window.scene.addItem(point)
                self.point_graphics.append(point)

            # Update the cursor annotation
            self.annotation_window.toggle_cursor_annotation(event.pos())

        elif event.modifiers() != Qt.ControlModifier:

            if event.button() == Qt.LeftButton and not self.drawing_rectangle:
                # Remove the hover annotation
                self.cancel_hover_annotation()
                # Get the start point
                self.start_point = self.annotation_window.mapToScene(event.pos())
                # Start drawing the rectangle
                self.drawing_rectangle = True

            elif event.button() == Qt.LeftButton and self.drawing_rectangle:
                # Get the end point
                self.end_point = self.annotation_window.mapToScene(event.pos())
                # Finish drawing the rectangle
                self.drawing_rectangle = False

            # Update the cursor annotation
            self.annotation_window.toggle_cursor_annotation()

        elif event.button() == Qt.RightButton and self.drawing_rectangle:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

        self.annotation_window.viewport().update()

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
            image_pixmap = self.annotation_window.image_pixmap
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and image_pixmap and cursor_in_window and self.start_point:
                self.annotation_window.toggle_cursor_annotation(self.end_point)
        else:
            if self.annotation_window.cursorInWindow(event.pos()):
                self.start_hover_timer(event.pos())
            else:
                self.cancel_hover_annotation()
                self.stop_hover_timer()

        self.annotation_window.viewport().update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles the key press event.
        
        Args:
            event (QKeyEvent): The key press event
        """

        if not event.key() == Qt.Key_Space:
            return

        # If there is no working area, set it
        if not self.working_area:
            self.set_working_area()
            self.sam_dialog.set_image(self.image, self.image_path)
        # If there is a bounding box, add the annotation
        elif self.start_point and self.end_point and not self.drawing_rectangle:
            self.annotation_window.add_annotation()
        # If there are positive points, add the annotation
        elif len(self.positive_points):
            # If there are positive points, add the annotation
            self.annotation_window.add_annotation()
        else:
            self.cancel_working_area()

        self.cancel_annotation()
        self.annotation_window.viewport().update()

    def set_working_area(self):
        """
        Set the working area for the tool.
        """
        self.annotation_window.setCursor(Qt.WaitCursor)

        # Cancel the current working area if it exists
        self.cancel_working_area()

        # Original image (grab current from the annotation window)
        self.image_path = self.annotation_window.current_image_path
        self.original_image = pixmap_to_numpy(self.annotation_window.image_pixmap)
        self.original_width = self.annotation_window.image_pixmap.size().width()
        self.original_height = self.annotation_window.image_pixmap.size().height()

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
        pen.setWidth(10)
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
        self.annotation_window.viewport().update()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        """
        Create an annotation based on the given scene position.
        
        Args:
            scene_pos (QPointF): The scene position
            finished (bool): Flag to indicate if the annotation is finished
        """
        if not self.annotation_window.active_image:
            return None
        
        if not self.annotation_window.image_pixmap:
            return None
        
        if not self.working_area:
            return None

        # Get the current transparency
        transparency = self.annotation_window.main_window.label_window.active_label.transparency

        # Get the positive and negative points, bbox
        positive = [[point.x(), point.y()] for point in self.positive_points]
        negative = [[point.x(), point.y()] for point in self.negative_points]
        bbox = np.array([])

        if self.hover_point:
            positive.append([self.hover_point.x(), self.hover_point.y()])
            transparency //= 4

        if self.start_point and self.end_point:
            bbox = np.array([self.top_left.x(),
                             self.top_left.y(),
                             self.bottom_right.x(),
                             self.bottom_right.y()])

        # Create the labels, points, and bbox as numpy arrays
        labels = np.array([1] * len(positive) + [0] * len(negative))
        points = np.array(positive + negative)

        # Predict the mask provided prompts
        results = self.sam_dialog.predict_from_prompts(bbox, points, labels)

        if not results:
            return None

        if results.boxes.conf[0] < self.annotation_window.main_window.get_uncertainty_thresh():
            return None
        
        # TODO use results processor
        # Get the points of the top1 mask
        top1_index = np.argmax(results.boxes.conf)
        predictions = results[top1_index].masks.xy[0]

        # Move the points back to the original image space
        working_area_top_left = self.working_area.rect().topLeft()
        points = [(point[0] + working_area_top_left.x(), point[1] + working_area_top_left.y()) for point in predictions]
        self.points = [QPointF(*point) for point in points]

        # Create the annotation
        annotation = PolygonAnnotation(self.points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       transparency)

        # Ensure the PolygonAnnotation is added to the scene after creation
        annotation.create_graphics_item(self.annotation_window.scene)

        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        # Handle hover graphics separately
        if self.hover_point:
            self.cancel_hover_annotation()
            self.hover_graphics = annotation.graphics_item
        else:
            # Remove the previous annotation graphics
            if self.annotation_graphics:
                self.annotation_window.scene.removeItem(self.annotation_graphics)
            self.annotation_graphics = annotation.graphics_item

        self.annotation_window.viewport().update()

        return annotation

    def cancel_hover_annotation(self):
        """
        Cancel the hover annotation.
        """
        if self.hover_graphics:
            self.annotation_window.scene.removeItem(self.hover_graphics)

        self.hover_point = None
        self.hover_graphics = None
        self.annotation_window.viewport().update()

    def cancel_rectangle_annotation(self):
        """
        Cancel the rectangle annotation.
        """
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False
        self.annotation_window.toggle_cursor_annotation()

        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        self.annotation_window.viewport().update()

    def cancel_annotation(self):
        """
        Cancel the annotation.
        """
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)

        if self.annotation_graphics:
            self.annotation_window.scene.removeItem(self.annotation_graphics)

        if self.hover_graphics:
            self.annotation_window.scene.removeItem(self.hover_graphics)

        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)

        self.points = []
        self.positive_points = []
        self.negative_points = []
        self.point_graphics = []
        self.rectangle_graphics = None

        self.cancel_hover_annotation()
        self.cancel_rectangle_annotation()
        self.annotation_window.viewport().update()

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