import warnings

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem

from toolbox.Tools.QtTool import Tool
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.sam_dialog = self.annotation_window.main_window.sam_deploy_model_dialog
        self.hover_active = True  # Ensure hover is active when SAMTool is activated

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.sam_dialog = None
        self.cancel_annotation()
        self.cancel_working_area()
        self.hover_active = False  # Ensure hover is inactive when SAMTool is deactivated

    def start_hover_timer(self, pos):
        if self.hover_active:
            self.hover_timer.start(3000)
            self.hover_pos = pos

    def stop_hover_timer(self):
        self.hover_timer.stop()
        self.display_hover_annotation()

    def display_hover_annotation(self):
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

        if not event.key() == Qt.Key_Space:
            return

        # If there is no working area, set it
        if not self.working_area:
            self.set_working_area()
            self.sam_dialog.set_image(self.image)
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
        self.annotation_window.setCursor(Qt.WaitCursor)

        # Cancel the current working area if it exists
        self.cancel_working_area()

        # Original image (grab current from the annotation window)
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

        # Create the graphic
        pen = QPen(Qt.green)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(10)
        self.working_area = self.annotation_window.scene.addRect(working_rect, pen=pen)

        # Crop the image based on the working_rect
        self.image = self.original_image[top:bottom, left:right]

        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.viewport().update()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):

        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap or not self.working_area:
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

        # Predict the mask
        results = self.sam_dialog.predict(bbox, points, labels)

        if not results:
            return None

        if results.boxes.conf[0] < self.sam_dialog.conf:
            return None

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
                                       transparency,
                                       show_msg=False)

        # Ensure the PolygonAnnotation is added to the scene after creation
        annotation.create_graphics_item(self.annotation_window.scene)
        self.annotation_window.scene.addItem(annotation.graphics_item)

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
        if self.hover_graphics:
            self.annotation_window.scene.removeItem(self.hover_graphics)

        self.hover_point = None
        self.hover_graphics = None
        self.annotation_window.viewport().update()

    def cancel_rectangle_annotation(self):
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False
        self.annotation_window.toggle_cursor_annotation()

        if self.rectangle_graphics:
            self.annotation_window.scene.removeItem(self.rectangle_graphics)
            self.rectangle_graphics = None

        self.annotation_window.viewport().update()

    def cancel_annotation(self):
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
        if self.working_area:
            self.annotation_window.scene.removeItem(self.working_area)
            self.working_area = None
            self.image = None