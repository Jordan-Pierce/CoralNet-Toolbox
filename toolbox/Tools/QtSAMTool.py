import warnings

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem

from toolbox.Tools.QtTool import Tool
from toolbox.QtPolygonAnnotation import PolygonAnnotation

from toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.sam_dialog = None

        self.cursor = Qt.CrossCursor
        self.points = []
        self.point_graphics = []

        self.image = None
        self.bbox = []
        self.positive_points = []
        self.negative_points = []

        self.working_area = None

        self.original_image = None
        self.original_width = None
        self.original_height = None

        self.complete = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.sam_dialog = self.annotation_window.main_window.sam_deploy_model_dialog

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.sam_dialog = None
        self.cancel_annotation()
        self.cancel_working_area()

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.modifiers() == Qt.ControlModifier:
            scene_pos = self.annotation_window.mapToScene(event.pos())

            # Get the adjusted position relative to the working area's top-left corner
            working_area_top_left = self.working_area.rect().topLeft()
            adjusted_pos = QPointF(scene_pos.x() - working_area_top_left.x(),
                                   scene_pos.y() - working_area_top_left.y())

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

        self.annotation_window.viewport().update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space and not len(self.positive_points):
            if not self.working_area:
                self.set_working_area()
                self.sam_dialog.set_image(self.image)
            else:
                self.cancel_working_area()
                self.cancel_annotation()

        if event.key() == Qt.Key_Space and len(self.positive_points):
            self.annotation_window.add_annotation()
            self.cancel_annotation()

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

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):

        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap:
            return None

        # Adjust points relative to the working area's top-left corner
        working_area_top_left = self.working_area.rect().topLeft()
        positive = [(point.x(), point.y()) for point in self.positive_points]
        negative = [(point.x(), point.y()) for point in self.negative_points]
        labels = [1] * len(positive) + [0] * len(negative)
        points = positive + negative

        # Predict the mask
        results = self.sam_dialog.predict(None, points, labels)

        if not results:
            return None

        # import matplotlib.pyplot as plt
        #
        # for mask in results.masks:
        #     plt.figure()
        #     points = mask.xy[0]
        #     plt.imshow(self.image)
        #     plt.plot(points.T[0], points.T[1], c='red')
        #     plt.show()

        # Move the points back to the original image space
        points = [(point[0] + working_area_top_left.x(), point[1] + working_area_top_left.y()) for point in points]
        self.points = [QPointF(*point) for point in points]

        # Create the annotation
        annotation = PolygonAnnotation(self.points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       self.annotation_window.main_window.label_window.active_label.transparency,
                                       show_msg=False)
        # Clear the points
        self.cancel_annotation()

        return annotation

    def cancel_annotation(self):
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)

        self.bbox = []
        self.points = []
        self.positive_points = []
        self.negative_points = []
        self.point_graphics = []

    def cancel_working_area(self):
        if self.working_area:
            self.annotation_window.scene.removeItem(self.working_area)
            self.working_area = None
            self.image = None