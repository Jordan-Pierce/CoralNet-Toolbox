import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QPixmap
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

        self.positive_points = []
        self.negative_points = []

        # Working area (dashed rectangle)
        self.working_area = None
        self.image = None

        self.complete = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.sam_dialog = self.annotation_window.main_window.sam_deploy_model_dialog

    def mousePressEvent(self, event: QMouseEvent):

        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                # Add a positive point
                self.positive_points.append(self.annotation_window.mapToScene(event.pos()))
                # Draw a green point on the scene
                point = self.annotation_window.scene.addEllipse(event.pos().x() - 2,
                                                                event.pos().y() - 2, 4, 4,
                                                                QPen(Qt.green),
                                                                QColor(Qt.green))
                self.point_graphics.append(point)
                self.annotation_window.scene.addItem(point)

            elif event.modifiers() == Qt.ShiftModifier:
                # Add a negative point
                self.negative_points.append(self.annotation_window.mapToScene(event.pos()))
                # Draw a red point on the scene
                point = self.annotation_window.scene.addEllipse(event.pos().x() - 2,
                                                                event.pos().y() - 2, 4, 4,
                                                                QPen(Qt.red),
                                                                QColor(Qt.red))
                self.point_graphics.append(point)
                self.annotation_window.scene.addItem(point)

    def mouseMoveEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event: QKeyEvent):
        # If no points have been added
        if event.key() == Qt.Key_Space and not len(self.positive_points):
            # If there isn't a working area, create one and set the image
            if not self.working_area:
                # Set the working area
                self.set_working_area()
                self.sam_dialog.set_image(self.image)
            else:
                self.cancel_working_area()
                self.cancel_annotation()

        # If points have been added
        if event.key() == Qt.Key_Space and len(self.positive_points):
            # Add the annotation
            self.annotation_window.add_annotation()
            self.cancel_annotation()

    def set_working_area(self):
        # Make the cursor busy
        self.annotation_window.setCursor(Qt.WaitCursor)

        # Remove the previous working area
        self.cancel_working_area()

        # Get the bounding rectangle of the image within the scene
        if self.annotation_window.image_pixmap:
            image_rect = self.annotation_window.image_pixmap.rect()
            scene_rect = self.annotation_window.mapToScene(image_rect).boundingRect()
        else:
            return

        # Capture the current view of the image within the scene
        view_rect = self.annotation_window.mapFromScene(scene_rect).boundingRect()
        qimage = self.annotation_window.grab(view_rect).toImage()
        # Convert QImage to QPixmap
        qpixmap = QPixmap.fromImage(qimage)
        # Convert QPixmap to numpy array
        self.image = pixmap_to_numpy(qpixmap)

        # Create a green dashed line rectangle around the image
        pen = QPen(Qt.green)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(5)
        self.working_area = self.annotation_window.scene.addRect(scene_rect, pen=pen)

        # Restore the cursor
        self.annotation_window.setCursor(Qt.CrossCursor)

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):

        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap:
            return None

        # Provide prompt to SAM model in form of numpy array
        positive = [(point.x(), point.y()) for point in self.positive_points]
        negative = [(point.x(), point.y()) for point in self.negative_points]
        labels = [1] * len(positive) + [0] * len(negative)
        points = positive + negative

        # Get the results from SAM model
        results = self.sam_dialog.predict(None, points, labels)

        if not results:
            return None

        # Convert the results to a PolygonAnnotation
        points = results.masks.xy[0].tolist()
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
        self.points = []
        self.positive_points = []
        self.negative_points = []

        return annotation

    def cancel_annotation(self):
        # Remove the positive and negative points
        for point in self.point_graphics:
            self.annotation_window.scene.removeItem(point)
        self.positive_points = []
        self.negative_points = []
        self.point_graphics = []

    def cancel_working_area(self):
        if self.working_area:
            self.annotation_window.scene.removeItem(self.working_area)
            self.working_area = None
            self.image = None