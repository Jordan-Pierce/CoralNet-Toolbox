import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy

import cv2
import numpy as np

import torch
import supervision as sv

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem, QApplication

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.ResultsProcessor import ResultsProcessor

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SeeAnythingTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)

        self.annotation_window = annotation_window
        self.main_window = self.annotation_window.main_window

        self.see_anything_dialog = None

        self.top_left = None

        self.cursor = Qt.CrossCursor
        self.annotation_graphics = None

        self.image = None
        self.rectangles = []  # Store rectangle coordinates for SeeAnything processing
        self.rectangle_items = []  # Store QGraphicsRectItem objects

        self.working_area = None
        self.shadow_area = None

        self.image_path = None
        self.original_image = None
        self.original_width = None
        self.original_height = None

        # Rectangle drawing attributes
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False
        self.current_rect_graphics = None  # For the rectangle currently being drawn
        self.rectangles_processed = False  # Track if rectangles have been processed

        self.annotations = []
        self.results = None

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        self.see_anything_dialog = self.main_window.see_anything_deploy_predictor_dialog

    def deactivate(self):
        """
        Deactivates the tool and cleans up all resources.
        """
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)

        # Clear annotations that haven't been confirmed
        if self.annotations:
            self.clear_annotations()

        # Clear rectangle data and graphics
        self.clear_all_rectangles()

        # Clean up working area and shadow
        self.cancel_working_area()

        # Clear detection data
        self.results = None

        # Update the viewport
        self.annotation_window.viewport().update()

    def get_workarea_thickness(self):
        """Calculate appropriate line thickness based on current view dimensions."""
        extent = self.annotation_window.viewportToScene()
        view_width = round(extent.width())
        view_height = round(extent.height())
        return max(5, min(20, max(view_width, view_height) // 1000))

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
        self.annotation_window.viewport().update()

    def create_rectangle_graphics(self):
        """
        Create a new rectangle graphics item for drawing with the selected label color.
        """
        if self.start_point and self.end_point:
            # Calculate the rectangle dimensions
            rect = QRectF(
                min(self.start_point.x(), self.end_point.x()),
                min(self.start_point.y(), self.end_point.y()),
                abs(self.end_point.x() - self.start_point.x()),
                abs(self.end_point.y() - self.start_point.y())
            )

            # Remove current rectangle being drawn if it exists
            if self.current_rect_graphics:
                self.annotation_window.scene.removeItem(self.current_rect_graphics)

            # Create a new rectangle graphics item
            self.current_rect_graphics = QGraphicsRectItem(rect)

            # Get color from the selected label
            color = self.annotation_window.selected_label.color

            # Style the rectangle
            pen = QPen(QColor(color))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            self.current_rect_graphics.setPen(pen)

            # Add to scene
            self.annotation_window.scene.addItem(self.current_rect_graphics)

    def update_rectangle_graphics(self):
        """
        Update the current rectangle graphics item while drawing.
        """
        if self.start_point and self.end_point and self.drawing_rectangle:
            # If no graphics item exists yet, create one
            if not self.current_rect_graphics:
                self.create_rectangle_graphics()
            else:
                # Update the existing rectangle
                rect = QRectF(
                    min(self.start_point.x(), self.end_point.x()),
                    min(self.start_point.y(), self.end_point.y()),
                    abs(self.end_point.x() - self.start_point.x()),
                    abs(self.end_point.y() - self.start_point.y())
                )
                self.current_rect_graphics.setRect(rect)

    def add_completed_rectangle(self):
        """
        Add the completed rectangle to the list of rectangles and their graphics.
        """
        if self.current_rect_graphics:
            # Add the current rectangle graphics item to the list
            self.rectangle_items.append(self.current_rect_graphics)

            # Calculate rectangle coordinates relative to working area
            working_area_top_left = self.working_area.rect().topLeft()

            top_left = QPointF(
                min(self.start_point.x(), self.end_point.x()) - working_area_top_left.x(),
                min(self.start_point.y(), self.end_point.y()) - working_area_top_left.y()
            )

            bottom_right = QPointF(
                max(self.start_point.x(), self.end_point.x()) - working_area_top_left.x(),
                max(self.start_point.y(), self.end_point.y()) - working_area_top_left.y()
            )

            # Add the rectangle coordinates to the list
            rectangle = np.array([top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()])
            self.rectangles.append(rectangle)

            # Reset the current rectangle graphics item
            self.current_rect_graphics = None
            # Set rectangles_processed to False since we have new rectangles
            self.rectangles_processed = False

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

        if event.button() == Qt.LeftButton and not self.drawing_rectangle:
            # Get the start point
            self.start_point = scene_pos
            # Start drawing the rectangle
            self.drawing_rectangle = True
            self.end_point = self.start_point  # Initialize end_point
            self.update_rectangle_graphics()

        elif event.button() == Qt.LeftButton and self.drawing_rectangle:
            # Get the end point
            self.end_point = scene_pos
            # Finish drawing the rectangle
            self.drawing_rectangle = False
            # Update the rectangle graphics before finalizing
            self.update_rectangle_graphics()

            # Add the completed rectangle to our lists
            self.add_completed_rectangle()

            # Reset drawing state
            self.start_point = None
            self.end_point = None

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
            self.update_rectangle_graphics()

        self.annotation_window.viewport().update()

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
                self.see_anything_dialog.set_image(self.image, self.image_path)

            # If there are rectangles already made and not processed, run the predictor
            elif len(self.rectangles) > 0 and not self.rectangles_processed:
                # Create annotation based on rectangles
                self.create_annotation(None)
                # After creating annotation, only clear the rectangle graphics but keep the data
                self.clear_rectangle_graphics()
                # Mark rectangles as processed
                self.rectangles_processed = True
            else:
                # If there's a working area but no rectangles or if rectangles have been processed,
                # confirm the annotations, and cancel the working area.
                if self.rectangles or self.annotations:
                    if self.see_anything_dialog.use_sam_dropdown.currentText() == "True":
                        self.apply_sam_model()
                    else:
                        # Confirm the annotations
                        self.confirm_annotations()
                # Cancel the working area
                self.cancel_working_area()

        elif event.key() == Qt.Key_Backspace:
            # Cancel current rectangle being drawn
            if self.drawing_rectangle:
                self.drawing_rectangle = False
                if self.current_rect_graphics:
                    self.annotation_window.scene.removeItem(self.current_rect_graphics)
                    self.current_rect_graphics = None
                self.start_point = None
                self.end_point = None
            # If we have a working area and annotations, clear them
            elif self.working_area and len(self.annotations) > 0:
                self.clear_annotations()
            # If not drawing and no annotations to clear, clear all rectangles
            else:
                self.clear_all_rectangles()

        self.annotation_window.viewport().update()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        """
        Create an annotation based on the given scene position and rectangles.

        Args:
            scene_pos (QPointF): The scene position (not used when using rectangles)
            finished (bool): Flag to indicate if the annotation is finished
        """
        if not self.annotation_window.active_image:
            return None

        if not self.annotation_window.pixmap_image:
            return None

        if not self.working_area:
            return None

        if len(self.rectangles) == 0:
            return None

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Move the points back to the original image space
        working_area_top_left = self.working_area.rect().topLeft()

        # Get the current transparency
        transparency = self.main_window.label_window.active_label.transparency

        # Predict the mask provided prompts
        results = self.see_anything_dialog.predict_from_prompts(self.rectangles)[0]

        if not results:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            return None

        self.results = results
        image_area = self.image.shape[0] * self.image.shape[1]

        # Clear the annotations (that haven't been confirmed)
        self.clear_annotations()

        # Loop through the results
        for result in self.results:
            # Extract values from result
            confidence = result.boxes.conf.item()

            if confidence < self.main_window.get_uncertainty_thresh():
                continue

            box = result.boxes.xyxyn.detach().cpu().numpy().squeeze()

            # Convert from normalized to pixel coordinates
            box = box * np.array([self.image.shape[1],
                                  self.image.shape[0],
                                  self.image.shape[1],
                                  self.image.shape[0]])

            # Convert to whole image coordinates
            box[0] += working_area_top_left.x()
            box[1] += working_area_top_left.y()
            box[2] += working_area_top_left.x()
            box[3] += working_area_top_left.y()

            # Check box area relative to image area
            box_area = (box[2] - box[0]) * (box[3] - box[1])

            # self.main_window.get_area_thresh_min()
            if box_area < self.main_window.get_area_thresh_min() * image_area:
                continue

            if box_area > self.main_window.get_area_thresh_max() * image_area:
                continue

            # Add the box to the list of rectangles (compounding automatic annotation)
            self.rectangles.append(box)

            if self.see_anything_dialog.task == "segment":
                # Get the mask from the result
                mask = result.masks.data.detach().cpu().numpy().squeeze().astype(int)
                # # Resize the mask to the resized image shape
                mask = cv2.resize(mask, (result.orig_img.shape[1], result.orig_img.shape[0]))
                # Convert to polygons
                polygons = sv.detection.utils.mask_to_polygons(mask)

                if len(polygons) == 1:
                    polygon = polygons[0]
                else:
                    # Grab the index of the largest polygon
                    polygon = max(polygons, key=lambda x: len(x))

                # Renormalize points by resized image dimensions
                normalized_points = polygon / np.array([result.orig_img.shape[1], result.orig_img.shape[0]])

                # Scale to working area dimensions
                points = normalized_points * np.array([self.image.shape[1], self.image.shape[0]])

                # Convert to whole image coordinates
                points[:, 0] += working_area_top_left.x()
                points[:, 1] += working_area_top_left.y()

                # Create the polygon annotation
                self.create_polygon_annotation(points, confidence, transparency)
            else:
                # Create the rectangle annotation
                self.create_rectangle_annotation(box, confidence, transparency)

        self.annotation_window.viewport().update()

        # Make cursor normal
        QApplication.restoreOverrideCursor()

    def create_rectangle_annotation(self, box, confidence, transparency):
        """
        Create rectangle annotations based on the given box coordinates.

        Args:
            box (np.ndarray): The bounding box coordinates.
            transparency (int): The transparency level for the annotation.
        """
        # Convert to QPointF
        top_left = QPointF(box[0], box[1])
        bottom_right = QPointF(box[2], box[3])

        # Create the annotation
        annotation = RectangleAnnotation(top_left,
                                         bottom_right,
                                         self.annotation_window.selected_label.short_label_code,
                                         self.annotation_window.selected_label.long_label_code,
                                         self.annotation_window.selected_label.color,
                                         self.annotation_window.current_image_path,
                                         self.annotation_window.selected_label.id,
                                         transparency)

        # Update the confidence score of annotation
        annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})

        # Ensure the annotation is added to the scene after creation (but not saved yet)
        annotation.create_graphics_item(self.annotation_window.scene)
        self.annotations.append(annotation)

    def create_polygon_annotation(self, points, confidence, transparency):
        """
        Create polygon annotations based on the given points.

        Args:
            points (np.ndarray): The polygon points.
            confidence (float): The confidence score for the annotation.
            transparency (int): The transparency level for the annotation.
        """
        # Convert to QPointF
        points = [QPointF(point[0], point[1]) for point in points]
        # Create the annotation
        annotation = PolygonAnnotation(points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       transparency)

        # Update the confidence score of annotation
        annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})

        # Ensure the annotation is added to the scene after creation (but not saved yet)
        annotation.create_graphics_item(self.annotation_window.scene)
        self.annotations.append(annotation)

    def confirm_annotations(self):
        """
        Confirm the annotations and clear the working area.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Confirming Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.annotations))

        # Confirm the annotations
        for annotation in self.annotations:
            # Connect signals needed for proper interaction
            annotation.selected.connect(self.annotation_window.select_annotation)
            annotation.annotationDeleted.connect(self.annotation_window.delete_annotation)
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)

            # Create cropped image if not already done
            if not annotation.cropped_image and self.annotation_window.rasterio_image:
                annotation.create_cropped_image(self.annotation_window.rasterio_image)

            # Add to annotation dict
            self.annotation_window.add_annotation_to_dict(annotation)
            # Update the table in ImageWindow
            self.annotation_window.annotationCreated.emit(annotation.id)

            # Update progress bar
            progress_bar.update_progress()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
        progress_bar = None

        # Clear the working area
        self.cancel_working_area()

        # Clear the annotations list
        self.annotations = []
        self.results = None

    def apply_sam_model(self):
        """Uses the Results with SAM predictor to create polygons instead of confirming the
        ones created by the SeeAnything predictor."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create a class mapping dictionary
        class_mapping = {0: self.annotation_window.selected_label}

        # Create a results processor
        results_processor = ResultsProcessor(
            self.main_window,
            class_mapping,
            uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
            iou_thresh=self.main_window.get_iou_thresh(),
            min_area_thresh=self.main_window.get_area_thresh_min(),
            max_area_thresh=self.main_window.get_area_thresh_max()
        )

        # Make a copy of the results
        results = copy.deepcopy(self.results)

        # Replace the resized image with the original image
        results.orig_img = self.original_image.copy()
        results.orig_shape = self.original_image.shape
        results.path = self.image_path
        results.names = {0: class_mapping[0].short_label_code}

        results = self.see_anything_dialog.sam_dialog.predict_from_results([results])

        # Process the results
        results_processor.process_segmentation_results(results)

        # Make cursor normal
        QApplication.restoreOverrideCursor()

        # Clear the previous, non-confirmed annotations
        self.clear_annotations()

        # Clear the working area
        self.cancel_working_area()

    def clear_annotations(self):
        """
        Clear all annotations created by this tool from the scene without confirming them.
        """
        # Remove all annotations from the scene
        for annotation in self.annotations:
            if annotation.graphics_item:
                self.annotation_window.scene.removeItem(annotation.graphics_item)

            annotation.delete()
            annotation = None

        # Clear the annotations list
        self.annotations = []

        # Also clear rectangles since they've been processed into annotations
        self.clear_rectangle_graphics()

        self.annotation_window.viewport().update()

    def clear_rectangle_graphics(self):
        """
        Clear rectangle graphics from the scene but keep the data.
        """
        # Remove all rectangle graphics from scene
        for rect_item in self.rectangle_items:
            self.annotation_window.scene.removeItem(rect_item)

        # Clear the rectangle graphics if one is being drawn
        if self.current_rect_graphics:
            self.annotation_window.scene.removeItem(self.current_rect_graphics)
            self.current_rect_graphics = None

        # Reset the graphics list
        self.rectangle_items = []

    def clear_rectangle_data(self):
        """
        Clear rectangle data structures but keep the graphics.
        """
        self.rectangles = []
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False
        self.rectangles_processed = False

    def clear_all_rectangles(self):
        """
        Clear all rectangle graphics and data.
        """
        self.clear_rectangle_graphics()
        self.clear_rectangle_data()
        self.rectangles_processed = False

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

        # Clear all rectangles when canceling the working area
        self.clear_all_rectangles()
        self.rectangles_processed = False

        self.annotations = []
        self.results = None
