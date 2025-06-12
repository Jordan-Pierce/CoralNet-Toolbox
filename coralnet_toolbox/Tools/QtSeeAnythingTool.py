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

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import CombineResults
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import simplify_polygon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SeeAnythingTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)

        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window

        self.see_anything_dialog = None

        self.top_left = None

        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor  # Add this for clarity
        self.annotation_graphics = None

        self.work_area_image = None
        self.rectangles = []       # Store rectangle coordinates for SeeAnything processing
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
        self.annotation_window.setCursor(self.cursor)
        self.see_anything_dialog = self.main_window.see_anything_deploy_predictor_dialog

    def deactivate(self):
        """
        Deactivates the tool and cleans up all resources.
        """
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)

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
        self.annotation_window.scene.update()

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

        # Get the thickness for the working area graphics
        pen_width = self.graphics_utility.get_workarea_thickness(self.annotation_window)

        # Create and add the working area graphics
        self.working_area.create_graphics(self.annotation_window.scene, pen_width)
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)

        # Create a semi-transparent overlay for the shadow
        shadow_brush = QBrush(QColor(0, 0, 0, 150))  # Semi-transparent black
        shadow_path = QPainterPath()
        shadow_path.addRect(self.annotation_window.scene.sceneRect())  # Cover the entire scene
        shadow_path.addRect(self.working_area.rect)  # Add the work area rect
        # Subtract the work area from the overlay
        shadow_path = shadow_path.simplified()

        # Create the shadow item
        self.shadow_area = QGraphicsPathItem(shadow_path)
        self.shadow_area.setBrush(shadow_brush)
        self.shadow_area.setPen(QPen(Qt.NoPen))  # No outline for the shadow

        # Add the shadow item to the scene
        self.annotation_window.scene.addItem(self.shadow_area)

        # Crop the image based on the working area
        self.work_area_image = self.original_image[top:bottom, left:right]

        # Set the image in the SeeAnything dialog
        self.see_anything_dialog.set_image(self.work_area_image, self.image_path)

        self.annotation_window.setCursor(Qt.CrossCursor)
        self.annotation_window.scene.update()

    def on_working_area_removed(self, work_area):
        """
        Handle when the work area is removed via its internal mechanism.
        """
        self.cancel_working_area()

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
                if self.current_rect_graphics in self.annotation_window.scene.items():
                    self.annotation_window.scene.removeItem(self.current_rect_graphics)
                self.current_rect_graphics = None

            # Create a new rectangle graphics item
            self.current_rect_graphics = QGraphicsRectItem(rect)

            # Get color from the selected label
            color = self.annotation_window.selected_label.color

            # Get the thickness for the rectangle graphics
            width = self.graphics_utility.get_rectangle_graphic_thickness(self.annotation_window)

            # Style the rectangle
            pen = QPen(QColor(color))
            pen.setWidth(width)
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
            # Add the current rectangle graphics item to the list if it's in the scene
            if self.current_rect_graphics in self.annotation_window.scene.items():
                self.rectangle_items.append(self.current_rect_graphics)
            else:
                # If it's not in the scene anymore for some reason, don't track it
                self.current_rect_graphics = None
                return

            # Calculate rectangle coordinates relative to working area
            working_area_top_left = self.working_area.rect.topLeft()

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

            # Reset the current rectangle graphics item without removing from scene
            # It's now tracked in rectangle_items
            self.current_rect_graphics = None

            # Set rectangles_processed to False since we have new user rectangles
            self.rectangles_processed = False  # Indicate prediction is needed

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
        if not self.working_area.contains_point(scene_pos):
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
            self.update_rectangle_graphics()

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

            # If there are user-drawn rectangles ready for processing, run the predictor
            elif len(self.rectangles) > 0 and not self.rectangles_processed:
                # Create annotation based on the user-drawn rectangles
                self.create_annotations_from_rectangles()
                # Clear the user-drawn rectangles (graphics and data) as they've been used
                self.clear_all_rectangles()
                # Mark rectangles as processed for this cycle
                self.rectangles_processed = True
            else:
                # If there's a working area but no new user rectangles,
                # or if rectangles have been processed, confirm the accumulated annotations.
                if self.annotations:  # Check if there are any annotations to confirm/process
                    if self.see_anything_dialog.use_sam_dropdown.currentText() == "True":
                        self.apply_sam_model()
                    else:
                        # Confirm the annotations accumulated so far
                        self.confirm_annotations()
                # Cancel the working area if no annotations were generated or after confirmation/SAM
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
            # If we have a working area and accumulated annotations, clear them
            elif self.working_area and len(self.annotations) > 0:
                self.clear_annotations()  # Clears unconfirmed annotations
            # If not drawing and no annotations to clear, clear any pending user-drawn rectangles
            else:
                self.clear_all_rectangles()  # Clears user input rectangles

        self.annotation_window.scene.update()

    def create_annotations_from_rectangles(self):
        """
        Create annotations based on the user-drawn rectangles.
        """
        if not self.annotation_window.active_image:
            return None

        if not self.annotation_window.pixmap_image:
            return None

        if not self.working_area:
            return None

        if len(self.rectangles) == 0:  # Check specifically for user-drawn rectangles
            return None

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Move the points back to the original image space
        working_area_top_left = self.working_area.rect.topLeft()

        # Predict the mask provided prompts (using only the current user-drawn rectangles)
        results = self.see_anything_dialog.predict_from_prompts(self.rectangles)[0]

        if not results:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            return None

        # Create a results processor to merge and filter results
        results_processor = ResultsProcessor(self.main_window,
                                             uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
                                             iou_thresh=self.main_window.get_iou_thresh(),
                                             min_area_thresh=self.main_window.get_area_thresh_min(),
                                             max_area_thresh=self.main_window.get_area_thresh_max())
        # Merge
        if self.results:
            results = CombineResults().combine_results([self.results, results])

        # Filter
        self.results = results_processor.apply_filters_to_results(results)

        # Calculate the area of the image
        image_area = self.work_area_image.shape[0] * self.work_area_image.shape[1]

        # Clear previous annotations if any
        self.clear_annotations()

        # Loop through the results from the current prediction
        for result in self.results:
            # Extract values from result
            confidence = result.boxes.conf.item()

            if confidence < self.main_window.get_uncertainty_thresh():
                continue

            # Get the bounding box coordinates (x1, y1, x2, y2) in normalized format
            box = result.boxes.xyxyn.detach().cpu().numpy().squeeze()

            # Convert from normalized coordinates directly to absolute pixel coordinates in the whole image
            box_abs = box.copy() * np.array([self.work_area_image.shape[1],
                                             self.work_area_image.shape[0],
                                             self.work_area_image.shape[1],
                                             self.work_area_image.shape[0]])

            # Add working area offset to get coordinates in the whole image
            box_abs[0] += working_area_top_left.x()
            box_abs[1] += working_area_top_left.y()
            box_abs[2] += working_area_top_left.x()
            box_abs[3] += working_area_top_left.y()

            # Check box area relative to **work area view** area
            box_area = (box_abs[2] - box_abs[0]) * (box_abs[3] - box_abs[1])

            # self.main_window.get_area_thresh_min()
            if box_area < self.main_window.get_area_thresh_min() * image_area:
                continue

            if box_area > self.main_window.get_area_thresh_max() * image_area:
                continue

            if self.see_anything_dialog.task == "segment":
                # Use polygons from result.masks.data.xyn (list of polygons, each Nx2, normalized to crop)
                polygon = result.masks.xyn[0]  # np.array of polygons, each as Nx2 array

                # Convert normalized polygon points directly to whole image coordinates
                polygon[:, 0] = polygon[:, 0] * self.work_area_image.shape[1] + working_area_top_left.x()
                polygon[:, 1] = polygon[:, 1] * self.work_area_image.shape[0] + working_area_top_left.y()

                polygon = simplify_polygon(polygon, 0.1)

                # Create the polygon annotation and add it to self.annotations
                self.create_polygon_annotation(polygon, confidence)
            else:
                # Create the rectangle annotation and add it to self.annotations
                self.create_rectangle_annotation(box_abs, confidence)

        self.annotation_window.scene.update()

        # Make cursor normal
        QApplication.restoreOverrideCursor()

    def create_rectangle_annotation(self, box, confidence):
        """
        Create rectangle annotations based on the given box coordinates.

        Args:
            box (np.ndarray): The bounding box coordinates.
            confidence (float): The confidence score for the annotation.
        """
        if len(box):
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
                                             self.annotation_window.main_window.get_transparency_value())

            # Update the confidence score of annotation
            annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})
            
            # Ensure the annotation is added to the scene after creation (but not saved yet)
            annotation.create_graphics_item(self.annotation_window.scene)
            
            # Animate the annotation
            annotation.animate(force=True)
            
            self.annotations.append(annotation)
            
    def update_transparency(self, value):
        """
        Update the transparency of all unconfirmed annotations in this tool.
        """
        for annotation in self.annotations:
            annotation.update_transparency(value)
        self.annotation_window.scene.update()

    def create_polygon_annotation(self, points, confidence):
        """
        Create polygon annotations based on the given points.

        Args:
            points (np.ndarray): The polygon points.
            confidence (float): The confidence score for the annotation.
        """
        if len(points) > 3:
            # Convert to QPointF
            points = [QPointF(point[0], point[1]) for point in points]
            # Create the annotation
            annotation = PolygonAnnotation(points,
                                           self.annotation_window.selected_label.short_label_code,
                                           self.annotation_window.selected_label.long_label_code,
                                           self.annotation_window.selected_label.color,
                                           self.annotation_window.current_image_path,
                                           self.annotation_window.selected_label.id,
                                           self.annotation_window.main_window.get_transparency_value())

            # Update the confidence score of annotation
            annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})
            
            # Ensure the annotation is added to the scene after creation (but not saved yet)
            annotation.create_graphics_item(self.annotation_window.scene)
            
            # Animate the annotation
            annotation.animate(force=True)
            
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
            
        for annotation in self.annotations:
            # Deanimate the annotation
            annotation.deanimate()
            # Create cropped image if not already done
            if not annotation.cropped_image and self.annotation_window.rasterio_image:
                annotation.create_cropped_image(self.annotation_window.rasterio_image)
            
            # Add the annotation using the add_annotation_from_tool method
            self.annotation_window.add_annotation_from_tool(annotation)

            # Update progress bar
            progress_bar.update_progress()
            
        # Update the scene to reflect deanimation
        self.annotation_window.scene.update()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
        progress_bar = None

        # Clear all rectangles explicitly before clearing the working area
        self.clear_all_rectangles()

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

        # Update the class mapping for the results
        results.names = {0: class_mapping[0].short_label_code}

        # Process the results with the SAM predictor using the new
        results = self.see_anything_dialog.sam_dialog.predict_from_results([results], self.image_path)

        # Get SAM resizing dimensions
        original_h, original_w = self.work_area_image.shape[:2]
        resized_h, resized_w = self.see_anything_dialog.sam_dialog.resized_image.shape[:2]

        # Calculate scaling factors
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h

        # Update mask coordinates to account for resizing
        for i, mask in enumerate(results[0].masks.xy):
            if len(mask) > 0:
                # Scale coordinates back to original size
                mask[:, 0] *= scale_x
                mask[:, 1] *= scale_y
                results[0].masks.xy[i] = mask

        # Get the raster
        raster = self.main_window.image_window.raster_manager.get_raster(self.image_path)

        # Map results from working area to the original image coordinates
        results = MapResults().map_results_from_work_area(
            results,
            raster,
            self.working_area,
            map_masks=True
        )

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
        Clear all *unconfirmed* annotations created by this tool from the scene.
        """
        for annotation in self.annotations:
            # Stop animation first
            annotation.deanimate()  # Deanimate the annotation before removing
            annotation.delete()  # Let the annotation handle all graphics cleanup
            annotation = None

        self.annotations = []
        self.annotation_window.scene.update()

    def clear_rectangle_graphics(self):
        """
        Clear rectangle graphics from the scene but keep the data.
        """
        # Remove all rectangle graphics from scene
        for rect_item in self.rectangle_items:
            if rect_item in self.annotation_window.scene.items():
                # Ensure any child items are removed first (like borders or handles)
                child_items = rect_item.childItems()
                for child in child_items:
                    self.annotation_window.scene.removeItem(child)

                # Remove the rectangle item itself
                self.annotation_window.scene.removeItem(rect_item)
            rect_item = None  # Explicitly dereference

        # Clear the rectangle graphics if one is being drawn
        if self.current_rect_graphics:
            if self.current_rect_graphics in self.annotation_window.scene.items():
                # Remove any child items first
                child_items = self.current_rect_graphics.childItems()
                for child in child_items:
                    self.annotation_window.scene.removeItem(child)

                self.annotation_window.scene.removeItem(self.current_rect_graphics)
            self.current_rect_graphics = None

        # Reset the graphics list
        self.rectangle_items = []

        # Force a full scene update and repaint
        self.annotation_window.scene.update()
        self.annotation_window.viewport().update()

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
        Clear all *user-drawn* rectangle graphics and data.
        """
        self.clear_rectangle_graphics()  # Clears items in self.rectangle_items and self.current_rect_graphics
        self.clear_rectangle_data()      # Clears self.rectangles list and drawing state

    def cancel_working_area(self):
        """
        Cancel the working area and clean up all associated resources.
        """
        if self.working_area:
            # Properly remove the working area using its method
            self.working_area.remove_from_scene()
            self.working_area = None

        if self.shadow_area:
            self.annotation_window.scene.removeItem(self.shadow_area)
            self.shadow_area = None

        self.image_path = None
        self.original_image = None
        self.work_area_image = None

        # Clear all rectangles when canceling the working area
        self.clear_all_rectangles()
        self.rectangles_processed = False

        self.annotations = []
        self.results = None

        # Force update to ensure graphics are removed visually
        self.annotation_window.scene.update()

