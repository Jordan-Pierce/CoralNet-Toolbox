import warnings

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem, QApplication, QInputDialog

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtAnnotation import RenderMode
from coralnet_toolbox.QtActions import MaskEditAction

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import CombineResults
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.Common import get_area_mode
from coralnet_toolbox.Common import raster_metrics
from coralnet_toolbox.Common import resolve_area_bounds_px

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.WorkArea import WorkArea

from coralnet_toolbox.SeeAnything.QtDeployPredictor import DEFAULT_OUTPUT_TYPE

from coralnet_toolbox.utilities import work_area_to_numpy_bgr

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.default_cursor = Qt.ArrowCursor  
        self.annotation_graphics = None

        self.work_area_image = None
        self.rectangles = []       
        self.rectangle_items = []  

        self.working_area = None

        self.image_path = None
        self.original_width = None
        self.original_height = None

        # Rectangle drawing attributes
        self.start_point = None
        self.end_point = None
        self.top_left = None
        self.bottom_right = None
        self.drawing_rectangle = False
        self.current_rect_graphics = None  
        self.rectangles_processed = False  

        # Add state variables for custom working area creation
        self.creating_working_area = False
        self.working_area_start = None
        self.working_area_temp_graphics = None
        
        # Add hover position tracking
        self.hover_pos = None

        self.annotations = []
        self.results = None

        # Output settings - synced from the dialog, as the SAM tool does
        self.output_type = DEFAULT_OUTPUT_TYPE

        # Text prompt (Ctrl+T) standing in for drawn boxes, or None
        self.text_prompt = None

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.see_anything_dialog = self.main_window.see_anything_deploy_predictor_dialog
        # Sync settings from dialog when the tool is activated
        self.sync_settings_from_dialog()
        self.report_state()

    def sync_settings_from_dialog(self):
        """Copy the output type from the dialog to this tool."""
        if self.see_anything_dialog:
            self.output_type = self.see_anything_dialog.get_output_type()

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
        # Cancel working area creation if in progress
        self.cancel_working_area_creation()
        # Clear detection data
        self.results = None
        self.text_prompt = None
        if self.see_anything_dialog is not None:
            self.see_anything_dialog.set_text_prompt(None)

        # If output type was Mask, unrasterize annotations to remove lock
        # protection, exactly as the SAM tool does.
        if self.output_type == "Mask":
            self.annotation_window.unrasterize_annotations()

        # Update the viewport
        self.annotation_window.scene.update()

        # Call parent deactivate to ensure crosshair is properly cleared
        super().deactivate()

    def leave(self):
        """Pointer left the window -- drop the rectangle being dragged out.

        The work area, the rectangles already placed and any unconfirmed
        predictions are deliberately kept: only the half-drawn rectangle, which
        tracks the cursor and would otherwise freeze mid-drag, is discarded.
        """
        if self.drawing_rectangle:
            self.cancel_rectangle_drawing()
        self.hover_pos = None
        super().leave()

    def report_state(self):
        """Say what Space and Backspace will do from here.

        Space means several different things depending on the state -- create
        the work area, predict, or confirm -- and nothing used to say which.
        Where a third option exists (drawing more reference boxes to widen the
        same prediction) it is named too, because nothing on screen suggests it.
        """
        if not self.active:
            return

        if self.creating_working_area:
            message = "Space: finish the work area  |  Backspace: cancel it"
        elif not self.working_area:
            message = "Space: use the current view as the work area, or drag one out"
        elif self.drawing_rectangle:
            message = "Click to finish the box  |  Backspace: cancel it"
        elif self.rectangles and not self.rectangles_processed:
            count = len(self.rectangles)
            message = (f"Space: predict from {count} reference box{'es' if count != 1 else ''}"
                       "  |  Draw another box to add an example"
                       "  |  Backspace: clear them")
        elif self.annotations:
            count = len(self.annotations)
            confirm = "refine with SAM and confirm" if self._sam_enabled() else "confirm"
            message = (f"Space: {confirm} {count} detection{'s' if count != 1 else ''}"
                       "  |  Draw another box to find more"
                       "  |  Backspace: discard them")
        elif self.text_prompt:
            message = (f"Space: predict from text prompt '{self.text_prompt}'"
                       "  |  Ctrl+T: change it  |  Or draw a box instead")
        else:
            message = ("Draw a box around an example, or Ctrl+T for a text prompt"
                       "  |  Space: close the work area")

        self.main_window.status_bar.showMessage(message, 6000)

    def _sam_enabled(self):
        """True when the dialog is set to refine detections with SAM."""
        dialog = self.see_anything_dialog
        return dialog is not None and dialog.use_sam_dropdown.currentText() == "True"

    def _sam_dialog(self):
        """The SAM predictor dialog, if one is loaded and SAM refinement is on."""
        if not self._sam_enabled():
            return None
        sam_dialog = getattr(self.see_anything_dialog, 'sam_dialog', None)
        if sam_dialog is None or getattr(sam_dialog, 'loaded_model', None) is None:
            return None
        return sam_dialog

    def _ensure_sam_image(self):
        """Make sure the shared SAM predictor holds this work area's features.

        The predictor is shared: the SAM tool and batch inference encode their
        own images into it. Encoding here when the work area is created runs the
        ViT pass under the wait cursor rather than at confirm time, and calling
        it again before refinement re-encodes if something else took the
        predictor in the meantime, instead of prompting against another image's
        features.

        Returns:
            bool: True if SAM can be prompted against this work area.
        """
        sam_dialog = self._sam_dialog()
        if sam_dialog is None or self.work_area_image is None:
            return False
        if sam_dialog.has_image(self.work_area_image):
            return True
        sam_dialog.set_image(self.work_area_image, self.image_path)
        return sam_dialog.has_image(self.work_area_image)

    def set_working_area(self):
        """
        Set the working area for the tool using the WorkArea class.
        """
        self.annotation_window.setCursor(Qt.WaitCursor)

        # Cancel the current working area if it exists
        self.cancel_working_area()

        # Original image (grab current from the annotation window)
        self.image_path = self.annotation_window.current_image_path
        self.original_width, self.original_height = self.annotation_window.get_image_dimensions()

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
        # Set animation manager

        # Create and add the working area graphics
        self.working_area.create_graphics(self.annotation_window.scene, 
                                          include_shadow=True, 
                                          image_rect=self.annotation_window.get_image_rect())
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)

        # Crop the image based on the working area
        # Read just the work area from the file rather than slicing a
        # full-image array taken off the display pixmap: that decoded the whole
        # raster to use a viewport-sized piece of it. BGR because ultralytics
        # documents its numpy input as cv2-order.
        self.work_area_image = work_area_to_numpy_bgr(
            self.annotation_window.rasterio_image, self.working_area)

        # Set the image in the SeeAnything dialog, and pre-encode it for SAM
        # while the wait cursor is still up
        self.see_anything_dialog.set_image(self.work_area_image, self.image_path)
        self._ensure_sam_image()

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
        right = min(self.annotation_window.get_image_dimensions()[0], 
                    int(max(start_point.x(), end_point.x())))
        bottom = min(self.annotation_window.get_image_dimensions()[1],
                     int(max(start_point.y(), end_point.y())))
        
        # Ensure minimum size (at least 10x10 pixels)
        if right - left < 10:
            right = min(left + 10, self.annotation_window.get_image_dimensions()[0])
        if bottom - top < 10:
            bottom = min(top + 10, self.annotation_window.get_image_dimensions()[1])
            
        # Original image information
        self.image_path = self.annotation_window.current_image_path
        self.original_width, self.original_height = self.annotation_window.get_image_dimensions()
            
        # Create the WorkArea instance
        self.working_area = WorkArea(left, top, right - left, bottom - top, self.image_path)
        # Set animation manager
        
        # Create and add the working area graphics
        self.working_area.create_graphics(self.annotation_window.scene, 
                                          include_shadow=True, 
                                          image_rect=self.annotation_window.get_image_rect())
        
        self.working_area.set_remove_button_visibility(False)
        self.working_area.removed.connect(self.on_working_area_removed)
        
        # Crop the image based on the working area
        # Read just the work area from the file rather than slicing a
        # full-image array taken off the display pixmap: that decoded the whole
        # raster to use a viewport-sized piece of it. BGR because ultralytics
        # documents its numpy input as cv2-order.
        self.work_area_image = work_area_to_numpy_bgr(
            self.annotation_window.rasterio_image, self.working_area)
        
        # Set the image in the SeeAnything dialog, and pre-encode it for SAM
        # while the wait cursor is still up
        self.see_anything_dialog.set_image(self.work_area_image, self.image_path)
        self._ensure_sam_image()
        
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

            # Style the rectangle
            pen = QPen(QColor(color))
            pen.setCosmetic(True)
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
            self.annotation_window.main_window.status_bar.showMessage(
                "A label must be selected before adding an annotation.", 4000)
            return None

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
        self.report_state()

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handles the mouse move event.

        Args:
            event (QMouseEvent): The mouse move event.
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
        if event.key() == Qt.Key_T and event.modifiers() == Qt.ControlModifier:
            self.prompt_for_text()
            return

        if event.key() == Qt.Key_Space:
            # If creating working area, confirm it
            if self.creating_working_area and self.working_area_start and self.hover_pos:
                self.set_custom_working_area(self.working_area_start, self.hover_pos)
                self.cancel_working_area_creation()
                self.report_state()
                return

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

            # No boxes drawn, but a text prompt is standing in for them
            elif not self.annotations and self.text_prompt:
                self.create_annotations_from_text()

            else:
                # If there's a working area but no new user rectangles,
                # or if rectangles have been processed, confirm the accumulated annotations.
                if self.annotations:  # Check if there are any annotations to confirm/process
                    if self._sam_enabled():
                        self.apply_sam_model()
                    else:
                        # Confirm the annotations accumulated so far
                        self.confirm_annotations()
                # Cancel the working area if no annotations were generated or after confirmation/SAM
                self.cancel_working_area()

        elif event.key() == Qt.Key_Backspace:
            # If creating working area, cancel it
            if self.creating_working_area:
                self.cancel_working_area_creation()
                self.report_state()
                return

            # Cancel current rectangle being drawn
            if self.drawing_rectangle:
                self.cancel_rectangle_drawing()
            # If we have a working area and accumulated annotations, clear them
            elif self.working_area and len(self.annotations) > 0:
                self.clear_annotations()  # Clears unconfirmed annotations
            # If not drawing and no annotations to clear, clear any pending user-drawn rectangles
            else:
                self.clear_all_rectangles()  # Clears user input rectangles

        self.annotation_window.scene.update()
        self.report_state()

    def cancel_rectangle_drawing(self):
        """Discard the rectangle currently being dragged out."""
        self.drawing_rectangle = False
        if self.current_rect_graphics:
            if self.current_rect_graphics.scene() is not None:
                self.annotation_window.scene.removeItem(self.current_rect_graphics)
            self.current_rect_graphics = None
        self.start_point = None
        self.end_point = None
        self.annotation_window.scene.update()

    def prompt_for_text(self):
        """Ask for a text prompt (Ctrl+T) to use in place of drawn boxes.

        Ultralytics turns the phrase into a class embedding via
        `YOLOE.get_text_pe`, so no reference boxes are needed at all. An empty
        entry clears the prompt and returns to box prompting.
        """
        if self.see_anything_dialog is None or self.see_anything_dialog.loaded_model is None:
            self.main_window.status_bar.showMessage(
                "Load a See Anything model before using a text prompt.", 4000)
            return

        text, accepted = QInputDialog.getText(
            self.annotation_window,
            "Text Prompt",
            "Describe what to find (leave empty to go back to box prompts):",
            text=self.text_prompt or "")
        if not accepted:
            return

        text = text.strip()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            applied = self.see_anything_dialog.set_text_prompt(text)
        except Exception as e:
            applied = False
            self.main_window.status_bar.showMessage(f"Could not set text prompt: {e}", 5000)
        finally:
            QApplication.restoreOverrideCursor()

        self.text_prompt = text if (text and applied) else None
        if self.text_prompt:
            self.main_window.status_bar.showMessage(
                f"Text prompt set to '{self.text_prompt}'. Press Space to predict.", 5000)
        elif not text:
            self.main_window.status_bar.showMessage("Text prompt cleared.", 3000)
        self.report_state()

    def create_annotations_from_rectangles(self):
        """
        Create annotations based on the user-drawn rectangles.
        """
        if not self.annotation_window.active_image:
            return None

        if not self.annotation_window.active_image:
            return None

        if not self.working_area:
            return None

        if len(self.rectangles) == 0:  # Check specifically for user-drawn rectangles
            return None

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        masks = None
        # Create masks from the rectangles (these are not polygons)
        if self.see_anything_dialog.get_task() == 'segment':
            masks = []
            for r in self.rectangles:
                x1, y1, x2, y2 = r
                masks.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))

        # Predict from prompts, providing masks if the task is segmentation.
        # The rectangles are in work-area pixels and are handed over unscaled:
        # ultralytics rasterizes them against its own letterbox (see
        # DeployPredictorDialog.build_prompts).
        results = self.see_anything_dialog.predict_from_prompts(self.rectangles, masks=masks)

        if not results:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            self.main_window.status_bar.showMessage(
                "See Anything returned nothing for those reference boxes.", 5000)
            return None

        self._build_annotations_from_results(results)

    def create_annotations_from_text(self):
        """Predict from the current text prompt (Ctrl+T) instead of drawn boxes."""
        if not self.annotation_window.active_image or not self.working_area:
            return None
        if not self.text_prompt:
            return None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        results = self.see_anything_dialog.predict_from_text()

        if not results:
            QApplication.restoreOverrideCursor()
            self.main_window.status_bar.showMessage(
                f"See Anything found nothing matching '{self.text_prompt}'.", 5000)
            return None

        self._build_annotations_from_results(results)

    def _build_annotations_from_results(self, results):
        """Filter a prediction and turn what survives into preview annotations.

        Shared by the box-prompt and text-prompt paths. Restores the cursor and
        reports how many detections were kept and how many the confidence and
        area thresholds dropped -- an empty result used to be silent, which read
        as a crash.

        Args:
            results (list): Ultralytics Results, as returned by the dialog.
        """
        # Move the points back to the original image space
        working_area_top_left = self.working_area.rect.topLeft()

        # Get the first result from the list
        results = results[0]

        # Create a results processor to merge and filter results
        results_processor = ResultsProcessor(self.main_window, {})
        # Merge
        if self.results:
            results = CombineResults().combine_results([self.results, results])

        # The model ran on the work-area crop, a numpy array, so Ultralytics
        # named the result after the array ("image0.jpg"). The area filter
        # resolves its bounds by looking that path up in the raster manager, and
        # a name no raster answers to leaves it with nothing to measure against:
        # a real-world bound went unresolved and filtered nothing at all, while
        # an image-share bound fell back to the fraction of the *crop*, which is
        # larger than the fraction of the image by the ratio between them -- 25x
        # for an 800x600 work area on a 4000x3000 raster, enough to reject
        # objects the same threshold keeps when the work area is not used.
        # Naming the raster is what makes both bounds whole-image.
        results.path = self.image_path

        # Filter
        self.results = results_processor.apply_filters_to_results(results)

        # Resolved through the same helper ResultsProcessor uses, so both filters
        # agree. Bounds are relative to the WHOLE image: scaling by the work-area
        # crop made the same threshold mean a different real size depending on how
        # far the view happened to be zoomed. `None` means the threshold cannot be
        # judged for this raster (a real-world bound with no scale), in which case
        # every detection is kept rather than silently dropped.
        try:
            _raster = self.main_window.image_window.raster_manager.get_raster(self.image_path)
        except Exception:
            _raster = None

        image_area, m2_per_px = raster_metrics(_raster)
        if not image_area:
            image_area = float(self.original_width or 0) * float(self.original_height or 0)
        if not image_area:
            image_area = self.work_area_image.shape[0] * self.work_area_image.shape[1]

        area_bounds = resolve_area_bounds_px(
            self.main_window.get_area_thresh_min(),
            self.main_window.get_area_thresh_max(),
            get_area_mode(self.main_window),
            image_area, m2_per_px)

        # Clear previous annotations if any
        self.clear_annotations()

        # Counted so the result can be reported rather than left to guess at
        dropped_confidence = 0
        dropped_area = 0

        # Process results based on the task type (creates polygons or rectangle annotations)
        if self.see_anything_dialog.get_task() == "segment":
            if self.results.masks:
                for i, polygon in enumerate(self.results.masks.xyn):
                    confidence = self.results.boxes.conf[i].item()
                    if confidence < self.main_window.get_uncertainty_thresh():
                        dropped_confidence += 1
                        continue

                    # Get absolute bounding box for area check (relative to work area)
                    box_work_area = self.results.boxes.xyxy[i].detach().cpu().numpy()
                    box_area = (box_work_area[2] - box_work_area[0]) * (box_work_area[3] - box_work_area[1])

                    # Area filtering
                    if area_bounds and not (area_bounds[0] <= box_area <= area_bounds[1]):
                        dropped_area += 1
                        continue

                    # Convert normalized polygon points to absolute coordinates in the whole image
                    polygon_abs = polygon.copy()
                    polygon_abs[:, 0] = polygon_abs[:, 0] * self.work_area_image.shape[1] + working_area_top_left.x()
                    polygon_abs[:, 1] = polygon_abs[:, 1] * self.work_area_image.shape[0] + working_area_top_left.y()

                    # No automatic simplification - preserve full precision
                    self.create_polygon_annotation(polygon_abs, confidence)

        else:  # Task is 'detect'
            if self.results.boxes:
                for i, box_norm in enumerate(self.results.boxes.xyxyn):
                    confidence = self.results.boxes.conf[i].item()
                    if confidence < self.main_window.get_uncertainty_thresh():
                        dropped_confidence += 1
                        continue

                    # Convert normalized box to absolute coordinates in the work area
                    box_abs_work_area = box_norm.detach().cpu().numpy() * np.array(
                        [self.work_area_image.shape[1], self.work_area_image.shape[0],
                         self.work_area_image.shape[1], self.work_area_image.shape[0]])
                    # Calculate the area of the bounding box
                    box_area = (box_abs_work_area[2] - box_abs_work_area[0]) * \
                               (box_abs_work_area[3] - box_abs_work_area[1])

                    # Area filtering
                    if area_bounds and not (area_bounds[0] <= box_area <= area_bounds[1]):
                        dropped_area += 1
                        continue

                    # Add working area offset to get coordinates in the whole image
                    box_abs_full = box_abs_work_area.copy()
                    box_abs_full[0] += working_area_top_left.x()
                    box_abs_full[1] += working_area_top_left.y()
                    box_abs_full[2] += working_area_top_left.x()
                    box_abs_full[3] += working_area_top_left.y()
                    self.create_rectangle_annotation(box_abs_full, confidence)

        self.annotation_window.scene.update()

        # Make cursor normal
        QApplication.restoreOverrideCursor()

        self.report_detection_counts(dropped_confidence, dropped_area)
        return len(self.annotations)

    def report_detection_counts(self, dropped_confidence, dropped_area):
        """Say how many detections were kept, and what the thresholds removed."""
        kept = len(self.annotations)
        message = f"{kept} detection{'s' if kept != 1 else ''} found"

        dropped = []
        if dropped_confidence:
            dropped.append(f"{dropped_confidence} below the uncertainty threshold")
        if dropped_area:
            dropped.append(f"{dropped_area} outside the area thresholds")
        if dropped:
            message += " (" + ", ".join(dropped) + " discarded)"

        if kept:
            confirm = "refine with SAM and confirm" if self._sam_enabled() else "confirm"
            message += (f". Space: {confirm}"
                        "  |  Draw another box to find more"
                        "  |  Backspace: discard.")
        elif dropped:
            message += ". Try lowering the thresholds, or draw another example box."
        else:
            message += ". Try another example box, or a different work area."

        self.main_window.status_bar.showMessage(message, 8000)

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
                                             self.annotation_window.selected_label,
                                             self.annotation_window.current_image_path,
                                             transparency=self.annotation_window.main_window.get_transparency_value(),
                                             show_confidence=False)

            # Update the confidence score of annotation
            annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})
            # Mark selected so it gets the dashed selected-state pen
            annotation.is_selected = True
            annotation.render_mode = RenderMode.FULL
            # Ensure the annotation is added to the scene after creation (but not saved yet)
            # Force hydrate so these tool-generated previews behave like normal Qt objects
            annotation.create_graphics_item(self.annotation_window.scene, force_hydrate=True)
            
            self.annotations.append(annotation)
            
    def refresh_label_preview(self):
        """Recolor the prompt rectangles and unconfirmed predictions for the new label.

        Every rectangle and every not-yet-confirmed annotation was drawn with
        the label that was selected at the time, so all of them move to the new
        one. Each annotation's confidence is re-keyed onto the new label so the
        score survives the switch instead of being dropped by update_label().
        """
        label = self.annotation_window.selected_label
        if label is None:
            return

        pen = QPen(QColor(label.color))
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)

        for rect_item in self.rectangle_items:
            rect_item.setPen(pen)
        if self.current_rect_graphics is not None:
            self.current_rect_graphics.setPen(pen)

        for annotation in self.annotations:
            confidence = None
            if annotation.machine_confidence:
                confidence = max(annotation.machine_confidence.values())
            annotation.update_label(label)
            if confidence is not None:
                annotation.update_machine_confidence({label: confidence})

        self.annotation_window.scene.update()

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
                                           self.annotation_window.selected_label,
                                           self.annotation_window.current_image_path,
                                           transparency=self.annotation_window.main_window.get_transparency_value(),
                                           show_confidence=False)

            # Update the confidence score of annotation
            annotation.update_machine_confidence({self.annotation_window.selected_label: confidence})
            # Mark selected so it gets the dashed selected-state pen
            annotation.is_selected = True
            annotation.render_mode = RenderMode.FULL
            # Ensure the annotation is added to the scene after creation (but not saved yet)
            # Force hydrate so these tool-generated previews behave like normal Qt objects
            annotation.create_graphics_item(self.annotation_window.scene, force_hydrate=True)
            
            self.annotations.append(annotation)

    def confirm_annotations(self, crop_annotations=False):
        """
        Confirm the annotations and clear the working area.
        """
        # Sync the latest output type from the dialog before committing
        self.sync_settings_from_dialog()

        if self.output_type == "Mask":
            self.confirm_annotations_as_mask()
            return

        # Confirm annotations, using a bulk path for large batches to avoid O(N^2) UI work.
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            total = len(self.annotations)
            if total == 0:
                return

            # Strip the preview-only selected state and graphics so annotations
            # enter the data model as normal unselected items (no dashed pen).
            for annotation in self.annotations:
                annotation.is_selected = False
                annotation.render_mode = RenderMode.PHANTOM
                if annotation.graphics_item_group is not None:
                    try:
                        if annotation.graphics_item_group.scene():
                            annotation.graphics_item_group.scene().removeItem(
                                annotation.graphics_item_group
                            )
                    except RuntimeError:
                        pass
                annotation.graphics_item_group = None
                annotation.graphics_item = None
                annotation.center_graphics_item = None
                annotation.bounding_box_graphics_item = None
                annotation.tag_item = None
                annotation.dimension_tag_item = None

            # Threshold for switching to the optimized, bulk path.
            BATCH_THRESHOLD = 10

            # Fast path for many annotations: batch-crop and add them in one operation.
            if total >= BATCH_THRESHOLD:
    
                # Batch-crop all annotations (shows its own progress UI) if requested.
                if crop_annotations:
                    self.annotation_window.crop_annotations(image_path=self.image_path,
                                                            annotations=self.annotations,
                                                            verbose=True)
                
                # Add them using the optimized bulk method which updates the UI once.
                self.annotation_window.add_annotations(self.annotations)

            else:
                # Small-number path: keep per-item feedback so UX remains immediate.
                progress_bar = ProgressBar(self.annotation_window, "Confirming Annotations")
                progress_bar.show()
                progress_bar.start_progress(total)

                for idx, annotation in enumerate(self.annotations):
                    # Allow canceling from the progress dialog
                    if progress_bar.wasCanceled():
                        break

                    
                    if crop_annotations and not annotation.cropped_image and self.annotation_window.rasterio_image:
                        annotation.create_cropped_image(self.annotation_window.rasterio_image)
                    
                    self.annotation_window.add_annotation_from_tool(annotation)

                    # Update progress bar every ~10% to avoid excessive UI updates
                    if total > 10:
                        if idx % (total // 10) == 0:
                            progress_bar.update_progress_percentage((idx / total) * 100)
                    else:
                        progress_bar.update_progress_percentage((idx / total) * 100)

                # Update the scene to reflect deanimation
                self.annotation_window.scene.update()

                # Tear down progress UI
                progress_bar.finish_progress()
                progress_bar.stop_progress()
                progress_bar.close()
   
        finally:
            # Ensure cleanup happens regardless of the path taken
            QApplication.restoreOverrideCursor()

            # Clear all rectangles explicitly before clearing the working area
            self.clear_all_rectangles()
            self.cancel_working_area()

            # Clear the annotations list and results cache
            self.annotations = []
            self.results = None

    def _mask_target(self):
        """The raster mask annotation and class ID to paint into, or (None, None).

        Mask output writes into the image's existing MaskAnnotation rather than
        creating vector annotations, the same way the SAM tool does.
        """
        mask_annotation = self.annotation_window.current_mask_annotation
        label = self.annotation_window.selected_label
        if mask_annotation is None or label is None:
            return None, None
        class_id = mask_annotation.label_id_to_class_id_map.get(label.id)
        if class_id is None:
            return None, None
        return mask_annotation, class_id

    def _commit_prediction_mask(self, prediction_mask, mask_annotation, painted):
        """Push a filled prediction mask onto the annotation and the undo stack."""
        if not painted:
            self.main_window.status_bar.showMessage(
                "Nothing to paint into the mask.", 4000)
            return

        history_action = MaskEditAction(mask_annotation, description="See Anything prediction")
        mask_annotation.update_mask_with_prediction_mask(
            prediction_mask,
            history_action=history_action,
        )
        if not history_action.is_empty():
            self.annotation_window.action_stack.push(history_action)

        self.main_window.status_bar.showMessage(
            f"Painted {painted} region{'s' if painted != 1 else ''} into the mask.", 5000)

    def confirm_annotations_as_mask(self):
        """Commit the preview annotations by painting them into the raster mask.

        The previews are polygons and rectangles because those draw cheaply;
        Mask output rasterizes them at confirm time, which is the same split the
        SAM tool uses between its preview and its committed result.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mask_annotation, class_id = self._mask_target()
            if mask_annotation is None:
                self.main_window.status_bar.showMessage(
                    "No raster mask on this image to paint into; "
                    "switch the output type or create a mask first.", 6000)
                return

            prediction_mask = np.zeros_like(mask_annotation.mask_data)
            painted = 0

            for annotation in self.annotations:
                polygon = self._annotation_to_polygon(annotation)
                if polygon is None:
                    continue
                cv2.fillPoly(prediction_mask, [polygon], int(class_id))
                painted += 1

            self._commit_prediction_mask(prediction_mask, mask_annotation, painted)
        finally:
            QApplication.restoreOverrideCursor()
            self.clear_annotations()
            self.clear_all_rectangles()
            self.cancel_working_area()
            self.annotations = []
            self.results = None

    @staticmethod
    def _annotation_to_polygon(annotation):
        """Return an annotation's outline as an int32 (N, 2) array for cv2.fillPoly."""
        if isinstance(annotation, PolygonAnnotation):
            points = [[p.x(), p.y()] for p in annotation.points]
        elif isinstance(annotation, RectangleAnnotation):
            top_left, bottom_right = annotation.top_left, annotation.bottom_right
            points = [[top_left.x(), top_left.y()],
                      [bottom_right.x(), top_left.y()],
                      [bottom_right.x(), bottom_right.y()],
                      [top_left.x(), bottom_right.y()]]
        else:
            return None

        if len(points) < 3:
            return None
        return np.round(np.array(points, dtype=np.float32)).astype(np.int32)

    def paint_results_into_mask(self, results_list):
        """Paint SAM-refined masks straight into the raster mask annotation.

        Used for Mask output when SAM refinement is on: the masks are already
        in whole-image coordinates by this point, so they are rasterized rather
        than turned into polygons and back.

        Args:
            results_list: Ultralytics Results (or a list of them) already mapped
                out of the work area.
        """
        mask_annotation, class_id = self._mask_target()
        if mask_annotation is None:
            self.main_window.status_bar.showMessage(
                "No raster mask on this image to paint into; "
                "switch the output type or create a mask first.", 6000)
            return

        if not isinstance(results_list, list):
            results_list = [results_list]

        prediction_mask = np.zeros_like(mask_annotation.mask_data)
        painted = 0

        for result in results_list:
            if result is None or result.masks is None:
                continue
            for polygon in result.masks.xy:
                if polygon is None or len(polygon) < 3:
                    continue
                cv2.fillPoly(prediction_mask,
                             [np.round(polygon).astype(np.int32)],
                             int(class_id))
                painted += 1

        self._commit_prediction_mask(prediction_mask, mask_annotation, painted)

    def apply_sam_model(self):
        """Uses the Results with SAM predictor to create polygons instead of confirming the
        ones created by the SeeAnything predictor."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Pick up the output type chosen in the dialog before committing
        self.sync_settings_from_dialog()

        # Create a class mapping dictionary
        class_mapping = {0: self.annotation_window.selected_label}

        # Create a results processor
        results_processor = ResultsProcessor(
            self.main_window,
            class_mapping
        )

        # OPTIMIZATION: Avoid copy.deepcopy() on PyTorch tensors.
        # Since self.results is discarded during cancel_working_area(),
        # we can safely modify the class dictionary in-place.
        results_to_process = self.results
        results_to_process.names = {0: class_mapping[0].short_label_code}

        # The boxes are already in work-area pixels and orig_img is already the
        # work-area crop: the prediction ran on that crop directly, letterboxed
        # by ultralytics, which inverts its own transform in scale_boxes.
        #
        # There used to be a block here rebinding boxes and orig_img through
        # xyxyn. It existed because this dialog pre-resized the crop with a
        # separate scale per axis, which ultralytics then unwound as if it were
        # a uniform letterbox -- so masks drifted further off the further they
        # sat from the centre of the work area. The resize is gone (see
        # DeployPredictorDialog.set_image), and so is the correction for it.

        # Re-encode if another tool or a batch run took the shared SAM
        # predictor since the work area was created.
        self._ensure_sam_image()

        # Process the results with the SAM predictor
        processed_results = self.see_anything_dialog.sam_dialog.predict_from_results([results_to_process],
                                                                                     self.image_path)

        # Get the raster
        raster = self.main_window.image_window.raster_manager.get_raster(self.image_path)

        # Map results from working area to the original image coordinates
        final_results = MapResults().map_results_from_work_area(
            processed_results,
            raster,
            self.working_area,
            map_masks=True,
            boundary_tolerance=self.see_anything_dialog.thresholds_widget.get_boundary_tolerance(),
        )

        # map_results_from_work_area resets result.path to raster.image_path;
        # for a virtual video frame (video.mp4::frame_N) that drops the frame
        # suffix and annotations get keyed to the wrong path. Restore it.
        if isinstance(final_results, list):
            for r in final_results:
                if r is not None:
                    r.path = self.image_path
        elif final_results is not None:
            final_results.path = self.image_path

        # Process the results, either as vector annotations or straight into
        # the image's raster mask
        if self.output_type == "Mask":
            self.paint_results_into_mask(final_results)
        else:
            results_processor.process_segmentation_results(final_results)

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

        self.image_path = None
        self.work_area_image = None

        # Clear all rectangles when canceling the working area
        self.clear_all_rectangles()
        self.rectangles_processed = False

        self.annotations = []
        self.results = None

        # Force update to ensure graphics are removed visually
        self.annotation_window.scene.update()
