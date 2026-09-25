import os
import warnings

import cv2
import numpy as np
import torch

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem, QApplication

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtAnnotation import RenderMode
from coralnet_toolbox.QtActions import MaskEditAction

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.Common import get_area_mode
from coralnet_toolbox.Common import raster_metrics
from coralnet_toolbox.Common import resolve_area_bounds_px

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.WorkArea import WorkArea

from coralnet_toolbox.SeeAnything.QtDeployPredictor import DEFAULT_OUTPUT_TYPE
from coralnet_toolbox.SeeAnything.QtPromptAlignment import (TextPromptDialog,
                                                            ensure_vocabulary,
                                                            project_label_names)
from coralnet_toolbox.SeeAnything.PromptSession import (KIND_BOXES,
                                                        KIND_DETECTION,
                                                        collapse_to_one_class)

from coralnet_toolbox.utilities import work_area_to_numpy_bgr

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Predictions are fetched down to this confidence and filtered for display, so
# Ctrl+wheel can lower the threshold without a new prediction. Lowering it past
# the floor fetches again.
PREDICT_CONFIDENCE_FLOOR = 0.05

# Ctrl+wheel moves the threshold this far per notch; Ctrl+Shift+wheel, the finer step.
THRESHOLD_STEP = 0.05
THRESHOLD_FINE_STEP = 0.01
# At or below this, wheel steps drop to THRESHOLD_FINE_STEP even without Shift, so
# values near zero stay reachable one notch at a time instead of overshooting.
THRESHOLD_FINE_CUTOFF = 0.10

# A right-button press and release closer than this (in screen pixels) is a click.
# Right-drag pans and Ctrl+right-drag rotates the canvas, and the annotation
# window hands the press to the tool before the canvas starts either, so acting on
# the press would remove a detection every time a pan began over one.
CLICK_SLOP_PX = 4

# A removed detection stays removed across re-runs if a new one overlaps it this much.
DROP_MATCH_IOU = 0.7


class PreviewDetection:
    """One detection from the cached prediction, whether or not it is on screen.

    The preview annotation is created the first time the detection is shown, so a
    prediction fetched down to the confidence floor does not pay for drawing
    detections the threshold hides.
    """

    __slots__ = ("index", "confidence", "box_work", "box_image", "polygon_image",
                 "annotation", "dropped")

    def __init__(self, index, confidence, box_work, box_image, polygon_image=None):
        self.index = index
        self.confidence = confidence
        self.box_work = box_work
        self.box_image = box_image
        self.polygon_image = polygon_image
        self.annotation = None
        self.dropped = False


def _box_iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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

        # Text prompt (Ctrl+T) standing in for drawn boxes, or None. A mirror of
        # the phrase held in the dialog's prompt session.
        self.text_prompt = None
        # Whether the session has already been run on this work area. Mirrors
        # `rectangles_processed`: without it, a prompt that finds nothing traps
        # Space on the predict branch and the work area cannot be closed.
        self.session_processed = False
        # Set when a run came back empty, so the status bar can keep saying so
        # rather than flashing it once and reverting.
        self.session_found_nothing = False

        # The last prediction, unfiltered by confidence, and every detection in it.
        # `self.annotations` and `self.results` are the filtered view of these that
        # is on screen; Ctrl+wheel and removals re-derive the view without a model call.
        self.raw_results = None
        self.raw_floor = None
        self.previews = []
        self.dropped_area = 0
        self._applied_threshold = None
        # Image-coordinate boxes the user removed, matched onto the next prediction
        # so a removal survives adding an example.
        self.dropped_boxes = []

        # Ctrl+Shift held: removed detections are revealed so they can be restored,
        # the way the Work Area tool reveals its remove buttons.
        self.revealing_dropped = False
        self.dropped_graphics = []
        self.hover_graphics = None
        # (screen position, modifiers) of a right-button press awaiting its release
        self._right_press = None
        self._signals_connected = False

    def activate(self):
        """
        Activates the tool.
        """
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.see_anything_dialog = self.main_window.see_anything_deploy_predictor_dialog
        # Sync settings from dialog when the tool is activated
        self.sync_settings_from_dialog()
        # The session outlives the tool; pick its phrase back up
        session = self._session()
        if session is not None:
            self.text_prompt = session.text_phrase()
        self._connect_signals()
        self.report_state()

    def _connect_signals(self):
        """Follow the global threshold slider and edits made in the session panel."""
        if self._signals_connected:
            return
        try:
            self.main_window.uncertaintyChanged.connect(self.on_uncertainty_changed)
        except (AttributeError, TypeError):
            pass
        edited = getattr(self.see_anything_dialog, 'sessionEdited', None)
        if edited is not None:
            edited.connect(self.on_session_edited)
        self._signals_connected = True

    def _disconnect_signals(self):
        if not self._signals_connected:
            return
        for signal, slot in ((getattr(self.main_window, 'uncertaintyChanged', None), self.on_uncertainty_changed),
                             (getattr(self.see_anything_dialog, 'sessionEdited', None), self.on_session_edited)):
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._signals_connected = False

    def _session(self):
        """The dialog's prompt session, or None."""
        return getattr(self.see_anything_dialog, 'session', None)

    def _has_prompt(self):
        """Whether there is anything to predict from without drawing a box."""
        session = self._session()
        return bool(self.text_prompt) or bool(session is not None and session.has_positives())

    def _prompt_label(self):
        """How to name the prompt in the status bar: the phrase, when that is all it is."""
        session = self._session()
        text_only = session is None or not session.has_positives() or session.modalities() == {"text"}
        if self.text_prompt and text_only:
            return f"'{self.text_prompt}'"
        return "The prompt session"

    def _threshold(self):
        """The global confidence threshold -- the one the Generator will run with."""
        return self.main_window.get_uncertainty_thresh()

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
        # Clear detection data. The prompt session is deliberately kept: it is
        # built here to be used elsewhere -- in the Generator, or back in this
        # tool after a detour to another one. Ctrl+Shift+Backspace clears it.
        self.results = None
        self.session_processed = False
        self.session_found_nothing = False
        self._set_revealing_dropped(False)
        self._clear_hover()
        self._right_press = None
        self._disconnect_signals()

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

        session = self._session()
        session_note = ""
        if session is not None and not session.is_empty():
            session_note = f"  |  Session: {session.summary()}  (Ctrl+Shift+Backspace: clear)"

        if self.creating_working_area:
            message = "Space: finish the work area  |  Backspace: cancel it"
        elif not self.working_area:
            message = ("Space: use the current view as the work area, or left-click, move, "
                       "left-click to draw one" + session_note)
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
                       "  |  Ctrl+click: more like this"
                       "  |  Ctrl+right-click: fewer like this"
                       "  |  Ctrl+Shift+right-click: remove one"
                       f"  |  Ctrl+wheel: threshold {self._threshold():.2f}"
                       "  |  Draw another box to find more"
                       "  |  Backspace: discard them")
        elif self._has_prompt() and self.session_found_nothing:
            # The dead end that used to trap the tool: Space kept re-running the
            # same empty prediction and Backspace had nothing to clear. Both do
            # something now, and both are named.
            message = (f"{self._prompt_label()} matched nothing here"
                       f"  |  Ctrl+wheel: lower the threshold ({self._threshold():.2f})"
                       "  |  Ctrl+T: try another word"
                       "  |  Backspace or Space: close the work area")
        elif self._has_prompt() and self.session_processed:
            message = (f"{self._prompt_label()} has already run"
                       "  |  Ctrl+T: try another word"
                       "  |  Backspace or Space: close the work area")
        elif self.text_prompt:
            message = (f"Space: predict from text prompt '{self.text_prompt}'"
                       "  |  Ctrl+T: change it  |  Backspace: close the work area"
                       "  |  Or draw a box instead" + session_note)
        elif self._has_prompt():
            message = ("Space: predict from the prompt session"
                       "  |  Draw a box to add an example"
                       "  |  Ctrl+T: add a phrase"
                       "  |  Backspace: close the work area" + session_note)
        else:
            message = ("Draw a box around an example, or Ctrl+T for a text prompt"
                       "  |  Backspace or Space: close the work area")

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

        # Right button: remember the press and decide on release. The canvas pans
        # on right-drag and rotates on Ctrl+right-drag, and it sees this same press
        # right after the tool does -- only a release without movement is a click.
        if event.button() == Qt.RightButton:
            self._right_press = (event.pos(), event.modifiers())
            return

        # Ctrl+click on a detection: more like this. Anywhere else, Ctrl+click
        # draws a box as a plain click does.
        modifiers = event.modifiers()
        if (event.button() == Qt.LeftButton
                and modifiers & Qt.ControlModifier
                and not modifiers & Qt.ShiftModifier
                and self.working_area is not None
                and not self.drawing_rectangle
                and self.detection_at(scene_pos) is not None):
            self.add_positive_at(scene_pos)
            self.report_state()
            return

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

        self.update_hover(scene_pos, event.modifiers())

        self.annotation_window.scene.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Act on a right-click once it is known not to be the start of a pan.

        Ctrl+right-click: fewer like this. Ctrl+Shift+right-click: remove or
        restore one detection.
        """
        if event.button() != Qt.RightButton or self._right_press is None:
            return
        press_pos, modifiers = self._right_press
        self._right_press = None
        if not self.is_click(press_pos, event.pos()):
            return

        scene_pos = self.annotation_window.mapToScene(event.pos())
        if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            self.toggle_drop_at(scene_pos)
        elif modifiers & Qt.ControlModifier:
            self.add_negative_at(scene_pos)
        else:
            return
        self.report_state()

    @staticmethod
    def is_click(press_pos, release_pos):
        """True when a press and release are close enough to be a click, not a drag."""
        delta = release_pos - press_pos
        return abs(delta.x()) + abs(delta.y()) <= CLICK_SLOP_PX

    def wheelEvent(self, event):
        """Ctrl+wheel: the confidence threshold, live. Ctrl+Shift+wheel: finer steps.

        The annotation window routes Ctrl+wheel here instead of zooming. The
        threshold changed is the global one, so the value that works here is the
        one the Generator runs with. Below THRESHOLD_FINE_CUTOFF the step drops
        to THRESHOLD_FINE_STEP even without Shift, since the normal step would
        overshoot the values that matter near zero.
        """
        if not event.modifiers() & Qt.ControlModifier:
            return
        # Shift+wheel is horizontal scrolling on some platforms, so the delta can
        # arrive on either axis.
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return
        near_zero = self._threshold() <= THRESHOLD_FINE_CUTOFF
        step = THRESHOLD_FINE_STEP if (event.modifiers() & Qt.ShiftModifier or near_zero) else THRESHOLD_STEP
        self.nudge_threshold(step if delta > 0 else -step)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles the key press event.

        Args:
            event (QKeyEvent): The key press event
        """
        if event.key() == Qt.Key_T and event.modifiers() == Qt.ControlModifier:
            self.prompt_for_text()
            return

        # Ctrl+Shift held: reveal removed detections so they can be restored, as
        # the Work Area tool reveals its remove buttons. With Backspace or Delete,
        # clear the whole prompt session -- the Work Area tool's clear-all chord.
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            self._set_revealing_dropped(True)
            if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
                self.clear_session()
            elif self.previews:
                self.main_window.status_bar.showMessage(
                    "Ctrl+Shift+right-click a detection to remove it, or a dotted red outline to "
                    "restore it  |  Ctrl+Shift+Backspace: clear the prompt session", 4000)
            self.annotation_window.scene.update()
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

            # No boxes drawn, but the session already holds a prompt -- a phrase,
            # or examples carried over from another work area. `session_processed`
            # matters when the prompt finds nothing: without it this branch matches
            # again on every Space -- annotations stay empty, so the prediction
            # re-runs forever and the work area can never be closed. One attempt
            # per work area, then Space means what it means everywhere else.
            elif not self.annotations and self._has_prompt() and not self.session_processed:
                self.create_annotations_from_session()
                self.session_processed = True

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

            # Backspace undoes one layer at a time, and the last layer is the work
            # area itself -- the same order as the SAM tool. A loaded phrase is not a
            # layer: it belongs to the prompt session, which outlives the work area,
            # so it must never stand between the user and closing it. (It used to:
            # Backspace cleared the phrase, then did nothing, and the only ways out
            # were Space -- which confirms -- or switching tools or images.)
            # Clear the phrase with an empty Ctrl+T, or the whole session with
            # Ctrl+Shift+Backspace.
            if self.drawing_rectangle:
                self.cancel_rectangle_drawing()
            # If we have a working area and accumulated annotations, clear them
            elif self.working_area and len(self.annotations) > 0:
                self.clear_annotations()  # Clears unconfirmed annotations
                # Cleared detections mean the prompt can be tried again.
                self.session_processed = False
            elif self.rectangles:
                self.clear_all_rectangles()  # Clears user input rectangles
            elif self.working_area:
                self.cancel_working_area()

        self.annotation_window.scene.update()
        self.report_state()

    def keyReleaseEvent(self, event: QKeyEvent):
        """Hide removed detections again once Ctrl+Shift is let go."""
        modifiers = event.modifiers()
        if self.revealing_dropped and not (modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier):
            self._set_revealing_dropped(False)
        if not modifiers & Qt.ControlModifier:
            self._clear_hover()
        self.annotation_window.scene.update()

    def clear_session(self):
        """Forget every example in the prompt session (Ctrl+Shift+Backspace)."""
        if self.see_anything_dialog is not None and hasattr(self.see_anything_dialog, 'clear_session'):
            self.see_anything_dialog.clear_session()
        self.text_prompt = None
        self.session_processed = False
        self.session_found_nothing = False
        self.main_window.status_bar.showMessage(
            "Prompt session cleared. Draw a box or Ctrl+T to start a new one.", 5000)

    def clear_text_prompt(self):
        """Drop the current phrase and go back to box prompting.

        Clears it on the dialog too, so the model is not left with a phrase's
        class embedding standing where the next visual prompt expects its own.
        """
        self.text_prompt = None
        self.session_processed = False
        self.session_found_nothing = False
        if self.see_anything_dialog is not None:
            try:
                self.see_anything_dialog.set_text_prompt(None)
            except Exception:
                pass
        self.main_window.status_bar.showMessage("Text prompt cleared.", 3000)

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

    def suggestion_candidates(self):
        """Phrases worth ranking against a drawn box.

        Every word the model knows, plus the project's own labels. The point of
        suggesting is to find a phrase the user would not have thought of, so
        restricting it to labels they already hand-picked would defeat it: a box
        around a bus should be able to come back "vehicle".

        The word list is built once per checkpoint, behind a confirmation, and
        cached. Declining leaves the project labels, which still work.

        Returns:
            list[str]: De-duplicated candidate phrases.
        """
        names = []
        try:
            names = ensure_vocabulary(self.annotation_window,
                                      self.see_anything_dialog.text_embedder()) or []
        except Exception as e:
            self.main_window.status_bar.showMessage(
                f"Could not load the model's word list: {e}", 5000)

        names = names + project_label_names(getattr(self.main_window, 'label_window', None))
        return list(dict.fromkeys(names))

    def suggest_text_for_rectangles(self):
        """Rank phrases against the boxes currently drawn in the work area.

        Returns:
            list[dict]: As `PromptAlignment.rank` returns, or empty if nothing
            is drawn or there is nothing to rank against.
        """
        # Rectangles first: gathering candidates can ask the user to build the
        # word list, which would be an odd thing to offer with nothing to rank.
        if not self.rectangles:
            return []
        candidates = self.suggestion_candidates()
        if not candidates:
            return []
        return self.see_anything_dialog.suggest_text_prompts(self.rectangles,
                                                             candidates=candidates)

    def prompt_for_text(self):
        """Ask for a text prompt (Ctrl+T) to use in place of drawn boxes.

        Ultralytics turns the phrase into a class embedding via
        `YOLOE.get_text_pe`, so no reference boxes are needed at all. An empty
        entry clears the prompt and returns to box prompting.

        A box already drawn is worth more than a guess, so the dialog can rank
        candidate phrases against it: the model's own embedding space knows what
        would find that object again, even when the user does not know what it
        is called.
        """
        if self.see_anything_dialog is None or self.see_anything_dialog.loaded_model is None:
            self.main_window.status_bar.showMessage(
                "Load a See Anything model before using a text prompt.", 4000)
            return

        suggest = self.suggest_text_for_rectangles if self.rectangles else None

        dialog = TextPromptDialog(current_text=self.text_prompt or "",
                                  suggest=suggest,
                                  parent=self.annotation_window)
        if dialog.exec_() != dialog.Accepted:
            return

        text = dialog.value()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            applied = self.see_anything_dialog.set_text_prompt(text)
        except Exception as e:
            applied = False
            self.main_window.status_bar.showMessage(f"Could not set text prompt: {e}", 5000)
        finally:
            QApplication.restoreOverrideCursor()

        self.text_prompt = text if (text and applied) else None
        # A new phrase has not been run yet, whatever the last one did.
        self.session_processed = False
        self.session_found_nothing = False
        if self.text_prompt:
            self.main_window.status_bar.showMessage(
                f"Text prompt set to '{self.text_prompt}'. Press Space to predict.", 5000)
        elif not text:
            self.main_window.status_bar.showMessage("Text prompt cleared.", 3000)
        self.report_state()

    def create_annotations_from_rectangles(self):
        """Turn the boxes just drawn into one example, then predict from the session.

        The boxes share `cls=0` and merge into a single embedding, exactly as a box
        prompt always has; that embedding joins the session as a positive example.
        Predicting from the session afterwards was measured to give the same
        detections as predicting with the boxes in-image, to the last digit -- and it
        is the path the Generator runs, which is what makes the session portable.
        """
        if not self.annotation_window.active_image:
            return None

        if not self.working_area:
            return None

        if len(self.rectangles) == 0:  # Check specifically for user-drawn rectangles
            return None

        masks = None
        # Create masks from the rectangles (these are not polygons)
        if self.see_anything_dialog.get_task() == 'segment':
            masks = []
            for r in self.rectangles:
                x1, y1, x2, y2 = r
                masks.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))

        # The rectangles are in work-area pixels and are handed over unscaled:
        # ultralytics rasterizes them against its own letterbox (see
        # DeployPredictorDialog.build_prompts).
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            embedding = self.see_anything_dialog.embed_boxes(self.rectangles, masks=masks)
        except Exception as e:
            embedding = None
            self.main_window.status_bar.showMessage(f"Could not read those reference boxes: {e}", 5000)
        finally:
            QApplication.restoreOverrideCursor()

        if embedding is None:
            self.main_window.status_bar.showMessage(
                "See Anything returned nothing for those reference boxes.", 5000)
            return None

        count = len(self.rectangles)
        self.see_anything_dialog.add_session_positive(
            KIND_BOXES, embedding,
            f"{count} box{'es' if count != 1 else ''} on {self._image_name()}")
        return self.run_session()

    def create_annotations_from_session(self):
        """Predict from the prompt session without drawing anything new.

        Covers a phrase set with Ctrl+T and examples carried over from another work
        area. An empty result is the normal outcome of a prompt that does not
        describe anything here, so it is reported as a state rather than a
        five-second flash -- `report_state` keeps saying it until the prompt changes
        or the work area closes.
        """
        if not self.annotation_window.active_image or not self.working_area:
            return None
        if not self._has_prompt():
            return None
        return self.run_session()

    def _image_name(self):
        return os.path.basename(str(self.image_path or self.annotation_window.current_image_path or "image"))

    def run_session(self):
        """Predict on this work area from every enabled example, and show the result.

        The prediction is fetched down to `PREDICT_CONFIDENCE_FLOOR` and cached raw;
        what is shown is filtered from that cache, which is what lets Ctrl+wheel
        move the threshold without predicting again.

        Returns:
            int: The number of detections now shown.
        """
        if not self.working_area or self.see_anything_dialog is None:
            return 0

        floor = min(self._threshold(), PREDICT_CONFIDENCE_FLOOR)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = self.see_anything_dialog.predict_from_session(conf=floor)
        finally:
            QApplication.restoreOverrideCursor()

        self.session_processed = True
        self.session_found_nothing = False
        self.raw_floor = floor

        result = results[0] if results else None
        if result is None or result.boxes is None or len(result.boxes) == 0:
            self._set_raw_results(None)
            self.session_found_nothing = True
            self.report_nothing_found()
            return 0

        self._set_raw_results(result)
        shown = self.apply_visibility(report=True)
        if not shown:
            self.session_found_nothing = True
            self.report_nothing_found()
        return shown

    def report_nothing_found(self):
        """Say that the prompt matched nothing, and what to do about it.

        A high alignment score does not promise a detection -- alignment compares
        a phrase against your reference crops, while detection compares that same
        phrase against regions of *this* image -- so "nothing found" needs to name
        the ways out rather than read as a failure.
        """
        threshold = f" at confidence {self._threshold():.2f}"
        if self.text_prompt and self._prompt_label().startswith("'"):
            subject = f"'{self.text_prompt}'"
        else:
            subject = "the prompt session"

        self.main_window.status_bar.showMessage(
            f"Nothing in this work area matched {subject}{threshold}. "
            f"Ctrl+wheel: lower the confidence threshold  |  Ctrl+T: try another word  |  "
            f"Backspace or Space: close the work area",
            10000)

    # --- The cached prediction and its on-screen view ---------------------------------------------------------------

    def _set_raw_results(self, result):
        """Replace the cached prediction and rebuild the list of detections in it.

        IoU and area filters run once, here: neither changes when the threshold
        does. Confidence is left to `apply_visibility`. Detections the user removed
        from the previous prediction stay removed if a new one lands on them.
        """
        self._discard_previews()

        if result is None:
            return

        collapse_to_one_class(result, self._label_name())

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
        result.path = self.image_path

        results_processor = ResultsProcessor(self.main_window, {})
        result = results_processor.filter_by_iou(result)
        result = results_processor.filter_by_area(result)

        self.raw_results = result
        self.previews = self._build_previews(result)

    def _label_name(self):
        label = self.annotation_window.selected_label
        return getattr(label, 'short_label_code', None) or "object"

    def _area_bounds(self):
        """Area bounds in pixels, relative to the WHOLE image.

        Resolved through the same helper ResultsProcessor uses, so both filters
        agree. Scaling by the work-area crop made the same threshold mean a
        different real size depending on how far the view happened to be zoomed.
        `None` means the threshold cannot be judged for this raster (a real-world
        bound with no scale), in which case every detection is kept rather than
        silently dropped.
        """
        try:
            _raster = self.main_window.image_window.raster_manager.get_raster(self.image_path)
        except Exception:
            _raster = None

        image_area, m2_per_px = raster_metrics(_raster)
        if not image_area:
            image_area = float(self.original_width or 0) * float(self.original_height or 0)
        if not image_area:
            image_area = self.work_area_image.shape[0] * self.work_area_image.shape[1]

        return resolve_area_bounds_px(
            self.main_window.get_area_thresh_min(),
            self.main_window.get_area_thresh_max(),
            get_area_mode(self.main_window),
            image_area, m2_per_px)

    def _build_previews(self, result):
        """One `PreviewDetection` per detection in the cached prediction."""
        previews = []
        self.dropped_area = 0
        if result is None or result.boxes is None:
            return previews

        offset = self.working_area.rect.topLeft()
        dx, dy = offset.x(), offset.y()
        height, width = self.work_area_image.shape[:2]
        area_bounds = self._area_bounds()
        segment = self.see_anything_dialog.get_task() == "segment" and result.masks is not None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        polygons = result.masks.xyn if segment else None

        for i, (box_work, confidence) in enumerate(zip(boxes, confidences)):
            box_area = (box_work[2] - box_work[0]) * (box_work[3] - box_work[1])
            if area_bounds and not (area_bounds[0] <= box_area <= area_bounds[1]):
                self.dropped_area += 1
                continue

            box_image = box_work + np.array([dx, dy, dx, dy], dtype=box_work.dtype)
            polygon_image = None
            if polygons is not None:
                # Normalized to the work-area crop; scale and offset into the image.
                polygon_image = polygons[i].copy()
                polygon_image[:, 0] = polygon_image[:, 0] * width + dx
                polygon_image[:, 1] = polygon_image[:, 1] * height + dy

            entry = PreviewDetection(i, float(confidence), box_work.copy(), box_image, polygon_image)
            entry.dropped = any(_box_iou(box_image, gone) >= DROP_MATCH_IOU for gone in self.dropped_boxes)
            previews.append(entry)

        return previews

    def _entry_visible(self, entry, threshold=None):
        threshold = self._threshold() if threshold is None else threshold
        return entry.confidence >= threshold and not entry.dropped

    def apply_visibility(self, report=False):
        """Show the detections that pass the threshold and were not removed.

        Re-derives `self.annotations` (what Space confirms) and `self.results` (what
        SAM refines) from the cache. No model call.

        Returns:
            int: The number of detections now shown.
        """
        threshold = self._threshold()
        shown_indices = []
        self.annotations = []
        below_threshold = 0
        removed = 0

        for entry in self.previews:
            if entry.confidence < threshold:
                below_threshold += 1
            elif entry.dropped:
                removed += 1

            if self._entry_visible(entry, threshold):
                if entry.annotation is None:
                    entry.annotation = self._make_preview_annotation(entry)
                if entry.annotation is None:
                    continue
                entry.annotation.set_visibility(True)
                self.annotations.append(entry.annotation)
                shown_indices.append(entry.index)
            elif entry.annotation is not None:
                entry.annotation.set_visibility(False)

        if shown_indices and self.raw_results is not None:
            self.results = self.raw_results[torch.as_tensor(shown_indices, dtype=torch.long)]
        else:
            self.results = None

        self._applied_threshold = threshold
        self._refresh_dropped_graphics()
        self.annotation_window.scene.update()

        if report:
            self.report_detection_counts(below_threshold, self.dropped_area, removed=removed)
        return len(self.annotations)

    def _make_preview_annotation(self, entry):
        if entry.polygon_image is not None:
            return self.create_polygon_annotation(entry.polygon_image, entry.confidence)
        return self.create_rectangle_annotation(entry.box_image, entry.confidence)

    def _discard_previews(self, keep=()):
        """Delete preview graphics, shown or hidden, except annotations in `keep`."""
        keep_ids = {id(a) for a in keep}
        doomed = [e.annotation for e in self.previews if e.annotation is not None]
        doomed += [a for a in self.annotations if all(a is not d for d in doomed)]
        for annotation in doomed:
            if id(annotation) in keep_ids:
                continue
            try:
                annotation.delete()
            except Exception:
                pass

        self.previews = []
        self.annotations = []
        self.raw_results = None
        self.results = None
        self._clear_dropped_graphics()

    # --- Threshold -----------------------------------------------------------------------------------------------------

    def nudge_threshold(self, delta):
        """Move the global confidence threshold and refilter what is shown.

        Returns:
            float: The threshold now in force.
        """
        current = self._threshold()
        new = round(min(1.0, max(0.0, current + delta)), 2)
        if new == round(current, 2):
            self.main_window.status_bar.showMessage(
                f"Confidence threshold is already {new:.2f}.", 2000)
            return current

        self.main_window.update_uncertainty_thresh(new)
        if self._applied_threshold is None or abs(self._applied_threshold - new) > 1e-9:
            self.apply_threshold(new)
        return new

    def on_uncertainty_changed(self, value):
        """Follow the main window's threshold slider as well as Ctrl+wheel."""
        if not self.active:
            return
        if self._applied_threshold is not None and abs(self._applied_threshold - value) <= 1e-9:
            return
        self.apply_threshold(value)

    def apply_threshold(self, value):
        """Refilter from the cache, or fetch again if the threshold went below the floor."""
        needs_fetch = (self.session_processed
                       and self.working_area is not None
                       and self._has_prompt()
                       and self.raw_floor is not None
                       and value < self.raw_floor - 1e-9)
        if needs_fetch:
            self.run_session()
        elif self.previews or self.raw_results is not None:
            self.apply_visibility()
        else:
            self._applied_threshold = value

        self.main_window.status_bar.showMessage(
            f"Confidence threshold {value:.2f}: {len(self.annotations)} "
            f"detection{'s' if len(self.annotations) != 1 else ''} shown"
            "  |  Ctrl+Shift+wheel: finer steps", 4000)

    # --- Examples and removals -----------------------------------------------------------------------------------------

    def detection_at(self, scene_pos, include_dropped=False):
        """The detection under a point, preferring the smallest when they overlap.

        Args:
            scene_pos (QPointF): Point in image coordinates.
            include_dropped (bool): Also consider removed detections, for restoring.

        Returns:
            PreviewDetection | None
        """
        x, y = scene_pos.x(), scene_pos.y()
        threshold = self._threshold()
        best, best_area = None, None
        for entry in self.previews:
            shown = self._entry_visible(entry, threshold) and entry.annotation is not None
            if not (shown or (include_dropped and entry.dropped)):
                continue
            x1, y1, x2, y2 = entry.box_image
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best_area:
                best, best_area = entry, area
        return best

    def toggle_drop_at(self, scene_pos):
        """Remove the detection under the point, or restore a removed one.

        Local to this work area: it changes what gets confirmed, not the prompt.

        Returns:
            bool: True if a detection was removed or restored.
        """
        entry = self.detection_at(scene_pos, include_dropped=True)
        if entry is None:
            self.main_window.status_bar.showMessage("No detection there to remove.", 3000)
            return False

        self._set_dropped(entry, not entry.dropped)
        self.apply_visibility()
        if entry.dropped:
            message = "Removed one detection. Ctrl+Shift+right-click it again to restore it."
        else:
            message = "Restored the detection."
        self.main_window.status_bar.showMessage(message, 4000)
        return True

    def _set_dropped(self, entry, dropped):
        entry.dropped = dropped
        if dropped:
            self.dropped_boxes.append(entry.box_image.copy())
        else:
            self.dropped_boxes = [b for b in self.dropped_boxes
                                  if _box_iou(b, entry.box_image) < DROP_MATCH_IOU]

    def _embed_entry(self, entry):
        """One embedding for a detection on this work area, or None."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            vectors = self.see_anything_dialog.embed_detections([entry.box_work])
        except Exception as e:
            vectors = []
            self.main_window.status_bar.showMessage(f"Could not read that detection: {e}", 5000)
        finally:
            QApplication.restoreOverrideCursor()
        return vectors[0] if vectors else None

    def add_positive_at(self, scene_pos):
        """More like this: add the detection under the point as a positive example.

        Each such example is its own class. Measured, a separate class adds its own
        kind of object and leaves the rest alone -- and it can be switched off by
        itself in the session panel to see whether it was helping.

        Returns:
            bool: True if the example was added.
        """
        entry = self.detection_at(scene_pos)
        if entry is None:
            return False

        embedding = self._embed_entry(entry)
        if embedding is None:
            return False

        self.see_anything_dialog.add_session_positive(
            KIND_DETECTION, embedding, f"detection on {self._image_name()}")
        shown = self.run_session()
        self.main_window.status_bar.showMessage(
            f"Added as a positive example; now showing {shown} "
            f"detection{'s' if shown != 1 else ''}.", 5000)
        return True

    def add_negative_at(self, scene_pos):
        """Fewer like this: remove the detection under the point AND add it as a negative.

        One gesture does both. The negative teaches the prompt -- it becomes a decoy
        class whose detections are thrown away -- but a decoy only wins objects that
        look more like it than like any positive, so the detection clicked is not
        guaranteed to vanish on its own. It is removed outright as well, and stays
        removed across the re-run.

        A decoy only competes against visual positives -- a phrase outscores an
        example crop of the same object by far (0.95 against 0.34) -- so with a
        text-only prompt the negative is kept in the session but has no effect yet,
        and the user is told what would make it count.

        Returns:
            bool: True if the detection was removed and kept as a negative example.
        """
        entry = self.detection_at(scene_pos)
        if entry is None:
            return False

        # Removed whatever else happens, so the click always does what it says.
        self._set_dropped(entry, True)

        embedding = self._embed_entry(entry)
        if embedding is None:
            self.apply_visibility()
            self.main_window.status_bar.showMessage(
                "Removed that detection, but could not read it as a negative example.", 6000)
            return False

        self.see_anything_dialog.add_session_negative(
            embedding, f"detection on {self._image_name()}")

        session = self._session()
        if session is not None and session.decoys_can_compete():
            shown = self.run_session()
            self.main_window.status_bar.showMessage(
                f"Removed and added as a negative example; now showing {shown} "
                f"detection{'s' if shown != 1 else ''}.", 5000)
        else:
            # Nothing a decoy can change yet, so no re-run -- just hide it.
            self.apply_visibility()
            self.main_window.status_bar.showMessage(
                "Removed and kept as a negative example. Negatives only take effect once the prompt "
                "has a visual example -- Ctrl+click a good detection to add one.", 8000)
        return True

    def on_session_edited(self):
        """An example was toggled or removed in the session panel: re-run the work area."""
        if not self.active or self.working_area is None:
            return
        session = self._session()
        self.text_prompt = session.text_phrase() if session is not None else None
        if self._has_prompt():
            self.run_session()
        else:
            self._set_raw_results(None)
            self.session_processed = False
        self.report_state()

    # --- Graphics for removals and hover -------------------------------------------------------------------------------

    def _set_revealing_dropped(self, revealing):
        if revealing == self.revealing_dropped:
            return
        self.revealing_dropped = revealing
        self.annotation_window.setCursor(Qt.PointingHandCursor if revealing else self.cursor)
        self._refresh_dropped_graphics()

    def _clear_dropped_graphics(self):
        for item in self.dropped_graphics:
            try:
                self.annotation_window.scene.removeItem(item)
            except Exception:
                pass
        self.dropped_graphics = []

    def _refresh_dropped_graphics(self):
        """Faint outlines on removed detections, only while Ctrl+Shift is held."""
        self._clear_dropped_graphics()
        if not self.revealing_dropped:
            return

        pen = QPen(QColor(230, 60, 60))
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.DotLine)
        for entry in self.previews:
            if not entry.dropped:
                continue
            x1, y1, x2, y2 = entry.box_image
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            item.setPen(pen)
            self.annotation_window.scene.addItem(item)
            self.dropped_graphics.append(item)

    def _clear_hover(self):
        if self.hover_graphics is not None:
            try:
                self.annotation_window.scene.removeItem(self.hover_graphics)
            except Exception:
                pass
            self.hover_graphics = None

    def update_hover(self, scene_pos, modifiers):
        """Outline the detection a Ctrl-click would act on, so it is never a guess."""
        self._clear_hover()
        if not self.previews or not modifiers & Qt.ControlModifier:
            return

        removing = bool(modifiers & Qt.ShiftModifier)
        entry = self.detection_at(scene_pos, include_dropped=removing)
        if entry is None:
            return

        pen = QPen(QColor(230, 60, 60) if removing else QColor(255, 215, 0))
        pen.setCosmetic(True)
        pen.setWidth(3)
        x1, y1, x2, y2 = entry.box_image
        self.hover_graphics = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.hover_graphics.setPen(pen)
        self.annotation_window.scene.addItem(self.hover_graphics)

    def report_detection_counts(self, dropped_confidence, dropped_area, removed=0):
        """Say how many detections were kept, and what the thresholds removed."""
        kept = len(self.annotations)
        message = f"{kept} detection{'s' if kept != 1 else ''} found"

        dropped = []
        if dropped_confidence:
            dropped.append(f"{dropped_confidence} below the uncertainty threshold")
        if dropped_area:
            dropped.append(f"{dropped_area} outside the area thresholds")
        if removed:
            dropped.append(f"{removed} removed by you")
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
        Create a preview rectangle annotation from image-coordinate box corners.

        Args:
            box (np.ndarray): The bounding box coordinates.
            confidence (float): The confidence score for the annotation.

        Returns:
            RectangleAnnotation | None: The preview, already in the scene.
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
            return annotation
        return None

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

        # Hidden previews too: the threshold can bring them back on screen.
        for annotation in self._preview_annotations():
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
        for annotation in self._preview_annotations():
            annotation.update_transparency(value)
        self.annotation_window.scene.update()

    def create_polygon_annotation(self, points, confidence):
        """
        Create a preview polygon annotation from image-coordinate points.

        Args:
            points (np.ndarray): The polygon points.
            confidence (float): The confidence score for the annotation.

        Returns:
            PolygonAnnotation | None: The preview, already in the scene, or None
            for a polygon too small to draw.
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
            return annotation
        return None

    def _preview_annotations(self):
        """Every preview annotation drawn so far, shown or hidden by the threshold."""
        annotations = [e.annotation for e in self.previews if e.annotation is not None]
        annotations += [a for a in self.annotations if all(a is not b for b in annotations)]
        return annotations

    def confirm_annotations(self, crop_annotations=False):
        """
        Confirm the annotations and clear the working area.
        """
        # Only what is on screen is confirmed; captured before any cleanup runs.
        confirmed = list(self.annotations)
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

            # The confirmed annotations now belong to the annotation window; the
            # previews the threshold or a removal hid do not, and are deleted.
            self._discard_previews(keep=confirmed)

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
        # Hidden previews included: the threshold hides them, it does not delete them.
        self._discard_previews()
        # Removals belong to the detections just discarded.
        self.dropped_boxes = []
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
        # The prompt session survives so the next work area can reuse it, but its
        # per-work-area run state does not.
        self.session_processed = False
        self.session_found_nothing = False
        self._set_revealing_dropped(False)
        self._clear_hover()

        # Hidden previews would otherwise stay in the scene with no owner.
        self._discard_previews()
        self.dropped_boxes = []
        self.raw_floor = None

        # Force update to ensure graphics are removed visually
        self.annotation_window.scene.update()
