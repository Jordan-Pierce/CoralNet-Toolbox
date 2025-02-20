import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QGraphicsPixmapItem)

from coralnet_toolbox.Annotations import (
    PatchAnnotation,
    PolygonAnnotation,
    RectangleAnnotation
)

from coralnet_toolbox.Tools import (
    PanTool,
    PatchTool,
    PolygonTool,
    RectangleTool,
    SAMTool,
    SelectTool,
    ZoomTool
)

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    viewChanged = pyqtSignal(int, int)  # Signal to emit when view is changed
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes
    annotationSelected = pyqtSignal(int)  # Signal to emit when annotation is selected
    annotationDeleted = pyqtSignal(str)  # Signal to emit when annotation is deleted
    annotationCreated = pyqtSignal(str)  # Signal to emit when annotation is created
    hover_point = pyqtSignal(QPointF)  # Signal to emit when mouse hovers over a point

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.annotation_size = 224
        self.annotation_color = None
        self.transparency = 128

        self.zoom_factor = 1.0
        self.pan_active = False
        self.pan_start = None
        self.drag_start_pos = None
        self.cursor_annotation = None

        self.annotations_dict = {}  # Dictionary to store annotations by UUID
        self.image_annotations_dict = {}  # Dictionary to store annotations by image path

        self.selected_annotations = []  # Stores the selected annotations
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.image_pixmap = None
        self.rasterio_image = None
        self.active_image = False
        self.current_image_path = None

        self.toolChanged.connect(self.set_selected_tool)

        self.tools = {
            "pan": PanTool(self),
            "zoom": ZoomTool(self),
            "select": SelectTool(self),
            "patch": PatchTool(self),
            "rectangle": RectangleTool(self),
            "polygon": PolygonTool(self),
            "sam": SAMTool(self),
        }

    def wheelEvent(self, event: QMouseEvent):
        # Handle zooming with the mouse wheel
        if self.selected_tool and event.modifiers() & Qt.ControlModifier:
            self.tools[self.selected_tool].wheelEvent(event)
        elif self.active_image:
            self.tools["zoom"].wheelEvent(event)

        self.viewChanged.emit(*self.get_image_dimensions())

    def mousePressEvent(self, event: QMouseEvent):
        if self.active_image:
            self.tools["pan"].mousePressEvent(event)
        if self.selected_tool:
            self.tools[self.selected_tool].mousePressEvent(event)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.active_image:
            self.tools["pan"].mouseMoveEvent(event)
        if self.selected_tool:
            self.tools[self.selected_tool].mouseMoveEvent(event)

        scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))
        self.hover_point.emit(scene_pos)

        if not self.cursorInWindow(event.pos()):
            self.toggle_cursor_annotation()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.active_image:
            self.tools["pan"].mouseReleaseEvent(event)
        if self.selected_tool:
            self.tools[self.selected_tool].mouseReleaseEvent(event)

        self.toggle_cursor_annotation()
        self.drag_start_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.active_image and self.selected_tool:
            self.tools[self.selected_tool].keyPressEvent(event)
        super().keyPressEvent(event)

        # Handle the hot key for deleting (backspace or delete) selected annotations
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                if self.selected_annotations:
                    self.delete_selected_annotation()

    def keyReleaseEvent(self, event):
        if self.active_image and self.selected_tool:
            self.tools[self.selected_tool].keyReleaseEvent(event)
        super().keyReleaseEvent(event)

    def cursorInWindow(self, pos, mapped=False):
        if not pos or not self.image_pixmap:
            return False

        image_rect = QGraphicsPixmapItem(self.image_pixmap).boundingRect()
        if not mapped:
            pos = self.mapToScene(pos)

        return image_rect.contains(pos)

    def cursorInViewport(self, pos):
        if not pos:
            return False

        return self.viewport().rect().contains(pos)

    def set_selected_tool(self, tool):
        if self.selected_tool:
            self.tools[self.selected_tool].deactivate()
        self.selected_tool = tool
        if self.selected_tool:
            self.tools[self.selected_tool].activate()

        self.unselect_annotations()
        self.toggle_cursor_annotation()

    def set_selected_label(self, label):
        self.selected_label = label
        self.annotation_color = label.color

        for annotation in self.selected_annotations:
            if annotation.label.id != label.id:
                annotation.update_user_confidence(self.selected_label)
                annotation.create_cropped_image(self.rasterio_image)
                self.main_window.confidence_window.display_cropped_image(annotation)

        if self.cursor_annotation:
            if self.cursor_annotation.label.id != label.id:
                self.toggle_cursor_annotation()

    def set_annotation_location(self, annotation_id, new_center_xy: QPointF):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            # Disconnect the confidence window from the annotation, so it won't update while moving
            annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
            annotation.update_location(new_center_xy)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            self.main_window.confidence_window.display_cropped_image(annotation)

    def set_annotation_size(self, size=None, delta=0):
        if size is not None:
            self.annotation_size = size
        else:
            self.annotation_size += delta
            self.annotation_size = max(1, self.annotation_size)

        # Cursor or 1 annotation selected
        if len(self.selected_annotations) == 1:
            annotation = self.selected_annotations[0]
            if not self.is_annotation_moveable(annotation):
                return
            
            # Disconnect the confidence window from the annotation, so it won't update while resizing
            annotation.annotationUpdated.disconnect(self.main_window.confidence_window.display_cropped_image)
            
            if isinstance(annotation, PatchAnnotation):
                annotation.update_annotation_size(self.annotation_size)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(self.annotation_size)
            elif isinstance(annotation, RectangleAnnotation):
                scale_factor = 1 + delta / 100.0
                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)
            elif isinstance(annotation, PolygonAnnotation):
                scale_factor = 1 + delta / 100.0
                annotation.update_annotation_size(scale_factor)
                if self.cursor_annotation:
                    self.cursor_annotation.update_annotation_size(scale_factor)

            # Create and display the cropped image in the confidence window
            annotation.create_cropped_image(self.rasterio_image)
            # Connect the confidence window back to the annotation
            annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
            # Display the cropped image in the confidence window
            self.main_window.confidence_window.display_cropped_image(annotation)

        # Only emit if 1 or no annotations are selected
        if len(self.selected_annotations) <= 1:
            # Emit that the annotation size has changed
            self.annotationSizeChanged.emit(self.annotation_size)

    def is_annotation_moveable(self, annotation):
        if annotation.show_message:
            self.unselect_annotations()
            annotation.show_warning_message()
            return False
        return True

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):

        if self.cursor_annotation:
            self.cursor_annotation.delete()
            self.cursor_annotation = None

        if not self.selected_label or not self.annotation_color:
            return

        if scene_pos:
            try:
                self.cursor_annotation = self.tools[self.selected_tool].create_annotation(scene_pos)
                self.cursor_annotation.create_graphics_item(self.scene)
            except:
                pass

    def display_image_item(self, image_item):
        # Clean up
        self.clear_scene()

        # Display NaN values the image dimensions in status bar
        self.imageLoaded.emit(0, 0)
        self.viewChanged.emit(0, 0)

        # Set the image representations
        self.image_pixmap = QPixmap(image_item)
        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        QApplication.processEvents()

    def set_image(self, image_path):

        # Clean up
        self.clear_scene()

        # Check that the image path is valid
        if image_path not in self.main_window.image_window.images:
            return

        # Set the image representations
        self.image_pixmap = QPixmap(self.main_window.image_window.images[image_path])
        self.rasterio_image = self.main_window.image_window.rasterio_images[image_path]

        self.current_image_path = image_path
        self.active_image = True

        self.tools["zoom"].reset_zoom()
        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.tools["zoom"].calculate_min_zoom()

        self.toggle_cursor_annotation()

        # Set the image dimensions, and current view in status bar
        self.imageLoaded.emit(self.image_pixmap.width(), self.image_pixmap.height())
        self.viewChanged.emit(self.image_pixmap.width(), self.image_pixmap.height())

        # Load all associated annotations
        self.load_annotations()
        # Update the image window's image dict
        self.main_window.image_window.update_image_annotations(image_path)
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

        QApplication.processEvents()

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def viewportToScene(self):
        # Map the top-left and bottom-right corners of the viewport to the scene coordinates
        top_left = self.mapToScene(self.viewport().rect().topLeft())
        bottom_right = self.mapToScene(self.viewport().rect().bottomRight())
        # Create and return a QRectF object from these points
        return QRectF(top_left, bottom_right)

    def get_image_dimensions(self):
        if self.image_pixmap:
            return self.image_pixmap.size().width(), self.image_pixmap.size().height()
        return 0, 0

    def center_on_annotation(self, annotation):
        # Create graphics item if it doesn't exist
        if not annotation.graphics_item:
            annotation.create_graphics_item(self.scene)
            
        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()
        annotation_center = annotation_rect.center()

        # Center the view on the annotation's center
        self.centerOn(annotation_center)

    def cycle_annotations(self, direction):
        # Get the annotations for the current image
        annotations = self.get_image_annotations()
        if not annotations:
            return

        if self.selected_tool == "select" and self.active_image:

            if self.main_window.label_window.label_locked:
                locked_label = self.main_window.label_window.locked_label
                indices = [i for i, a in enumerate(annotations) if a.label.id == locked_label.id]

                if not indices:
                    return

                if self.selected_annotations:
                    current_index = annotations.index(self.selected_annotations[0])
                else:
                    current_index = indices[0]

                if current_index in indices:
                    # Find position in indices list and cycle within that
                    current_pos = indices.index(current_index)
                    new_pos = (current_pos + direction) % len(indices)
                    new_index = indices[new_pos]  # Get the actual annotation index
                else:
                    # Find next valid index based on direction
                    if direction > 0:
                        next_indices = [i for i in indices if i > current_index]
                        new_index = next_indices[0] if next_indices else indices[0]
                    else:
                        prev_indices = [i for i in indices if i < current_index]
                        new_index = prev_indices[-1] if prev_indices else indices[-1]

            elif self.selected_annotations:
                # Cycle through all the annotations
                current_index = annotations.index(self.selected_annotations[0])
                new_index = (current_index + direction) % len(annotations)

            else:
                # Select the first annotation
                new_index = 0

            if 0 <= new_index < len(annotations):
                # Select the new annotation
                self.select_annotation(annotations[new_index])
                # Center the view on the new annotation
                self.center_on_annotation(annotations[new_index])

    def select_annotation(self, annotation, ctrl_pressed=False):
        if not ctrl_pressed:
            self.unselect_annotations()

        if annotation in self.selected_annotations:
            self.unselect_annotation(annotation)
            return

        if annotation not in self.selected_annotations:
            self.selected_annotations.append(annotation)
            annotation.select()

            # Set the label and color for selected annotation
            self.selected_label = annotation.label
            self.annotation_color = annotation.label.color

            # Emit a signal indicating the selected annotations label to LabelWindow
            # which then update the label in the label window, followed by transparency.
            # Only do this if only one annotation is selected, otherwise all selected
            # annotations will change label
            self.annotationSelected.emit(annotation.id)

            if len(self.selected_annotations) == 1:
                self.labelSelected.emit(annotation.label.id)

                if not annotation.cropped_image:
                    # Crop the image from annotation using current image item
                    annotation.create_cropped_image(self.rasterio_image)

                # Display the selected annotation in confidence window
                annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
                self.main_window.confidence_window.display_cropped_image(annotation)

        if len(self.selected_annotations) > 1:
            self.main_window.label_window.deselect_active_label()
            self.main_window.confidence_window.clear_display()

    def select_annotations(self):
        """Select all annotations in the current image."""
        annotations = self.get_image_annotations()
        for annotation in annotations:
            if self.main_window.label_window.label_locked:
                if annotation.label.id == self.main_window.label_window.locked_label.id:
                    self.select_annotation(annotation, ctrl_pressed=True)
            else:
                self.select_annotation(annotation, ctrl_pressed=True)

    def unselect_annotation(self, annotation):
        if annotation in self.selected_annotations:
            annotation.deselect()
            self.selected_annotations.remove(annotation)
            self.main_window.confidence_window.clear_display()

    def unselect_annotations(self):
        for annotation in self.selected_annotations:
            annotation.deselect()
        self.selected_annotations = []
        self.main_window.confidence_window.clear_display()

    def load_annotation(self, annotation):
        # Remove the graphics item from its current scene if it exists
        if annotation.graphics_item and annotation.graphics_item.scene():
            annotation.graphics_item.scene().removeItem(annotation.graphics_item)

        # Create the graphics item (scene previously cleared)
        annotation.create_graphics_item(self.scene)
        # Connect essential update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
        # Update the view   
        self.viewport().update()

    def load_annotations(self, image_path=None, annotations=None):
        """
        Loads annotations for the current image by default or for a given image and annotations.
        """
        # Crop annotations (if image_path and annotations are provided, they are used)
        annotations = self.crop_annotations(image_path, annotations)
        total = len(annotations)
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Loading Annotations")
        progress_bar.show()
        progress_bar.start_progress(total)
        
        try:
            # Load each annotation and update progress
            for idx, annotation in enumerate(annotations):
                if progress_bar.wasCanceled():
                    break
                
                # Load the annotation
                self.load_annotation(annotation)

                # Update every 10% of the annotations (or for each item if total is small)
                if total > 10:
                    if idx % (total // 10) == 0:
                        progress_bar.update_progress_percentage((idx / total) * 100)
                else:
                    progress_bar.update_progress_percentage((idx / total) * 100)
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        QApplication.processEvents()
        self.viewport().update()

    def get_image_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        return self.image_annotations_dict.get(image_path, [])

    def get_image_review_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path and annotation.label.id == '-1':
                annotations.append(annotation)

        return annotations

    def crop_annotations(self, image_path=None, annotations=None, return_annotations=True, verbose=True):
        """
        Crop annotations for the specified image.

        If annotations is None, crop all annotations associated with the image (using self.get_image_annotations).
        If return_annotations is True, returns the list of annotations after cropping.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        if not image_path:
            image_path = self.current_image_path

        if annotations is None:
            annotations = self.get_image_annotations(image_path)

        progress_bar = None
        if verbose:
            progress_bar = ProgressBar(self, title="Cropping Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(annotations))

        try:
            rasterio_image = self.main_window.image_window.rasterio_open(image_path)
            for annotation in annotations:
                if not getattr(annotation, "cropped_image", False):
                    annotation.create_cropped_image(rasterio_image)
                if verbose:
                    progress_bar.update_progress()
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
        finally:
            QApplication.restoreOverrideCursor()
            if verbose:
                progress_bar.stop_progress()
                progress_bar.close()

        if return_annotations:
            return annotations

    def add_annotation(self, scene_pos: QPointF = None):
        if not self.selected_label:
            QMessageBox.warning(self,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return

        # Return if the isn't an active image
        if not self.active_image or not self.image_pixmap:
            return

        # Return if the position provided isn't in the window
        if scene_pos:
            if not self.cursorInWindow(scene_pos, mapped=True):
                return

        # Create the annotation for the selected tool
        annotation = self.tools[self.selected_tool].create_annotation(scene_pos, finished=True)

        if annotation is None:
            self.toggle_cursor_annotation()
            return

        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotationDeleted.connect(self.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
        
        # Create the graphics item
        annotation.create_graphics_item(self.scene)
        annotation.create_cropped_image(self.rasterio_image)
        
        # Display the cropped image in the confidence window
        self.main_window.confidence_window.display_cropped_image(annotation)

        # Add to annotation dict
        self.add_annotation_to_dict(annotation)
        
        # Update the table in ImageWindow
        self.annotationCreated.emit(annotation.id)

    def add_annotation_to_dict(self, annotation):       
        # Add to annotation dict
        self.annotations_dict[annotation.id] = annotation
        # Add to image annotations dict (if not already present)
        if annotation.image_path not in self.image_annotations_dict:
            self.image_annotations_dict[annotation.image_path] = []
        if annotation not in self.image_annotations_dict[annotation.image_path]:
            self.image_annotations_dict[annotation.image_path].append(annotation)

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations_dict:
            # Get the annotation from dict
            annotation = self.annotations_dict[annotation_id]
            # Remove from image annotations dict
            if annotation.image_path in self.image_annotations_dict:
                self.image_annotations_dict[annotation.image_path].remove(annotation)
                if not self.image_annotations_dict[annotation.image_path]:
                    del self.image_annotations_dict[annotation.image_path]

            # Delete the annotation
            annotation.delete()
            del self.annotations_dict[annotation_id]
            self.annotationDeleted.emit(annotation_id)
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

    def delete_selected_annotation(self):
        for annotation in self.selected_annotations:
            self.delete_annotation(annotation.id)
        self.unselect_annotations()

    def delete_annotations(self, annotations):
        for annotation in annotations:
            self.delete_annotation(annotation.id)

    def delete_label_annotations(self, label):
        for annotation in list(self.annotations_dict.values()):
            if annotation.label.id == label.id:
                annotation.delete()
                del self.annotations_dict[annotation.id]

    def delete_image_annotations(self, image_path):
        """Efficiently delete all annotations for a given image path"""
        if image_path in self.image_annotations_dict:
            annotations = self.image_annotations_dict[image_path]
            # Delete graphics items and annotations in batch
            for annotation in annotations:
                annotation.delete()
                del self.annotations_dict[annotation.id]
                self.annotationDeleted.emit(annotation.id)

            del self.image_annotations_dict[image_path]

            # Clear confidence window if needed
            if self.current_image_path == image_path:
                self.main_window.confidence_window.clear_display()

    def delete_image(self, image_path):
        # Delete all annotations associated with image path
        self.delete_annotations(self.get_image_annotations(image_path))
        # Delete the image
        if self.current_image_path == image_path:
            self.scene.clear()
            self.main_window.confidence_window.clear_display()
            self.current_image_path = None
            self.image_pixmap = None
            self.rasterio_image = None
            self.active_image = False

    def clear_scene(self):
        # Clean up
        self.unselect_annotations()

        # Clear the previous scene and delete its items
        if self.scene:
            for item in self.scene.items():
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
                    del item
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
