import warnings
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QGraphicsPixmapItem)

from toolbox.QtPatchAnnotation import PatchAnnotation
from toolbox.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Tools.QtPanTool import PanTool
from toolbox.Tools.QtPatchTool import PatchTool
from toolbox.Tools.QtPolygonTool import PolygonTool
from toolbox.Tools.QtSAMTool import SAMTool
from toolbox.Tools.QtSelectTool import SelectTool
from toolbox.Tools.QtZoomTool import ZoomTool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes
    annotationSelected = pyqtSignal(int)  # Signal to emit when annotation is selected
    transparencyChanged = pyqtSignal(int)  # Signal to emit when transparency changes

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

        self.selected_annotation = None  # Stores the selected annotation
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.image_pixmap = None
        self.rasterio_image = None
        self.active_image = False  # Flag to check if the image has been set
        self.current_image_path = None

        self.toolChanged.connect(self.set_selected_tool)

        self.tools = {
            "pan": PanTool(self),
            "zoom": ZoomTool(self),
            "select": SelectTool(self),
            "patch": PatchTool(self),
            "polygon": PolygonTool(self),
            "sam": SAMTool(self),
        }

    def wheelEvent(self, event: QMouseEvent):
        if self.active_image:
            self.tools["zoom"].wheelEvent(event)
        if self.selected_tool:
            self.tools[self.selected_tool].wheelEvent(event)

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
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.tools["pan"].mouseReleaseEvent(event)
        self.toggle_cursor_annotation()
        self.drag_start_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.active_image and self.selected_tool:
            self.tools[self.selected_tool].keyPressEvent(event)
        super().keyPressEvent(event)

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

    def set_selected_tool(self, tool):
        if self.selected_tool:
            self.tools[self.selected_tool].deactivate()
        self.selected_tool = tool
        if self.selected_tool:
            self.tools[self.selected_tool].activate()

        self.unselect_annotation()
        self.toggle_cursor_annotation()

    def set_selected_label(self, label):
        self.selected_label = label
        self.annotation_color = label.color

        if self.selected_annotation:
            if self.selected_annotation.label.id != label.id:
                self.selected_annotation.update_user_confidence(self.selected_label)
                self.selected_annotation.create_cropped_image(self.rasterio_image)
                self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        if self.cursor_annotation:
            if self.cursor_annotation.label.id != label.id:
                self.toggle_cursor_annotation()

    def set_annotation_size(self, size=None, delta=0):
        if size is not None:
            self.annotation_size = size
        else:
            self.annotation_size += delta
            self.annotation_size = max(1, self.annotation_size)

        if isinstance(self.selected_annotation, PatchAnnotation):
            self.selected_annotation.update_annotation_size(self.annotation_size)
            if self.cursor_annotation:
                self.cursor_annotation.update_annotation_size(self.annotation_size)
        elif isinstance(self.selected_annotation, PolygonAnnotation):
            scale_factor = 1 + delta / 100.0
            self.selected_annotation.update_annotation_size(scale_factor)
            if self.cursor_annotation:
                self.cursor_annotation.update_annotation_size(scale_factor)

        if self.selected_annotation:
            self.selected_annotation.create_cropped_image(self.rasterio_image)
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        # Emit that the annotation size has changed
        self.annotationSizeChanged.emit(self.annotation_size)

    def set_annotation_location(self, annotation_id, new_center_xy: QPointF):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            annotation.update_location(new_center_xy)

    def set_annotation_transparency(self, transparency):
        if self.selected_annotation:
            # Update the label's transparency in the LabelWindow
            self.main_window.label_window.set_label_transparency(transparency)
            label = self.selected_annotation.label
            for annotation in self.annotations_dict.values():
                if annotation.label.id == label.id:
                    annotation.update_transparency(transparency)

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):

        if self.cursor_annotation:
            self.cursor_annotation.delete()
            self.cursor_annotation = None

        if not self.selected_label or not self.annotation_color:
            return

        if scene_pos:
            self.cursor_annotation = self.tools[self.selected_tool].create_annotation(scene_pos)
            self.cursor_annotation.create_graphics_item(self.scene)

    def display_image_item(self, image_item):
        # Clean up
        self.clear_scene()

        # Display NaN values the image dimensions in status bar
        self.imageLoaded.emit(-0, -0)

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

        # Set the image representations
        self.image_pixmap = QPixmap(self.main_window.image_window.images[image_path])
        self.rasterio_image = self.main_window.image_window.rasterio_images[image_path]

        self.current_image_path = image_path
        self.active_image = True

        # Set the image dimensions in status bar
        self.imageLoaded.emit(self.image_pixmap.width(), self.image_pixmap.height())

        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.toggle_cursor_annotation()

        # Load all associated annotations in parallel
        self.load_annotations_parallel()
        # Update the image window's image dict
        self.main_window.image_window.update_image_annotations(image_path)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        QApplication.processEvents()

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def center_on_annotation(self, annotation):
        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()
        annotation_center = annotation_rect.center()

        # Center the view on the annotation's center
        self.centerOn(annotation_center)

    def cycle_annotations(self, direction):
        if self.selected_tool == "select" and self.active_image:
            annotations = self.get_image_annotations()
            if annotations:
                if self.selected_annotation:
                    current_index = annotations.index(self.selected_annotation)
                    new_index = (current_index + direction) % len(annotations)
                else:
                    new_index = 0
                # Select the new annotation
                self.select_annotation(annotations[new_index])
                # Center the view on the new annotation
                self.center_on_annotation(annotations[new_index])

    def select_annotation(self, annotation):
        if self.selected_annotation != annotation:
            if self.selected_annotation:
                self.unselect_annotation()
            # Select the annotation
            self.selected_annotation = annotation
            self.selected_annotation.select()
            # Set the label and color for selected annotation
            self.selected_label = self.selected_annotation.label
            self.annotation_color = self.selected_annotation.label.color
            # Emit a signal indicating the selected annotations label to LabelWindow
            self.labelSelected.emit(annotation.label.id)
            # Emit a signal indicating the selected annotation's transparency to MainWindow
            self.transparencyChanged.emit(annotation.transparency)
            # Crop the image from annotation using current image item
            if not self.selected_annotation.cropped_image:
                self.selected_annotation.create_cropped_image(self.rasterio_image)
            # Display the selected annotation in confidence window
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

    def unselect_annotation(self):
        if self.selected_annotation:
            self.selected_annotation.deselect()
            self.selected_annotation = None

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

    def load_annotation(self, annotation):
        # Create the graphics item (scene previously cleared)
        annotation.create_graphics_item(self.scene)
        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)
        annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

    def load_annotations(self):
        # Crop all the annotations for current image (if not already cropped)
        annotations = self.crop_image_annotations(return_annotations=True)

        # Connect update signals for all the annotations
        for annotation in annotations:
            self.load_annotation(annotation)

    def load_annotations_parallel(self):
        # Crop all the annotations for current image (if not already cropped)
        annotations = self.crop_image_annotations(return_annotations=True)

        # Use ThreadPoolExecutor to process annotations in parallel
        with ThreadPoolExecutor() as executor:
            for annotation in annotations:
                executor.submit(self.load_annotation, annotation)

    def get_image_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path:
                annotations.append(annotation)

        return annotations

    def get_image_review_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path and annotation.label.id == '-1':
                annotations.append(annotation)

        return annotations

    def crop_image_annotations(self, image_path=None, return_annotations=False):
        if not image_path:
            # Set the image path if not provided
            image_path = self.current_image_path
        # Get the annotations for the image
        annotations = self.get_image_annotations(image_path)
        self._crop_annotations_batch(image_path, annotations)
        # Return the annotations if flag is set
        if return_annotations:
            return annotations

    def crop_these_image_annotations(self, image_path, annotations):
        # Crop these annotations for this image
        self._crop_annotations_batch(image_path, annotations)
        return annotations

    def _crop_annotations_batch(self, image_path, annotations):
        # Get the rasterio representation
        rasterio_image = self.main_window.image_window.rasterio_open(image_path)
        # Loop through the annotations, crop the image if not already cropped
        for annotation in annotations:
            if not annotation.cropped_image:
                annotation.create_cropped_image(rasterio_image)

    def add_annotation(self, scene_pos: QPointF = None):
        if not self.selected_label:
            QMessageBox.warning(self, "No Label Selected", "A label must be selected before adding an annotation.")
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

        annotation.create_graphics_item(self.scene)
        annotation.create_cropped_image(self.rasterio_image)

        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)
        annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

        self.annotations_dict[annotation.id] = annotation

        self.main_window.confidence_window.display_cropped_image(annotation)

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            del self.annotations_dict[annotation_id]

    def delete_selected_annotation(self):
        if self.selected_annotation:
            self.delete_annotation(self.selected_annotation.id)
            self.selected_annotation = None
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

    def delete_annotations(self, annotations):
        for annotation in annotations:
            self.delete_annotation(annotation.id)

    def delete_label_annotations(self, label):
        for annotation in list(self.annotations_dict.values()):
            if annotation.label.id == label.id:
                annotation.delete()
                del self.annotations_dict[annotation.id]

    def delete_image_annotations(self, image_path):
        annotations = self.get_image_annotations(image_path)
        self.delete_annotations(annotations)

    def delete_image(self, image_path):
        # Delete all annotations associated with image path
        self.delete_annotations(self.get_image_annotations(image_path))
        # Delete the image
        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.image_pixmap = None
            self.rasterio_image = None
            self.active_image = False  # Reset image_set flag

    def clear_scene(self):
        # Clean up
        self.unselect_annotation()

        # Clear the previous scene and delete its items
        if self.scene:
            for item in self.scene.items():
                self.scene.removeItem(item)
                del item
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)