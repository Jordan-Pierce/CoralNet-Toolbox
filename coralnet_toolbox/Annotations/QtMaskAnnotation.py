import warnings

import base64
import rasterio

import numpy as np

from scipy.ndimage import label as ndimage_label
from skimage.measure import find_contours

from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pycocotools import mask

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QApplication
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush, QPolygonF

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MaskGraphicsItem(QGraphicsItem):
    def __init__(self, mask_annotation):
        super().__init__()
        self.mask_annotation = mask_annotation
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def boundingRect(self):
        height, width = self.mask_annotation.mask_data.shape
        return QRectF(0, 0, width, height)

    def paint(self, painter, option, widget):
        if self.mask_annotation.qimage:
            # Apply transparency at render time for instant updates
            transparency = self.mask_annotation.get_current_transparency()
            if transparency < 255:
                painter.setOpacity(transparency / 255.0)
            painter.drawImage(0, 0, self.mask_annotation.qimage)
            if transparency < 255:
                painter.setOpacity(1.0)  # Reset opacity for other drawing operations


class MaskAnnotation(Annotation):
    LOCK_BIT = 2**7  # For uint8, this is 128

    def __init__(self,
                 image_path: str,
                 mask_data: np.ndarray,
                 initial_labels: list,
                 transparency: int = 128,
                 show_msg: bool = False,
                 rasterio_src=None):
        """
        Initialize a full-image semantic segmentation annotation.
        There should only be one MaskAnnotation per image.
        """
        # --- This block is for context ---
        if not initial_labels:
            raise ValueError("initial_labels cannot be empty.")
        placeholder_label = initial_labels[0]

        super().__init__(
            short_label_code=placeholder_label.short_label_code,
            long_label_code=placeholder_label.long_label_code,
            color=placeholder_label.color,
            image_path=image_path,
            label_id=placeholder_label.id,
            transparency=transparency,
            show_msg=show_msg
        )
        
        self.mask_data = mask_data.astype(np.uint8)
        
        self.class_id_to_label_map = {}
        self.label_id_to_class_id_map = {}
        self.visible_label_ids = set()
        
        self.next_class_id = 1
        self.sync_label_map(initial_labels)
        # Update visible labels to include all labels currently in the map
        self.visible_label_ids = set(self.label_id_to_class_id_map.keys())

        self.offset = QPointF(0, 0)
        self.rasterio_src = rasterio_src
        
        # Initialize the colored canvas and QImage once at creation time.
        # This moves the expensive, one-time image generation from the first
        # brush stroke to the object's creation.
        self.colored_mask = None
        self.qimage = None
        self._initialize_canvas()
        
        self.set_centroid()
        self.set_cropped_bbox()

    def sync_label_map(self, current_labels_in_project: list):
        """Ensures the internal maps are synced with the project's labels,
           assigning new, stable IDs to any new labels without changing existing ones."""
        
        # Add any new labels from the project list to our maps
        for label in current_labels_in_project:
            if label.id not in self.label_id_to_class_id_map:
                new_id = self.next_class_id
                self.class_id_to_label_map[new_id] = label
                self.label_id_to_class_id_map[label.id] = new_id
                self.next_class_id += 1
                
                # Also add the new label's ID to the set of visible labels.
                self.visible_label_ids.add(label.id)
                
    def _build_color_map(self):
        """Builds a numpy array mapping class IDs to RGBA colors."""
        # Ensure the map is large enough to handle locked class IDs
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        map_size = max(256, max_id + self.LOCK_BIT + 1)
        color_map = np.zeros((map_size, 4), dtype=np.uint8)

        for class_id, label in self.class_id_to_label_map.items():
            color = label.color
            # Always use full alpha in the array - transparency applied at render time
            alpha = 255 if label.id in self.visible_label_ids else 0
            
            # Set the color for both the normal and the locked version of the class ID
            rgba = [color.red(), color.green(), color.blue(), alpha]
            color_map[class_id] = rgba
            if class_id + self.LOCK_BIT < map_size:
                color_map[class_id + self.LOCK_BIT] = rgba

        return color_map
    
    def _initialize_canvas(self):
        """
        Creates the initial color canvas and QImage. This is the one-time expensive
        operation that happens when the mask is first loaded.
        """
        height, width = self.mask_data.shape
        color_map = self._build_color_map()
        
        # Create the full-size 4-channel RGBA numpy array
        self.colored_mask = color_map[self.mask_data]
        
        # Create a QImage that is a VIEW on the numpy array's data buffer.
        # Modifying the numpy array will now automatically update the QImage.
        self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)

    def _update_full_canvas(self):
        """Regenerates the entire color canvas. Used for global changes like label color edits."""
        color_map = self._build_color_map()
        # Update the existing colored_mask array in-place
        np.copyto(self.colored_mask, color_map[self.mask_data])

    def _update_canvas_slice(self, update_rect):
        """
        Efficiently updates only a small rectangular slice of the color canvas.
        Used for tools like the brush.
        """
        x1, y1, x2, y2 = update_rect
        
        # Get the slice of the data that has changed
        data_slice = self.mask_data[y1:y2, x1:x2]
        
        # Rebuild the color map in case label properties have changed
        color_map = self._build_color_map()
        
        # Perform the color lookup ONLY for the small data slice
        color_slice = color_map[data_slice]
        
        # Update the corresponding slice in our persistent color canvas
        self.colored_mask[y1:y2, x1:x2] = color_slice

    def set_centroid(self):
        """Set the centroid to the center of the image."""
        height, width = self.mask_data.shape
        self.center_xy = QPointF(width / 2.0, height / 2.0)

    def set_cropped_bbox(self):
        """Set the bounding box to the full dimensions of the image."""
        height, width = self.mask_data.shape
        self.cropped_bbox = (0, 0, width, height)
        self.annotation_size = int(max(width, height))
                
    def contains_point(self, point: QPointF) -> bool:
        """Check if a point is within the mask's classified area."""
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if 0 <= y < height and 0 <= x < width:
            return self.mask_data[y, x] > 0
        return False

    def get_area(self):
        """Return the total number of non-background pixels."""
        return np.count_nonzero(self.mask_data)

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box (always 0,0)."""
        return QPointF(0, 0)

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        height, width = self.mask_data.shape
        return QPointF(width, height)

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create a QGraphicsPixmapItem to display the mask."""
        self.graphics_item = MaskGraphicsItem(self)
        scene.addItem(self.graphics_item)

    def update_graphics_item(self, update_rect=None):
        """Update the graphics item when mask data has changed."""
        if self.graphics_item is None:
            return
        
        if update_rect:
            # Localized update for brush strokes - update only the changed area
            self._update_canvas_slice(update_rect)
            qt_rect = QRectF(update_rect[0], 
                             update_rect[1], 
                             update_rect[2] - update_rect[0], 
                             update_rect[3] - update_rect[1])
            self.graphics_item.update(qt_rect)
        else:
            # Full update for global changes (e.g., label color changes)
            self._update_full_canvas()
            self.graphics_item.update()

    def update_mask(self, brush_location: QPointF, brush_mask: np.ndarray, new_class_id: int, annotation_window):
        """
        Modify the mask data based on a brush stroke, creating on-the-fly
        protection for vector annotations in the brush area.
        """
        x_start, y_start = int(brush_location.x()), int(brush_location.y())
        brush_h, brush_w = brush_mask.shape
        mask_h, mask_w = self.mask_data.shape

        # Define the update area and clip it to the mask's bounds
        x_end = min(x_start + brush_w, mask_w)
        y_end = min(y_start + brush_h, mask_h)
        clipped_x_start = max(x_start, 0)
        clipped_y_start = max(y_start, 0)

        if clipped_x_start >= x_end or clipped_y_start >= y_end:
            return
            
        # This is the rectangular region of the main mask we are about to modify
        update_rect = QRectF(clipped_x_start, 
                             clipped_y_start, 
                             x_end - clipped_x_start, 
                             y_end - clipped_y_start)
        
        # 1. Quickly find any vector annotations that overlap with our update area
        intersecting_annos = annotation_window.get_intersecting_annotations(update_rect)
        
        # 2. If there are any, create a localized boolean mask of the protected pixels
        if intersecting_annos:
            local_lock_mask = self._create_local_lock_mask(update_rect, intersecting_annos)
        else:
            local_lock_mask = None

        # Get the slice of the main mask data we will be updating
        target_slice = self.mask_data[clipped_y_start:y_end, clipped_x_start:x_end]
        
        # Clip the user's brush mask to match the on-screen portion
        brush_x_offset = clipped_x_start - x_start
        brush_y_offset = clipped_y_start - y_start
        clipped_brush_mask = brush_mask[brush_y_offset:brush_y_offset + target_slice.shape[0],
                                        brush_x_offset:brush_x_offset + target_slice.shape[1]]
        
        # 3. Apply protection: remove protected pixels from the brush mask
        if local_lock_mask is not None:
            # The brush can only paint where the original brush mask is True
            # AND the local lock mask is False.
            final_brush_mask = clipped_brush_mask & ~local_lock_mask
        else:
            final_brush_mask = clipped_brush_mask

        # Apply the final, protected brush mask to the data
        target_slice[final_brush_mask] = new_class_id
        
        # Trigger a visual update for the changed rectangle
        changed_rect_coords = (clipped_x_start, clipped_y_start, x_end, y_end)
        self.update_graphics_item(update_rect=changed_rect_coords)
        
        self.annotationUpdated.emit(self)
        
    def get_current_transparency(self):
        """Get the current transparency value for rendering."""
        # Try to get the active label's transparency from the application
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                main_window = app.activeWindow()
                if hasattr(main_window, 'label_window') and hasattr(main_window.label_window, 'active_label'):
                    active_label = main_window.label_window.active_label
                    if active_label:
                        return active_label.transparency
        except Exception:
            pass
        return self.transparency
    
    def update_transparency(self, transparency):
        """Update transparency instantly using render-time application."""
        transparency = max(0, min(255, transparency))  # Clamp to valid range
        if self.transparency != transparency:
            self.transparency = transparency
            # Update transparency on all visible labels
            for label_id in self.visible_label_ids:
                label = next((lbl for lbl in self.class_id_to_label_map.values() if lbl.id == label_id), None)
                if label:
                    label.transparency = transparency
            
            # Trigger repaint - transparency applied during rendering
            if self.graphics_item:
                self.graphics_item.update()
            
    def update_visible_labels(self, visible_ids: set):
        """
        Updates the set of visible label IDs and triggers a redraw of the mask.

        Args:
            visible_ids (set): A set containing the UUIDs of labels that should be visible.
        """
        self.visible_label_ids = visible_ids
        self.update_graphics_item()
            
    def remove_from_scene(self):
        """Removes the graphics item from its scene, if it exists."""
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            
        # Remove the graphics item reference
        self.graphics_item = None

    # --- Data Manipulation & Editing Methods ---

    def fill_region(self, point: QPointF, new_class_id: int, annotation_window):
        """
        Fills a contiguous region with a new class ID, protecting against
        overwriting vector annotations within the fill area.
        """
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if not (0 <= y < height and 0 <= x < width):
            return

        old_class_id = self.mask_data[y, x]
        if old_class_id == new_class_id:
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Find all contiguous pixels with the same starting class ID
        labeled_array, num_features = ndimage_label(self.mask_data == old_class_id)
        region_label = labeled_array[y, x]
        fill_mask = labeled_array == region_label
        
        if not np.any(fill_mask):
            QApplication.restoreOverrideCursor()
            return

        # Find bounding box of the area to be filled to limit our protection check
        coords = np.where(fill_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        fill_bbox = QRectF(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        
        # Find vector annotations that overlap with the fill area
        intersecting_annos = annotation_window.get_intersecting_annotations(fill_bbox)

        # If any exist, create a local protection mask
        if intersecting_annos:
            # Call the new, faster rasterio-based helper
            local_lock_mask = self._create_lock_mask_with_rasterio(fill_bbox, intersecting_annos)
            
            # Get the slice of the fill_mask corresponding to the lock_mask
            fill_mask_slice = fill_mask[y_min: y_max + 1, x_min: x_max + 1]
            
            # Remove protected pixels from the fill mask
            protected_fill_mask_slice = fill_mask_slice & ~local_lock_mask
            
            # Place the updated, protected slice back into the full-size fill mask
            fill_mask[y_min: y_max + 1, x_min: x_max + 1] = protected_fill_mask_slice

        # Apply the final, protected fill mask to the data
        self.mask_data[fill_mask] = new_class_id
        
        # Use localized update for efficiency
        update_rect = (x_min, y_min, x_max + 1, y_max + 1)
        self.update_graphics_item(update_rect=update_rect)

        QApplication.restoreOverrideCursor()
        self.annotationUpdated.emit(self)

    def replace_class(self, old_class_id: int, new_class_id: int):
        """Replaces all pixels of one class ID with another across the entire mask."""
        self.mask_data[self.mask_data == old_class_id] = new_class_id
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def _rasterize_annotation_group(self, annotations: list, class_id: int):
        """Burns a list of vector annotations OF THE SAME CLASS into the mask."""
        height, width = self.mask_data.shape

        geometries = []
        for anno in annotations:
            if hasattr(anno, 'get_polygon'):
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                if len(points) >= 3:
                    geometries.append(Polygon(points))

        if not geometries:
            return

        # Rasterize all geometries into a single boolean mask
        stencil_np = rasterize(
            geometries,
            out_shape=(height, width),
            transform=from_origin(0, 0, 1, 1),
            fill=0,          # Pixels outside shapes are 0 (not burned)
            default_value=1, # Pixels inside shapes are 1 (burned)
            dtype=np.uint8
        ).astype(bool)

        # Add the LOCK_BIT to the class_id to mark these pixels as protected.
        self.mask_data[stencil_np] = class_id + self.LOCK_BIT
        
    def _create_stencil_for_group(self, annotations: list) -> np.ndarray:
        """Creates a boolean numpy stencil for a list of annotations using rasterio."""
        height, width = self.mask_data.shape

        geometries = []
        for anno in annotations:
            if hasattr(anno, 'get_polygon'):
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                if len(points) >= 3:
                    geometries.append(Polygon(points))

        if not geometries:
            return np.zeros((height, width), dtype=bool)

        # Rasterize all geometries into a single boolean mask
        stencil = rasterize(
            geometries,
            out_shape=(height, width),
            transform=from_origin(0, 0, 1, 1),
            fill=0,          # Pixels outside shapes are 0 (not in stencil)
            default_value=1, # Pixels inside shapes are 1 (in stencil)
            dtype=np.uint8
        ).astype(bool)
        
        return stencil
    
    def _create_local_lock_mask(self, rect: QRectF, annotations: list) -> np.ndarray:
        """
        Creates a boolean numpy stencil for a given rectangular area by rasterizing
        the provided annotations. True means the pixel is protected.
        """
        width, height = int(rect.width()), int(rect.height())
        if width <= 0 or height <= 0:
            return np.zeros((height, width), dtype=bool)

        geometries = []
        for anno in annotations:
            if hasattr(anno, 'get_polygon'):
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                if len(points) >= 3:
                    geometries.append(Polygon(points))

        if not geometries:
            return np.zeros((height, width), dtype=bool)

        # Create a transform that maps the global coordinates of the geometries
        # to the local coordinate system of our output numpy array.
        transform = from_origin(rect.x(), rect.y(), 1, 1)

        # Rasterize the shapes into a numpy array.
        lock_mask = rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=0,          # Pixels outside the shapes are 0 (not locked)
            default_value=1, # Pixels inside the shapes are 1 (locked)
            dtype=np.uint8
        ).astype(bool)
        
        return lock_mask
    
    def _create_lock_mask_with_rasterio(self, rect: QRectF, annotations: list) -> np.ndarray:
        """
        Creates a boolean lock mask for a given rectangular area using the
        highly optimized rasterio.features.rasterize function.
        """
        geometries = []
        for anno in annotations:
            if hasattr(anno, 'get_polygon'):
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                if len(points) >= 3:
                    geometries.append(Polygon(points))
        
        if not geometries:
            return np.zeros((int(rect.height()), int(rect.width())), dtype=bool)
    
        # Create a transform that maps the global coordinates of the geometries
        # to the local coordinate system of our output numpy array.
        transform = from_origin(rect.x(), rect.y(), 1, 1)
    
        # Rasterize the shapes into a numpy array.
        lock_mask = rasterize(
            geometries,
            out_shape=(int(rect.height()), int(rect.width())),
            transform=transform,
            fill=0,          # Pixels outside the shapes are 0 (not locked)
            default_value=1, # Pixels inside the shapes are 1 (locked)
            dtype=np.uint8
        ).astype(bool)
        
        return lock_mask

    def rasterize_annotations(self, all_annotations: list):
        """
        Mark pixels covered by vector annotations as locked to prevent painting over them.
        Uses rasterio.features.rasterize for robust and efficient polygon rasterization.
        Vector annotations remain visible while their pixel areas become protected.
        
        Args:
            all_annotations: List of vector annotations to protect
        """
        if not all_annotations:
            return

        height, width = self.mask_data.shape

        geometries = []
        for annotation in all_annotations:
            if not hasattr(annotation, 'get_polygon'):
                continue
            try:
                polygon = annotation.get_polygon()
                if polygon is None or polygon.isEmpty():
                    continue
                points = [(polygon.at(i).x(), polygon.at(i).y()) for i in range(polygon.count())]
                if len(points) < 3:
                    continue
                geometries.append(Polygon(points))
            except Exception as e:
                print(f"Warning: Could not process annotation {annotation.id}: {e}")
                continue

        if not geometries:
            return

        # Rasterize all geometries into a single boolean mask
        lock_mask = rasterize(
            geometries,
            out_shape=(height, width),
            transform=from_origin(0, 0, 1, 1),
            fill=0,           # Pixels outside shapes are 0 (not locked)
            default_value=1,  # Pixels inside shapes are 1 (locked)
            dtype=np.uint8
        ).astype(bool)

        # Apply locking: Add LOCK_BIT to pixels that are not already locked
        to_lock = lock_mask & (self.mask_data < self.LOCK_BIT)
        self.mask_data[to_lock] += self.LOCK_BIT

    def unrasterize_annotations(self):
        """
        Remove lock protection from all pixels that were marked as locked.
        This allows mask editing over previously protected vector annotation areas.
        """
        # Find all pixels that have the lock bit set and remove it
        locked_pixels = self.mask_data >= self.LOCK_BIT
        
        # Remove the lock bit from these pixels, keeping their original class
        self.mask_data[locked_pixels] = self.mask_data[locked_pixels] - self.LOCK_BIT

    def clear_pixels_for_class(self, class_id: int):
        """Finds all pixels matching a class ID (both locked and unlocked) and resets them to 0."""
        if class_id == 0:  # Cannot clear background class
            return

        # Create a boolean mask of all pixels whose real class ID matches.
        pixels_to_clear = (self.mask_data % self.LOCK_BIT) == class_id
        
        # Set these pixels back to 0 (unclassified).
        self.mask_data[pixels_to_clear] = 0
        
    def clear_pixels_for_annotations(self, annotations_to_clear: list):
        """
        Rasterizes a list of vector annotations and sets the corresponding
        pixels in the mask_data to 0 (unclassified).
        """
        if not annotations_to_clear:
            return

        # 1. Convert all annotation polygons to Shapely Polygons.
        geometries = []
        for anno in annotations_to_clear:
            if hasattr(anno, 'get_polygon'):
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                if len(points) >= 3:
                    geometries.append(Polygon(points))

        if not geometries:
            return

        # 2. Rasterize all shapes at once into a boolean mask.
        # This creates a numpy array where 'True' indicates a pixel is covered
        # by at least one of the vector annotations.
        height, width = self.mask_data.shape
        clear_mask = rasterize(
            geometries,
            out_shape=(height, width),
            fill=0,
            default_value=1,
            dtype=np.uint8
        ).astype(bool)

        # 3. Apply the mask to the data, setting pixels to 0.
        self.mask_data[clear_mask] = 0

        # 4. Trigger a full repaint of the mask to show the changes.
        self.update_graphics_item()

    # --- Analysis & Information Retrieval Methods ---

    def get_class_statistics(self) -> dict:
        """Returns a dictionary with pixel counts and percentages for each class."""
        stats = {}
        total_pixels = self.get_area()
        if total_pixels == 0:
            return stats

        class_ids, counts = np.unique(self.mask_data[self.mask_data > 0], return_counts=True)
        
        for cid, count in zip(class_ids, counts):
            label = self.class_id_to_label_map.get(int(cid))
            if label:
                stats[label.short_label_code] = {
                    "pixel_count": int(count),
                    "percentage": (count / total_pixels) * 100
                }
        return stats

    def get_class_at_point(self, point: QPointF) -> int:
        """Returns the class ID at a specific point."""
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if 0 <= y < height and 0 <= x < width:
            return self.mask_data[y, x]
        return 0  # Return background class if outside bounds

    # --- Conversion & Exporting Methods ---
    
    def get_binary_mask(self, class_id: int) -> np.ndarray:
        """Returns a boolean numpy array where True corresponds to the given class ID."""
        return self.mask_data == class_id

    def to_instance_polygons(self, class_id: int) -> list:
        """Converts all contiguous regions of a class ID into PolygonAnnotations."""
        binary_mask = self.get_binary_mask(class_id)
        # Add padding to handle contours touching the border
        padded_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        
        # Level is 0.5 to find contours between 0 and 1 values
        contours = find_contours(padded_mask, level=0.5)
        
        annotations = []
        for contour in contours:
            # Remove padding offset and swap (row, col) to (x, y)
            points = [QPointF(p[1] - 1, p[0] - 1) for p in contour]
            if len(points) > 2:  # Must have at least 3 points for a valid polygon
                # Use the label associated with the class_id for the new annotation
                label = self.class_id_to_label_map[class_id]
                anno = PolygonAnnotation(
                    points=points,
                    short_label_code=label.short_label_code,
                    long_label_code=label.long_label_code,
                    color=label.color,
                    image_path=self.image_path,
                    label_id=label.id
                )
                annotations.append(anno)
        return annotations

    def export_as_png(self, path: str, use_label_colors: bool = True):
        """Saves the mask to a PNG file."""
        if use_label_colors:
            # Use current colored image for export
            self.qimage.save(path)
        else:
            # Save the raw class IDs as a grayscale image
            height, width = self.mask_data.shape
            # Ensure data is in a format QImage can handle (e.g., 8-bit grayscale)
            if self.mask_data.max() < 256:
                img_data = self.mask_data.astype(np.uint8)
                q_image = QImage(img_data.data, width, height, QImage.Format_Grayscale8)
                q_image.save(path)
            else:
                warnings.warn("Mask contains class IDs > 255; cannot save as 8-bit grayscale PNG.")

    def export_as_raster(self, path: str):
        """Saves the mask data to a raster file (e.g., GeoTIFF) using rasterio."""
        profile = {
            'driver': 'GTiff',
            'height': self.mask_data.shape[0],
            'width': self.mask_data.shape[1],
            'count': 1,
            'dtype': self.mask_data.dtype
        }
        
        # If the original image was opened with rasterio, copy its spatial metadata
        if self.rasterio_src:
            profile['crs'] = self.rasterio_src.crs
            profile['transform'] = self.rasterio_src.transform

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(self.mask_data, 1)

    # --- Serialization & Deserialization ---

    def to_dict(self):
        """Serialize the annotation to a dictionary, with RLE for the mask."""
        base_dict = super().to_dict()
        
        # Encode each class's binary mask using pycocotools
        rle_list = []
        unique_classes = np.unique(self.mask_data)
        for class_id in unique_classes:
            if class_id == 0:
                continue
            binary_mask = (self.mask_data == class_id).astype(np.uint8)
            rle = mask.encode(np.asfortranarray(binary_mask))
            rle['counts'] = base64.b64encode(rle['counts']).decode('ascii')
            rle_list.append({'class_id': int(class_id), 'rle': rle})
        
        # Convert the label map to a serializable format
        serializable_label_map = {}
        for cid, label in self.class_id_to_label_map.items():
            serializable_label_map[cid] = label.short_label_code

        base_dict.update({
            'shape': self.mask_data.shape,
            'rle_masks': rle_list,
            'label_map': serializable_label_map
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Instantiate a MaskAnnotation from a dictionary."""
        # Get all labels currently in the project. This is needed for the constructor.
        all_project_labels = list(label_window.labels)
        if not all_project_labels:
            raise ValueError("Cannot import a MaskAnnotation without any labels loaded in the project.")

        # Decode the RLE mask data
        shape = tuple(data['shape'])
        mask_data = np.zeros(shape, dtype=np.uint8)
        for item in data['rle_masks']:
            class_id = item['class_id']
            rle = item['rle']
            try:
                rle['counts'] = base64.b64decode(rle['counts'])
                binary_mask = mask.decode(rle).astype(bool)
                if binary_mask.shape != shape:
                    print(f"Warning: RLE decoded shape {binary_mask.shape} does not match expected shape {shape}")
                    continue
                mask_data[binary_mask] = class_id
            except Exception as e:
                print(f"Error decoding RLE for class {class_id}: {e}")
                continue

        # Create the base annotation instance. It will have a generic label map initially.
        annotation = cls(
            image_path=data['image_path'],
            mask_data=mask_data,
            initial_labels=all_project_labels
        )
        
        # Clear the generic maps created by the constructor.
        annotation.class_id_to_label_map.clear()
        annotation.label_id_to_class_id_map.clear()
        
        max_id_found = 0
        if 'label_map' in data and data['label_map']:
            # Iterate through the saved map {class_id: short_code}
            for cid_str, short_code in data['label_map'].items():
                class_id = int(cid_str)
                label = label_window.get_label_by_short_code(short_code)
                
                if label:
                    # Rebuild the maps with the correct associations
                    annotation.class_id_to_label_map[class_id] = label
                    annotation.label_id_to_class_id_map[label.id] = class_id
                    if class_id > max_id_found:
                        max_id_found = class_id
                else:
                    print(f"Warning: Label with short code '{short_code}' not found in project during mask import.")

        # Ensure the next class ID is set correctly to avoid future conflicts.
        annotation.next_class_id = max_id_found + 1

        # Restore other annotation properties
        annotation.id = data.get('id', annotation.id)
        annotation.data = data.get('data', {})
        
        return annotation

    @classmethod
    def from_rasterio(cls, file_path: str, image_path: str, all_labels: list):
        """Creates a MaskAnnotation instance by loading data from a raster file."""
        with rasterio.open(file_path) as src:
            mask_data = src.read(1)
            return cls(
                image_path=image_path,
                mask_data=mask_data,
                initial_labels=all_labels,
                rasterio_src=src
            )

    # --- Compatibility Methods ---
    def get_perimeter(self):
        height, width = self.mask_data.shape
        return 2 * (width + height)
    
    def get_polygon(self):
        height, width = self.mask_data.shape
        return QPolygonF(QRectF(0, 0, width, height))

    def __repr__(self):
        return (f"MaskAnnotation(id={self.id}, image_path={self.image_path}, "
                f"shape={self.mask_data.shape})")