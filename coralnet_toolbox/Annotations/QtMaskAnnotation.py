import warnings

import base64
import rasterio

import numpy as np

from scipy.ndimage import label as ndimage_label
from skimage.measure import find_contours
from pycocotools import mask

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush, QPolygonF

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
import time

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
            painter.drawImage(0, 0, self.mask_annotation.qimage)


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
        self.next_class_id = 1
        self.sync_label_map(initial_labels)

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
                
    def _build_color_map(self):
        """Builds a numpy array mapping class IDs to RGBA colors."""
        # Ensure the map is large enough to handle locked class IDs
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        map_size = max(256, max_id + self.LOCK_BIT + 1)
        color_map = np.zeros((map_size, 4), dtype=np.uint8)

        for class_id, label in self.class_id_to_label_map.items():
            color = label.color
            alpha = label.transparency if label.id in self.visible_label_ids else 0
            
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
                
    def _update_qimage(self, full_update=True, update_rect=None):
        """Update the QImage used for rendering the mask overlay."""
        height, width = self.mask_data.shape
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        color_map = np.zeros((max_id + 1, 4), dtype=np.uint8)

        for class_id, label in self.class_id_to_label_map.items():
            color = label.color
            # MODIFIED: Get transparency directly from the Label object itself.
            # The global self.transparency is no longer used for rendering individual classes.
            alpha = label.transparency if label.id in self.visible_label_ids else 0
            color_map[class_id] = [color.red(), color.green(), color.blue(), alpha]

        real_class_ids = self.mask_data % self.LOCK_BIT
        if full_update or self.colored_mask is None:
            self.colored_mask = color_map[real_class_ids]
            self.qimage = QImage(self.colored_mask.data, width, height, QImage.Format_RGBA8888)
        else:
            # Update only the specified rect
            if update_rect:
                x1, y1, x2, y2 = update_rect
                self.colored_mask[y1:y2, x1:x2] = color_map[real_class_ids[y1:y2, x1:x2]]

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
        """Update the pixmap if the mask data has changed."""
        if self.graphics_item is None:
            return
        
        start_time = time.time()
        if update_rect:
            # A small area changed (e.g., brush stroke) - THE FAST PATH
            self._update_canvas_slice(update_rect)
            qt_rect = QRectF(update_rect[0], 
                             update_rect[1], 
                             update_rect[2] - update_rect[0], 
                             update_rect[3] - update_rect[1])
            self.graphics_item.update(qt_rect)
        else:
            # A global change happened (e.g., label color changed) - THE SLOW PATH
            self._update_full_canvas()
            self.graphics_item.update()
        
        end_time = time.time()
        print(f"Finished update_graphics_item at {end_time}, total elapsed: {end_time - start_time}")

    def update_mask(self, brush_location: QPointF, brush_mask: np.ndarray, new_class_id: int):
        """Modify the mask data based on a brush stroke, avoiding locked pixels."""
        start_time = time.time()
        print(f"Starting update_mask at {start_time}")
        
        x_start, y_start = int(brush_location.x()), int(brush_location.y())
        brush_h, brush_w = brush_mask.shape
        mask_h, mask_w = self.mask_data.shape
        
        init_time = time.time()
        print(f"After initial calculations at {init_time}, elapsed: {init_time - start_time}")
        
        x_end = min(x_start + brush_w, mask_w)
        y_end = min(y_start + brush_h, mask_h)
        clipped_x_start = max(x_start, 0)
        clipped_y_start = max(y_start, 0)
        
        clip_time = time.time()
        print(f"After clipping calculations at {clip_time}, elapsed: {clip_time - init_time}")
        
        if clipped_x_start >= x_end or clipped_y_start >= y_end:
            return
        
        target_slice = self.mask_data[clipped_y_start:y_end, clipped_x_start:x_end]
        
        slice_time = time.time()
        print(f"After target slice at {slice_time}, elapsed: {slice_time - clip_time}")
        
        # Find which pixels in the target area are already locked.
        locked_pixels = target_slice >= self.LOCK_BIT
        
        lock_time = time.time()
        print(f"After finding locked pixels at {lock_time}, elapsed: {lock_time - slice_time}")
        
        brush_x_offset = clipped_x_start - x_start
        brush_y_offset = clipped_y_start - y_start
        clipped_brush_mask = brush_mask[brush_y_offset:brush_y_offset + target_slice.shape[0],
                                        brush_x_offset:brush_x_offset + target_slice.shape[1]]
        
        brush_clip_time = time.time()
        print(f"After clipping brush mask at {brush_clip_time}, elapsed: {brush_clip_time - lock_time}")
        
        # Subtract the locked pixels from the brush mask. The brush can only paint
        # where the original brush mask is True AND the target pixel is NOT locked.
        final_brush_mask = clipped_brush_mask & ~locked_pixels
        
        final_mask_time = time.time()
        print(f"After creating final brush mask at {final_mask_time}, elapsed: {final_mask_time - brush_clip_time}")
        
        # Apply the final, protected brush mask.
        target_slice[final_brush_mask] = new_class_id
        
        apply_time = time.time()
        print(f"After applying mask at {apply_time}, elapsed: {apply_time - final_mask_time}")
        
        changed_rect = (clipped_x_start, clipped_y_start, x_end, y_end)
        self.update_graphics_item(update_rect=changed_rect)
        
        update_time = time.time()
        print(f"After update_graphics_item at {update_time}, elapsed: {update_time - apply_time}")
        
        self.annotationUpdated.emit(self)
        
        end_time = time.time()
        print(f"Finished update_mask at {end_time}, total elapsed: {end_time - start_time}")
        
    def update_transparency(self, transparency):
        """Update the transparency of the mask annotation and re-render the graphics item."""
        transparency = max(0, min(255, transparency))  # Clamp to valid range
        if self.transparency != transparency:
            self.transparency = transparency
            
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

    def fill_region(self, point: QPointF, new_class_id: int):
        """Fills a contiguous region with a new class ID (paint bucket tool)."""
        x, y = int(point.x()), int(point.y())
        height, width = self.mask_data.shape
        if not (0 <= y < height and 0 <= x < width):
            return

        old_class_id = self.mask_data[y, x]
        if old_class_id == new_class_id:
            return

        labeled_array, num_features = ndimage_label(self.mask_data == old_class_id)
        region_label = labeled_array[y, x]
        
        # Only fill pixels that are part of the region and not locked
        region_mask = labeled_array == region_label
        unlocked_region_mask = region_mask & (self.mask_data < self.LOCK_BIT)
        self.mask_data[unlocked_region_mask] = new_class_id
        
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def replace_class(self, old_class_id: int, new_class_id: int):
        """Replaces all pixels of one class ID with another across the entire mask."""
        self.mask_data[self.mask_data == old_class_id] = new_class_id
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def _rasterize_annotation_group(self, annotations: list, class_id: int):
        """Burns a list of vector annotations OF THE SAME CLASS into the mask."""
        height, width = self.mask_data.shape
        stencil = QImage(width, height, QImage.Format_Alpha8)
        stencil.fill(Qt.transparent)

        painter = QPainter(stencil)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("white")))
        
        for anno in annotations:
            path = anno.get_painter_path()
            painter.drawPath(path)
        painter.end()

        # Fix for QImage memory padding issue
        stride = stencil.bytesPerLine()
        ptr = stencil.bits()
        ptr.setsize(stencil.byteCount())

        arr_view = np.ndarray(shape=(height, width), 
                              buffer=ptr, 
                              dtype=np.uint8, 
                              strides=(stride, 1))

        stencil_np = np.copy(arr_view) > 0

        # Add the LOCK_BIT to the class_id to mark these pixels as protected.
        self.mask_data[stencil_np] = class_id + self.LOCK_BIT
        
    def _create_stencil_for_group(self, annotations: list) -> np.ndarray:
        """Creates a boolean numpy stencil for a list of annotations using QPainter."""
        height, width = self.mask_data.shape
        # Use a memory-efficient 1-bit format for the stencil
        stencil_image = QImage(width, height, QImage.Format_Mono)
        stencil_image.fill(0)  # 0 is transparent/off

        painter = QPainter(stencil_image)
        # 1 is the color that will be "on"
        painter.setPen(QColor(1, 1, 1))
        painter.setBrush(QColor(1, 1, 1))
        
        for anno in annotations:
            # get_painter_path() is an assumed method on vector annotation objects
            path = anno.get_painter_path()
            painter.drawPath(path)
        painter.end()

        # Efficiently convert the QImage buffer to a numpy array
        ptr = stencil_image.bits()
        ptr.setsize(stencil_image.byteCount())
        
        # Note: QImage.Format_Mono pads each line to be a multiple of 32 bits.
        # We must account for this by using the correct stride (bytesPerLine).
        arr_view = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stencil_image.bytesPerLine()))
        
        # The final boolean mask is derived by un-packing the bits and slicing to the correct width
        packed = np.unpackbits(arr_view, axis=1)
        return packed[:, :width].astype(bool)

    def rasterize_annotations(self, all_annotations: list):
        """
        Mark pixels covered by vector annotations as locked to prevent painting over them.
        Uses an efficient polygon fill algorithm instead of expensive QPainter operations.
        Vector annotations remain visible while their pixel areas become protected.
        
        Args:
            all_annotations: List of vector annotations to protect
        """
        if not all_annotations:
            return

        height, width = self.mask_data.shape
        
        # Process each annotation to mark its pixels as locked
        for annotation in all_annotations:
            if not hasattr(annotation, 'get_polygon'):
                continue
                
            try:
                # Get the polygon points for this annotation
                polygon = annotation.get_polygon()
                if polygon is None or polygon.isEmpty():
                    continue
                
                # Convert QPolygonF to a list of (x, y) tuples
                points = [(polygon.at(i).x(), polygon.at(i).y()) for i in range(polygon.count())]
                if len(points) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Use a simple polygon fill algorithm to mark pixels as locked
                self._mark_polygon_pixels_as_locked(points, width, height)
                
            except Exception as e:
                # Skip annotations that cause errors rather than failing completely
                print(f"Warning: Could not rasterize annotation {annotation.id}: {e}")
                continue

    def unrasterize_annotations(self):
        """
        Remove lock protection from all pixels that were marked as locked.
        This allows mask editing over previously protected vector annotation areas.
        """
        # Find all pixels that have the lock bit set and remove it
        locked_pixels = self.mask_data >= self.LOCK_BIT
        
        # Remove the lock bit from these pixels, keeping their original class
        self.mask_data[locked_pixels] = self.mask_data[locked_pixels] - self.LOCK_BIT
    
    def _mark_polygon_pixels_as_locked(self, points, width, height):
        """
        Efficiently mark pixels inside a polygon as locked using a scanline fill algorithm.
        This is much faster than QPainter for simple polygon filling.
        
        Args:
            points: List of (x, y) tuples defining the polygon vertices
            width: Image width
            height: Image height
        """
        if len(points) < 3:
            return
            
        # Convert to integer coordinates and clip to image bounds
        int_points = []
        for x, y in points:
            x = max(0, min(width - 1, int(x)))
            y = max(0, min(height - 1, int(y)))
            int_points.append((x, y))
        
        # Find bounding box to limit scan area
        min_y = min(p[1] for p in int_points)
        max_y = max(p[1] for p in int_points)
        
        # For each scanline (row) in the bounding box
        for y in range(min_y, max_y + 1):
            # Find intersections of the scanline with polygon edges
            intersections = []
            n_points = len(int_points)
            
            for i in range(n_points):
                j = (i + 1) % n_points
                x1, y1 = int_points[i]
                x2, y2 = int_points[j]
                
                # Check if edge crosses the scanline
                if (y1 <= y < y2) or (y2 <= y < y1):
                    # Calculate intersection x-coordinate
                    if y2 != y1:  # Avoid division by zero
                        x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        intersections.append(int(x_intersect))
            
            # Sort intersections and fill between pairs
            intersections.sort()
            for i in range(0, len(intersections) - 1, 2):
                x_start = max(0, min(width - 1, intersections[i]))
                x_end = max(0, min(width - 1, intersections[i + 1]))
                
                # Mark pixels in this horizontal segment as locked
                for x in range(x_start, x_end + 1):
                    current_val = self.mask_data[y, x]
                    if current_val < self.LOCK_BIT:  # Only lock if not already locked
                        self.mask_data[y, x] = current_val + self.LOCK_BIT

    def clear_pixels_for_class(self, class_id: int):
        """Finds all pixels matching a class ID (both locked and unlocked) and resets them to 0."""
        if class_id == 0:  # Cannot clear background class
            return

        # Create a boolean mask of all pixels whose real class ID matches.
        pixels_to_clear = (self.mask_data % self.LOCK_BIT) == class_id
        
        # Set these pixels back to 0 (unclassified).
        self.mask_data[pixels_to_clear] = 0

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
            # Use rendering logic to create a colored image
            self._update_qimage()
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