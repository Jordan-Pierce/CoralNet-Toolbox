import warnings

import base64
import rasterio

import numpy as np

from scipy.ndimage import label as ndimage_label
from skimage.measure import find_contours
from pycocotools import mask

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush, QPolygonF

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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

        Args:
            image_path (str): Path to the source image.
            mask_data (np.ndarray): 2D numpy array of integer class IDs, matching image dimensions.
            initial_labels (list): A list of all Label_objects currently in the project.
            transparency (int): The alpha value for displaying the mask overlay.
            rasterio_src: Optional rasterio dataset object for the source image.
        """
        # For a full-image mask, the concept of a single "primary label" is ambiguous.
        # We'll use the first available label as a placeholder to satisfy the base class.
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
        
        self.class_id_to_label_map = {}  # Replaces the old label_map
        self.label_id_to_class_id_map = {}
        self.next_class_id = 1  # Start class IDs at 1 (0 is background)
        self.sync_label_map(initial_labels)

        self.offset = QPointF(0, 0)
        self.rasterio_src = rasterio_src
        
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

    def set_centroid(self):
        """Set the centroid to the center of the image."""
        height, width = self.mask_data.shape
        self.center_xy = QPointF(width / 2.0, height / 2.0)

    def set_cropped_bbox(self):
        """Set the bounding box to the full dimensions of the image."""
        height, width = self.mask_data.shape
        self.cropped_bbox = (0, 0, width, height)
        self.annotation_size = int(max(width, height))

    def _render_mask_to_pixmap(self) -> QPixmap:
        """Converts the numpy mask_data into a colored QPixmap for display."""
        height, width = self.mask_data.shape
        
        # Create a color map for fast lookup from class ID to RGBA color
        max_id = max(self.class_id_to_label_map.keys()) if self.class_id_to_label_map else 0
        color_map = np.zeros((max_id + 1, 4), dtype=np.uint8)
        for class_id, label in self.class_id_to_label_map.items():
            color = label.color
            color_map[class_id] = [color.red(), color.green(), color.blue(), self.transparency]

        # Get the real class IDs by removing the lock bit before color mapping.
        real_class_ids = self.mask_data % self.LOCK_BIT
        colored_mask = color_map[real_class_ids]
        
        q_image = QImage(colored_mask.data, width, height, QImage.Format_RGBA8888)
        
        return QPixmap.fromImage(q_image)

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
        pixmap = self._render_mask_to_pixmap()
        self.graphics_item = QGraphicsPixmapItem(pixmap)
        self.graphics_item.setPos(self.offset)
        # Directly add to scene without calling super(), as pixmap items don't support setBrush
        scene.addItem(self.graphics_item)

    def update_graphics_item(self):
        """Update the pixmap if the mask data has changed."""
        overall_start_time = time.time()
        
        if self.graphics_item:
            start_time = time.time()
            pixmap = self._render_mask_to_pixmap()
            end_time = time.time()
            print(f"_render_mask_to_pixmap took {end_time - start_time:.4f} seconds")
            start_time = time.time()
            self.graphics_item.setPixmap(pixmap)
            end_time = time.time()
            print(f"self.graphics_item.setPixmap took {end_time - start_time:.4f} seconds")

        start_time = time.time()
        super().update_graphics_item()
        end_time = time.time()
        print(f"super().update_graphics_item() took {end_time - start_time:.4f} seconds")
        
        overall_end_time = time.time()
        print(f"update_graphics_item took {overall_end_time - overall_start_time:.4f} seconds")

    def update_mask(self, brush_location: QPointF, brush_mask: np.ndarray, new_class_id: int):
        """Modify the mask data based on a brush stroke, avoiding locked pixels."""
        overall_start_time = time.time()
        
        start_time = time.time()
        x_start, y_start = int(brush_location.x()), int(brush_location.y())
        brush_h, brush_w = brush_mask.shape
        mask_h, mask_w = self.mask_data.shape

        x_end = min(x_start + brush_w, mask_w)
        y_end = min(y_start + brush_h, mask_h)
        clipped_x_start = max(x_start, 0)
        clipped_y_start = max(y_start, 0)

        if clipped_x_start >= x_end or clipped_y_start >= y_end:
            return
        end_time = time.time()
        print(f"Initial calculations took {end_time - start_time:.4f} seconds")

        start_time = time.time()
        target_slice = self.mask_data[clipped_y_start:y_end, clipped_x_start:x_end]
        end_time = time.time()
        print(f"Getting target_slice took {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        # Find which pixels in the target area are already locked.
        locked_pixels = target_slice >= self.LOCK_BIT
        end_time = time.time()
        print(f"Finding locked_pixels took {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        brush_x_offset = clipped_x_start - x_start
        brush_y_offset = clipped_y_start - y_start
        clipped_brush_mask = brush_mask[brush_y_offset:brush_y_offset + target_slice.shape[0],
                                        brush_x_offset:brush_x_offset + target_slice.shape[1]]
        end_time = time.time()
        print(f"Calculating offsets and clipped_brush_mask took {end_time - start_time:.4f} seconds")

        start_time = time.time()
        # Subtract the locked pixels from the brush mask. The brush can only paint
        # where the original brush mask is True AND the target pixel is NOT locked.
        final_brush_mask = clipped_brush_mask & ~locked_pixels
        end_time = time.time()
        print(f"Computing final_brush_mask took {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        # Apply the final, protected brush mask.
        target_slice[final_brush_mask] = new_class_id
        end_time = time.time()
        print(f"Applying the mask took {end_time - start_time:.4f} seconds")

        start_time = time.time()
        self.update_graphics_item()
        end_time = time.time()
        print(f"update_graphics_item took {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        self.annotationUpdated.emit(self)
        end_time = time.time()
        print(f"annotationUpdated.emit took {end_time - start_time:.4f} seconds")
        
        overall_end_time = time.time()
        print(f"update_mask took {overall_end_time - overall_start_time:.4f} seconds")
        
    def update_transparency(self, transparency):
        """Update the transparency of the mask annotation and re-render the graphics item."""
        transparency = max(0, min(255, transparency))  # Clamp to valid range
        if self.transparency != transparency:
            self.transparency = transparency
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

    def rasterize_annotations(self, all_annotations: list):
        """
        Groups a list of vector annotations by label and rasterizes each group
        onto this mask's data layer.
        """
        if not all_annotations:
            return

        # Group annotations by their label ID for efficient batch processing
        annotations_by_label = {}
        for anno in all_annotations:
            if anno.label.id not in annotations_by_label:
                annotations_by_label[anno.label.id] = []
            annotations_by_label[anno.label.id].append(anno)

        # For each label group, burn the corresponding annotations into the mask
        for label_id, annos_to_burn in annotations_by_label.items():
            # Now a fast, direct O(1) dictionary lookup using the UUID!
            class_id = self.label_id_to_class_id_map.get(label_id)
            if class_id:
                # Call the internal helper to do the actual drawing
                self._rasterize_annotation_group(annos_to_burn, class_id)
        
        # After all groups are rasterized, update the graphics item to show the changes
        self.update_graphics_item()

    def unrasterize_annotations(self):
        """
        Clears all pixels that were rasterized from vector annotations.
        This is identified by finding all pixels marked with the LOCK_BIT.
        """
        # Find all pixels that have the lock bit set.
        locked_pixels = self.mask_data >= self.LOCK_BIT
        
        # Reset these pixels back to 0 (unclassified).
        self.mask_data[locked_pixels] = 0
        
        # After clearing the pixels, update the graphics item to show the changes.
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

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
            pixmap = self._render_mask_to_pixmap()
            pixmap.toImage().save(path)
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
