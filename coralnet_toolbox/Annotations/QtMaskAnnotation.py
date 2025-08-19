import warnings

import zlib
import base64
import rasterio

import numpy as np

from scipy.ndimage import label as ndimage_label
from skimage.measure import find_contours

from PyQt5.QtCore import Qt, QPointF, QRectF, QPolygonF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QBrush

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions for Serialization
# ----------------------------------------------------------------------------------------------------------------------


def rle_encode(mask):
    """
    Encodes a 2D numpy array using Run-Length Encoding.
    Returns a compressed string representation.
    """
    pixels = mask.flatten()
    pixels = np.append(pixels, -1)  # Append a sentinel value
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1:] = runs[1:] - runs[:-1]
    values = pixels[np.cumsum(np.append(0, runs[:-1]))]
    
    # Pair values and runs, then convert to a string
    rle_pairs = ",".join([f"{v},{r}" for v, r in zip(values, runs)])
    
    # Further compress with zlib and encode in base64 for JSON compatibility
    compressed = zlib.compress(rle_pairs.encode('utf-8'))
    return base64.b64encode(compressed).decode('ascii')


def rle_decode(rle_string, shape):
    """
    Decodes a Run-Length Encoded string back into a 2D numpy array.
    """
    # Decode from base64 and decompress with zlib
    decoded_b64 = base64.b64decode(rle_string)
    decompressed = zlib.decompress(decoded_b64).decode('utf-8')
    
    # Parse the value,run pairs
    pairs = decompressed.split(',')
    values = [int(v) for v in pairs[0::2]]
    runs = [int(r) for r in pairs[1::2]]
    
    # Reconstruct the pixel array
    pixels = np.repeat(values, runs)
    
    # Reshape to the original 2D mask dimensions
    return pixels.reshape(shape)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MaskAnnotation(Annotation):
    def __init__(self,
                 image_path: str,
                 mask_data: np.ndarray,
                 label_map: dict,
                 transparency: int = 128,
                 show_msg: bool = False,
                 rasterio_src=None):
        """
        Initialize a full-image semantic segmentation annotation.
        There should only be one MaskAnnotation per image.

        Args:
            image_path (str): Path to the source image.
            mask_data (np.ndarray): 2D numpy array of integer class IDs, matching image dimensions.
            label_map (dict): A map of {class_id: Label_object}.
            transparency (int): The alpha value for displaying the mask overlay.
            rasterio_src: Optional rasterio dataset object for the source image.
        """
        # For a full-image mask, the concept of a single "primary label" is ambiguous.
        # We'll use the first available label as a placeholder to satisfy the base class.
        if not label_map:
            raise ValueError("label_map cannot be empty.")
        placeholder_label = next(iter(label_map.values()))

        super().__init__(
            short_label_code=placeholder_label.short_label_code,
            long_label_code=placeholder_label.long_label_code,
            color=placeholder_label.color,
            image_path=image_path,
            label_id=placeholder_label.id,
            transparency=transparency,
            show_msg=show_msg
        )
        
        self.mask_data = mask_data
        self.label_map = label_map
        self.offset = QPointF(0, 0)  # A full-image mask has no offset
        self.rasterio_src = rasterio_src  # Store rasterio source if provided
        
        # Set geometric properties from mask
        self.set_centroid()
        self.set_cropped_bbox()

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
        max_id = max(self.label_map.keys()) if self.label_map else 0
        color_map = np.zeros((max_id + 1, 4), dtype=np.uint8)
        for class_id, label in self.label_map.items():
            color = label.color
            color_map[class_id] = [color.red(), color.green(), color.blue(), self.transparency]

        colored_mask = color_map[self.mask_data]
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
        super().create_graphics_item(scene)

    def update_graphics_item(self):
        """Update the pixmap if the mask data has changed."""
        if self.graphics_item:
            pixmap = self._render_mask_to_pixmap()
            self.graphics_item.setPixmap(pixmap)
        super().update_graphics_item()

    def update_mask(self, brush_location: QPointF, brush_mask: np.ndarray, new_class_id: int):
        """Modify the mask data based on a brush stroke."""
        x_start, y_start = int(brush_location.x()), int(brush_location.y())
        brush_h, brush_w = brush_mask.shape
        mask_h, mask_w = self.mask_data.shape

        x_end = min(x_start + brush_w, mask_w)
        y_end = min(y_start + brush_h, mask_h)
        clipped_x_start = max(x_start, 0)
        clipped_y_start = max(y_start, 0)

        if clipped_x_start >= x_end or clipped_y_start >= y_end:
            return

        target_slice = self.mask_data[clipped_y_start:y_end, clipped_x_start:x_end]
        brush_x_offset = clipped_x_start - x_start
        brush_y_offset = clipped_y_start - y_start
        clipped_brush_mask = brush_mask[brush_y_offset:brush_y_offset + target_slice.shape[0],
                                        brush_x_offset:brush_x_offset + target_slice.shape[1]]

        target_slice[clipped_brush_mask] = new_class_id

        self.update_graphics_item()
        self.annotationUpdated.emit(self)

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
        
        self.mask_data[labeled_array == region_label] = new_class_id
        
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def replace_class(self, old_class_id: int, new_class_id: int):
        """Replaces all pixels of one class ID with another across the entire mask."""
        self.mask_data[self.mask_data == old_class_id] = new_class_id
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def import_annotations(self, annotations: list, class_id: int):
        """Burns a list of vector annotations into the mask with a given class ID."""
        height, width = self.mask_data.shape
        # Use Format_Alpha8 for an efficient 8-bit stencil mask
        stencil = QImage(width, height, QImage.Format_Alpha8)
        stencil.fill(Qt.transparent)

        painter = QPainter(stencil)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("white")))  # Opaque white
        
        for anno in annotations:
            path = anno.get_painter_path()
            painter.drawPath(path)
        painter.end()

        # Convert the QImage stencil to a boolean numpy array
        ptr = stencil.bits()
        ptr.setsize(stencil.byteCount())
        stencil_np = np.array(ptr).reshape(height, width) > 0  # True where painted

        # Update the mask data where the stencil is True
        self.mask_data[stencil_np] = class_id
        
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    # --- Analysis & Information Retrieval Methods ---

    def get_class_statistics(self) -> dict:
        """Returns a dictionary with pixel counts and percentages for each class."""
        stats = {}
        total_pixels = self.get_area()
        if total_pixels == 0:
            return stats

        class_ids, counts = np.unique(self.mask_data[self.mask_data > 0], return_counts=True)
        
        for cid, count in zip(class_ids, counts):
            label = self.label_map.get(int(cid))
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
                label = self.label_map[class_id]
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
        rle_string = rle_encode(self.mask_data)
        base_dict.update({
            'shape': self.mask_data.shape,
            'rle_mask': rle_string,
            'label_map': {cid: label.short_label_code for cid, label in self.label_map.items()}
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Instantiate a MaskAnnotation from a dictionary."""
        label_map = {}
        for cid_str, short_code in data['label_map'].items():
            label = label_window.get_label_by_short_code(short_code)
            if label:
                label_map[int(cid_str)] = label

        mask_data = rle_decode(data['rle_mask'], data['shape'])

        annotation = cls(
            image_path=data['image_path'],
            mask_data=mask_data,
            label_map=label_map
        )
        annotation.id = data.get('id', annotation.id)
        annotation.data = data.get('data', {})
        return annotation

    @classmethod
    def from_rasterio(cls, file_path: str, image_path: str, label_map: dict):
        """Creates a MaskAnnotation instance by loading data from a raster file."""
        with rasterio.open(file_path) as src:
            mask_data = src.read(1)
            return cls(
                image_path=image_path,
                mask_data=mask_data,
                label_map=label_map,
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
